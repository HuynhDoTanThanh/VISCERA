"""Composition root + CLI. Wires infrastructure into the application and runs the pipeline.

    python -m agent_system.cli --split val
    python -m agent_system.cli --split unlabeled --limit 9937 --workers 48
    python -m agent_system.cli --split train --name race_v1

Outputs:
    raw_store/      logs/<run>.log · raw_labels/*.json (resumable) · runs/<run>/manifest.json
    training_store/[<name>/]  images/<stem>.jpg  +  labels/<stem>.json   (one label per image)
"""
from __future__ import annotations
import argparse
import asyncio
import dataclasses
import json
from datetime import datetime
from pathlib import Path

from .application.aggregation import Aggregator
from .application.extraction import Extractor
from .application.pipeline import LabelPipeline
from .application.strategies import MultiStageStrategy, SingleShotStrategy
from .application.trust import Curator, TrustScorer
from .config import Settings
from .domain.entities import Anchor
from .infrastructure.cache import FileVoteCache
from .infrastructure.dataset import DatasetLoader
from .infrastructure.logging import get_logger
from .infrastructure.vlm import ProxyVLMClient
from .outputs.writers import RawStore, TrainingStore


def load_anchors(path: Path) -> list[Anchor]:
    data = json.loads(Path(path).read_text())
    out = []
    for a in data:
        out.append(Anchor(path=a["path"], kind=a.get("kind") or a["cls"],
                          center=a.get("center", ""), hard=bool(a.get("hard", False))))
    return out


def build_settings(args) -> Settings:
    overrides = {}
    if args.workers:
        w = str(args.workers).lower()
        if w.startswith("auto"):
            overrides["adaptive_concurrency"] = True   # AIMD auto-tune up to the max limit
            if ":" in w:                               # auto:N → set the adaptive ceiling (e.g. 256×gateways)
                overrides["max_workers"] = int(w.split(":", 1)[1])
        else:
            overrides["max_workers"] = int(w)
    if args.votes:
        overrides["votes_per_expert"] = args.votes
        overrides["ms_votes_per_expert"] = args.votes
    if args.strategy:
        overrides["extraction_strategy"] = args.strategy
    if args.self_verify:
        overrides["self_verify"] = True
    if args.experts:
        base = Settings()
        overrides["experts"] = tuple(e for e in base.experts if e.key in args.experts)
    if args.dataset_root:
        overrides["dataset_root"] = Path(args.dataset_root)
    if args.unlabeled_dir:
        overrides["unlabeled_path"] = Path(args.unlabeled_dir)
    if args.out:
        overrides["training_store"] = Path(args.out).resolve()
    if args.name:
        overrides["dataset_name"] = args.name
    return dataclasses.replace(Settings(), **overrides)


def main():
    ap = argparse.ArgumentParser(description="agent_system — generate best-sure foundation labels")
    ap.add_argument("--split", choices=["train", "val", "unlabeled"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap unlabeled frames")
    ap.add_argument("--dataset-root", default=None, help="override dataset root dir")
    ap.add_argument("--unlabeled-dir", default=None,
                    help="path to the unlabeled images (recursive); overrides dataset/unlabeled_data")
    ap.add_argument("--out", default=None,
                    help="output dir for the dataset (default: agent_system/artifacts/training_store). "
                         "Writes <out>/images/*.jpg + <out>/labels/*.json")
    ap.add_argument("--name", default=None,
                    help="nest dataset under <out>/<name>/ (else flat images/+labels/)")
    ap.add_argument("--anchors", default=None, help="anchors JSON (default: artifacts/anchors.json "
                    "then outputs/reports/fewshot_anchors.json)")
    ap.add_argument("--strategy", choices=["multistage", "single"], default=None)
    ap.add_argument("--self-verify", action="store_true")
    ap.add_argument("--votes", type=int, default=None)
    ap.add_argument("--workers", default=None,
                    help="int = fixed concurrency cap; 'auto' = AIMD adaptive auto-tuning")
    ap.add_argument("--experts", nargs="+", default=None, help="subset of expert keys")
    args = ap.parse_args()

    settings = build_settings(args)
    settings.ensure_dirs()
    run_id = f"{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log = get_logger("pipeline", settings.log_dir, run_id)

    # resolve anchors
    cand = [Path(args.anchors)] if args.anchors else [
        settings.raw_store.parent / "anchors.json",
        Settings.dataset_root.parent / "outputs" / "reports" / "fewshot_anchors.json",
    ]
    anchor_path = next((p for p in cand if p.exists()), None)
    if anchor_path is None:
        log.error("no anchors file found; run tools.select_anchors first or pass --anchors")
        return
    anchors = load_anchors(anchor_path)
    log.info(f"run={run_id} experts={[e.key for e in settings.experts]} "
             f"votes={settings.votes_per_expert} anchors={len(anchors)} ({anchor_path.name})")

    # wire dependencies (composition root)
    clients = {spec.key: ProxyVLMClient(spec, settings) for spec in settings.experts}
    cache = FileVoteCache(settings.raw_label_dir)
    strategy = (MultiStageStrategy(settings) if settings.extraction_strategy == "multistage"
                else SingleShotStrategy(settings))
    if settings.extraction_strategy == "multistage":
        votes_desc = f"votes/expert={settings.ms_votes_per_expert}"
    else:
        votes_desc = "votes/expert=" + ", ".join(
            f"{e.key}:{e.votes or settings.votes_per_expert}" for e in settings.experts)
    log.info(f"strategy={settings.extraction_strategy} {votes_desc}")
    pipeline = LabelPipeline(
        extractor=Extractor(clients, cache, settings, strategy),
        aggregator=Aggregator(),
        scorer=TrustScorer(settings),
        curator=Curator(settings),
        settings=settings, logger=log,
    )

    frames = DatasetLoader(settings).load(args.split, args.limit)
    log.info(f"loaded {len(frames)} {args.split} frames")

    training, raw = TrainingStore(settings), RawStore(settings)
    counts: dict[str, int] = {}
    n_cells = 0

    def on_result(cf):
        nonlocal n_cells
        training.write_pair(cf)                          # images/<stem>.jpg + labels/<stem>.json
        counts[cf.decision.value] = counts.get(cf.decision.value, 0) + 1
        n_cells += len(cf.supervised)

    results = asyncio.run(pipeline.arun(frames, anchors, on_result=on_result))

    raw.write_run(run_id, args.split, anchors,
                  {"frames": len(frames), "curated": len(results),
                   "supervised_cells": n_cells, "decisions": counts})
    log.info(f"DONE: {len(results)}/{len(frames)} curated · cells={n_cells} · decisions={counts}")
    log.info(f"dataset -> {training.root}/  (images/*.jpg + labels/*.json)")


if __name__ == "__main__":
    main()
