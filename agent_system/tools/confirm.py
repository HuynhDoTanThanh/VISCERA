"""Confirmation harness driver — validate the label set before training the foundation.

Extracts (cached) train + val with the production strategy/panel, runs the statistical gates
(confirm.py), and optionally the agentic faithfulness audit + meta-review (audit.py).

    # offline gates only (fast if train/val already extracted by the cli):
    python -m agent_system.tools.confirm --train-ndbe 200 --val-ndbe 200 --workers 48
    # + agentic faithfulness audit & meta-verdict:
    python -m agent_system.tools.confirm --audit --audit-n 60 --workers 48

Writes outputs/reports/label_confirmation.{md,json}.
"""
from __future__ import annotations
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..application.aggregation import Aggregator
from ..application.audit import audit_faithfulness, meta_review, stratify
from ..application.confirm import confirm, format_report
from ..application.extraction import Extractor
from ..application.strategies import MultiStageStrategy, SingleShotStrategy
from ..config import Settings
from ..domain.concepts import ROBUST_CORE
from ..infrastructure.cache import FileVoteCache
from ..infrastructure.dataset import DatasetLoader
from ..infrastructure.vlm import ProxyVLMClient
from ..cli import load_anchors


def extract_aggs(frames, ext, agg, workers):
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(lambda fr: (fr, ext.extract(fr, ANCHORS)), f): f for f in frames}
        done = 0
        for f in as_completed(futs):
            fr, (votes, _) = f.result()
            done += 1
            if votes:
                out[fr.path] = agg.aggregate(fr, votes)
            if done % 50 == 0:
                print(f"    extracted {done}/{len(frames)}", flush=True)
    return out


def sample(frames, n_ndbe, seed=0):
    neo = [f for f in frames if f.label == 1]
    ndbe = [f for f in frames if f.label == 0]
    return neo + random.Random(seed).sample(ndbe, min(n_ndbe, len(ndbe)))


def main():
    global ANCHORS
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ndbe", type=int, default=200, help="ndbe sampled from train (neo: all)")
    ap.add_argument("--val-ndbe", type=int, default=200, help="ndbe sampled from val (neo: all)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--audit-n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--no-meta", action="store_true")
    args = ap.parse_args()

    s = Settings(); s.ensure_dirs()
    ANCHORS = load_anchors(s.raw_store.parent / "anchors.json")
    clients = {sp.key: ProxyVLMClient(sp, s) for sp in s.experts}
    strat = (MultiStageStrategy(s) if s.extraction_strategy == "multistage" else SingleShotStrategy(s))
    ext = Extractor(clients, FileVoteCache(s.raw_label_dir), s, strat)
    agg = Aggregator()
    loader = DatasetLoader(s)

    tr = sample(loader.train(), args.train_ndbe)
    va = sample(loader.val(), args.val_ndbe)
    print(f"[confirm] extracting train={len(tr)} val={len(va)} "
          f"(strategy={s.extraction_strategy}, experts={[e.key for e in s.experts]})…")
    tr_aggs = extract_aggs(tr, ext, agg, args.workers)
    va_aggs = extract_aggs(va, ext, agg, args.workers)

    rep = confirm(tr_aggs, tr, va_aggs, va, boot=args.boot)
    text = format_report(rep)
    print("\n" + text)

    if args.audit:
        core = rep["confirmed_core"] or list(ROBUST_CORE)
        samples = stratify(va_aggs, va, core, n=args.audit_n)
        adj_client = clients.get("proagent") or next(iter(clients.values()))
        print(f"\n[confirm] agentic audit: re-grading {len(samples)} val frames on {len(core)} core…")
        audit = audit_faithfulness(samples, core, adj_client, workers=args.workers)
        rep["audit"] = audit
        text += "\n\n## Agentic faithfulness audit (independent re-grade)\n"
        text += (f"- graded baseline={audit['n_baseline']} risk={audit['n_risk']}\n\n"
                 f"{'concept':22}{'MAE_base':>9}{'agr_base':>9}{'MAE_risk':>9}{'agr_risk':>9}\n"
                 + "-" * 58 + "\n")
        pb, pr = audit["per_concept_baseline"], audit["per_concept_risk"]
        for c in core:
            b, r = pb.get(c), pr.get(c)
            bm = f"{b['mae']:.3f}" if b else "-"
            ba = f"{b['agree@0.5']:.2f}" if b else "-"
            rm = f"{r['mae']:.3f}" if r else "-"
            ra = f"{r['agree@0.5']:.2f}" if r else "-"
            text += f"{c:22}{bm:>9}{ba:>9}{rm:>9}{ra:>9}\n"
        if not args.no_meta:
            verdict = meta_review(text, audit, adj_client)
            if verdict:
                rep["verdict"] = verdict
                text += "\n## Meta-review verdict\n```json\n" + json.dumps(verdict, indent=2) + "\n```\n"
                print("VERDICT:", verdict.get("verdict"))

    rdir = s.raw_store.parent.parent.parent / "outputs" / "reports"
    (rdir / "label_confirmation.md").write_text(text)
    (rdir / "label_confirmation.json").write_text(json.dumps(rep, indent=2))
    print(f"\n[confirm] CONFIRMED core ({len(rep['confirmed_core'])}): {', '.join(rep['confirmed_core'])}")
    print(f"[confirm] wrote {rdir/'label_confirmation.md'}")


if __name__ == "__main__":
    main()
