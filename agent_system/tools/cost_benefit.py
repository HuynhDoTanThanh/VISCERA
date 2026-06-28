"""Single vs multistage: accuracy, reliability, coverage, and API cost on the same val sample.
Reuses the cached votes (no new API calls)."""
from __future__ import annotations
import dataclasses
import json
import random

import numpy as np

from ..domain.concept_schema import CONCEPTS
from ..config import Settings
from ..domain.concepts import ROBUST_CORE
from ..infrastructure.cache import FileVoteCache
from ..infrastructure.dataset import DatasetLoader
from ..infrastructure.vlm import ProxyVLMClient
from ..application.aggregation import Aggregator
from ..application.extraction import Extractor, anchor_signature
from ..application.strategies import MultiStageStrategy, SingleShotStrategy
from ..cli import load_anchors
from .ab_strategy import run_strategy, metrics


def api_calls(cache, frames, anchors, expert, strategy, version):
    sig = anchor_signature(anchors)
    ns = f"{expert}.{strategy}" + (f".{version}" if strategy == "multistage" else "")
    total = 0
    for fr in frames:
        payload_file = cache._file(cache.key(fr.path, ns, sig))  # noqa: SLF001
        if not payload_file.exists():
            continue
        p = json.loads(payload_file.read_text())
        if strategy == "single":
            total += len(p.get("votes", []))                     # 1 call per vote
        else:
            for m in p.get("meta", []):
                st = m.get("stages", [])
                # 'chunks' = 3 calls (surface/vascular/focal); others = 1
                total += sum(3 if s == "chunks" else 1 for s in st)
    return total


def disc_reliability(aggs, frames, names):
    ok = [fr for fr in frames if fr.path in aggs]
    rel, ass = [], []
    for n in names:
        rel += [aggs[fr.path].cells[n].reliability for fr in ok]
        ass += [aggs[fr.path].cells[n].mask for fr in ok]
    return float(np.mean(rel)), float(np.mean(ass))


def main():
    base = Settings(); base.ensure_dirs()
    spec = next(e for e in base.experts if e.key == "proagent")
    settings = dataclasses.replace(base, experts=(spec,))
    anchors = load_anchors(settings.raw_store.parent / "anchors.json")
    frames = DatasetLoader(settings).val()
    neo = [f for f in frames if f.label == 1]; ndbe = [f for f in frames if f.label == 0]
    sample = neo + random.Random(0).sample(ndbe, 60)
    clients = {spec.key: ProxyVLMClient(spec, settings)}
    cache = FileVoteCache(settings.raw_label_dir)
    names = [c.name for c in CONCEPTS]
    disc = [c.name for c in CONCEPTS if c.role == "discriminative"]

    s_single = dataclasses.replace(settings, extraction_strategy="single", votes_per_expert=3)
    s_multi = dataclasses.replace(settings, extraction_strategy="multistage", ms_votes_per_expert=2)
    a_s = run_strategy("single", s_single, clients, cache, sample, anchors, 16)
    a_m = run_strategy("multistage", s_multi, clients, cache, sample, anchors, 16)
    m_s, _, pos = metrics(a_s, sample, names)
    m_m, _, _ = metrics(a_m, sample, names)

    def g(m, grp, f): return float(np.nanmean([m[n][f] for n in grp]))
    rel_s, ass_s = disc_reliability(a_s, sample, disc)
    rel_m, ass_m = disc_reliability(a_m, sample, disc)
    calls_s = api_calls(cache, sample, anchors, "proagent", "single", base.extraction_version)
    calls_m = api_calls(cache, sample, anchors, "proagent", "multistage", base.extraction_version)

    print(f"# Single vs Multistage — val {len(sample)} frames ({pos} neo), expert=proagent\n")
    print(f"{'metric':34}{'single':>10}{'multistage':>12}{'Δ':>9}")
    print("-" * 65)
    print(f"{'core AUROC (assessable)':34}{g(m_s,ROBUST_CORE,'auroc_ass'):>10.3f}"
          f"{g(m_m,ROBUST_CORE,'auroc_ass'):>12.3f}{g(m_m,ROBUST_CORE,'auroc_ass')-g(m_s,ROBUST_CORE,'auroc_ass'):>+9.3f}")
    print(f"{'discriminative AUROC (assessable)':34}{g(m_s,disc,'auroc_ass'):>10.3f}"
          f"{g(m_m,disc,'auroc_ass'):>12.3f}{g(m_m,disc,'auroc_ass')-g(m_s,disc,'auroc_ass'):>+9.3f}")
    print(f"{'reliability r (discriminative)':34}{rel_s:>10.3f}{rel_m:>12.3f}{rel_m-rel_s:>+9.3f}")
    print(f"{'assessable m (discriminative)':34}{ass_s:>10.3f}{ass_m:>12.3f}{ass_m-ass_s:>+9.3f}")
    print(f"{'API calls (total, sample)':34}{calls_s:>10d}{calls_m:>12d}{f'{calls_m/max(calls_s,1):.1f}x':>9}")
    print(f"{'API calls / frame':34}{calls_s/len(sample):>10.1f}{calls_m/len(sample):>12.1f}"
          f"{f'{calls_m/max(calls_s,1):.1f}x':>9}")


if __name__ == "__main__":
    main()
