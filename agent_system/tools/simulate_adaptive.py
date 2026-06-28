"""Simulate an ADAPTIVE self-cascade on already-cached val votes (NO new API calls).

Idea: pro-agent votes once on every frame. If that 1-vote suspicion is clearly low/high
(confident), keep it (1 call). Only frames in the uncertain band escalate to the full panel.
Confident negatives — the ~93% majority — stop early, so average calls drop sharply while the
hard, ranking-critical frames still get full quality.

Sweeps the band to find the cheapest policy that keeps susp-AUROC(neo) >= target.

    python -m agent_system.tools.simulate_adaptive --target 0.90
"""
from __future__ import annotations
import argparse
import asyncio
import dataclasses
import random

import numpy as np

from ..config import Settings
from ..application.aggregation import Aggregator
from ..application.extraction import Extractor
from ..application.strategies import SingleShotStrategy
from ..application.trust import Curator, TrustScorer
from ..infrastructure.cache import FileVoteCache
from ..infrastructure.dataset import DatasetLoader
from ..infrastructure.vlm import ProxyVLMClient
from .ab_strategy import _auroc
from .probe_capacity import load_anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ndbe", type=int, default=100)
    ap.add_argument("--target", type=float, default=0.90)
    a = ap.parse_args()

    s = dataclasses.replace(Settings(), votes_per_expert=3)
    s.ensure_dirs()
    experts = [e.key for e in s.experts]              # [proagent, flash3]
    clients = {e.key: ProxyVLMClient(e, s) for e in s.experts}
    ext = Extractor(clients, FileVoteCache(s.raw_label_dir), s, SingleShotStrategy(s))
    anchors = load_anchors(s.raw_store.parent / "anchors.json")
    agg, scorer, curator = Aggregator(), TrustScorer(s), Curator(s)

    frames = DatasetLoader(s).val()
    neo = [f for f in frames if f.label == 1]
    ndbe = [f for f in frames if f.label == 0]
    sample = neo + random.Random(0).sample(ndbe, min(a.n_ndbe, len(ndbe)))

    async def load():
        sem = asyncio.Semaphore(48)
        for c in clients.values():
            c.set_semaphore(sem)
        out = {}
        for fr in sample:
            votes, _ = await ext.aextract(fr, anchors)   # cache hit → no API call
            out[fr.path] = votes
        return out
    votes_by = asyncio.run(load())

    def susp(fr, votes):
        if not votes:
            return None
        ag = agg.aggregate(fr, votes)
        return curator.curate(ag, scorer.score(ag, None), verified=False).suspicion

    # per-frame suspicion for each vote-subset we might use
    rows = []
    for fr in sample:
        vs = votes_by.get(fr.path, [])
        sub = lambda ne, nv: [v for v in vs if v.expert in set(experts[:ne]) and v.lens < nv]
        rows.append({
            "y": fr.label,
            "s_pro1": susp(fr, sub(1, 1)),    # 1 call
            "s_full": susp(fr, sub(2, 3)),    # 6 calls (escalation target)
            "s_2x2":  susp(fr, sub(2, 2)),    # 4 calls (alt escalation target)
        })
    rows = [r for r in rows if r["s_pro1"] is not None and r["s_full"] is not None]
    n = len(rows)
    y = np.array([r["y"] for r in rows])

    base_full = _auroc(y, np.array([r["s_full"] for r in rows]))
    base_pro1 = _auroc(y, np.array([r["s_pro1"] for r in rows]))
    print(f"[adaptive] n={n} ({int(y.sum())} neo) | baselines: 1-call={base_pro1:.3f}  "
          f"6-call(full)={base_full:.3f}  target≥{a.target}\n")

    def simulate(lo, hi, esc_key):
        final, calls = [], []
        for r in rows:
            s1 = r["s_pro1"]
            if lo <= s1 <= hi:                       # uncertain → escalate
                final.append(r[esc_key])
                calls.append(6 if esc_key == "s_full" else 4)
            else:                                    # confident → keep 1-call
                final.append(s1)
                calls.append(1)
        au = _auroc(y, np.array(final))
        return au, float(np.mean(calls)), float(np.mean([c > 1 for c in calls]))

    print(f"{'band(lo-hi)':>14}{'escTo':>7}{'AUROC':>8}{'avg_calls':>11}{'escalate%':>11}{'speedup':>9}")
    print("-" * 62)
    best = None
    for esc_key, esc_calls in [("s_full", 6), ("s_2x2", 4)]:
        for lo in [0.05, 0.08, 0.10, 0.12, 0.15]:
            for hi in [0.55, 0.65, 0.75, 0.90]:
                au, avg, frac = simulate(lo, hi, esc_key)
                if au >= a.target:
                    spd = 6.0 / avg
                    tag = ""
                    if best is None or avg < best[3]:
                        best = (lo, hi, esc_key, avg, au, frac, spd); tag = ""
                    print(f"{f'{lo:.2f}-{hi:.2f}':>14}{esc_key[2:]:>7}{au:>8.3f}"
                          f"{avg:>11.2f}{frac*100:>10.0f}%{spd:>8.1f}x")
    if best:
        lo, hi, ek, avg, au, frac, spd = best
        print(f"\n>>> cheapest policy ≥{a.target}: escalate band [{lo:.2f},{hi:.2f}] to "
              f"{'2x3' if ek=='s_full' else '2x2'} → AUROC {au:.3f}, avg {avg:.2f} calls/img "
              f"({spd:.1f}x faster than 6), escalates {frac*100:.0f}% of frames.")
    else:
        print(f"\n>>> no banded policy reached {a.target}. Try --target 0.89 or escalate wider.")


if __name__ == "__main__":
    main()
