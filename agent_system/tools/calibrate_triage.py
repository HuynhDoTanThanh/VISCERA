"""Calibrate the cascade triage stage: can a cheap, low-reasoning model (optionally WITHOUT the
5 anchor images) rank neoplasia above benign well enough to escalate only a small fraction to the
pro panel — without missing positives?

Runs 1-vote single-shot triage on val frames, computes suspicion via the normal agg→score→curate
path, then reports AUROC(neo vs ndbe) and a threshold table (neo-recall vs escalate-fraction).

    python -m agent_system.tools.calibrate_triage                                  # flash-extra-low, no anchors
    python -m agent_system.tools.calibrate_triage --model ag/gemini-3.5-flash-low --anchors
    python -m agent_system.tools.calibrate_triage --n-ndbe 200
"""
from __future__ import annotations
import argparse
import asyncio
import dataclasses
import random
import time

from ..config import Settings, ModelSpec
from ..application.aggregation import Aggregator
from ..application.strategies import SingleShotStrategy
from ..application.trust import Curator, TrustScorer
from ..infrastructure.dataset import DatasetLoader
from ..infrastructure.vlm import ProxyVLMClient
from .probe_capacity import load_anchors


def auroc(labels, scores):
    """Mann-Whitney AUROC (rank-based), no sklearn dependency."""
    pairs = sorted(zip(scores, labels))
    pos = sum(labels); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    # rank-sum over positives
    ranks = {}
    i = 0
    srt = sorted(scores)
    while i < len(srt):
        j = i
        while j < len(srt) and srt[j] == srt[i]:
            j += 1
        avg_rank = (i + j - 1) / 2 + 1   # 1-based average rank for ties
        for k in range(i, j):
            ranks.setdefault(srt[k], avg_rank)
        i = j
    rank_sum = sum(ranks[s] for s, l in zip(scores, labels) if l == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ag/gemini-3.5-flash-extra-low")
    ap.add_argument("--anchors", action="store_true", help="send the 5 anchor images (default: none)")
    ap.add_argument("--n-ndbe", type=int, default=150)
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()

    s = Settings(); s.ensure_dirs()
    client = ProxyVLMClient(ModelSpec("triage", a.model), s)
    anchors = load_anchors(s.raw_store.parent / "anchors.json")
    use_anchors = anchors if a.anchors else []
    agg, scorer, curator = Aggregator(), TrustScorer(s), Curator(s)
    strat = SingleShotStrategy(s)

    frames = DatasetLoader(s).val()
    neo = [f for f in frames if f.label == 1]
    ndbe = [f for f in frames if f.label == 0]
    sample = neo + random.Random(0).sample(ndbe, min(a.n_ndbe, len(ndbe)))
    print(f"[triage] model={a.model} anchors={'yes' if a.anchors else 'NO'} votes={a.votes} "
          f"| {len(neo)} neo + {len(sample)-len(neo)} ndbe")

    from ..domain.entities import Vote

    async def score_frame(frame):
        sem = score_frame.sem
        client.set_semaphore(sem)
        votes = []
        results = await asyncio.gather(
            *[strat.aread(frame, use_anchors, client, seed=i + 1) for i in range(a.votes)])
        for i, (values, _) in enumerate(results):
            if values:
                votes.append(Vote(expert="triage", lens=i, values=values))
        if not votes:
            return None
        ag = agg.aggregate(frame, votes)
        cells = scorer.score(ag, None)
        cf = curator.curate(ag, cells, verified=False)
        return (frame.label, cf.suspicion)

    async def go():
        score_frame.sem = asyncio.Semaphore(a.concurrency)
        t0 = time.time()
        out = await asyncio.gather(*[score_frame(f) for f in sample])
        dt = time.time() - t0
        out = [o for o in out if o is not None]
        labels = [l for l, _ in out]
        scores = [sc for _, sc in out]
        n = len(out)
        print(f"[triage] scored {n}/{len(sample)} in {dt:.0f}s ({n/dt*60:.0f} frames/min)")
        au = auroc(labels, scores)
        print(f"[triage] AUROC(neo vs ndbe) on suspicion = {au:.3f}\n")
        print(f"{'thr':>6}{'neo_recall':>12}{'escalate%':>11}{'neo_kept_cheap':>16}")
        print("-" * 46)
        npos = sum(labels)
        for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            esc = sum(1 for sc in scores if sc >= thr)
            neo_caught = sum(1 for l, sc in out if l == 1 and sc >= thr)
            neo_missed = npos - neo_caught
            print(f"{thr:>6.2f}{neo_caught/npos if npos else 0:>12.2f}"
                  f"{esc/n*100:>10.0f}%{neo_missed:>16}")
        print("\nPick the highest thr with neo_recall=1.00 (0 missed). escalate% = pro-panel cost; "
              "the rest get the cheap triage label.")
    asyncio.run(go())


if __name__ == "__main__":
    main()
