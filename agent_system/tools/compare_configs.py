"""Compare extraction configs (experts × votes) on the SAME val votes, using the established
ab_strategy metrics. Extracts the max config (2 experts × 3 votes) once — cached/resumable — then
SUBSETS votes to score every smaller config, so all rows share identical underlying readings.

Metrics per config (your ab_strategy definitions):
  AUROC = core-7 AUROC over ASSESSABLE cells (the headline discriminative quality, masked excluded)
  AUR   = discriminative-15 AUROC over assessable cells
  ASS   = mean core assessable fraction (coverage — how many core cells were not masked)
  susp  = AUROC(neo vs ndbe) on the curated frame suspicion (the cascade/triage signal)

    python -m agent_system.tools.compare_configs --n-ndbe 100
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
from ..domain.concepts import ROBUST_CORE, DISCRIMINATIVE_ALL, ALL_CONCEPTS
from ..infrastructure.cache import FileVoteCache
from ..infrastructure.dataset import DatasetLoader
from ..infrastructure.vlm import ProxyVLMClient
from .ab_strategy import _auroc, metrics
from .probe_capacity import load_anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ndbe", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--configs", default="1x1,1x2,1x3,2x1,2x2,2x3",
                    help="comma list of ExV (experts x votes); 2x3 = current method")
    a = ap.parse_args()

    s = dataclasses.replace(Settings(), votes_per_expert=3)  # extract the max; subset down
    s.ensure_dirs()
    expert_order = [e.key for e in s.experts]                 # [proagent, flash3]
    clients = {e.key: ProxyVLMClient(e, s) for e in s.experts}
    cache = FileVoteCache(s.raw_label_dir)
    ext = Extractor(clients, cache, s, SingleShotStrategy(s))
    anchors = load_anchors(s.raw_store.parent / "anchors.json")
    agg, scorer, curator = Aggregator(), TrustScorer(s), Curator(s)

    frames = DatasetLoader(s).val()
    neo = [f for f in frames if f.label == 1]
    ndbe = [f for f in frames if f.label == 0]
    sample = neo + random.Random(0).sample(ndbe, min(a.n_ndbe, len(ndbe)))
    print(f"[cmp] extracting 2 experts × 3 votes on {len(sample)} frames "
          f"({len(neo)} neo) — cached/resumable…")

    async def go():
        sem = asyncio.Semaphore(a.concurrency)
        for c in clients.values():
            c.set_semaphore(sem)
        votes_by = {}
        done = 0
        tasks = [asyncio.create_task(ext.aextract(fr, anchors)) for fr in sample]
        for fr, t in zip(sample, tasks):
            votes, _ = await t
            votes_by[fr.path] = votes
            done += 1
            if done % 25 == 0:
                print(f"    extracted {done}/{len(sample)}", flush=True)
        return votes_by

    votes_by = asyncio.run(go())

    def score_config(n_exp, n_votes):
        keep = set(expert_order[:n_exp])
        aggs = {}
        susp_y, susp_s = [], []
        for fr in sample:
            sub = [v for v in votes_by.get(fr.path, [])
                   if v.expert in keep and v.lens < n_votes]
            if not sub:
                continue
            ag = agg.aggregate(fr, sub)
            aggs[fr.path] = ag
            cf = curator.curate(ag, scorer.score(ag, None), verified=False)
            susp_y.append(fr.label); susp_s.append(cf.suspicion)
        rows, n, npos = metrics(aggs, sample, list(ALL_CONCEPTS))
        core = float(np.nanmean([rows[c]["auroc_ass"] for c in ROBUST_CORE]))
        disc = float(np.nanmean([rows[c]["auroc_ass"] for c in DISCRIMINATIVE_ALL]))
        ass = float(np.nanmean([rows[c]["assess"] for c in ROBUST_CORE]))
        susp = _auroc(np.array(susp_y), np.array(susp_s)) if len(set(susp_y)) == 2 else float("nan")
        return n, core, disc, ass, susp

    print(f"\n{'experts':>8}{'votes':>6}{'calls':>6}{'AUROC':>8}{'AUR':>7}{'ASS':>7}{'susp':>7}  notes")
    print(f"  (core-ass)            {'(core)':>8}{'(disc)':>7}{'(cov)':>7}{'(neo)':>7}")
    print("-" * 70)
    for cfg in a.configs.split(","):
        e, v = (int(x) for x in cfg.lower().split("x"))
        n, core, disc, ass, susp = score_config(e, v)
        note = "← CURRENT" if (e, v) == (2, 3) else ("cheapest" if (e, v) == (1, 1) else "")
        print(f"{e:>8}{v:>6}{e*v:>6}{core:>8.3f}{disc:>7.3f}{ass:>7.2f}{susp:>7.3f}  {note}")
    print("\nAUROC=core-7 assessable · AUR=disc-15 assessable · ASS=core coverage · "
          "susp=neo-vs-ndbe on frame suspicion")


if __name__ == "__main__":
    main()
