"""Optimize generation time for the single+quality+2-model config.

Two experiments:
  --mode votes  : extract MAX votes once per (frame, model), then SUBSAMPLE to measure
                  accuracy + reliability at votes_per_model = 1..MAX (no extra API). Finds the
                  fewest votes that keep performance.
  --mode batch  : throughput probe — extract a small disjoint set at several worker counts,
                  report frames/min + error rate to find where the proxy saturates.

    python -m agent_system.tools.optimize_generation --mode votes --n-ndbe 60 --max-votes 5
    python -m agent_system.tools.optimize_generation --mode batch --probe 40 --workers 8 16 32 48
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

from ..domain.concept_schema import CONCEPTS
from ..config import Settings
from ..domain.concepts import ROBUST_CORE
from ..domain.entities import Vote
from ..application.aggregation import Aggregator
from ..infrastructure.dataset import DatasetLoader
from ..infrastructure.vlm import ProxyVLMClient
from ..prompts import LENSES, SYSTEM_QUALITY, quality_query_prompt
from ..cli import load_anchors

PANEL = ("proagent", "flash3")


def _auroc(y, col):
    sel = ~np.isnan(col)
    if sel.sum() < 10 or len(set(y[sel])) < 2:
        return float("nan")
    a = roc_auc_score(y[sel], col[sel])
    return float(max(a, 1 - a))


def extract_max_votes(clients, frames, anchors, base_temp, max_votes, workers, cache_path):
    """{path: {model: [vote,...]}} — MAX votes per (frame, model). Resumable via cache_path."""
    store = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def work(frame):
        out = {}
        for key, client in clients.items():
            have = store.get(frame.path, {}).get(key, [])
            if len(have) >= max_votes:
                out[key] = have
                continue
            votes = list(have)
            for i in range(len(votes), max_votes):
                v = client.read(SYSTEM_QUALITY, anchors, frame.path,
                                quality_query_prompt(LENSES[i % len(LENSES)]),
                                base_temp + 0.1 * (i % 3), i + 1)
                if v is not None:
                    votes.append(v)
            out[key] = votes
        return frame.path, out

    todo = [f for f in frames
            if any(len(store.get(f.path, {}).get(k, [])) < max_votes for k in clients)]
    t0, calls = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, f) for f in todo]):
            path, out = fut.result()
            store[path] = out
            calls += sum(len(v) for v in out.values())
    cache_path.write_text(json.dumps(store))
    return store, time.time() - t0


def sweep_votes(store, frames, max_votes):
    agg = Aggregator()
    names = [c.name for c in CONCEPTS]
    disc = [c.name for c in CONCEPTS if c.role == "discriminative"]
    rows = []
    for v in range(1, max_votes + 1):
        aggs, y = {}, []
        for fr in frames:
            rec = store.get(fr.path)
            if not rec:
                continue
            pooled = []
            for key in PANEL:
                pooled += [Vote(key, i, val) for i, val in enumerate(rec.get(key, [])[:v])]
            if not pooled:
                continue
            aggs[fr.path] = agg.aggregate(fr, pooled)
            y.append(fr.label)
        y = np.array(y)
        ok = [fr for fr in frames if fr.path in aggs]

        def grp_auroc(group):
            vals = []
            for n in group:
                col = np.array([aggs[fr.path].cells[n].value
                                if aggs[fr.path].cells[n].mask > 0 else np.nan for fr in ok])
                vals.append(_auroc(y, col))
            return float(np.nanmean(vals))

        rel = float(np.mean([aggs[fr.path].cells[n].reliability for fr in ok for n in disc]))
        rows.append({"votes_per_model": v, "total_votes": v * len(PANEL),
                     "core_auroc": grp_auroc(ROBUST_CORE), "disc_auroc": grp_auroc(disc),
                     "reliability": rel})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["votes", "batch"], default="votes")
    ap.add_argument("--n-ndbe", type=int, default=60)
    ap.add_argument("--max-votes", type=int, default=5)
    ap.add_argument("--workers", type=int, nargs="+", default=[16])
    ap.add_argument("--probe", type=int, default=40)
    args = ap.parse_args()

    s = Settings(); s.ensure_dirs()
    clients = {sp.key: ProxyVLMClient(sp, s) for sp in s.experts if sp.key in PANEL}
    anchors = load_anchors(s.raw_store.parent / "anchors.json")
    frames = DatasetLoader(s).val()
    neo = [f for f in frames if f.label == 1]; ndbe = [f for f in frames if f.label == 0]
    rep_dir = s.raw_store.parent.parent.parent / "outputs" / "reports"

    if args.mode == "votes":
        sample = neo + random.Random(0).sample(ndbe, args.n_ndbe)
        cache = s.raw_store / "opt_votes.json"
        print(f"[opt] extracting up to {args.max_votes} votes × {len(clients)} models on "
              f"{len(sample)} frames…")
        store, secs = extract_max_votes(clients, sample, anchors, s.base_temp, args.max_votes,
                                        args.workers[0], cache)
        rows = sweep_votes(store, sample, args.max_votes)
        full = rows[-1]
        L = [f"# Vote sweep — single+quality, panel={'+'.join(PANEL)} "
             f"(val {len(sample)} frames, {len(neo)} neo)\n",
             f"{'v/model':>8}{'total':>7}{'core_AUROC':>12}{'disc_AUROC':>12}{'reliab':>8}"
             f"{'Δdisc vs max':>13}",
             "-" * 60]
        for r in rows:
            L.append(f"{r['votes_per_model']:>8}{r['total_votes']:>7}{r['core_auroc']:>12.3f}"
                     f"{r['disc_auroc']:>12.3f}{r['reliability']:>8.3f}"
                     f"{r['disc_auroc']-full['disc_auroc']:>+13.3f}")
        report = "\n".join(L)
        (rep_dir / "optimize_votes.md").write_text(report)
        print("\n" + report)
        print(f"\n[opt] extraction of {len(sample)} frames @ {args.workers[0]} workers: {secs:.0f}s")
    else:
        # disjoint probe set per worker count (avoid cache reuse skewing timing)
        pool = random.Random(1).sample(ndbe, min(args.probe * len(args.workers), len(ndbe)))
        chunks = [pool[i * args.probe:(i + 1) * args.probe] for i in range(len(args.workers))]
        print(f"[opt] batch probe: {args.probe} frames × {len(clients)} models per worker setting")
        L = [f"# Batch/throughput probe — single+quality, panel={'+'.join(PANEL)}\n",
             f"{'workers':>8}{'frames':>8}{'sec':>8}{'frames/min':>12}{'calls/min':>11}{'errors':>8}",
             "-" * 56]
        for w, chunk in zip(args.workers, chunks):
            t0 = time.time()
            errs = 0
            def work(frame):
                e = 0
                for key, client in clients.items():
                    v = client.read(SYSTEM_QUALITY, anchors, frame.path,
                                    quality_query_prompt(LENSES[0]), s.base_temp, 1)
                    if v is None:
                        e += 1
                return e
            with ThreadPoolExecutor(max_workers=w) as ex:
                for fut in as_completed([ex.submit(work, f) for f in chunk]):
                    errs += fut.result()
            secs = time.time() - t0
            calls = len(chunk) * len(clients)
            L.append(f"{w:>8}{len(chunk):>8}{secs:>8.0f}{len(chunk)/secs*60:>12.1f}"
                     f"{calls/secs*60:>11.1f}{errs:>8}")
        report = "\n".join(L)
        (rep_dir / "optimize_batch.md").write_text(report)
        print("\n" + report)


if __name__ == "__main__":
    main()
