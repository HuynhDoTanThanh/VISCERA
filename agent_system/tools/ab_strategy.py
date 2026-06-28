"""A/B the extraction strategies on val: does multi-stage beat single-shot on LABEL ACCURACY?

For a val sample (all neo + sampled ndbe), runs each strategy through the real Extractor+Aggregator
(same anchors, same frames, same expert) and scores the aggregated concept values against the
neo/ndbe label:

  AUROC   per-concept discriminativeness (direction-agnostic) — the accuracy proxy
  reliab  mean within-reading agreement
  assess  mean assessable fraction (multi-stage masks low-confidence cells → expected lower)

    python -m agent_system.tools.ab_strategy --expert proagent --n-ndbe 60 \
        --single-votes 3 --ms-votes 2 --workers 16

Writes outputs/reports/ab_strategy.md.
"""
from __future__ import annotations
import argparse
import dataclasses
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

from ..application.aggregation import Aggregator
from ..application.extraction import Extractor
from ..application.strategies import MultiStageStrategy, SingleShotStrategy
from ..config import Settings
from ..domain.concepts import ROBUST_CORE
from ..infrastructure.cache import FileVoteCache
from ..infrastructure.dataset import DatasetLoader
from ..infrastructure.vlm import ProxyVLMClient
from ..cli import load_anchors
from ..domain.concept_schema import CONCEPTS


def _auroc(y, col):
    try:
        a = roc_auc_score(y, col)
        return float(max(a, 1 - a))
    except ValueError:
        return float("nan")


def run_strategy(name, settings, clients, cache, frames, anchors, workers):
    strat = MultiStageStrategy(settings) if name == "multistage" else SingleShotStrategy(settings)
    ext = Extractor(clients, cache, settings, strat)
    agg = Aggregator()

    def work(fr):
        votes, _ = ext.extract(fr, anchors)
        return fr, (agg.aggregate(fr, votes) if votes else None)

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, fr) for fr in frames]
        done = 0
        for f in as_completed(futs):
            fr, a = f.result()
            done += 1
            if a is not None:
                out[fr.path] = a
            if done % 20 == 0:
                print(f"    [{name}] {done}/{len(frames)}", flush=True)
    return out


def metrics(aggs, frames, names):
    ok = [fr for fr in frames if fr.path in aggs]
    y = np.array([fr.label for fr in ok])
    rows = {}
    for n in names:
        col = np.array([aggs[fr.path].cells[n].value for fr in ok])
        msk = np.array([aggs[fr.path].cells[n].mask for fr in ok])
        rel = np.mean([aggs[fr.path].cells[n].reliability for fr in ok])
        # fair metric: AUROC over ASSESSABLE cells only (masked = not supervised, so excluded)
        sel = msk > 0
        auroc_ass = (_auroc(y[sel], col[sel]) if sel.sum() >= 10 and len(set(y[sel])) == 2
                     else float("nan"))
        rows[n] = {"auroc": _auroc(y, col), "auroc_ass": auroc_ass,
                   "rel": float(rel), "assess": float(np.mean(msk))}
    return rows, len(ok), int(y.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="proagent")
    ap.add_argument("--n-ndbe", type=int, default=60)
    ap.add_argument("--single-votes", type=int, default=3)
    ap.add_argument("--ms-votes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = Settings()
    spec = next(e for e in base.experts if e.key == args.expert)
    settings = dataclasses.replace(base, experts=(spec,))
    settings.ensure_dirs()

    anchors = load_anchors(settings.raw_store.parent / "anchors.json")
    frames = DatasetLoader(settings).val()
    neo = [f for f in frames if f.label == 1]
    ndbe = [f for f in frames if f.label == 0]
    rng = random.Random(args.seed)
    sample = neo + rng.sample(ndbe, min(args.n_ndbe, len(ndbe)))
    print(f"[ab] expert={args.expert} sample={len(sample)} ({len(neo)} neo) anchors={len(anchors)}")

    clients = {spec.key: ProxyVLMClient(spec, settings)}
    cache = FileVoteCache(settings.raw_label_dir)
    names = [c.name for c in CONCEPTS]

    s_single = dataclasses.replace(settings, extraction_strategy="single",
                                   votes_per_expert=args.single_votes)
    s_multi = dataclasses.replace(settings, extraction_strategy="multistage",
                                  ms_votes_per_expert=args.ms_votes)

    print("[ab] running single-shot…")
    a_single = run_strategy("single", s_single, clients, cache, sample, anchors, args.workers)
    print("[ab] running multistage…")
    a_multi = run_strategy("multistage", s_multi, clients, cache, sample, anchors, args.workers)

    m_single, n_s, pos = metrics(a_single, sample, names)
    m_multi, n_m, _ = metrics(a_multi, sample, names)

    disc = [c.name for c in CONCEPTS if c.role == "discriminative"]
    allc = [c.name for c in CONCEPTS]

    def gmean(m, group, fld):
        return float(np.nanmean([m[n][fld] for n in group]))

    def line(label, group):
        ss, mm = gmean(m_single, group, "auroc_ass"), gmean(m_multi, group, "auroc_ass")
        return f"**{label} AUROC(assessable)  single {ss:.3f} → multistage {mm:.3f}  (Δ {mm-ss:+.3f})**"

    L = [f"# A/B: extraction strategy — val ({len(sample)} frames, {pos} neo, expert={args.expert})\n",
         f"- single-shot: {args.single_votes} votes · multistage {base.extraction_version}: "
         f"{args.ms_votes} votes (context/region/chunks/morphology/refine/confirm) · "
         f"ALL {len(allc)} classes enabled\n",
         "- AUROC = aggregated value vs label (masked→0); AUROC_ass = ASSESSABLE cells only "
         "(fair view — masked cells aren't supervised)\n",
         line("core (7)", ROBUST_CORE),
         line("discriminative (15)", disc),
         line(f"ALL classes ({len(allc)})", allc),
         "",
         f"{'concept':22} {'AUR_s':>6} {'AUR_m':>6} {'Δ':>6} | {'ASSok_s':>7} {'ASSok_m':>7} {'Δass':>6} "
         f"{'ass_s':>6} {'ass_m':>6}",
         "-" * 92]
    for n in ROBUST_CORE + tuple(c for c in names if c not in ROBUST_CORE):
        s, m = m_single[n], m_multi[n]
        d = m["auroc"] - s["auroc"]
        da = (m["auroc_ass"] - s["auroc_ass"]) if not (np.isnan(m["auroc_ass"]) or np.isnan(s["auroc_ass"])) else float("nan")
        mark = "  ←core" if n in ROBUST_CORE else ""
        L.append(f"{n:22} {s['auroc']:6.3f} {m['auroc']:6.3f} {d:+6.3f} | "
                 f"{s['auroc_ass']:7.3f} {m['auroc_ass']:7.3f} {da:+6.3f} "
                 f"{s['assess']:6.2f} {m['assess']:6.2f}{mark}")
    report = "\n".join(L)
    (settings.raw_store.parent.parent.parent / "outputs" / "reports" / "ab_strategy.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
