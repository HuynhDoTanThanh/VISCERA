"""Compare candidate VLM labelers on the SAME 5 few-shot anchors and the SAME train sample.

Which model should generate the foundation labels? We hold the anchors (fewshot_anchors.json)
and the evaluation frames fixed, then extract the concept vector with each model and score:

  AUROC      per-concept discriminativeness vs the neo/ndbe label (direction-agnostic)
  reliab (r) within-vote agreement  (how self-consistent the model is)
  assess (m) how often it can rate the feature at all
  agree      mean pairwise agreement@0.5 with the other models (consensus / independence)

A good labeler maximizes AUROC and reliability over the robust core while staying assessable.

    .venv/bin/python -m scripts.compare_models --n-neo 127 --n-ndbe 200 --workers 24 \
        --models flash3 flash35 proagent

Writes outputs/reports/model_comparison.{md,json}. Extraction is cached per (frame, model,
anchors), so re-runs are free and adding a model only extracts the new one.
"""
from __future__ import annotations
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from cf import config
from cf.aggregate import aggregate_frame
from cf.concept_schema import BY_NAME, CONCEPTS
from cf.confirm import load_split
from cf.extract import extract_frame

ROBUST_CORE = ["mucosal_irregularity", "nodularity", "demarcation", "lesion_present",
               "focal_erythema", "surface_effacement", "colocalization"]


def load_anchors():
    p = config.REPORT_DIR / "fewshot_anchors.json"
    a = json.loads(p.read_text())
    return [(x["path"], x["cls"], x.get("center", "")) for x in a]


def auroc(y, col):
    try:
        a = roc_auc_score(y, col)
        return float(max(a, 1 - a))
    except ValueError:
        return float("nan")


def extract_all(sample, anchors, model_key, workers):
    """Re-extract the sample with one model; return {path: aggregate_frame dict}."""
    def work(fr):
        try:
            rec = extract_frame(fr["path"], anchors, experts=[model_key])
            return fr["path"], aggregate_frame(rec)
        except Exception:  # noqa: BLE001
            return fr["path"], None
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, fr) for fr in sample]
        done = 0
        for f in as_completed(futs):
            p, agg = f.result()
            done += 1
            if agg is not None:
                out[p] = agg
            if done % 25 == 0:
                print(f"    {model_key}: {done}/{len(sample)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["flash3", "flash35", "proagent"])
    ap.add_argument("--n-neo", type=int, default=127)
    ap.add_argument("--n-ndbe", type=int, default=200)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    anchors = load_anchors()
    anchor_paths = {p for p, _, _ in anchors}
    _, frames = load_split(str(config.CACHE_DIR / "index_train.json"))
    neo = [f for f in frames if f["label"] == 1 and f["path"] not in anchor_paths]
    ndbe = [f for f in frames if f["label"] == 0 and f["path"] not in anchor_paths]
    rng = random.Random(args.seed)
    sample = (rng.sample(neo, min(args.n_neo, len(neo)))
              + rng.sample(ndbe, min(args.n_ndbe, len(ndbe))))
    y = np.array([f["label"] for f in sample])
    print(f"[compare] models={args.models}  sample={len(sample)} "
          f"({y.sum()} neo / {(y == 0).sum()} ndbe)  anchors={len(anchors)}")

    names = [c.name for c in CONCEPTS]
    per_model = {}
    for mk in args.models:
        print(f"[compare] extracting with {mk} ({config.COMPARE_MODELS.get(mk, mk)})…", flush=True)
        aggs = extract_all(sample, anchors, mk, args.workers)
        # align matrices over frames that succeeded for this model
        ok = [f for f in sample if f["path"] in aggs]
        yk = np.array([f["label"] for f in ok])
        C = {n: np.array([aggs[f["path"]][n]["c"] for f in ok]) for n in names}
        R = {n: np.array([aggs[f["path"]][n]["r"] for f in ok]) for n in names}
        M = {n: np.array([aggs[f["path"]][n]["m"] for f in ok]) for n in names}
        per_concept = {n: {"auroc": auroc(yk, C[n]), "r": float(R[n].mean()),
                           "m": float(M[n].mean())} for n in names}
        per_model[mk] = {"model": config.COMPARE_MODELS.get(mk, mk), "n": len(ok),
                         "per_concept": per_concept, "_C": C, "_paths": [f["path"] for f in ok]}

    # inter-model agreement@0.5 on the robust core (over frames all models share)
    shared = set.intersection(*[set(per_model[mk]["_paths"]) for mk in args.models]) if per_model else set()
    agree = {mk: [] for mk in args.models}
    if len(args.models) > 1 and shared:
        idx = {mk: {p: i for i, p in enumerate(per_model[mk]["_paths"])} for mk in args.models}
        for n in ROBUST_CORE:
            for a in range(len(args.models)):
                for b in range(a + 1, len(args.models)):
                    ma, mb = args.models[a], args.models[b]
                    va = np.array([per_model[ma]["_C"][n][idx[ma][p]] for p in shared])
                    vb = np.array([per_model[mb]["_C"][n][idx[mb][p]] for p in shared])
                    ag = float(np.mean((va >= 0.5) == (vb >= 0.5)))
                    agree[ma].append(ag)
                    agree[mb].append(ag)

    # summaries
    summary = {}
    for mk in args.models:
        pc = per_model[mk]["per_concept"]
        core_auroc = float(np.nanmean([pc[n]["auroc"] for n in ROBUST_CORE]))
        core_r = float(np.mean([pc[n]["r"] for n in ROBUST_CORE]))
        core_m = float(np.mean([pc[n]["m"] for n in ROBUST_CORE]))
        summary[mk] = {"model": per_model[mk]["model"], "n": per_model[mk]["n"],
                       "core_auroc": core_auroc, "core_reliability": core_r,
                       "core_assessable": core_m,
                       "mean_intermodel_agree": float(np.mean(agree[mk])) if agree[mk] else None,
                       "labeler_score": round(core_auroc * core_r, 4)}

    # report
    out = {"models": args.models, "n_sample": len(sample), "pos": int(y.sum()),
           "anchors": len(anchors), "robust_core": ROBUST_CORE,
           "summary": summary,
           "per_concept": {mk: per_model[mk]["per_concept"] for mk in args.models}}
    (config.REPORT_DIR / "model_comparison.json").write_text(json.dumps(out, indent=2))

    L = [f"# Model comparison — VLM labeler ({len(sample)} train frames, {int(y.sum())} neo, "
         f"5 few-shot anchors)\n",
         f"{'model':14} {'backend':22} {'coreAUROC':>9} {'coreRel':>8} {'coreAssess':>10} "
         f"{'agree':>6} {'score':>6}",
         "-" * 84]
    for mk in sorted(args.models, key=lambda m: -summary[m]["labeler_score"]):
        s = summary[mk]
        ag = f"{s['mean_intermodel_agree']:.2f}" if s["mean_intermodel_agree"] is not None else "  -"
        L.append(f"{mk:14} {s['model']:22} {s['core_auroc']:9.3f} {s['core_reliability']:8.2f} "
                 f"{s['core_assessable']:10.2f} {ag:>6} {s['labeler_score']:6.3f}")
    best = max(args.models, key=lambda m: summary[m]["labeler_score"])
    L += ["", f"**Best labeler (core AUROC × reliability): `{best}` "
          f"({summary[best]['model']})**", "",
          "## Per-concept AUROC (robust core)",
          f"{'concept':22} " + " ".join(f"{m:>10}" for m in args.models)]
    for n in ROBUST_CORE:
        L.append(f"{n:22} " + " ".join(f"{per_model[m]['per_concept'][n]['auroc']:10.3f}"
                                       for m in args.models))
    (config.REPORT_DIR / "model_comparison.md").write_text("\n".join(L))
    print("\n" + "\n".join(L))
    print(f"\n[compare] wrote {config.REPORT_DIR/'model_comparison.md'} and .json")


if __name__ == "__main__":
    main()
