"""Mine PSEUDO-POSITIVES from the unlabeled pool — the positive-side lever.

WHY THIS EXISTS
---------------
Every use of the 288k-frame pool so far has been on the NEGATIVE / regularisation side: Mean-Teacher
consistency, one-sided-PU confident-negative targets, hard-FP mining. But the measured leaderboard
deficit is AUPRC (exp6 0.390 vs the RARE25 winner's 0.822) with AUROC nearly matched (0.860 vs 0.920).
AUPRC is dominated by how well the HARD POSITIVES rank — and the binding constraint on that is that we
train on 127 positives.

The pool composition (measured on out/, 288,711 frames):
    CONFIDENT_NEGATIVE  214,584      HARD_NEG_CANDIDATE  61,567      ABSTAIN  12,558
    VLM suspicion > 0.9  16,961
At a plausible ~1% true prevalence the pool holds on the order of ~2,900 unlabeled true positives —
about 23x the labeled positive count — concentrated exactly in the two buckets the pipeline quarantines.

THE PU PROBLEM AND THE TRIPLE GATE
----------------------------------
Naively promoting high-suspicion frames is dangerous: a pseudo-positive that is really NDBE teaches the
model that NDBE looks neoplastic, which RAISES FPR@90R — the precise opposite of the goal.

BUT: measured against ground truth on the 3,095 labeled frames, VLM suspicion is a MUCH better
high-precision detector than the docs assumed:

    suspicion >= 0.5   n=139   precision 66.9%   recall 58.9%
    suspicion >= 0.8   n= 70   precision 90.0%   recall 39.9%
    suspicion >= 0.9   n= 32   precision 100.0%  recall 20.3%      (base rate 5.11%)

Zero false positives among 2,937 labeled negatives at s>=0.9. The docs' "VLM baseline PPV@90R ~0.04"
is a different operating point entirely — the VLM is a poor ranker at HIGH RECALL but an excellent
detector at the TOP. We only need the top.

THE ONE UNKNOWN: the pool has 5.88% of frames at s>=0.9 versus 1.03% in the labeled set (5.7x more,
proportionally). Either the VLM over-flags raw video frames, or the pool is genuinely positive-rich —
both fit the data, and they imply very different precision (<=17% if the pool is 1% positive, ~85% if
it is 5%). So suspicion is used as a GATE, not as a label, the set is CAPPED, and the target is SOFT.

A frame is promoted only if four weakly-correlated signals agree:

  A. CONCEPT gate  — trust-weighted score on the DECISIVE architectural/vascular hallmarks
     (demarcation, nodularity, vascular_irregularity, ...) at or above the labeled-positive percentile.
     This is exactly the inverse of the PU guard mine_hardneg.py already uses to EXCLUDE likely positives.
  B. MODEL gate    — high probability under the trained detector. Pass models from a DIFFERENT fold than
     the one you will train (`--score-with`), otherwise self-training just amplifies existing bias.
  C. BUCKET gate   — candidates come only from HARD_NEG_CANDIDATE / ABSTAIN. CONFIDENT_NEGATIVE is where
     the VLM was sure it is normal mucosa; promoting from there is almost pure noise.
  D. SUSPICION gate — VLM overall suspicion >= --susp-min (default 0.9), the strongest single signal above.

Output is a capped, ranked path list. Train with `finetune.py --pos-list ... --pos-soft 0.85`, which adds
them as SOFT positives and — critically — excludes them from the soft-pAUC threshold quantile so a wrong
pseudo-positive can never set the operating point.

    python -m phase3.mine_pseudopos --concept-targets phase3/cache/concept_targets.npz \
        --manifest phase3/cache/unl_manifest.npz \
        --score-with fold1_seed0.pt,fold1_seed1.pt --topn 300 --out phase3/cache/unl_pseudopos.txt
"""
from __future__ import annotations
import argparse
import os
import numpy as np


def concept_score(z, concepts):
    """Trust-weighted mean of the given concepts (same formula as mine_hardneg.cscore)."""
    names = [str(x) for x in z["concept_names"]]
    idx = {n: i for i, n in enumerate(names)}
    di = [idx[n] for n in concepts if n in idx]
    if not di:
        raise SystemExit(f"none of {concepts} are in the concept matrix")
    val, trust = z["value"], z["trust"]
    w = trust[:, di]
    return (val[:, di] * w).sum(1) / np.clip(w.sum(1), 1e-6, None), [names[i] for i in di]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept-targets", default="phase3/cache/concept_targets.npz")
    ap.add_argument("--manifest", default="phase3/cache/unl_manifest.npz",
                    help="unl_manifest.npz — supplies the VLM decision bucket per frame")
    ap.add_argument("--score-with", default="",
                    help="comma-sep .pt detectors used for the MODEL gate. Use models from a DIFFERENT fold "
                         "than the one you will train, or self-training amplifies its own bias. "
                         "Empty = concept+bucket gates only (weaker; not recommended for a ship).")
    ap.add_argument("--decisive",
                    default="demarcation,nodularity,vascular_irregularity,focal_abnormal_vessels,"
                            "depression_ulceration,surface_effacement,dilated_vessels",
                    help="DECISIVE hallmark concepts (AUROC ~0.87-0.91 vs the neo label)")
    ap.add_argument("--pos-pct", type=float, default=50.0,
                    help="concept gate: keep candidates whose decisive score >= this percentile of the LABELED "
                         "positives. 50 = the median positive. Raise for higher precision / fewer frames.")
    ap.add_argument("--buckets", default="HARD_NEG_CANDIDATE,ABSTAIN",
                    help="VLM decision buckets to mine. Deliberately EXCLUDES CONFIDENT_NEGATIVE.")
    ap.add_argument("--model-min", type=float, default=0.5,
                    help="model gate: minimum detector probability")
    ap.add_argument("--susp-min", type=float, default=0.9,
                    help="gate D — VLM suspicion. MEASURED against ground truth on the 3,095 labeled frames: "
                         "s>=0.8 -> 90.0%% precision (n=70), s>=0.9 -> 100%% precision (n=32, ZERO false positives "
                         "among 2,937 negatives) vs a 5.11%% base rate. This is the single strongest gate. "
                         "CAVEAT: the pool has 5.88%% of frames at s>=0.9 vs 1.03%% in the labeled set (5.7x), so "
                         "that precision does NOT transfer directly — either the VLM over-flags raw video frames, or "
                         "the pool is genuinely positive-rich. Hence the cap and the soft target.")
    ap.add_argument("--topn", type=int, default=300,
                    help="cap. Keep this well under the labeled positive count (127) unless the LOCO gate says "
                         "otherwise — pseudo-positives are the highest-variance lever in the pipeline.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-score", type=int, default=40000,
                    help="cap how many concept-gated candidates get scored by the model (GPU time)")
    ap.add_argument("--out", default="phase3/cache/unl_pseudopos.txt")
    a = ap.parse_args()

    z = np.load(a.concept_targets, allow_pickle=True)
    paths = z["paths"].astype(str)
    lab = z["label"]
    decis, used = concept_score(z, a.decisive.split(","))
    print(f"concept matrix: {len(paths):,} frames | decisive concepts used: {used}")

    # ---- gate A: decisive-concept score vs the labeled positives -------------------------------------
    labeled = lab >= 0
    if not (lab == 1).any():
        raise SystemExit("no labeled positives in the concept matrix — cannot calibrate the concept gate")
    thr = float(np.percentile(decis[labeled & (lab == 1)], a.pos_pct))
    unl = lab < 0
    gate_a = unl & (decis >= thr)
    print(f"gate A (concept >= p{a.pos_pct:g} of labeled positives = {thr:.4f}): "
          f"{int(gate_a.sum()):,} / {int(unl.sum()):,} unlabeled")

    # ---- gates C (bucket) + D (VLM suspicion) -------------------------------------------------------
    want = {b.strip() for b in a.buckets.split(",")}
    m = np.load(a.manifest, allow_pickle=True)
    mp = m["img_path"].astype(str)
    dec_by_path = dict(zip(mp, m["decision"].astype(str)))
    susp_by_path = dict(zip(mp, m["suspicion"].astype(float)))
    in_bucket = np.array([dec_by_path.get(p, "?") in want for p in paths])
    hi_susp = np.array([susp_by_path.get(p, 0.0) >= a.susp_min for p in paths])
    gate_ac = gate_a & in_bucket & hi_susp
    print(f"gate C (decision in {sorted(want)}): {int((gate_a & in_bucket).sum()):,} survive A+C")
    print(f"gate D (VLM suspicion >= {a.susp_min}): {int(gate_ac.sum()):,} survive A+C+D")
    if gate_ac.sum() == 0:
        raise SystemExit("no candidates survived the concept+bucket gates — loosen --pos-pct")

    cand = paths[gate_ac]
    cand_decis = decis[gate_ac]
    cand = np.array([p for p in cand if os.path.exists(p)])
    if len(cand) == 0:
        raise SystemExit("candidate images do not exist on this machine")
    order_c = np.argsort(-cand_decis[:len(cand)])
    if len(cand) > a.max_score:                       # score the most concept-confident first
        cand = cand[order_c[:a.max_score]]
        print(f"  capped to the {a.max_score:,} most concept-confident candidates for scoring")

    # ---- gate B: the detector ------------------------------------------------------------------------
    if a.score_with:
        from phase3.infer import _score_finetuned
        models = [x.strip() for x in a.score_with.split(",") if x.strip()]
        print(f"scoring {len(cand):,} candidates with {models} ...")
        sc = _score_finetuned(models, list(cand), a.batch_size)
        keep = sc >= a.model_min
        print(f"gate B (model prob >= {a.model_min}): {int(keep.sum()):,} survive A+B+C")
        cand, sc = cand[keep], sc[keep]
        order = np.argsort(-sc)
    else:
        print("gate B SKIPPED (no --score-with) — concept+bucket only, much weaker precision")
        sc = np.full(len(cand), np.nan)
        order = np.arange(len(cand))

    picked = cand[order[:a.topn]]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(picked.tolist()) + ("\n" if len(picked) else ""))
    n_real = int((lab == 1).sum())
    print(f"\nwrote {len(picked)} pseudo-positives -> {a.out}")
    if a.score_with and len(picked):
        print(f"  model prob range: {sc[order[0]]:.3f} .. {sc[order[min(len(picked), len(order)) - 1]]:.3f}")
    print(f"  that is +{len(picked) / max(n_real, 1):.0%} on top of the {n_real} labeled positives")
    print("  TRAIN WITH:  --pos-list %s --pos-soft 0.85" % a.out)
    print("  GATE IT: these are UNVERIFIED labels. A wrong pseudo-positive teaches 'NDBE looks neoplastic'")
    print("  and RAISES FPR@90R. Always A/B on a leak-free LOCO leg before shipping.")


if __name__ == "__main__":
    main()
