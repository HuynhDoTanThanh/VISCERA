"""Build a manifest of all unlabeled frames and emit featurization path-lists for hard-neg mining.

The confident-false-positive tail is what kills PPV@90R. The ~31k HARD_NEG_CANDIDATE frames (high VLM
suspicion, almost-certainly negative at 1% prevalence) are the exact NDBE look-alikes we must mine as
negatives. PU caveat: high-suspicion frames carry >1% true-positive contamination -> the head must treat
the suspicious extremes with PU/abstention care, NOT hard-label them all negative.

Outputs (under phase3/cache/):
  unl_manifest.npz        name, dir, img_path, suspicion, decision, frame_trust  (all unlabeled)
  unl_hardneg.txt         image paths for HARD_NEG_CANDIDATE (mineable hard negatives)
  unl_confneg.txt         a sample of CONFIDENT_NEGATIVE (easy-negative manifold)
  unl_suspicious.txt      top-suspicion frames (PU-uncertain extremes; abstain/inspect, do not hard-label)

Usage:
    .venv/bin/python -m phase3.mine_hardneg --confneg-sample 30000 --suspicious-top 5000
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import numpy as np


def scan(label_root="out"):
    rows = []
    files = glob.glob(os.path.join(label_root, "[0-9]*", "labels", "*.json"))
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        dpart = fp.split(os.sep)
        # out/<dir>/labels/<name>.json -> out/<dir>/images/<name>.jpg
        dir_ = dpart[-3]
        name = d.get("name", os.path.splitext(dpart[-1])[0])
        img = os.path.join(label_root, dir_, "images", f"{name}.jpg")
        rows.append((name, dir_, img, float(d.get("suspicion", 0.0)),
                     d.get("decision", ""), float(d.get("frame_trust", 0.0))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-root", default="out")
    ap.add_argument("--out-dir", default="phase3/cache")
    ap.add_argument("--confneg-sample", type=int, default=30000)
    ap.add_argument("--suspicious-top", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest-only", action="store_true",
                    help="only build unl_manifest.npz (paths+suspicion for the SEMI loss); skip all .txt bucket writes")
    # ---- model-in-the-loop (one-sided PU) mining: score the VLM-negative pool with the CURRENT model,
    #      emit its top false-positives as hard negatives for the next fine-tune round ----
    ap.add_argument("--score-with", default="",
                    help="comma-sep fine-tuned .pt to score the pool with; emits unl_modelFP.txt (model's top FPs "
                         "among VLM-negative frames = the hard negatives that decide PPV@90R). Skips the static writes.")
    ap.add_argument("--pool", default="HARD_NEG_CANDIDATE,CONFIDENT_NEGATIVE",
                    help="VLM decisions eligible to be mined as negatives (very-likely-negative buckets only)")
    ap.add_argument("--topn", type=int, default=3000, help="how many model-FP hard negatives to emit")
    ap.add_argument("--skip-top", type=int, default=200,
                    help="drop the top-scoring candidates (PU guard: the very top may be real positives, not FPs)")
    ap.add_argument("--exclude-dir", default="",
                    help="comma-sep out/<dir> to exclude (anti-leakage: hold the LOCO-val center's pool out of the mine)")
    # ---- CTM: Concept-Confounded Tail Mining. Rank the unlabeled pool by a trust-weighted DIAGNOSTIC-concept
    #      score D (the axis on which NDBE look-alikes confuse the model); emit the confounded band as hard negs.
    #      Runs offline from concept_targets.npz alone (no images / GPU). ----
    ap.add_argument("--concept-rank", default="",
                    help="concept_targets.npz -> mine concept-confounded hard negatives by diagnostic-concept score D")
    ap.add_argument("--concept-conf",
                    default="whitish_focal_area,focal_erythema,color_heterogeneity,color_change_locality,mucosal_irregularity",
                    help="SURFACE/COLOR confounder concepts (NDBE look-alike axis) — mine HIGH on these")
    ap.add_argument("--concept-decisive",
                    default="demarcation,nodularity,vascular_irregularity,focal_abnormal_vessels,depression_ulceration,"
                            "surface_effacement,dilated_vessels",
                    help="ARCHITECTURE/VASCULAR hallmark concepts (strongest neo signal, AUROC~0.90) — PU guard drops "
                         "unlabeled frames HIGH on these (they are the likely unlabeled POSITIVES, not hard negatives)")
    ap.add_argument("--concept-out", default="unl_conceptFP.txt")
    a = ap.parse_args()

    if a.concept_rank:
        # CTM: mine SURFACE-confounded but ARCHITECTURALLY-BLAND unlabeled frames as hard negatives. Concept-space
        # alone can't cleanly separate confounded-neg from unlabeled-pos (positives score high on everything;
        # measured contamination only 5.1%->3.7%), so the real lever is the DECISIVE-hallmark PU GUARD: drop
        # unlabeled frames whose decisive score >= the labeled-positive median (= likely unlabeled positives).
        # Deployable CTM intersects this with the model-FP mine (--score-with) on Colab; this offline pass is the
        # PU-safe candidate set + interpretability. Runs from concept_targets.npz alone (no images/GPU).
        os.makedirs(a.out_dir, exist_ok=True)
        z = np.load(a.concept_rank, allow_pickle=True)
        names = [str(x) for x in z["concept_names"]]; idx = {n: i for i, n in enumerate(names)}
        val, trust, lab, paths = z["value"], z["trust"], z["label"], z["paths"]

        def cscore(concepts):
            di = [idx[n] for n in concepts if n in idx]; w = trust[:, di]
            return (val[:, di] * w).sum(1) / np.clip(w.sum(1), 1e-6, None)

        conf = cscore(a.concept_conf.split(","))             # surface confounder axis
        decis = cscore(a.concept_decisive.split(","))        # decisive hallmarks (neo signal)
        labi = lab >= 0
        pos_decis_med = float(np.median(decis[labi][lab[labi] == 1])) if (lab == 1).any() else 1.0
        unl = lab < 0
        keep = unl & (decis < pos_decis_med)                 # PU guard: exclude likely unlabeled positives
        Ck, Pk = conf[keep], paths[keep]
        order = np.argsort(-Ck)                              # most surface-confounded first
        sel = order[a.skip_top:a.skip_top + a.topn]
        picked = Pk[sel]
        outp = os.path.join(a.out_dir, os.path.basename(a.concept_out))
        with open(outp, "w") as f:
            f.write("\n".join(map(str, picked.tolist())) + ("\n" if len(picked) else ""))
        print(f"CTM: {len(picked)} concept-confounded hard negatives -> {outp}")
        if len(sel):
            print(f"  PU guard kept {int(keep.sum()):,}/{int(unl.sum()):,} unlabeled (decisive<{pos_decis_med:.3f}=pos-median); "
                  f"mined confounder {Ck[sel[0]]:.3f}..{Ck[sel[-1]]:.3f}, skip-top {a.skip_top}")
        return
    os.makedirs(a.out_dir, exist_ok=True)

    rows = scan(a.label_root)
    print(f"scanned {len(rows)} unlabeled frames")
    name = np.array([r[0] for r in rows]); dir_ = np.array([r[1] for r in rows])
    img = np.array([r[2] for r in rows]); susp = np.array([r[3] for r in rows], float)
    dec = np.array([r[4] for r in rows]); ftrust = np.array([r[5] for r in rows], float)
    np.savez_compressed(os.path.join(a.out_dir, "unl_manifest.npz"),
                        name=name, dir=dir_, img_path=img, suspicion=susp, decision=dec, frame_trust=ftrust)

    from collections import Counter
    print("decision counts:", dict(Counter(dec.tolist())))

    if a.manifest_only:                      # SEMI only needs unl_manifest.npz -> skip the unused .txt bucket writes
        print("manifest-only: wrote unl_manifest.npz, skipping .txt buckets")
        return

    if a.score_with:
        from phase3.infer import _score_finetuned
        keep = np.isin(dec, [d.strip() for d in a.pool.split(",")])
        cand = [p for p in img[keep].tolist() if os.path.exists(p)]
        if a.exclude_dir:
            ex = set(d.strip() for d in a.exclude_dir.split(","))
            cand = [p for p in cand if p.split(os.sep)[-3] not in ex]   # out/<dir>/images/<name>
        cand = np.array(cand)
        print(f"scoring {len(cand)} VLM-negative candidates ({a.pool}) with {a.score_with} ...")
        models = [m.strip() for m in a.score_with.split(",") if m.strip()]
        sc = _score_finetuned(models, list(cand))
        order = np.argsort(-sc)                                          # highest model score = hardest FP
        sel = order[a.skip_top:a.skip_top + a.topn]
        picked = cand[sel]
        outp = os.path.join(a.out_dir, "unl_modelFP.txt")
        with open(outp, "w") as f:
            f.write("\n".join(picked.tolist()) + ("\n" if len(picked) else ""))
        rng_lo = sc[order[a.skip_top]] if len(order) > a.skip_top else float("nan")
        rng_hi = sc[sel[-1]] if len(sel) else float("nan")
        print(f"  unl_modelFP.txt: {len(picked)} model-FP hard negatives "
              f"(skipped top {a.skip_top} as PU guard; model-prob range {rng_lo:.3f}..{rng_hi:.3f})")
        return

    rng = np.random.default_rng(a.seed)
    # only keep frames whose image actually exists on disk
    def write(paths, fn):
        exist = [p for p in paths if os.path.exists(p)]
        with open(os.path.join(a.out_dir, fn), "w") as f:
            f.write("\n".join(exist) + ("\n" if exist else ""))
        print(f"  {fn}: {len(exist)} paths (of {len(paths)} requested)")

    hardneg = img[dec == "HARD_NEG_CANDIDATE"]
    write(list(hardneg), "unl_hardneg.txt")

    confneg = img[dec == "CONFIDENT_NEGATIVE"]
    if len(confneg) > a.confneg_sample:
        confneg = confneg[rng.choice(len(confneg), a.confneg_sample, replace=False)]
    write(list(confneg), "unl_confneg.txt")

    order = np.argsort(-susp)
    write(list(img[order[:a.suspicious_top]]), "unl_suspicious.txt")
    print(f"suspicion: >0.7 -> {(susp>0.7).sum()}  >0.85 -> {(susp>0.85).sum()}")


if __name__ == "__main__":
    main()
