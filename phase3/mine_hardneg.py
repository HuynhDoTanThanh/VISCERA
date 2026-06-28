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
    a = ap.parse_args()
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
