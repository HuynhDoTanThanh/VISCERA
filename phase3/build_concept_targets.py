"""Build the concept-distillation target matrix for Stage-1 pretraining.

For every frame with a VLM concept JSON, emit (image_path, value[35], trust[35], supervise[35], center, label).
This is the supervision for concept-supervised pretraining (pretrain_concept.py): the encoder learns to predict
the VLM's clinical concepts on the large 100k corpus, trust-weighted and supervise-masked.

Concept ordering = agent_system/domain/concept_schema.py (single source of truth, 35 concepts).

    .venv/bin/python -m phase3.build_concept_targets --out phase3/cache/concept_targets.npz
"""
from __future__ import annotations
import argparse
import csv as csvmod
import glob
import json
import os
import numpy as np

from phase3.dataset import CONCEPT_NAMES, NC


def _arrays(rec):
    cj = rec.get("concepts", {})
    v = np.zeros(NC, np.float32); t = np.zeros(NC, np.float32); s = np.zeros(NC, np.float32)
    for i, nm in enumerate(CONCEPT_NAMES):
        c = cj.get(nm)
        if c:
            v[i] = float(c.get("value", 0.0)); t[i] = float(c.get("trust", 0.0))
            s[i] = 1.0 if c.get("supervise", False) else 0.0
    return v, t, s


def scan_unlabeled(label_root="out"):
    rows = []
    for fp in glob.glob(os.path.join(label_root, "[0-9]*", "labels", "*.json")):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        parts = fp.split(os.sep)
        dir_ = parts[-3]; name = d.get("name", os.path.splitext(parts[-1])[0])
        img = os.path.join(label_root, dir_, "images", f"{name}.jpg")
        rows.append((img, name, "", int(d.get("label", -1)), _arrays(d)))
    return rows


def scan_labeled(csv_path, label_dir):
    """Derive labeled frames + concepts DIRECTLY from out/<split>/labels JSONs
    (img = out/<split>/images/<name>.jpg). Self-contained: needs NO dataset.zip and NO CSV — works on the
    Colab out-folder layout and locally. (csv_path is accepted for CLI back-compat but is not required.)"""
    split_root = os.path.dirname(label_dir.rstrip("/"))  # e.g. out/train
    rows = []
    for fp in glob.glob(os.path.join(label_dir, "*.json")):
        d = json.load(open(fp))
        if int(d.get("label", -1)) < 0:
            continue
        nm = d.get("name", os.path.splitext(os.path.basename(fp))[0])
        img = os.path.join(split_root, "images", f"{nm}.jpg")
        rows.append((img, nm, d.get("center", ""), int(d["label"]), _arrays(d)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase3/cache/concept_targets.npz")
    ap.add_argument("--label-root", default="out")
    ap.add_argument("--train-csv", default="dataset/train.csv")
    ap.add_argument("--include-labeled", action="store_true", default=True)
    a = ap.parse_args()

    rows = scan_unlabeled(a.label_root)
    print(f"unlabeled with concepts: {len(rows)}")
    if a.include_labeled:
        lab = scan_labeled(a.train_csv, os.path.join(a.label_root, "train", "labels"))
        print(f"labeled train with concepts: {len(lab)}")
        rows += lab

    # keep only rows whose image exists
    keep = [r for r in rows if os.path.exists(r[0])]
    print(f"images present on disk: {len(keep)} / {len(rows)}")

    paths = np.array([r[0] for r in keep]); names = np.array([r[1] for r in keep])
    center = np.array([r[2] for r in keep]); label = np.array([r[3] for r in keep], np.int64)
    value = np.stack([r[4][0] for r in keep]); trust = np.stack([r[4][1] for r in keep])
    sup = np.stack([r[4][2] for r in keep])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, paths=paths, names=names, center=center, label=label,
                        value=value, trust=trust, supervise=sup, concept_names=np.array(CONCEPT_NAMES))
    print(f"saved {value.shape} concept targets -> {a.out}")
    # quick reliability summary
    print("mean supervise% per concept (top/bottom):")
    sm = sup.mean(0) * 100
    order = np.argsort(-sm)
    for i in list(order[:5]) + list(order[-5:]):
        print(f"  {CONCEPT_NAMES[i]:24s} sup={sm[i]:5.1f}%  trust={trust[:,i].mean():.2f}")


if __name__ == "__main__":
    main()
