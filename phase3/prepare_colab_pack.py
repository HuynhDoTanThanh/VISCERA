"""Package a self-contained Colab bundle to run the FAIR concept-pretraining test on a GPU.

Bundles: phase3 code + dinov2.pth + a STRATIFIED subset of the concept-labeled frames (oversampling
graded-positive-like frames + diverse negatives across source dirs) + labeled train/val images, with all
paths remapped to relative locations. Run the result on Colab via colab_concept_test.ipynb.

    .venv/bin/python -m phase3.prepare_colab_pack --n 30000 --tar rare26_concept.tar.gz
"""
from __future__ import annotations
import argparse
import csv
import os
import shutil
import tarfile
import numpy as np

from phase3.pretrain_concept import curated_concepts


def dir_of(p):
    parts = p.split(os.sep)
    return parts[-3] if "images" in parts else "lab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="phase3/cache/concept_targets.npz")
    ap.add_argument("--train-csv", default="dataset/train.csv")
    ap.add_argument("--val-csv", default="dataset/val.csv")
    ap.add_argument("--n", type=int, default=30000, help="unlabeled subset size for Stage-1")
    ap.add_argument("--stage", default="rare26_concept")
    ap.add_argument("--tar", default="rare26_concept.tar.gz")
    ap.add_argument("--with-val", action="store_true", help="also bundle val images (not needed for the gate; +~1.9GB)")
    a = ap.parse_args()

    D = np.load(a.targets, allow_pickle=True)
    paths, value, trust, sup = D["paths"], D["value"], D["trust"], D["supervise"]
    center, label = D["center"], D["label"]
    main_idx, _, _ = curated_concepts()
    # unlabeled rows only for Stage-1 subset (labeled handled via csv)
    is_unl = label < 0
    upaths = paths[is_unl]; uval = value[is_unl]; utr = trust[is_unl]; usup = sup[is_unl]
    core_max = uval[:, main_idx].max(1)
    rng = np.random.default_rng(0)
    pos_like = np.where(core_max > 0.4)[0]; rest = np.where(core_max <= 0.4)[0]
    n_pos = min(len(pos_like), a.n // 2)
    n_rest = min(a.n - n_pos, len(rest))
    # diversify negatives across source dirs (round-robin)
    by_dir = {}
    for i in rest:
        by_dir.setdefault(dir_of(upaths[i]), []).append(i)
    order = []
    keys = list(by_dir.values())
    while len(order) < n_rest and any(keys):
        for lst in keys:
            if lst:
                order.append(lst.pop())
            if len(order) >= n_rest:
                break
    sel = np.concatenate([rng.choice(pos_like, n_pos, replace=False), np.array(order[:n_rest], int)])
    print(f"subset: {n_pos} graded-positive-like + {len(sel)-n_pos} diverse negatives across {len(by_dir)} dirs")

    root = a.stage
    img_dir = os.path.join(root, "data", "img"); lab_dir = os.path.join(root, "data", "labimg")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
    os.makedirs(os.path.join(root, "phase3"), exist_ok=True)

    # stage unlabeled subset images with unique names + remap targets
    new_paths = []
    keep = np.zeros(len(sel), bool)
    for k, i in enumerate(sel):
        src = upaths[i]; staged = f"{dir_of(src)}__{os.path.basename(src)}"
        dst = os.path.join(img_dir, staged)
        if os.path.exists(src):
            shutil.copy(src, dst); new_paths.append(os.path.join("data", "img", staged)); keep[k] = True
        else:
            new_paths.append("")
    keep_idx = sel[keep]
    np.savez_compressed(os.path.join(root, "data", "concept_targets.npz"),
                        paths=np.array([p for p in new_paths if p]),
                        value=uval[keep_idx], trust=utr[keep_idx], supervise=usup[keep_idx],
                        center=np.array([""] * int(keep.sum())), label=np.full(int(keep.sum()), -1, np.int64),
                        concept_names=D["concept_names"])
    print(f"staged {int(keep.sum())} unlabeled images")

    # stage labeled train/val images + rewrite csvs
    def stage_csv(src_csv, out_csv):
        rows = list(csv.DictReader(open(src_csv))); n = 0
        for r in rows:
            if os.path.exists(r["path"]):
                b = os.path.basename(r["path"]); shutil.copy(r["path"], os.path.join(lab_dir, b))
                r["path"] = os.path.join("data", "labimg", b); n += 1
        with open(os.path.join(root, "data", os.path.basename(out_csv)), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        return n
    print(f"staged train images: {stage_csv(a.train_csv, a.train_csv)}")
    if a.with_val:
        print(f"staged val images:   {stage_csv(a.val_csv, a.val_csv)}")

    # code + backbone
    for f in os.listdir("phase3"):
        if f.endswith(".py"):
            shutil.copy(os.path.join("phase3", f), os.path.join(root, "phase3", f))
    shutil.copy("dinov2.pth", os.path.join(root, "dinov2.pth"))

    print(f"taring -> {a.tar} ...")
    with tarfile.open(a.tar, "w:gz") as t:
        t.add(root, arcname=os.path.basename(root))
    sz = os.path.getsize(a.tar) / 1e9
    print(f"DONE: {a.tar} ({sz:.1f} GB). Upload to Google Drive, then run phase3/colab_concept_test.ipynb")


if __name__ == "__main__":
    main()
