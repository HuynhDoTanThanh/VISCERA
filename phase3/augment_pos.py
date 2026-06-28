"""Positive augmentation featurizer — expand the 127 train neo with cross-center-relevant views.

With only 127 positives, the head can't learn a center-invariant positive boundary. Photometric jitter
(brightness/contrast/gamma/saturation) simulates the scope/lighting/color differences BETWEEN centers —
the exact nuisance variation a new center introduces — while hflip/small-rotation add geometric variety.
Each augmented view is featurized through the frozen backbone and added as an extra positive at train time.

Usage:
    .venv/bin/python -m phase3.augment_pos --out phase3/cache/feats_train_neo_aug.npz --views 8
"""
from __future__ import annotations
import argparse
import csv
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from phase3.featurize import load_backbone, _MEAN, _STD, IMG, POOL_NAMES, _pool


def _views(im: Image.Image, k: int):
    """Deterministic augmentation views simulating cross-center nuisance variation."""
    base = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.adjust_brightness(x, 1.25),
        lambda x: TF.adjust_brightness(x, 0.8),
        lambda x: TF.adjust_contrast(x, 1.3),
        lambda x: TF.adjust_gamma(x, 0.8),
        lambda x: TF.adjust_saturation(x, 1.3),
        lambda x: TF.hflip(TF.adjust_brightness(x, 0.85)),
        lambda x: TF.rotate(x, 8, fill=0),
        lambda x: TF.rotate(x, -8, fill=0),
        lambda x: TF.adjust_hue(x, 0.03),
    ]
    return [f(im) for f in base[:k]]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/train.csv")
    ap.add_argument("--out", default="phase3/cache/feats_train_neo_aug.npz")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv)) if int(r.get("label", -1)) == 1]
    print(f"{len(rows)} train positives x {a.views} views = {len(rows)*a.views} augmented features")
    model, dev = load_backbone()

    feats, names, centers = [], [], []
    batch, meta = [], []
    def flush():
        if not batch:
            return
        xs = torch.stack(batch).to(dev)
        pooled = _pool(model.forward_features(xs)).float().cpu().numpy()
        feats.append(pooled)
        for nm, ce in meta:
            names.append(nm); centers.append(ce)
        batch.clear(); meta.clear()

    for r in rows:
        im = Image.open(r["path"]).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
        nm0 = os.path.splitext(os.path.basename(r["path"]))[0]
        for vi, v in enumerate(_views(im, a.views)):
            x = (torch.from_numpy(np.asarray(v, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0 - _MEAN) / _STD
            batch.append(x); meta.append((f"{nm0}__v{vi}", r["center"]))
            if len(batch) >= a.batch_size:
                flush()
    flush()
    feats = np.concatenate(feats)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, feats=feats, names=np.array(names), center=np.array(centers),
                        label=np.ones(len(names), np.int64), pool_names=np.array(POOL_NAMES), embed=768)
    print(f"saved {feats.shape} -> {a.out}")


if __name__ == "__main__":
    main()
