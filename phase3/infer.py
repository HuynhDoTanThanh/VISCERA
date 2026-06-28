"""Offline container entrypoint — image folder/list -> calibrated neoplasia scores (no VLM, no network).

Pipeline: image -> frozen domain-adapted DINOv2 ViT-B/14-reg -> 4-pool embedding -> shipped ensemble
(phase3/ship.py artifact) -> calibrated score. This is the only thing the --network=none container runs.

    .venv/bin/python -m phase3.infer --model phase3/cache/ship_model.pkl --csv dataset/val.csv --out preds.csv
    .venv/bin/python -m phase3.infer --model phase3/cache/ship_model.pkl --images-dir /path/to/test --out preds.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import pickle
import numpy as np

from phase3.featurize import featurize_paths
from phase3.ship import score_features


def _score_finetuned(pt_path, paths, bs=32):
    """Score with a fine-tuned Net (.pt), hflip-TTA averaged."""
    import torch
    from phase3.finetune import Net, FrameDS, device
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    dev = device()
    net = Net(ckpt["cfg"].get("unfreeze", 4)).to(dev)
    net.load_state_dict(ckpt["model"]); net.eval()
    ds = FrameDS(list(paths), [0] * len(paths), train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=0)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(dev)
            s = (torch.sigmoid(net(x)) + torch.sigmoid(net(torch.flip(x, dims=[-1])))) / 2  # hflip TTA
            out.append(s.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phase3/cache/ship_model.pkl")
    ap.add_argument("--csv", help="csv with a 'path' column")
    ap.add_argument("--images-dir", help="directory of images to score")
    ap.add_argument("--out", default="preds.csv")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    finetuned = a.model.endswith(".pt")
    if not finetuned:
        with open(a.model, "rb") as f:
            artifact = pickle.load(f)

    if a.csv:
        rows = list(csv.DictReader(open(a.csv)))
        paths = [r["path"] for r in rows]
        labels = [r.get("label", "") for r in rows]
    else:
        paths = sorted(glob.glob(os.path.join(a.images_dir, "**", "*.*"), recursive=True))
        paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]
        labels = [""] * len(paths)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    print(f"scoring {len(paths)} images ({'fine-tuned .pt' if finetuned else 'frozen ensemble .pkl'})", flush=True)
    if finetuned:
        scores = _score_finetuned(a.model, paths, a.batch_size)
    else:
        z = featurize_paths(paths, batch_size=a.batch_size, workers=a.workers)
        scores = score_features(artifact, z)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "score", "label"])
        for n, s, l in zip(names, scores, labels):
            w.writerow([n, f"{s:.6f}", l])
    print(f"wrote {len(scores)} predictions -> {a.out}")


if __name__ == "__main__":
    main()
