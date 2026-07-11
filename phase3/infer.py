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


# TTA views — the 5view set is IDENTICAL to RARE25-Submission/model/viscera_model.py so val == the shipped container.
_TTA_VIEWS = {
    "orig":   lambda x: x,
    "hflip":  lambda x: __import__("torch").flip(x, dims=[-1]),
    "vflip":  lambda x: __import__("torch").flip(x, dims=[-2]),
    "rot90":  lambda x: __import__("torch").rot90(x, 1, dims=[-2, -1]),
    "rot270": lambda x: __import__("torch").rot90(x, 3, dims=[-2, -1]),
}


def _score_finetuned(pt_paths, paths, bs=32, tta="hflip"):
    """Score one or more fine-tuned Nets (.pt): mean-prob over TTA views per model, then prob-averaged across models.
    tta='hflip' (orig+hflip) or '5view' (orig/hflip/vflip/rot90/rot270 = EXACTLY the shipped container)."""
    import torch
    from phase3.finetune import Net, FrameDS, device
    if isinstance(pt_paths, str):
        pt_paths = [pt_paths]
    views = ["orig", "hflip"] if tta == "hflip" else ["orig", "hflip", "vflip", "rot90", "rot270"]
    dev = device()
    ds = FrameDS(list(paths), [0] * len(paths), train=False)
    acc = np.zeros(len(paths), dtype=np.float64)
    for pt in pt_paths:
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {})
        net = Net(cfg.get("unfreeze", 4), cg_head=cfg.get("cg_head", False), backbone=cfg.get("backbone", "dinov2")).to(dev)
        net.load_state_dict(ckpt["model"]); net.eval()
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=0)
        out = []
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(dev)
                s = sum(torch.sigmoid(net(_TTA_VIEWS[v](x))) for v in views) / len(views)  # mean-prob over views
                out.append(s.float().cpu().numpy())
        acc += np.concatenate(out); print(f"  scored with {pt} (tta={tta})")
    return acc / len(pt_paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phase3/cache/ship_model.pkl")
    ap.add_argument("--csv", help="csv with a 'path' column")
    ap.add_argument("--images-dir", help="directory of images to score")
    ap.add_argument("--out", default="preds.csv")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tta", choices=["hflip", "5view"], default="hflip",
                    help="5view = orig/hflip/vflip/rot90/rot270 = EXACTLY the shipped container (use for val scoring)")
    a = ap.parse_args()

    finetuned = a.model.endswith(".pt") or "," in a.model   # one or more .pt (comma-sep) = fine-tuned ensemble
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
        models = [m.strip() for m in a.model.split(",") if m.strip()]
        scores = _score_finetuned(models, paths, a.batch_size, tta=a.tta)
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
