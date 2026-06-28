"""Phase-3 featurizer: frozen domain-adapted DINOv2 ViT-B/14-reg -> cached pooled embeddings.

Design-invariant foundation for ANY downstream head. Caches a RICH pooled superset per image
(cls / register-mean / patch-mean / patch-max = 4x768 = 3072-d) so linear / logistic / MLP / kNN /
one-class heads can all be built from the cache without re-running the backbone.

Backbone = dinov2.pth EMA `teacher` (backbone.*), loaded into timm vit_base_patch14_reg4_dinov2 at
336px via timm's official DINOv2 converter. Frozen.

Speed + resilience (the external volume is flaky):
  - parallel image decode via DataLoader workers (the bottleneck is PNG/JPG decode + volume IO)
  - RESUMABLE sharding for big runs (--shard-dir): writes shard_*.npz every --shard-size images and
    skips shards already on disk, so a volume hiccup never loses more than one shard.

Usage:
    # labeled (small, single npz):
    .venv/bin/python -m phase3.featurize --csv dataset/train.csv --out phase3/cache/feats_train.npz --workers 6
    .venv/bin/python -m phase3.featurize --csv dataset/val.csv   --out phase3/cache/feats_val.npz   --workers 6
    # unlabeled mining (big, resumable shards then aggregate):
    .venv/bin/python -m phase3.featurize --list phase3/cache/unl_hardneg.txt --shard-dir phase3/cache/unl_hardneg --shard-size 2000 --workers 8
    .venv/bin/python -m phase3.featurize --aggregate phase3/cache/unl_hardneg --out phase3/cache/feats_unl_hardneg.npz
"""
from __future__ import annotations
import argparse
import csv as csvmod
import glob
import os
import time
import numpy as np
import torch
import timm
from PIL import Image

CKPT = "dinov2.pth"
IMG = 336
EMBED = 768
POOL_NAMES = ("cls", "reg_mean", "patch_mean", "patch_max")
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_backbone(dev=None, backbone_ckpt=None):
    """Load the SSL teacher backbone; if backbone_ckpt is given, overwrite with that backbone state dict
    (e.g. a concept-pretrained encoder from pretrain_concept.py) for apples-to-apples re-featurization."""
    dev = dev or device()
    teacher = torch.load(CKPT, map_location="cpu", weights_only=False)["teacher"]
    bk = {k[len("backbone."):]: v for k, v in teacher.items() if k.startswith("backbone.")}
    m = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False, img_size=IMG, num_classes=0)
    from timm.models import vision_transformer as vit_mod
    converted = vit_mod.checkpoint_filter_fn(bk, m)
    converted.pop("mask_token", None)
    miss, unexp = m.load_state_dict(converted, strict=False)
    assert not miss and not unexp, f"backbone load mismatch missing={miss} unexpected={unexp}"
    if backbone_ckpt:
        ck = torch.load(backbone_ckpt, map_location="cpu", weights_only=False)
        state = ck.get("backbone", ck)
        m2, u2 = m.load_state_dict(state, strict=False)
        print(f"[featurize] alternate backbone {backbone_ckpt} (missing={len(m2)} unexpected={len(u2)})", flush=True)
    m.eval().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, dev


def _load_tensor(path, hflip=False):
    im = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    x = (x - _MEAN) / _STD
    if hflip:
        x = torch.flip(x, dims=[-1])
    return x


class _ImgDS(torch.utils.data.Dataset):
    def __init__(self, paths, hflip=False):
        self.paths = paths; self.hflip = hflip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return i, _load_tensor(self.paths[i], self.hflip)
        except Exception:
            return i, torch.zeros(3, IMG, IMG)  # unreadable -> zeros (rare); kept to preserve order


@torch.no_grad()
def _pool(out):
    cls = out[:, 0]; reg = out[:, 1:5].mean(1); patch = out[:, 5:]
    return torch.cat([cls, reg, patch.mean(1), patch.max(1).values], dim=-1)


@torch.no_grad()
def featurize_paths(paths, batch_size=32, workers=6, hflip=False, model=None, dev=None, log_every=2000, backbone_ckpt=None):
    if model is None:
        model, dev = load_backbone(backbone_ckpt=backbone_ckpt)
    ds = _ImgDS(paths, hflip)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=workers,
                                     shuffle=False, drop_last=False, persistent_workers=workers > 0)
    feats = np.zeros((len(paths), len(POOL_NAMES) * EMBED), dtype=np.float32)
    t0 = time.time(); done = 0
    for idx, xs in dl:
        xs = xs.to(dev)
        pooled = _pool(model.forward_features(xs)).float().cpu().numpy()
        feats[idx.numpy()] = pooled
        done += len(idx)
        if done % log_every < batch_size:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(paths)}  {rate:.1f} img/s  eta {(len(paths)-done)/max(rate,1e-6):.0f}s", flush=True)
    return feats


def _save(out, feats, names, paths, meta):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, feats=feats, names=np.array(names), paths=np.array(paths),
                        pool_names=np.array(POOL_NAMES), embed=EMBED, **meta)
    os.replace(tmp, out)


def run_sharded(paths, names, shard_dir, shard_size, batch_size, workers):
    """Resumable: each shard_<k>.npz holds shard_size rows; existing shards are skipped."""
    os.makedirs(shard_dir, exist_ok=True)
    model, dev = load_backbone()
    nsh = (len(paths) + shard_size - 1) // shard_size
    for k in range(nsh):
        sp = os.path.join(shard_dir, f"shard_{k:05d}.npz")
        if os.path.exists(sp):
            continue
        lo, hi = k * shard_size, min((k + 1) * shard_size, len(paths))
        print(f"shard {k+1}/{nsh}  rows {lo}:{hi}", flush=True)
        feats = featurize_paths(paths[lo:hi], batch_size, workers, model=model, dev=dev)
        tmp = os.path.join(shard_dir, f"_tmp_{k:05d}.npz")  # must end .npz (np.savez appends it otherwise)
        np.savez_compressed(tmp, feats=feats, names=np.array(names[lo:hi]), paths=np.array(paths[lo:hi]),
                            pool_names=np.array(POOL_NAMES), embed=EMBED)
        os.replace(tmp, sp)
    print(f"all {nsh} shards present in {shard_dir}", flush=True)


def aggregate(shard_dir, out):
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    feats, names, paths = [], [], []
    for sp in shards:
        d = np.load(sp, allow_pickle=True)
        feats.append(d["feats"]); names.append(d["names"]); paths.append(d["paths"])
    feats = np.concatenate(feats); names = np.concatenate(names); paths = np.concatenate(paths)
    _save(out, feats, names, paths, {"label": np.full(len(names), -1, np.int64)})
    print(f"aggregated {len(shards)} shards -> {out}  {feats.shape}", flush=True)


def read_csv_rows(path):
    with open(path) as f:
        return list(csvmod.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--list")
    ap.add_argument("--out")
    ap.add_argument("--shard-dir")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--aggregate")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backbone", default="", help="alternate backbone ckpt (e.g. concept_encoder.pt) to featurize with")
    a = ap.parse_args()

    if a.aggregate:
        aggregate(a.aggregate, a.out); return

    meta = {}
    if a.csv:
        rows = read_csv_rows(a.csv)
        if a.limit:
            rows = rows[:a.limit]
        paths = [r["path"] for r in rows]
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        meta["label"] = np.array([int(r.get("label", -1)) for r in rows], dtype=np.int64)
        meta["center"] = np.array([r.get("center", "") for r in rows])
        if rows and "source_path" in rows[0]:
            meta["source"] = np.array([os.path.splitext(os.path.basename(r.get("source_path", "")))[0] for r in rows])
            meta["aug"] = np.array([r.get("aug", "") for r in rows])
    else:
        with open(a.list) as f:
            paths = [ln.strip() for ln in f if ln.strip()]
        if a.limit:
            paths = paths[:a.limit]
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        meta["label"] = np.full(len(paths), -1, dtype=np.int64)

    print(f"featurizing {len(paths)} images on {device()} (workers={a.workers})", flush=True)
    if a.shard_dir:
        run_sharded(paths, names, a.shard_dir, a.shard_size, a.batch_size, a.workers)
        if a.out:
            aggregate(a.shard_dir, a.out)
    else:
        feats = featurize_paths(paths, a.batch_size, a.workers, hflip=a.hflip, backbone_ckpt=a.backbone or None)
        _save(a.out, feats, names, paths, meta)
        print(f"saved {feats.shape} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
