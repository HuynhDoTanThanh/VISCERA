"""Phase-3 end-to-end fine-tune — the cross-center CEILING-RAISER (run on a CUDA GPU).

A frozen probe on cached DINOv2 features tops out ~0.2-0.28 LOCO-mean PPV@90R because it cannot change
the features. Fine-tuning the last K transformer blocks + head, with HEAVY photometric+geometric
augmentation (simulating the scope/lighting nuisance an unseen center introduces) and a LOCO-selected
operating objective, is the lever that can learn genuinely center-invariant neoplasia features.

Device-portable (cuda preferred, mps/cpu fallback). AMP on cuda. Selects on a held-out CENTER (LOCO)
via the PPV@90R bootstrap harness — the honest new-center proxy. Saves backbone+head for the offline
--network=none container (image -> model -> score; no VLM).

Run on cloud GPU:
    python -m phase3.finetune --unfreeze 4 --epochs 40 --holdout center_2 --bs 64 \
        --neg-list phase3/cache/unl_confneg.txt --neg-cap 6000 --out phase3/cache/ft_holdout_c2.pt
    # then swap --holdout center_1 to see both legs; ship a model trained on BOTH centers (--holdout none).
"""
from __future__ import annotations
import argparse
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as T

from phase3.featurize import CKPT, IMG, _MEAN, _STD
from phase3 import evaluate as ev


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------------------------------------------- model
class Net(nn.Module):
    """DINOv2 ViT-B/14-reg backbone (last K blocks trainable) + [cls ⊕ patch_mean] linear head.

    init_ckpt: optional concept-pretrained encoder (pretrain_concept.py output) to start from instead
    of the raw SSL teacher — this is Stage-2 of the concept-supervised pipeline.
    """
    def __init__(self, unfreeze=4, init_ckpt=None):
        super().__init__()
        m = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False, img_size=IMG, num_classes=0)
        teacher = torch.load(CKPT, map_location="cpu", weights_only=False)["teacher"]
        bk = {k[len("backbone."):]: v for k, v in teacher.items() if k.startswith("backbone.")}
        from timm.models import vision_transformer as vit_mod
        conv = vit_mod.checkpoint_filter_fn(bk, m); conv.pop("mask_token", None)
        miss, unexp = m.load_state_dict(conv, strict=False)
        assert not miss and not unexp, f"backbone mismatch {miss} {unexp}"
        if init_ckpt:
            ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            m2, u2 = m.load_state_dict(ck["backbone"], strict=False)
            print(f"[init] loaded concept-pretrained backbone from {init_ckpt} (missing={len(m2)} unexpected={len(u2)})")
        self.backbone = m
        # freeze all, then unfreeze last `unfreeze` blocks + final norm
        for p in m.parameters():
            p.requires_grad_(False)
        nblocks = len(m.blocks)
        for i in range(max(0, nblocks - unfreeze), nblocks):
            for p in m.blocks[i].parameters():
                p.requires_grad_(True)
        for p in m.norm.parameters():
            p.requires_grad_(True)
        self.head = nn.Sequential(nn.LayerNorm(2 * 768), nn.Linear(2 * 768, 1))

    def forward(self, x):
        f = self.backbone.forward_features(x)            # (B, 1+4+576, 768)
        feat = torch.cat([f[:, 0], f[:, 5:].mean(1)], -1)
        return self.head(feat).squeeze(-1)


# ----------------------------------------------------------------- data
class FrameDS(torch.utils.data.Dataset):
    def __init__(self, paths, labels, train=True):
        self.paths, self.labels, self.train = paths, labels, train
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(IMG, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.3, 0.04),       # cross-center scope/lighting nuisance
                T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], 0.2),
                T.RandomRotation(10),
            ])
        else:
            self.tf = T.Resize((IMG, IMG))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        im = self.tf(im) if self.train else self.tf(im)
        x = (torch.from_numpy(np.asarray(im.resize((IMG, IMG)), np.float32).copy()).permute(2, 0, 1) / 255. - _MEAN) / _STD
        return x, float(self.labels[i])


# ----------------------------------------------------------------- losses (90R-targeted)
def pairwise_rank_loss(logits, y, margin=1.0):
    """Encourage every positive to outrank every negative (batch-wise). Directly improves ranking = PPV."""
    pos, neg = logits[y == 1], logits[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return logits.sum() * 0.0
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)          # (P, N)
    return torch.nn.functional.softplus(margin - diff).mean()


def soft_pauc90(logits, y, q=0.1):
    """Soft partial-AUC at the 90R operating point: penalize negatives scoring above the
    ~10th-percentile positive (the threshold where recall=90%). Targets exactly the FP tail PPV@90R sees."""
    pos, neg = logits[y == 1], logits[y == 0]
    if len(pos) < 2 or len(neg) == 0:
        return logits.sum() * 0.0
    thr = torch.quantile(pos, q)
    return torch.nn.functional.softplus(neg - thr).mean()


def layerwise_param_groups(net, base_lr, decay=0.75, head_lr_mult=10.0):
    """Lower LR for earlier (frozen-thawed) blocks, higher for head — standard ViT fine-tune practice."""
    groups = []
    nblocks = len(net.backbone.blocks)
    for i, blk in enumerate(net.backbone.blocks):
        ps = [p for p in blk.parameters() if p.requires_grad]
        if ps:
            groups.append({"params": ps, "lr": base_lr * (decay ** (nblocks - 1 - i))})
    norm_ps = [p for p in net.backbone.norm.parameters() if p.requires_grad]
    if norm_ps:
        groups.append({"params": norm_ps, "lr": base_lr})
    groups.append({"params": net.head.parameters(), "lr": base_lr * head_lr_mult})
    return groups


def load_split(csv_path, neg_list="", neg_cap=0):
    rows = list(csv.DictReader(open(csv_path)))
    paths = [r["path"] for r in rows]; labels = [int(r["label"]) for r in rows]; centers = [r["center"] for r in rows]
    extra_neg = []
    if neg_list and os.path.exists(neg_list):
        extra_neg = [ln.strip() for ln in open(neg_list) if ln.strip()][:neg_cap] if neg_cap else [ln.strip() for ln in open(neg_list)]
    return np.array(paths), np.array(labels), np.array(centers), extra_neg


@torch.no_grad()
def evaluate_center(net, paths, labels, centers, dev, bs=64):
    net.eval()
    ds = FrameDS(paths, labels, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=0)  # eval set is small; avoids spawn issues
    sc = []
    for x, _ in dl:
        sc.append(torch.sigmoid(net(x.to(dev))).float().cpu().numpy())
    s = np.concatenate(sc)
    return ev.bootstrap(np.array(labels), s, np.array(centers), target=0.9, prevalence=0.01, B=500)["curve"]["median"], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="dataset/train.csv")
    ap.add_argument("--holdout", default="center_2", help="center to hold out for LOCO val; 'none' = train on all (ship)")
    ap.add_argument("--neg-list", default=""); ap.add_argument("--neg-cap", type=int, default=6000)
    ap.add_argument("--unfreeze", type=int, default=4)
    ap.add_argument("--init", default="", help="concept-pretrained encoder (pretrain_concept.py) for Stage-2")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--loss", default="bce+rank+pauc", help="bce | rank | pauc | bce+rank+pauc")
    ap.add_argument("--warmup", type=int, default=2, help="epochs of pure BCE before adding tail terms")
    ap.add_argument("--lr-decay", type=float, default=0.75, help="layer-wise LR decay factor")
    ap.add_argument("--out", default="phase3/cache/ft.pt")
    a = ap.parse_args()
    dev = device(); print(f"device={dev}")

    paths, labels, centers, extra_neg = load_split(a.train_csv, a.neg_list, a.neg_cap)
    if a.holdout != "none":
        trm = centers != a.holdout; vam = centers == a.holdout
    else:
        trm = np.ones(len(paths), bool); vam = np.zeros(len(paths), bool)
    tp, tl = list(paths[trm]), list(labels[trm])
    if extra_neg:
        tp += extra_neg; tl += [0] * len(extra_neg)
        print(f"+ {len(extra_neg)} unlabeled negatives")
    net = Net(a.unfreeze, init_ckpt=a.init or None).to(dev)
    ntrain = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"train frames={len(tp)} pos={int(np.sum(tl))} | trainable params={ntrain/1e6:.1f}M | holdout={a.holdout}")

    ds = FrameDS(tp, tl, train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=6, drop_last=True, persistent_workers=True)
    pos_w = torch.tensor([float((np.array(tl) == 0).sum()) / max(int((np.array(tl) == 1).sum()), 1)],
                         device=dev, dtype=torch.float32)  # MPS rejects float64
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(layerwise_param_groups(net, a.lr, a.lr_decay), weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    amp = dev == "cuda"; scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def compute_loss(logits, y, tail):
        terms = a.loss.split("+")
        L = logits.sum() * 0.0
        if "bce" in terms or not tail:
            L = L + bce(logits, y)
        if tail and "rank" in terms:
            L = L + 0.5 * pairwise_rank_loss(logits, y)
        if tail and "pauc" in terms:
            L = L + 0.5 * soft_pauc90(logits, y)
        return L

    best = -1
    for ep in range(a.epochs):
        net.train(); tot = 0
        tail = ep >= a.warmup           # warm-start on BCE, then add the 90R tail terms
        for x, y in dl:
            x, y = x.to(dev), y.float().to(dev)   # cast float64->float32 on CPU BEFORE moving (MPS rejects float64)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = compute_loss(net(x), y, tail)
            if amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        msg = f"ep{ep+1}/{a.epochs} loss={tot/len(dl):.4f}"
        if vam.any():
            ppv, _ = evaluate_center(net, list(paths[vam]), list(labels[vam]), list(centers[vam]), dev, a.bs)
            msg += f"  LOCO-val({a.holdout}) PPV@90R={ppv:.4f}"
            if ppv > best:
                best = ppv
                torch.save({"model": net.state_dict(), "cfg": vars(a), "loco_ppv90": best}, a.out)
                msg += "  *saved*"
        print(msg, flush=True)
    if not vam.any():
        torch.save({"model": net.state_dict(), "cfg": vars(a)}, a.out)
        print(f"saved ship model -> {a.out}")
    print(f"best LOCO-val PPV@90R = {best:.4f}")


if __name__ == "__main__":
    main()
