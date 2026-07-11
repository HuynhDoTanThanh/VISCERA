"""Phase-3 end-to-end fine-tune — the cross-center CEILING-RAISER (run on a CUDA GPU).

Two Stage-2 regimes, selectable per the encoder you start from:
  * FINE-TUNE (default, --unfreeze K): unfreeze the last K blocks + head, HEAVY photometric+geometric
    augmentation (simulating the scope/lighting nuisance an unseen center introduces) + a LOCO-selected
    operating objective. On the RAW SSL backbone a frozen probe tops out ~0.2-0.28 LOCO-mean PPV@90R
    (it cannot change the features), so moving the blocks is the lever that learns center-invariant features.
  * HEAD-ONLY (--head-only): freeze the ENTIRE encoder, train only the linear head. This is the natural
    readout for the CONCEPT-PRETRAINED encoder (pretrain_concept.py --init): Phase-1 already made the
    features clinically-grounded + center-invariant, and fine-tuning the blocks would DISTORT them out-of-
    distribution (Kumar et al. 2022, "fine-tuning can distort pretrained features and underperform OOD").
    Fewest params -> least overfit when positives are scarce and SD >> margins. It is only as strong as the
    Phase-1 encoder, so treat FINE-TUNE vs HEAD-ONLY as a LOCO A/B, never a foregone conclusion.
    (LP-FT = head-only first, then a short low-LR unfreeze — the best-of-both; WiSE-FT is its weight-space cousin.)

Device-portable (cuda preferred, mps/cpu fallback). AMP on cuda. Selects on a held-out CENTER (LOCO)
via the PPV@90R bootstrap harness — the honest new-center proxy. Saves backbone+head for the offline
--network=none container (image -> model -> score; no VLM).

Run on cloud GPU:
    python -m phase3.finetune --unfreeze 4 --epochs 40 --holdout center_2 --bs 64 \
        --neg-list phase3/cache/unl_confneg.txt --neg-cap 6000 --out phase3/cache/ft_holdout_c2.pt
    # head-only readout of the concept encoder (preserves Phase-1 invariance):
    python -m phase3.finetune --head-only --init phase3/cache/concept_encoder.pt --epochs 30 \
        --holdout center_2 --out phase3/cache/lp_holdout_c2.pt
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
    def __init__(self, unfreeze=4, init_ckpt=None, head_only=False):
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
        self.head_only = head_only
        # freeze all; then (unless head_only) unfreeze last `unfreeze` blocks + final norm. head_only keeps the
        # ENTIRE encoder frozen (pure linear probe) — the head's own LayerNorm still adapts feature scaling.
        for p in m.parameters():
            p.requires_grad_(False)
        if not head_only:
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


class PosBalancedBatchSampler(torch.utils.data.Sampler):
    """Each batch carries exactly `pos_per_batch` positives (oversampled) + negatives, so the soft-pAUC /
    pairwise-rank tail losses ALWAYS fire. Without this, at ~1.5% positive rate a shuffled batch holds <1
    positive and the operating-point losses are silent no-ops (audit finding)."""
    def __init__(self, labels, batch_size, pos_per_batch=8, seed=0):
        self.labels = np.asarray(labels); self.bs = batch_size
        self.pos = np.where(self.labels == 1)[0]; self.neg = np.where(self.labels == 0)[0]
        self.ppb = min(pos_per_batch, max(1, len(self.pos))); self.npb = max(1, batch_size - self.ppb)
        self.nbatches = max(1, len(self.neg) // self.npb); self.epoch = 0; self.seed = seed

    def __len__(self):
        return self.nbatches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch); self.epoch += 1
        neg = rng.permutation(self.neg)
        for b in range(self.nbatches):
            nb = neg[b * self.npb:(b + 1) * self.npb]
            pb = rng.choice(self.pos, self.ppb, replace=len(self.pos) < self.ppb)
            idx = np.concatenate([pb, nb]); rng.shuffle(idx)
            yield idx.tolist()


# ----------------------------------------------------------------- losses (90R-targeted)
def pairwise_rank_loss(logits, y, margin=1.0, ohem_k=0):
    """Encourage every positive to outrank every negative (batch-wise). Directly improves ranking = PPV.

    ohem_k>0 = TAIL-WEIGHTED MARGIN: for each positive keep only its k HARDEST negatives (highest-scoring
    negatives = the pairs that actually sit at the 90R threshold). A uniform mean over all P*N pairs spends
    most of its gradient on already-separated easy pairs; top-k concentrates the margin exactly where
    PPV@90R is decided. No new params, no graph change."""
    pos, neg = logits[y == 1], logits[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return logits.sum() * 0.0
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)          # (P, N)
    loss = torch.nn.functional.softplus(margin - diff)  # (P, N) per-pair hinge
    if ohem_k and ohem_k < loss.shape[1]:
        loss = loss.topk(ohem_k, dim=1).values          # k hardest negatives per positive
    return loss.mean()


def soft_pauc90(logits, y, q=0.2):
    """Soft partial-AUC at the 90R operating point: penalize negatives scoring above the low-percentile
    positive (the threshold where recall~90%). q=0.2 (not 0.1) since only ~8 positives/batch. Targets the
    FP tail PPV@90R sees. (quantile needs float32 — AMP gives half.)"""
    pos, neg = logits[y == 1].float(), logits[y == 0].float()
    if len(pos) < 2 or len(neg) == 0:
        return logits.sum() * 0.0
    thr = torch.quantile(pos, q)
    return torch.nn.functional.softplus(neg - thr).mean().to(logits.dtype)


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
    r = ev.report_full(np.array(labels), s, np.array(centers), target=0.9, prevalence=0.01, B=500)
    return r, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="dataset/train.csv")
    ap.add_argument("--holdout", default="center_2", help="center to hold out for LOCO val; 'none' = train on all (ship)")
    ap.add_argument("--neg-list", default=""); ap.add_argument("--neg-cap", type=int, default=6000)
    ap.add_argument("--unfreeze", type=int, default=4)
    ap.add_argument("--head-only", action="store_true",
                    help="freeze the ENTIRE encoder; train only the linear head (LP readout of the concept "
                         "encoder — preserves Phase-1 center-invariance; Kumar et al. 2022: LP>FT under OOD shift)")
    ap.add_argument("--init", default="", help="concept-pretrained encoder (pretrain_concept.py) for Stage-2")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--pos-per-batch", type=int, default=8, help="positives guaranteed per batch (makes tail loss fire)")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="for multi-seed ensembling + reproducibility")
    ap.add_argument("--wise-ft", type=float, default=1.0,
                    help="WiSE-FT: final backbone = a*FT + (1-a)*init; 1.0=pure FT, <1 interpolates toward SSL init (robustness)")
    ap.add_argument("--loss", default="bce+rank+pauc", help="bce | rank | pauc | bce+rank+pauc")
    ap.add_argument("--warmup", type=int, default=2, help="epochs of pure BCE before adding tail terms")
    ap.add_argument("--lr-decay", type=float, default=0.75, help="layer-wise LR decay factor")
    ap.add_argument("--ohem-k", type=int, default=0,
                    help="tail-weighted margin: keep k hardest negatives per positive in the pairwise-rank loss "
                         "(0=off=uniform mean; recipe ~= pos_per_batch, e.g. 8). Targets the 90R tail, no new params.")
    ap.add_argument("--swad", action="store_true",
                    help="SWAD: ship the running mean of weights over the last --swad-last-n epochs (flat-minima "
                         "variance floor + cross-center robustness). Averaged model has the SAME graph as best-epoch.")
    ap.add_argument("--swad-last-n", type=int, default=5, help="number of final epochs to average for SWAD")
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
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    net = Net(a.unfreeze, init_ckpt=a.init or None, head_only=a.head_only).to(dev)
    init_bb = {k: v.detach().cpu().clone() for k, v in net.backbone.state_dict().items()}  # for WiSE-FT
    ntrain = sum(p.numel() for p in net.parameters() if p.requires_grad)
    mode = "HEAD-ONLY (frozen encoder linear probe)" if a.head_only else f"unfreeze last {a.unfreeze} blocks"
    print(f"train frames={len(tp)} pos={int(np.sum(tl))} | {mode} | trainable params={ntrain/1e6:.3f}M | holdout={a.holdout}")

    ds = FrameDS(tp, tl, train=True)
    sampler = PosBalancedBatchSampler(tl, a.bs, pos_per_batch=a.pos_per_batch, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=6, persistent_workers=True)
    # with a balanced sampler the in-batch neg/pos ratio is ~npb/ppb (mild), NOT the ~159 dataset ratio
    pos_w = torch.tensor([sampler.npb / sampler.ppb], device=dev, dtype=torch.float32)
    print(f"sampler: {sampler.ppb} pos + {sampler.npb} neg / batch, {sampler.nbatches} batches, pos_weight={pos_w.item():.1f}")
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
            L = L + 0.5 * pairwise_rank_loss(logits, y, ohem_k=a.ohem_k)
        if tail and "pauc" in terms:
            L = L + 0.5 * soft_pauc90(logits, y)
        return L

    swad_sum, swad_n = None, 0           # SWAD: running sum of state_dict over the last-N epochs
    best = -1
    for ep in range(a.epochs):
        net.train(); tot = 0
        if net.head_only:
            net.backbone.eval()         # frozen encoder -> deterministic features (no drop_path/dropout noise)
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
            r, s = evaluate_center(net, list(paths[vam]), list(labels[vam]), list(centers[vam]), dev, a.bs)
            ppv = r["ppv90"]
            # AUROC/AUPRC are threshold-free & STABLE across epochs — trust them to read the trend;
            # PPV@90R bounces on few positives. Selection stays on PPV (the leaderboard metric).
            msg += (f"  LOCO-val({a.holdout}) PPV@90R={ppv:.4f} CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
                    f" AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f}")
            if ppv > best:
                best = ppv
                torch.save({"model": net.state_dict(), "cfg": vars(a), "loco_ppv90": best}, a.out)
                # persist the primary (best-epoch) model's held-out y/center/scores for the PAIRED gate (ev.gate).
                # Frame order is identical across runs (same csv + holdout + sequential loader) -> sA,sB align.
                np.savez(a.out[:-3] + "_loco.npz", y=np.array(labels[vam]), c=np.array(centers[vam]), s=s)
                msg += "  *saved*"
        print(msg, flush=True)
        if a.swad and ep >= a.epochs - a.swad_last_n:   # accumulate the flat tail of the trajectory
            sd = net.state_dict()
            if swad_sum is None:
                swad_sum = {k: v.detach().float().cpu().clone() for k, v in sd.items() if v.is_floating_point()}
            else:
                for k in swad_sum:
                    swad_sum[k] += sd[k].detach().float().cpu()
            swad_n += 1

    # ---- SWAD: build the averaged model (same graph; frozen params average to themselves) ----
    swad_out = None
    if a.swad and swad_sum is not None:
        ref = net.state_dict()
        swad_state = {k: v.clone() for k, v in ref.items()}
        for k in swad_sum:
            swad_state[k] = (swad_sum[k] / swad_n).to(ref[k].dtype)
        swad_out = a.out if not vam.any() else a.out[:-3] + "_swad.pt"
        torch.save({"model": swad_state, "cfg": vars(a), "swad_last_n": swad_n}, swad_out)
        print(f"SWAD: averaged last {swad_n} epochs -> {swad_out}")
        if vam.any():        # LOCO ablation: score SWAD on the held-out center vs the best-epoch model (a.out)
            net.load_state_dict(swad_state)
            r, s = evaluate_center(net, list(paths[vam]), list(labels[vam]), list(centers[vam]), dev, a.bs)
            # SWAD is the shippable model under --swad -> its held-out scores are what the paired gate should compare
            np.savez(a.out[:-3] + "_loco.npz", y=np.array(labels[vam]), c=np.array(centers[vam]), s=s)
            print(f"SWAD LOCO-val({a.holdout}) PPV@90R={r['ppv90']:.4f} CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
                  f"AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f}  (compare to best-epoch above)")

    if not vam.any() and not (a.swad and swad_out):     # ship: SWAD already wrote a.out; else write final net
        torch.save({"model": net.state_dict(), "cfg": vars(a)}, a.out)
        print(f"saved ship model -> {a.out}")
    print(f"best LOCO-val PPV@90R = {best:.4f}")

    # WiSE-FT: interpolate the saved backbone toward the SSL init (robustness; pick alpha on inner val).
    # No-op under --head-only (the backbone never moved), so skip it. Applied to BOTH best-epoch and SWAD.
    def apply_wise_ft(path):
        if not (a.wise_ft < 1.0 and not a.head_only and path and os.path.exists(path)):
            return
        ck = torch.load(path, map_location="cpu", weights_only=False); st = ck["model"]
        for k, v in init_bb.items():
            bk = f"backbone.{k}"
            if bk in st and st[bk].shape == v.shape:
                st[bk] = (a.wise_ft * st[bk].float() + (1 - a.wise_ft) * v.float()).to(st[bk].dtype)
        ck["model"] = st; torch.save(ck, path)
        print(f"applied WiSE-FT alpha={a.wise_ft} -> {path}")
    apply_wise_ft(a.out)
    if swad_out and swad_out != a.out:
        apply_wise_ft(swad_out)


if __name__ == "__main__":
    main()
