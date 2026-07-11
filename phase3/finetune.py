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
BACKBONES = {   # name -> (timm model, weights file). BOTH have 5 prefix tokens (1 cls + 4 reg) + embed 768 -> SAME head.
    "dinov2": ("vit_base_patch14_reg4_dinov2", "dinov2.pth"),   # teacher-format ckpt; patch14 -> 576 patches @336
    "dinov3": ("vit_base_patch16_dinov3", "dinov3.pth"),        # plain timm state_dict; patch16 -> 784 patches @448
}


class AttnPool(nn.Module):
    """Gated attention-MIL pooling (Ilse et al. 2018) over patch tokens: a subtle lesion of a FEW patches can
    dominate the pooled vector instead of being divided by ~1024 (mean-pool dilution). Per-image (softmax over the
    token axis) — no batch stats, ships in the LayerNorm-only graph. Returns (pooled, attention)."""
    def __init__(self, dim=768, hid=128):
        super().__init__()
        self.V = nn.Linear(dim, hid, bias=False)
        self.U = nn.Linear(dim, hid, bias=False)
        self.w = nn.Linear(hid, 1, bias=False)

    def forward(self, p):                                     # p: (B, N, dim)
        a = self.w(torch.tanh(self.V(p)) * torch.sigmoid(self.U(p))).squeeze(-1)   # (B, N)
        a = torch.softmax(a, dim=1)
        return (a.unsqueeze(-1) * p).sum(1), a               # (B, dim), (B, N)


class Net(nn.Module):
    """DINOv2 ViT-B/14-reg backbone (last K blocks trainable) + [cls ⊕ pooled-patch] linear head.
    Pooling = mean (default) or gated attention-MIL (cg_head=True) that lifts a few-patch lesion (the tail lever).

    init_ckpt: optional concept-pretrained encoder (pretrain_concept.py output) to start from instead
    of the raw SSL teacher — this is Stage-2 of the concept-supervised pipeline.
    """
    def __init__(self, unfreeze=4, init_ckpt=None, head_only=False, cg_head=False, backbone="dinov3"):
        super().__init__()
        from timm.models import vision_transformer as vit_mod
        model_name, ckpt_path = BACKBONES[backbone]
        m = timm.create_model(model_name, pretrained=False, img_size=IMG, num_classes=0)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if backbone == "dinov2":                                    # teacher-format checkpoint -> extract backbone.*
            sd = {k[len("backbone."):]: v for k, v in sd["teacher"].items() if k.startswith("backbone.")}
        conv = vit_mod.checkpoint_filter_fn(sd, m); conv.pop("mask_token", None)
        miss, unexp = m.load_state_dict(conv, strict=False)
        assert not miss and not unexp, f"backbone {backbone} mismatch: missing={list(miss)[:4]} unexpected={list(unexp)[:4]}"
        if init_ckpt:
            ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            bk2 = vit_mod.checkpoint_filter_fn(ck["backbone"], m)   # interpolates pos_embed if IMG changed (336->448)
            bk2.pop("mask_token", None)
            m2, u2 = m.load_state_dict(bk2, strict=False)
            # fail loud (like the SSL load above) — else a future key drift silently reverts to raw SSL.
            assert not m2 and not u2, f"concept-init key mismatch: missing={m2[:4]} unexpected={u2[:4]}"
            print(f"[init] loaded concept-pretrained backbone from {init_ckpt}")
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
        self.cg_head = cg_head
        self.attn = AttnPool(768, 128) if cg_head else None   # ~0.2M params; only added key set vs mean-pool

    def forward(self, x, return_attn=False):
        f = self.backbone.forward_features(x)            # (B, 1+4+N, 768)  N=576@336 / 1024@448
        patches = f[:, 5:]
        if self.cg_head:
            pooled, a = self.attn(patches)               # attention-weighted pool (lifts few-patch lesion)
        else:
            pooled, a = patches.mean(1), None            # mean-pool (dilutes)
        logit = self.head(torch.cat([f[:, 0], pooled], -1)).squeeze(-1)   # cls kept as stable residual
        return (logit, a) if return_attn else logit


# ----------------------------------------------------------------- data
class FrameDS(torch.utils.data.Dataset):
    def __init__(self, paths, labels, train=True, aug="mild", aug_config="rand-m6-mstd0.6-inc1"):
        self.paths, self.labels, self.train = paths, labels, train
        if not train:
            self.tf = T.Resize((IMG, IMG))                # eval: deterministic, NO aug (train/serve parity)
        elif aug == "strong":
            # Endoscopy-curated RandAugment (domain randomization): with only 2 centers we cannot LEARN cross-center
            # invariance, so we SYNTHESIZE acquisition diversity. EXCLUDE Invert/Solarize/Posterize/SolarizeAdd —
            # they invert/quantize exactly the mucosal-color + vascular cues that separate neo from NDBE (signal
            # erasure). Keep geometric (scope framing/orientation) + photometric (scope/lighting nuisance) ops only.
            from timm.data.auto_augment import rand_augment_transform
            ENDO_OPS = ["AutoContrast", "Equalize", "Rotate", "ColorIncreasing", "ContrastIncreasing",
                        "BrightnessIncreasing", "SharpnessIncreasing", "ShearX", "ShearY", "TranslateXRel", "TranslateYRel"]
            ra = rand_augment_transform(aug_config, {"translate_const": int(IMG * 0.3), "img_mean": (124, 116, 104)},
                                        transforms=ENDO_OPS)
            self.tf = T.Compose([
                T.RandomResizedCrop(IMG, scale=(0.5, 1.0), ratio=(0.8, 1.25)),   # wider framing variation than mild
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(),                # endoscopy has no canonical up
                ra,                                                              # curated RandAugment (m6/mstd0.6)
                T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], 0.2),
            ])
        else:  # mild (default) == the exp1 baseline augmentation, unchanged
            self.tf = T.Compose([
                T.RandomResizedCrop(IMG, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.3, 0.04),       # cross-center scope/lighting nuisance
                T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], 0.2),
                T.RandomRotation(10),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        im = self.tf(im) if self.train else self.tf(im)
        x = (torch.from_numpy(np.asarray(im.resize((IMG, IMG)), np.float32).copy()).permute(2, 0, 1) / 255. - _MEAN) / _STD
        return x, float(self.labels[i])


class UnlabeledDS(torch.utils.data.Dataset):
    """VLM-scored unlabeled frames for SEMI-SUPERVISED training. Returns (weak_view, strong_view, vlm_suspicion):
    weak = mild aug (EMA-teacher target), strong = endoscopy RandAugment (student input) — FixMatch/Mean-Teacher
    consistency. The 168k-frame pool regularizes the model instead of memorizing the 127 positives (anti-overfit),
    and the VLM suspicion gives a one-sided-PU confident-NEGATIVE signal (low suspicion = clearly normal mucosa)."""
    def __init__(self, img_paths, susp, aug_config="rand-m6-mstd0.6-inc1"):
        self.paths = list(img_paths); self.susp = np.asarray(susp, np.float32)
        self.weak = T.Compose([T.RandomResizedCrop(IMG, scale=(0.7, 1.0)), T.RandomHorizontalFlip()])
        from timm.data.auto_augment import rand_augment_transform
        ENDO_OPS = ["AutoContrast", "Equalize", "Rotate", "ColorIncreasing", "ContrastIncreasing",
                    "BrightnessIncreasing", "SharpnessIncreasing", "ShearX", "ShearY", "TranslateXRel", "TranslateYRel"]
        ra = rand_augment_transform(aug_config, {"translate_const": int(IMG * 0.3), "img_mean": (124, 116, 104)},
                                    transforms=ENDO_OPS)
        self.strong = T.Compose([T.RandomResizedCrop(IMG, scale=(0.5, 1.0)),
                                 T.RandomHorizontalFlip(), T.RandomVerticalFlip(), ra])

    def __len__(self):
        return len(self.paths)

    def _norm(self, im):
        return (torch.from_numpy(np.asarray(im.resize((IMG, IMG)), np.float32).copy()).permute(2, 0, 1) / 255. - _MEAN) / _STD

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return self._norm(self.weak(im)), self._norm(self.strong(im)), float(self.susp[i])


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
    ap.add_argument("--aug", choices=["mild", "strong"], default="mild",
                    help="mild=exp1 baseline; strong=endoscopy-curated RandAugment (domain randomization -> synthesize "
                         "the acquisition diversity 2 centers can't provide; excludes signal-erasing Invert/Solarize/Posterize)")
    ap.add_argument("--aug-config", default="rand-m6-mstd0.6-inc1", help="timm RandAugment config for --aug strong")
    # ---- semi-supervised over the VLM-scored unlabeled pool: Mean-Teacher consistency (anti-overfit) + one-sided-PU
    #      confident-negative distillation (demotes the FP tail). Uses the 168k frames the labeled set can't. ----
    ap.add_argument("--semi-manifest", default="",
                    help="unl_manifest.npz (VLM-scored pool) -> enable semi-supervised loss. '' = off.")
    ap.add_argument("--semi-n", type=int, default=20000, help="unlabeled frames to sample for the semi loss")
    ap.add_argument("--semi-bs", type=int, default=48, help="unlabeled batch size for the semi loss")
    ap.add_argument("--semi-weight", type=float, default=0.5, help="weight of the semi loss (ramped 0->1 over --semi-rampup)")
    ap.add_argument("--semi-lo", type=float, default=0.15,
                    help="VLM suspicion below this = confident-NEGATIVE pseudo-label (target 0). Positives NOT pseudo-"
                         "labeled (high suspicion includes NDBE look-alikes) -> one-sided PU.")
    ap.add_argument("--ema-decay", type=float, default=0.99, help="EMA teacher decay for the consistency target")
    ap.add_argument("--semi-rampup", type=int, default=5, help="epochs to ramp the semi weight 0->1 after --warmup")
    # ---- CG-AMIL head: gated attention-MIL pooling (lifts a few-patch lesion vs mean-pool) + entropy floor ----
    ap.add_argument("--backbone", choices=["dinov2", "dinov3"], default="dinov3",
                    help="dinov3 = ViT-B/16 (stronger dense/patch features, needs dinov3.pth); dinov2 = ViT-B/14-reg "
                         "(Apache-2.0, dinov2.pth, the exp1 0.018 path). Both: 5 prefix + embed 768 -> same head.")
    ap.add_argument("--cg-head", action="store_true",
                    help="use gated attention-MIL pooling instead of mean-pool (the tail lever; ~0.2M params, "
                         "regularized by the SEMI 288k-pool consistency + entropy floor). Ship graph gains attn.* keys.")
    ap.add_argument("--attn-entropy", type=float, default=0.02,
                    help="entropy regularizer on the attention map (maximize H -> anti-collapse: forbids 1-hot "
                         "attention that memorizes a single patch/center cue). Only used with --cg-head.")
    ap.add_argument("--semi-steps", type=int, default=1,
                    help="unlabeled batches per labeled step (grad-accumulated). >1 uses MORE of the pool per epoch "
                         "with fresh strong-aug each, WITHOUT repeating the labeled set more. Coverage/epoch = "
                         "26*semi_steps*semi_bs frames. e.g. 8 -> ~20k/epoch at semi_bs=96.")
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
    a.img = IMG                              # stamp the training image size into cfg so the container reconstructs exactly
    dev = device(); print(f"device={dev} | backbone={a.backbone} | img={IMG} | cg_head={a.cg_head}")
    if a.holdout == "none" and a.wise_ft >= 1.0 and not a.head_only:
        print("WARNING: shipping PURE FT (--wise-ft 1.0, no OOD anchor). The recipe is --wise-ft 0.7 [--swad]; "
              "a bare invocation ships the least-robust model.", flush=True)

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
    net = Net(a.unfreeze, init_ckpt=a.init or None, head_only=a.head_only, cg_head=a.cg_head, backbone=a.backbone).to(dev)
    init_bb = {k: v.detach().cpu().clone() for k, v in net.backbone.state_dict().items()}  # for WiSE-FT
    ntrain = sum(p.numel() for p in net.parameters() if p.requires_grad)
    mode = "HEAD-ONLY (frozen encoder linear probe)" if a.head_only else f"unfreeze last {a.unfreeze} blocks"
    print(f"train frames={len(tp)} pos={int(np.sum(tl))} | {mode} | trainable params={ntrain/1e6:.3f}M | holdout={a.holdout}")

    ds = FrameDS(tp, tl, train=True, aug=a.aug, aug_config=a.aug_config)
    sampler = PosBalancedBatchSampler(tl, a.bs, pos_per_batch=a.pos_per_batch, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=6, persistent_workers=True)
    # with a balanced sampler the in-batch neg/pos ratio is ~npb/ppb (mild), NOT the ~159 dataset ratio
    pos_w = torch.tensor([sampler.npb / sampler.ppb], device=dev, dtype=torch.float32)
    print(f"sampler: {sampler.ppb} pos + {sampler.npb} neg / batch, {sampler.nbatches} batches, pos_weight={pos_w.item():.1f}")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(layerwise_param_groups(net, a.lr, a.lr_decay), weight_decay=a.wd)
    # SWAD averages the last-N epochs: if the cosine tail has annealed to ~0 those iterates are near-identical and
    # the average is a no-op (== final epoch). Floor the tail at lr*0.1 (=1e-5, below every layerwise group's base
    # LR so no group inverts) ONLY when --swad, so the averaging window actually explores the basin. Plain cosine->0
    # otherwise. Whether the (now-functional) SWAD helps is decided by the paired LOCO gate.
    eta_min = a.lr * 0.1 if a.swad else 0.0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=eta_min)
    amp = dev == "cuda"; scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ---- semi-supervised setup: sample the VLM pool + build an EMA teacher (off unless --semi-manifest) ----
    semi_dl, ema = None, None
    if a.semi_manifest and os.path.exists(a.semi_manifest):
        import copy
        z = np.load(a.semi_manifest, allow_pickle=True)
        up, us = z["img_path"], z["suspicion"]
        ridx = np.random.default_rng(a.seed).choice(len(up), min(a.semi_n * 2, len(up)), replace=False)
        up2, us2 = up[ridx], us[ridx]
        keep = np.array([os.path.exists(p) for p in up2])          # check only the sample, not all 168k
        up2, us2 = up2[keep][:a.semi_n], us2[keep][:a.semi_n]
        uds = UnlabeledDS(up2, us2, a.aug_config)
        semi_dl = torch.utils.data.DataLoader(uds, batch_size=a.semi_bs, shuffle=True, num_workers=8,
                                              persistent_workers=True, drop_last=True, prefetch_factor=4)
        ema = copy.deepcopy(net)
        for p in ema.parameters():
            p.requires_grad_(False)
        ema.eval()
        print(f"semi: {len(up2)} VLM frames | weight={a.semi_weight} ema={a.ema_decay} conf-neg<{a.semi_lo} "
              f"rampup={a.semi_rampup}ep (consistency + one-sided-PU)")

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
    semi_iter = iter(semi_dl) if semi_dl is not None else None
    best = -1
    for ep in range(a.epochs):
        net.train(); tot = 0
        if net.head_only:
            net.backbone.eval()         # frozen encoder -> deterministic features (no drop_path/dropout noise)
        tail = ep >= a.warmup           # warm-start on BCE, then add the 90R tail terms
        semi_w = (min(1.0, max(0, ep - a.warmup + 1) / max(1, a.semi_rampup)) * a.semi_weight) if semi_dl is not None else 0.0
        for x, y in dl:
            x, y = x.to(dev), y.float().to(dev)   # cast float64->float32 on CPU BEFORE moving (MPS rejects float64)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                if a.cg_head:
                    logit, attn = net(x, return_attn=True)
                    sup = compute_loss(logit, y, tail)
                    ent = -(attn * (attn + 1e-8).log()).sum(1).mean()   # attention entropy (per image, mean)
                    sup = sup - a.attn_entropy * ent                    # maximize H -> anti-collapse (anti-memorize)
                else:
                    sup = compute_loss(net(x), y, tail)                 # supervised (labeled) loss
            if amp:
                scaler.scale(sup).backward()
            else:
                sup.backward()
            tot += sup.item()
            # SEMI: run --semi-steps unlabeled batches per labeled step (grads ACCUMULATED into the same opt.step),
            # so the semi loss sweeps MUCH more of the pool per epoch (fresh strong-aug each) WITHOUT repeating the
            # labeled set more -> no extra overfitting. Coverage/epoch = nbatches * semi_steps * semi_bs frames.
            if semi_dl is not None and tail and semi_w > 0:
                for _ in range(a.semi_steps):
                    try:
                        xw, xs, susp = next(semi_iter)
                    except StopIteration:
                        semi_iter = iter(semi_dl); xw, xs, susp = next(semi_iter)
                    xw, xs, susp = xw.to(dev), xs.to(dev), susp.float().to(dev)   # susp collates to f64 -> f32 (MPS-safe)
                    with torch.autocast(device_type="cuda", enabled=amp):
                        with torch.no_grad():
                            pt = torch.sigmoid(ema(xw))              # EMA-teacher prob on the WEAK view
                        ls = net(xs); ps = torch.sigmoid(ls)         # student on the STRONG view
                        semi = ((ps - pt) ** 2).mean()               # Mean-Teacher consistency (label-free -> anti-overfit)
                        neg = susp < a.semi_lo                       # one-sided PU: confident VLM negatives -> target 0
                        if neg.any():
                            semi = semi + nn.functional.binary_cross_entropy_with_logits(ls[neg], torch.zeros_like(ls[neg]))
                        semi = (semi_w * semi) / a.semi_steps        # average over the K accumulated unlabeled batches
                    if amp:
                        scaler.scale(semi).backward()
                    else:
                        semi.backward()
            if amp:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            if ema is not None:                                      # update the EMA teacher after the student step
                with torch.no_grad():
                    for pe, psrc in zip(ema.parameters(), net.parameters()):
                        pe.mul_(a.ema_decay).add_(psrc.detach(), alpha=1 - a.ema_decay)
                    for be, bsrc in zip(ema.buffers(), net.buffers()):
                        be.copy_(bsrc)
        sched.step()
        msg = f"ep{ep+1}/{a.epochs} loss={tot/len(dl):.4f}"
        if vam.any():
            r, s = evaluate_center(net, list(paths[vam]), list(labels[vam]), list(centers[vam]), dev, a.bs)
            ppv = r["ppv90"]
            # SELECT on AUPRC, not max PPV@90R: taking the epoch-max of PPV@90R over ~49-78 held-out positives is
            # selection-on-noise (its MDE dwarfs the margins) and it biases the SWAD-vs-best comparison. AUPRC is
            # threshold-free and stable across epochs. PPV@90R stays a printed diagnostic. (nan-safe fallback to PPV.)
            sel = r["auprc"] if not np.isnan(r["auprc"]) else ppv
            msg += (f"  LOCO-val({a.holdout}) PPV@90R={ppv:.4f} CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
                    f" AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f}")
            if sel > best:
                best = sel
                torch.save({"model": net.state_dict(), "cfg": vars(a), "loco_auprc": best, "loco_ppv90": ppv}, a.out)
                # persist the selected model's held-out y/center/scores for the PAIRED gate (ev.gate).
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
    print(f"best LOCO-val AUPRC (selection metric) = {best:.4f}")

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
