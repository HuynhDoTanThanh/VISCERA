"""Stage-1: VLM-CONCEPT-SUPERVISED PRETRAINING (the user's core idea).

Teach the encoder the VLM's clinical concepts on the large ~100k corpus, so its features are
clinically-grounded and (crucially) CENTER-INVARIANT — concepts like demarcation / mucosal_irregularity
discriminate neoplasia consistently across centers (measured: c1 & c2 AUROC both 0.8-0.9), unlike raw pixels.

Init from the domain-adapted DINOv2 (keep SSL features), unfreeze last-K blocks + heads, and:
  - MAIN heads distill the reliable discriminative/quality concepts (trust-weighted, supervise-masked BCE).
  - CENTER heads predict the center_cue concepts (black_border/overlay_graphics) through a GRADIENT-REVERSAL
    layer -> the encoder is pushed to be INVARIANT to center artifacts (the nuisance that breaks cross-center).
Output encoder -> Stage-2 downstream binary (finetune.py --init <ckpt>).

GPU (cloud). Device-portable; MPS-runnable for a smoke. Reads phase3/cache/concept_targets.npz (build_concept_targets.py).

    python -m phase3.pretrain_concept --epochs 15 --unfreeze 6 --bs 96 --out phase3/cache/concept_encoder.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image

from phase3.featurize import CKPT, IMG, _MEAN, _STD
from phase3.dataset import CONCEPT_NAMES, CONCEPT_ROLE


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------- gradient reversal (DANN-style)
class _GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd=1.0):
    return _GradRev.apply(x, lambd)


# --------------------------------------------------------------- model
class ConceptNet(nn.Module):
    def __init__(self, main_idx, center_idx, unfreeze=6):
        super().__init__()
        m = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False, img_size=IMG, num_classes=0)
        teacher = torch.load(CKPT, map_location="cpu", weights_only=False)["teacher"]
        bk = {k[len("backbone."):]: v for k, v in teacher.items() if k.startswith("backbone.")}
        from timm.models import vision_transformer as vit_mod
        conv = vit_mod.checkpoint_filter_fn(bk, m); conv.pop("mask_token", None)
        miss, unexp = m.load_state_dict(conv, strict=False)
        assert not miss and not unexp, f"backbone mismatch {miss} {unexp}"
        for p in m.parameters():
            p.requires_grad_(False)
        for i in range(max(0, len(m.blocks) - unfreeze), len(m.blocks)):
            for p in m.blocks[i].parameters():
                p.requires_grad_(True)
        for p in m.norm.parameters():
            p.requires_grad_(True)
        self.backbone = m
        self.main_idx = main_idx; self.center_idx = center_idx
        self.main_head = nn.Sequential(nn.LayerNorm(2 * 768), nn.Linear(2 * 768, len(main_idx)))
        self.center_head = nn.Sequential(nn.LayerNorm(2 * 768), nn.Linear(2 * 768, max(len(center_idx), 1)))

    def feat(self, x):
        f = self.backbone.forward_features(x)
        return torch.cat([f[:, 0], f[:, 5:].mean(1)], -1)

    def forward(self, x, grl=1.0):
        z = self.feat(x)
        return self.main_head(z), self.center_head(grad_reverse(z, grl))


# --------------------------------------------------------------- data
class ConceptDS(torch.utils.data.Dataset):
    def __init__(self, paths, value, trust, sup, main_idx, main_w, center_idx, train=True):
        self.paths = paths; self.value = value; self.trust = trust; self.sup = sup
        self.main_idx = main_idx; self.main_w = main_w; self.center_idx = center_idx
        self.tf = T.Compose([
            T.RandomResizedCrop(IMG, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
            T.RandomHorizontalFlip(), T.ColorJitter(0.35, 0.35, 0.35, 0.05),
            T.RandomApply([T.GaussianBlur(5, (0.1, 1.6))], 0.2), T.RandomRotation(12),
        ]) if train else T.Resize((IMG, IMG))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = self.tf(Image.open(self.paths[i]).convert("RGB"))
        except Exception:
            im = Image.new("RGB", (IMG, IMG))
        x = (torch.from_numpy(np.asarray(im.resize((IMG, IMG)), np.float32).copy()).permute(2, 0, 1) / 255. - _MEAN) / _STD
        # weight = supervise * role_w (trust DROPPED per audit: trust CV 3-12%, uncorrelated with correctness -> no-op)
        mv = torch.tensor(self.value[i, self.main_idx]); mw = torch.tensor(self.sup[i, self.main_idx] * self.main_w)
        cv = torch.tensor(self.value[i, self.center_idx]); cw = torch.tensor(self.sup[i, self.center_idx])
        return x, mv, mw, cv, cw


# Curated discriminative core (dense, high-trust; measured AUROC 0.81-0.91 vs neo, sign-consistent in BOTH
# centers) + a small quality head for FP-suppression. Context/gestalt/dead-constant concepts are EXCLUDED:
# uniform-weight context (modality/view/...) encodes center-specific acquisition style and would make the
# trunk MORE center-aware — the opposite of the goal. center_cue -> gradient reversal head.
CORE_W = {
    "demarcation": 1.0, "mucosal_irregularity": 1.0, "focal_erythema": 1.0, "lesion_present": 1.0,
    "nodularity": 1.0, "colocalization": 1.0, "surface_effacement": 1.0, "color_change_locality": 1.0,
    "depression_ulceration": 1.0,
    "blood": 0.3, "debris": 0.3, "mucus_bubbles": 0.3,
}


def curated_concepts(value=None, sup=None):
    """Return (main_idx, main_role_w, center_idx) — the curated core (NOT role-agnostic auto-select)."""
    main, w = [], []
    for i, nm in enumerate(CONCEPT_NAMES):
        if nm in CORE_W:
            main.append(i); w.append(CORE_W[nm])
    center = [i for i, nm in enumerate(CONCEPT_NAMES) if CONCEPT_ROLE.get(nm) == "center_cue"]
    return main, np.array(w, np.float32), center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="phase3/cache/concept_targets.npz")
    ap.add_argument("--out", default="phase3/cache/concept_encoder.pt")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--unfreeze", type=int, default=6); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--grl", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="debug subset")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev = device(); print(f"device={dev}")

    D = np.load(a.targets, allow_pickle=True)
    paths, value, trust, sup = D["paths"], D["value"], D["trust"], D["supervise"]
    main_idx, main_w, center_idx = curated_concepts(value, sup)
    if a.limit and a.limit < len(paths):
        # FAIR subset: oversample graded-positive-like frames (corpus is ~74% confident-negative, so a
        # random subset starves the discriminative heads and would unfairly null the idea).
        core_max = value[:, main_idx].max(1)
        rng = np.random.default_rng(0)
        pos_like = np.where(core_max > 0.4)[0]; rest = np.where(core_max <= 0.4)[0]
        n_pos = min(len(pos_like), a.limit // 2); n_rest = min(a.limit - n_pos, len(rest))
        sel = np.concatenate([rng.choice(pos_like, n_pos, replace=False),
                              rng.choice(rest, n_rest, replace=False)])
        paths, value, trust, sup = paths[sel], value[sel], trust[sel], sup[sel]
        print(f"stratified subset: {n_pos} graded-positive-like + {n_rest} other")
    print(f"frames={len(paths)} | MAIN core={len(main_idx)} ({[CONCEPT_NAMES[i] for i in main_idx]})")
    print(f"CENTER (GRL) concepts={[CONCEPT_NAMES[i] for i in center_idx]}")

    ds = ConceptDS(paths, value, trust, sup, main_idx, main_w, center_idx, train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                                     drop_last=True, persistent_workers=a.workers > 0)
    net = ConceptNet(main_idx, center_idx, a.unfreeze).to(dev)
    tp = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"trainable params={tp/1e6:.1f}M")
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    amp = dev == "cuda"; scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for ep in range(a.epochs):
        net.train(); tot = tm = tc = 0.0; n = 0
        lam = a.grl * (2.0 / (1.0 + np.exp(-10.0 * ep / max(a.epochs - 1, 1))) - 1.0)  # DANN ramp 0->grl
        for x, mv, mw, cv, cw in dl:
            x, mv, mw, cv, cw = x.to(dev), mv.float().to(dev), mw.float().to(dev), cv.float().to(dev), cw.float().to(dev)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                pm, pc = net(x, lam)
                lm = (bce(pm, mv) * mw).sum() / (mw.sum() + 1e-6)            # trust-weighted concept distillation
                lc = (bce(pc, cv) * cw).sum() / (cw.sum() + 1e-6) if len(center_idx) else x.sum() * 0
                loss = lm + lc                                              # GRL makes lc push center info OUT
            if amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            tot += loss.item(); tm += lm.item(); tc += float(lc); n += 1
        sched.step()
        print(f"ep{ep+1}/{a.epochs} loss={tot/n:.4f} (concept={tm/n:.4f} center_grl={tc/n:.4f} lam={lam:.2f})", flush=True)
        torch.save({"backbone": net.backbone.state_dict(), "main_idx": main_idx, "center_idx": center_idx,
                    "cfg": vars(a)}, a.out)
    print(f"saved concept-pretrained encoder -> {a.out}")


if __name__ == "__main__":
    main()
