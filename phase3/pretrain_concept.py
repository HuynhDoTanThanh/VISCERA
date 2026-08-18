"""Stage-1: VLM-CONCEPT-SUPERVISED PRETRAINING (the user's core idea).

Teach the encoder the VLM's clinical concepts on the large ~170k corpus, so its features are
clinically-grounded and (crucially) CENTER-INVARIANT — concepts like demarcation / mucosal_irregularity
discriminate neoplasia consistently across centers (measured: c1 & c2 AUROC both 0.8-0.9), unlike raw pixels.

Init from the domain-adapted DINOv2 (keep SSL features), unfreeze last-K blocks + heads, and:
  - MAIN heads distill the reliable discriminative/quality concepts (certain/uncertain/smoothed soft-BCE).
  - CENTER heads predict the center_cue concepts (black_border/overlay_graphics) through a GRADIENT-REVERSAL
    layer -> the encoder is pushed to be INVARIANT to center artifacts (the nuisance that breaks cross-center).
  - AUX heads (z.detach()) read alive context/acquisition concepts for calibration/interpretability WITHOUT
    shaping the trunk (so "use all 35" is honored without re-injecting center style).
Output encoder -> Stage-2 downstream binary (finetune.py --init <ckpt>).

FULL-35 loss (--discrim full15 + the certain/uncertain/smoothing knobs). Every knob DEFAULTS to the prior
behavior, so `masked_concept_loss` at (certain_floor=1, smooth_eps=0, unc_w=0, pos_weight_cap=1) is
algebraically identical to the old `(bce(pm,mv)*mw).sum()/mw.sum()`. This keeps each term cleanly ablatable
so concept_gate.py LOCO can attribute any delta (SD >> margins -> mild terms, leave-one-out gating).

REALITY CHECK (measured on phase3/cache/concept_targets.npz): 7 of the 35 concepts are DEAD CONSTANTS
(value == 0 for every frame: modality/distance/view/landmark/interpretable_fraction/dominant_color/lesion_size
are categoricals with no `abnormal` set -> abnormality-scalar is always 0). They are auto-dropped (you cannot
supervise a constant; GRL/aux on them is a no-op). overall_suspicion is 3.3%-observed -> dropped. So "all 35"
really trains ~24 concepts; the substance of "more labels" is discriminative 9->15.

GPU (cloud). Device-portable; MPS-runnable for a smoke. Reads phase3/cache/concept_targets.npz (build_concept_targets.py).

    # current behavior (12 curated concepts, plain masked BCE):
    python -m phase3.pretrain_concept --epochs 15 --unfreeze 6 --bs 96 --out phase3/cache/concept_encoder.pt
    # recommended v2 (full discriminative core + certain/uncertain/smoothing + rare-positive rebalancing):
    python -m phase3.pretrain_concept --epochs 15 --unfreeze 6 --bs 96 --discrim full15 \
        --certain_floor 0.7 --smooth_eps 0.05 --unc_w 0.1 --pos_weight_cap 10 --context_route detach \
        --out phase3/cache/concept_encoder_v2.pt
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
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
    """Backbone + 3 heads. main -> grad to trunk (the invariant clinical signal); center -> reversed grad
    (push center cues OUT); aux -> z.detach() (read alive context WITHOUT shaping the trunk)."""

    def __init__(self, main_idx, center_idx, aux_idx, unfreeze=6, backbone="dinov3"):
        super().__init__()
        from timm.models import vision_transformer as vit_mod
        from phase3.finetune import BACKBONES
        model_name, ckpt_path = BACKBONES[backbone]
        m = timm.create_model(model_name, pretrained=False, img_size=IMG, num_classes=0)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if backbone == "dinov2":                                    # teacher-format checkpoint -> extract backbone.*
            sd = {k[len("backbone."):]: v for k, v in sd["teacher"].items() if k.startswith("backbone.")}
        conv = vit_mod.checkpoint_filter_fn(sd, m); conv.pop("mask_token", None)
        miss, unexp = m.load_state_dict(conv, strict=False)
        assert not miss and not unexp, f"backbone {backbone} mismatch: {list(miss)[:4]} {list(unexp)[:4]}"
        for p in m.parameters():
            p.requires_grad_(False)
        for i in range(max(0, len(m.blocks) - unfreeze), len(m.blocks)):
            for p in m.blocks[i].parameters():
                p.requires_grad_(True)
        for p in m.norm.parameters():
            p.requires_grad_(True)
        self.backbone = m
        self.main_idx = main_idx; self.center_idx = center_idx; self.aux_idx = aux_idx
        self.main_head = nn.Sequential(nn.LayerNorm(2 * 768), nn.Linear(2 * 768, max(len(main_idx), 1)))
        self.center_head = nn.Sequential(nn.LayerNorm(2 * 768), nn.Linear(2 * 768, max(len(center_idx), 1)))
        self.aux_head = nn.Sequential(nn.LayerNorm(2 * 768), nn.Linear(2 * 768, max(len(aux_idx), 1)))

    def feat(self, x):
        f = self.backbone.forward_features(x)
        return torch.cat([f[:, 0], f[:, 5:].mean(1)], -1)

    def forward(self, x, grl=1.0):
        z = self.feat(x)
        return (self.main_head(z),                          # grad -> trunk
                self.center_head(grad_reverse(z, grl)),     # grad reversed -> center info pushed OUT
                self.aux_head(z.detach()))                  # grad blocked -> aux never shapes the trunk


# --------------------------------------------------------------- data
class ConceptDS(torch.utils.data.Dataset):
    """Yields (x, value[35], supervise[35]). Routing/weighting/pos_weight are applied in the loss (so the
    same batch feeds all three heads by index-slice)."""

    def __init__(self, paths, value, sup, train=True, img=None):
        self.paths = paths; self.value = value; self.sup = sup
        # Capture the resolution HERE (parent process) instead of reading the module global inside __getitem__.
        # main() rebinds IMG via `global IMG`, but DataLoader workers started with *spawn* (macOS) re-import this
        # module and see featurize.IMG (448), so --img silently did not reach the data pipeline off-Linux.
        # Colab/Linux uses fork, which inherits the rebound value — which is why real runs were unaffected.
        IMG = self.img = int(img) if img is not None else globals()["IMG"]
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
            # DO NOT silently substitute a black frame. This used to swallow every failure, so a target matrix
            # whose paths don't resolve on THIS machine (e.g. a Drive-cached npz built locally with
            # `dataset/train/...png` paths, unzipped on Colab where only `out/` exists) trained the encoder to map
            # a BLACK IMAGE to real concept targets — for 30 epochs, on the 2,476 labeled frames incl. all 127
            # positives. main() preflights this, so reaching here means a genuinely corrupt file.
            raise RuntimeError(
                f"ConceptDS: cannot read {self.paths[i]!r}. Refusing to train on a blank substitute — "
                f"rebuild the target matrix on this machine (`python -m phase3.build_concept_targets`).")
        x = (torch.from_numpy(np.asarray(im.resize((self.img, self.img)), np.float32).copy())
             .permute(2, 0, 1) / 255. - _MEAN) / _STD
        return x, torch.from_numpy(self.value[i].copy()), torch.from_numpy(self.sup[i].copy())


# --------------------------------------------------------------- unified certain / uncertain / smoothing loss
def masked_concept_loss(logits, value, sup, role_w, prior, pos_w,
                        certain_floor=1.0, smooth_eps=0.0, unc_w=0.0):
    """One soft-BCE cell expression carrying all three terms. Shapes: logits/value/sup (B,K); role_w/prior/pos_w (K,).

    CERTAIN  : assessable cells (s=1) supervised on the smoothed value, weighted by margin (up-weight confident
               calls), class-balanced by per-concept pos_weight (rare-positive discriminative binaries).
    UNCERTAIN: not_assessable cells (s=0) get a small-weight (unc_w) prior-pull toward p_j -- do NOT invent a
               label (treating NA as negative would teach "unassessable = negative", a center shortcut).
    SMOOTHING: target shrinks toward the per-concept prior p_j (marginal, not 0.5) so the head is never more
               confident than the noisy VLM warrants (overconfident heads memorize center cues).

    Defaults (certain_floor=1, smooth_eps=0, unc_w=0, pos_w=1) reduce EXACTLY to masked role-weighted BCE.
    """
    v, s = value, sup
    margin = 2.0 * (v - 0.5).abs()                                              # 1 = hard call, 0 = ambiguous mid
    c = role_w * (s * (certain_floor + (1.0 - certain_floor) * margin)         # certain (assessable) weight ...
                  + (1.0 - s) * unc_w)                                          # ... + uncertain (masked) weight
    tgt = s * ((1.0 - smooth_eps) * v + smooth_eps * prior) + (1.0 - s) * prior  # smoothed / prior-anchored target
    pw = s * pos_w + (1.0 - s) * 1.0                                            # pos_weight only on the certain part
    # numerically-stable soft-BCE-with-pos_weight:  pw*t*softplus(-x) + (1-t)*softplus(x)
    bce = pw * tgt * F.softplus(-logits) + (1.0 - tgt) * F.softplus(logits)
    return (c * bce).sum() / (c.sum() + 1e-6)


# --------------------------------------------------------------- concept routing (the "which head" table)
# Curated discriminative core historically used (dense, high-trust; measured AUROC 0.81-0.91 vs neo,
# sign-consistent in BOTH centers). "full15" adds the 6 alive-but-unused discriminative concepts (the real
# substance of "more labels").
CURATED9 = ["demarcation", "mucosal_irregularity", "focal_erythema", "lesion_present", "nodularity",
            "colocalization", "surface_effacement", "color_change_locality", "depression_ulceration"]
SCENE_QUALITY = ["blood", "debris", "mucus_bubbles"]     # genuine scene FP-suppression, not center-specific -> MAIN @0.3
ACQ_QUALITY = ["blur", "glare", "exposure"]              # acquisition/scope style -> AUX (detached) by default
ALIVE_CONTEXT = ["magnification", "mucosal_pattern_type", "vessels_assessable"]  # informative context, contested -> AUX


def _idx(names):
    s = set(names)
    return [i for i, nm in enumerate(CONCEPT_NAMES) if nm in s]


def route_concepts(value, sup, discrim="curated9", context_route="detach"):
    """Return (main_idx, center_idx, aux_idx, role_w35). Dead-constant concepts (value with ~0 variance among
    supervised) are ALWAYS excluded from every head -- you cannot supervise a constant. context_route governs
    the alive context + acquisition-quality concepts: detach (AUX, default) / main (MAIN, ablation) /
    grl (CENTER) / drop (exclude)."""
    role_w = np.zeros(len(CONCEPT_NAMES), np.float32)

    # 1) detect dead constants empirically (robust to schema changes): std of value among supervised ~ 0.
    alive = np.zeros(len(CONCEPT_NAMES), bool)
    for i in range(len(CONCEPT_NAMES)):
        m = sup[:, i] > 0.5
        alive[i] = bool(m.sum()) and float(value[m, i].std()) > 1e-4
    dead = [CONCEPT_NAMES[i] for i in range(len(CONCEPT_NAMES)) if not alive[i]]

    # 2) MAIN: discriminative core (+ scene-quality FP-suppression @0.3)
    disc = CURATED9 if discrim == "curated9" else [nm for nm in CONCEPT_NAMES if CONCEPT_ROLE[nm] == "discriminative"]
    main = [i for i in _idx(disc) if alive[i]]
    for i in main:
        role_w[i] = 1.0
    for i in _idx(SCENE_QUALITY):
        if alive[i]:
            main.append(i); role_w[i] = 0.3

    # 3) CENTER (GRL): the real center cues (black_border/overlay_graphics)
    center = [i for i in range(len(CONCEPT_NAMES)) if CONCEPT_ROLE[CONCEPT_NAMES[i]] == "center_cue" and alive[i]]
    for i in center:
        role_w[i] = 1.0

    # 4) contested = alive context + acquisition-quality (routed by context_route)
    contested = [i for i in _idx(ALIVE_CONTEXT + ACQ_QUALITY) if alive[i]]
    aux = []
    if context_route == "detach":
        aux = contested
        for i in aux:
            role_w[i] = 0.3 if CONCEPT_NAMES[i] in ACQ_QUALITY else 1.0
    elif context_route == "main":
        for i in contested:
            main.append(i); role_w[i] = 0.3 if CONCEPT_NAMES[i] in ACQ_QUALITY else 1.0
    elif context_route == "grl":
        for i in contested:
            center.append(i); role_w[i] = 1.0
    # context_route == "drop": contested excluded everywhere

    return sorted(set(main)), sorted(set(center)), sorted(set(aux)), role_w, dead


def priors_and_posw(value, sup, pos_weight_cap):
    """Per-concept assessable base rate p_j (pooled across centers) and pos_weight=(1-p)/p capped."""
    s = sup.astype(np.float32)
    prior = (s * value).sum(0) / (s.sum(0) + 1e-6)
    posw = np.clip((1.0 - prior) / np.clip(prior, 1e-3, 1.0), 1.0, pos_weight_cap).astype(np.float32)
    return prior.astype(np.float32), posw


def main():
    global IMG                               # --img overrides the resolution used by ConceptNet/ConceptDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=IMG,
                    help="Stage-1 input resolution. MUST match the Stage-2 --img you will fine-tune at, or the "
                         "concept encoder is trained on a different token grid than it is consumed at (pos_embed is "
                         "interpolated at load, but the blocks were adapted to the other scale). featurize.IMG was "
                         "flipped 336->448 for the dinov3 era (commit f4271e2), so any Stage-1 built since then "
                         "silently ran @448 while the winning ship runs @336 — pass 336 for the dinov2 recipe.")
    ap.add_argument("--resume", action="store_true",
                    help="continue from an existing --out checkpoint (Stage-1 is ~7.5h on 291k frames @336 and "
                         "Colab sessions drop). Restores epoch/optimizer/schedule when the checkpoint has them; "
                         "otherwise warm-starts from its backbone weights.")
    ap.add_argument("--holdout", default="none",
                    help="LEAK GUARD for LOCO gating. concept_targets.npz contains the 2,476 LABELED train frames "
                         "(127 pos / 2,349 neg, center_1 1823 / center_2 653) alongside the 167,724 unlabeled ones. "
                         "Stage-1 never reads the binary label, but it DOES distil concepts that are 0.87-0.91 AUROC "
                         "proxies for it (mucosal_irregularity, demarcation, nodularity) on those exact images — so a "
                         "`finetune --holdout center_2 --init concept_encoder.pt` LOCO run evaluates on frames the "
                         "encoder was trained on for 30 epochs with label-correlated targets. That makes EVERY "
                         "concept-init LOCO number optimistic, including --loco-no-semi runs. "
                         "'center_1'/'center_2' = drop that center's labeled frames; 'labeled' = drop all 2,476 "
                         "(1.5% of the corpus) for ONE encoder that is honest for BOTH legs; 'none' = ship setting "
                         "(correct: the hidden test was never in Stage-1, so the leaderboard is unaffected).")
    ap.add_argument("--targets", default="phase3/cache/concept_targets.npz")
    ap.add_argument("--out", default="phase3/cache/concept_encoder.pt")
    ap.add_argument("--backbone", choices=["dinov2", "dinov3"], default="dinov3",
                    help="dinov3 = ViT-B/16 (dinov3.pth) | dinov2 = ViT-B/14-reg (dinov2.pth). Must match Stage-2 --backbone.")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--unfreeze", type=int, default=6); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--grl", type=float, default=1.0)
    ap.add_argument("--l2sp", type=float, default=1.0,
                    help="L2-SP anchor to SSL init (keeps concepts from OVERWRITING SSL features; 0=off). KEY lever.")
    # ---- full-35 / certain-uncertain-smoothing knobs (all default to the PRIOR behavior) ----
    ap.add_argument("--discrim", choices=["curated9", "full15"], default="curated9",
                    help="MAIN discriminative set. full15 = +6 alive-but-unused discriminative concepts ('more labels').")
    ap.add_argument("--context_route", choices=["detach", "main", "grl", "drop"], default="detach",
                    help="alive context+acq-quality -> AUX(detach, safe default) / MAIN(ablation) / CENTER(grl) / drop.")
    ap.add_argument("--certain_floor", type=float, default=1.0,
                    help="CERTAIN: floor of margin weight (gamma). 1.0=off (plain supervise mask). 0.7=down-weight ambiguous.")
    ap.add_argument("--smooth_eps", type=float, default=0.0,
                    help="SMOOTHING: target shrink toward per-concept prior. 0=off. 0.05 recommended.")
    ap.add_argument("--unc_w", type=float, default=0.0,
                    help="UNCERTAIN: prior-pull weight on not_assessable (s=0) cells. 0=off (hard-mask). 0.1 recommended.")
    ap.add_argument("--pos_weight_cap", type=float, default=1.0,
                    help="rare-positive rebalancing: pos_weight=(1-p)/p capped here. 1.0=off. 10 recommended.")
    ap.add_argument("--limit", type=int, default=0, help="debug subset")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    IMG = a.img                              # override before ConceptNet/ConceptDS read the module global
    dev = device(); print(f"device={dev} | img={IMG} (Stage-2 --img MUST match this)")

    D = np.load(a.targets, allow_pickle=True)
    paths, value, sup = D["paths"], D["value"], D["supervise"]
    cen_cur = D["center"].astype(str)          # travels with `paths` through holdout + --limit filters
    if a.holdout != "none":
        # Drop labeled frames so a LOCO gate is not evaluated on images this encoder was trained on.
        # Labeled rows are exactly those with a non-empty `center` (the 167,724 unlabeled ones have center == "").
        cen = D["center"].astype(str)
        drop = (cen != "") if a.holdout == "labeled" else (cen == a.holdout)
        keep = ~drop
        print(f"LEAK GUARD --holdout {a.holdout}: dropped {int(drop.sum())} labeled frames "
              f"({int(drop.sum())/len(paths):.2%} of the corpus) -> {int(keep.sum())} remain. "
              f"Use this encoder ONLY for gating; the ship encoder must be --holdout none.")
        paths, value, sup, cen_cur = paths[keep], value[keep], sup[keep], cen_cur[keep]
    main_idx, center_idx, aux_idx, role_w35, dead = route_concepts(value, sup, a.discrim, a.context_route)
    prior35, posw35 = priors_and_posw(value, sup, a.pos_weight_cap)
    if a.limit and a.limit < len(paths):
        # FAIR subset: oversample graded-positive-like frames (corpus is ~74% confident-negative, so a
        # random subset starves the discriminative heads and would unfairly null the idea).
        core_max = value[:, main_idx].max(1) if main_idx else np.zeros(len(value))
        rng = np.random.default_rng(0)
        pos_like = np.where(core_max > 0.4)[0]; rest = np.where(core_max <= 0.4)[0]
        n_pos = min(len(pos_like), a.limit // 2); n_rest = min(a.limit - n_pos, len(rest))
        sel = np.concatenate([rng.choice(pos_like, n_pos, replace=False),
                              rng.choice(rest, n_rest, replace=False)])
        paths, value, sup, cen_cur = paths[sel], value[sel], sup[sel], cen_cur[sel]
        print(f"stratified subset: {n_pos} graded-positive-like + {n_rest} other")
    # PREFLIGHT: verify the target matrix's paths actually resolve HERE. A matrix built on another machine (or
    # restored from Drive) can carry paths that do not exist in this environment; the old blank-image fallback
    # made that invisible and silently poisoned Stage-1. Check the labeled rows in full (they are few and carry
    # the strongest, most label-correlated supervision) plus a sample of the rest.
    _lab_idx = np.where(cen_cur != "")[0]
    _chk = np.unique(np.concatenate([_lab_idx, np.arange(0, len(paths), max(1, len(paths) // 2000))]))
    _missing = [paths[i] for i in _chk if not os.path.exists(paths[i])]
    if _missing:
        raise FileNotFoundError(
            f"{len(_missing)} of {len(_chk)} checked concept-target paths do not exist here "
            f"(e.g. {_missing[:3]}). This matrix was built elsewhere — on Colab the labeled rows are stored as "
            f"'dataset/...' but only 'out/' is unzipped, so Stage-1 would have trained {len(_lab_idx)} labeled "
            f"frames (incl. every positive) as BLANK images. Rebuild it here: "
            f"`python -m phase3.build_concept_targets --out {a.targets}`")
    print(f"preflight OK: {len(_chk)} sampled paths resolve ({len(_lab_idx)} labeled rows checked in full)")
    nm = lambda idx: [CONCEPT_NAMES[i] for i in idx]
    print(f"frames={len(paths)}")
    print(f"MAIN (grad->trunk) [{len(main_idx)}]: {nm(main_idx)}")
    print(f"CENTER (GRL)       [{len(center_idx)}]: {nm(center_idx)}")
    print(f"AUX (detached)     [{len(aux_idx)}]: {nm(aux_idx)}")
    print(f"DROPPED (dead-const / gestalt / route=drop) [{len(dead)}]: {dead}")
    print(f"knobs: discrim={a.discrim} context_route={a.context_route} certain_floor={a.certain_floor} "
          f"smooth_eps={a.smooth_eps} unc_w={a.unc_w} pos_weight_cap={a.pos_weight_cap}")

    ds = ConceptDS(paths, value, sup, train=True, img=IMG)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                                     drop_last=True, persistent_workers=a.workers > 0)
    net = ConceptNet(main_idx, center_idx, aux_idx, a.unfreeze, backbone=a.backbone).to(dev)
    tp = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"trainable params={tp/1e6:.1f}M  l2sp={a.l2sp}")
    # L2-SP reference: snapshot the trainable backbone weights at SSL init (anchor to keep features)
    sp_ref = {n: p.detach().clone() for n, p in net.backbone.named_parameters() if p.requires_grad} if a.l2sp > 0 else {}
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    amp = dev == "cuda"; scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # per-concept buffers, sliced per head (pos_weight/prior/role_w travel with their head's concepts)
    rw = torch.tensor(role_w35, device=dev); pr = torch.tensor(prior35, device=dev); pw = torch.tensor(posw35, device=dev)
    mi = torch.tensor(main_idx, dtype=torch.long, device=dev)
    ci = torch.tensor(center_idx, dtype=torch.long, device=dev)
    ai = torch.tensor(aux_idx, dtype=torch.long, device=dev)
    kw = dict(certain_floor=a.certain_floor, smooth_eps=a.smooth_eps, unc_w=a.unc_w)

    # ---- RESUME: Stage-1 is ~7.5h on 291k frames @336 and Colab disconnects. Restart from the last epoch
    # instead of losing the run. Full resume needs epoch/opt/sched/scaler (saved below); a checkpoint from an
    # older build has only the backbone, so we warm-start from its weights and restart the schedule.
    start_ep = 0
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location="cpu", weights_only=False)
        net.backbone.load_state_dict(ck["backbone"])
        for h in ("main_head", "center_head", "aux_head"):
            if h in ck:
                getattr(net, h).load_state_dict(ck[h])
        if "epoch" in ck:
            start_ep = int(ck["epoch"])
            if "opt" in ck: opt.load_state_dict(ck["opt"])
            if "sched" in ck: sched.load_state_dict(ck["sched"])
            if "scaler" in ck and amp: scaler.load_state_dict(ck["scaler"])
            print(f"RESUMED from {a.out} at epoch {start_ep}/{a.epochs} (optimizer + schedule restored)")
        else:
            print(f"WARM-START from {a.out} (pre-resume checkpoint: backbone only, no epoch/optimizer state -> "
                  f"the LR schedule and GRL ramp restart from 0; set --epochs to the REMAINING count)")
        if start_ep >= a.epochs:
            print("already complete — nothing to do"); return

    for ep in range(start_ep, a.epochs):
        net.train(); tot = tm = tc = ta = 0.0; n = 0
        lam = a.grl * (2.0 / (1.0 + np.exp(-10.0 * ep / max(a.epochs - 1, 1))) - 1.0)  # DANN ramp 0->grl
        for x, v, s in dl:
            x, v, s = x.to(dev), v.float().to(dev), s.float().to(dev)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                pm, pc, pa = net(x, lam)
                lm = masked_concept_loss(pm, v[:, mi], s[:, mi], rw[mi], pr[mi], pw[mi], **kw) if len(main_idx) else x.sum() * 0
                lc = masked_concept_loss(pc, v[:, ci], s[:, ci], rw[ci], pr[ci], pw[ci], **kw) if len(center_idx) else x.sum() * 0
                la = masked_concept_loss(pa, v[:, ai], s[:, ai], rw[ai], pr[ai], pw[ai], **kw) if len(aux_idx) else x.sum() * 0
                loss = lm + lc + la                                         # GRL makes lc push center info OUT; la detached
                if sp_ref:                                                  # L2-SP: anchor to SSL init (anti-forgetting)
                    sp = sum(((p - sp_ref[n]) ** 2).mean() for n, p in net.backbone.named_parameters() if n in sp_ref)
                    loss = loss + a.l2sp * sp
            if amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            tot += loss.item(); tm += float(lm); tc += float(lc); ta += float(la); n += 1
        sched.step()
        print(f"ep{ep+1}/{a.epochs} loss={tot/n:.4f} (main={tm/n:.4f} center_grl={tc/n:.4f} aux={ta/n:.4f} lam={lam:.2f})", flush=True)
        # save heads + optimizer/schedule/scaler + epoch so --resume can continue exactly (Stage-1 is ~7.5h)
        torch.save({"backbone": net.backbone.state_dict(), "main_idx": main_idx, "center_idx": center_idx,
                    "aux_idx": aux_idx, "cfg": vars(a), "epoch": ep + 1,
                    "main_head": net.main_head.state_dict(), "center_head": net.center_head.state_dict(),
                    "aux_head": net.aux_head.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict() if amp else None}, a.out)
    # final save: drop the optimizer/scaler state (~170MB of AdamW moments) so the encoder shipped to Stage-2
    # and cached on Drive stays lean. Stage-2 only reads ["backbone"] and ["cfg"].
    torch.save({"backbone": net.backbone.state_dict(), "main_idx": main_idx, "center_idx": center_idx,
                "aux_idx": aux_idx, "cfg": vars(a), "epoch": a.epochs,
                "main_head": net.main_head.state_dict(), "center_head": net.center_head.state_dict(),
                "aux_head": net.aux_head.state_dict()}, a.out)
    print(f"saved concept-pretrained encoder -> {a.out}")


if __name__ == "__main__":
    main()
