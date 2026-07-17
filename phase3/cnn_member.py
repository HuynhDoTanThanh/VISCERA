"""D2F+ decorrelated CNN member (ConvNeXt-T / ResNet50) — a DIFFERENT architecture family from our ViTs, so it
makes DIFFERENT cross-center mistakes (local texture/edge vs patch-token acquisition style). Averaged with the
GastroNet-DINOv2-ViT-B anchor it cancels per-family center bias = the winner's DOMINANT lever (ResNet50 ⊕ ViT).

Self-contained (does NOT touch the ViT pipeline). Two stages, mirroring the ViT recipe:
  Stage-1 (concept): ImageNet-init CNN, concept-supervised on the 144k VLM-concept pool (masked soft-BCE, 35 concepts)
                     -> center-agnostic, clinically-grounded features (concept_targets.npz).
  Stage-2 (finetune): fine-tune on the 127 labels (BCE + pairwise-rank, pos-balanced), WiSE-FT toward the concept init.
Usage:
  python -m phase3.cnn_member --stage concept  --arch convnext_tiny --targets phase3/cache/concept_targets.npz --epochs 15 --out cnn_concept.pt
  python -m phase3.cnn_member --stage finetune --arch convnext_tiny --init cnn_concept.pt --train-csv train_colab.csv --holdout none --epochs 20 --wise-ft 0.7 --out cnn_member.pt
Score (ensemble/container): CNNMember('cnn_member.pt').score_frames(list_of_PIL_or_HWC) -> np.array[0,1].
"""
import argparse, os, sys, numpy as np, torch, torch.nn as nn
from PIL import Image
import timm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3.dataset import CONCEPT_NAMES

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NC = len(CONCEPT_NAMES)


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class CNNNet(nn.Module):
    """timm CNN (num_classes=0 -> global-pooled features) + a concept head (Stage-1) + a binary head (Stage-2)."""
    def __init__(self, arch="convnext_tiny", pretrained=True):
        super().__init__()
        self.arch = arch
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.dim = d
        self.concept_head = nn.Linear(d, NC)
        self.cls_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def features(self, x):
        return self.backbone(x)                       # (B, d) pooled

    def forward(self, x):                             # binary neoplasia logit
        return self.cls_head(self.features(x)).squeeze(-1)

    def forward_concepts(self, x):                    # (B, 35) concept logits
        return self.concept_head(self.features(x))


# ---------------- data ----------------
def _load_img(p, img, train):
    im = Image.open(p).convert("RGB")
    if train:
        # light aug only (CNN Stage-2 on 127 labels — heavy aug overfits; matches the light ViT recipe)
        import torchvision.transforms as T
        tf = T.Compose([T.RandomResizedCrop(img, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                        T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.3, 0.04), T.RandomRotation(10)])
        im = tf(im)
    else:
        im = im.resize((img, img), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im.resize((img, img)), np.float32).copy()).permute(2, 0, 1) / 255.
    return (x - _MEAN[0]) / _STD[0]


class ConceptDS(torch.utils.data.Dataset):
    def __init__(self, paths, value, sup, img):
        self.paths, self.value, self.sup, self.img = paths, value, sup, img

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        return (_load_img(self.paths[i], self.img, True),
                torch.from_numpy(self.value[i].copy()), torch.from_numpy(self.sup[i].copy()))


class LabelDS(torch.utils.data.Dataset):
    def __init__(self, paths, labels, img, train=True):
        self.paths, self.labels, self.img, self.train = paths, labels, img, train

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        return _load_img(self.paths[i], self.img, self.train), float(self.labels[i])


# ---------------- losses ----------------
def masked_concept_loss(logits, value, sup):
    """Plain masked soft-BCE over the 35 concepts (supervised cells only)."""
    l = nn.functional.binary_cross_entropy_with_logits(logits, value, reduction="none")
    return (l * sup).sum() / sup.sum().clamp_min(1.0)


def set_trainable(backbone, unfreeze_stages):
    """LP-FT freeze control. unfreeze_stages=0 => FULLY FROZEN encoder (head-only linear probe on the converged
    concept features = our best cross-center config). >0 => unfreeze the last N stages + final norm (light FT)."""
    for p in backbone.parameters():
        p.requires_grad_(False)
    if unfreeze_stages <= 0:
        return
    if hasattr(backbone, "stages"):                       # ConvNeXt: stages[0..3]
        stages = list(backbone.stages)
    elif hasattr(backbone, "layer4"):                     # ResNet: layer1..4
        stages = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
    else:
        stages = list(backbone.children())
    for st in stages[-unfreeze_stages:]:
        for p in st.parameters():
            p.requires_grad_(True)
    for nm in ("norm", "norm_pre", "head_norm"):          # final norm adapts feature scale
        if hasattr(backbone, nm):
            for p in getattr(backbone, nm).parameters():
                p.requires_grad_(True)


def pairwise_rank(s, y, margin=1.0):
    """softplus tail-margin over all pos x neg pairs in the batch (fires only when both classes present)."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return s.new_tensor(0.0)
    d = pos.unsqueeze(1) - neg.unsqueeze(0)          # (P, N)
    return nn.functional.softplus(margin - d).mean()


# ---------------- training ----------------
def train_concept(a):
    dev = _dev()
    d = np.load(a.targets, allow_pickle=True)
    cen = d["center"]; keep = np.isin(cen, ["center_1", "center_2"]) if a.labeled_only else np.ones(len(cen), bool)
    paths = d["paths"][keep]; value = d["value"][keep].astype(np.float32); sup = d["supervise"][keep].astype(np.float32)
    print(f"[concept] {len(paths)} frames | supervised cells/concept avg={sup.mean(0).mean():.2f}", flush=True)
    net = CNNNet(a.arch, pretrained=not a.scratch).to(dev)
    dl = torch.utils.data.DataLoader(ConceptDS(paths, value, sup, a.img), batch_size=a.bs, shuffle=True,
                                     num_workers=a.workers, drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    for ep in range(a.epochs):
        net.train(); tot = 0.0
        for x, v, s in dl:
            x, v, s = x.to(dev), v.to(dev), s.to(dev)
            loss = masked_concept_loss(net.forward_concepts(x), v, s)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"  ep{ep+1}/{a.epochs} concept_loss={tot/len(dl):.4f}", flush=True)
    torch.save({"arch": a.arch, "backbone": net.backbone.state_dict()}, a.out)
    print(f"[concept] saved -> {a.out}", flush=True)


def train_finetune(a):
    import csv
    dev = _dev()
    rows = [r for r in csv.DictReader(open(a.train_csv))]
    if a.holdout and a.holdout != "none":
        rows = [r for r in rows if r["center"] != a.holdout]
    paths = [r["path"] for r in rows]; labels = np.array([int(r["label"]) for r in rows])
    print(f"[finetune] {len(paths)} frames pos={labels.sum()} holdout={a.holdout}", flush=True)
    net = CNNNet(a.arch, pretrained=True).to(dev)
    init0 = None
    if a.init:
        ck = torch.load(a.init, map_location="cpu"); net.backbone.load_state_dict(ck["backbone"]);
        init0 = {k: v.clone() for k, v in net.backbone.state_dict().items()}     # WiSE-FT anchor
        print(f"[finetune] concept-init from {a.init}", flush=True)
    set_trainable(net.backbone, a.unfreeze_stages)                              # LP-FT: freeze encoder (0=head-only)
    ntr = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    print(f"[finetune] unfreeze_stages={a.unfreeze_stages} -> trainable {ntr:.1f}M "
          f"({'FROZEN encoder / head-only LP' if a.unfreeze_stages == 0 else 'light FT'})", flush=True)
    # pos-balanced sampler: oversample positives so rank/pAUC pairs fire every batch
    w = np.where(labels == 1, (labels == 0).sum() / max((labels == 1).sum(), 1), 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True)
    dl = torch.utils.data.DataLoader(LabelDS(paths, labels, a.img, True), batch_size=a.bs, sampler=sampler,
                                     num_workers=a.workers, drop_last=True)
    pw = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)], device=dev)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    for ep in range(a.epochs):
        net.train(); tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            s = net(x)
            loss = nn.functional.binary_cross_entropy_with_logits(s, y, pos_weight=pw) + 0.5 * pairwise_rank(s, y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"  ep{ep+1}/{a.epochs} loss={tot/len(dl):.4f}", flush=True)
    if init0 is not None and a.unfreeze_stages > 0 and 0.0 < a.wise_ft < 1.0:    # WiSE-FT only if encoder moved
        sd = net.backbone.state_dict()
        for k in sd:
            if sd[k].dtype.is_floating_point:
                sd[k] = a.wise_ft * sd[k] + (1 - a.wise_ft) * init0[k].to(sd[k].device)
        net.backbone.load_state_dict(sd); print(f"[finetune] WiSE-FT a={a.wise_ft}", flush=True)
    torch.save({"arch": a.arch, "img": a.img, "model": net.state_dict()}, a.out)
    print(f"[finetune] saved member -> {a.out}", flush=True)


class CNNMember:
    """Scorer for the ensemble / container. Mirrors frozen_lp_member.score_frames API."""
    def __init__(self, ckpt, device=None, bs=32):
        self.device = device or _dev(); self.bs = bs
        ck = torch.load(ckpt, map_location="cpu")
        self.img = int(ck.get("img", 224))
        self.net = CNNNet(ck["arch"], pretrained=False)
        self.net.load_state_dict(ck["model"]); self.net.to(self.device).eval()

    @torch.no_grad()
    def score_frames(self, frames):
        out = np.zeros(len(frames), np.float64)
        for st in range(0, len(frames), self.bs):
            xs = []
            for im in frames[st:st + self.bs]:
                if isinstance(im, np.ndarray):
                    if im.dtype != np.uint8:
                        mx = float(im.max()); im = (im / mx * 255).clip(0, 255).astype(np.uint8) if mx > 255 else im.astype(np.uint8)
                    im = Image.fromarray(im)
                im = im.convert("RGB").resize((self.img, self.img), Image.BILINEAR)
                xs.append(torch.from_numpy(np.asarray(im, np.float32).copy()).permute(2, 0, 1) / 255.)
            x = ((torch.stack(xs) - _MEAN) / _STD).to(self.device)
            out[st:st + x.shape[0]] = torch.sigmoid(self.net(x)).float().cpu().numpy()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["concept", "finetune"], required=True)
    ap.add_argument("--arch", default="convnext_tiny", help="convnext_tiny (768-d) | resnet50 (2048-d, winner's family)")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--targets", default="phase3/cache/concept_targets.npz")
    ap.add_argument("--labeled-only", action="store_true", help="Stage-1 on labeled frames only (fast smoke); default = full pool")
    ap.add_argument("--train-csv", default="train_colab.csv")
    ap.add_argument("--holdout", default="none", help="center to hold out (LOCO); none = ship")
    ap.add_argument("--init", default="", help="Stage-1 concept ckpt for Stage-2 init")
    ap.add_argument("--unfreeze-stages", type=int, default=0,
                    help="LP-FT freeze depth for Stage-2: 0 = FROZEN encoder / head-only linear probe (OOD-safe default, "
                         "preserves converged concept features); 1-2 = light FT of the last stage(s). Gate on LOCO.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--wise-ft", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scratch", action="store_true", help="Stage-1 from random init (default = ImageNet)")
    ap.add_argument("--out", default="cnn_member.pt")
    a = ap.parse_args()
    (train_concept if a.stage == "concept" else train_finetune)(a)


if __name__ == "__main__":
    main()
