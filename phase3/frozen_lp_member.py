"""Frozen-DINOv2 linear-probe member — the top LOCO config (§19.2b: cross-center AUROC 0.91-0.95,
vs DINOv3 0.78-0.90 and fine-tuned exps/2 hidden 0.854). Maximally preserves foundation features
(Kumar LP-FT): a linear head on FROZEN DINOv2 [cls⊕mean-pool]. Self-contained, offline, no fine-tuning.

Two uses:
  1. STANDALONE submission member: score_frames(frames)->probs (drop into a container).
  2. DECORRELATED ensemble member for exps/2 (dinov2 FT ⊕ dinov2 frozen-LP — same backbone family, but
     frozen-vs-FT decorrelates; NEVER pair with dinov3 which drags the LOCO, §19.2b).

Fit the LP once with refit() on ALL labeled data (train+val); ship the tiny (mean,scale,coef,intercept) npz.
"""
import os, numpy as np, torch
from PIL import Image
import timm, timm.models.vision_transformer as vit_mod

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_IMG = 336


def _dinov2_frozen(weight_path, device):
    m = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False, img_size=_IMG, num_classes=0)
    sd = torch.load(weight_path, map_location="cpu")
    sd = sd["teacher"] if isinstance(sd, dict) and "teacher" in sd else sd.get("model", sd)
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")} or sd
    try: sd = vit_mod.checkpoint_filter_fn(sd, m)
    except Exception: pass
    miss, _ = m.load_state_dict(sd, strict=False)
    assert not miss, f"frozen dinov2 miss={miss[:4]}"
    return m.to(device).eval()


class FrozenLPMember:
    def __init__(self, dinov2_weights, lp_npz, device=None, bs=32):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available()
                                              else "cuda" if torch.cuda.is_available() else "cpu")
        self.bs = bs
        self.net = _dinov2_frozen(dinov2_weights, self.device)
        d = np.load(lp_npz)
        self.mean, self.scale, self.coef, self.b = d["mean"], d["scale"], d["coef"], float(np.ravel(d["intercept"])[0])

    @torch.no_grad()
    def _feats(self, frames):
        F = np.zeros((len(frames), 1536), np.float32)
        for st in range(0, len(frames), self.bs):
            ims = []
            for im in frames[st:st + self.bs]:
                if isinstance(im, np.ndarray):
                    if im.dtype != np.uint8:
                        mx = float(im.max()); im = (im / mx * 255).clip(0, 255).astype(np.uint8) if mx > 255 else im.astype(np.uint8)
                    im = Image.fromarray(im)
                ims.append(im.convert("RGB").resize((_IMG, _IMG), Image.BILINEAR))
            xb = torch.stack([torch.from_numpy(np.asarray(i, np.float32).copy()).permute(2, 0, 1) / 255. for i in ims])
            xb = ((xb - _MEAN) / _STD).to(self.device)
            f = self.net.forward_features(xb)
            F[st:st + xb.shape[0]] = torch.cat([f[:, 0], f[:, 5:].mean(1)], -1).float().cpu().numpy()
        return F

    def score_frames(self, frames):
        """frames: list of PIL or HWC-numpy -> np.array of neoplasia probs in [0,1]."""
        z = (self._feats(frames) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.b)))


def refit(dinov2_weights, paths, labels, out_npz, device=None):
    """Fit the LP on ALL labeled frozen features (train+val) and save the shippable member npz."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    m = FrozenLPMember.__new__(FrozenLPMember)
    m.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu"); m.bs = 32
    m.net = _dinov2_frozen(dinov2_weights, m.device)
    F = m._feats([Image.open(p) for p in paths]); y = np.asarray(labels)
    sc = StandardScaler().fit(F); lp = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(sc.transform(F), y)
    np.savez(out_npz, mean=sc.mean_, scale=sc.scale_, coef=lp.coef_[0], intercept=lp.intercept_)
    print(f"[frozen-LP] refit on {len(y)} frames ({int(y.sum())} pos) -> {out_npz}", flush=True)


if __name__ == "__main__":   # quick self-check on val: score + AUROC (should ~match harness LP)
    import csv
    from sklearn.metrics import roc_auc_score
    ROOT = "/Volumes/Shin/RARE2026"
    rows = [r for r in csv.DictReader(open(f"{ROOT}/dataset/val.csv")) if r["aug"] == "orig"]
    mem = FrozenLPMember(f"{ROOT}/dinov2.pth", f"{ROOT}/phase3/cache/frozen_lp_dinov2.npz")
    probs = mem.score_frames([Image.open(r["path"]) for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    print(f"self-check: N={len(y)} pos={y.sum()} AUROC={roc_auc_score(y, probs):.4f} "
          f"(fit-on-val so optimistic; sanity only) range=[{probs.min():.3f},{probs.max():.3f}]")
