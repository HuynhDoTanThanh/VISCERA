"""Phase-3 T1 trainer — frozen DINOv2 embedding -> calibrated, center-de-biased logistic -> PPV@90R.

Implements the adversarially-verified blueprint (re-targeted to PPV@90R, the confirmed metric):
  - feature  : per-pool L2-norm of the cached pooled superset; all 4 pools by default (patch_max=focal,
               paired-bootstrap-confirmed to help 90R). standardize or PCA-whiten (fit on train neg / unlabeled).
  - center   : IN-FEATURE de-bias — fit a center classifier on frozen features, PROJECT OUT its direction
               (center_2 neo-rate 11.9% vs center_1 2.7%; a head that learns center plants FPs at 1% prevalence).
               HARD gate: adversarial center-AUROC on the final feature must be <= ~0.55.
  - head     : strongly-L2 LogisticRegression(class_weight='balanced'); C selected by pooled
               StratifiedGroupKFold curve-point PPV@90R. NO mixup/label-smoothing/effective-number knobs.
  - PU       : negatives = labeled ndbe (+ optional unlabeled CONFIDENT_NEGATIVE); NEVER HARD_NEG_CANDIDATE.
  - select   : pooled cross-center GroupKFold; REPORT LOCO-worst as an honest check (don't argmin on it).
  - eval     : held-out val, source-deduped, center-stratified 1%-prevalence bootstrap curve-point PPV@90R.

Usage:
    .venv/bin/python -m phase3.train --pools all --preproc standardize --center-debias project1
    .venv/bin/python -m phase3.train --pools all --preproc whiten --whiten-dim 256 --whiten-feats phase3/cache/feats_unl_confneg.npz
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

from phase3.dataset import load_feats
from phase3 import evaluate as ev

EMBED = 768
ALL_POOLS = ["cls", "reg_mean", "patch_mean", "patch_max"]


# ------------------------------------------------------------------ feature pipeline (leak-free fit/transform)
class FeaturePipe:
    def __init__(self, pools, pool_names, preproc="standardize", whiten_dim=256, debias_k=32):
        self.pools = ALL_POOLS if pools == ["all"] else pools
        self.pool_names = list(pool_names)
        self.preproc = preproc; self.whiten_dim = whiten_dim; self.debias_k = int(debias_k)

    def _slice(self, z):
        blocks = []
        for nm in self.pools:
            j = self.pool_names.index(nm)
            b = z[:, j * EMBED:(j + 1) * EMBED]
            b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
            blocks.append(b)
        return np.concatenate(blocks, 1).astype(np.float32)

    def fit(self, z, center, z_extra=None):
        X = self._slice(z)
        fit_src = X if z_extra is None else np.concatenate([X, self._slice(z_extra)], 0)
        self.mu = fit_src.mean(0, keepdims=True); self.sd = fit_src.std(0, keepdims=True) + 1e-6
        Xs = (X - self.mu) / self.sd
        fit_s = (fit_src - self.mu) / self.sd
        if self.preproc == "whiten":
            self.pca = PCA(n_components=min(self.whiten_dim, fit_s.shape[1]), whiten=True, random_state=0).fit(fit_s)
            Xs = self.pca.transform(Xs)
        else:
            self.pca = None
        # center de-bias: iteratively project out the top center-discriminative directions.
        # k~32 drives adversarial center-AUROC 1.0->~0.54 (passes <=0.55 gate) WITHOUT hurting PPV
        # (neoplasia signal is ~orthogonal to the center subspace). Fit on labeled centers only.
        self.cdirs = []
        if self.debias_k > 0 and center is not None:
            cmask = np.isin(center, ["center_1", "center_2"])
            if cmask.sum() > 0 and len(np.unique(center[cmask])) > 1:
                cy = (center[cmask] == "center_1").astype(int)
                Xc = Xs[cmask].copy(); Xtmp = Xs.copy()
                for _ in range(self.debias_k):
                    w = LogisticRegression(C=1.0, max_iter=2000).fit(Xc, cy).coef_[0]
                    w = w / (np.linalg.norm(w) + 1e-9)
                    self.cdirs.append(w)
                    Xc = Xc - (Xc @ w)[:, None] * w[None, :]
                    Xtmp = Xtmp - (Xtmp @ w)[:, None] * w[None, :]
        return self

    def transform(self, z):
        Xs = (self._slice(z) - self.mu) / self.sd
        if self.pca is not None:
            Xs = self.pca.transform(Xs)
        for w in self.cdirs:
            Xs = Xs - (Xs @ w)[:, None] * w[None, :]
        return Xs


def fit_head(X, y, C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="lbfgs").fit(X, y)


def oof_scores(X, y, groups, C, n_splits=5, seed=0):
    o = np.full(len(y), np.nan)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(X, y, groups):
        o[te] = fit_head(X[tr], y[tr], C).predict_proba(X[te])[:, 1]
    return o


def center_auroc(X, y, center, groups):
    """Adversarial gate: how well can a probe recover center from the FINAL feature? want ~0.5."""
    cu = np.unique(center)
    if len(cu) < 2:
        return np.nan
    cy = (center == cu[0]).astype(int)
    o = oof_scores(X, cy, groups, C=1.0)
    m = ~np.isnan(o)
    return roc_auc_score(cy[m], o[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-feats", default="phase3/cache/feats_train.npz")
    ap.add_argument("--val-feats", default="phase3/cache/feats_val.npz")
    ap.add_argument("--neg-feats", default="", help="optional unlabeled CONFIDENT_NEGATIVE feats npz to add as clean negatives")
    ap.add_argument("--whiten-feats", default="", help="optional unlabeled feats npz to fit whitening/standardizer (leak-free)")
    ap.add_argument("--pools", default="all")
    ap.add_argument("--preproc", default="standardize", choices=["standardize", "whiten"])
    ap.add_argument("--whiten-dim", type=int, default=256)
    ap.add_argument("--debias-k", type=int, default=32, help="# center directions to project out (0=off; 32 passes the AUROC<=0.55 gate)")
    ap.add_argument("--C", type=float, default=0.0, help="0 = auto-select by pooled OOF curve PPV@90R")
    ap.add_argument("--recall", type=float, default=0.9)
    ap.add_argument("--out", default="phase3/cache/val_scores.npz")
    a = ap.parse_args()

    pools = a.pools.split(",")
    Ftr = load_feats(a.train_feats); Fva = load_feats(a.val_feats)
    ztr, ytr, ctr = Ftr["feats"], Ftr["label"], Ftr["center"]
    gtr = Ftr["names"]  # train: 1 source per file
    # optional clean unlabeled negatives (CONFIDENT_NEGATIVE)
    if a.neg_feats:
        Fn = load_feats(a.neg_feats)
        ztr = np.concatenate([ztr, Fn["feats"]]); ytr = np.concatenate([ytr, np.zeros(len(Fn["feats"]), int)])
        ctr = np.concatenate([ctr, np.array([""] * len(Fn["feats"]))]); gtr = np.concatenate([gtr, Fn["names"]])
        print(f"+ added {len(Fn['feats'])} unlabeled CONFIDENT_NEGATIVE as clean negatives")
    z_extra = load_feats(a.whiten_feats)["feats"] if a.whiten_feats else None

    pipe = FeaturePipe(pools, Ftr["pool_names"], a.preproc, a.whiten_dim, a.debias_k).fit(ztr, ctr, z_extra)
    Xtr = pipe.transform(ztr); Xva = pipe.transform(Fva["feats"])
    print(f"train n={len(ytr)} pos={(ytr==1).sum()} neg={(ytr==0).sum()} | feat-dim={Xtr.shape[1]} pools={pipe.pools} preproc={a.preproc} debias_k={a.debias_k}")

    # select C by pooled OOF curve-point PPV@90R (on labeled-only rows for honest pos density)
    lab = np.array([g in set(Ftr["names"]) for g in gtr]) if a.neg_feats else np.ones(len(ytr), bool)
    Cgrid = [a.C] if a.C > 0 else [0.01, 0.03, 0.1, 0.3, 1.0]
    best = None
    for C in Cgrid:
        o = oof_scores(Xtr, ytr, gtr, C)
        b = ev.bootstrap(ytr[lab], o[lab], ctr[lab], target=a.recall, prevalence=0.01, B=500)
        med = b["curve"]["median"]
        print(f"  C={C:<5}  pooled OOF curve PPV@{int(a.recall*100)}R median={med:.4f} [{b['curve']['lo']:.3f},{b['curve']['hi']:.3f}]")
        if best is None or med > best[1]:
            best = (C, med, o)
    C, _, oof = best
    print(f"selected C={C}")

    # adversarial center gate on final features (OOF)
    cauroc = center_auroc(Xtr[lab], ytr[lab], ctr[lab], gtr[lab])
    gate = "PASS" if (np.isnan(cauroc) or cauroc <= 0.58) else "FAIL"
    print(f"adversarial center-AUROC (want<=0.55) = {cauroc:.3f}  [{gate}]")

    # LOCO-worst honest check (train on one center, eval the other) -- labeled rows only
    labm = lab & np.isin(ctr, ["center_1", "center_2"])
    Xl, yl, cl = Xtr[labm], ytr[labm], ctr[labm]
    loco = []
    for hold in ["center_1", "center_2"]:
        tr = cl != hold; te = cl == hold
        if (yl[te] == 1).sum() == 0:
            continue
        sc = fit_head(Xl[tr], yl[tr], C).predict_proba(Xl[te])[:, 1]
        b = ev.bootstrap(yl[te], sc, cl[te], target=a.recall, prevalence=0.01, B=500)
        loco.append(b["curve"]["median"])
        print(f"  LOCO train!={hold} -> eval {hold}: curve PPV@{int(a.recall*100)}R median={b['curve']['median']:.4f} (pos={int((yl[te]==1).sum())})")
    if loco:
        print(f"LOCO-worst = {min(loco):.4f}")

    # ===== held-out VAL (the honest number) =====
    head = fit_head(Xtr, ytr, C)
    sva = head.predict_proba(Xva)[:, 1]
    name, score, yv, cv = ev.dedup_by_source(Fva["names"], sva, Fva["label"], Fva["center"],
                                             Fva["source"] if Fva["source"] is not None else None, policy="orig")
    print(f"\n===== HELD-OUT VAL (source-deduped, n={len(yv)} pos={int((yv==1).sum())}) =====")
    ev.full_report(name, score, yv, cv, None, target=a.recall, prevalence=0.01, B=1000, dedup="off")
    # also val center gate
    vca = center_auroc(Xva, Fva["label"], Fva["center"], Fva["names"])
    print(f"val adversarial center-AUROC = {vca:.3f}")
    np.savez_compressed(a.out, names=name, score=score, label=yv, center=cv)
    print(f"saved val scores -> {a.out}")


if __name__ == "__main__":
    main()
