"""Phase-3 DG experiment runner — LOCO-primary (the confirmed NEW-CENTER target).

The hidden test is a held-out center, so the honest metric is leave-one-center-out:
train on one labeled center (+ optional diverse unlabeled negatives), evaluate the other.
Reports LOCO leg1/leg2/mean/worst (primary) + pooled same-center OOF (optimistic upper bound) +
the adversarial center-AUROC. All PPV@90R via the corrected curve-point 1%-prevalence bootstrap.

Use as the lever-ablation harness: each lever must raise LOCO-mean by a paired bootstrap, not pooled.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from phase3.train import FeaturePipe, fit_head
from phase3 import evaluate as ev

CENTERS = ["center_1", "center_2"]


def _bootstrap_curve(y, s, c, recall, B):
    return ev.bootstrap(y, s, c, target=recall, prevalence=0.01, B=B)["curve"]["median"]


def loco_leg(z, y, c, g, pool_names, pools, preproc, whiten_dim,
             train_center, eval_center, C, neg_z=None, recall=0.9, B=500, return_scores=False,
             pos_z=None, pos_center=None):
    """Train on train_center (+ optional unlabeled negatives + augmented positives), evaluate eval_center."""
    tr = c == train_center
    te = c == eval_center
    ztr, ytr, ctr, gtr = z[tr], y[tr], c[tr], g[tr]
    if neg_z is not None and len(neg_z):
        ztr = np.concatenate([ztr, neg_z]); ytr = np.concatenate([ytr, np.zeros(len(neg_z), int)])
        ctr = np.concatenate([ctr, np.array([""] * len(neg_z))]); gtr = np.concatenate([gtr, np.arange(len(neg_z)) + 10_000_000])
    if pos_z is not None and len(pos_z):
        m = pos_center == train_center  # only this center's augmented positives
        if m.sum():
            ztr = np.concatenate([ztr, pos_z[m]]); ytr = np.concatenate([ytr, np.ones(int(m.sum()), int)])
            ctr = np.concatenate([ctr, np.array([train_center] * int(m.sum()))]); gtr = np.concatenate([gtr, np.arange(int(m.sum())) + 20_000_000])
    # single training center -> no labeled center-debias possible; debias_k=0
    pipe = FeaturePipe(pools, pool_names, preproc, whiten_dim, debias_k=0).fit(ztr, ctr)
    Xtr, Xte = pipe.transform(ztr), pipe.transform(z[te])
    sc = fit_head(Xtr, ytr, C).predict_proba(Xte)[:, 1]
    med = _bootstrap_curve(y[te], sc, c[te], recall, B)
    if return_scores:
        return med, sc, y[te], c[te]
    return med


def pooled_oof(z, y, c, g, pool_names, pools, preproc, whiten_dim, debias_k, C, recall=0.9, B=500):
    pipe = FeaturePipe(pools, pool_names, preproc, whiten_dim, debias_k).fit(z, c)
    X = pipe.transform(z)
    o = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        o[te] = fit_head(X[tr], y[tr], C).predict_proba(X[te])[:, 1]
    med = _bootstrap_curve(y, o, c, recall, B)
    # adversarial center gate
    cy = (c == CENTERS[0]).astype(int)
    oc = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, cy, g):
        oc[te] = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], cy[tr]).predict_proba(X[te])[:, 1]
    m = ~np.isnan(oc)
    return med, roc_auc_score(cy[m], oc[m]), o


def run(z, y, c, g, pool_names, *, pools=("all",), preproc="standardize", whiten_dim=256,
        debias_k=32, C=0.1, neg_z=None, pos_z=None, pos_center=None, recall=0.9, B=500, title=""):
    pools = list(pools)
    leg12, s2, y2, c2 = loco_leg(z, y, c, g, pool_names, pools, preproc, whiten_dim, "center_1", "center_2", C, neg_z, recall, B, True, pos_z, pos_center)
    leg21, s1, y1, c1 = loco_leg(z, y, c, g, pool_names, pools, preproc, whiten_dim, "center_2", "center_1", C, neg_z, recall, B, True, pos_z, pos_center)
    lmean, lworst = (leg12 + leg21) / 2, min(leg12, leg21)
    pmed, cauroc, _ = pooled_oof(z, y, c, g, pool_names, pools, preproc, whiten_dim, debias_k, C, recall, B)
    print(f"{title:42s} LOCO[c1->c2={leg12:.3f} c2->c1={leg21:.3f}] mean={lmean:.3f} worst={lworst:.3f} | pooled={pmed:.3f} cAUROC={cauroc:.2f}")
    return dict(leg_c1c2=leg12, leg_c2c1=leg21, loco_mean=lmean, loco_worst=lworst, pooled=pmed, center_auroc=cauroc,
                loco_scores=(np.concatenate([s1, s2]), np.concatenate([y1, y2]), np.concatenate([c1, c2])))


if __name__ == "__main__":
    from phase3.dataset import load_feats
    F = load_feats("phase3/cache/feats_train.npz")
    z, y, c, g = F["feats"], F["label"], F["center"], F["names"]
    pn = F["pool_names"]
    print("=== C sweep on the NEW-CENTER (LOCO) target, all-pools standardize ===")
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        run(z, y, c, g, pn, pools=["all"], C=C, title=f"C={C}")
