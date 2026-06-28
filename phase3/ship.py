"""Ship the frozen-probe model — train the deployable scorer on ALL labeled data, save artifact.

Best frozen config (LOCO-selected): all-4-pool DINOv2 embedding, per-pool L2, standardize,
center-debias (project 32 center directions), strong logistic (C=1.0), small diverse ensemble,
Platt calibration. This is the offline --network=none scorer: image -> backbone -> pooled -> pipe ->
mean(member logistic probs) -> Platt -> score. Concepts/VLM are NOT used at test (training-only).

    .venv/bin/python -m phase3.ship --out phase3/cache/ship_model.pkl
"""
from __future__ import annotations
import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from phase3.dataset import load_feats
from phase3.train import FeaturePipe, fit_head
from phase3 import evaluate as ev

# diverse ensemble members: (pools, C) — averaged in probability space
MEMBERS = [(["all"], 1.0), (["all"], 0.3), (["cls", "reg_mean", "patch_mean", "patch_max"], 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-feats", default="phase3/cache/feats_train.npz")
    ap.add_argument("--debias-k", type=int, default=32)
    ap.add_argument("--out", default="phase3/cache/ship_model.pkl")
    a = ap.parse_args()

    F = load_feats(a.train_feats)
    z, y, c, g = F["feats"], F["label"], F["center"], F["names"]
    pn = list(F["pool_names"])

    members, oof_probs = [], []
    for pools, C in MEMBERS:
        pipe = FeaturePipe(pools, pn, "standardize", 256, a.debias_k).fit(z, c)
        X = pipe.transform(z)
        # OOF for calibration data
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
            oof[te] = fit_head(X[tr], y[tr], C).predict_proba(X[te])[:, 1]
        oof_probs.append(oof)
        head = fit_head(X, y, C)  # refit on all data for deployment
        members.append({"pools": pools, "C": C, "pipe": pipe, "head": head})

    mean_oof = np.mean(oof_probs, 0)
    # Platt calibration (1-D logistic on the averaged OOF prob) — rank-preserving, for score output
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(mean_oof.reshape(-1, 1), y)

    # honest LOCO read of the shipped ensemble (on OOF mean as a proxy)
    bs = ev.bootstrap(y, mean_oof, c, target=0.9, prevalence=0.01, B=1000)
    print(f"shipped ensemble pooled-OOF PPV@90R median={bs['curve']['median']:.4f} [{bs['curve']['lo']:.3f},{bs['curve']['hi']:.3f}]")

    artifact = {"members": members, "platt": platt, "pool_names": pn, "debias_k": a.debias_k,
                "config": "all-pools+std+debias32+logistic-ensemble+platt"}
    with open(a.out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"saved ship artifact -> {a.out}  ({len(members)} members)")


def score_features(artifact, z):
    """Score a (N,3072) pooled-feature matrix with the shipped ensemble. Returns calibrated scores."""
    probs = []
    for m in artifact["members"]:
        X = m["pipe"].transform(z)
        probs.append(m["head"].predict_proba(X)[:, 1])
    mean_p = np.mean(probs, 0)
    return artifact["platt"].predict_proba(mean_p.reshape(-1, 1))[:, 1]


if __name__ == "__main__":
    main()
