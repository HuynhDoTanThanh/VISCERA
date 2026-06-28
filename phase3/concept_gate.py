"""FAIR retention gate for the concept-supervised pretraining idea.

Compares, apples-to-apples (SAME 4-pool feature, SAME logistic head, SAME LOCO splits, paired bootstrap):
  - SSL backbone features  (dinov2.pth)
  - CONCEPT-pretrained backbone features (pretrain_concept.py -> concept_encoder.pt)
on the binary neo/ndbe task's NEW-CENTER proxy (leave-one-center-out PPV@90R).

The user's idea (train backbone on 170k x 35 concepts) is ACCEPTED iff the concept-encoder features
match-or-beat SSL LOCO-mean WITHOUT degrading the worst leg, by a paired bootstrap. This is the honest
test that updates the backbone (not a frozen proxy, not a 35-scalar bottleneck).

    .venv/bin/python -m phase3.concept_gate --ssl phase3/cache/feats_train.npz --concept phase3/cache/feats_train_concept.npz
"""
from __future__ import annotations
import argparse
import numpy as np

from phase3.dataset import load_feats
from phase3.experiment import loco_leg
from phase3 import evaluate as ev


def _loco_scores(z, y, c, g, pn, C):
    """Return held-out scores for both legs (concatenated), aligned to a fixed frame order."""
    parts = []
    for trc, tec in [("center_1", "center_2"), ("center_2", "center_1")]:
        med, sc, yte, cte = loco_leg(z, y, c, g, pn, ["all"], "standardize", 256, trc, tec, C,
                                     return_scores=True, B=400)
        parts.append((tec, sc, yte, cte, med))
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl", default="phase3/cache/feats_train.npz")
    ap.add_argument("--concept", default="phase3/cache/feats_train_concept.npz")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--recall", type=float, default=0.9)
    a = ap.parse_args()

    Fs, Fc = load_feats(a.ssl), load_feats(a.concept)
    assert list(Fs["names"]) == list(Fc["names"]), "SSL and concept feats must cover the SAME frames in the SAME order"
    y, c, g, pn = Fs["label"], Fs["center"], Fs["names"], Fs["pool_names"]

    print(f"=== FAIR GATE (LOCO PPV@{int(a.recall*100)}R, all-pool, logistic C={a.C}) ===")
    res = {}
    for name, z in [("SSL", Fs["feats"]), ("CONCEPT", Fc["feats"])]:
        legs = _loco_scores(z, y, c, g, pn, a.C)
        meds = [m for *_, m in legs]
        res[name] = legs
        print(f"  {name:8s} LOCO c1->c2={legs[0][4]:.3f}  c2->c1={legs[1][4]:.3f}  mean={np.mean(meds):.3f}  worst={min(meds):.3f}")

    # paired bootstrap per leg (same held-out frames -> aligned scores)
    print("\n  paired bootstrap (CONCEPT - SSL), per held-out center:")
    overall_ok = True
    for i, leg in enumerate(["center_2", "center_1"]):
        _, s_ssl, yte, cte, _ = res["SSL"][i]
        _, s_con, yte2, _, _ = res["CONCEPT"][i]
        pb = ev.paired_bootstrap(yte, s_con, s_ssl, cte, target=a.recall, prevalence=0.01, B=1000, mode="curve")
        print(f"    held-out {leg}: P(concept>ssl)={pb['p_gt0']:.3f}  medianΔ={pb['median_delta']:+.4f}  [{pb['lo']:+.3f},{pb['hi']:+.3f}]")
        if pb["median_delta"] < 0:
            overall_ok = False

    ssl_mean = np.mean([m for *_, m in res["SSL"]]); con_mean = np.mean([m for *_, m in res["CONCEPT"]])
    ssl_worst = min(m for *_, m in res["SSL"]); con_worst = min(m for *_, m in res["CONCEPT"])
    print("\n  VERDICT:")
    print(f"    concept mean {con_mean:.3f} vs SSL {ssl_mean:.3f} | concept worst {con_worst:.3f} vs SSL {ssl_worst:.3f}")
    if con_mean >= ssl_mean - 0.005 and con_worst >= ssl_worst - 0.005 and overall_ok:
        print("    => PASS: concept-pretraining matches-or-beats SSL. The idea earns its place; proceed to Stage-2.")
    else:
        print("    => FAIL: concept-pretraining does NOT beat SSL features on the new-center metric. Keep SSL; concepts -> interpretability/mining only.")


if __name__ == "__main__":
    main()
