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
    # Power indicator: PPV@90R on a held-out center is estimated from THAT center's positives only, then
    # reweighted to 1% prevalence. Few positives -> wide CI -> the gate cannot resolve small deltas. Print it
    # up front so a lucky/unlucky point estimate is never read as a real capability difference.
    npos = {tec: int(((c == tec) & (y == 1)).sum()) for tec in ("center_2", "center_1")}
    print(f"  [power] held-out positives: center_2={npos['center_2']}  center_1={npos['center_1']}"
          f"  — PPV@90R CI width scales ~1/sqrt(pos); treat legs with <~50 pos as low-power")

    res = {}  # name -> (legs, cis) ; legs[i]=(tec,sc,yte,cte,med), cis[i]=curve summary {median,lo,hi}
    for name, zf in [("SSL", Fs["feats"]), ("CONCEPT", Fc["feats"])]:
        legs = _loco_scores(zf, y, c, g, pn, a.C)
        cis = [ev.bootstrap(yte, sc, cte, target=a.recall, prevalence=0.01, B=1000)["curve"]
               for _, sc, yte, cte, _ in legs]
        res[name] = (legs, cis)
        meds = [ci["median"] for ci in cis]
        print(f"  {name:8s} c1->c2={cis[0]['median']:.3f}[{cis[0]['lo']:.3f},{cis[0]['hi']:.3f}]"
              f"  c2->c1={cis[1]['median']:.3f}[{cis[1]['lo']:.3f},{cis[1]['hi']:.3f}]"
              f"  mean={np.mean(meds):.3f}  worst={min(meds):.3f}")

    # paired bootstrap per leg (same held-out frames -> aligned scores). The delta CI crossing 0 is the
    # honest "cannot distinguish" signal — more reliable than comparing two noisy point estimates.
    print("\n  paired bootstrap (CONCEPT - SSL), per held-out center:")
    deltas = {}
    for i, leg in enumerate(["center_2", "center_1"]):
        _, s_ssl, yte, cte, _ = res["SSL"][0][i]
        _, s_con, _, _, _ = res["CONCEPT"][0][i]
        pb = ev.paired_bootstrap(yte, s_con, s_ssl, cte, target=a.recall, prevalence=0.01, B=1000, mode="curve")
        deltas[leg] = pb
        crosses = pb["lo"] <= 0 <= pb["hi"]  # inclusive: a CI touching 0 (or a dead tie) is not significant
        tag = "not distinguishable (CI includes 0)" if crosses else ("CONCEPT better" if pb["median_delta"] > 0 else "CONCEPT worse")
        print(f"    held-out {leg}: P(concept>ssl)={pb['p_gt0']:.3f}  medianΔ={pb['median_delta']:+.4f}"
              f"  [{pb['lo']:+.3f},{pb['hi']:+.3f}]  -> {tag}")

    ssl_meds = [ci["median"] for ci in res["SSL"][1]]; con_meds = [ci["median"] for ci in res["CONCEPT"][1]]
    ssl_mean, con_mean = float(np.mean(ssl_meds)), float(np.mean(con_meds))
    ssl_worst, con_worst = float(min(ssl_meds)), float(min(con_meds))
    both_inconclusive = all(d["lo"] <= 0 <= d["hi"] for d in deltas.values())
    concept_wins = (con_mean >= ssl_mean - 0.005 and con_worst >= ssl_worst - 0.005
                    and all(d["median_delta"] >= 0 for d in deltas.values()))
    print("\n  VERDICT:")
    print(f"    concept mean {con_mean:.3f} vs SSL {ssl_mean:.3f} | concept worst {con_worst:.3f} vs SSL {ssl_worst:.3f}")
    if concept_wins:
        print("    => PASS: concept-pretraining matches-or-beats SSL. The idea earns its place; proceed to Stage-2.")
    elif both_inconclusive:
        print("    => INCONCLUSIVE: both LOCO legs' paired Δ CIs cross 0 — at this positive count the gate")
        print("       cannot separate concept from SSL. Do NOT kill the idea on this run: train Stage-1 to")
        print("       convergence and/or add labeled positives, then re-gate.")
    else:
        print("    => FAIL: concept-pretraining does NOT beat SSL on the new-center metric (paired Δ<0 with")
        print("       CI clear of 0 on a decisive leg). Keep SSL; concepts -> interpretability/mining only.")


if __name__ == "__main__":
    main()
