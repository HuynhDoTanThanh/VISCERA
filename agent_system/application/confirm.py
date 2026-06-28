"""Label-confirmation gates — "can these concepts be trusted as foundation labels?"

Operates on aggregated readings (FrameAggregate) for a TRAIN split (fit) and a held-out VAL split.
Per discriminative concept it enforces four gates and emits PASS / REVIEW / FAIL:

  generalizes  : held-out val AUROC 95% bootstrap-CI lower bound > 0.50
  reliable     : inter-vote/model agreement >= 0.60 (on val)
  measurable   : assessable in >= 50% of frames (else it is mostly a mask, not a label)
  honest       : within-center label-AUROC >= 0.55 AND concept-predicts-center AUROC < 0.70
                 (center_2 has ~4x the neo prevalence of center_1, so a center-coupled concept
                 fakes discriminativeness on the pooled set while teaching a site shortcut)

Pure offline: consumes aggregates, calls no model.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from ..domain.concept_schema import BY_NAME, CONCEPTS
from ..domain.entities import Frame, FrameAggregate

AGREE_MIN = 0.60
ASSESS_MIN = 0.50
CENTER_LEAK_MAX = 0.70
WITHIN_AUROC_MIN = 0.55
CI_FLOOR = 0.50
BOOT = 2000
CENTER_POS = "center_2"


def _matrix(aggs: dict[str, FrameAggregate], frames: list[Frame]):
    names = [c.name for c in CONCEPTS]
    rows = [fr for fr in frames if fr.path in aggs]
    Xc = np.array([[aggs[fr.path].cells[n].value for n in names] for fr in rows])
    Xr = np.array([[aggs[fr.path].cells[n].reliability for n in names] for fr in rows])
    Xm = np.array([[aggs[fr.path].cells[n].mask for n in names] for fr in rows])
    y = np.array([fr.label for fr in rows])
    centers = np.array([fr.center for fr in rows])
    return Xc, Xr, Xm, y, centers, names


def _auroc(y, col):
    try:
        return float(roc_auc_score(y, col))
    except ValueError:
        return float("nan")


def _orient(y, col):
    a = _auroc(y, col)
    return (-col, True) if (not np.isnan(a) and a < 0.5) else (col, False)


def _boot_auroc(y, col, boot=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        a = _auroc(ys, col[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    v = np.asarray(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _within_center(y, col, centers):
    out = {}
    for c in sorted(set(centers)):
        m = centers == c
        yc = y[m]
        if m.sum() < 10 or yc.min() == yc.max():
            continue
        out[c] = _auroc(yc, col[m])
    return out


def _verdict(agree, assess, ci_low, within_min, center_leak):
    if np.isnan(ci_low) or ci_low <= CI_FLOOR:
        return "FAIL", [f"val AUROC CI-low {ci_low:.2f} <= {CI_FLOOR} (no held-out signal)"]
    reasons = []
    if agree < AGREE_MIN:
        reasons.append(f"agreement {agree:.2f} < {AGREE_MIN}")
    if assess < ASSESS_MIN:
        reasons.append(f"assessable {assess:.2f} < {ASSESS_MIN} (mostly a mask)")
    if not np.isnan(center_leak) and center_leak >= CENTER_LEAK_MAX:
        reasons.append(f"predicts center (AUROC {center_leak:.2f})")
    if not np.isnan(within_min) and within_min < WITHIN_AUROC_MIN:
        reasons.append(f"within-center AUROC {within_min:.2f} < {WITHIN_AUROC_MIN} (center confound)")
    return ("PASS" if not reasons else "REVIEW"), reasons


def confirm(train_aggs, train_frames, val_aggs, val_frames, boot=BOOT) -> dict:
    Xc_t, Xr_t, Xm_t, y_t, ctr_t, names = _matrix(train_aggs, train_frames)
    Xc_v, Xr_v, Xm_v, y_v, ctr_v, _ = _matrix(val_aggs, val_frames)
    center_bin = (ctr_t == CENTER_POS).astype(int)
    mi = mutual_info_classif(Xc_t, y_t, discrete_features=False, random_state=0)

    rows = []
    for j, n in enumerate(names):
        c = BY_NAME[n]
        _, flip = _orient(y_t, Xc_t[:, j])
        s = -1.0 if flip else 1.0
        tr_auroc = _auroc(y_t, s * Xc_t[:, j])
        va_mean, va_lo, va_hi = _boot_auroc(y_v, s * Xc_v[:, j], boot)
        agree = float(np.mean(Xr_v[:, j]))
        assess = float(np.mean(Xm_v[:, j]))
        leak = _auroc(center_bin, Xc_t[:, j])
        leak = max(leak, 1 - leak) if not np.isnan(leak) else leak
        within = _within_center(y_v, s * Xc_v[:, j], ctr_v)
        within_min = min(within.values()) if within else float("nan")
        status, why = ("n/a", [])
        if c.role == "discriminative":
            status, why = _verdict(agree, assess, va_lo, within_min, leak)
        rows.append({"concept": n, "tier": c.tier, "role": c.role, "agree": agree,
                     "assess": assess, "mi_train": float(mi[j]), "auroc_train": tr_auroc,
                     "auroc_val": va_mean, "auroc_val_ci": [va_lo, va_hi],
                     "within_center_min": within_min, "center_leak": leak,
                     "status": status, "reasons": why})
    rows.sort(key=lambda r: (-r["mi_train"], -r["auroc_train"]))
    disc = [r for r in rows if r["role"] == "discriminative"]
    return {"n_train": int(len(y_t)), "n_val": int(len(y_v)),
            "pos_train": int(y_t.sum()), "pos_val": int(y_v.sum()), "rows": rows,
            "confirmed_core": [r["concept"] for r in disc if r["status"] == "PASS"],
            "review": [r["concept"] for r in disc if r["status"] == "REVIEW"],
            "failed": [r["concept"] for r in disc if r["status"] == "FAIL"]}


def format_report(rep: dict) -> str:
    L = ["# Label Confirmation — usability as foundation supervision\n",
         f"- train {rep['n_train']} ({rep['pos_train']} neo) · val held-out {rep['n_val']} "
         f"({rep['pos_val']} neo)",
         f"- gates: agree>={AGREE_MIN}, assess>={ASSESS_MIN}, val-AUROC-CI-low>{CI_FLOOR}, "
         f"within-center>={WITHIN_AUROC_MIN}, center-leak<{CENTER_LEAK_MAX}\n",
         f"{'concept':22}{'tier':5}{'role':14}{'agree':>6}{'assess':>7}{'AUROCval[CI]':>20}"
         f"{'within':>7}{'cLeak':>6}  verdict",
         "-" * 100]
    for r in rep["rows"]:
        ci = r["auroc_val_ci"]
        cis = f"{r['auroc_val']:.2f}[{ci[0]:.2f},{ci[1]:.2f}]" if not np.isnan(r["auroc_val"]) else "n/a"
        wm = f"{r['within_center_min']:.2f}" if not np.isnan(r["within_center_min"]) else "-"
        cl = f"{r['center_leak']:.2f}" if not np.isnan(r["center_leak"]) else "-"
        v = r["status"] if r["role"] == "discriminative" else ""
        L.append(f"{r['concept']:22}{r['tier']:5}{r['role']:14}{r['agree']:>6.2f}{r['assess']:>7.2f}"
                 f"{cis:>20}{wm:>7}{cl:>6}  {v}")
    L.append(f"\n## CONFIRMED core ({len(rep['confirmed_core'])}): {', '.join(rep['confirmed_core'])}")
    if rep["review"]:
        L.append(f"## REVIEW ({len(rep['review'])}): {', '.join(rep['review'])}")
    if rep["failed"]:
        L.append(f"## FAILED ({len(rep['failed'])}): {', '.join(rep['failed'])}")
    return "\n".join(L)
