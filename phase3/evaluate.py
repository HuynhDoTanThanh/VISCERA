"""Phase-3 evaluation harness — the honest local proxy for the RARE26 leaderboard (PPV@90R).

Hardened per adversarial review. Computes, at ~1% prevalence over center-stratified bootstrap resamples:
  - CURVE-POINT PPV@R   : precision at the threshold whose recall first reaches R, recomputed per
                          resample. Mirrors how a PR-curve leaderboard reads PPV@90R. PRIMARY.
  - FIXED-THRESHOLD PPV : precision at a threshold CHOSEN ON HELD-OUT data and applied unchanged to
                          each resample. The deployable number a --network=none container can hit.
  - oracle max-precision: optimistic upper bound (the old eval_metrics.py number) — diagnostic only.
Acceptance of a lever is a PAIRED bootstrap on SHARED resamples (P(delta>0) and median delta), NOT
independent-CI separation (impossible at ~48 positives). Source-level de-aug with a symmetric aug
policy across classes. LOCO honest read. OOD negative stress test helper.

At fixed recall R, prevalence pi:  precision = R*pi / (R*pi + FPR*(1-pi)).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# ----------------------------------------------------------------- metric primitives
def _recall_threshold(y, s, target):
    """Highest score threshold t such that recall(>=t) >= target, on THIS set. Returns (thr, precision)."""
    P = int((y == 1).sum())
    if P == 0:
        return np.nan, np.nan
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    recall = tp / P
    k = np.searchsorted(recall, target, side="left")  # first index with recall>=target
    if k >= len(recall):
        return np.nan, np.nan
    prec = tp[k] / max(tp[k] + fp[k], 1)
    return float(s[order][k]), float(prec)


def ppv_curvepoint(y, s, target=0.9):
    """Precision at the threshold whose recall first reaches `target` (leaderboard-mirror PPV@R)."""
    _, prec = _recall_threshold(y, s, target)
    return prec


def ppv_fixed(y, s, thr, target=0.9):
    """Operating-point precision at a FIXED threshold; also returns realized recall (diagnostic)."""
    pred = s >= thr
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return prec, rec


def ppv_oracle(y, s, target=0.9):
    """Optimistic: max precision over all thresholds with recall>=target (the old number)."""
    P = int((y == 1).sum())
    if P == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort"); ys = y[order]
    tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
    recall = tp / P; precision = tp / np.maximum(tp + fp, 1)
    m = recall >= target
    return float(precision[m].max()) if m.any() else np.nan


def fpr_at_recall(y, s, target=0.9):
    thr, _ = _recall_threshold(y, s, target)
    if np.isnan(thr):
        return np.nan
    neg = y == 0
    return float((s[neg] >= thr).sum() / max(neg.sum(), 1))


# ----------------------------------------------------------------- threshold-free ranking metrics
def auc_metrics(y, s):
    """AUROC + AUPRC — threshold-free, STABLE even at few positives (unlike PPV@90R). AUROC is
    prevalence-independent; AUPRC is at the set's NATURAL prevalence (not reweighted to 1%). Both measure
    RANKING quality: a high AUROC can coexist with a near-floor PPV@90R@1% — trust them for stability, not
    as the operating-point score."""
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return dict(auroc=float("nan"), auprc=float("nan"))
    return dict(auroc=float(roc_auc_score(y, s)), auprc=float(average_precision_score(y, s)))


def report_full(y, s, center=None, target=0.9, prevalence=0.01, B=2000, seed=12345):
    """The 5 trusted numbers in one call: PPV@{target}R (bootstrap median) + 95% CI [lo,hi] at `prevalence`,
    plus AUROC + AUPRC. Returns a flat dict; callers just format it."""
    b = bootstrap(y, s, center, target=target, prevalence=prevalence, B=B, seed=seed)["curve"]
    a = auc_metrics(y, s)
    return dict(ppv90=b.get("median", float("nan")), ci_lo=b.get("lo", float("nan")), ci_hi=b.get("hi", float("nan")),
                auroc=a["auroc"], auprc=a["auprc"], n=int(len(y)), pos=int((np.asarray(y) == 1).sum()))


# ----------------------------------------------------------------- resampling
def _resample_idx(rng, pos, neg_by_center, npos, prevalence, center_mix):
    """Indices for one resample: npos positives + negatives to hit prevalence, center-stratified."""
    nneg = int(round(npos * (1 - prevalence) / prevalence))
    pi = rng.choice(pos, npos, replace=True)
    chunks = [pi]
    for ce, frac in center_mix.items():
        nc = max(int(round(nneg * frac)), 0)
        pool = neg_by_center.get(ce)
        if pool is not None and len(pool) and nc:
            chunks.append(rng.choice(pool, nc, replace=True))
    return np.concatenate(chunks)


def _center_mix(center, y):
    neg = y == 0
    cs, counts = np.unique(center[neg], return_counts=True)
    tot = counts.sum()
    return {c: cnt / tot for c, cnt in zip(cs, counts)}


def bootstrap(y, s, center=None, target=0.9, prevalence=0.01, B=1000, seed=12345,
              fixed_thr=None, center_mix=None):
    """Bootstrap median PPV@target. Returns curve-point + (optional) fixed-threshold + oracle medians."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        nan = dict(median=np.nan, lo=np.nan, hi=np.nan, mean=np.nan)
        return dict(curve=nan, fixed=nan, oracle=nan, fpr_median=np.nan,
                    n_pos=len(pos), n_neg=len(neg), target=target, prevalence=prevalence, B=0)
    if center is None:
        center = np.zeros(len(y), dtype=int)
    neg_by_center = {c: neg[center[neg] == c] for c in np.unique(center)}
    mix = center_mix or _center_mix(center, y)
    npos = len(pos)
    cp = np.empty(B); fx = np.empty(B); orc = np.empty(B); fpr = np.empty(B)
    for b in range(B):
        idx = _resample_idx(rng, pos, neg_by_center, npos, prevalence, mix)
        yb, sb = y[idx], s[idx]
        cp[b] = ppv_curvepoint(yb, sb, target)
        orc[b] = ppv_oracle(yb, sb, target)
        fpr[b] = fpr_at_recall(yb, sb, target)
        fx[b] = ppv_fixed(yb, sb, fixed_thr, target)[0] if fixed_thr is not None else np.nan
    def msummary(a):
        a = a[~np.isnan(a)]
        return dict(median=float(np.median(a)), lo=float(np.percentile(a, 2.5)),
                    hi=float(np.percentile(a, 97.5)), mean=float(np.mean(a))) if len(a) else dict(median=np.nan)
    return dict(curve=msummary(cp), fixed=msummary(fx), oracle=msummary(orc),
                fpr_median=float(np.nanmedian(fpr)), n_pos=npos, n_neg=int(round(npos*(1-prevalence)/prevalence)),
                target=target, prevalence=prevalence, B=B,
                _curve_samples=cp, _fixed_samples=fx)


def _metric_on(yb, sb, target=0.9, mode="curve", thr=None):
    """One metric value on one resample, dispatched by mode: curve=PPV@target(curve-point), fixed=PPV@thr,
    auroc / auprc = threshold-free ranking (nan if a resample is single-class)."""
    if mode == "curve":
        return ppv_curvepoint(yb, sb, target)
    if mode == "fixed":
        return ppv_fixed(yb, sb, thr, target)[0]
    if len(np.unique(yb)) < 2:
        return np.nan
    if mode == "auroc":
        return roc_auc_score(yb, sb)
    if mode == "auprc":
        return average_precision_score(yb, sb)
    raise ValueError(f"unknown mode {mode}")


def paired_bootstrap(y, sA, sB, center=None, target=0.9, prevalence=0.01, B=1000, seed=12345,
                     mode="curve", thrA=None, thrB=None):
    """Δ = metric(A) - metric(B) on SHARED resamples. Returns P(Δ>0), median Δ, and the 95% Δ CI. The
    lever-acceptance test. mode = curve | fixed | auroc | auprc (use auroc/auprc to accept levers — stable;
    curve/PPV@90R is a noisy tie-breaker only)."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if center is None:
        center = np.zeros(len(y), dtype=int)
    neg_by_center = {c: neg[center[neg] == c] for c in np.unique(center)}
    mix = _center_mix(center, y); npos = len(pos)
    d = np.empty(B)
    for b in range(B):
        idx = _resample_idx(rng, pos, neg_by_center, npos, prevalence, mix)
        yb = y[idx]
        d[b] = _metric_on(yb, sA[idx], target, mode, thrA) - _metric_on(yb, sB[idx], target, mode, thrB)
    d = d[~np.isnan(d)]
    return dict(p_gt0=float((d > 0).mean()), median_delta=float(np.median(d)),
                mean_delta=float(np.mean(d)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), B=len(d))


# ----------------------------------------------------------------- acceptance gate / noise floor (contribution C)
def gate(y, sA, sB, center=None, metric="auroc", target=0.9, prevalence=0.01, B=2000, seed=12345):
    """THE lever-acceptance gate. Is A reliably better than B on `metric`? Δ = metric(A) - metric(B) on shared
    resamples; PASS = 95% Δ CI clear ABOVE 0, FAIL = clear below, else INCONCLUSIVE (= noise, do not ship).
    Accept levers on metric='auroc'/'auprc' (stable at few positives); 'curve' (PPV@90R) is a tie-breaker only."""
    r = paired_bootstrap(y, sA, sB, center, target, prevalence, B, seed, mode=metric)
    verdict = "PASS" if r["lo"] > 0 else ("FAIL" if r["hi"] < 0 else "INCONCLUSIVE")
    return dict(metric=metric, delta=r["median_delta"], lo=r["lo"], hi=r["hi"], p_gt0=r["p_gt0"],
                verdict=verdict, B=r["B"])


def mde(y, s, center=None, metric="curve", target=0.9, prevalence=0.01, B=2000, seed=12345):
    """Minimum Detectable Effect = the noise floor of `metric` on THIS set. Returns the metric's bootstrap SD
    and MDE=2*SD: a change smaller than MDE is NOT MEASURABLE here (fall back to AUROC/AUPRC). This is the
    quantitative reason PPV@90R@1% at ~127 positives cannot rank configs — its MDE dwarfs the contest margins."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if center is None:
        center = np.zeros(len(y), dtype=int)
    neg_by_center = {c: neg[center[neg] == c] for c in np.unique(center)}
    mix = _center_mix(center, y); npos = len(pos)
    vals = np.empty(B)
    for b in range(B):
        idx = _resample_idx(rng, pos, neg_by_center, npos, prevalence, mix)
        vals[b] = _metric_on(y[idx], s[idx], target, metric)
    vals = vals[~np.isnan(vals)]
    sd = float(np.std(vals))
    return dict(metric=metric, median=float(np.median(vals)), sd=sd, mde=2 * sd, B=len(vals))


def seed_summary(y, score_list, metric="auroc", target=0.9):
    """Seed-averaged metric: given a list of per-seed score arrays for the SAME y, report mean±sd across seeds
    ('a win must not be one lucky init'). Point metric per seed (not bootstrap)."""
    vals = [_metric_on(np.asarray(y), np.asarray(s), target, metric) for s in score_list]
    vals = np.array([v for v in vals if not np.isnan(v)])
    return dict(metric=metric, mean=float(np.mean(vals)), sd=float(np.std(vals)), n_seeds=int(len(vals)),
                vals=[float(v) for v in vals])


# ----------------------------------------------------------------- dedup / loaders
def dedup_by_source(name, score, label, center, source=None, policy="mean"):
    """One row per SOURCE frame. policy: mean | max | orig(first). Symmetric across classes (no leakage)."""
    key = source if source is not None else name
    groups = {}
    for i, k in enumerate(key):
        groups.setdefault(k, []).append(i)
    agg = {"mean": np.mean, "max": np.max}.get(policy)
    N, S, Y, C = [], [], [], []
    for k, idxs in groups.items():
        idxs = np.array(idxs)
        v = float(agg(score[idxs])) if agg else float(score[idxs[0]])
        N.append(str(k)); S.append(v); Y.append(int(label[idxs][0])); C.append(str(center[idxs][0]))
    return np.array(N), np.array(S, float), np.array(Y, int), np.array(C)


def load_label_dir(dirpath):
    name, s, y, c = [], [], [], []
    for fp in glob.glob(os.path.join(dirpath, "*.json")):
        d = json.load(open(fp))
        if int(d.get("label", -1)) < 0:
            continue
        name.append(d["name"]); s.append(float(d["suspicion"])); y.append(int(d["label"])); c.append(d.get("center", ""))
    return np.array(name), np.array(s, float), np.array(y, int), np.array(c)


def load_scores(path):
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        src = d["source"] if "source" in d.files else None
        return d["names"], d["score"].astype(float), d["label"].astype(int), d["center"].astype(str), src
    import csv
    rows = list(csv.DictReader(open(path)))
    g = lambda k, d="": np.array([r.get(k, d) for r in rows])
    return (g("name"), g("score").astype(float), g("label").astype(int), g("center"),
            g("source") if rows and "source" in rows[0] else None)


# ----------------------------------------------------------------- report
def full_report(name, s, y, c, source, target=0.9, prevalence=0.01, B=1000, dedup="orig", held_thr=None):
    if dedup and dedup != "off":
        name, s, y, c = dedup_by_source(name, s, y, c, source, policy=dedup)
    P, N = int((y == 1).sum()), int((y == 0).sum())
    print(f"n_sources={len(y)}  pos={P}  neg={N}  dedup={dedup}")
    cp = ppv_curvepoint(y, s, target); orc = ppv_oracle(y, s, target); fp = fpr_at_recall(y, s, target)
    print(f"[pooled]  curve PPV@{int(target*100)}R={cp:.4f}  oracle={orc:.4f}  FPR={fp:.4f}")
    bs = bootstrap(y, s, c, target, prevalence, B, fixed_thr=held_thr)
    print(f"[bootstrap x{B} @ {prevalence:.0%}]  curve median={bs['curve']['median']:.4f} "
          f"[{bs['curve']['lo']:.4f},{bs['curve']['hi']:.4f}]  oracle median={bs['oracle']['median']:.4f}  medFPR={bs['fpr_median']:.4f}")
    if held_thr is not None:
        print(f"                                 fixed-thr({held_thr:.4f}) median={bs['fixed']['median']:.4f} [{bs['fixed']['lo']:.4f},{bs['fixed']['hi']:.4f}]")
    centers = sorted(set(c.tolist()))
    if len(centers) > 1:
        print("[per-center / LOCO honest read]")
        for ce in centers:
            m = c == ce
            if (y[m] == 1).sum() == 0:
                continue
            b = bootstrap(y[m], s[m], c[m], target, prevalence, B)
            print(f"  {ce:10s} pos={int((y[m]==1).sum()):3d} neg={int((y[m]==0).sum()):4d}  curve median={b['curve']['median']:.4f}")
    return bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir"); ap.add_argument("--scores")
    ap.add_argument("--recall", type=float, default=0.9)
    ap.add_argument("--prevalence", type=float, default=0.01)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--dedup", default="orig", choices=["off", "orig", "mean", "max"])
    a = ap.parse_args()
    if a.labels_dir:
        name, s, y, c = load_label_dir(a.labels_dir); source = None
    else:
        name, s, y, c, source = load_scores(a.scores)
    full_report(name, s, y, c, source, a.recall, a.prevalence, a.bootstrap, a.dedup)


if __name__ == "__main__":
    main()
