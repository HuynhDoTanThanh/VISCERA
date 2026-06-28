"""PPV@90R per concept + cross-center threshold transfer (the metric that actually matters)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3.dataset import assemble, CONCEPT_ROLE

D = assemble("phase3/cache/feats_train.npz", "out/train/labels")
yb = (D["y"] == 1).astype(int)
C = D["c_value"]; names = list(D["concept_names"]); center = D["center"]
c1 = center == "center_1"; c2 = center == "center_2"

def ppv_at_recall(score, lab, recall=0.90):
    """precision at the threshold achieving >= target recall (pooled)."""
    lab = np.asarray(lab); score = np.asarray(score)
    P = lab.sum()
    if P == 0: return np.nan, np.nan
    order = np.argsort(-score)
    s_sorted = score[order]; l_sorted = lab[order]
    tp = np.cumsum(l_sorted)
    recall_curve = tp / P
    # smallest k reaching target recall
    idx = np.searchsorted(recall_curve, recall)
    if idx >= len(score): idx = len(score)-1
    thr = s_sorted[idx]
    pred = score >= thr
    if pred.sum() == 0: return np.nan, thr
    return lab[pred].sum() / pred.sum(), thr

def transfer_ppv(score_tr, lab_tr, score_te, lab_te, recall=0.90):
    """Fit recall-90 threshold on train-center, apply to test-center; report PPV on test-center."""
    P = lab_tr.sum()
    if P == 0 or lab_te.sum()==0: return np.nan
    order = np.argsort(-score_tr)
    tp = np.cumsum(lab_tr[order]); rec = tp/P
    idx = np.searchsorted(rec, recall); idx = min(idx, len(score_tr)-1)
    thr = np.sort(score_tr)[::-1][idx]
    pred = score_te >= thr
    if pred.sum()==0: return np.nan
    return lab_te[pred].sum()/pred.sum()

prev_all = yb.mean(); prev1 = yb[c1].mean(); prev2 = yb[c2].mean()
print(f"# prevalence: all={prev_all:.3f}  c1={prev1:.3f}  c2={prev2:.3f}  (TEST is ~0.01)")
print(f"# PPV@90R pooled, and CROSS-CENTER transfer (thr fit on one center, applied to other)")
print(f"{'concept':26s} {'role':13s} {'PPV90all':>8s} {'PPV90c1':>8s} {'PPV90c2':>8s} {'c1->c2':>7s} {'c2->c1':>7s}")
res=[]
for i, nm in enumerate(names):
    v = C[:, i]
    p_all,_ = ppv_at_recall(v, yb)
    p_c1,_ = ppv_at_recall(v[c1], yb[c1])
    p_c2,_ = ppv_at_recall(v[c2], yb[c2])
    t12 = transfer_ppv(v[c1], yb[c1], v[c2], yb[c2])
    t21 = transfer_ppv(v[c2], yb[c2], v[c1], yb[c1])
    res.append((nm, CONCEPT_ROLE[nm], p_all, p_c1, p_c2, t12, t21))
def f(x): return f"{x:8.3f}" if x==x else "     nan"
for r in sorted(res, key=lambda x: -(x[5] if x[5]==x[5] else -1) - (x[6] if x[6]==x[6] else 0)):
    nm, role, p_all, p_c1, p_c2, t12, t21 = r
    print(f"{nm:26s} {role:13s} {f(p_all)} {f(p_c1)} {f(p_c2)} {f(t12)[:7]:>7s} {f(t21)[:7]:>7s}")

# Combined trust-weighted discriminative score (simple sum of top concepts) as a sanity ceiling
from phase3.dataset import DISCRIMINATIVE  # not exported; fallback
