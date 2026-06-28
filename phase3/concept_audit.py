"""Data-grounding audit: is the 35-concept VLM supervision good enough to distill?"""
import os, sys, json, glob
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3.dataset import assemble, CONCEPT_ROLE

np.set_printoptions(suppress=True)

D = assemble("phase3/cache/feats_train.npz", "out/train/labels")
y = D["y"]
yb = (y == 1).astype(int)
C = D["c_value"]; R = D["c_trust"]; S = D["c_supervise"]
names = list(D["concept_names"])
center = D["center"]
has = D["has_concept"]
print(f"# train n={len(y)} pos={yb.sum()} neg={(yb==0).sum()} has_concept={has.mean()*100:.1f}%")
print(f"# centers: {dict(zip(*np.unique(center, return_counts=True)))}")
cmask = {c: center == c for c in np.unique(center)}

def auroc(score, lab):
    lab = np.asarray(lab)
    if len(np.unique(lab)) < 2:
        return np.nan
    try:
        return roc_auc_score(lab, score)
    except Exception:
        return np.nan

def signed_auc(score, lab):
    a = auroc(score, lab)
    return a

# center label for leakage test (center_2 vs center_1)
cen_bin = (center == "center_2").astype(int)

rows = []
for i, nm in enumerate(names):
    role = CONCEPT_ROLE[nm]
    v = C[:, i]
    a_all = auroc(v, yb)
    a_c1 = auroc(v[cmask["center_1"]], yb[cmask["center_1"]])
    a_c2 = auroc(v[cmask["center_2"]], yb[cmask["center_2"]])
    a_center = auroc(v, cen_bin)  # does the concept value separate centers?
    mtrust = R[:, i].mean()
    psup = S[:, i].mean() * 100
    # mean value by class
    mv_pos = v[yb == 1].mean(); mv_neg = v[yb == 0].mean()
    rows.append((nm, role, a_all, a_c1, a_c2, a_center, mtrust, psup, mv_pos, mv_neg))

# discriminative power: |AUC-0.5| folded
def fold(a):
    return abs(a - 0.5) if a == a else np.nan

print("\n## PER-CONCEPT (train)  |  AUC vs neo (all/c1/c2), AUC vs center, trust, %sup, meanval pos/neg")
hdr = f"{'concept':26s} {'role':13s} {'AUCall':>6s} {'AUCc1':>6s} {'AUCc2':>6s} {'AUCcen':>6s} {'trust':>5s} {'%sup':>5s} {'vPos':>5s} {'vNeg':>5s}"
print(hdr)
for r in sorted(rows, key=lambda x: -(fold(x[2]) if x[2]==x[2] else -1)):
    nm, role, a_all, a_c1, a_c2, a_center, mtrust, psup, mvp, mvn = r
    def f(x): return f"{x:6.3f}" if x==x else "   nan"
    print(f"{nm:26s} {role:13s} {f(a_all)} {f(a_c1)} {f(a_c2)} {f(a_center)} {mtrust:5.2f} {psup:5.1f} {mvp:5.2f} {mvn:5.2f}")

# Trust-weighted discriminability & consistency across centers
print("\n## CROSS-CENTER CONSISTENCY (AUC sign must agree c1 & c2 to be a generalizable signal)")
for r in sorted(rows, key=lambda x: -(fold(x[2]) if x[2]==x[2] else -1))[:18]:
    nm, role, a_all, a_c1, a_c2, a_center, mtrust, psup, mvp, mvn = r
    s1 = (a_c1-0.5) if a_c1==a_c1 else np.nan
    s2 = (a_c2-0.5) if a_c2==a_c2 else np.nan
    agree = "AGREE" if (s1==s1 and s2==s2 and np.sign(s1)==np.sign(s2)) else "DISAGREE/na"
    print(f"{nm:26s} d_c1={s1:+.3f} d_c2={s2:+.3f} {agree:12s} trust={mtrust:.2f} sup={psup:.0f}%")
