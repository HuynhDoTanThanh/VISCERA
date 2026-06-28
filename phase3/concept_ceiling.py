"""Multi-concept ceiling: logistic on concept VALUES (train), AUROC + cross-center PPV@90R transfer.
Compares: all 35 concepts / trusted-discriminative subset / +trust-weighting, vs the embedding baseline note."""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3.dataset import assemble, CONCEPT_ROLE

D = assemble("phase3/cache/feats_train.npz", "out/train/labels")
yb = (D["y"] == 1).astype(int)
C = D["c_value"]; R = D["c_trust"]; names = list(D["concept_names"]); center = D["center"]
c1 = center=="center_1"; c2 = center=="center_2"

def ppv_transfer(score_tr, lab_tr, score_te, lab_te, recall=0.90):
    P=lab_tr.sum()
    if P==0 or lab_te.sum()==0: return np.nan
    order=np.argsort(-score_tr); tp=np.cumsum(lab_tr[order]); rec=tp/P
    idx=min(np.searchsorted(rec,recall), len(score_tr)-1)
    thr=np.sort(score_tr)[::-1][idx]
    pred=score_te>=thr
    return lab_te[pred].sum()/pred.sum() if pred.sum()>0 else np.nan

def ppv_pooled(score, lab, recall=0.90):
    P=lab.sum(); order=np.argsort(-score); tp=np.cumsum(lab[order]); rec=tp/P
    idx=min(np.searchsorted(rec,recall), len(score)-1); thr=np.sort(score)[::-1][idx]
    pred=score>=thr
    return lab[pred].sum()/pred.sum() if pred.sum()>0 else np.nan

# concept subsets
TRUSTED_DISC = ["mucosal_irregularity","focal_erythema","demarcation","nodularity","lesion_present",
                "color_change_locality","surface_effacement","colocalization","whitish_focal_area",
                "color_heterogeneity","mucosal_pattern_type"]
idx_all = list(range(len(names)))
idx_disc = [names.index(n) for n in TRUSTED_DISC]

def fit_eval(Xcols, tag):
    X = C[:, Xcols]
    # within-center 5-fold-ish: just report cross-center transfer + pooled CV-free (train=test upper bound)
    # cross-center: fit on c1, eval c2 ; fit on c2, eval c1
    def fit(tr, te):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X[tr], yb[tr])
        s = clf.predict_proba(X[te])[:,1]
        return s
    s12 = fit(c1, c2); s21 = fit(c2, c1)
    auc12 = roc_auc_score(yb[c2], s12); auc21 = roc_auc_score(yb[c1], s21)
    p12 = ppv_transfer(fit(c1,c1), yb[c1], s12, yb[c2])  # thr from c1-train scores
    # better: fit on c1, get c1 scores for threshold, apply model+thr to c2
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[c1], yb[c1])
    s_c1 = clf.predict_proba(X[c1])[:,1]; s_c2 = clf.predict_proba(X[c2])[:,1]
    p_12 = ppv_transfer(s_c1, yb[c1], s_c2, yb[c2])
    clf2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[c2], yb[c2])
    s2_c2 = clf2.predict_proba(X[c2])[:,1]; s2_c1 = clf2.predict_proba(X[c1])[:,1]
    p_21 = ppv_transfer(s2_c2, yb[c2], s2_c1, yb[c1])
    print(f"{tag:30s} AUROC c1->c2={auc12:.3f} c2->c1={auc21:.3f} | PPV@90R c1->c2={p_12:.3f} c2->c1={p_21:.3f}")

print(f"# prevalence c1={yb[c1].mean():.3f} c2={yb[c2].mean():.3f} | LOCO-worst wall reference ~0.04")
fit_eval(idx_all, "all 35 concept values")
fit_eval(idx_disc, "11 trusted-discriminative")
# trust-weighted features
Cw = C * R
def fit_eval_w(Xcols, tag):
    global C
    save=C; C=Cw; fit_eval(Xcols, tag); C=save
fit_eval_w(idx_disc, "11 trusted-disc x trust")
