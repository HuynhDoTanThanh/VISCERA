"""CRISP — Concept-Residualized Invariant-operating-point Scoring (§20). RANK-INDEPENDENT mechanism test.

Deploy path (faithful): nuisance-concept heads predict n̂(x) from FROZEN GastroNet-DINOv2 features
(trained on the 2476 labeled train frames that have VLM concepts); the detector logit s(x) is
FWL-residualized against n̂ using μ₀,σ₀ fit on SOURCE NEGATIVES only:  r = (s − μ₀(n̂)) / σ₀(n̂).

Headline claim (falsifiable WITHOUT the noisy PPV): residualizing REDUCES the cross-center NEGATIVE-score
drift (KS distance between source-neg and target-neg score distributions) while PRESERVING recall when the
source 90%-recall threshold is transferred to the target center. Measured on both LOCO legs on the held-out val.
"""
import numpy as np, os
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
ROOT="/Volumes/Shin/RARE2026"; C=f"{ROOT}/phase3/cache"
NUIS=["modality","magnification","distance","view","landmark","interpretable_fraction",
      "blur","glare","exposure","mucus_bubbles","debris","black_border","overlay_graphics","dominant_color"]

ft=np.load(f"{C}/feats_train.npz",allow_pickle=True); ct=np.load(f"{C}/concept_targets.npz",allow_pickle=True); fv=np.load(f"{C}/feats_val.npz",allow_pickle=True)
cn=[str(x) for x in ct["concept_names"]]; nidx=[cn.index(n) for n in NUIS]
# align concepts to feats_train by name
cmap={str(n):i for i,n in enumerate(ct["names"])}
tr_names=[str(x) for x in ft["names"]]; ci=np.array([cmap[n] for n in tr_names])
Xtr=ft["feats"].astype(np.float32); ytr=ft["label"].astype(int); ctr=ft["center"]; Ntr=ct["value"][ci][:,nidx]   # (2476,14) nuisance concept values
o=fv["aug"]=="orig"; Xva=fv["feats"][o].astype(np.float32); yva=fv["label"][o].astype(int); cva=fv["center"][o]
print(f"train {Xtr.shape} pos={ytr.sum()} | val {Xva.shape} pos={yva.sum()} | nuisance concepts={len(NUIS)}",flush=True)

sc=StandardScaler().fit(Xtr); Ztr=sc.transform(Xtr); Zva=sc.transform(Xva)
# nuisance heads: Ridge per concept (predict n̂ from features), fit on train
nheads=[Ridge(alpha=10.0).fit(Ztr,Ntr[:,k]) for k in range(len(NUIS))]
Nhat_tr=np.stack([h.predict(Ztr) for h in nheads],1); Nhat_va=np.stack([h.predict(Zva) for h in nheads],1)
r2=[roc_auc_score((Ntr[:,k]>np.median(Ntr[:,k])).astype(int), Nhat_tr[:,k]) if len(set(Ntr[:,k]>np.median(Ntr[:,k])))>1 else np.nan for k in range(len(NUIS))]
# concept audit: |corr(concept,label)| (want LOW) and cross-center shift |mean_c1-mean_c2| (want HIGH)
print("\nnuisance concept audit (|corr label| LOW ok, center-shift HIGH = useful, n̂ AUROC = predictable):")
for k,nm in enumerate(NUIS):
    corr=abs(np.corrcoef(Ntr[:,k],ytr)[0,1]); shift=abs(Ntr[ctr=='center_1',k].mean()-Ntr[ctr=='center_2',k].mean())
    print(f"  {nm:22s} |corr_label|={corr:.3f}  center_shift={shift:.3f}  head_AUROC={r2[k]:.3f}")

def ks(a,b): return ks_2samp(a,b).statistic
def recall_at(thr,s,y): return (s[y==1]>=thr).mean()
def fpr_at(thr,s,y): return (s[y==0]>=thr).mean()
def thr_90R(s,y):     # score threshold giving 90% recall on this (source) set
    sp=np.sort(s[y==1])[::-1]; k=int(np.ceil(0.9*len(sp)))-1; return sp[min(k,len(sp)-1)]

print("\n=== CRISP mechanism test (held-out val, both LOCO legs) ===")
print(f"{'leg':>10} {'metric':>26} {'RAW s':>9} {'RESID r':>9} {'better?':>8}")
for S,T in [("center_1","center_2"),("center_2","center_1")]:
    trS=ctr==S
    det=LogisticRegression(max_iter=3000,C=1.0,class_weight="balanced").fit(Ztr[trS],ytr[trS])
    sv=det.predict_proba(Zva)[:,1]; sv=np.log(np.clip(sv,1e-6,1-1e-6)/(1-np.clip(sv,1e-6,1-1e-6)))   # logit
    # FWL: fit μ₀,σ₀ on SOURCE (train) NEGATIVES
    st=det.predict_proba(Ztr[trS])[:,1]; st=np.log(np.clip(st,1e-6,1-1e-6)/(1-np.clip(st,1e-6,1-1e-6)))
    negS=ytr[trS]==0
    mu0=Ridge(alpha=1.0).fit(Nhat_tr[trS][negS], st[negS]); resid=st[negS]-mu0.predict(Nhat_tr[trS][negS]); sig0=resid.std()+1e-6
    rv=(sv-mu0.predict(Nhat_va))/sig0
    vS=cva==S; vT=cva==T
    # (1) negative-score cross-center DRIFT (want RESID < RAW)
    d_raw=ks(sv[vS&(yva==0)], sv[vT&(yva==0)]); d_res=ks(rv[vS&(yva==0)], rv[vT&(yva==0)])
    print(f"{S[-1]+'->'+T[-1]:>10} {'neg-drift KS':>26} {d_raw:9.3f} {d_res:9.3f} {'YES' if d_res<d_raw else 'no':>8}")
    # (2) threshold-transfer recall: set 90R thr on SOURCE, apply to TARGET (want recall stay ~0.90)
    for name,s in [("raw",sv),("resid",rv)]:
        pass
    thr_raw=thr_90R(sv[vS],yva[vS]); rec_raw=recall_at(thr_raw,sv[vT],yva[vT]); fpr_raw=fpr_at(thr_raw,sv[vT],yva[vT])
    thr_res=thr_90R(rv[vS],yva[vS]); rec_res=recall_at(thr_res,rv[vT],yva[vT]); fpr_res=fpr_at(thr_res,rv[vT],yva[vT])
    print(f"{'':>10} {'transfer recall@90R':>26} {rec_raw:9.3f} {rec_res:9.3f} {'(want ~0.90)':>8}")
    print(f"{'':>10} {'transfer FPR@90R (target)':>26} {fpr_raw:9.3f} {fpr_res:9.3f} {'YES' if fpr_res<fpr_raw else 'no':>8}")
    # (3) target-center AUROC (should be preserved — residual must not destroy discrimination)
    a_raw=roc_auc_score(yva[vT],sv[vT]); a_res=roc_auc_score(yva[vT],rv[vT])
    print(f"{'':>10} {'target AUROC (preserve)':>26} {a_raw:9.3f} {a_res:9.3f} {'ok' if a_res>=a_raw-0.02 else 'DROP':>8}")
print("\nDONE",flush=True)
