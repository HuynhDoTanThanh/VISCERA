"""Frozen-feature LOCO decision harness — the HONEST cross-center compass (§19).

Our ships saw BOTH centers, so true LOCO needs retraining. This probe instead does the
label-free-backbone LOCO: fit a linear probe on ONE center's FROZEN features, test the OTHER
center. That transfer number predicted the leaderboard (frozen DINOv3-LP c1->c2 = 0.776 ~=
hidden 0.756). Use it to decide: which frozen backbone transfers best, does a dinov2(+)dinov3
ensemble help cross-center, and it exports the frozen-LP member weights for the container.

Metrics per test center: AUROC, PPV@90R + FPR@90R (curve point), and the BOOTSTRAP-MEDIAN
PPV@90R + 95% CI (matches the competition estimator). Features are cached -> re-runs are instant.

Run:  .venv/bin/python phase3/loco_probe.py
Cache: phase3/cache/loco_feats.npz   Export: phase3/cache/frozen_lp_<bb>.npz
"""
import os, csv, sys, json, time, warnings
import numpy as np, torch
from PIL import Image
warnings.filterwarnings("ignore")
import timm, timm.models.vision_transformer as vit_mod
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = "/Volumes/Shin/RARE2026"
CACHE = f"{ROOT}/phase3/cache"; os.makedirs(CACHE, exist_ok=True)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); STD = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
# frozen backbones: (timm name, weight file, unwrap fn, img)
def _dinov2_sd():
    sd = torch.load(f"{ROOT}/dinov2.pth", map_location="cpu")["teacher"]
    return {k[len("backbone."):]: v for k,v in sd.items() if k.startswith("backbone.")}
def _dinov3_sd():
    sd = torch.load(f"{ROOT}/dinov3.pth", map_location="cpu"); return sd.get("model", sd) if isinstance(sd,dict) else sd
BACKBONES = {"dinov2": ("vit_base_patch14_reg4_dinov2", _dinov2_sd, 336),
             "dinov3": ("vit_base_patch16_dinov3", _dinov3_sd, 448)}

def load_val():
    rows = [r for r in csv.DictReader(open(f"{ROOT}/dataset/val.csv")) if r["aug"]=="orig"]
    return ([r["path"] for r in rows], np.array([int(r["label"]) for r in rows]),
            np.array([r["center"] for r in rows]))

def extract(paths):
    cf = f"{CACHE}/loco_feats.npz"
    if os.path.exists(cf):
        d = np.load(cf); print(f"[cache] loaded {cf}", flush=True); return {k: d[k] for k in d.files}
    feats = {}
    for bb,(tname,sdfn,img) in BACKBONES.items():
        m = timm.create_model(tname, pretrained=False, img_size=img, num_classes=0)
        sd = sdfn()
        try: sd = vit_mod.checkpoint_filter_fn(sd, m)
        except Exception: pass
        miss,unexp = m.load_state_dict(sd, strict=False); assert len(miss)==0, f"{bb} miss={miss[:4]}"
        m.to(DEV).eval()
        F = np.zeros((len(paths), 1536), np.float32); t0=time.time()
        with torch.no_grad():
            for st in range(0, len(paths), 32):
                ims = [Image.open(p).convert("RGB").resize((img,img), Image.BILINEAR) for p in paths[st:st+32]]
                xb = torch.stack([torch.from_numpy(np.asarray(im,np.float32).copy()).permute(2,0,1)/255. for im in ims])
                xb = ((xb-MEAN)/STD).to(DEV); f = m.forward_features(xb)
                F[st:st+xb.shape[0]] = torch.cat([f[:,0], f[:,5:].mean(1)], -1).float().cpu().numpy()
        feats[bb] = F; print(f"[extract] {bb} {img}px done ({time.time()-t0:.0f}s)", flush=True)
    np.savez(cf, **feats); return feats

# ---- metrics ----
def fpr90(y,s,R=.9):
    P=y.sum(); Nn=(y==0).sum()
    if P==0 or Nn==0: return np.nan
    o=np.argsort(-s,kind="mergesort"); ys=y[o]; tp=np.cumsum(ys); fp=np.cumsum(1-ys); rc=tp/P
    return fp[min(np.searchsorted(rc,R),len(rc)-1)]/Nn
def ppv1(f): return np.nan if np.isnan(f) else .01/(.01+.99*f)   # 1% prevalence (was .009 — prevalence bug)
def auc(y,s): return roc_auc_score(y,s) if len(set(y))>1 else float("nan")
def boot_ppv(y,s,B=1000):
    rng=np.random.RandomState(0); n=len(y); out=[]
    for _ in range(B):
        idx=rng.randint(0,n,n)
        if y[idx].sum()==0 or (y[idx]==0).sum()==0: continue
        out.append(ppv1(fpr90(y[idx],s[idx])))
    out=np.array(out); return (round(float(np.median(out)),4), round(float(np.percentile(out,2.5)),4), round(float(np.percentile(out,97.5)),4))

def lp_scores(Xtr,ytr,Xte):
    sc=StandardScaler().fit(Xtr); lp=LogisticRegression(max_iter=3000,C=1.0,class_weight="balanced").fit(sc.transform(Xtr),ytr)
    return lp.predict_proba(sc.transform(Xte))[:,1], (sc,lp)

def main():
    paths,y,cen = load_val(); feats = extract(paths)
    c1 = cen=="center_1"; c2 = cen=="center_2"
    print(f"\nval N={len(y)} pos={y.sum()} | c1={c1.sum()}(pos {y[c1].sum()}) c2={c2.sum()}(pos {y[c2].sum()})\n", flush=True)
    configs = {"dinov2":["dinov2"], "dinov3":["dinov3"], "ens(dv2+dv3)":["dinov2","dinov3"]}
    rows=[]
    for name,bbs in configs.items():
        for tr,te,tag in [(c1,c2,"c1->c2"),(c2,c1,"c2->c1")]:
            # per-backbone LP fit on train center, prob-avg for ensemble
            sc_te = np.mean([lp_scores(feats[bb][tr],y[tr],feats[bb][te])[0] for bb in bbs],0)
            f=fpr90(y[te],sc_te); m,lo,hi=boot_ppv(y[te],sc_te)
            rows.append({"config":name,"dir":tag,"AUROC":round(auc(y[te],sc_te),4),
                         "PPV@90R":round(ppv1(f),4),"FPR@90R":round(float(f),4),
                         "PPV_bootmedian":m,"PPV_CI":[lo,hi]})
    # reference: exps/2 actual scores per center (IN-DISTRIBUTION, not LOCO) if cached
    gs=f"{CACHE.replace('phase3/cache','')}".rstrip('/')
    g2=f"/private/tmp/claude-501/-Volumes-Shin-RARE2026/6faefad9-c3e7-49bc-a2d7-612d55d69ac8/scratchpad/ground2_scores.npz"
    if os.path.exists(g2):
        d=np.load(g2); s2=d["s2"]
        for te,tag in [(c2,"c2(in-dist)"),(c1,"c1(in-dist)")]:
            f=fpr90(y[te],s2[te]); rows.append({"config":"exps2-REF(not LOCO)","dir":tag,"AUROC":round(auc(y[te],s2[te]),4),
                "PPV@90R":round(ppv1(f),4),"FPR@90R":round(float(f),4),"PPV_bootmedian":boot_ppv(y[te],s2[te])[0],"PPV_CI":[None,None]})
    # export frozen-LP members fit on ALL val (for the container ensemble member)
    for bb in ["dinov2","dinov3"]:
        _,(sc,lp)=lp_scores(feats[bb],y,feats[bb])
        np.savez(f"{CACHE}/frozen_lp_{bb}.npz", mean=sc.mean_, scale=sc.scale_, coef=lp.coef_[0], intercept=lp.intercept_)
    json.dump(rows, open(f"{CACHE}/loco_probe_results.json","w"), indent=1)
    print("=== LOCO frozen-LP transfer (honest cross-center compass) ===")
    print(f"{'config':>20} {'dir':>12} {'AUROC':>7} {'PPV@90R':>8} {'FPR@90R':>8} {'PPV_bootmed':>11} {'CI':>18}")
    for r in rows:
        print(f"{r['config']:>20} {r['dir']:>12} {r['AUROC']:>7} {r['PPV@90R']:>8} {r['FPR@90R']:>8} {r['PPV_bootmedian']:>11} {str(r['PPV_CI']):>18}")
    print("\nExported frozen_lp_dinov2.npz, frozen_lp_dinov3.npz (container members). DONE", flush=True)

if __name__=="__main__": main()
