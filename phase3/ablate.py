"""ABLATION on the honest frozen-DINOv2 LOCO compass — answers 2 design questions cleanly (only ONE factor
varies, same frozen backbone, so no confound like exps/3 had):
  Q1 IMAGE SIZE: does 336 -> 448 -> 518 improve CROSS-CENTER transfer? (exps/2@336 beat exps/3@448 but that
     was confounded by backbone; here it's pure resolution.)
  Q2 POOLING / CG-AMIL: does gated attention-MIL beat mean-pool for cross-center? (attention REGRESSED in
     exps/3, but confounded; here: same frozen features, only the pool head differs, trained identically.)
Metric = LOCO AUROC (c1->c2 and c2->c1) via linear/attention probe on FROZEN features."""
import os, csv, sys, time, json, warnings
import numpy as np, torch, torch.nn as nn
from PIL import Image
warnings.filterwarnings("ignore")
import timm, timm.models.vision_transformer as vit_mod
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT="/Volumes/Shin/RARE2026"; DEV=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
rows=[r for r in csv.DictReader(open(f"{ROOT}/dataset/val.csv")) if r["aug"]=="orig"]
paths=[r["path"] for r in rows]; y=np.array([int(r["label"]) for r in rows])
cen=np.array([r["center"] for r in rows]); c1=cen=="center_1"; c2=cen=="center_2"; N=len(rows)

def frozen_dinov2(img):
    m=timm.create_model("vit_base_patch14_reg4_dinov2",pretrained=False,img_size=img,num_classes=0)
    sd=torch.load(f"{ROOT}/dinov2.pth",map_location="cpu")["teacher"]
    sd={k[len("backbone."):]:v for k,v in sd.items() if k.startswith("backbone.")}
    sd=vit_mod.checkpoint_filter_fn(sd,m); miss,_=m.load_state_dict(sd,strict=False); assert not miss
    return m.to(DEV).eval()

def extract(img, keep_patches=False):
    m=frozen_dinov2(img); cls=np.zeros((N,768),np.float32); mean=np.zeros((N,768),np.float32); mx=np.zeros((N,768),np.float32)
    P=[] if keep_patches else None; t0=time.time()
    with torch.no_grad():
        for st in range(0,N,32):
            ims=[Image.open(p).convert("RGB").resize((img,img),Image.BILINEAR) for p in paths[st:st+32]]
            xb=torch.stack([torch.from_numpy(np.asarray(i,np.float32).copy()).permute(2,0,1)/255. for i in ims])
            xb=((xb-MEAN)/STD).to(DEV); f=m.forward_features(xb); pt=f[:,5:]
            cls[st:st+xb.shape[0]]=f[:,0].float().cpu().numpy()
            mean[st:st+xb.shape[0]]=pt.mean(1).float().cpu().numpy()
            mx[st:st+xb.shape[0]]=pt.amax(1).float().cpu().numpy()
            if keep_patches: P.append(pt.float().cpu().numpy())
    print(f"  extracted @{img}px ({time.time()-t0:.0f}s)",flush=True)
    return cls, mean, mx, (np.concatenate(P) if keep_patches else None)

def auc(yy,ss): return roc_auc_score(yy,ss) if len(set(yy))>1 else float("nan")
def lp_loco(X, tag):
    out={}
    for tr,te,d in [(c1,c2,"c1->c2"),(c2,c1,"c2->c1")]:
        sc=StandardScaler().fit(X[tr]); lp=LogisticRegression(max_iter=3000,C=1.0,class_weight="balanced").fit(sc.transform(X[tr]),y[tr])
        out[d]=round(auc(y[te],lp.predict_proba(sc.transform(X[te]))[:,1]),4)
    out["mean"]=round((out["c1->c2"]+out["c2->c1"])/2,4); return out

class AttnMIL(nn.Module):
    def __init__(s,d=768,h=128):
        super().__init__(); s.V=nn.Linear(d,h,False); s.U=nn.Linear(d,h,False); s.w=nn.Linear(h,1,False); s.head=nn.Linear(2*d,1)
    def forward(s,p,cls):
        a=torch.softmax(s.w(torch.tanh(s.V(p))*torch.sigmoid(s.U(p))).squeeze(-1),1)
        return s.head(torch.cat([cls,(a.unsqueeze(-1)*p).sum(1)],-1)).squeeze(-1)

def attn_loco(P, cls, epochs=60):
    """Train gated attention-MIL on frozen patches of the train center, eval the other. Fair vs mean-pool
    (a Linear on cls+mean trained the SAME torch way)."""
    res={"attnMIL":{}, "meanpool(torch)":{}}
    for tr,te,d in [(c1,c2,"c1->c2"),(c2,c1,"c2->c1")]:
        Ptr=torch.tensor(P[tr],device=DEV); ctr=torch.tensor(cls[tr],device=DEV); ytr=torch.tensor(y[tr],dtype=torch.float32,device=DEV)
        Pte=torch.tensor(P[te],device=DEV); cte=torch.tensor(cls[te],device=DEV)
        pw=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],device=DEV)
        for name,net in [("attnMIL",AttnMIL().to(DEV)),
                         ("meanpool(torch)",nn.Sequential())]:
            if name=="meanpool(torch)":
                lin=nn.Linear(2*768,1).to(DEV); opt=torch.optim.Adam(lin.parameters(),1e-3,weight_decay=1e-3)
                Xtr=torch.cat([ctr,Ptr.mean(1)],-1); Xte=torch.cat([cte,Pte.mean(1)],-1)
                for _ in range(epochs):
                    opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(lin(Xtr).squeeze(-1),ytr,pos_weight=pw); loss.backward(); opt.step()
                with torch.no_grad(): s=torch.sigmoid(lin(Xte).squeeze(-1)).cpu().numpy()
            else:
                opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-3)
                for _ in range(epochs):
                    opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(net(Ptr,ctr),ytr,pos_weight=pw); loss.backward(); opt.step()
                with torch.no_grad(): s=torch.sigmoid(net(Pte,cte)).cpu().numpy()
            res[name][d]=round(auc(y[te],s),4)
    for k in res: res[k]["mean"]=round((res[k]["c1->c2"]+res[k]["c2->c1"])/2,4)
    return res

torch.manual_seed(0)
R={}
# Q1: image size (cls+mean linear probe)
print("=== Q1 image size ===",flush=True)
for img in [336,448]:
    cls,mean,mx,P=extract(img, keep_patches=(img==336))
    R[f"size{img}_cls+mean"]=lp_loco(np.concatenate([cls,mean],1),f"{img}")
    if img==336:
        cls336,mean336,mx336,P336=cls,mean,mx,P
# Q2: pooling at 336 (sklearn LP on frozen)
print("=== Q2 pooling (linear probe on frozen 336) ===",flush=True)
R["pool_mean"]=lp_loco(mean336,"mean"); R["pool_cls"]=lp_loco(cls336,"cls")
R["pool_max"]=lp_loco(mx336,"max"); R["pool_cls+mean"]=lp_loco(np.concatenate([cls336,mean336],1),"cls+mean")
R["pool_cls+max"]=lp_loco(np.concatenate([cls336,mx336],1),"cls+max")
# Q2b: attention-MIL vs mean-pool (both torch-trained, fair)
print("=== Q2b attention-MIL vs mean-pool (torch, frozen 336) ===",flush=True)
am=attn_loco(P336,cls336); R["CG-AMIL(attn)"]=am["attnMIL"]; R["meanpool(torch,ref)"]=am["meanpool(torch)"]

json.dump(R,open(f"{ROOT}/phase3/cache/ablate_results.json","w"),indent=1)
print("\n================ ABLATION (frozen DINOv2 LOCO AUROC) ================")
print(f"{'config':>22} {'c1->c2':>8} {'c2->c1':>8} {'MEAN':>8}")
for k,v in R.items(): print(f"{k:>22} {v['c1->c2']:>8} {v['c2->c1']:>8} {v['mean']:>8}")
print("\nDONE",flush=True)
