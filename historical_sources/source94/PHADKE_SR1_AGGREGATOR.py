#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4.1-SR1 | SOURCE-PROTOCOL RECONSTRUCTION
Appearance-only ResNet18 + CORN under Phadke et al. 2026 evaluation protocol.

Scientific purpose
------------------
Validate that our source-reconstructed implementation expresses the capability
reported by the 2026 same-dataset paper BEFORE transporting the identical method
to our pooled-OOF protocol.

Critical isolation:
- exact same reconstructed model recipe as P4.1-C
- exact same P4.1-C source40 ROI cache (no re-extraction)
- only evaluation/training protocol changes:
    published: 374-patient development set -> 5-fold CV, 94-patient locked test,
               average five fold-model probabilities on locked test
    P4.1-C:    our fixed-inner 5-fold pooled OOF across all 468 patients

This is a source-protocol reconstruction, NOT exact historical reproduction:
the public repository does not contain the final campaign output artifacts or
the exact final multi-seed manifest.
"""
from pathlib import Path
import os, sys, json, math, time, traceback, hashlib, zipfile, shutil
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import cohen_kappa_score

# ------------------------- immutable parents / public reference
EXPECTED_P4C_SHA="4089ed9d6e446abd76b16433f54a41a38c263a612bacefdee250d8f61e21804f"
EXPECTED_REPO_SHA="f2bc251fffaf90460703df1338b5d811957b493f73c2e1c222f948fd11496bf4"
REPO_COMMIT="df2b3020fe2ecb36203b09812803acfb847ba6d5"
P4C_RUN=Path("/PATH/TO/RESEARCH_WORKSPACE/server_runs/paper2_p4_1c_source_reconstructed_resnet18_corn_formal_20260918")
PY_EXPECTED="/PATH/TO/AUTHOR_HOME/miniforge3/envs/lss-foraminal-v2/bin/python"

PUBLISHED=dict(
    paper="Phadke et al., Journal of Imaging 2026",
    appearance_only_qwk=0.806,
    appearance_only_qwk_ci=[0.756,0.844],
    locked_test_patients=94,
    locked_test_rois=580,
    locked_test_grade_counts={0:378,1:104,2:53,3:45},
    development_patients=374,
    development_rois=2398,
    primary_endpoint="QWK",
    bootstrap_resamples=2000,
)

# ------------------------- exact P4.1-C recipe lock
K=4
SEED=1337
RECIPE=dict(
    backbone="resnet18", img_size=224,
    epochs=60, patience=14, bs=32,
    lr_backbone=2e-4, lr_head=1e-3, wd=1e-4,
    warmup=3, freeze_blocks=0, drop=0.3, drop_path=0.0,
    ema=0.995, sampler="balanced", loss_w="none", smooth=0.0, clip=1.0,
    rot_deg=10.0, trans=0.05, scale=0.10, bright=0.20, contrast=0.20,
    noise=0.0, erase_p=0.25, n_tta=1,
)
GRAY_MEAN,GRAY_STD=0.449,0.226

def mark(s): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}",flush=True)
def req(c,m):
    if not c: raise RuntimeError(m)
def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def bytes_sha(b): return hashlib.sha256(b).hexdigest()
def norm_pid(v):
    s=str(v).strip()
    if s.isdigit(): return s.zfill(4)
    import re
    m=re.findall(r"(?<!\d)(\d{4})(?!\d)",s)
    return m[-1] if m else s

# ------------------------- parent validation / split
def extract_repo(repo_zip,run):
    ext=run/"repo_extract"
    if ext.exists(): shutil.rmtree(ext)
    ext.mkdir()
    with zipfile.ZipFile(repo_zip) as z: z.extractall(ext)
    roots=[p for p in ext.iterdir() if p.is_dir()]
    req(len(roots)==1,"REPO_ROOT_FAIL")
    return roots[0]

def validate_p4c_parent(p4c_zip):
    with zipfile.ZipFile(p4c_zip) as z:
        contract=json.loads(z.read("06_RECONSTRUCTION_CONTRACT.json"))
        idx_bytes=z.read("roi_source40_index.csv")
        status=json.loads(z.read("00_FINAL_STATUS.json"))
    req(status.get("status")=="PASS" and status.get("models_complete")==5,"P4C_PARENT_STATUS_FAIL")
    parent_recipe=contract.get("recipe",{})
    # numeric/string exactness after json normalization
    req(parent_recipe==RECIPE,
        "P4C_RECIPE_DRIFT|parent="+json.dumps(parent_recipe,sort_keys=True)+
        "|expected="+json.dumps(RECIPE,sort_keys=True))
    server_idx=P4C_RUN/"roi_source40_index.csv"
    cache=P4C_RUN/"cache"/"rois_source40_224"
    req(server_idx.exists(),"P4C_SERVER_ROI_INDEX_MISSING")
    req(bytes_sha(idx_bytes)==sha256(server_idx),"P4C_RESULT_VS_SERVER_ROI_INDEX_SHA_MISMATCH")
    req(cache.is_dir(),"P4C_ROI_CACHE_MISSING")
    pngs=list(cache.glob("*.png"))
    req(len(pngs)==2978,f"P4C_ROI_CACHE_COUNT_FAIL:{len(pngs)}")
    idx=pd.read_csv(server_idx)
    req(len(idx)==2978 and idx["foramen_id"].nunique()==2978,"P4C_ROI_INDEX_CLOSURE_FAIL")
    return idx,cache,contract

def load_source_split(repo,idx):
    sp=pd.read_csv(repo/"data"/"splits.csv")
    req(len(sp)==469,"SOURCE_SPLIT_ROWS_FAIL")
    req({"patient","split"}.issubset(sp.columns),"SOURCE_SPLIT_COLUMNS_FAIL")
    sp["patient"]=sp["patient"].map(norm_pid)
    req(sp.patient.nunique()==469,"SOURCE_SPLIT_PATIENT_UNIQUE_FAIL")
    counts=sp["split"].value_counts().to_dict()
    exp={"test":94,"fold0":75,"fold1":75,"fold2":75,"fold3":75,"fold4":75}
    req(counts==exp,f"SOURCE_SPLIT_COUNTS_FAIL:{counts}")
    smap=dict(zip(sp.patient,sp["split"]))
    req(smap.get("0329")=="fold1","EXCLUDED_PATIENT_SPLIT_FAIL")
    d=idx.copy()
    d["patient"]=d["patient"].map(norm_pid)
    req(d.patient.nunique()==468 and len(d)==2978,"INDEX_COHORT_FAIL")
    d["source_split"]=d["patient"].map(smap)
    req(d.source_split.notna().all(),"ANALYZABLE_PATIENT_MISSING_FROM_SOURCE_SPLIT")
    te=d[d.source_split=="test"].copy()
    dev=d[d.source_split!="test"].copy()
    req(te.patient.nunique()==94 and len(te)==580,
        f"LOCKED_TEST_SIZE_FAIL patients={te.patient.nunique()} rois={len(te)}")
    req(dev.patient.nunique()==374 and len(dev)==2398,
        f"DEVELOPMENT_SIZE_FAIL patients={dev.patient.nunique()} rois={len(dev)}")
    g=te["grade"].astype(int).value_counts().sort_index().to_dict()
    req(g==PUBLISHED["locked_test_grade_counts"],f"TEST_GRADE_COUNTS_FAIL:{g}")
    # patient purity
    req(not (set(te.patient)&set(dev.patient)),"SOURCE_PROTOCOL_PATIENT_LEAKAGE")
    devfold={}
    for f in range(5):
        x=dev[dev.source_split==f"fold{f}"]
        devfold[f]={"patients":int(x.patient.nunique()),"rois":int(len(x))}
    return d,dev,te,counts,devfold

# ------------------------- image bank / network
class RoiBank:
    def __init__(self,cache,ids):
        arr=[]; keep=[]
        for fid in ids:
            p=cache/f"{fid}.png"
            req(p.exists(),"ROI_MISSING:"+str(fid))
            a=np.asarray(Image.open(p).convert("L"),dtype=np.uint8)
            req(a.shape==(224,224),"ROI_SHAPE_FAIL:"+str(fid))
            arr.append(a); keep.append(str(fid))
        self.X=torch.from_numpy(np.stack(arr))
        self.row={f:i for i,f in enumerate(keep)}
        mark(f"ROI_BANK_READY shape={tuple(self.X.shape)}")
    def rows_for(self,ids): return np.array([self.row[str(x)] for x in ids],dtype=np.int64)

class Net(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        self.backbone=timm.create_model("resnet18",pretrained=pretrained,num_classes=0,global_pool="avg",in_chans=1)
        self.drop=nn.Dropout(RECIPE["drop"])
        self.head=nn.Linear(self.backbone.num_features,K-1)
    def forward(self,x): return self.head(self.drop(self.backbone(x)))

class EMA:
    def __init__(self,m,d):
        self.d=d
        self.s={k:v.detach().clone().float() for k,v in m.state_dict().items()}
    @torch.no_grad()
    def update(self,m):
        for k,v in m.state_dict().items():
            if v.dtype.is_floating_point:
                self.s[k].mul_(self.d).add_(v.detach().float(),alpha=1-self.d)
            else: self.s[k].copy_(v.detach())
    def copy_to(self,m): m.load_state_dict(self.s,strict=True)

def corn_loss(logits,y):
    total=torch.tensor(0.0,device=logits.device); n=torch.tensor(0.0,device=logits.device)
    for k in range(K-1):
        mask=torch.ones_like(y,dtype=torch.bool) if k==0 else y>(k-1)
        if mask.sum()==0: continue
        l=F.binary_cross_entropy_with_logits(logits[mask,k],(y[mask]>k).float(),reduction="sum")
        total+=l; n+=mask.sum()
    return total/torch.clamp(n,min=1.0)

def corn_probs(logits):
    p=torch.sigmoid(logits)
    c=torch.cumprod(p,dim=1)
    o=torch.zeros(logits.shape[0],K,device=logits.device)
    o[:,0]=1-c[:,0]; o[:,1]=c[:,0]-c[:,1]; o[:,2]=c[:,1]-c[:,2]; o[:,3]=c[:,2]
    return o.clamp_min(1e-6)

_PAD=None
def augment(x,g):
    global _PAD
    B,_,H,W=x.shape
    dev=x.device
    def u(n=B): return (torch.rand(n,generator=g)*2-1).to(dev)
    ang=u()*RECIPE["rot_deg"]*math.pi/180
    sc=1+u()*RECIPE["scale"]; tx=u()*RECIPE["trans"]*2; ty=u()*RECIPE["trans"]*2
    co,si=torch.cos(ang)/sc,torch.sin(ang)/sc
    th=torch.zeros(B,2,3,device=dev)
    th[:,0,0]=co; th[:,0,1]=-si; th[:,0,2]=tx
    th[:,1,0]=si; th[:,1,1]=co; th[:,1,2]=ty
    grid=F.affine_grid(th,(B,1,H,W),align_corners=False)
    if _PAD is None:
        try:
            F.grid_sample(x[:1],grid[:1],padding_mode="reflection",align_corners=False)
            _PAD="reflection"
        except Exception: _PAD="zeros"
    x=F.grid_sample(x,grid,mode="bilinear",padding_mode=_PAD,align_corners=False)
    x=x+(u()*RECIPE["bright"]).view(B,1,1,1)
    m=x.mean((1,2,3),keepdim=True)
    x=(x-m)*(1+u()*RECIPE["contrast"]).view(B,1,1,1)+m
    hit=(torch.rand(B,generator=g)<RECIPE["erase_p"]).nonzero().flatten()
    for i in hit.tolist():
        eh=int(H*(0.08+0.20*torch.rand(1,generator=g).item()))
        ew=int(W*(0.08+0.20*torch.rand(1,generator=g).item()))
        y0=int(torch.randint(0,max(1,H-eh),(1,),generator=g).item())
        x0=int(torch.randint(0,max(1,W-ew),(1,),generator=g).item())
        x[i,:,y0:y0+eh,x0:x0+ew]=0
    return x

def qwk(y,p): return float(cohen_kappa_score(y,p,weights="quadratic",labels=[0,1,2,3]))
def sample_prob(y):
    cnt=np.maximum(np.bincount(y,minlength=K).astype(float),1)
    w=1/cnt; p=w[y]; return p/p.sum()

# ------------------------- train source CV fold
def train_fold(dev,test,bank,fold,run,device):
    fd=run/f"fold{fold}"; fd.mkdir(exist_ok=True)
    done=fd/"COMPLETE.json"
    if done.exists() and (fd/"val_predictions.npz").exists() and (fd/"test_predictions.npz").exists():
        mark(f"FOLD{fold} already complete")
        return json.load(open(done))
    val_name=f"fold{fold}"
    tr=dev[dev.source_split!=val_name].copy()
    va=dev[dev.source_split==val_name].copy()
    req(not(set(tr.patient)&set(va.patient)),"DEV_CV_PATIENT_LEAKAGE")
    req(not(set(tr.patient)&set(test.patient)) and not(set(va.patient)&set(test.patient)),
        "LOCKED_TEST_PATIENT_LEAKAGE")
    tr_idx=bank.rows_for(tr.foramen_id.values); va_idx=bank.rows_for(va.foramen_id.values); te_idx=bank.rows_for(test.foramen_id.values)
    ytr=tr.grade.values.astype(int); yva=va.grade.values.astype(int)

    run_seed=SEED*1000+fold
    np.random.seed(run_seed); torch.manual_seed(run_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(run_seed)
    gen=torch.Generator().manual_seed(run_seed)

    model=Net(True).to(device); shell=Net(False).to(device)
    bb=[p for n,p in model.named_parameters() if n.startswith("backbone")]
    hd=[p for n,p in model.named_parameters() if not n.startswith("backbone")]
    opt=torch.optim.AdamW([{"params":bb,"lr":RECIPE["lr_backbone"]},{"params":hd,"lr":RECIPE["lr_head"]}],
                          weight_decay=RECIPE["wd"])
    base=[g["lr"] for g in opt.param_groups]
    ema=EMA(model,RECIPE["ema"])
    ytr_t=torch.from_numpy(ytr)
    prob=sample_prob(ytr)
    steps=max(1,len(ytr)//RECIPE["bs"])
    best=-2; best_ep=0; best_state=None; wait=0; hist=[]

    def prep(ix):
        x=bank.X[ix].to(device).float().div_(255)
        return ((x-GRAY_MEAN)/GRAY_STD).unsqueeze(1)
    @torch.no_grad()
    def infer(m,ix):
        m.eval(); out=[]
        for i in range(0,len(ix),64):
            out.append(corn_probs(m(prep(ix[i:i+64]))).cpu())
        return torch.cat(out).numpy()

    mark(f"FOLD{fold} START train={len(tr)}({tr.patient.nunique()}p) val={len(va)}({va.patient.nunique()}p) locked_test={len(test)}(94p)")
    for ep in range(RECIPE["epochs"]):
        if ep<RECIPE["warmup"]: fac=(ep+1)/RECIPE["warmup"]
        else:
            prog=(ep-RECIPE["warmup"])/max(1,RECIPE["epochs"]-RECIPE["warmup"])
            fac=.5*(1+math.cos(math.pi*min(1,prog)))
        for g,b in zip(opt.param_groups,base): g["lr"]=b*fac
        model.train()
        order=np.random.choice(len(ytr),size=steps*RECIPE["bs"],replace=True,p=prob)
        losses=[]
        for s in range(steps):
            sel=order[s*RECIPE["bs"]:(s+1)*RECIPE["bs"]]
            x=augment(prep(tr_idx[sel]),gen); y=ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss=corn_loss(model(x),y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),RECIPE["clip"])
            opt.step(); ema.update(model); losses.append(float(loss.detach().cpu()))
        ema.copy_to(shell)
        Pv=infer(shell,va_idx)
        k=qwk(yva,Pv.argmax(1))
        if k>best:
            best=k; best_ep=ep+1; wait=0
            best_state={n:v.detach().cpu().clone() for n,v in shell.state_dict().items()}
        else: wait+=1
        hist.append({"epoch":ep+1,"loss":float(np.mean(losses)),"val_qwk_argmax":k,"best_qwk":best,"wait":wait})
        if ep==0 or (ep+1)%5==0 or wait>=RECIPE["patience"]:
            mark(f"FOLD{fold} ep={ep+1} valQWK={k:.4f} best={best:.4f}@{best_ep} wait={wait}")
        if wait>=RECIPE["patience"]: break

    req(best_state is not None,"BEST_STATE_MISSING")
    shell.load_state_dict(best_state)
    Pv=infer(shell,va_idx)
    Pt=infer(shell,te_idx)  # test labels are NOT used here
    np.savez_compressed(fd/"val_predictions.npz",
        foramen_id=va.foramen_id.values.astype(str),patient=va.patient.values.astype(str),
        y=yva,P=Pv,source_fold=np.full(len(va),fold,int))
    np.savez_compressed(fd/"test_predictions.npz",
        foramen_id=test.foramen_id.values.astype(str),patient=test.patient.values.astype(str),
        P=Pt,source_fold=np.full(len(test),fold,int))
    torch.save(best_state,fd/"best_ema_state.pt")
    pd.DataFrame(hist).to_csv(fd/"history.csv",index=False)
    meta={"fold":fold,"train_rows":len(tr),"train_patients":tr.patient.nunique(),
          "val_rows":len(va),"val_patients":va.patient.nunique(),
          "locked_test_rows":len(test),"locked_test_patients":test.patient.nunique(),
          "best_val_qwk_argmax":best,"best_epoch":best_ep,"epochs_ran":len(hist),
          "test_labels_used_for_training_or_selection":False,"run_seed":run_seed}
    done.write_text(json.dumps(meta,indent=2)+"\n")
    mark(f"FOLD{fold} COMPLETE bestValQWK={best:.4f}@{best_ep}")
    del model,shell,ema,best_state
    torch.cuda.empty_cache()
    return meta

# ------------------------- source rule: fit on full dev OOF, apply once to locked test
def apply_expected(P,thr):
    e=P@np.arange(K)
    return np.digitize(e,np.sort(np.asarray(thr)))
def apply_cumulative(P,taus):
    cum=np.cumsum(P[:,::-1],axis=1)[:,::-1]
    out=np.zeros(len(P),int)
    for k in range(1,K): out+=(cum[:,k]>taus[k-1]).astype(int)
    return out
def fit_expected(P,y,passes=4,step=.02):
    th=[.5,1.5,2.5]; best=qwk(y,apply_expected(P,th)); grid=np.arange(0,3.0001,step)
    for _ in range(passes):
        imp=False
        for i in range(3):
            cur=th[i]
            for g in grid:
                c=list(th); c[i]=float(g)
                if not(c[0]<=c[1]<=c[2]): continue
                s=qwk(y,apply_expected(P,c))
                if s>best+1e-9: best,cur,imp=s,float(g),True
            th[i]=cur
        if not imp: break
    return th,best
def fit_cumulative(P,y,passes=4,step=.02):
    t=[.5,.5,.5]; best=qwk(y,apply_cumulative(P,t)); grid=np.arange(.02,.99,step)
    for _ in range(passes):
        imp=False
        for i in range(3):
            cur=t[i]
            for g in grid:
                c=list(t); c[i]=float(g); s=qwk(y,apply_cumulative(P,c))
                if s>best+1e-9: best,cur,imp=s,float(g),True
            t[i]=cur
        if not imp: break
    return t,best
def fit_source_rule(P,y,margin=.01):
    arg=P.argmax(1); s_arg=qwk(y,arg)
    et,s_exp=fit_expected(P,y); ct,s_cum=fit_cumulative(P,y)
    scores={"argmax":s_arg,"expected":s_exp,"cumulative":s_cum}
    best=max(scores.values())
    chosen=next(x for x in ["argmax","expected","cumulative"] if scores[x]>=best-margin)
    return {"chosen":chosen,"scores":scores,"expected_thresholds":et,"cumulative_thresholds":ct,
            "margin":margin,"fit_data":"development OOF only","locked_test_labels_seen":False}

def apply_rule(P,r):
    if r["chosen"]=="argmax": return P.argmax(1)
    if r["chosen"]=="expected": return apply_expected(P,r["expected_thresholds"])
    return apply_cumulative(P,r["cumulative_thresholds"])

# ------------------------- metrics / paper-style bootstrap
def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int)
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
    return {
        "Accuracy":float(accuracy_score(y,p)),
        "BalancedAccuracy":float(balanced_accuracy_score(y,p)),
        "MacroF1":float(f1_score(y,p,average="macro",labels=[0,1,2,3],zero_division=0)),
        "WeightedF1":float(f1_score(y,p,average="weighted",labels=[0,1,2,3],zero_division=0)),
        "QWK":qwk(y,p),
        "MAE":float(np.mean(np.abs(y-p))),
        "AdjacentAccuracy":float(np.mean(np.abs(y-p)<=1)),
    }

def patient_bootstrap(test,pred_col,B=2000,seed=1337):
    pats=np.array(sorted(test.patient.unique()))
    groups={p:np.where(test.patient.values==p)[0] for p in pats}
    if "grade" in test.columns:
        y=test["grade"].values.astype(int)
    elif "y_true" in test.columns:
        y=test["y_true"].values.astype(int)
    else:
        raise RuntimeError("BOOTSTRAP_LABEL_COLUMN_MISSING")
    pred=test[pred_col].values.astype(int)
    rng=np.random.default_rng(seed)
    names=list(metrics(y,pred).keys())
    vals={m:np.empty(B,float) for m in names}
    for b in range(B):
        draw=rng.choice(pats,size=len(pats),replace=True)
        ix=np.concatenate([groups[p] for p in draw])
        mm=metrics(y[ix],pred[ix])
        for m in names: vals[m][b]=mm[m]
    pt=metrics(y,pred)
    return pd.DataFrame([{"metric":m,"point":pt[m],
                          "ci_low":float(np.quantile(vals[m],.025)),
                          "ci_high":float(np.quantile(vals[m],.975)),
                          "bootstrap_n":B,"seed":seed} for m in names])

# ------------------------- aggregate / one-time locked-test opening
def aggregate(dev,test,run,fold_meta):
    # 1) assemble development OOF
    v=[]
    test_probs=[]
    test_ids=None
    for f in range(5):
        z=np.load(run/f"fold{f}"/"val_predictions.npz",allow_pickle=True)
        P=z["P"]
        v.append(pd.DataFrame({
            "foramen_id":z["foramen_id"].astype(str),"patient":z["patient"].astype(str),
            "y_true":z["y"].astype(int),"source_fold":z["source_fold"].astype(int),
            "p0":P[:,0],"p1":P[:,1],"p2":P[:,2],"p3":P[:,3]}))
        t=np.load(run/f"fold{f}"/"test_predictions.npz",allow_pickle=True)
        ids=t["foramen_id"].astype(str)
        if test_ids is None: test_ids=ids
        else: req(np.array_equal(test_ids,ids),"TEST_ORDER_MISMATCH_ACROSS_FOLDS")
        test_probs.append(t["P"])
    oof=pd.concat(v,ignore_index=True)
    req(len(oof)==2398 and oof.foramen_id.nunique()==2398 and oof.patient.nunique()==374,
        "DEVELOPMENT_OOF_CLOSURE_FAIL")
    rule=fit_source_rule(oof[["p0","p1","p2","p3"]].values,oof.y_true.values,margin=.01)
    (run/"04_RULE_FROZEN_BEFORE_TEST_OPEN.json").write_text(json.dumps(rule,indent=2)+"\n")
    mark("SOURCE_DECISION_RULE_FROZEN="+rule["chosen"])
    mark("LOCKED_TEST_LABELS_NOW_OPENED_FOR_FINAL_EVALUATION")

    # 2) average five fold models on locked test
    P=np.mean(np.stack(test_probs,axis=0),axis=0)
    # align test labels AFTER rule freeze
    lab=test.set_index("foramen_id").loc[test_ids]
    req(len(lab)==580 and lab.patient.nunique()==94,"LOCKED_TEST_ALIGNMENT_FAIL")
    pred=apply_rule(P,rule)
    arg=P.argmax(1)
    res=pd.DataFrame({
        "foramen_id":test_ids,"patient":lab.patient.values.astype(str),
        "y_true":lab.grade.values.astype(int),
        "p0":P[:,0],"p1":P[:,1],"p2":P[:,2],"p3":P[:,3],
        "pred_source_rule":pred,"pred_argmax":arg
    })
    req(res.y_true.value_counts().sort_index().to_dict()==PUBLISHED["locked_test_grade_counts"],
        "FINAL_TEST_GRADE_COUNTS_FAIL")
    point_source=metrics(res.y_true,res.pred_source_rule)
    point_arg=metrics(res.y_true,res.pred_argmax)
    ci_source=patient_bootstrap(res.rename(columns={"grade":"y_true"}), "pred_source_rule", B=2000, seed=SEED)

    # supplement: if source rule differs from argmax, bootstrap argmax too
    ci_arg=patient_bootstrap(res, "pred_argmax", B=2000, seed=SEED)
    ci_source["system"]="source_rule"; ci_arg["system"]="argmax"
    cis=pd.concat([ci_source,ci_arg],ignore_index=True)

    pub=PUBLISHED["appearance_only_qwk"]
    q=point_source["QWK"]
    qrow=ci_source[ci_source.metric=="QWK"].iloc[0]
    bridge={
        "published_reference":PUBLISHED,
        "source_protocol_reconstruction":{
            "qwk":q,"qwk_ci_95":[float(qrow.ci_low),float(qrow.ci_high)],
            "absolute_delta_vs_published_qwk":float(q-pub),
            "published_qwk_inside_reconstruction_ci":bool(qrow.ci_low<=pub<=qrow.ci_high),
            "rule_chosen":rule["chosen"],
            "test_patients":94,"test_rois":580,
        },
        "protocol_transport_p4_1c":{
            "qwk":0.7679975933383562,
            "qwk_ci_95":[0.7442711924060808,0.789236146813469],
            "mae":0.2995298858294157,
            "patients":468,"rois":2978,
            "protocol":"our fixed-inner five-fold pooled OOF"
        },
        "interpretation_constraints":[
            "source-protocol reconstruction is not exact historical reproduction",
            "do not tune after viewing locked-test result",
            "published 0.806 and P4.1-C 0.768 are not direct head-to-head estimates because evaluation protocols differ",
            "P4.1-D paired comparisons must be performed only under the common OUR protocol"
        ]
    }

    oof.to_csv(run/"01_DEVELOPMENT_OOF.csv",index=False)
    res.to_csv(run/"02_LOCKED_TEST_PREDICTIONS.csv",index=False)
    pd.DataFrame([{"system":"source_rule",**point_source},{"system":"argmax",**point_arg}]).to_csv(run/"03_LOCKED_TEST_POINT_ESTIMATES.csv",index=False)
    cis.to_csv(run/"05_PAPER_STYLE_BOOTSTRAP_2000.csv",index=False)
    pd.DataFrame(fold_meta).to_csv(run/"06_FOLD_TRAINING_SUMMARY.csv",index=False)
    (run/"07_PROTOCOL_BRIDGE.json").write_text(json.dumps(bridge,indent=2)+"\n")

    status={
        "status":"PASS",
        "name":"P4.1-SR1 source-protocol reconstruction",
        "exact_historical_reproduction":False,
        "source_rule":rule["chosen"],
        "locked_test_qwk":q,
        "locked_test_qwk_ci_95":[float(qrow.ci_low),float(qrow.ci_high)],
        "published_appearance_qwk":pub,
        "published_qwk_inside_reconstruction_ci":bool(qrow.ci_low<=pub<=qrow.ci_high),
        "absolute_delta_qwk_vs_published":float(q-pub),
        "locked_test_patients":94,"locked_test_rois":580,
        "development_patients":374,"development_rois":2398,
        "models_complete":5,
        "test_used_for_model_or_rule_selection":False,
        "next":"Return to P4.1-D1 paired comparison under OUR protocol"
    }
    (run/"00_FINAL_STATUS.json").write_text(json.dumps(status,indent=2)+"\n")

    lines=[
        "P4.1-SR1 | SOURCE-PROTOCOL RECONSTRUCTION",
        "",
        "STATUS=PASS",
        "EXACT_HISTORICAL_REPRODUCTION=False",
        "MODELS=5/5",
        "DEVELOPMENT=374 patients / 2398 ROIs",
        "LOCKED_TEST=94 patients / 580 ROIs",
        "PRIMARY_ENDPOINT=QWK",
        "BOOTSTRAP=2000 patient-cluster (paper-style)",
        "",
        f"SOURCE_RULE_CHOSEN={rule['chosen']}",
        f"DEV_OOF_RULE_SCORES={rule['scores']}",
        "",
        "LOCKED_TEST_SOURCE_RULE:",
        json.dumps(point_source,sort_keys=True),
        f"QWK_95CI=[{qrow.ci_low:.9f}, {qrow.ci_high:.9f}]",
        "",
        "PUBLISHED_APPEARANCE_ONLY:",
        "QWK=0.806 | 95% CI [0.756, 0.844]",
        f"RECON_MINUS_PUBLISHED_QWK={q-pub:+.9f}",
        f"PUBLISHED_QWK_INSIDE_RECON_CI={bool(qrow.ci_low<=pub<=qrow.ci_high)}",
        "",
        "PROTOCOL TRANSPORT REFERENCE (P4.1-C, OUR pooled OOF):",
        "QWK=0.7679975933 | 95% CI [0.7442711924, 0.7892361468]",
        "",
        "INTERPRETATION LOCK:",
        "- This validates the source-reconstructed implementation under the published evaluation structure.",
        "- It is not an exact historical reproduction because final campaign multi-seed artifacts are absent.",
        "- No post-test tuning is permitted.",
        "- P4.1-D1 returns to OUR common protocol for paired comparison."
    ]
    (run/"08_HUMAN_READABLE_RESULTS.txt").write_text("\n".join(lines)+"\n")

    name="PAPER2_P4_1_SR1_SOURCE_PROTOCOL_RECONSTRUCTION_RESULT.zip"
    rz=run/name
    with zipfile.ZipFile(rz,"w",zipfile.ZIP_DEFLATED) as z:
        for n in ["00_FINAL_STATUS.json","01_DEVELOPMENT_OOF.csv","02_LOCKED_TEST_PREDICTIONS.csv",
                  "03_LOCKED_TEST_POINT_ESTIMATES.csv","04_RULE_FROZEN_BEFORE_TEST_OPEN.json",
                  "05_PAPER_STYLE_BOOTSTRAP_2000.csv","06_FOLD_TRAINING_SUMMARY.csv",
                  "07_PROTOCOL_BRIDGE.json","08_HUMAN_READABLE_RESULTS.txt"]:
            z.write(run/n,arcname=n)
    s=sha256(rz); (run/(name+".sha256")).write_text(f"{s}  {name}\n")
    (run/"DONE").write_text("PASS\n")
    mark("RESULT_ZIP="+str(rz)); mark("RESULT_SHA256="+s); mark("P4_1_SR1=PASS")

def main():
    if len(sys.argv)!=4:
        raise SystemExit("usage: script RUN_DIR P4C_RESULT_ZIP SOURCE_REPO_ZIP")
    run=Path(sys.argv[1]); p4c=Path(sys.argv[2]); repo_zip=Path(sys.argv[3]); run.mkdir(parents=True,exist_ok=True)
    req(sha256(p4c)==EXPECTED_P4C_SHA,"P4C_PARENT_SHA_FAIL")
    req(sha256(repo_zip)==EXPECTED_REPO_SHA,"SOURCE_REPO_SHA_FAIL")
    req(Path(sys.executable).resolve()==Path(PY_EXPECTED).resolve(),"PYTHON_ENV_FAIL")
    req(torch.cuda.is_available(),"CUDA_NOT_AVAILABLE")
    mark(f"ENV torch={torch.__version__} timm={getattr(timm,'__version__','?')} gpu={torch.cuda.get_device_name(0)}")
    idx,cache,parent_contract=validate_p4c_parent(p4c)
    repo=extract_repo(repo_zip,run)
    d,dev,test,split_counts,devfold=load_source_split(repo,idx)
    mark(f"SOURCE_PROTOCOL_COHORT=PASS dev={len(dev)}/{dev.patient.nunique()}p test={len(test)}/{test.patient.nunique()}p")
    mark("DEV_FOLD_COUNTS="+json.dumps(devfold,sort_keys=True))
    pre={
        "status":"LOCKED_BEFORE_TRAINING",
        "p4c_parent_sha256":EXPECTED_P4C_SHA,
        "source_repo_sha256":EXPECTED_REPO_SHA,
        "source_repo_commit":REPO_COMMIT,
        "recipe":RECIPE,"seed":SEED,
        "published_reference":PUBLISHED,
        "same_roi_cache_as_p4_1c":True,
        "test_labels_allowed_for_selection":False,
        "source_decision_rule":"fit on development OOF; simplest within 0.01 QWK; then one-time locked-test evaluation",
        "paper_style_bootstrap":"2000 patient-cluster resamples",
        "environment_note":"server environment differs from reported Python 3.9.6 / torch 2.8.0 environment"
    }
    (run/"SCIENTIFIC_LOCK.json").write_text(json.dumps(pre,indent=2)+"\n")
    bank=RoiBank(cache,d.foramen_id.tolist())
    device=torch.device("cuda")
    fold_meta=[]
    for f in range(5):
        fold_meta.append(train_fold(dev,test,bank,f,run,device))
    aggregate(dev,test,run,fold_meta)

if __name__=="__main__":
    try: main()
    except Exception:
        try:
            rd=Path(sys.argv[1]); rd.mkdir(parents=True,exist_ok=True)
            (rd/"FAILED").write_text(traceback.format_exc())
        except Exception: pass
        traceback.print_exc(); raise
