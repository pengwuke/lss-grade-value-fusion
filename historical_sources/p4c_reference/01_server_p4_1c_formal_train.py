#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4.1-C | Source-reconstructed 2026 contemporary matched baseline
FORMAL TRAINING

Frozen purpose
--------------
Reconstruct the 2026 same-dataset appearance-only method under OUR already-frozen
468-patient / 2978-ROI patient-disjoint 5-fold outer protocol with fixed-inner
validation.

This is intentionally NOT called an exact historical reproduction:
the public repository does not ship the final campaign best_arch/best_reg/final
manifest artifacts, so its final multi-seed recipe cannot be recovered byte-for-byte.

Pre-registered reconstruction used here
---------------------------------------
- raw expert foraminal annotation boxes
- 40% box-relative peri-foraminal margin
- grayscale direct square resize to 224x224, PIL bilinear
- ImageNet-pretrained timm ResNet-18, in_chans=1
- CORN K-1 independent conditional logits
- source explicit arch_r18_224 candidate + train2.py BASE:
  epochs=60, patience=14, batch=32
  AdamW backbone lr=2e-4, head lr=1e-3, wd=1e-4
  warmup=3 then cosine
  dropout=0.3, EMA=0.995
  grade-balanced sampling
  rot=10deg, trans=.05, scale=.10, brightness=.20, contrast=.20, erase=.25
  no label smoothing, no loss weights, grad clip=1
  n_tta=1
- source reproducibility seed 1337 (the only explicit global seed published/repo-locked)
- checkpoint selected ONLY by fixed-inner validation QWK
- outer TEST used only once after checkpoint freeze
- source decision-rule logic: nested 5-fold OOF rule selection among
  argmax / expected-grade thresholds / cumulative thresholds, simplest within 0.01 QWK
- harmonized argmax result reported separately
- primary endpoint QWK
- 20,000 patient-cluster bootstrap, seed 42

No endpoint switching. No fusion. No morphometry. No geometry G1.
"""
from pathlib import Path
import os, sys, json, csv, math, time, traceback, hashlib, zipfile, shutil, re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import cohen_kappa_score

# --------------------------------------------------------------------- frozen facts
EXPECTED_P4_1B_SHA = "09dc6ad632f5e5f454dc775777f60fead600c9359513c1a63d81bef44734f635"
EXPECTED_REPO_SHA = "f2bc251fffaf90460703df1338b5d811957b493f73c2e1c222f948fd11496bf4"
EXPECTED_MANIFEST_SHA = "5e64c21b9a7d70cb9eae5c336a08ffc0a317c904b1be36a7d01f660dc4c56d22"

N = 2978
NPAT = 468
GRADE_COUNTS = {0:2009,1:507,2:254,3:208}
FOLD_COUNTS = {0:618,1:596,2:557,3:616,4:591}
FIXED_VAL = {0:1,1:2,2:3,3:4,4:0}
LEVELS = ["L1-L2","L2-L3","L3-L4","L4-L5","L5-S1"]
K = 4
SEED = 1337

RAW_ROOT = Path("/data/LSS/raw/LSS_MRI_AISSLab_Dataset_V0.1")
MANIFEST = Path("/PATH/TO/RESEARCH_WORKSPACE/server_runs/step02a0_fix1_resolve_roi_paths_20260813/unified_5fold_manifest_server_resolved.csv")
PYTHON_EXPECTED = "/PATH/TO/AUTHOR_HOME/miniforge3/envs/lss-foraminal-v2/bin/python"

RECIPE = dict(
    backbone="resnet18", img_size=224,
    epochs=60, patience=14, bs=32,
    lr_backbone=2e-4, lr_head=1e-3, wd=1e-4,
    warmup=3, freeze_blocks=0, drop=0.3, drop_path=0.0,
    ema=0.995, sampler="balanced", loss_w="none", smooth=0.0, clip=1.0,
    rot_deg=10.0, trans=0.05, scale=0.10, bright=0.20, contrast=0.20,
    noise=0.0, erase_p=0.25, n_tta=1,
)
GRAY_MEAN, GRAY_STD = 0.449, 0.226

def mark(s):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}", flush=True)

def req(c,msg):
    if not c:
        raise RuntimeError(msg)

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def norm_pid(v):
    s=str(v).strip()
    if s.isdigit(): return s.zfill(4)
    m=re.findall(r"(?<!\d)(\d{4})(?!\d)",s)
    return m[-1] if m else s

def fold_num(v):
    s=str(v).strip().lower()
    if s.isdigit() and int(s) in range(5): return int(s)
    m=re.search(r"(?:fold[_-]?)?([0-4])$",s)
    return int(m.group(1)) if m else None

# ---------------------------------------------------------------- authoritative manifest
def load_manifest():
    req(MANIFEST.exists(),"AUTHORITATIVE_MANIFEST_NOT_FOUND")
    req(sha256(MANIFEST)==EXPECTED_MANIFEST_SHA,"AUTHORITATIVE_MANIFEST_SHA_MISMATCH")
    d=pd.read_csv(MANIFEST)
    req(len(d)==N,f"MANIFEST_N_FAIL:{len(d)}")
    req("patient_id" in d.columns and "severity" in d.columns and "unified_outer_fold" in d.columns,
        "MANIFEST_REQUIRED_COLUMNS_MISSING")
    d["patient_id"]=d["patient_id"].map(norm_pid)
    d["severity"]=d["severity"].astype(int)
    d["unified_outer_fold"]=d["unified_outer_fold"].astype(int)
    req(d.patient_id.nunique()==NPAT,"MANIFEST_PATIENT_COUNT_FAIL")
    req(d.severity.value_counts().sort_index().to_dict()==GRADE_COUNTS,"MANIFEST_GRADE_COUNTS_FAIL")
    req(d.unified_outer_fold.value_counts().sort_index().to_dict()==FOLD_COUNTS,"MANIFEST_FOLD_COUNTS_FAIL")
    # patient-level fold purity
    nun=d.groupby("patient_id")["unified_outer_fold"].nunique()
    req(int(nun.max())==1,"PATIENT_CROSSES_OUTER_FOLDS")
    p2f=d.groupby("patient_id")["unified_outer_fold"].first().to_dict()
    return d,p2f

# ---------------------------------------------------------------- raw source annotations
def parse_raw(p2f):
    det=RAW_ROOT/"Foramina_Detection"
    req(det.is_dir(),"RAW_FORAMINA_DETECTION_MISSING")
    rows=[]
    for xp in sorted(det.glob("*/*.xml")):
        patient=norm_pid(xp.parent.name)
        if patient not in p2f:
            # includes the one excluded patient 0329
            continue
        slice_idx=xp.stem
        png=xp.with_suffix(".png")
        if not png.exists():
            continue
        try:
            root=ET.parse(xp).getroot()
        except Exception:
            continue
        for obj in root.findall("object"):
            name=(obj.findtext("name") or "").strip()
            m=re.match(r"^([LR])FS([0-3])$",name)
            if not m:
                continue
            level=(obj.findtext("level") or "").strip()
            if level not in LEVELS:
                continue
            bb=obj.find("bndbox")
            if bb is None:
                continue
            try:
                xmin=int(float(bb.findtext("xmin"))); ymin=int(float(bb.findtext("ymin")))
                xmax=int(float(bb.findtext("xmax"))); ymax=int(float(bb.findtext("ymax")))
            except Exception:
                continue
            if xmax<=xmin or ymax<=ymin:
                continue
            rows.append(dict(
                patient=patient,slice_idx=str(slice_idx),level=level,side=m.group(1),
                grade=int(m.group(2)),xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,
                png_path=str(png),outer_fold=int(p2f[patient])
            ))
    d=pd.DataFrame(rows)
    req(len(d)==N,f"RAW_ANALYZABLE_N_FAIL:{len(d)}")
    req(d.patient.nunique()==NPAT,"RAW_ANALYZABLE_PATIENT_FAIL")
    req(d.grade.value_counts().sort_index().to_dict()==GRADE_COUNTS,"RAW_GRADE_COUNTS_FAIL")
    req(d.outer_fold.value_counts().sort_index().to_dict()==FOLD_COUNTS,"RAW_FOLD_COUNTS_FAIL")

    # exact source-paper foramen_id construction
    d=d.sort_values(["patient","slice_idx","level","side","ymin"]).reset_index(drop=True)
    dup=d.groupby(["patient","slice_idx","level","side"]).cumcount().astype(str)
    d["foramen_id"]=(d["patient"]+"_"+d["slice_idx"].astype(str)+"_"+
                     d["level"]+"_"+d["side"]+"_"+dup)
    req(d.foramen_id.is_unique,"FORAMEN_ID_NOT_UNIQUE")
    return d

def audit_patient_grade_multiset(raw, manifest):
    a=raw.groupby("patient")["grade"].apply(lambda s: tuple(sorted(map(int,s)))).to_dict()
    b=manifest.groupby("patient_id")["severity"].apply(lambda s: tuple(sorted(map(int,s)))).to_dict()
    req(set(a)==set(b),"PATIENT_SET_MISMATCH_RAW_MANIFEST")
    bad=[p for p in a if a[p]!=b[p]]
    req(not bad,"PATIENT_GRADE_MULTISET_MISMATCH:"+json.dumps(bad[:20]))
    return True

# ---------------------------------------------------------------- ROI cache
def build_roi_cache(df, cache_dir):
    cache_dir.mkdir(parents=True,exist_ok=True)
    index_csv=cache_dir.parent/"roi_source40_index.csv"
    # regenerate only missing files; crop semantics exactly source extract_rois.py
    n_new=0
    cache_img={}
    idx=[]
    for r in df.itertuples(index=False):
        op=cache_dir/f"{r.foramen_id}.png"
        bw=r.xmax-r.xmin; bh=r.ymax-r.ymin
        if not op.exists():
            if r.png_path not in cache_img:
                cache_img[r.png_path]=Image.open(r.png_path).convert("L")
            im=cache_img[r.png_path]
            W,H=im.size
            x0=max(0,int(r.xmin-0.40*bw)); x1=min(W,int(r.xmax+0.40*bw))
            y0=max(0,int(r.ymin-0.40*bh)); y1=min(H,int(r.ymax+0.40*bh))
            req(x1>x0 and y1>y0,"INVALID_SOURCE40_CROP:"+r.foramen_id)
            crop=im.crop((x0,y0,x1,y1)).resize((224,224),Image.BILINEAR)
            crop.save(op)
            n_new+=1
        idx.append(dict(foramen_id=r.foramen_id,patient=r.patient,outer_fold=int(r.outer_fold),
                        grade=int(r.grade),roi_path=str(op),png_path=r.png_path,
                        slice_idx=r.slice_idx,level=r.level,side=r.side,
                        xmin=r.xmin,ymin=r.ymin,xmax=r.xmax,ymax=r.ymax))
    pd.DataFrame(idx).to_csv(index_csv,index=False)
    files=list(cache_dir.glob("*.png"))
    req(len(files)==N,f"ROI_CACHE_COUNT_FAIL:{len(files)}")
    mark(f"ROI_CACHE_READY count={len(files)} newly_written={n_new}")
    return index_csv

class RoiBank:
    def __init__(self, cache_dir, ids):
        arrs=[]; keep=[]
        for fid in ids:
            p=cache_dir/f"{fid}.png"
            req(p.exists(),"ROI_MISSING:"+fid)
            a=np.asarray(Image.open(p).convert("L"),dtype=np.uint8)
            req(a.shape==(224,224),"ROI_SHAPE_FAIL:"+fid)
            arrs.append(a); keep.append(fid)
        self.X=torch.from_numpy(np.stack(arrs))
        self.row={f:i for i,f in enumerate(keep)}
        mark(f"RoiBank shape={tuple(self.X.shape)} MB={self.X.numel()/1e6:.1f}")
    def rows_for(self, ids):
        return np.array([self.row[x] for x in ids],dtype=np.int64)

# ---------------------------------------------------------------- source model/loss/aug
class Net(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone=timm.create_model(
            "resnet18",pretrained=pretrained,num_classes=0,global_pool="avg",in_chans=1
        )
        self.drop=nn.Dropout(RECIPE["drop"])
        self.head=nn.Linear(self.backbone.num_features,K-1)
    def forward(self,x):
        return self.head(self.drop(self.backbone(x)))

class EMA:
    def __init__(self,model,decay):
        self.decay=decay
        self.shadow={k:v.detach().clone().float() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self,model):
        for k,v in model.state_dict().items():
            s=self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(),alpha=1-self.decay)
            else:
                s.copy_(v.detach())
    def copy_to(self,model):
        model.load_state_dict({k:v for k,v in self.shadow.items()},strict=True)

def corn_loss(logits,y):
    total,n=0.0,0
    for k in range(K-1):
        mask=torch.ones_like(y,dtype=torch.bool) if k==0 else y>(k-1)
        if mask.sum()==0: continue
        lk=logits[mask,k]
        tk=(y[mask]>k).float()
        l=F.binary_cross_entropy_with_logits(lk,tk,reduction="none")
        total=total+l.sum()
        n=n+mask.sum()
    return total/torch.clamp(n.float(),min=1.0)

def corn_probs(logits):
    p=torch.sigmoid(logits)
    cum=torch.cumprod(p,dim=1)
    out=torch.zeros(logits.shape[0],K,device=logits.device)
    out[:,0]=1-cum[:,0]
    out[:,1]=cum[:,0]-cum[:,1]
    out[:,2]=cum[:,1]-cum[:,2]
    out[:,3]=cum[:,2]
    return out.clamp_min(1e-6)

_PAD_MODE=None
def _pad_mode(x,grid):
    global _PAD_MODE
    if _PAD_MODE is None:
        for cand in ("reflection","zeros"):
            try:
                F.grid_sample(x[:1],grid[:1],mode="bilinear",padding_mode=cand,align_corners=False)
                _PAD_MODE=cand; break
            except RuntimeError:
                continue
        if _PAD_MODE is None: _PAD_MODE="zeros"
    return _PAD_MODE

def augment(x,r,gen):
    B,_,H,W=x.shape
    dev=x.device
    def u(n=B):
        return (torch.rand(n,generator=gen)*2-1).to(dev)
    if r["rot_deg"]>0 or r["trans"]>0 or r["scale"]>0:
        ang=u()*r["rot_deg"]*math.pi/180.0
        sc=1.0+u()*r["scale"]
        tx,ty=u()*r["trans"]*2,u()*r["trans"]*2
        cos,sin=torch.cos(ang)/sc,torch.sin(ang)/sc
        theta=torch.zeros(B,2,3,device=dev)
        theta[:,0,0],theta[:,0,1],theta[:,0,2]=cos,-sin,tx
        theta[:,1,0],theta[:,1,1],theta[:,1,2]=sin,cos,ty
        grid=F.affine_grid(theta,(B,1,H,W),align_corners=False)
        x=F.grid_sample(x,grid,mode="bilinear",padding_mode=_pad_mode(x,grid),align_corners=False)
    if r["bright"]>0:
        x=x+(u()*r["bright"]).view(B,1,1,1)
    if r["contrast"]>0:
        m=x.mean(dim=(1,2,3),keepdim=True)
        x=(x-m)*(1.0+u()*r["contrast"]).view(B,1,1,1)+m
    if r["noise"]>0:
        x=x+torch.randn(x.shape,generator=gen).to(dev)*r["noise"]
    if r["erase_p"]>0:
        hit=(torch.rand(B,generator=gen)<r["erase_p"]).nonzero().flatten()
        for i in hit.tolist():
            eh=int(H*(0.08+0.20*torch.rand(1,generator=gen).item()))
            ew=int(W*(0.08+0.20*torch.rand(1,generator=gen).item()))
            y0=int(torch.randint(0,max(1,H-eh),(1,),generator=gen).item())
            x0=int(torch.randint(0,max(1,W-ew),(1,),generator=gen).item())
            x[i,:,y0:y0+eh,x0:x0+ew]=0.0
    return x

def sample_prob(y):
    cnt=np.maximum(np.bincount(y,minlength=K).astype(float),1)
    w=1.0/cnt
    p=w[y]
    return p/p.sum()

def qwk(y,p):
    return float(cohen_kappa_score(y,p,weights="quadratic",labels=[0,1,2,3]))

# ---------------------------------------------------------------- train one outer fold
def train_outer(df,bank,outer,run_dir,device):
    fold_dir=run_dir/f"fold{outer}"
    fold_dir.mkdir(parents=True,exist_ok=True)
    marker=fold_dir/"COMPLETE.json"
    pred_npz=fold_dir/"test_predictions.npz"
    val_npz=fold_dir/"val_predictions.npz"
    if marker.exists() and pred_npz.exists() and val_npz.exists():
        mark(f"OUTER{outer} already complete; skip")
        return json.load(open(marker))

    val=FIXED_VAL[outer]
    tr=df[~df.outer_fold.isin([outer,val])].copy()
    va=df[df.outer_fold==val].copy()
    te=df[df.outer_fold==outer].copy()
    expected={
        0:(1764,596,618),1:(1825,557,596),2:(1805,616,557),3:(1771,591,616),4:(1769,618,591)
    }[outer]
    req((len(tr),len(va),len(te))==expected,
        f"ROLE_COUNT_FAIL outer={outer} got={(len(tr),len(va),len(te))} exp={expected}")
    # patient disjointness
    a,b,c=set(tr.patient),set(va.patient),set(te.patient)
    req(not (a&b or a&c or b&c),f"PATIENT_LEAKAGE_OUTER{outer}")

    ids_tr=bank.rows_for(tr.foramen_id.values)
    ids_va=bank.rows_for(va.foramen_id.values)
    ids_te=bank.rows_for(te.foramen_id.values)
    y_tr=tr.grade.values.astype(np.int64)
    y_va=va.grade.values.astype(np.int64)
    y_te=te.grade.values.astype(np.int64)

    run_seed=SEED*1000+outer
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    model=Net(pretrained=True).to(device)
    shell=Net(pretrained=False).to(device)
    bb=[p for n,p in model.named_parameters() if n.startswith("backbone") and p.requires_grad]
    hd=[p for n,p in model.named_parameters() if not n.startswith("backbone") and p.requires_grad]
    opt=torch.optim.AdamW([{"params":bb,"lr":RECIPE["lr_backbone"]},
                           {"params":hd,"lr":RECIPE["lr_head"]}],weight_decay=RECIPE["wd"])
    base_lrs=[g["lr"] for g in opt.param_groups]
    ema=EMA(model,RECIPE["ema"])
    gen=torch.Generator().manual_seed(run_seed)
    p_samp=sample_prob(y_tr)
    ntr=len(y_tr)
    steps=max(1,ntr//RECIPE["bs"])
    ytr_t=torch.from_numpy(y_tr)
    best_k=-2.0; best_state=None; best_ep=0; wait=0

    def prep(indices):
        x=bank.X[indices].to(device).float().div_(255.0)
        x=(x-GRAY_MEAN)/GRAY_STD
        return x.unsqueeze(1)

    @torch.no_grad()
    def infer(m,indices):
        m.eval()
        out=[]
        bs=64
        for i in range(0,len(indices),bs):
            x=prep(indices[i:i+bs])
            out.append(corn_probs(m(x)).float().cpu())
        return torch.cat(out).numpy()

    hist=[]
    mark(f"OUTER{outer} START train={len(tr)} val={len(va)} test={len(te)} fixed_val={val} device={device}")
    for ep in range(RECIPE["epochs"]):
        if ep<RECIPE["warmup"]:
            f=(ep+1)/max(1,RECIPE["warmup"])
        else:
            prog=(ep-RECIPE["warmup"])/max(1,RECIPE["epochs"]-RECIPE["warmup"])
            f=0.5*(1+math.cos(math.pi*min(1.0,prog)))
        for g,b in zip(opt.param_groups,base_lrs):
            g["lr"]=b*f

        model.train()
        order=np.random.choice(ntr,size=steps*RECIPE["bs"],replace=True,p=p_samp)
        losses=[]
        for s in range(steps):
            sel=order[s*RECIPE["bs"]:(s+1)*RECIPE["bs"]]
            if len(sel)<2: continue
            x=prep(ids_tr[sel])
            x=augment(x,RECIPE,gen)
            y=ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss=corn_loss(model(x),y)
            loss.backward()
            if RECIPE["clip"]>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),RECIPE["clip"])
            opt.step()
            ema.update(model)
            losses.append(float(loss.detach().cpu()))
        ema.copy_to(shell)
        Pv=infer(shell,ids_va)
        kval=qwk(y_va,Pv.argmax(1))
        if kval>best_k:
            best_k=kval; best_ep=ep+1; wait=0
            best_state={k:v.detach().cpu().clone() for k,v in shell.state_dict().items()}
        else:
            wait+=1
        hist.append(dict(epoch=ep+1,loss=float(np.mean(losses)),val_qwk=kval,best_qwk=best_k,wait=wait))
        if ep==0 or (ep+1)%5==0 or wait>=RECIPE["patience"]:
            mark(f"OUTER{outer} ep={ep+1} loss={np.mean(losses):.4f} valQWK={kval:.4f} best={best_k:.4f}@{best_ep} wait={wait}")
        if wait>=RECIPE["patience"]:
            break

    req(best_state is not None,"BEST_STATE_MISSING")
    shell.load_state_dict(best_state,strict=True)
    Pv=infer(shell,ids_va)
    Pt=infer(shell,ids_te)
    np.savez_compressed(val_npz,
        foramen_id=va.foramen_id.values.astype(str),patient=va.patient.values.astype(str),
        y=y_va,P=Pv,fold=np.full(len(va),val,dtype=int))
    np.savez_compressed(pred_npz,
        foramen_id=te.foramen_id.values.astype(str),patient=te.patient.values.astype(str),
        y=y_te,P=Pt,fold=np.full(len(te),outer,dtype=int))
    torch.save(best_state,fold_dir/"best_ema_state.pt")
    pd.DataFrame(hist).to_csv(fold_dir/"history.csv",index=False)
    meta=dict(
        outer=outer,fixed_val=val,train_rows=len(tr),val_rows=len(va),test_rows=len(te),
        train_patients=tr.patient.nunique(),val_patients=va.patient.nunique(),test_patients=te.patient.nunique(),
        seed=SEED,run_seed=run_seed,best_val_qwk=float(best_k),best_epoch=int(best_ep),
        epochs_ran=len(hist),test_used_for_selection=False,
        checkpoint_selection="fixed-inner validation QWK only",
        test_inference_after_checkpoint_freeze=True
    )
    marker.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    mark(f"OUTER{outer} COMPLETE bestValQWK={best_k:.4f}@{best_ep}")
    del model,shell,ema,best_state
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return meta

# ---------------------------------------------------------------- source decision rule
def apply_expected(P,thr):
    e=P@np.arange(K)
    return np.digitize(e,np.sort(np.asarray(thr)))

def apply_cumulative(P,taus):
    cum=np.cumsum(P[:,::-1],axis=1)[:,::-1]
    out=np.zeros(len(P),dtype=int)
    for k in range(1,K):
        out+=(cum[:,k]>taus[k-1]).astype(int)
    return out

def fit_expected(P,y,passes=4,lo=0.0,hi=3.0,step=0.02):
    thr=[0.5,1.5,2.5]
    grid=np.arange(lo,hi+1e-9,step)
    best=qwk(y,apply_expected(P,thr))
    for _ in range(passes):
        improved=False
        for i in range(3):
            cur=thr[i]
            for g in grid:
                cand=list(thr); cand[i]=float(g)
                if not (cand[0]<=cand[1]<=cand[2]): continue
                s=qwk(y,apply_expected(P,cand))
                if s>best+1e-9:
                    best,cur,improved=s,float(g),True
            thr[i]=cur
        if not improved: break
    return thr,best

def fit_cumulative(P,y,passes=4,step=0.02):
    taus=[0.5,0.5,0.5]
    grid=np.arange(0.02,0.99,step)
    best=qwk(y,apply_cumulative(P,taus))
    for _ in range(passes):
        improved=False
        for i in range(3):
            cur=taus[i]
            for g in grid:
                cand=list(taus); cand[i]=float(g)
                s=qwk(y,apply_cumulative(P,cand))
                if s>best+1e-9:
                    best,cur,improved=s,float(g),True
            taus[i]=cur
        if not improved: break
    return taus,best

def choose_source_rule(oof,margin=0.01):
    P=oof[["p0","p1","p2","p3"]].values
    y=oof.y_true.values.astype(int)
    folds=oof.outer_fold.values.astype(int)
    def nested(fitter,applier):
        pred=np.zeros(len(y),dtype=int)
        pars={}
        for f in sorted(np.unique(folds)):
            m=folds==f
            par,_=fitter(P[~m],y[~m])
            pars[str(f)]=[float(x) for x in par]
            pred[m]=applier(P[m],par)
        return pred,pars
    arg=P.argmax(1)
    exp,exp_par=nested(fit_expected,apply_expected)
    cum,cum_par=nested(fit_cumulative,apply_cumulative)
    scores={"argmax":qwk(y,arg),"expected":qwk(y,exp),"cumulative":qwk(y,cum)}
    best=max(scores.values())
    simple=["argmax","expected","cumulative"]
    chosen=next(r for r in simple if scores[r]>=best-margin)
    pred={"argmax":arg,"expected":exp,"cumulative":cum}[chosen]
    detail=dict(
        margin=margin,scores=scores,best_nested_qwk=float(best),chosen=chosen,
        fold_specific_expected_thresholds=exp_par,
        fold_specific_cumulative_thresholds=cum_par,
        principle="simplest rule within 0.01 QWK of best nested OOF score",
        primary_method_specific_prediction="chosen nested source-rule output",
        harmonized_prediction="plain argmax"
    )
    return pred,arg,detail

# ---------------------------------------------------------------- metrics/bootstrap
def conf_metrics(y,p,w=None):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=int)
    if w is None: w=np.ones(len(y),dtype=float)
    w=np.asarray(w,dtype=float)
    C=np.bincount(y*K+p,weights=w,minlength=K*K).reshape(K,K).astype(float)
    n=C.sum()
    acc=float(np.trace(C)/n)
    row=C.sum(1); col=C.sum(0)
    recall=np.divide(np.diag(C),row,out=np.zeros(K),where=row>0)
    ba=float(recall.mean())
    f1=[]
    for k in range(K):
        tp=C[k,k]; fp=col[k]-tp; fn=row[k]-tp
        den=2*tp+fp+fn
        f1.append(float(2*tp/den) if den>0 else 0.0)
    mf1=float(np.mean(f1))
    wf1=float(np.sum(np.asarray(f1)*row)/n)
    mae=float(np.sum(C*np.abs(np.arange(K)[:,None]-np.arange(K)[None,:]))/n)
    # weighted kappa
    W=(np.arange(K)[:,None]-np.arange(K)[None,:])**2/((K-1)**2)
    E=np.outer(row,col)/n
    den=float((W*E).sum())
    q=float(1-(W*C).sum()/den) if den>0 else 0.0
    return dict(Accuracy=acc,BalancedAccuracy=ba,MacroF1=mf1,WeightedF1=wf1,QWK=q,MAE=mae)

def bootstrap_patient(oof,pred_cols,B=20000,seed=42):
    pats=np.array(sorted(oof.patient.unique()))
    pidx={p:i for i,p in enumerate(pats)}
    row_pi=np.array([pidx[p] for p in oof.patient],dtype=int)
    y=oof.y_true.values.astype(int)
    rng=np.random.default_rng(seed)
    metrics=["Accuracy","BalancedAccuracy","MacroF1","WeightedF1","QWK","MAE"]
    store={(pc,m):np.empty(B,dtype=float) for pc in pred_cols for m in metrics}
    for b in range(B):
        draw=rng.integers(0,len(pats),size=len(pats))
        mult=np.bincount(draw,minlength=len(pats)).astype(float)
        rw=mult[row_pi]
        for pc in pred_cols:
            mm=conf_metrics(y,oof[pc].values.astype(int),rw)
            for m in metrics:
                store[(pc,m)][b]=mm[m]
        if (b+1)%5000==0: mark(f"BOOTSTRAP {b+1}/{B}")
    rows=[]
    for pc in pred_cols:
        point=conf_metrics(y,oof[pc].values.astype(int))
        for m in metrics:
            a=store[(pc,m)]
            rows.append(dict(system=pc,metric=m,point=point[m],
                             ci_low=float(np.quantile(a,.025)),ci_high=float(np.quantile(a,.975)),
                             bootstrap_n=B,seed=seed))
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- aggregate
def aggregate(run_dir,df,fold_meta):
    chunks=[]
    for outer in range(5):
        z=np.load(run_dir/f"fold{outer}"/"test_predictions.npz",allow_pickle=True)
        P=z["P"]
        t=pd.DataFrame(dict(
            foramen_id=z["foramen_id"].astype(str),
            patient=z["patient"].astype(str),
            y_true=z["y"].astype(int),
            outer_fold=z["fold"].astype(int),
            p0=P[:,0],p1=P[:,1],p2=P[:,2],p3=P[:,3]
        ))
        chunks.append(t)
    oof=pd.concat(chunks,ignore_index=True)
    req(len(oof)==N,"OOF_N_FAIL")
    req(oof.foramen_id.is_unique,"OOF_ID_NOT_UNIQUE")
    req(oof.patient.nunique()==NPAT,"OOF_PATIENT_FAIL")
    req(oof.y_true.value_counts().sort_index().to_dict()==GRADE_COUNTS,"OOF_GRADE_FAIL")
    req(oof.outer_fold.value_counts().sort_index().to_dict()==FOLD_COUNTS,"OOF_FOLD_FAIL")
    # full id-set closure against source raw dataframe
    req(set(oof.foramen_id)==set(df.foramen_id),"OOF_SOURCE_ID_SET_MISMATCH")

    source_pred,arg_pred,rule=choose_source_rule(oof,margin=0.01)
    oof["pred_source_rule_nested"]=source_pred
    oof["pred_argmax_harmonized"]=arg_pred
    oof["expected_grade"]=oof[["p0","p1","p2","p3"]].values@np.arange(K)

    point=[]
    for syscol in ["pred_source_rule_nested","pred_argmax_harmonized"]:
        mm=conf_metrics(oof.y_true.values,oof[syscol].values)
        point.append({"system":syscol,**mm})
    point=pd.DataFrame(point)
    boot=bootstrap_patient(oof,["pred_source_rule_nested","pred_argmax_harmonized"],B=20000,seed=42)

    oof.to_csv(run_dir/"01_OOF_PREDICTIONS.csv",index=False)
    point.to_csv(run_dir/"02_POINT_ESTIMATES.csv",index=False)
    boot.to_csv(run_dir/"03_PATIENT_BOOTSTRAP20K.csv",index=False)
    (run_dir/"04_RULE_SELECTION.json").write_text(json.dumps(rule,indent=2)+"\n",encoding="utf-8")
    pd.DataFrame(fold_meta).to_csv(run_dir/"05_FOLD_TRAINING_SUMMARY.csv",index=False)

    contract=dict(
        status="FORMAL_P4_1C_SOURCE_RECONSTRUCTED_CONTEMPORARY_MATCHED_BASELINE",
        exact_historical_reproduction=False,
        reconstruction_name="source-reconstructed 2026 contemporary matched baseline",
        target="appearance-only ResNet18 + CORN",
        source_global_seed=SEED,
        final_multi_seed_exact_recipe_unavailable=True,
        reason="public repository omits final best_arch/best_reg/final campaign artifacts and exact final multi-seed list",
        recipe=RECIPE,
        roi="expert bbox expanded 40% each side by box dimension; grayscale; PIL bilinear direct square 224",
        protocol="our frozen patient-disjoint 5-fold outer with fixed-inner validation mapping 0->1,1->2,2->3,3->4,4->0",
        checkpoint="validation QWK only; outer test inference only after checkpoint freeze",
        primary_endpoint="QWK",
        primary_method_specific_inference="nested source decision-rule selection",
        harmonized_comparator="plain argmax",
        uncertainty="20,000 patient-cluster bootstrap seed42",
        published_raw_metric_direct_head_to_head_allowed=False,
        geometry_p4_reopened=False,
        training_models=5
    )
    (run_dir/"06_RECONSTRUCTION_CONTRACT.json").write_text(json.dumps(contract,indent=2)+"\n",encoding="utf-8")

    sm=point.set_index("system").to_dict("index")
    txt=[
        "P4.1-C | SOURCE-RECONSTRUCTED 2026 CONTEMPORARY MATCHED BASELINE",
        "",
        "STATUS=PASS",
        "MODELS=5/5",
        f"PATIENTS={NPAT}",
        f"ROIS={N}",
        "PRIMARY_ENDPOINT=QWK",
        "BOOTSTRAP=20000 patient-cluster seed42",
        "",
        "SOURCE_RULE_NESTED:",
        json.dumps(sm["pred_source_rule_nested"],sort_keys=True),
        "",
        "ARGMAX_HARMONIZED:",
        json.dumps(sm["pred_argmax_harmonized"],sort_keys=True),
        "",
        f"RULE_CHOSEN={rule['chosen']}",
        f"RULE_NESTED_SCORES={rule['scores']}",
        "",
        "INTERPRETATION LOCK:",
        "- This is a source-reconstructed contemporary matched baseline, not an exact historical reproduction.",
        "- Published QWK 0.806 used a different locked-test protocol and must not be numerically ranked head-to-head against this pooled OOF.",
        "- G0 remains the geometry causal control.",
        "- No geometry experiment was reopened.",
        "- Result sign/magnitude was not used to alter endpoint, recipe, or decision-rule margin."
    ]
    (run_dir/"07_HUMAN_READABLE_RESULTS.txt").write_text("\n".join(txt)+"\n",encoding="utf-8")

    status=dict(
        status="PASS",models_complete=5,patients=NPAT,rois=N,
        primary_endpoint="QWK",bootstrap_n=20000,bootstrap_seed=42,
        rule_chosen=rule["chosen"],
        source_rule_qwk=float(sm["pred_source_rule_nested"]["QWK"]),
        argmax_qwk=float(sm["pred_argmax_harmonized"]["QWK"]),
        exact_historical_reproduction=False,
        published_metric_direct_comparison_allowed=False,
        next="P4.1-D integrate contemporary baseline into frozen Paper2 evidence matrix"
    )
    (run_dir/"00_FINAL_STATUS.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")

    # compact result zip; intentionally excludes checkpoints/cache
    result_name="PAPER2_P4_1C_SOURCE_RECONSTRUCTED_RESNET18_CORN_FORMAL_RESULT.zip"
    rz=run_dir/result_name
    if rz.exists(): rz.unlink()
    members=[
        "00_FINAL_STATUS.json","01_OOF_PREDICTIONS.csv","02_POINT_ESTIMATES.csv",
        "03_PATIENT_BOOTSTRAP20K.csv","04_RULE_SELECTION.json","05_FOLD_TRAINING_SUMMARY.csv",
        "06_RECONSTRUCTION_CONTRACT.json","07_HUMAN_READABLE_RESULTS.txt",
        "roi_source40_index.csv"
    ]
    with zipfile.ZipFile(rz,"w",zipfile.ZIP_DEFLATED) as z:
        for n in members:
            p=run_dir/n
            if p.exists(): z.write(p,arcname=n)
    s=sha256(rz)
    (run_dir/(result_name+".sha256")).write_text(f"{s}  {result_name}\n",encoding="utf-8")
    (run_dir/"DONE").write_text("PASS\n",encoding="utf-8")
    mark("FORMAL_RESULT="+str(rz))
    mark("FORMAL_RESULT_SHA256="+s)
    mark("P4_1C_FORMAL=PASS")

def main():
    if len(sys.argv)!=4:
        raise SystemExit("usage: train.py RUN_DIR P4_1B_RESULT_ZIP SOURCE_REPO_ZIP")
    run_dir=Path(sys.argv[1]); p4b=Path(sys.argv[2]); repo=Path(sys.argv[3])
    run_dir.mkdir(parents=True,exist_ok=True)
    log_file=run_dir/"scientific_contract.json"
    req(sha256(p4b)==EXPECTED_P4_1B_SHA,"P4_1B_PARENT_SHA_FAIL")
    req(sha256(repo)==EXPECTED_REPO_SHA,"SOURCE_REPO_SHA_FAIL")
    req(Path(sys.executable).resolve()==Path(PYTHON_EXPECTED).resolve(),
        f"PYTHON_ENV_FAIL:{sys.executable}")
    req(torch.cuda.is_available(),"CUDA_NOT_AVAILABLE")
    device=torch.device("cuda")
    mark(f"ENV torch={torch.__version__} timm={getattr(timm,'__version__','?')} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
    mark("PARENT_SHA_GATES=PASS")

    manifest,p2f=load_manifest()
    raw=parse_raw(p2f)
    audit_patient_grade_multiset(raw,manifest)
    mark("COHORT_ALIGNMENT=PASS 469/2979 -> 468/2978; patient-grade multiset exact")

    # materialize immutable row index and source ROI cache
    raw.to_csv(run_dir/"00_SOURCE_ROW_INDEX.csv",index=False)
    cache=run_dir/"cache"/"rois_source40_224"
    idx=build_roi_cache(raw,cache)
    shutil.copy2(idx,run_dir/"roi_source40_index.csv")

    contract_pre=dict(
        parent_p4_1b_sha256=EXPECTED_P4_1B_SHA,
        source_repo_sha256=EXPECTED_REPO_SHA,
        authoritative_manifest_sha256=EXPECTED_MANIFEST_SHA,
        raw_root=str(RAW_ROOT),seed=SEED,recipe=RECIPE,
        status="PRE_REGISTERED_BEFORE_FORMAL_MODEL_FIT",
        test_selection_leakage_allowed=False,
        source_rule_margin=0.01,
        primary_endpoint="QWK"
    )
    log_file.write_text(json.dumps(contract_pre,indent=2)+"\n",encoding="utf-8")

    bank=RoiBank(cache,raw.foramen_id.tolist())
    fold_meta=[]
    for outer in range(5):
        meta=train_outer(raw,bank,outer,run_dir,device)
        fold_meta.append(meta)

    aggregate(run_dir,raw,fold_meta)

if __name__=="__main__":
    try:
        main()
    except Exception:
        # best-effort failure marker in first CLI arg run dir
        try:
            rd=Path(sys.argv[1]); rd.mkdir(parents=True,exist_ok=True)
            (rd/"FAILED").write_text(traceback.format_exc(),encoding="utf-8")
        except Exception:
            pass
        traceback.print_exc()
        raise
