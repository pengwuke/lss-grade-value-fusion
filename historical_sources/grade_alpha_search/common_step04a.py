#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, mean_absolute_error
)

ROOT = Path("/PATH/TO/RESEARCH_WORKSPACE")
PKG = ROOT / "server_packages/secondgen_step04a_firstgenbest_iq8_nohe_g1_20260821"
SCI = ROOT / "server_runs/secondgen_step04a_firstgenbest_coralmsaux_fixeddeit_iq8_nohe_g1_20260821"
CTRL = ROOT / "server_runs/secondgen_step04a_firstgenbest_control_20260821"

SOURCE54 = Path("/PATH/TO/PROJECT/experiments/baseline_v1/scripts/54_train_coral_risk_protocol_v3_full468_oof.py")
SHA54 = "b8e5360657fa467e74db846c4d03e7f5f80901aa2108e66587b04a58e89cb259"
SOURCE52 = Path("/PATH/TO/PROJECT/experiments/baseline_v1/scripts/52_train_official_deit_protocol_v3_full468_oof.py")
SHA52 = "dc8c0b4b1aafe640d7ce764c3efff984e5154430621c946ea7025c0b4d3887ae"
SOURCE393 = Path("/PATH/TO/PROJECT/experiments/baseline_v1/scripts/393_train_step14u25d_convnext_ce_differential_lr.py")
SHA393 = "5eef55010d4c019d26601a270a78be5e51b264bb396ba3a8b70871b7f4dfae04"
FORMAL_WRAPPER = ROOT / "server_runs/step04a_protocol_matched80_20_authoritative_source_preflight_20260813/current_unified_wrappers/step02b_train_convnext_corn_unified5fold.py"
SHA_WRAPPER = "5ecf4154e8dccffcb6ab9a9ee0798a7ebdc7cbf17099fd47c03f9b14ce9a7ee4"
R2_MANIFEST = ROOT / "server_runs/secondgen_step01f_r2_patient_split_403link_20260820/01_FROZEN_SOURCE_MANIFEST_WITH_PATIENT_SPLIT.csv"
SHA_R2_MANIFEST = "39371acae04df30197c19be73a2a3441379f4596f494a6651d93255ad7c8d907"
CACHE_ROOT = ROOT / "server_cache/secondgen_step01f_matched_ipng_iq8_if_20260820"

OLD_CORAL = ROOT / "server_packages/step06a7_fix2_attention_checkpoint_case_lock_20260815/oof_inputs/CORAL_MSaux_OOF.csv"
OLD_CORAL_SHA = "5e3235114612a7dad280e105e484e186e0a167f006918dd2f95e4ad6d9cec6ce"
OLD_DEIT = ROOT / "server_packages/step06a7_fix2_attention_checkpoint_case_lock_20260815/oof_inputs/Original_DeiT_OOF.csv"
OLD_DEIT_SHA = "6d8ea865ff43a8fcfad2f2701d6925e415dfd5dc9a5db237985623edb1b78e9e"
OLD_FUSION = ROOT / "server_packages/step06a7_fix2_attention_checkpoint_case_lock_20260815/oof_inputs/CORAL_MSaux_plus_fixedDeiT_OOF.csv"
OLD_FUSION_SHA = "4652e56526a808099ee36aabdc139f9a59ddb1015ba3d023dff8cedcffcb6886"

N_ROI, N_PAT, SEED = 2978, 468, 42
EXPECTED_TEST = {0:618, 1:596, 2:557, 3:616, 4:591}
EXPECTED_VAL = {0:1, 1:2, 2:3, 3:4, 4:0}

FUSION_BRANCH_WEIGHT, FUSION_DEIT_WEIGHT = 0.40, 0.60
CNN_EPOCHS, CNN_BATCH = 200, 32
CNN_LR, CNN_WD, CNN_FC_L2, CNN_DROPOUT = 0.0007723, 0.002453, 0.01, 0.5
MSAUX_WEIGHT = 0.50
DEIT_EPOCHS, DEIT_BATCH = 20, 16
DEIT_LR, DEIT_WD, DEIT_DROPOUT = 2e-5, 0.01, 0.25

METRIC_NAMES = [
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
    "qwk", "mae", "recall_normal", "recall_mild",
    "recall_moderate", "recall_severe", "moderate_to_severe_recall"
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

def set_all_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def label_col(df: pd.DataFrame) -> str:
    for c in ["severity", "grade", "label", "y_true", "target"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().all() and set(vals.astype(int).unique()).issubset({0,1,2,3}):
                return c
    raise RuntimeError(f"No 4-grade label column. columns={list(df.columns)}")

def patient_col(df: pd.DataFrame) -> str:
    for c in ["patient_id", "patient", "pid"]:
        if c in df.columns:
            return c
    raise RuntimeError("patient_id column not found")

def identity_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["protocol_v4_row_id", "row_id", "roi_id"]:
        if c in df.columns and df[c].astype(str).nunique() == len(df):
            return c
    return None

def metrics_from_arrays(y, p) -> Dict[str, float]:
    y, p = np.asarray(y, int), np.asarray(p, int)
    out = {
        "accuracy": float(accuracy_score(y,p)),
        "balanced_accuracy": float(balanced_accuracy_score(y,p)),
        "macro_f1": float(f1_score(y,p,average="macro",zero_division=0)),
        "weighted_f1": float(f1_score(y,p,average="weighted",zero_division=0)),
        "qwk": float(cohen_kappa_score(y,p,weights="quadratic")),
        "mae": float(mean_absolute_error(y,p)),
    }
    for k, name in enumerate(["normal","mild","moderate","severe"]):
        mask = y == k
        out[f"recall_{name}"] = float(np.mean(p[mask] == k)) if mask.any() else float("nan")
    ms = y >= 2
    out["moderate_to_severe_recall"] = float(np.mean(p[ms] >= 2)) if ms.any() else float("nan")
    return out

def metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    cm = np.asarray(cm, float)
    n = cm.sum()
    ts, ps, tp = cm.sum(1), cm.sum(0), np.diag(cm)
    rec = np.divide(tp, ts, out=np.full(4,np.nan), where=ts>0)
    pre = np.divide(tp, ps, out=np.zeros(4), where=ps>0)
    f1 = np.divide(2*pre*rec, pre+rec, out=np.zeros(4), where=(pre+rec)>0)
    idx = np.arange(4.0)
    w = ((idx[:,None]-idx[None,:])**2)/9.0
    obs = float((w*cm).sum())
    exp = float((w*(ts[:,None]*ps[None,:]/n)).sum())
    qwk = float("nan") if exp<=0 else 1.0-obs/exp
    mae = float((np.abs(idx[:,None]-idx[None,:])*cm).sum()/n)
    ms = float(cm[2:,2:].sum()/(ts[2]+ts[3]))
    return {
        "accuracy":float(tp.sum()/n),
        "balanced_accuracy":float(np.nanmean(rec)),
        "macro_f1":float(np.nanmean(f1)),
        "weighted_f1":float(np.nansum(f1*ts)/n),
        "qwk":qwk,"mae":mae,
        "recall_normal":float(rec[0]),"recall_mild":float(rec[1]),
        "recall_moderate":float(rec[2]),"recall_severe":float(rec[3]),
        "moderate_to_severe_recall":ms,
    }

def batch_metrics(cms: np.ndarray) -> Dict[str, np.ndarray]:
    cms = np.asarray(cms,float)
    n = cms.sum((1,2))
    ts, ps = cms.sum(2), cms.sum(1)
    tp = np.diagonal(cms,axis1=1,axis2=2)
    rec = np.divide(tp,ts,out=np.full_like(tp,np.nan),where=ts>0)
    pre = np.divide(tp,ps,out=np.zeros_like(tp),where=ps>0)
    f1 = np.divide(2*pre*rec,pre+rec,out=np.zeros_like(tp),where=(pre+rec)>0)
    idx=np.arange(4.0); w=((idx[:,None]-idx[None,:])**2)/9.0
    obs=np.einsum("bij,ij->b",cms,w)
    ex=ts[:,:,None]*ps[:,None,:]/n[:,None,None]
    exp=np.einsum("bij,ij->b",ex,w)
    qwk=1.0-np.divide(obs,exp,out=np.full_like(obs,np.nan),where=exp>0)
    mae=np.einsum("bij,ij->b",cms,np.abs(idx[:,None]-idx[None,:]))/n
    ms=np.divide(cms[:,2:,2:].sum((1,2)),ts[:,2]+ts[:,3],out=np.full(len(cms),np.nan),where=(ts[:,2]+ts[:,3])>0)
    return {
        "accuracy":tp.sum(1)/n,"balanced_accuracy":np.nanmean(rec,1),
        "macro_f1":np.nanmean(f1,1),"weighted_f1":np.nansum(f1*ts,1)/n,
        "qwk":qwk,"mae":mae,
        "recall_normal":rec[:,0],"recall_mild":rec[:,1],
        "recall_moderate":rec[:,2],"recall_severe":rec[:,3],
        "moderate_to_severe_recall":ms,
    }

def g1_resize_pad(image: np.ndarray, target: int) -> np.ndarray:
    x=np.asarray(image)
    if x.ndim not in (2,3): raise RuntimeError(f"G1 bad ndim {x.ndim}")
    h,w=x.shape[:2]
    scale=min(target/float(h),target/float(w))
    nh=max(1,int(round(h*scale))); nw=max(1,int(round(w*scale)))
    y=cv2.resize(x,(nw,nh),interpolation=cv2.INTER_LINEAR)
    top=(target-nh)//2; bottom=target-nh-top
    left=(target-nw)//2; right=target-nw-left
    y=cv2.copyMakeBorder(y,top,bottom,left,right,borderType=cv2.BORDER_REPLICATE)
    if y.shape[:2]!=(target,target): raise RuntimeError(f"G1 shape {y.shape}")
    return y

def read_iq8(path: str) -> np.ndarray:
    p=Path(path)
    x=np.load(p) if p.suffix.lower()==".npy" else cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
    if x is None: raise RuntimeError(f"Cannot decode {p}")
    x=np.asarray(x)
    if x.ndim==3:
        if x.shape[2]==1: x=x[:,:,0]
        elif x.shape[2] in (3,4):
            y=x[:,:,:3]
            if not (np.array_equal(y[:,:,0],y[:,:,1]) and np.array_equal(y[:,:,0],y[:,:,2])):
                raise RuntimeError(f"IQ8 non-gray {p}")
            x=y[:,:,0]
    if x.ndim!=2: raise RuntimeError(f"IQ8 bad shape {x.shape}")
    xf=x.astype(np.float32)
    if not np.all(np.isfinite(xf)) or xf.min()<-1e-4 or xf.max()>255.0001:
        raise RuntimeError(f"IQ8 domain invalid {p}")
    if np.max(np.abs(xf-np.rint(xf)))>1e-4:
        raise RuntimeError(f"IQ8 not integer-like {p}")
    return np.rint(xf).astype(np.uint8)

def make_native_aug(source_module, seed: int):
    import albumentations as A
    train_tf, eval_tf = source_module.build_transforms(seed)
    tn=[type(t).__name__ for t in train_tf.transforms]
    en=[type(t).__name__ for t in eval_tf.transforms]
    tail=["Resize","Normalize","ToTensorV2"]
    if tn[-3:]!=tail or en[-3:]!=tail:
        raise RuntimeError(f"Unexpected transform tail train={tn} eval={en}")
    native=A.Compose([copy.deepcopy(t) for t in train_tf.transforms[:-3]],seed=seed)
    return native,tn,en

class SelectedPreprocDataset(Dataset):
    def __init__(self, frame, branch, training, source_module, seed=SEED):
        self.frame=frame.reset_index(drop=True).copy()
        self.branch=branch; self.training=bool(training); self.label=label_col(self.frame)
        if "iq8_path" not in self.frame.columns: raise RuntimeError("iq8_path missing")
        self.native_aug=None
        if self.training:
            self.native_aug,_,_=make_native_aug(source_module,seed)

    def __len__(self): return len(self.frame)

    def __getitem__(self, idx):
        row=self.frame.iloc[int(idx)]
        img=read_iq8(str(row["iq8_path"]))
        y=int(row[self.label])
        if self.branch=="coral_msaux":
            if self.training: img=self.native_aug(image=img)["image"]
            img=g1_resize_pad(img,64)
            x=(img.astype(np.float32)/255.0-0.5)/0.5
            return torch.from_numpy(np.ascontiguousarray(x[None])).float(), torch.tensor(y).long(), torch.tensor(float(y>=2)).float(), int(idx)
        if self.branch=="deit":
            rgb=np.repeat(img[:,:,None],3,axis=2)
            if self.training: rgb=self.native_aug(image=rgb)["image"]
            rgb=g1_resize_pad(rgb,224)
            x=(rgb.astype(np.float32)/255.0-0.5)/0.5
            x=np.transpose(x,(2,0,1))
            return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.tensor(y).long(), int(idx)
        raise ValueError(self.branch)

def resolve_relative_path(v: str, csv_parent: Path) -> Optional[Path]:
    s=str(v).strip()
    if not s or s.lower() in {"nan","none"}: return None
    p=Path(s)
    tests=[p] if p.is_absolute() else [csv_parent/p,CACHE_ROOT/p,ROOT/p]
    for t in tests:
        if t.is_file(): return t.resolve()
    return None

def _norm_series(s): return s.astype(str).str.strip()

def _join_mapping(formal, cand, path_col):
    for rid in ["protocol_v4_row_id","row_id","roi_id"]:
        if rid in formal.columns and rid in cand.columns:
            cc=cand[[rid,path_col]].copy()
            if cc[rid].astype(str).duplicated().any(): continue
            lookup=dict(zip(_norm_series(cc[rid]),cc[path_col]))
            vals=_norm_series(formal[rid]).map(lookup)
            if vals.notna().all(): return vals.tolist(),f"key:{rid}"
    keysets=[
        ["patient_id","image_name","object_index"],
        ["patient_id","image_name","roi_index"],
        ["patient_id","image_basename","object_index"],
        ["patient_id","image_basename","roi_index"],
    ]
    for keys in keysets:
        if all(k in formal.columns and k in cand.columns for k in keys):
            fk=formal[keys].astype(str).agg("||".join,axis=1)
            ck=cand[keys].astype(str).agg("||".join,axis=1)
            if ck.duplicated().any(): continue
            vals=fk.map(dict(zip(ck,cand[path_col])))
            if vals.notna().all(): return vals.tolist(),"key:"+"+".join(keys)
    if len(cand)==len(formal):
        evidence=[c for c in ["patient_id","image_name","object_index","severity"] if c in cand.columns and c in formal.columns]
        if len(evidence)>=2 and all(_norm_series(cand[c]).reset_index(drop=True).equals(_norm_series(formal[c]).reset_index(drop=True)) for c in evidence):
            return cand[path_col].tolist(),"row_order_exact:"+"+".join(evidence)
    return None,None

def discover_iq8_mapping(formal):
    csvs=[R2_MANIFEST]
    roots=[
        CACHE_ROOT,
        ROOT/"server_runs/secondgen_step01f_matched_ipng_iq8_if_cache_20260820",
        ROOT/"server_runs/secondgen_step02s_fix5b_iq8_matched_20260820",
        ROOT/"server_runs/secondgen_step02s_fix5b_iq8_png8_matched_20260820",
    ]
    for sr in roots:
        if sr.exists(): csvs.extend(list(sr.rglob("*.csv"))[:100])
    runs=ROOT/"server_runs"
    if runs.exists():
        for d in runs.glob("secondgen_step0*"):
            n=d.name.lower()
            if ("step01f" in n or "step02s" in n) and "20260820" in n:
                csvs.extend(list(d.rglob("*.csv"))[:50])
    seen=set(); csvs2=[]
    for p in csvs:
        p=Path(p)
        if p.is_file() and str(p) not in seen:
            seen.add(str(p)); csvs2.append(p)
    solutions=[]; diagnostics=[]
    for csvp in csvs2:
        try: c=pd.read_csv(csvp,encoding="utf-8-sig")
        except Exception as e:
            diagnostics.append({"csv":str(csvp),"status":f"read_fail:{type(e).__name__}"}); continue
        variants=[("all",c)]
        for ac in [x for x in c.columns if x.lower() in {"arm","source_arm","intensity_arm","representation"}]:
            mask=c[ac].astype(str).str.lower().str.contains("iq8",na=False)
            if mask.any(): variants.append((f"{ac}=iq8",c.loc[mask].copy()))
        for vname,cc in variants:
            pathcols=[col for col in cc.columns if "iq8" in col.lower() and ("path" in col.lower() or "file" in col.lower())]
            if vname!="all":
                for col in cc.columns:
                    cl=col.lower()
                    if ("path" in cl or cl.endswith("_file") or cl in {"file","filename"}) and col not in pathcols:
                        pathcols.append(col)
            for pc in pathcols:
                vals,mode=_join_mapping(formal,cc,pc)
                if vals is None: continue
                resolved=[]; ok=True
                for v in vals:
                    rp=resolve_relative_path(v,csvp.parent)
                    if rp is None: ok=False; break
                    resolved.append(str(rp))
                diagnostics.append({"csv":str(csvp),"variant":vname,"path_col":pc,"alignment":mode,"rows":len(cc),"all_files_exist":ok})
                if ok and len(resolved)==N_ROI and len(set(resolved))==N_ROI:
                    mh=hashlib.sha256("\n".join(resolved).encode()).hexdigest()
                    solutions.append({"paths":resolved,"csv":str(csvp),"variant":vname,"path_col":pc,"alignment":mode,"mapping_sha":mh})
    uniq={}
    for s in solutions: uniq.setdefault(s["mapping_sha"],s)
    if len(uniq)==1: return next(iter(uniq.values())),diagnostics
    if len(uniq)>1:
        raise RuntimeError("Multiple distinct perfect IQ8 mappings; fail closed: "+ "; ".join(f"{v['csv']}:{v['path_col']}" for v in uniq.values()))
    files=[]
    if CACHE_ROOT.exists():
        for p in CACHE_ROOT.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".npy",".png",".tif",".tiff",".jpg",".jpeg"} and "iq8" in str(p).lower():
                files.append(p.resolve())
    files=sorted(set(files))
    if len(files)==N_ROI:
        by={}
        for p in files:
            by.setdefault(p.name,[]).append(p); by.setdefault(p.stem,[]).append(p)
        for csvp in csvs2:
            try: cc=pd.read_csv(csvp,encoding="utf-8-sig")
            except Exception: continue
            for col in cc.columns:
                if not any(k in col.lower() for k in ["file","name","path"]): continue
                vals,mode=_join_mapping(formal,cc,col)
                if vals is None: continue
                resolved=[]; ok=True
                for v in vals:
                    hits=by.get(Path(str(v)).name,[])
                    if len(hits)!=1: hits=by.get(Path(str(v)).stem,[])
                    if len(hits)!=1: ok=False; break
                    resolved.append(str(hits[0]))
                if ok and len(set(resolved))==N_ROI:
                    return {"paths":resolved,"csv":str(csvp),"variant":"fallback_cache_scan","path_col":col,"alignment":mode,"mapping_sha":hashlib.sha256("\n".join(resolved).encode()).hexdigest()},diagnostics
    raise RuntimeError(f"Could not uniquely resolve IQ8 2978/2978; csv_candidates={len(csvs2)} direct_solutions={len(uniq)} cache_files={len(files)}")

def detect_prob_cols(df):
    for cols in [[f"prob_{i}" for i in range(4)],[f"prob_class_{i}" for i in range(4)],[f"p{i}" for i in range(4)]]:
        if all(c in df.columns for c in cols): return cols
    by={}
    for c in df.columns:
        m=re.search(r"(?:^|_)prob(?:ability)?_?([0-3])(?:_|$)",c.lower())
        if m: by[int(m.group(1))]=c
    if len(by)==4: return [by[i] for i in range(4)]
    names=["normal","mild","moderate","severe"]; cols=[]
    for name in names:
        hits=[c for c in df.columns if "prob" in c.lower() and name in c.lower()]
        if len(hits)!=1: return []
        cols.append(hits[0])
    return cols

def align_by_row_id(reference, other):
    rid=identity_col(reference)
    if rid is None or rid not in other.columns: raise RuntimeError("unique row id required")
    oo=other.copy(); oo["_rid"]=oo[rid].astype(str)
    if oo["_rid"].duplicated().any(): raise RuntimeError("duplicate row id")
    oo=oo.set_index("_rid",drop=False)
    keys=reference[rid].astype(str)
    if not all(k in oo.index for k in keys): raise RuntimeError("OOF row-id alignment missing")
    return oo.loc[keys].reset_index(drop=True)

def archive_partial(out: Path):
    if not out.exists():
        out.mkdir(parents=True); return
    if (out/"PASS_FOLD.json").is_file(): return
    k=0
    while True:
        dst=out.with_name(out.name+("_PARTIAL_RESIDUE" if k==0 else f"_PARTIAL_RESIDUE_{k}"))
        if not dst.exists():
            out.rename(dst); out.mkdir(parents=True); return
        k+=1

def torch_load(path, map_location="cpu"):
    try: return torch.load(path,map_location=map_location,weights_only=False)
    except TypeError: return torch.load(path,map_location=map_location)
