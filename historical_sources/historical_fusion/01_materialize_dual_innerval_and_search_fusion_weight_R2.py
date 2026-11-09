#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pathlib
from contextlib import contextmanager
import ast, hashlib, importlib.util, json, shutil, sys, zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score

ROOT=Path("/PATH/TO/RESEARCH_WORKSPACE")
R5ZIP=ROOT/"server_runs/paper2_p10_c3_r5_selected_alpha_outer_eval_20261005/P10_C3_R5_SELECTED_ALPHA_OUTER_EVAL_RESULT.zip"
R4ZIP=ROOT/"server_runs/paper2_p10_c3_r4_innerval_alpha_search_20261005/P10_C3_R4_INNERVAL_ALPHA_SEARCH_RESULT.zip"
FIXED=ROOT/"server_packages/step06a2_fix2_explicit_locked_deit_20260814/Original_DeiT_native_loss_fixed_inner_OOF.csv"
SOURCE52=Path("/PATH/TO/PROJECT/experiments/baseline_v1/scripts/52_train_official_deit_protocol_v3_full468_oof.py")

R5RUN=ROOT/"server_runs/paper2_p10_c3_r5_selected_alpha_outer_eval_20261005"
R4RUN=ROOT/"server_runs/paper2_p10_c3_r4_innerval_alpha_search_20261005"
FIXED_INNER_REBUILD=ROOT/"server_runs/paper2_source_g1_corrected_fixed_inner_formal_p3_l4_20260917"
LOCK=ROOT/"server_packages/step06a7_fix5_fix2_exact_sha_final_lock_20260815"

OUT=ROOT/"server_runs/paper2_p10_c3_r6_r4_r2_crossplatform_deit_load_fusion_weight_search_20261005"
DER=OUT/"derived"; PROBS=OUT/"inner_probs"; SEARCH=OUT/"fusion_search"; SETUP=OUT/"_isolated_setup"
for p in [DER,PROBS,SEARCH,SETUP]:
    p.mkdir(parents=True,exist_ok=True)

EXP={
 "r5":"4f57e4c186f2b30ae246e6a8dc158eb802da4bfa10aae88ca435052332e63a8e",
 "r4":"ca30bf25c1b4f99872c8c1cdcbc8d929269454d13b67ecc66c6ca59d4f1d977f",
 "fixed":"6d8ea865ff43a8fcfad2f2701d6925e415dfd5dc9a5db237985623edb1b78e9e",
 "source52":"dc8c0b4b1aafe640d7ce764c3efff984e5154430621c946ea7025c0b4d3887ae",
}
SELECTED={0:0.10,1:0.125,2:0.125,3:0.10,4:0.125}
INNER={0:1,1:2,2:3,3:4,4:0}
SELECTED_CORAL_SHA={
 0:"92f3c1609d08eeb872517dbc4e9e003b24f3cfe2bf497d2c9d2763d4ec077701",
 1:"7dd0ec3ae79cf4d02145a0d4260f818cc48f4b2d49a2ec7e5457ed3dc50f8ac8",
 2:"8e2167377188f1d4e31416a7a8548e4336bc11d1ed8d05ba75f577370c1f20bf",
 3:"94304d52175e3958fc90b03e00e4dbdabca7c1e8c7f2fb6a167f9ebfd0fb71b4",
 4:"d21ab6c567f9177c758eeb00739f7295bcec6ac1499cf4faffc8959988137d89",
}
DEIT_SHA={
 0:"ac17fc555a07b6d0c00b2b56cf384b3c8d8cbf11bb452654b52d98df13e67956",
 1:"5a1e0e6be8b86980927afd015dbfa087e28b7726882e5802a9a9da7e490066ac",
 2:"c1a4f872e5b09304cf4679fe88449bb0f2b3a3d993fad98d07e802c977613ea8",
 3:"23e734e60579870fcce261c6e901d4542f4a391266f61cf103d7863e3245b461",
 4:"880a10df50ee64266714979b229b5bcafdb3dc76bd4e363aaf6f4a43145de05c",
}
DEIT_EPOCH={0:15,1:11,2:16,3:12,4:14}
GRID=[0.30,0.40,0.50,0.60,0.70]

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def req(c,m):
    if not c: raise RuntimeError(m)
def gate(n,p,e):
    a=sha(p) if p.is_file() else "MISSING"
    print(f"SHA_GATE|{n}|{a}",flush=True)
    req(a==e,f"SHA_FAIL {n} expected={e} actual={a}")
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    req(spec is not None and spec.loader is not None,f"IMPORT_SPEC_FAIL:{path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
def line_start(text,lineno):
    ls=text.splitlines(True); return sum(len(x) for x in ls[:lineno-1])
def find_dataframe(obj, expected_len):
    for attr in ["df","frame","data","source_df"]:
        v=getattr(obj,attr,None)
        if isinstance(v,pd.DataFrame) and len(v)==expected_len: return v.reset_index(drop=True).copy()
    for name in dir(obj):
        if name.startswith("_"): continue
        try: v=getattr(obj,name)
        except Exception: continue
        if isinstance(v,pd.DataFrame) and len(v)==expected_len: return v.reset_index(drop=True).copy()
    raise RuntimeError(f"DATAFRAME_NOT_FOUND dataset={type(obj)} expected_len={expected_len}")
def coral_probs(logits):
    c=torch.sigmoid(logits); c=torch.cummin(c,dim=1).values
    p=torch.stack([1-c[:,0],c[:,0]-c[:,1],c[:,1]-c[:,2],c[:,2]],dim=1)
    p=torch.clamp(p,min=0.0)
    return p/torch.clamp(p.sum(dim=1,keepdim=True),min=1e-12)

@contextmanager
def windows_path_pickle_compat():
    """Historical Step06A7 Fix5-Fix1 compatibility adapter; checkpoint bytes remain unchanged."""
    original = pathlib.WindowsPath
    try:
        pathlib.WindowsPath = pathlib.PosixPath
        yield
    finally:
        pathlib.WindowsPath = original

def torch_load_historical_windows_checkpoint(path, *, map_location):
    with windows_path_pickle_compat():
        return torch.load(path, map_location=map_location, weights_only=False)

print("="*196)
print("P10-C3-R6-R4-R2 | CROSSPLATFORM DEIT LOAD HOTFIX")
print("CAUSE=R6-R4 setup capture pointed SCI at R4 search tree and removed outer0 alpha=.10 checkpoint before load")
print("FIX=R1 isolation/recovery retained + historical WindowsPath->PosixPath torch.load adapter for Original-DeiT only")
print("NO TRAINING | NO OUTER TEST | NO OUTER-OOF WEIGHT SELECTION")
print("="*196)

for n,p,e in [("R5",R5ZIP,EXP["r5"]),("R4",R4ZIP,EXP["r4"]),("FIXED_OOF",FIXED,EXP["fixed"]),("SOURCE52",SOURCE52,EXP["source52"])]:
    gate(n,p,e)

# minimal recovery of the file damaged by failed R6-R4
r4_damaged=R4RUN/"search"/"alpha_0p1"/"coral_msaux"/"outer_0"/"best_checkpoint_min_validation_ordinal_objective.pth"
r5_src=R5RUN/"final_oof"/"coral_msaux"/"outer_0"/"best_checkpoint_min_validation_ordinal_objective.pth"
gate("R5_RECOVERY_SOURCE_OUTER0",r5_src,SELECTED_CORAL_SHA[0])
recovery={"was_missing":not r4_damaged.is_file(),"recovered":False}
if not r4_damaged.is_file():
    r4_damaged.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(r5_src,r4_damaged)
    recovery["recovered"]=True
gate("R4_RECOVERED_OUTER0",r4_damaged,SELECTED_CORAL_SHA[0])
print("R4_CHECKPOINT_RECOVERY_AUDIT="+json.dumps(recovery,sort_keys=True))

for outer in range(5):
    ck=R5RUN/"final_oof"/"coral_msaux"/f"outer_{outer}"/"best_checkpoint_min_validation_ordinal_objective.pth"
    gate(f"R5_SELECTED_CORAL_OUTER{outer}",ck,SELECTED_CORAL_SHA[outer])

def infer_coral(outer,alpha):
    tag=str(float(alpha)).replace(".","p")
    mat=R4RUN/"candidates"/f"alpha_{tag}"
    src=mat/"03_train_coral_progression_aux_INNER_SEARCH.py"
    ck=R5RUN/"final_oof"/"coral_msaux"/f"outer_{outer}"/"best_checkpoint_min_validation_ordinal_objective.pth"
    req(src.is_file(),f"CORAL_SETUP_SOURCE_MISSING outer={outer}")
    gate(f"CORAL_SELECTED_CK_OUTER{outer}",ck,SELECTED_CORAL_SHA[outer])

    text=src.read_text(encoding="utf-8")
    tree=ast.parse(text)
    main=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main"]
    req(len(main)==1,"CORAL_MAIN_MISSING")
    loops=[n for n in ast.walk(main[0]) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="epoch"]
    req(len(loops)==1,f"CORAL_EPOCH_LOOP_COUNT:{len(loops)}")
    pos=line_start(text,loops[0].lineno); indent=" "*loops[0].col_offset
    derived=text[:pos]+indent+"globals()['_R6_SETUP_CTX']=dict(locals())\n"+indent+"return\n"+text[pos:]
    compile(derived,f"setup_{outer}.py","exec")
    dp=DER/f"coral_setup_alpha_{tag}.py"
    if not dp.exists(): dp.write_text(derived,encoding="utf-8")

    iso=SETUP/f"outer_{outer}_alpha_{tag}"
    if iso.exists(): shutil.rmtree(iso)
    iso.mkdir(parents=True)

    sys.path.insert(0,str(mat))
    try: mod=load(dp,f"r6r1_coral_{outer}_{tag}")
    finally:
        try: sys.path.remove(str(mat))
        except ValueError: pass
    mod.SCI=iso
    oldargv=sys.argv[:]; sys.argv=[str(dp),"--outer-fold",str(outer)]
    try: mod.main()
    finally: sys.argv=oldargv

    ctx=getattr(mod,"_R6_SETUP_CTX",None)
    req(isinstance(ctx,dict),"CORAL_SETUP_CTX_MISSING")
    for k in ["model","val_loader","device"]: req(k in ctx,f"CORAL_CTX_KEY_MISSING:{k}")
    model,loader,device=ctx["model"],ctx["val_loader"],ctx["device"]
    gate(f"CORAL_CK_POST_SETUP_OUTER{outer}",ck,SELECTED_CORAL_SHA[outer])

    obj=torch.load(ck,map_location=device,weights_only=False)
    sd=obj["model_state_dict"] if isinstance(obj,dict) and "model_state_dict" in obj else obj
    model.load_state_dict(sd,strict=True); model=model.to(device); model.eval()
    dsdf=find_dataframe(loader.dataset,len(loader.dataset))
    probs=[]; ys=[]; idxs=[]
    with torch.no_grad():
        for batch in loader:
            x=batch[0].to(device,non_blocking=True); y=batch[1]; idx=batch[-1]
            out=model(x); logits=out[0] if isinstance(out,(tuple,list)) else out
            probs.append(coral_probs(logits).cpu().numpy())
            ys.extend(torch.as_tensor(y).cpu().numpy().astype(int).tolist())
            idxs.extend(torch.as_tensor(idx).cpu().numpy().astype(int).tolist())
    prob=np.concatenate(probs)
    sub=dsdf.iloc[np.asarray(idxs,int)].reset_index(drop=True).copy()
    req("protocol_v4_row_id" in sub.columns,"CORAL_ROW_ID_MISSING")
    sub["y_true_infer"]=np.asarray(ys,int)
    for i in range(4): sub[f"coral_p{i}"]=prob[:,i]
    return sub,sha(ck)

def infer_deit(outer,source52):
    inner=INNER[outer]
    ck=LOCK/"deit_inputs"/f"outer_{outer}"/"best_val_loss.pth"
    gate(f"DEIT_CK_OUTER{outer}",ck,DEIT_SHA[outer])
    man=FIXED_INNER_REBUILD/"training"/"g0"/"deit"/f"outer_{outer}"/"fixed_inner_trval_manifest.csv"
    req(man.is_file(),f"DEIT_MANIFEST_MISSING outer={outer}")
    df=pd.read_csv(man)
    req("protocol_v3_cv_fold" in df.columns,"DEIT_MANIFEST_FOLD_MISSING")
    path_col,label_col,_=source52.detect_columns(df)
    folds=pd.to_numeric(df["protocol_v3_cv_fold"],errors="raise").astype(int)
    val=df.loc[folds==inner].copy().reset_index(drop=True)
    req("protocol_v4_row_id" in val.columns and len(val)>0,f"DEIT_VAL_BAD outer={outer}")

    frozen=pd.read_csv(FIXED,encoding="utf-8-sig")
    req("protocol_v3_cv_fold" in frozen.columns,"FROZEN_FOLD_MISSING")
    exp=frozen.loc[pd.to_numeric(frozen["protocol_v3_cv_fold"],errors="raise").astype(int)==inner,"protocol_v4_row_id"].astype(str)
    req(set(val["protocol_v4_row_id"].astype(str))==set(exp),f"DEIT_ROWSET_DRIFT outer={outer}")

    _,eval_tf=source52.build_transforms(42)
    ds=source52.ROIDataset(val,path_col,label_col,eval_tf)
    loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=source52.timm.create_model("deit_tiny_patch16_224.fb_in1k",pretrained=False,num_classes=4,drop_rate=0.25).to(device)
    obj=torch_load_historical_windows_checkpoint(ck,map_location=device)
    sd=obj["model_state_dict"] if isinstance(obj,dict) and "model_state_dict" in obj else obj
    model.load_state_dict(sd,strict=True); model.eval()
    probs=[]; ys=[]; idxs=[]
    with torch.no_grad():
        for x,y,idx in loader:
            probs.append(torch.softmax(model(x.to(device,non_blocking=True)),dim=1).cpu().numpy())
            ys.extend(y.cpu().numpy().astype(int).tolist()); idxs.extend(idx.cpu().numpy().astype(int).tolist())
    prob=np.concatenate(probs)
    sub=val.iloc[np.asarray(idxs,int)].reset_index(drop=True).copy()
    sub["y_true_infer"]=np.asarray(ys,int)
    for i in range(4): sub[f"deit_p{i}"]=prob[:,i]
    return sub,sha(ck),sha(man)

def met(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int); g1=y==1; den=max(int(g1.sum()),1)
    return {
      "qwk":float(cohen_kappa_score(y,p,weights="quadratic")),
      "mae":float(np.mean(np.abs(y-p))),
      "balanced_accuracy":float(balanced_accuracy_score(y,p)),
      "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
      "accuracy":float(accuracy_score(y,p)),
      "g1_to0_rate":float(np.sum(g1&(p==0))/den),
    }

source52=load(SOURCE52,"r6r1_source52")
all_rows=[]; selected=[]; materialized=[]
for outer in range(5):
    coral,csha=infer_coral(outer,SELECTED[outer])
    deit,dsha,msha=infer_deit(outer,source52)
    coral["_id"]=coral["protocol_v4_row_id"].astype(str)
    deit["_id"]=deit["protocol_v4_row_id"].astype(str)
    a=coral[["_id","y_true_infer"]+[f"coral_p{i}" for i in range(4)]].merge(
      deit[["_id","y_true_infer"]+[f"deit_p{i}" for i in range(4)]],
      on="_id",suffixes=("_coral","_deit"),how="inner"
    )
    req(len(a)==len(coral)==len(deit),f"ALIGN_COUNT_FAIL outer={outer}")
    req(np.array_equal(a["y_true_infer_coral"].astype(int),a["y_true_infer_deit"].astype(int)),f"LABEL_FAIL outer={outer}")
    y=a["y_true_infer_coral"].astype(int).to_numpy()
    cp=a[[f"coral_p{i}" for i in range(4)]].to_numpy(float)
    dp=a[[f"deit_p{i}" for i in range(4)]].to_numpy(float)
    req(np.max(np.abs(cp.sum(1)-1))<1e-5 and np.max(np.abs(dp.sum(1)-1))<1e-5,f"PROB_SUM_FAIL outer={outer}")
    a["outer"]=outer; a["inner_fold"]=INNER[outer]; a["selected_alpha"]=SELECTED[outer]
    a.to_csv(PROBS/f"outer_{outer}_dual_inner_validation_probs.csv",index=False,encoding="utf-8-sig")
    mi={"outer":outer,"inner":INNER[outer],"selected_alpha":SELECTED[outer],"rows":len(a),"coral_checkpoint_sha256":csha,"deit_checkpoint_sha256":dsha,"deit_manifest_sha256":msha}
    materialized.append(mi); print("DUAL_INNER_PROB_GATE=PASS "+json.dumps(mi,sort_keys=True))

    rows=[]
    for w in GRID:
        pred=(w*cp+(1-w)*dp).argmax(1)
        rows.append({"outer":outer,"coral_weight":w,"deit_weight":1-w,**met(y,pred)})
    sdf=pd.DataFrame(rows); sdf["dist040"]=(sdf["coral_weight"]-0.40).abs()
    sdf=sdf.sort_values(["qwk","mae","balanced_accuracy","dist040","coral_weight"],ascending=[False,True,False,True,True],kind="mergesort")
    sdf.to_csv(SEARCH/f"outer_{outer}_fusion_weight_grid.csv",index=False)
    win=sdf.iloc[0]
    sel={"outer":outer,"selected_coral_weight":float(win["coral_weight"]),"selected_deit_weight":float(win["deit_weight"]),"val_qwk":float(win["qwk"]),"val_mae":float(win["mae"]),"val_balanced_accuracy":float(win["balanced_accuracy"]),"val_macro_f1":float(win["macro_f1"]),"val_g1_to0_rate":float(win["g1_to0_rate"])}
    selected.append(sel); all_rows.extend(rows); print("FUSION_WEIGHT_SELECTION|"+json.dumps(sel,sort_keys=True))

seltab=pd.DataFrame(selected)
seltab.to_csv(SEARCH/"01_SELECTED_FUSION_WEIGHT_BY_OUTER.csv",index=False)
pd.DataFrame(all_rows).to_csv(SEARCH/"02_ALL_FUSION_WEIGHT_INNER_RESULTS.csv",index=False)
boundary=int(seltab["selected_coral_weight"].isin([0.30,0.70]).sum())
counts={str(k):int(v) for k,v in seltab["selected_coral_weight"].value_counts().sort_index().to_dict().items()}
next_action=("Boundary selected in >=3/5 folds: expand inner-only fusion grid before any selected-weight outer evaluation."
             if boundary>=3 else
             "Freeze per-outer selected fusion weights; evaluate on frozen selected-alpha CORAL outer probabilities + fixed Original-DeiT OOF; then patient-cluster bootstrap versus frozen strong.")
summary={
 "status":"PASS_P10_C3_R6_R4_R2_DUAL_INNERVAL_FUSION_WEIGHT_SEARCH",
 "hotfix":{"parent_r1_script_sha256":"23ea4320433f90ef895fb92aecaf6fd9ad30af7986ad26ad21277fef65080860","r1_failure_mode":"Original-DeiT checkpoint contains pickled WindowsPath; Linux torch.load raised NotImplementedError","repair":"retain R1 isolated setup/recovery and temporarily map pathlib.WindowsPath to pathlib.PosixPath only during DeiT torch.load","checkpoint_bytes_modified":False,"recovery":recovery},
 "fusion_grid":GRID,
 "selection_rule":"max QWK; lower MAE; higher BA; closer to original 0.40",
 "selection_counts":counts,
 "boundary_selected_folds":boundary,
 "selected":selected,
 "materialized":materialized,
 "training_performed":False,
 "outer_test_opened":False,
 "outer_oof_used_for_weight_selection":False,
 "next_action":next_action,
}
(OUT/"00_P10_C3_R6_R4_R2_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

result=OUT/"P10_C3_R6_R4_R2_DUAL_INNERVAL_FUSION_WEIGHT_SEARCH_RESULT.zip"
with zipfile.ZipFile(result,"w",zipfile.ZIP_DEFLATED) as z:
    for base in [PROBS,SEARCH]:
        for p in sorted(base.iterdir()):
            if p.is_file(): z.write(p,f"{base.name}/{p.name}")
    z.write(OUT/"00_P10_C3_R6_R4_R2_SUMMARY.json","00_P10_C3_R6_R4_R2_SUMMARY.json")
with zipfile.ZipFile(result) as z: req(z.testzip() is None,"ZIP_TEST_FAIL")
print("P10_C3_R6_R4_R2_DUAL_INNERVAL_FUSION_WEIGHT_SEARCH=PASS")
print("R4_CHECKPOINT_RECOVERY="+json.dumps(recovery,sort_keys=True))
print("SELECTION_COUNTS="+json.dumps(counts,sort_keys=True))
print(f"BOUNDARY_SELECTED_FOLDS={boundary}")
print("TRAINING_PERFORMED=False")
print("OUTER_TEST_OPENED=False")
print("OUTER_OOF_USED_FOR_WEIGHT_SELECTION=False")
print("NEXT_ACTION="+next_action)
print(f"RESULT_ZIP={result}")
print(f"RESULT_ZIP_SHA256={sha(result)}")
