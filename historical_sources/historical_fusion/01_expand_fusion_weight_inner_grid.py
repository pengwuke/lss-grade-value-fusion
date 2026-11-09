#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import hashlib, io, json, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score

ROOT=Path("/PATH/TO/RESEARCH_WORKSPACE")
PARENT_RUN=ROOT/"server_runs/paper2_p10_c3_r6_r4_r2_crossplatform_deit_load_fusion_weight_search_20261005"
PARENT=PARENT_RUN/"P10_C3_R6_R4_R2_DUAL_INNERVAL_FUSION_WEIGHT_SEARCH_RESULT.zip"
OUT=ROOT/"server_runs/paper2_p10_c3_r6_r5_fusion_weight_boundary_expansion_20261005"
OUT.mkdir(parents=True,exist_ok=True)

EXPECTED_PARENT="bfdaff98e1bae26afecee85e7478d1a1ba20295e5a95a0f1b72803463ba7ed36"
GRID=[0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def req(c,m):
    if not c: raise RuntimeError(m)

def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int)
    g1=y==1; den=max(int(g1.sum()),1)
    return {
      "qwk":float(cohen_kappa_score(y,p,weights="quadratic")),
      "mae":float(np.mean(np.abs(y-p))),
      "balanced_accuracy":float(balanced_accuracy_score(y,p)),
      "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
      "accuracy":float(accuracy_score(y,p)),
      "g1_to0_rate":float(np.sum(g1&(p==0))/den),
    }

print("="*190)
print("P10-C3-R6-R5 | FUSION-WEIGHT INNER-VALIDATION BOUNDARY EXPANSION")
print("PARENT GRID=0.30..0.70 | BOUNDARY 0.70 SELECTED IN 3/5 FOLDS")
print("EXPANDED GRID=0.30..1.00 STEP 0.10 | REUSE FROZEN DUAL INNER-VALIDATION PROBABILITIES")
print("NO TRAINING | NO INFERENCE | NO OUTER TEST | NO OUTER-OOF WEIGHT SELECTION")
print("="*190)

actual=sha(PARENT) if PARENT.is_file() else "MISSING"
print(f"PARENT_SHA_GATE expected={EXPECTED_PARENT} actual={actual}")
req(actual==EXPECTED_PARENT,"PARENT_SHA_MISMATCH")

with zipfile.ZipFile(PARENT) as z:
    names=set(z.namelist())
    req("00_P10_C3_R6_R4_R2_SUMMARY.json" in names,"PARENT_SUMMARY_MISSING")
    parent_summary=json.loads(z.read("00_P10_C3_R6_R4_R2_SUMMARY.json"))
    req(parent_summary.get("outer_oof_used_for_weight_selection") is False,"PARENT_OUTER_OOF_SELECTION_DRIFT")
    req(parent_summary.get("outer_test_opened") is False,"PARENT_OUTER_TEST_DRIFT")
    req(int(parent_summary.get("boundary_selected_folds",-1))==3,"PARENT_BOUNDARY_COUNT_DRIFT")
    old_counts=parent_summary.get("selection_counts")
    print("PARENT_SELECTION_COUNTS="+json.dumps(old_counts,sort_keys=True))

    all_rows=[]
    selections=[]
    fold_deltas=[]
    for outer in range(5):
        member=f"inner_probs/outer_{outer}_dual_inner_validation_probs.csv"
        req(member in names,f"PARENT_INNER_PROB_MISSING:{member}")
        df=pd.read_csv(io.BytesIO(z.read(member)),encoding="utf-8-sig")
        required=["y_true_infer_coral","y_true_infer_deit"]+[f"coral_p{i}" for i in range(4)]+[f"deit_p{i}" for i in range(4)]
        for c in required:
            req(c in df.columns,f"COLUMN_MISSING outer={outer} col={c}")
        req(len(df)>0,f"EMPTY_INNER_PROBS outer={outer}")
        y1=df["y_true_infer_coral"].astype(int).to_numpy()
        y2=df["y_true_infer_deit"].astype(int).to_numpy()
        req(np.array_equal(y1,y2),f"LABEL_ALIGNMENT_FAIL outer={outer}")
        cp=df[[f"coral_p{i}" for i in range(4)]].to_numpy(float)
        dp=df[[f"deit_p{i}" for i in range(4)]].to_numpy(float)
        req(np.max(np.abs(cp.sum(1)-1))<1e-5,f"CORAL_PROB_SUM_FAIL outer={outer}")
        req(np.max(np.abs(dp.sum(1)-1))<1e-5,f"DEIT_PROB_SUM_FAIL outer={outer}")
        print(f"DUAL_INNER_REUSE_GATE=PASS outer={outer} rows={len(df)}")

        rows=[]
        for w in GRID:
            pred=(w*cp+(1.0-w)*dp).argmax(1)
            m=metrics(y1,pred)
            rows.append({"outer":outer,"coral_weight":w,"deit_weight":1.0-w,**m})
        sdf=pd.DataFrame(rows)
        sdf["distance_from_fixed_040"]=(sdf["coral_weight"]-0.40).abs()
        sdf=sdf.sort_values(
            ["qwk","mae","balanced_accuracy","distance_from_fixed_040","coral_weight"],
            ascending=[False,True,False,True,True],
            kind="mergesort"
        )
        sdf.to_csv(OUT/f"outer_{outer}_expanded_fusion_weight_grid.csv",index=False)

        win=sdf.iloc[0]
        base=sdf.loc[np.isclose(sdf["coral_weight"],0.40)].iloc[0]
        pure=sdf.loc[np.isclose(sdf["coral_weight"],1.00)].iloc[0]
        sel={
            "outer":outer,
            "selected_coral_weight":float(win["coral_weight"]),
            "selected_deit_weight":float(win["deit_weight"]),
            "val_qwk":float(win["qwk"]),
            "val_mae":float(win["mae"]),
            "val_balanced_accuracy":float(win["balanced_accuracy"]),
            "val_macro_f1":float(win["macro_f1"]),
            "val_g1_to0_rate":float(win["g1_to0_rate"]),
            "delta_qwk_vs_fixed_040":float(win["qwk"]-base["qwk"]),
            "delta_mae_vs_fixed_040":float(win["mae"]-base["mae"]),
            "delta_qwk_vs_coral_only":float(win["qwk"]-pure["qwk"]),
            "delta_mae_vs_coral_only":float(win["mae"]-pure["mae"]),
        }
        selections.append(sel)
        all_rows.extend(rows)
        fold_deltas.append({
            "outer":outer,
            "winner_weight":float(win["coral_weight"]),
            "winner_minus_fixed040_qwk":float(win["qwk"]-base["qwk"]),
            "winner_minus_fixed040_mae":float(win["mae"]-base["mae"]),
            "winner_minus_coralonly_qwk":float(win["qwk"]-pure["qwk"]),
            "winner_minus_coralonly_mae":float(win["mae"]-pure["mae"]),
        })
        print("EXPANDED_FUSION_WEIGHT_SELECTION|"+json.dumps(sel,sort_keys=True))

sel=pd.DataFrame(selections)
sel.to_csv(OUT/"01_SELECTED_FUSION_WEIGHT_BY_OUTER_EXPANDED.csv",index=False)
pd.DataFrame(all_rows).to_csv(OUT/"02_ALL_EXPANDED_FUSION_WEIGHT_INNER_RESULTS.csv",index=False)
pd.DataFrame(fold_deltas).to_csv(OUT/"03_INNER_WINNER_DELTAS.csv",index=False)

counts={str(k):int(v) for k,v in sel["selected_coral_weight"].value_counts().sort_index().to_dict().items()}
endpoint_100=int(np.isclose(sel["selected_coral_weight"],1.00).sum())
upper_090_100=int((sel["selected_coral_weight"]>=0.90-1e-12).sum())
mean_dq=float(sel["delta_qwk_vs_fixed_040"].mean())
mean_dm=float(sel["delta_mae_vs_fixed_040"].mean())

if endpoint_100>=3:
    interpretation="CORAL-only endpoint selected in >=3/5 folds; fixed-DeiT fusion contributes no stable inner-validation benefit for most folds."
    next_action="Freeze per-outer expanded-grid selections including the meaningful CORAL-only endpoint, then evaluate selected routes on frozen outer OOF exactly once. Compare both selected-weight system and tuned-CORAL-only route against frozen strong with patient-cluster bootstrap."
else:
    interpretation="Expanded grid has an interior/upper-but-not-majority-endpoint solution; fusion remains potentially useful."
    next_action="Freeze the per-outer expanded-grid weights and evaluate selected-weight fusion on frozen outer OOF exactly once; then patient-cluster bootstrap versus frozen strong and versus fixed 0.40/0.60 tuned-alpha fusion."

summary={
    "status":"PASS_P10_C3_R6_R5_FUSION_WEIGHT_BOUNDARY_EXPANSION",
    "parent_result_sha256":EXPECTED_PARENT,
    "reason":"previous upper boundary coral_weight=0.70 selected in 3/5 folds",
    "expanded_grid":GRID,
    "selection_rule":"max validation QWK; tie lower MAE; higher balanced accuracy; closer to original CORAL weight 0.40",
    "selection_counts":counts,
    "selected":selections,
    "coral_only_endpoint_selected_folds":endpoint_100,
    "weight_ge_0p90_selected_folds":upper_090_100,
    "mean_inner_delta_qwk_vs_fixed_040":mean_dq,
    "mean_inner_delta_mae_vs_fixed_040":mean_dm,
    "interpretation":interpretation,
    "training_performed":False,
    "inference_performed":False,
    "outer_test_opened":False,
    "outer_oof_used_for_weight_selection":False,
    "next_action":next_action,
}
(OUT/"00_P10_C3_R6_R5_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

result=OUT/"P10_C3_R6_R5_FUSION_WEIGHT_BOUNDARY_EXPANSION_RESULT.zip"
with zipfile.ZipFile(result,"w",zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p!=result:
            z.write(p,p.name)
with zipfile.ZipFile(result) as z:
    req(z.testzip() is None,"RESULT_ZIP_TEST_FAIL")

print("P10_C3_R6_R5_FUSION_WEIGHT_BOUNDARY_EXPANSION=PASS")
print("SELECTION_COUNTS="+json.dumps(counts,sort_keys=True))
print(f"CORAL_ONLY_ENDPOINT_SELECTED_FOLDS={endpoint_100}")
print(f"WEIGHT_GE_0P90_SELECTED_FOLDS={upper_090_100}")
print(f"MEAN_INNER_DELTA_QWK_VS_FIXED040={mean_dq:.9f}")
print(f"MEAN_INNER_DELTA_MAE_VS_FIXED040={mean_dm:.9f}")
print("TRAINING_PERFORMED=False")
print("INFERENCE_PERFORMED=False")
print("OUTER_TEST_OPENED=False")
print("OUTER_OOF_USED_FOR_WEIGHT_SELECTION=False")
print("NEXT_ACTION="+next_action)
print(f"RESULT_ZIP={result}")
print(f"RESULT_ZIP_SHA256={sha(result)}")
