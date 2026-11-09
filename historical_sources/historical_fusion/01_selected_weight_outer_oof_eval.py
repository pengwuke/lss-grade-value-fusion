#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import hashlib, io, json, re, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score

ROOT=Path("/PATH/TO/RESEARCH_WORKSPACE")
R6R5=ROOT/"server_runs/paper2_p10_c3_r6_r5_fusion_weight_boundary_expansion_20261005/P10_C3_R6_R5_FUSION_WEIGHT_BOUNDARY_EXPANSION_RESULT.zip"
R5=ROOT/"server_runs/paper2_p10_c3_r5_selected_alpha_outer_eval_20261005/P10_C3_R5_SELECTED_ALPHA_OUTER_EVAL_RESULT.zip"
R5RUN=ROOT/"server_runs/paper2_p10_c3_r5_selected_alpha_outer_eval_20261005"
FIXED_DEIT=ROOT/"server_packages/step06a2_fix2_explicit_locked_deit_20260814/Original_DeiT_native_loss_fixed_inner_OOF.csv"
STRONG=ROOT/"server_runs/secondgen_step04a_firstgenbest_coralmsaux_fixeddeit_iq8_nohe_g1_20260821/analysis/02_NEW_CORAL_MSAUX_POOLED_OOF.csv"
if not STRONG.exists():
    STRONG=ROOT/"server_packages/secondgen_step04a_firstgenbest_iq8_nohe_g1_20260821/analysis/02_NEW_CORAL_MSAUX_POOLED_OOF.csv"
B2R1=ROOT/"server_runs/paper2_p10_b2_r1_selective_rescue_ceiling_hard103_audit_20261004/P10_B2_R1_SELECTIVE_RESCUE_CEILING_HARD103_AUDIT_RESULT.zip"

OUT=ROOT/"server_runs/paper2_p10_c3_r6_r6_selected_weight_outer_oof_eval_20261005"
OUT.mkdir(parents=True,exist_ok=True)

EXP={
 "r6r5":"08d0316826dff5fcbccd60a98776b53962f5e675ecac36287014be2463d805d4",
 "r5":"4f57e4c186f2b30ae246e6a8dc158eb802da4bfa10aae88ca435052332e63a8e",
 "deit":"6d8ea865ff43a8fcfad2f2701d6925e415dfd5dc9a5db237985623edb1b78e9e",
 "strong":"35e93a1a2deab1f72e8e7d704383c98058e1a27abdf43fee31fd64a3d9df1074",
 "b2r1":"0c9f29f276e5c1f21bf1823b298c2b4380c8b5938449d18f82811a15e5dd71d6",
}
EXPECTED_WEIGHTS={0:0.7,1:0.6,2:0.6,3:0.9,4:0.9}
FROZEN_QWK=0.7973902091308339
FROZEN_MAE=0.2834116856950974
R5_FIXED_QWK=0.8033509177355884
R5_FIXED_MAE=0.27602417730020146
R5_CORAL_QWK=0.7974260406320567
R5_CORAL_MAE=0.26326393552719946
N_BOOT=20000
SEED=20261005

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

def detect_id(df):
    for c in ["protocol_v4_row_id","protocol_v3_row_id","row_id","roi_id"]:
        if c in df.columns: return c
    cs=[c for c in df.columns if "row_id" in c.lower() or c.lower().endswith("_id")]
    for c in cs:
        if df[c].nunique()==len(df): return c
    raise RuntimeError(f"ID_COLUMN_NOT_FOUND columns={list(df.columns)}")

def detect_prob_cols(df):
    prefs=[
      [f"prob_{i}" for i in range(4)],
      [f"prob{i}" for i in range(4)],
      [f"p{i}" for i in range(4)],
      [f"class_{i}_prob" for i in range(4)],
      ["prob_0_normal","prob_1_mild","prob_2_moderate","prob_3_severe"],
    ]
    for cols in prefs:
        if all(c in df.columns for c in cols): return cols
    hits=[]
    for c in df.columns:
        m=re.search(r"(\d)$",c)
        if m and "prob" in c.lower() and int(m.group(1)) in range(4):
            hits.append((int(m.group(1)),c))
    if len({i for i,c in hits})==4:
        d={i:c for i,c in hits}; return [d[i] for i in range(4)]
    raise RuntimeError(f"PROB_COLS_NOT_FOUND columns={list(df.columns)}")

def detect_patient(df):
    for c in ["patient_id","patient","PatientID","patientid"]:
        if c in df.columns: return c
    cs=[c for c in df.columns if "patient" in c.lower()]
    if cs: return cs[0]
    raise RuntimeError("PATIENT_COLUMN_NOT_FOUND")

def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int)
    g1=y==1; g1n=max(int(g1.sum()),1)
    return {
      "accuracy":float(accuracy_score(y,p)),
      "balanced_accuracy":float(balanced_accuracy_score(y,p)),
      "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
      "weighted_f1":float(f1_score(y,p,average="weighted",zero_division=0)),
      "mae":float(np.mean(np.abs(y-p))),
      "qwk":float(cohen_kappa_score(y,p,weights="quadratic")),
      "grade1_recall":float(np.sum(g1&(p==1))/g1n),
      "grade1_to0_count":int(np.sum(g1&(p==0))),
      "grade1_to0_rate":float(np.sum(g1&(p==0))/g1n),
    }

def qwk_from_conf(conf):
    active=np.where((conf.sum(axis=0)+conf.sum(axis=1))>0)[0]
    if len(active)<=1: return float("nan")
    c=conf[np.ix_(active,active)].astype(float); n=c.sum()
    if n<=0: return float("nan")
    hist1=c.sum(axis=1); hist2=c.sum(axis=0)
    e=np.outer(hist1,hist2)/n; k=len(active)
    ii=np.arange(k)[:,None]; jj=np.arange(k)[None,:]
    w=((ii-jj)**2)/float((k-1)**2)
    den=float((w*e).sum()); num=float((w*c).sum())
    return 1.0-num/den if den>0 else float("nan")

def cluster_tables(patient,y,p):
    pats=pd.Series(patient).astype(str).to_numpy()
    uniq=np.array(sorted(pd.unique(pats))); mp={v:i for i,v in enumerate(uniq)}
    P=len(uniq)
    conf=np.zeros((P,4,4),dtype=np.int32)
    n=np.zeros(P,dtype=np.int32); cor=np.zeros(P,dtype=np.int32); ae=np.zeros(P,dtype=np.int32)
    g1den=np.zeros(P,dtype=np.int32); g1to0=np.zeros(P,dtype=np.int32)
    for pat,yy,pp in zip(pats,y,p):
        i=mp[pat]; yy=int(yy); pp=int(pp)
        conf[i,yy,pp]+=1; n[i]+=1; cor[i]+=int(yy==pp); ae[i]+=abs(yy-pp)
        if yy==1:
            g1den[i]+=1; g1to0[i]+=int(pp==0)
    return uniq,{"conf":conf,"n":n,"cor":cor,"ae":ae,"g1den":g1den,"g1to0":g1to0}

def bootstrap_contrast(patient,y,base,cand,name):
    up,btab=cluster_tables(patient,y,base); up2,ctab=cluster_tables(patient,y,cand)
    req(np.array_equal(up,up2),"BOOT_PATIENT_ALIGNMENT_FAIL")
    P=len(up); rng=np.random.default_rng(SEED)
    vals={k:np.empty(N_BOOT,float) for k in ["qwk","mae","accuracy","g1_to0"]}
    for b in range(N_BOOT):
        cnt=rng.multinomial(P,np.full(P,1.0/P))
        def one(tab):
            conf=np.tensordot(cnt,tab["conf"],axes=(0,0))
            nn=float(np.dot(cnt,tab["n"])); cor=float(np.dot(cnt,tab["cor"])); ae=float(np.dot(cnt,tab["ae"]))
            gd=float(np.dot(cnt,tab["g1den"])); gz=float(np.dot(cnt,tab["g1to0"]))
            return {"qwk":qwk_from_conf(conf),"mae":ae/nn,"accuracy":cor/nn,"g1_to0":gz/gd if gd>0 else float("nan")}
        bm=one(btab); cm=one(ctab)
        for k in vals: vals[k][b]=cm[k]-bm[k]
        if (b+1)%2000==0:
            print(f"BOOTSTRAP_PROGRESS {name} {b+1}/{N_BOOT}",flush=True)
    mb=metrics(y,base); mc=metrics(y,cand); out={"contrast":name}
    for metric,key in [("qwk","qwk"),("mae","mae"),("accuracy","accuracy"),("g1_to0","grade1_to0_rate")]:
        arr=vals[metric]
        out[f"point_delta_{metric if metric!='g1_to0' else 'g1_to0_rate'}"]=float(mc[key]-mb[key])
        out[f"delta_{metric if metric!='g1_to0' else 'g1_to0'}_ci_low"]=float(np.nanpercentile(arr,2.5))
        out[f"delta_{metric if metric!='g1_to0' else 'g1_to0'}_ci_high"]=float(np.nanpercentile(arr,97.5))
    print("BOOTSTRAP_CONTRAST|"+json.dumps(out,sort_keys=True),flush=True)
    return out

def local_account(y,base,cand,hard,system):
    y=np.asarray(y,int); b=np.asarray(base,int); c=np.asarray(cand,int); h=np.asarray(hard,bool)
    db=np.abs(y-b); dc=np.abs(y-c)
    return {
      "system":system,
      "distance_benefit_rows":int(np.sum(dc<db)),
      "distance_harm_rows":int(np.sum(dc>db)),
      "distance_saved_units":int(np.maximum(db-dc,0).sum()),
      "distance_added_units":int(np.maximum(dc-db,0).sum()),
      "grade1_errors_corrected":int(np.sum((y==1)&(b!=1)&(c==1))),
      "grade1_to0_corrected":int(np.sum((y==1)&(b==0)&(c==1))),
      "hard103_correct":int(np.sum(h&(c==y))),
      "hard103_distance_benefit":int(np.sum(h&(dc<db))),
      "hard103_distance_harm":int(np.sum(h&(dc>db))),
    }

print("="*192)
print("P10-C3-R6-R6 | INNER-SELECTED FUSION WEIGHT OUTER-OOF SYSTEM EVALUATION")
print("WEIGHTS BY OUTER=[0.7,0.6,0.6,0.9,0.9] CORAL; WEIGHT SELECTION USED INNER VALIDATION ONLY")
print("NO TRAINING | NO NEW MODEL INFERENCE | ALGEBRAIC EVALUATION ON FROZEN OUTER OOF PROBABILITIES")
print("="*192)
for n,p,e in [("R6R5",R6R5,EXP["r6r5"]),("R5",R5,EXP["r5"]),("FIXED_DEIT",FIXED_DEIT,EXP["deit"]),("STRONG_CORAL",STRONG,EXP["strong"]),("B2R1",B2R1,EXP["b2r1"])]:
    gate(n,p,e)

with zipfile.ZipFile(R6R5) as z:
    req("00_P10_C3_R6_R5_SUMMARY.json" in z.namelist(),"R6R5_SUMMARY_MISSING")
    s=json.loads(z.read("00_P10_C3_R6_R5_SUMMARY.json"))
    req(s.get("outer_oof_used_for_weight_selection") is False,"WEIGHT_SELECTION_OUTER_OOF_DRIFT")
    req(s.get("outer_test_opened") is False,"R6R5_OUTER_TEST_DRIFT")
    got={int(x["outer"]):float(x["selected_coral_weight"]) for x in s["selected"]}
    for k,v in EXPECTED_WEIGHTS.items():
        req(abs(got[k]-v)<1e-12,f"WEIGHT_FREEZE_DRIFT outer={k} expected={v} got={got[k]}")
print("FUSION_WEIGHT_FREEZE_GATE=PASS "+json.dumps(EXPECTED_WEIGHTS,sort_keys=True))

# Candidate selected-alpha CORAL frozen outer OOF.
folds=[]
for outer in range(5):
    pred=R5RUN/"final_oof"/"coral_msaux"/f"outer_{outer}"/"outer_test_predictions.csv"
    req(pred.is_file(),f"CAND_OUTER_PRED_MISSING outer={outer}")
    d=pd.read_csv(pred,encoding="utf-8-sig")
    d["_outer_eval"]=outer
    folds.append(d)
    print(f"CAND_OUTER_GATE=PASS outer={outer} rows={len(d)}")
cand=pd.concat(folds,ignore_index=True)
cid=detect_id(cand); cp=detect_prob_cols(cand)
req(len(cand)==2978 and cand[cid].nunique()==2978,f"CAND_POOL_FAIL rows={len(cand)} unique={cand[cid].nunique()}")
req("y_true" in cand.columns and "y_pred" in cand.columns,"CAND_Y_MISSING")

deit=pd.read_csv(FIXED_DEIT,encoding="utf-8-sig"); did=detect_id(deit); dp=detect_prob_cols(deit)
strong=pd.read_csv(STRONG,encoding="utf-8-sig"); sid=detect_id(strong)

with zipfile.ZipFile(B2R1) as z:
    hits=[n for n in z.namelist() if n.endswith("08_RESPONDER_MANIFEST.csv")]
    req(len(hits)==1,f"RESPONDER_MANIFEST_COUNT={len(hits)}")
    resp=pd.read_csv(io.BytesIO(z.read(hits[0])),encoding="utf-8-sig")
rid=detect_id(resp)
ycol="y_true" if "y_true" in resp.columns else "grade"
pred_auth="pred_frozen_strong_authority" if "pred_frozen_strong_authority" in resp.columns else [c for c in resp.columns if "pred" in c.lower() and "frozen" in c.lower() and "strong" in c.lower()][0]
hardcol="hard103" if "hard103" in resp.columns else [c for c in resp.columns if "hard103" in c.lower()][0]
patcol=detect_patient(resp)

for df,col in [(cand,cid),(deit,did),(strong,sid),(resp,rid)]:
    df["_id"]=df[col].astype(str)

base=resp[["_id",ycol,pred_auth,hardcol,patcol]].rename(columns={ycol:"y_true_auth",pred_auth:"pred_frozen",hardcol:"hard103",patcol:"patient_id"})
base=base.merge(cand[["_id","y_true","y_pred","_outer_eval"]+cp].rename(columns={"y_true":"y_true_cand","y_pred":"pred_cand",**{c:f"cand_p{i}" for i,c in enumerate(cp)}}),on="_id",how="inner")
base=base.merge(deit[["_id"]+([ "outer_fold" ] if "outer_fold" in deit.columns else [])+dp].rename(columns={c:f"deit_p{i}" for i,c in enumerate(dp)}),on="_id",how="inner")
req(len(base)==2978 and base["_id"].nunique()==2978,f"ALIGNMENT_FAIL rows={len(base)}")
req(np.array_equal(base["y_true_auth"].astype(int),base["y_true_cand"].astype(int)),"Y_ALIGNMENT_FAIL")
if "outer_fold" in base.columns:
    req(np.array_equal(base["_outer_eval"].astype(int),pd.to_numeric(base["outer_fold"],errors="raise").astype(int)),"OUTER_FOLD_ALIGNMENT_FAIL")

y=base["y_true_auth"].astype(int).to_numpy()
patient=base["patient_id"].astype(str).to_numpy()
hard=base["hard103"].astype(bool).to_numpy()
pred_frozen=base["pred_frozen"].astype(int).to_numpy()
pred_coral=base["pred_cand"].astype(int).to_numpy()
cand_prob=base[[f"cand_p{i}" for i in range(4)]].to_numpy(float)
deit_prob=base[[f"deit_p{i}" for i in range(4)]].to_numpy(float)
req(np.max(np.abs(cand_prob.sum(1)-1))<1e-4,"CAND_PROB_SUM_FAIL")
req(np.max(np.abs(deit_prob.sum(1)-1))<1e-4,"DEIT_PROB_SUM_FAIL")

fixed_prob=0.40*cand_prob+0.60*deit_prob
pred_fixed=fixed_prob.argmax(1).astype(int)
weights=np.array([EXPECTED_WEIGHTS[int(o)] for o in base["_outer_eval"].astype(int)],float)
selected_prob=weights[:,None]*cand_prob+(1.0-weights[:,None])*deit_prob
pred_selected=selected_prob.argmax(1).astype(int)

m_frozen=metrics(y,pred_frozen)
m_coral=metrics(y,pred_coral)
m_fixed=metrics(y,pred_fixed)
m_selected=metrics(y,pred_selected)
req(abs(m_frozen["qwk"]-FROZEN_QWK)<1e-9 and abs(m_frozen["mae"]-FROZEN_MAE)<1e-12,f"FROZEN_IDENTITY_FAIL:{m_frozen}")
req(abs(m_fixed["qwk"]-R5_FIXED_QWK)<1e-9 and abs(m_fixed["mae"]-R5_FIXED_MAE)<1e-12,f"R5_FIXED_IDENTITY_FAIL:{m_fixed}")
req(abs(m_coral["qwk"]-R5_CORAL_QWK)<1e-9 and abs(m_coral["mae"]-R5_CORAL_MAE)<1e-12,f"R5_CORAL_IDENTITY_FAIL:{m_coral}")
print("BASELINE_AND_PARENT_METRIC_IDENTITY_GATE=PASS")
print("POINT|selected_weight_fusion|"+json.dumps(m_selected,sort_keys=True))
print("POINT|fixed040_tunedalpha_fusion|"+json.dumps(m_fixed,sort_keys=True))
print("POINT|tuned_progression_coral|"+json.dumps(m_coral,sort_keys=True))
print("POINT|frozen_strong|"+json.dumps(m_frozen,sort_keys=True))

locals_=[
    local_account(y,pred_frozen,pred_selected,hard,"selected_weight_vs_frozen_strong"),
    local_account(y,pred_fixed,pred_selected,hard,"selected_weight_vs_fixed040_tunedalpha"),
    local_account(y,pred_coral,pred_selected,hard,"selected_weight_vs_tuned_coral"),
]
for x in locals_: print("LOCAL_ACCOUNT|"+json.dumps(x,sort_keys=True))

boots=[
    bootstrap_contrast(patient,y,pred_frozen,pred_selected,"selected_weight_fusion_minus_frozen_strong"),
    bootstrap_contrast(patient,y,pred_fixed,pred_selected,"selected_weight_fusion_minus_fixed040_tunedalpha_fusion"),
    bootstrap_contrast(patient,y,pred_coral,pred_selected,"selected_weight_fusion_minus_tuned_progression_coral"),
]

b_frozen,b_fixed,b_coral=boots
flags={
 "selected_vs_frozen_qwk_gain_established":b_frozen["delta_qwk_ci_low"]>0,
 "selected_vs_frozen_mae_improvement_established":b_frozen["delta_mae_ci_high"]<0,
 "selected_vs_fixed_qwk_gain_established":b_fixed["delta_qwk_ci_low"]>0,
 "selected_vs_fixed_mae_improvement_established":b_fixed["delta_mae_ci_high"]<0,
 "selected_vs_coral_qwk_gain_established":b_coral["delta_qwk_ci_low"]>0,
 "selected_vs_coral_mae_improvement_established":b_coral["delta_mae_ci_high"]<0,
}
flags["fusion_subsystem_replacement_supported"]=bool(
    flags["selected_vs_fixed_qwk_gain_established"] and
    (flags["selected_vs_fixed_mae_improvement_established"] or b_fixed["point_delta_mae"]<=0)
)
flags["system_promotion_vs_frozen_supported"]=bool(
    flags["selected_vs_frozen_qwk_gain_established"] and
    (flags["selected_vs_frozen_mae_improvement_established"] or b_frozen["point_delta_mae"]<=0)
)
print("DECISION_FLAGS="+json.dumps(flags,sort_keys=True))

out=base[["_id","patient_id","y_true_auth","_outer_eval"]].copy()
out["coral_weight_selected"]=weights
out["pred_selected_weight_fusion"]=pred_selected
out["pred_fixed040_tunedalpha_fusion"]=pred_fixed
out["pred_tuned_progression_coral"]=pred_coral
out["pred_frozen_strong"]=pred_frozen
out["hard103"]=hard.astype(int)
for i in range(4):
    out[f"selected_weight_fusion_prob_{i}"]=selected_prob[:,i]
    out[f"fixed040_tunedalpha_fusion_prob_{i}"]=fixed_prob[:,i]
    out[f"tuned_progression_coral_prob_{i}"]=cand_prob[:,i]
out.to_csv(OUT/"01_SELECTED_WEIGHT_SYSTEM_POOLED_OOF.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(boots).to_csv(OUT/"02_BOOTSTRAP20K_CONTRASTS.csv",index=False)
(OUT/"03_LOCAL_ACCOUNTS.json").write_text(json.dumps(locals_,indent=2),encoding="utf-8")

summary={
 "status":"PASS_P10_C3_R6_R6_SELECTED_WEIGHT_OUTER_OOF_EVAL",
 "development_status":"nested-development OOF; weight selection used fixed inner-validation only, but outer OOF had been observed in prior route development",
 "selected_coral_weight_by_outer":EXPECTED_WEIGHTS,
 "points":{
   "selected_weight_fusion":m_selected,
   "fixed040_tunedalpha_fusion":m_fixed,
   "tuned_progression_coral":m_coral,
   "frozen_strong":m_frozen,
 },
 "bootstrap":boots,
 "local_accounts":locals_,
 "decision_flags":flags,
 "training_performed":False,
 "new_model_inference_performed":False,
 "outer_oof_used_for_weight_selection":False,
 "next_action":(
   "If fusion subsystem replacement is supported, freeze R11 as the current best system route, update the Graphviz master route map and handoff, then move to the next highest-value replaceable position only if meaningful net improvement space remains."
   if flags["fusion_subsystem_replacement_supported"] else
   "If replacement is not statistically supported, retain R10/R09 evidence and stop further fusion-weight tuning; move to a different replaceable position or higher-scale route."
 )
}
(OUT/"00_P10_C3_R6_R6_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

result=OUT/"P10_C3_R6_R6_SELECTED_WEIGHT_OUTER_OOF_EVAL_RESULT.zip"
with zipfile.ZipFile(result,"w",zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p!=result: z.write(p,p.name)
with zipfile.ZipFile(result) as z:
    req(z.testzip() is None,"RESULT_ZIP_TEST_FAIL")
print("P10_C3_R6_R6_SELECTED_WEIGHT_OUTER_OOF_EVAL=PASS")
print("NEXT_ACTION="+summary["next_action"])
print(f"RESULT_ZIP={result}")
print(f"RESULT_ZIP_SHA256={sha(result)}")
