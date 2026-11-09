#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W25-A4: authentic fixed DeiT + four A3 arms + Phadke P4.1-C.
Lifecycle/protection inherited from successful A3; actual fusion replaces model
inference. Input identity derives from verified A0/P5R2 and exact DeiT archive.
No image access, model loading, training, weight/threshold search or A2/A3 rerun.
R2: D1-locked P4C numeric compatibility + label-free decision-invariance check.
Original primary values and fusion/statistics rules remain unchanged.
"""
from __future__ import annotations
import argparse, csv, fcntl, hashlib, importlib.util, io, json, math, os, re, shutil
import subprocess, sys, time, traceback, zipfile
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
# CPU-only analysis; no GPU framework is imported or invoked.
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
HERE=Path(__file__).resolve().parent
RUN_PARENT=Path('/PATH/TO/RESEARCH_WORKSPACE/server_runs')
PREFIX='paper2_w25_a4_r2_fixed_deit_fusion_20261025_'
MANIFEST='10_FUSION_SOURCE_SHA256.json'
DESIGN='03_FUSION_DESIGN.json'
COMPLETE='PASS_FIXED_DEIT_MATCHED_FUSION_AND_PAIRED_ANALYSIS'
FAILED='FAIL_A4_DIAGNOSTIC_SAVED'
ARMS=('A0_NATURAL','A1_SCALED','A2_BINARY_P','A3_GRADE_FIXED')
COUNTS=(618,596,557,616,591)
PATIENT_COUNTS=(94,94,94,93,93)
PROTECTED={}
STAGE='INITIAL_CHECKS'
class GateError(RuntimeError):pass

def need(ok,code):
    if not ok:raise GateError(code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+'.partial')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(path)

def gate(path,h,label):
    path=Path(path);need(path.is_file() and not path.is_symlink(),label+'_MISSING_OR_SYMLINK')
    need(sha(path)==h,label+'_SHA_DRIFT');PROTECTED[str(path)]=h

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None,'MODULE_SPEC_MISSING')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def safe_child(root,relative):
    root=Path(root).resolve();p=Path(relative)
    need(not p.is_absolute() and '..' not in p.parts and len(p.parts)>0,'UNSAFE_RELATIVE_PATH')
    out=root/p;need(not out.is_symlink() and out.resolve().is_relative_to(root),'PATH_OR_SYMLINK_ESCAPE')
    return out

def run_for_design(h):
    need(re.fullmatch('[0-9a-f]{64}',h) is not None,'DESIGN_HASH_INVALID')
    return RUN_PARENT/(PREFIX+h[:12])

def require_run(run):
    need(run.parent==RUN_PARENT and re.fullmatch(PREFIX+'[0-9a-f]{12}',run.name) is not None,'RUN_PATH_INVALID')
    need(RUN_PARENT.is_dir() and not run.is_symlink(),'RUN_ROOT_INVALID')

def postcheck():
    for p,h in PROTECTED.items():need(Path(p).is_file() and sha(p)==h,'PROTECTED_SOURCE_OR_CHECKPOINT_CHANGED')
    return len(PROTECTED)

def process_identity(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text();parts=s[s.rfind(')')+2:].split()
        return {'state':parts[0],'start_ticks':parts[19]}
    except (OSError,IndexError):return None

def worker_live(run):
    p=run/'WORKER_PID.json'
    if not p.exists():return False
    x=read(p);now=process_identity(x['pid'])
    if not now or now['state']=='Z' or now['start_ticks']!=x['start_ticks']:return False
    try:b=(Path('/proc')/str(x['pid'])/'cmdline').read_bytes()
    except OSError:return False
    return str(run/'source'/Path(__file__).name).encode() in b and b'worker' in b

def integer(v):
    s=str(v).strip();need(re.fullmatch(r'-?[0-9]+(?:\.0+)?',s) is not None,'INTEGER_ID_OR_LABEL_INVALID')
    return int(s.split('.')[0])

def csvrows(b):
    reader=csv.DictReader(io.StringIO(b.decode('utf-8-sig')))
    need(reader.fieldnames is not None and len(set(reader.fieldnames))==len(reader.fieldnames),'CSV_HEADER_INVALID')
    return list(reader)

def read_csv(path):return csvrows(Path(path).read_bytes())

def require_columns(rows,columns,label):
    need(len(rows)>0 and all(set(columns)<=set(r) for r in rows),label+'_SCHEMA')

def unique_rows(rows,column,label):
    require_columns(rows,[column],label);out={}
    for r in rows:
        i=integer(r[column]);need(i not in out,label+'_DUPLICATE_ID');out[i]=r
    need(set(out)==set(range(2978)),label+'_COHORT_ID_SET')
    return out

def probabilities(rows,columns,label):
    require_columns(rows,columns,label)
    p=np.asarray([[float(r[c]) for c in columns] for r in rows],dtype=np.float64)
    need(p.shape==(len(rows),4) and np.isfinite(p).all(),label+'_PROB_SHAPE_OR_FINITE')
    need((p>=0).all() and (p<=1+1e-6).all(),label+'_PROB_RANGE')
    need(np.allclose(p.sum(1),1,rtol=0,atol=2e-6),label+'_PROB_SUM')
    return p

def fixed_fusion(p,deit):
    need(p.shape==deit.shape and p.ndim==2 and p.shape[1]==4,'FUSION_ALIGNMENT')
    # Never renormalize, clip, round or optimize probabilities or weights.
    return 0.40*p+0.60*deit

def validate_sources(root=HERE):
    m=read(root/MANIFEST)
    need(m['schema']=='W25_A4_FUSION_SOURCE_MANIFEST_V1','SOURCE_MANIFEST_SCHEMA')
    for n,h in m['files'].items():
        need(Path(n).name==n and re.fullmatch('[0-9a-f]{64}',h) is not None,'SOURCE_ENTRY_INVALID')
        gate(root/n,h,'FUSION_SOURCE_'+n)
    d=read(root/DESIGN)
    need(d['schema']=='W25_A4_FIXED_DEIT_MATCHED_FUSION_V1','FUSION_DESIGN_SCHEMA')
    need(tuple(d['arms'])==ARMS and d['outer_folds']==list(range(5)),'ARMS_OR_FOLDS_DRIFT')
    expected=list(ARMS)+['PHADKE_P4C','ORIGINAL_DEIT']+[n+'_F040' for n in list(ARMS)+['PHADKE_P4C']]
    need(d['systems']==expected,'FIXED_SYSTEM_SET_DRIFT')
    need(d['fixed_branch_weight']==.4 and d['fixed_deit_weight']==.6,'FIXED_WEIGHT_DRIFT')
    need(d['deit']['historical_metrics_6dp']==[.695433,.626829,.594757,.719990,.760831,.341169],'DEIT_AUTHORITY_METRIC_CONTRACT_DRIFT')
    need(d['p4c_archive']['expected_single_qwk']==.767997593,'P4C_AUTHORITY_METRIC_CONTRACT_DRIFT')
    need(d['bootstrap_repeats']==20000 and d['bootstrap_seed']==42,'BOOTSTRAP_DRIFT')
    for k in ('new_training','new_model_inference','weight_search','tuning'):need(d[k] is False,'SCOPE_DRIFT_'+k)
    need(d['fusion'] is True and d['new_bootstrap'] is True and len(d['contrasts'])==15,'SCOPE_OR_CONTRAST_COUNT')
    # All original scientific design fields are byte-derived from R1; only a
    # reviewed numeric representation policy is appended.
    n=d.get('numeric_compatibility_review',{})
    gate(root/'18_ORIGINAL_A4_DESIGN.json',n['original_design_sha256'],'R1_DESIGN_EXACT')
    original=read(root/'18_ORIGINAL_A4_DESIGN.json')
    need({k:v for k,v in d.items() if k!='numeric_compatibility_review'}==original,'ORIGINAL_FUSION_DESIGN_CHANGED')
    gate(root/'17_REVIEWED_D1_REPORT.json',n['d1_report_sha256'],'D1_REVIEW_EXACT')
    gate(root/'19_REVIEWED_A4_FAILURE.json',n['old_failure_report_sha256'],'R1_FAILURE_EXACT')
    need(n['policy']=='EXACT_D1_PROFILE_AND_ZERO_DECISION_CHANGE' and
         n['primary_raw_arrays_unchanged'] and n['normalization_on_diagnostic_copies_only'] and
         n['nonzero_decision_change']=='STOP_BEFORE_METRICS_BOOTSTRAP', 'NUMERIC_POLICY_DRIFT')
    helper=load_module(root/'16_d1_probability_compatibility.py','a4_r2_numeric_validation')
    try:helper.validate_evidence(read(root/'17_REVIEWED_D1_REPORT.json'),d['p4c_archive'])
    except helper.NumericCompatibilityError as e:raise GateError(str(e)) from e
    return m,d

def verify_old_failed_run(d):
    # Observe and protect the exact failed R1 run; never resume, mutate or delete.
    n=d['numeric_compatibility_review'];old=Path(n['old_run_directory'])
    expected=RUN_PARENT/('paper2_w25_a4_fixed_deit_fusion_20261023_'+n['original_design_sha256'][:12])
    need(old==expected and old.is_dir() and not old.is_symlink(),'R1_FAILED_RUN_IDENTITY')
    need(not external_worker_live(old,'01_remote_fixed_deit_fusion.py'),'R1_WORKER_STILL_ACTIVE')
    gate(old/'01_FIXED_FUSION_REVIEW.json',n['old_failure_report_sha256'],'R1_FAILURE_PRESERVED')
    gate(old/'02_FIXED_FUSION_SAFE_RESULT.zip',n['old_failure_bundle_sha256'],'R1_FAILURE_BUNDLE_PRESERVED')
    gate(old/'source/03_FUSION_DESIGN.json',n['original_design_sha256'],'R1_DESIGN_PRESERVED')
    gate(old/'source/10_FUSION_SOURCE_SHA256.json',n['original_source_manifest_sha256'],'R1_SOURCE_MANIFEST_PRESERVED')
    f=read(old/'01_FIXED_FUSION_REVIEW.json')
    need(f['status']==FAILED and f['error_code']=='P4C_PROB_SUM' and
         f['fixed_fusion_complete'] is False and f['bootstrap_executed'] is False,'R1_FAILURE_STATE_CHANGED')
    need(read(old/'00_PROGRESS.json')['state']==FAILED,'R1_PROGRESS_STATE_CHANGED')
    for name,h in read(old/'source/10_FUSION_SOURCE_SHA256.json')['files'].items():
        gate(safe_child(old/'source',name),h,'R1_SOURCE_PRESERVED')
    count=0
    for p in sorted(old.rglob('*')):
        need(not p.is_symlink(),'R1_TREE_SYMLINK')
        if p.is_file():PROTECTED[str(p)]=sha(p);count+=1
    return {'exact_failed_R1_verified':True,'old_worker_live':False,
            'old_run_restarted_or_modified':False,'protected_files':count,
            'producer_cause_confirmed':False,'d1_report_sha256':n['d1_report_sha256']}

def safe_progress(run,state,**kwargs):
    x={'schema':'W25_A4_FUSION_PROGRESS_V1','state':state,'stage':STAGE,
       'updated_unix':time.time(),'worker_pid':os.getpid(),'new_training':False,
       'new_model_inference':False,'completed_systems':0}
    x.update(kwargs);dump(run/'00_PROGRESS.json',x)

def external_worker_live(run,filename):
    p=run/'WORKER_PID.json'
    if not p.exists():return False
    x=read(p);now=process_identity(x['pid'])
    if not now or now['state']=='Z' or now['start_ticks']!=x['start_ticks']:return False
    try:cmd=(Path('/proc')/str(x['pid'])/'cmdline').read_bytes()
    except OSError:return False
    return str(run/'source'/filename).encode() in cmd and b'worker' in cmd

def verify_a3(d):
    a3=Path(d['a3_run_directory'])
    need(a3.parent==RUN_PARENT and re.fullmatch('paper2_w25_a3_frozen_outer_eval_20261023_[0-9a-f]{12}',a3.name) is not None,'A3_PATH_INVALID')
    gate(a3/'01_OUTER_EVALUATION_REVIEW.json',d['a3_report_sha256'],'A3_REPORT')
    gate(a3/'02_EVAL_SAFE_RESULT.zip',d['a3_safe_bundle_sha256'],'A3_SAFE_BUNDLE')
    gate(a3/'source/10_EVAL_SOURCE_SHA256.json',d['a3_source_manifest_sha256'],'A3_SOURCE_MANIFEST')
    gate(a3/'source/03_EVALUATION_DESIGN.json',d['a3_design_sha256'],'A3_DESIGN')
    gate(a3/'source/01_remote_evaluate_frozen_aux.py',d['a3_controller_sha256'],'A3_CONTROLLER')
    for n,h in read(a3/'source/10_EVAL_SOURCE_SHA256.json')['files'].items():gate(safe_child(a3/'source',n),h,'A3_FROZEN_SOURCE')
    need(not external_worker_live(a3,'01_remote_evaluate_frozen_aux.py'),'A3_STILL_RUNNING')
    need(read(a3/'00_PROGRESS.json')['state']=='PASS_FROZEN_OUTER_EVALUATION_AND_PAIRED_ANALYSIS','A3_NOT_COMPLETED')
    r=read(a3/'01_OUTER_EVALUATION_REVIEW.json')
    need(r['status']=='PASS_FROZEN_OUTER_EVALUATION_AND_PAIRED_ANALYSIS' and r['outer_prediction_complete'] and r['bootstrap_executed'],'A3_REPORT_INCOMPLETE')
    need(sha(HERE/'04_A3_COMPLETED_REPORT.json')==d['a3_report_sha256'],'A3_PACKAGED_REPORT_MISMATCH')
    for p in sorted(a3.rglob('*')):
        need(not p.is_symlink(),'A3_TREE_SYMLINK')
        if p.is_file():PROTECTED[str(p)]=sha(p)
    identity=read(a3/'99_RESULT_IDENTITY.json')
    need(identity['report_sha256']==d['a3_report_sha256'] and identity['bundle_sha256']==d['a3_safe_bundle_sha256'],'A3_RESULT_IDENTITY')
    ph=read(a3/d['a3_hash_manifest'])
    need(set(ph)=={d['a3_probabilities'],d['a3_decisions']},'A3_PREDICTION_MANIFEST_SCHEMA')
    for n,h in ph.items():gate(safe_child(a3,n),h,'A3_SERVER_REGISTERED_PREDICTIONS')
    return a3,r,ph

def zip_table(config,key='member',hashkey='member_sha256'):
    gate(Path(config['path']),config['sha256'],'FROZEN_ARCHIVE')
    with zipfile.ZipFile(config['path']) as z:
        need(z.namelist().count(config[key])==1,'ARCHIVE_MEMBER_NOT_UNIQUE')
        b=z.read(config[key]);need(hashlib.sha256(b).hexdigest()==config[hashkey],'ARCHIVE_MEMBER_SHA')
    return csvrows(b)

def identity_tuple(row,pidcol,foldcol,ycol,parent):
    pid=parent.normalize_pid(row[pidcol]);fold=integer(row[foldcol]);y=integer(row[ycol])
    need(pid is not None and fold in range(5) and y in range(4),'PATIENT_FOLD_GRADE_INVALID')
    return pid,fold,y

def authoritative_rows(d,parent):
    rows=zip_table(d['r11_archive']);auth=unique_rows(rows,'_id','R11')
    required=['patient_id','_outer_eval','y_true_auth','pred_tuned_progression_coral','pred_fixed040_tunedalpha_fusion','pred_selected_weight_fusion']
    require_columns(rows,required,'R11')
    ids={i:identity_tuple(r,'patient_id','_outer_eval','y_true_auth',parent) for i,r in auth.items()}
    need(len({t[0] for t in ids.values()})==468,'R11_PATIENT_COUNT')
    pidfold=defaultdict(set)
    for p,f,y in ids.values():pidfold[p].add(f)
    need(all(len(s)==1 for s in pidfold.values()),'R11_PATIENT_SPLIT')
    for f in range(5):
        q=[t for t in ids.values() if t[1]==f]
        need(len(q)==COUNTS[f] and len({t[0] for t in q})==PATIENT_COUNTS[f],'R11_FOLD_COUNTS')
    need(np.bincount([ids[i][2] for i in range(2978)],minlength=4).tolist()==[2009,507,254,208],'R11_GRADE_COUNTS')
    return auth,ids

def align_deit(d,ids,auth,parent,stats):
    cfg=d['deit'];paths=[Path(p) for p in cfg['paths'] if Path(p).is_file()]
    need(len(paths)>0,'AUTHENTIC_DEIT_FILE_MISSING_NO_BACKSOLVING_PERMITTED')
    # Copies may be selected only by the prespecified path order and identical SHA.
    for p in paths:gate(p,cfg['sha256'],'EXACT_ORIGINAL_DEIT')
    rows=read_csv(paths[0]);by=unique_rows(rows,cfg['id_column'],'DEIT')
    required=[cfg[k] for k in ('patient_column','fold_column','inner_fold_column','label_column','prediction_column')]+['checkpoint_epoch','checkpoint_monitor','checkpoint_selection_protocol']
    require_columns(rows,required,'DEIT')
    for i,r in by.items():
        got=identity_tuple(r,cfg['patient_column'],cfg['fold_column'],cfg['label_column'],parent)
        need(got==ids[i],'DEIT_PATIENT_OUTER_LABEL_MISMATCH')
        need(integer(r[cfg['inner_fold_column']])==(ids[i][1]+1)%5,'DEIT_INNER_VALIDATION_ROLE_MISMATCH')
        need(integer(r['checkpoint_epoch'])>=0 and bool(r['checkpoint_monitor']) and bool(r['checkpoint_selection_protocol']),'DEIT_SELECTION_METADATA_MISSING')
    ordered=[by[i] for i in range(2978)]
    p=probabilities(ordered,cfg['probability_columns'],'DEIT')
    pred=np.asarray([integer(r[cfg['prediction_column']]) for r in ordered],dtype=int)
    need(np.array_equal(pred,p.argmax(1)),'DEIT_NATIVE_ARGMAX_MISMATCH')
    y=np.asarray([ids[i][2] for i in range(2978)],dtype=int)
    pt=stats.point(y,pred)
    # Historic six-decimal metrics are a sanity gate, not the row-identity proof.
    expected=cfg['historical_metrics_6dp']
    need(all(abs(pt['metrics'][k]-v)<=1e-6 for k,v in zip(stats.METRICS,expected)),'DEIT_HISTORICAL_METRIC_MISMATCH')
    ar=[auth[i] for i in range(2978)]
    r09=probabilities(ar,['tuned_progression_coral_prob_'+str(j) for j in range(4)],'HISTORICAL_R09')
    r10=probabilities(ar,['fixed040_tunedalpha_fusion_prob_'+str(j) for j in range(4)],'HISTORICAL_R10')
    need(np.array_equal(r09.argmax(1),np.asarray([integer(r['pred_tuned_progression_coral']) for r in ar])),'R09_NATIVE_ARGMAX')
    recreated=fixed_fusion(r09,p)
    error=float(np.max(np.abs(recreated-r10)))
    mismatch=int(np.count_nonzero(recreated.argmax(1)!=np.asarray([integer(r['pred_fixed040_tunedalpha_fusion']) for r in ar])))
    need(error<=d['r10_forward_reconstruction_atol'] and mismatch==0,'ORIGINAL_R10_FORWARD_PARITY_FAILED_NO_SUBSTITUTION')
    return p,{'sha256':cfg['sha256'],'path_identity_index':cfg['paths'].index(str(paths[0])),
        'exact_sha_copies_present':len(paths),'rows':2978,'patients':468,
        'patient_roi_outer_inner_label_mismatches':0,'native_argmax_mismatches':0,
        'forward_R09_plus_original_DeiT_R10_prediction_mismatches':mismatch,'forward_R10_probability_max_abs':error,
        'forward_only_not_backsolved':True,'same_probability_array_used_for_all_fusions':True,'point':pt}

def align_phadke(d,ids,parent,stats):
    cfg=d['p4c_archive'];idx=zip_table(cfg,'index_member','index_sha256');rows=zip_table(cfg)
    gate(Path(d['r2_manifest']['path']),d['r2_manifest']['sha256'],'R2_IDENTITY_BRIDGE')
    r2=read_csv(d['r2_manifest']['path'])
    cross=parent.try_exact_r2_roi_crosswalk(r2,idx)
    need(cross['status']=='EXACT_2978_ROI_IMAGE_BBOX_CROSSWALK','P4C_IMAGE_BBOX_CROSSWALK_UNPROVEN')
    conf=cross['successful_key']
    def key(row,isp4):
        pp=parent.normalize_pid(row['patient' if isp4 else 'patient_id'])
        n=parent.filename_part(row.get(conf['p4c' if isp4 else 'r2'],''),conf['mode']=='stem')
        b=parent.bbox_from(row,('xmin','ymin','xmax','ymax') if isp4 else ('x0','y0','x1','y1'))
        need(pp and n and b is not None,'BRIDGE_NULL_ID');return pp,n,b
    keyindex={key(r,True):r for r in idx};fid_to_id={}
    for row in r2:
        i=integer(row['row_index']);matched=keyindex[key(row,False)];fid=str(matched['foramen_id']).strip()
        need(i in ids and fid not in fid_to_id,'BRIDGE_ID_DUPLICATE_OR_MISSING')
        got=(parent.normalize_pid(row['patient_id']),integer(row['outer_fold']),integer(matched['grade']))
        need(got==ids[i],'R2_MAPPING_VS_ORIGINAL_R11_IDENTITY')
        fid_to_id[fid]=i
    need(len(fid_to_id)==2978 and set(fid_to_id.values())==set(ids),'BRIDGE_NOT_BIJECTIVE')
    by={}
    for row in rows:
        fid=str(row['foramen_id']).strip();need(fid in fid_to_id,'P4C_FORAMEN_MISSING')
        i=fid_to_id[fid];need(i not in by,'P4C_DUPLICATE_ID')
        need(identity_tuple(row,'patient','outer_fold','y_true',parent)==ids[i],'P4C_IDENTITY_MISMATCH')
        by[i]=row
    need(set(by)==set(ids),'P4C_ID_SET_MISMATCH')
    ordered=[by[i] for i in range(2978)]
    numeric=load_module(HERE/'16_d1_probability_compatibility.py','a4_r2_p4c_numeric')
    try:p,numeric_audit=numeric.reviewed_raw_p4c(ordered,cfg,read(HERE/'17_REVIEWED_D1_REPORT.json'))
    except numeric.NumericCompatibilityError as e:raise GateError(str(e)) from e
    a=np.asarray([integer(r['pred_argmax_harmonized']) for r in ordered],dtype=int)
    s=np.asarray([integer(r['pred_source_rule_nested']) for r in ordered],dtype=int)
    need(np.array_equal(a,p.argmax(1)) and np.array_equal(a,s),'P4C_ARCHIVED_DECODER_IDENTITY_CHANGED')
    y=np.asarray([ids[i][2] for i in range(2978)],dtype=int);pt=stats.point(y,a)
    need(abs(pt['metrics']['qwk']-cfg['expected_single_qwk'])<=2e-9,'P4C_FROZEN_QWK_MISMATCH')
    summaries=[]
    with zipfile.ZipFile(cfg['path']) as z:
        for info in z.infolist():
            if not info.filename.lower().endswith('.csv') or info.file_size>2000000:continue
            rr=csvrows(z.read(info))
            if len(rr)==5 and {'outer','fixed_val','train_rows','val_rows','test_rows','test_used_for_selection'}<=set(rr[0]):
                summaries.append(rr)
    need(len(summaries)==1,'PHADKE_ROLE_SUMMARY_NOT_UNIQUE')
    roles=summaries[0]
    need({integer(x['outer']) for x in roles}==set(range(5)),'PHADKE_ROLE_FOLDS')
    safe_roles=[]
    for row in roles:
        f=integer(row['outer']);vf=(f+1)%5
        need(integer(row['fixed_val'])==vf,'PHADKE_INNER_VALIDATION_FOLD')
        need((integer(row['train_rows']),integer(row['val_rows']),integer(row['test_rows']))==
             (2978-COUNTS[f]-COUNTS[vf],COUNTS[vf],COUNTS[f]),'PHADKE_TRAIN_VAL_TEST_COUNTS')
        need(str(row['test_used_for_selection']).strip().lower() in ('false','0'),'PHADKE_TEST_SELECTION_FLAG')
        if 'test_inference_after_checkpoint_freeze' in row:
            need(str(row['test_inference_after_checkpoint_freeze']).strip().lower() in ('true','1'),'PHADKE_FREEZE_ROLE_FLAG')
        safe_roles.append({k:row[k] for k in ('outer','fixed_val','train_rows','val_rows','test_rows','checkpoint_selection') if k in row})
    return p,{'source_sha256':cfg['sha256'],'oof_member_sha256':cfg['member_sha256'],'rows':2978,
       'image_bbox_crosswalk':cross,'direct_R11_patient_fold_label_mismatches':0,'source_vs_harmonized_prediction_mismatches':0,
       'archived_role_summaries':safe_roles,'point':pt,'numeric_mass_compatibility':numeric_audit}

def points_equal(a,b,tol=1e-12):
    return a['confusion_matrix']==b['confusion_matrix'] and all(
        (a['metrics'][k] is None and b['metrics'][k] is None) or
        (a['metrics'][k] is not None and b['metrics'][k] is not None and abs(a['metrics'][k]-b['metrics'][k])<=tol)
        for k in a['metrics'])

def align_a3(a3,d,ids,parent,stats,r):
    rows=read_csv(a3/d['a3_probabilities']);need(len(rows)==11912,'A3_TOTAL_PROBABILITY_ROWS')
    need({x.get('arm') for x in rows}==set(ARMS),'A3_ARM_NAMES')
    decisions=unique_rows(read_csv(a3/d['a3_decisions']),'protocol_v4_row_id','A3_DECISIONS')
    for i,row in decisions.items():need(identity_tuple(row,'patient_id','unified_outer_fold','y_true',parent)==ids[i],'A3_DECISION_IDENTITY')
    out=[];y=np.asarray([ids[i][2] for i in range(2978)],dtype=int)
    fold=np.asarray([ids[i][1] for i in range(2978)],dtype=int)
    for arm in ARMS:
        by=unique_rows([x for x in rows if x['arm']==arm],'protocol_v4_row_id','A3_'+arm)
        for i,row in by.items():need(identity_tuple(row,'patient_id','unified_outer_fold','y_true',parent)==ids[i],'A3_PROBABILITY_IDENTITY')
        ordered=[by[i] for i in range(2978)];p=probabilities(ordered,['p'+str(j) for j in range(4)],'A3_'+arm)
        native=np.asarray([integer(row['prediction']) for row in ordered],dtype=int)
        need(np.array_equal(native,p.argmax(1)),'A3_NATIVE_ARGMAX_MISMATCH')
        need(np.array_equal(native,np.asarray([integer(decisions[i][arm]) for i in range(2978)])),'A3_PROBABILITY_DECISION_PARITY')
        need(points_equal(stats.point(y,native),r['pooled_results'][arm]),'A3_POOLED_CONFUSION_OR_METRIC_CHANGED')
        for f in range(5):
            old=next(x for x in r['fold_results'] if x['outer_fold']==f)['arms'][arm]
            need(points_equal(stats.point(y[fold==f],native[fold==f]),old),'A3_FOLD_CONFUSION_OR_METRIC_CHANGED')
        out.append(p)
    return out,{'rows':11912,'systems':4,'direct_authority_identity_mismatches':0,
          'all_24_original_confusion_matrices_and_six_metrics_reproduced':True,'probability_argmax_decision_parity':True,
          'a3_server_hash_manifest_scope':'A3 server-side declared hashes reverified; these CSV hashes were not included in the earlier exported aggregate report; that report independently anchors 24 confusion matrices and metrics.'}

def evaluate_all(run,report,d):
    global STAGE
    stats=load_module(HERE/'02_fixed_fusion_metrics.py','a4_metrics')
    parent=load_module(HERE/'05_P5R2_IDENTITY_PARENT.py','a4_exact_p5r2')
    STAGE='VERIFY_COMPLETED_A3_AND_FROZEN_PROBABILITIES';safe_progress(run,'VERIFYING_FROZEN_PROBABILITIES')
    report['reviewed_numeric_policy_and_R1_protection']=verify_old_failed_run(d)
    a3,r,ph=verify_a3(d)
    auth,ident=authoritative_rows(d,parent)
    arms,a3audit=align_a3(a3,d,ident,parent,stats,r)
    report['a3_probability_reuse_audit']=a3audit;report['a3_server_registered_prediction_sha256']=ph
    STAGE='AUTHENTIC_DEIT_ROLES_AND_HISTORICAL_R10_PARITY';safe_progress(run,'VERIFYING_DEIT_IDENTITY')
    deit,deitaudit=align_deit(d,ident,auth,parent,stats);report['deit_identity']=deitaudit
    print('W25_A4_DEIT_EXACT_IDENTITY_AND_R10_FORWARD_PARITY=PASS',flush=True)
    STAGE='PHADKE_IMAGE_BBOX_AND_ORIGINAL_AUTHORITY_CROSSWALK';safe_progress(run,'VERIFYING_PHADKE_IDENTITY')
    phadke,phadkeaudit=align_phadke(d,ident,parent,stats);report['phadke_identity']=phadkeaudit
    print('W25_A4_FOUR_ARMS_PHADKE_DEIT_PATIENT_ROI_FOLD_IDENTITY=PASS',flush=True)
    branches=arms+[phadke]
    STAGE='D1_COMPATIBILITY_AND_NORMALIZATION_DECISION_PARITY'
    safe_progress(run,'VERIFYING_NUMERICAL_COMPATIBILITY')
    numeric=load_module(HERE/'16_d1_probability_compatibility.py','a4_r2_decision_parity')
    parity=numeric.decision_invariance(branches,deit,d['systems'])
    report['numeric_representation_sensitivity']=parity
    need(parity['all_11_decisions_invariant'],'NUMERIC_REPRESENTATION_CHANGES_DECISION_STOP_BEFORE_EFFICACY')
    print('W25_A4_R2_D1_PROFILE_REPRODUCED_AND_ALL11_DECISIONS_INVARIANT=PASS',flush=True)
    fused=[fixed_fusion(p,deit) for p in branches]
    allprobs=branches+[deit]+fused
    preds=np.stack([p.argmax(1) for p in allprobs],axis=1).astype(int)
    y=np.asarray([ident[i][2] for i in range(2978)],dtype=int)
    patients=np.asarray([ident[i][0] for i in range(2978)],dtype=str)
    folds=np.asarray([ident[i][1] for i in range(2978)],dtype=int)
    STAGE='FIXED_FUSION_NO_SELECTION';safe_progress(run,'FUSING_FIXED_PROBABILITIES',completed_systems=11)
    report['pooled_results']={n:stats.point(y,preds[:,j]) for j,n in enumerate(d['systems'])}
    report['fold_results']=[{'outer_fold':f,'patients':int(len(set(patients[folds==f]))),'rois':int((folds==f).sum()),
       'systems':{n:stats.point(y[folds==f],preds[folds==f,j]) for j,n in enumerate(d['systems'])}} for f in range(5)]
    # Context is inherited, never substituted for matched A3_GRADE_FIXED.
    report['historical_context']={name:stats.point(y,np.asarray([integer(auth[i][col]) for i in range(2978)],dtype=int))
       for name,col in [('R09_SELECTED_ALPHA','pred_tuned_progression_coral'),('R10_FIXED_FUSION','pred_fixed040_tunedalpha_fusion'),('R11_SELECTED_FUSION','pred_selected_weight_fusion')]}
    report['historical_context_is_not_newly_tuned_or_replaced']=True
    with (run/'SERVER_ONLY_ALIGNED_SINGLE_AND_FUSED_PROBABILITIES.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['protocol_v4_row_id','patient_id','outer_fold','y_true','system','prediction','p0','p1','p2','p3'])
        for i in range(2978):
            for j,name in enumerate(d['systems']):writer.writerow([i,patients[i],int(folds[i]),int(y[i]),name,int(preds[i,j]),*allprobs[j][i].tolist()])
    report['server_only_fused_probability_sha256']=sha(run/'SERVER_ONLY_ALIGNED_SINGLE_AND_FUSED_PROBABILITIES.csv')
    report['source_probability_identity_complete']=True
    STAGE='NEW_MATCHED_PATIENT_BOOTSTRAP';last=[-1]
    def progress(done,total):
        if done==total or done//1000!=last[0]:
            last[0]=done//1000;safe_progress(run,'BOOTSTRAPPING',completed_systems=11,bootstrap_completed=done,bootstrap_total=total)
            print(f'W25_A4_PAIRED_BOOTSTRAP={done}/{total}',flush=True)
    report['paired_bootstrap']=stats.paired_fusion_bootstrap(y,preds,patients,d['systems'],d['contrasts'],
        iterations=d['bootstrap_repeats'],seed=d['bootstrap_seed'],callback=progress)
    need(all(v['valid_replicates']==d['bootstrap_repeats'] for c in report['paired_bootstrap']['contrasts'] for v in c['metrics'].values()),'UNDEFINED_BOOTSTRAP_REPLICATES_REVIEW_REQUIRED')
    report['fixed_fusion_complete']=True;report['bootstrap_executed']=True
    report['phadke_with_same_deit_fixed_fusion_completed']=True
    report['validation_selected_fair_fusion_completed']=False
    report['equal_backbone_compute_and_input_pixels_claimed']=False
    STAGE='POST_PROTECTION';report['protected_source_files_unchanged']=postcheck()
    report['no_original_scientific_file_replaced']=True
def export_result(run,report):
    dump(run/'01_FIXED_FUSION_REVIEW.json',report)
    payload={'01_FIXED_FUSION_REVIEW.json':(run/'01_FIXED_FUSION_REVIEW.json').read_bytes(),
        DESIGN:(HERE/DESIGN).read_bytes(),MANIFEST:(HERE/MANIFEST).read_bytes(),
        '04_A3_COMPLETED_REPORT.json':(HERE/'04_A3_COMPLETED_REPORT.json').read_bytes(),
        '17_REVIEWED_D1_REPORT.json':(HERE/'17_REVIEWED_D1_REPORT.json').read_bytes()}
    payload['98_MEMBER_SHA256.json']=(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in payload.items()},indent=2)+'\n').encode()
    tmp=run/'02_FIXED_FUSION_SAFE_RESULT.zip.partial'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
    tmp.replace(run/'02_FIXED_FUSION_SAFE_RESULT.zip')
    dump(run/'99_RESULT_IDENTITY.json',{'report_sha256':sha(run/'01_FIXED_FUSION_REVIEW.json'),'bundle_sha256':sha(run/'02_FIXED_FUSION_SAFE_RESULT.zip')})

def validate_run_sources(run,manifest_sha):
    need(not (run/'source').is_symlink(),'SOURCE_SYMLINK')
    x=read(run/'00_LAUNCH.json')
    need(x['source_manifest_sha256']==manifest_sha and x['design_sha256']==sha(HERE/DESIGN),'FUSION_RUN_IDENTITY_MISMATCH')
    need(sha(run/'source'/MANIFEST)==manifest_sha,'INSTALLED_FUSION_MANIFEST_DRIFT');validate_sources(run/'source')

def status_snapshot(run,manifest_sha):
    base={'schema':'W25_A4_FIXED_FUSION_STATUS_V1','run_directory':str(run),'source_manifest_sha256':manifest_sha,
          'design_sha256':sha(HERE/DESIGN),'new_training':False,'new_model_inference':False,'tuning':False,'fusion':True,'restart_permitted':False}
    if not run.exists():base.update(state='NOT_STARTED',worker_live=False,safe_result_available=False);return base
    validate_run_sources(run,manifest_sha)
    p=read(run/'00_PROGRESS.json') if (run/'00_PROGRESS.json').is_file() else {'state':'STARTING'}
    live=worker_live(run);base.update(progress=p,worker_live=live)
    ident=run/'99_RESULT_IDENTITY.json'
    if ident.exists():
        h=read(ident);rr=run/'01_FIXED_FUSION_REVIEW.json';zz=run/'02_FIXED_FUSION_SAFE_RESULT.zip'
        need(sha(rr)==h['report_sha256'] and sha(zz)==h['bundle_sha256'],'RESULT_HASH_DRIFT')
        r=read(rr);need(not r['new_training'] and not r['new_model_inference'] and not r['tuning'] and r['fusion'],'RESULT_SCOPE_DRIFT')
        base.update(state=r['status'],report_sha256=h['report_sha256'],bundle_sha256=h['bundle_sha256'],safe_result_available=True,
            numeric_decision_invariance=r.get('numeric_representation_sensitivity',{}).get('all_11_decisions_invariant'),
            error_code=r.get('error_code'),failed_stage=r.get('failed_stage'))
    elif live:base.update(state=p['state'],safe_result_available=False)
    else:base.update(state='STOPPED_OR_STARTUP_UNCONFIRMED_NO_AUTORESTART',safe_result_available=False)
    return base

def worker(run,manifest_sha):
    require_run(run);validate_run_sources(run,manifest_sha)
    need(HERE==run/'source','WORKER_NOT_PERSISTENT_SOURCE')
    with (run/'WORKER.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise GateError('FUSION_ALREADY_RUNNING')
        cur=process_identity(os.getpid());need(cur is not None,'PID_IDENTITY_UNAVAILABLE')
        dump(run/'WORKER_PID.json',{'pid':os.getpid(),'start_ticks':cur['start_ticks']})
        safe_progress(run,'PREPARING_FIXED_FUSION')
        _,d=validate_sources()
        r={'schema':'W25_A4_FIXED_DEIT_FUSION_REVIEW_V1','status':'RUNNING','created_unix':time.time(),
           'run_directory':str(run),'a3_run_directory':d['a3_run_directory'],'a3_report_sha256':d['a3_report_sha256'],
           'source_manifest_sha256':manifest_sha,'design_sha256':sha(HERE/DESIGN),
           'new_training':False,'new_model_inference':False,'tuning':False,'fusion':True,
           'fixed_branch_weight':.4,'fixed_deit_weight':.6,'fixed_fusion_complete':False,'bootstrap_executed':False,
           'single_seed_internal_retrospective':True,'source_probability_identity_complete':False,
           'statistical_superiority_claim_automatically_authorized':False}
        code=0
        try:evaluate_all(run,r,d);r['status']=COMPLETE
        except (Exception,SystemExit) as e:
            code=2;r['status']=FAILED;r['failed_stage']=STAGE
            # Only whitelisted gate codes, no exception values containing patient rows.
            r['error_code']=str(e) if isinstance(e,GateError) else 'RUNTIME_'+type(e).__name__
            (run/'SERVER_ONLY_EXCEPTION.log').write_text(traceback.format_exc(),encoding='utf-8')
            try:r['protected_files_checked_after_failure']=postcheck();r['protected_files_unchanged']=True
            except Exception:r['protected_files_unchanged']=False
        r['finished_unix']=time.time();export_result(run,r)
        safe_progress(run,r['status'],completed_systems=len(r.get('pooled_results',{})),safe_result_available=True,error_code=r.get('error_code'))
        print('W25_A4_FINAL_STATUS='+r['status'],flush=True);return code

def evaluate_once(run,manifest_sha):
    require_run(run)
    with (RUN_PARENT/(run.name+'.START.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise GateError('ANOTHER_EVALUATE_REQUEST_ACTIVE_USE_STATUS')
        if run.exists():
            s=status_snapshot(run,manifest_sha);s['action']='EXISTING_FUSION_NOT_RESTARTED';return s
        m,d=validate_sources()
        # Only read completed A3 metadata. No GPU contention gate or model loading.
        verify_old_failed_run(d)
        verify_a3(d)
        run.mkdir();source=run/'source';source.mkdir()
        for n in m['files']:shutil.copyfile(HERE/n,source/n)
        shutil.copyfile(HERE/MANIFEST,source/MANIFEST);validate_sources(source)
        dump(run/'00_LAUNCH.json',{'schema':'W25_A4_FUSION_LAUNCH_V1','created_unix':time.time(),
             'design_sha256':sha(HERE/DESIGN),'source_manifest_sha256':manifest_sha,
             'a3_report_sha256':d['a3_report_sha256'],'new_training':False,'new_model_inference':False,'automatic_restart_or_resume':False})
        safe_progress(run,'STARTING')
        cmd=[sys.executable,'-B',str(source/Path(__file__).name),'--mode','worker','--manifest-sha',manifest_sha]
        with (run/'SERVER_ONLY_WORKER.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=source,start_new_session=True,close_fds=True)
        for _ in range(100):
            if (run/'WORKER_PID.json').exists() or proc.poll() is not None:break
            time.sleep(.1)
        s=status_snapshot(run,manifest_sha);s['action']='NEW_FIXED_FUSION_ANALYSIS_SCHEDULED';return s

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['evaluate','status','worker'],required=True)
    ap.add_argument('--manifest-sha',required=True);ap.add_argument('--out',type=Path);ap.add_argument('--bundle',type=Path)
    a=ap.parse_args();need(sha(HERE/MANIFEST)==a.manifest_sha,'EXPECTED_FUSION_MANIFEST_HASH')
    validate_sources();run=run_for_design(sha(HERE/DESIGN));require_run(run)
    if a.mode=='worker':return worker(run,a.manifest_sha)
    need(a.out is not None and a.bundle is not None,'CLIENT_OUTPUT_ARGUMENTS')
    for p in (a.out,a.bundle):need(re.fullmatch(r'/PATH/TO/TEMP/lss_w25a4r2_[A-Za-z0-9]{8}',str(p.parent)) is not None,'CLIENT_OUTPUT_PATH_INVALID')
    s=evaluate_once(run,a.manifest_sha) if a.mode=='evaluate' else status_snapshot(run,a.manifest_sha)
    dump(a.out,s)
    if s.get('safe_result_available'):shutil.copyfile(run/'02_FIXED_FUSION_SAFE_RESULT.zip',a.bundle)
    print('W25_A4_JOB_STATE='+s['state'],flush=True)
    print('W25_A4_RUN_DIRECTORY='+str(run),flush=True)
    if s.get('error_code'):print('W25_A4_R2_ERROR='+str(s['error_code'])+'|STAGE='+str(s.get('failed_stage')),flush=True)
    if s.get('numeric_decision_invariance') is not None:print('W25_A4_R2_ALL11_DECISIONS_INVARIANT='+str(s['numeric_decision_invariance']),flush=True)
    p=s.get('progress',{});print(f"W25_A4_PROGRESS|systems={p.get('completed_systems',0)}/11|bootstrap={p.get('bootstrap_completed','-')}/{p.get('bootstrap_total','-')}",flush=True)
    print('W25_A4_TRAINING_AND_MODEL_INFERENCE=False|NO_WEIGHT_SEARCH=True|STATUS_NEVER_STARTS_WORKER=True',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
