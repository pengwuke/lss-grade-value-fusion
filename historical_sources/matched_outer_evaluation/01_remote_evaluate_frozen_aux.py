#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W25-A3: reviewed frozen A2 checkpoints -> outer OOF -> paired statistics.
Lifecycle derived from the successful A2 controller; model, dataset, probability
conversion and evaluate() are imported only from A2's SHA-verified source snapshot.
No training routine, optimizer, checkpoint selection, fusion, or tuning is called.
Evaluate never restarts an existing evaluation; Status never starts any worker.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, importlib.util, json, math, os, re, shutil
import subprocess, sys, time, traceback, zipfile
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
HERE=Path(__file__).resolve().parent
RUN_PARENT=Path('/PATH/TO/RESEARCH_WORKSPACE/server_runs')
PREFIX='paper2_w25_a3_frozen_outer_eval_20261023_'
MANIFEST='10_EVAL_SOURCE_SHA256.json'
DESIGN='03_EVALUATION_DESIGN.json'
COMPLETE='PASS_FROZEN_OUTER_EVALUATION_AND_PAIRED_ANALYSIS'
FAILED='FAIL_A3_DIAGNOSTIC_SAVED'
ARMS=('A0_NATURAL','A1_SCALED','A2_BINARY_P','A3_GRADE_FIXED')
COUNTS=(618,596,557,616,591)
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

def validate_sources(root=HERE):
    m=read(root/MANIFEST)
    need(m['schema']=='W25_A3_EVALUATION_SOURCE_MANIFEST_V1','SOURCE_MANIFEST_SCHEMA')
    for n,h in m['files'].items():
        need(Path(n).name==n and re.fullmatch('[0-9a-f]{64}',h) is not None,'SOURCE_ENTRY_INVALID')
        gate(root/n,h,'EVAL_SOURCE_'+n)
    d=read(root/DESIGN)
    need(d['schema']=='W25_A3_FROZEN_OUTER_EVALUATION_V1','EVALUATION_DESIGN_SCHEMA')
    need(tuple(d['arms'])==ARMS and d['outer_folds']==list(range(5)),'ARMS_OR_FOLDS_DRIFT')
    need(d['bootstrap_repeats']==20000 and d['bootstrap_seed']==42 and d['evaluation_batch_size']==32,'STATISTIC_OR_BATCH_DRIFT')
    need(d['contrasts']==[[3,1],[3,0],[3,2],[1,0]],'CONTRAST_DRIFT')
    need(d['new_training'] is False and d['tuning'] is False and d['fusion'] is False,'SCOPE_DRIFT')
    need(d['new_outer_inference'] is True and d['new_bootstrap'] is True,'EVALUATION_AUTHORIZATION_DRIFT')
    return m,d

def run_for_design(h):
    need(re.fullmatch('[0-9a-f]{64}',h) is not None,'DESIGN_HASH_INVALID')
    return RUN_PARENT/(PREFIX+h[:12])

def require_run(run):
    need(run.parent==RUN_PARENT and re.fullmatch(PREFIX+'[0-9a-f]{12}',run.name) is not None,'RUN_PATH_INVALID')
    need(RUN_PARENT.is_dir() and not run.is_symlink(),'RUN_ROOT_INVALID')

def check_freeze_report(r,d):
    need(r['status']=='PASS_FORMAL5_TRAINING_AND_FREEZE_ONLY','A2_NOT_FROZEN')
    need(r['run_directory']==d['a2_run_directory'],'A2_RUN_MISMATCH')
    need(r['design_sha256']==d['a2_design_sha256'] and r['source_manifest_sha256']==d['a2_source_manifest_sha256'],'A2_SOURCE_IDENTITY_DRIFT')
    for k in ('formal_training_complete','all_20_checkpoints_frozen_before_any_outer_inference',
              'all_4000_training_epochs_completed','all_five_folds_four_arms_input_pairing_verified',
              'source_and_training_validation_inputs_unchanged'):
        need(r[k] is True,'A2_COMPLETION_GATE_'+k)
    for k in ('outer_test_inference','bootstrap_executed','hyperparameter_search',
              'fusion_or_weight_selection_executed','historical_checkpoints_reused'):
        need(r[k] is False,'A2_SCOPE_GATE_'+k)
    models=r['trained_models']
    need(len(models)==20 and {(x['outer_fold'],x['arm']) for x in models}=={(f,a) for f in range(5) for a in ARMS},'A2_MODEL_SET_INVALID')
    need(len({x['checkpoint_sha256'] for x in models})==20,'A2_CHECKPOINT_SHA_SET_INVALID')
    for x in models:
        f=x['outer_fold'];vf=(f+1)%5
        need(x['fixed_validation_fold']==vf and x['epochs_completed']==200,'A2_MODEL_EPOCH_OR_ROLE')
        need(x['train_rois']==2978-COUNTS[f]-COUNTS[vf] and x['validation_rois']==COUNTS[vf],'A2_MODEL_COUNTS')
        need(x['optimizer_steps']==math.ceil(x['train_rois']/32)*200,'A2_OPTIMIZER_STEPS')
        need(1<=x['best_epoch']<=200 and math.isfinite(x['best_validation_ordinal_objective']),'A2_SELECTION_FIELDS')
        for k in ('shared_initialization_verified','all_epoch_sampling_and_augmented_inputs_paired','gradient_gates_pass','checkpoint_roundtrip'):
            need(x[k] is True,'A2_MODEL_GATE_'+k)
    return models

def a2_source_and_state(d):
    a2=Path(d['a2_run_directory'])
    need(a2.parent==RUN_PARENT and re.fullmatch('paper2_w25_a2_matched_aux_formal5_20261022_[0-9a-f]{12}',a2.name) is not None,'A2_PATH_INVALID')
    gate(a2/'01_FORMAL_TRAIN_FREEZE_REVIEW.json',d['a2_freeze_report_sha256'],'A2_REPORT')
    gate(a2/'02_FORMAL_SAFE_RESULT.zip',d['a2_safe_bundle_sha256'],'A2_BUNDLE')
    gate(a2/'source/13_FORMAL_SOURCE_SHA256.json',d['a2_source_manifest_sha256'],'A2_MANIFEST')
    need(sha(HERE/'04_A2_FROZEN_TRAIN_REPORT.json')==d['a2_freeze_report_sha256'],'PACKAGED_A2_REPORT_DRIFT')
    r=read(a2/'01_FORMAL_TRAIN_FREEZE_REVIEW.json');check_freeze_report(r,d)
    fm=read(a2/'source/13_FORMAL_SOURCE_SHA256.json')
    for n,h in fm['files'].items():gate(safe_child(a2/'source',n),h,'A2_SOURCE_'+n)
    formal=load_module(a2/'source/11_remote_matched_aux_formal5.py','a3_verified_a2_controller')
    formal.validate_sources(a2/'source')
    need(not formal.worker_live(a2),'A2_TRAINING_WORKER_STILL_LIVE')
    need(read(a2/'00_PROGRESS.json')['state']=='PASS_FORMAL5_TRAINING_AND_FREEZE_ONLY','A2_PROGRESS_NOT_FINAL')
    return a2,r,formal

def snapshot_a2(a2):
    files=[]
    for p in sorted(a2.rglob('*')):
        need(not p.is_symlink(),'A2_TREE_SYMLINK')
        if p.is_file():PROTECTED[str(p)]=sha(p);files.append(p)
    need(len(files)>20,'A2_TREE_INCOMPLETE')
    return len(files)

def verify_histories(a2,r):
    """Read actual 4000 epoch records, not only the aggregate pass flags."""
    n=0;stream_checks=0;source_histories=0
    for f in range(5):
        reference=None
        for arm in ARMS:
            x=next(x for x in r['trained_models'] if x['outer_fold']==f and x['arm']==arm)
            ck=safe_child(a2,x['checkpoint_relative']);gate(ck,x['checkpoint_sha256'],'FROZEN_CHECKPOINT')
            history=read(ck.parent/'SERVER_ONLY_TRAINING_HISTORY.json')
            stream=read(ck.parent/'SERVER_ONLY_MATCHED_INPUT_HASHES.json')
            summary=read(ck.parent/'00_TRAINED_AND_FROZEN.json')
            need(summary==x,'PER_MODEL_FREEZE_REPORT_DRIFT')
            need(len(history)==len(stream)==200,'EPOCH_HISTORY_LENGTH')
            need([h['epoch'] for h in history]==list(range(1,201)),'EPOCH_HISTORY_ORDER')
            need(all(math.isfinite(h['train_objective']) and math.isfinite(h['validation_ordinal_objective']) for h in history),'NONFINITE_HISTORY')
            best=min(history,key=lambda h:h['validation_ordinal_objective'])
            need(best['epoch']==x['best_epoch'] and best['validation_ordinal_objective']==x['best_validation_ordinal_objective'],'FIXED_MINIMUM_SELECTION_MISMATCH')
            for e,t in enumerate(stream,1):
                need(t['epoch']==e and t['seen_rows']==x['train_rois'],'INPUT_STREAM_RECORD_INVALID')
                need(all(re.fullmatch('[0-9a-f]{64}',t[k]) for k in ('sample_sha256','augmented_sha256')),'INPUT_STREAM_DIGEST_INVALID')
            if reference is None:reference=stream
            else:
                need(stream==reference,'FULL_EPOCH_INPUT_PAIRING_MISMATCH');stream_checks+=200
            n+=200;source_histories+=1
    return {'epoch_history_rows':n,'training_histories':source_histories,'epoch_pairwise_stream_checks':stream_checks,
            'minimum_validation_selection_checked_models':20,'all_pass':True}

def safe_progress(run,state,**kwargs):
    x={'schema':'W25_A3_EVALUATION_PROGRESS_V1','state':state,'stage':STAGE,
       'updated_unix':time.time(),'worker_pid':os.getpid(),'new_training':False,'completed_models':0}
    x.update(kwargs);dump(run/'00_PROGRESS.json',x)

def postcheck():
    for p,h in PROTECTED.items():need(Path(p).is_file() and sha(p)==h,'PROTECTED_SOURCE_OR_CHECKPOINT_CHANGED')
    return len(PROTECTED)

def freeze_model_metadata(ck,x,d):
    need(ck['arm']==x['arm'] and ck['outer_fold']==x['outer_fold'] and ck['fixed_validation_fold']==x['fixed_validation_fold'],'CHECKPOINT_ROLE_MISMATCH')
    need(ck['best_epoch']==x['best_epoch'] and ck['best_validation_ordinal_objective']==x['best_validation_ordinal_objective'],'CHECKPOINT_SELECTION_MISMATCH')
    need(ck['alpha']==.125 and ck['seed']==42 and ck['training_budget_epochs']==200,'CHECKPOINT_CONFIGURATION_DRIFT')
    need(ck['design_sha256']==d['a2_design_sha256'] and ck['outer_test_opened'] is False and ck['smoke_checkpoint_reused'] is False,'CHECKPOINT_FREEZE_METADATA_DRIFT')

def evaluate_all(run,report,d):
    global STAGE
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    stats=load_module(HERE/'02_patient_paired_metrics.py','a3_paired_stats')
    a2,r,formal=a2_source_and_state(d)
    report['a2_files_protected_before']=snapshot_a2(a2)
    STAGE='VERIFY_ACTUAL_CHECKPOINTS_AND_4000_EPOCH_RECORDS';safe_progress(run,'VERIFYING_FROZEN_INPUTS')
    report['history_and_selection_audit']=verify_histories(a2,r)
    core=formal.core_load();core.validate_source_bundle()
    report['gpu_gate']=core.gpu_gate()
    need(torch.cuda.is_available(),'CUDA_REQUIRED')
    runtime={'python':sys.version.split()[0],'torch':str(torch.__version__),'cuda':torch.version.cuda,
             'cudnn':torch.backends.cudnn.version(),'gpu_name':torch.cuda.get_device_name(0)}
    need(runtime==r['runtime'],'A2_TO_A3_RUNTIME_CHANGED_REVIEW_REQUIRED');report['runtime']=runtime
    device=torch.device('cuda:0');free,_=torch.cuda.mem_get_info(device);need(free>=1024**3,'GPU_MEMORY_GATE')
    common=core.source_load(a2/'source/common_step04a.py','common_step04a')
    prog=core.source_load(a2/'source/ornet_progression_aux.py','ornet_progression_aux')
    s54=core.source_load(a2/'source/source54.py','a3_source54')
    obj=core.source_load(a2/'source/matched_aux_objectives.py','matched_aux_objectives')
    common.SCI=run
    legacy=core.source_load(a2/'source/parent_r09_trainer.py.txt','a3_legacy_r09')
    need(common.CNN_BATCH==32 and tuple(obj.ARM_NAMES)==ARMS,'INFERENCE_CONFIG_DRIFT')
    formal_source=read(a2/'source/14_A1_SMOKE_REVIEW.json')
    manifest=common.CTRL/'02_RESOLVED_EXPERIMENT_MANIFEST.csv'
    gate(manifest,formal_source['control_manifest_sha256_observed_and_identity_checked'],'CONTROL_MANIFEST')
    gate(common.FORMAL_WRAPPER,common.SHA_WRAPPER,'ROLE_AUDITOR')
    gate(core.CANON_PATH,core.CANON_SHA,'CANONICAL');gate(core.R11_PATH,core.R11_SHA,'R11_IDENTITY')
    with zipfile.ZipFile(core.R11_PATH) as z:
        b=z.read('01_SELECTED_WEIGHT_SYSTEM_POOLED_OOF.csv')
        need(hashlib.sha256(b).hexdigest()==core.R11_MEMBER_SHA,'R11_MEMBER_SHA_DRIFT')
        auth=core.csv_bytes(b)
    df=pd.read_csv(manifest,encoding='utf-8-sig');canon=pd.read_csv(core.CANON_PATH,encoding='utf-8-sig')
    report['cohort_identity']=core.validate_rows(df,canon,auth);del canon,auth
    # The A2 input hashes are retrospective integrity references, not inferred here.
    hashes=read(a2/'SERVER_ONLY_PROTECTED_INPUT_HASHES.json')
    need(len(hashes)==r['protected_files_checked'],'A2_INPUT_HASH_COUNT')
    for p,h in hashes.items():gate(Path(p),h,'A2_PROTECTED_INPUT')
    for p in set(map(str,df.iq8_path)):
        need(p in hashes,'INPUT_ABSENT_FROM_A2_PROTECTED_MANIFEST')
    fw=common.load_module(common.FORMAL_WRAPPER,'a3_role_auditor')
    streams=[];report['inference_checks']=[]
    for f in range(5):
        vf=(f+1)%5;tr,va,te,audit=fw.role_audit(df,f)
        need(core.integer(audit['fixed_validation_fold'])==vf,'ROLE_VF_MISMATCH')
        expected=(df[~df.unified_outer_fold.astype(int).isin([f,vf])],df[df.unified_outer_fold.astype(int)==vf],df[df.unified_outer_fold.astype(int)==f])
        for actual,exp in zip((tr,va,te),expected):
            need(set(map(core.integer,actual.protocol_v4_row_id))==set(map(core.integer,exp.protocol_v4_row_id)),'ROLE_ROW_MISMATCH')
        sets=[set(map(core.pid,x.patient_id)) for x in (tr,va,te)]
        need(not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'PATIENT_ROLE_LEAKAGE')
        need(len(te)==COUNTS[f] and len(va)==COUNTS[vf],'INFERENCE_COUNTS')
        del tr,expected,sets
        target=te[common.label_col(te)].astype(int).to_numpy()
        all_predictions={}
        for arm in ARMS:
            STAGE='VALIDATION_PARITY_THEN_FROZEN_OUTER_INFERENCE'
            safe_progress(run,'EVALUATING',outer_fold=f,arm=arm,completed_models=len(report['inference_checks']))
            x=next(x for x in r['trained_models'] if x['outer_fold']==f and x['arm']==arm)
            ckpath=safe_child(a2,x['checkpoint_relative']);gate(ckpath,x['checkpoint_sha256'],'CHECKPOINT_PRE_INFERENCE')
            ck=common.torch_load(ckpath,'cpu');freeze_model_metadata(ck,x,d)
            common.set_all_seeds(42);model=core.create_model(common,s54,prog,device)
            obj.configure_arm(model,arm);model.load_state_dict(ck['model_state_dict'],strict=True)
            before=core.state_digest(model.state_dict())
            val_ds=common.SelectedPreprocDataset(va,'coral_msaux',False,s54,42)
            val_loader=DataLoader(val_ds,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
            vl,vy,vp,vpred=legacy.evaluate(model,val_loader,device,s54)
            need(abs(vl-x['best_validation_ordinal_objective'])<=1e-8,'FROZEN_VALIDATION_REPRODUCTION_FAILED')
            need(np.array_equal(vy,va[common.label_col(va)].astype(int).to_numpy()),'VAL_ORDER_MISMATCH')
            # Fixed-checkpoint validation is retained only as server-side provenance;
            # it is never used to select a checkpoint, arm, alpha or weight here.
            valout=va[['protocol_v4_row_id','patient_id','unified_outer_fold']].copy()
            valout['y_true']=vy;valout['prediction']=vpred
            for j in range(4):valout['p'+str(j)]=vp[:,j]
            valout.to_csv(run/f'SERVER_ONLY_VAL_outer{f}_{arm}.csv',index=False)
            test_ds=common.SelectedPreprocDataset(te,'coral_msaux',False,s54,42)
            need(not test_ds.training and test_ds.native_aug is None,'TEST_AUGMENTATION_FORBIDDEN')
            need(list(map(core.integer,test_ds.frame.protocol_v4_row_id))==list(map(core.integer,te.protocol_v4_row_id)),'TEST_INDEX_FRAME_ALIGNMENT')
            test_loader=DataLoader(test_ds,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
            _,yy,pp,pred=legacy.evaluate(model,test_loader,device,s54)
            need(np.array_equal(yy,target),'OUTER_TRUE_LABEL_ORDER')
            need(pp.shape==(len(te),4) and np.isfinite(pp).all() and (pp>=0).all() and (pp<=1+1e-6).all() and np.allclose(pp.sum(1),1,atol=2e-6),'OUTER_PROBABILITY_INVALID')
            need(np.array_equal(pred,np.argmax(pp,axis=1)),'DECODER_DRIFT')
            need(before==core.state_digest(model.state_dict()),'INFERENCE_MODEL_STATE_CHANGED')
            gate(ckpath,x['checkpoint_sha256'],'CHECKPOINT_POST_INFERENCE')
            out=te[['protocol_v4_row_id','patient_id','unified_outer_fold']].copy()
            out['y_true']=yy;out['prediction']=pred;out['arm']=arm
            for j in range(4):out['p'+str(j)]=pp[:,j]
            streams.append(out);all_predictions[arm]=stats.point(yy,pred)
            report['inference_checks'].append({'outer_fold':f,'arm':arm,'outer_rois':len(te),'fixed_validation_rois':len(va),
                'checkpoint_sha256':x['checkpoint_sha256'],'checkpoint_selected_before_test':True,
                'validation_loss_reproduction_max_abs':float(abs(vl-x['best_validation_ordinal_objective'])),
                'prediction_decoder':'frozen CORAL probabilities argmax; auxiliary scalar unused',
                'checkpoint_and_in_memory_state_unchanged':True})
            print(f'W25_A3_OUTER_INFERENCE|fold={f}|arm={arm}|rows={len(te)}|FROZEN_SHA=PASS',flush=True)
            del model,ck,val_ds,val_loader,test_ds,test_loader;torch.cuda.empty_cache()
        report.setdefault('fold_results',[]).append({'outer_fold':f,'patients':int(te.patient_id.nunique()),'rois':len(te),'arms':all_predictions})
    pooled=pd.concat(streams,ignore_index=True)
    need(len(pooled)==11912,'POOLED_PREDICTION_COUNT')
    wide=None;y=None;patients=None
    report['pooled_results']={}
    for arm in ARMS:
        q=pooled[pooled.arm==arm].sort_values('protocol_v4_row_id').reset_index(drop=True)
        need(len(q)==2978 and q.protocol_v4_row_id.nunique()==2978 and q.patient_id.nunique()==468,'POOLED_ARM_IDENTITIES')
        identity=q[['protocol_v4_row_id','patient_id','unified_outer_fold','y_true']]
        if wide is None:
            wide=identity.copy();y=q.y_true.to_numpy(dtype=int);patients=q.patient_id.map(core.pid).to_numpy()
        else:need(identity.equals(wide[['protocol_v4_row_id','patient_id','unified_outer_fold','y_true']]),'ARM_POOLED_IDENTITY_DRIFT')
        wide[arm]=q.prediction.to_numpy(dtype=int)
        report['pooled_results'][arm]=stats.point(y,wide[arm].to_numpy(dtype=int))
    pooled.to_csv(run/'SERVER_ONLY_OUTER_PROBABILITIES_ALL_ARMS.csv',index=False)
    wide.to_csv(run/'SERVER_ONLY_MATCHED_OOF_2978.csv',index=False)
    dump(run/'SERVER_ONLY_PREDICTION_SHA.json',{n:sha(run/n) for n in ('SERVER_ONLY_OUTER_PROBABILITIES_ALL_ARMS.csv','SERVER_ONLY_MATCHED_OOF_2978.csv')})
    report['outer_prediction_complete']=True
    STAGE='PAIRED_PATIENT_CLUSTER_BOOTSTRAP'
    last=[-1]
    def progress(done,total):
        if done==total or done//1000!=last[0]:
            last[0]=done//1000;safe_progress(run,'BOOTSTRAPPING',completed_models=20,bootstrap_completed=done,bootstrap_total=total)
            print(f'W25_A3_PATIENT_BOOTSTRAP|completed={done}/{total}',flush=True)
    report['paired_bootstrap']=stats.paired_patient_bootstrap(y,wide[list(ARMS)].to_numpy(dtype=int),patients,
        iterations=d['bootstrap_repeats'],seed=d['bootstrap_seed'],callback=progress)
    report['bootstrap_executed']=True
    # Preserve undefined counts in the report and fail rather than silently report
    # fewer than the frozen number of replicates for this 468-patient cohort.
    need(all(v['valid_replicates']==d['bootstrap_repeats'] for c in report['paired_bootstrap']['contrasts'] for v in c['metrics'].values()),'UNDEFINED_BOOTSTRAP_REPLICATES_REVIEW_REQUIRED')
    STAGE='POST_PROTECTION'
    report['protected_files_checked_after']=postcheck()
    report['a2_checkpoints_sources_histories_inputs_unchanged']=True
    report['phadke_matched_fusion_completed']=False
    report['historical_R09_R10_R11_replaced']=False
    report['statistical_result_interpretation_requires_review']=True

def export_result(run,report):
    dump(run/'01_OUTER_EVALUATION_REVIEW.json',report)
    payload={'01_OUTER_EVALUATION_REVIEW.json':(run/'01_OUTER_EVALUATION_REVIEW.json').read_bytes(),
       DESIGN:(HERE/DESIGN).read_bytes(),MANIFEST:(HERE/MANIFEST).read_bytes(),
       '04_A2_FROZEN_TRAIN_REPORT.json':(HERE/'04_A2_FROZEN_TRAIN_REPORT.json').read_bytes()}
    payload['98_MEMBER_SHA256.json']=(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in payload.items()},indent=2)+'\n').encode()
    tmp=run/'02_EVAL_SAFE_RESULT.zip.partial'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
    tmp.replace(run/'02_EVAL_SAFE_RESULT.zip')
    dump(run/'99_RESULT_IDENTITY.json',{'report_sha256':sha(run/'01_OUTER_EVALUATION_REVIEW.json'),'bundle_sha256':sha(run/'02_EVAL_SAFE_RESULT.zip')})

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

def validate_run_sources(run,manifest_sha):
    need(not (run/'source').is_symlink(),'SOURCE_SYMLINK')
    x=read(run/'00_LAUNCH.json')
    need(x['source_manifest_sha256']==manifest_sha and x['design_sha256']==sha(HERE/DESIGN),'EVALUATION_RUN_IDENTITY_MISMATCH')
    need(sha(run/'source'/MANIFEST)==manifest_sha,'INSTALLED_EVAL_MANIFEST_DRIFT');validate_sources(run/'source')

def status_snapshot(run,manifest_sha):
    base={'schema':'W25_A3_FROZEN_EVALUATION_STATUS_V1','run_directory':str(run),'source_manifest_sha256':manifest_sha,
          'design_sha256':sha(HERE/DESIGN),'new_training':False,'tuning':False,'fusion':False,'restart_permitted':False}
    if not run.exists():base.update(state='NOT_STARTED',worker_live=False,safe_result_available=False);return base
    validate_run_sources(run,manifest_sha)
    p=read(run/'00_PROGRESS.json') if (run/'00_PROGRESS.json').is_file() else {'state':'STARTING'}
    live=worker_live(run);base.update(progress=p,worker_live=live)
    ident=run/'99_RESULT_IDENTITY.json'
    if ident.exists():
        h=read(ident);rr=run/'01_OUTER_EVALUATION_REVIEW.json';zz=run/'02_EVAL_SAFE_RESULT.zip'
        need(sha(rr)==h['report_sha256'] and sha(zz)==h['bundle_sha256'],'RESULT_HASH_DRIFT')
        r=read(rr);need(r['new_training'] is False and r['tuning'] is False and r['fusion'] is False,'RESULT_SCOPE_DRIFT')
        base.update(state=r['status'],report_sha256=h['report_sha256'],bundle_sha256=h['bundle_sha256'],safe_result_available=True)
    elif live:base.update(state=p['state'],safe_result_available=False)
    else:base.update(state='STOPPED_OR_STARTUP_UNCONFIRMED_NO_AUTORESTART',safe_result_available=False)
    return base

def worker(run,manifest_sha):
    require_run(run);validate_run_sources(run,manifest_sha)
    need(HERE==run/'source','WORKER_NOT_PERSISTENT_SOURCE')
    with (run/'WORKER.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise GateError('EVALUATION_ALREADY_RUNNING')
        cur=process_identity(os.getpid());need(cur is not None,'PID_IDENTITY_UNAVAILABLE')
        dump(run/'WORKER_PID.json',{'pid':os.getpid(),'start_ticks':cur['start_ticks']})
        safe_progress(run,'PREPARING_EVALUATION')
        _,d=validate_sources()
        r={'schema':'W25_A3_FROZEN_OUTER_REVIEW_V1','status':'RUNNING','created_unix':time.time(),
           'run_directory':str(run),'a2_run_directory':d['a2_run_directory'],
           'a2_freeze_report_sha256':d['a2_freeze_report_sha256'],'source_manifest_sha256':manifest_sha,
           'design_sha256':sha(HERE/DESIGN),'new_training':False,'tuning':False,'fusion':False,
           'outer_prediction_complete':False,'bootstrap_executed':False,
           'single_seed_internal_retrospective':True,'statistical_superiority_claim_automatically_authorized':False}
        code=0
        try:
            evaluate_all(run,r,d);r['status']=COMPLETE
        except (Exception,SystemExit) as e:
            code=2;r['status']=FAILED;r['failed_stage']=STAGE
            r['error_code']=str(e) if isinstance(e,GateError) or e.__class__.__name__=='GateError' else 'RUNTIME_'+type(e).__name__
            (run/'SERVER_ONLY_EXCEPTION.log').write_text(traceback.format_exc(),encoding='utf-8')
            try:r['protected_files_checked_after_failure']=postcheck();r['protected_files_unchanged']=True
            except Exception:r['protected_files_unchanged']=False
        r['finished_unix']=time.time();export_result(run,r)
        safe_progress(run,r['status'],completed_models=len(r.get('inference_checks',[])),safe_result_available=True,error_code=r.get('error_code'))
        print('W25_A3_FINAL_STATUS='+r['status'],flush=True);return code

def evaluate_once(run,manifest_sha):
    require_run(run)
    with (RUN_PARENT/(run.name+'.START.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise GateError('ANOTHER_EVALUATE_REQUEST_ACTIVE_USE_STATUS')
        if run.exists():
            s=status_snapshot(run,manifest_sha);s['action']='EXISTING_EVALUATION_NOT_RESTARTED';return s
        m,d=validate_sources();a2,r,formal=a2_source_and_state(d)
        # Check GPU contention before scheduling, repeated inside the worker.
        core=formal.core_load();core.gpu_gate()
        run.mkdir();source=run/'source';source.mkdir()
        for n in m['files']:shutil.copyfile(HERE/n,source/n)
        shutil.copyfile(HERE/MANIFEST,source/MANIFEST);validate_sources(source)
        dump(run/'00_LAUNCH.json',{'schema':'W25_A3_EVALUATION_LAUNCH_V1','created_unix':time.time(),
             'design_sha256':sha(HERE/DESIGN),'source_manifest_sha256':manifest_sha,
             'a2_freeze_report_sha256':d['a2_freeze_report_sha256'],'new_training':False,'automatic_restart_or_resume':False})
        safe_progress(run,'STARTING')
        cmd=[sys.executable,'-B',str(source/Path(__file__).name),'--mode','worker','--manifest-sha',manifest_sha]
        with (run/'SERVER_ONLY_WORKER.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                  cwd=source,start_new_session=True,close_fds=True)
        for _ in range(100):
            if (run/'WORKER_PID.json').exists() or proc.poll() is not None:break
            time.sleep(.1)
        s=status_snapshot(run,manifest_sha);s['action']='NEW_FROZEN_EVALUATION_SCHEDULED';return s

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['evaluate','status','worker'],required=True)
    ap.add_argument('--manifest-sha',required=True);ap.add_argument('--out',type=Path);ap.add_argument('--bundle',type=Path)
    a=ap.parse_args();need(sha(HERE/MANIFEST)==a.manifest_sha,'EXPECTED_EVAL_MANIFEST_HASH')
    validate_sources();run=run_for_design(sha(HERE/DESIGN));require_run(run)
    if a.mode=='worker':return worker(run,a.manifest_sha)
    need(a.out is not None and a.bundle is not None,'CLIENT_OUTPUT_ARGUMENTS')
    for p in (a.out,a.bundle):need(re.fullmatch(r'/PATH/TO/TEMP/lss_w25a3_[A-Za-z0-9]{8}',str(p.parent)) is not None,'CLIENT_OUTPUT_PATH_INVALID')
    s=evaluate_once(run,a.manifest_sha) if a.mode=='evaluate' else status_snapshot(run,a.manifest_sha)
    dump(a.out,s)
    if s.get('safe_result_available'):shutil.copyfile(run/'02_EVAL_SAFE_RESULT.zip',a.bundle)
    print('W25_A3_JOB_STATE='+s['state'],flush=True)
    print('W25_A3_RUN_DIRECTORY='+str(run),flush=True)
    p=s.get('progress',{});print(f"W25_A3_PROGRESS|models={p.get('completed_models',0)}/20|bootstrap={p.get('bootstrap_completed','-')}/{p.get('bootstrap_total','-')}",flush=True)
    print('W25_A3_NEW_TRAINING=False|NO_RETUNING=True|STATUS_NEVER_STARTS_WORKER=True',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
