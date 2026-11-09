#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W25-A2: four matched arms x five folds x 200 epochs; TRAIN + INNER VAL ONLY.
Derives model, objectives, preprocessing, evaluation and source gates from the
byte-exact successful A1 package. Does not run A1.main or the legacy trainer.main.
Start returns after creating a durable job; Status never starts/resumes training.
No outer prediction, no efficacy statistics, no tuning, no automatic restart.
"""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, importlib.util, json, math, os, re
import shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path
sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
HERE = Path(__file__).resolve().parent
RUN_PARENT = Path('/PATH/TO/RESEARCH_WORKSPACE/server_runs')
PREFIX = 'paper2_w25_a2_matched_aux_formal5_20261022_'
ARM_NAMES = ('A0_NATURAL', 'A1_SCALED', 'A2_BINARY_P', 'A3_GRADE_FIXED')
STAGE = 'PREPARING'
SOURCE_MANIFEST = '13_FORMAL_SOURCE_SHA256.json'
DESIGN_NAME = '12_FORMAL_DESIGN.json'
SMOKE_NAME = '14_A1_SMOKE_REVIEW.json'
COMPLETE = 'PASS_FORMAL5_TRAINING_AND_FREEZE_ONLY'
FAILED = 'FAIL_FORMAL5_DIAGNOSTIC_SAVED'
TEST_COUNTS = (618,596,557,616,591)

class GateError(RuntimeError): pass

def need(ok, code):
    if not ok: raise GateError(code)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def run_for_design(design_sha):
    need(re.fullmatch('[0-9a-f]{64}',design_sha) is not None,'INVALID_DESIGN_SHA')
    return RUN_PARENT/(PREFIX+design_sha[:12])

def require_inside_run(run):
    need(run.parent==RUN_PARENT and re.fullmatch(PREFIX+'[0-9a-f]{12}',run.name) is not None,'INVALID_RUN_PATH')
    need(not run.is_symlink() and RUN_PARENT.is_dir(),'RUN_ROOT_UNSAFE_OR_MISSING')

def validate_design(d):
    need(d['schema']=='W25_A2_FORMAL5_TRAIN_FREEZE_DESIGN_V1','DESIGN_SCHEMA')
    need(d['outer_folds']==[0,1,2,3,4] and d['fixed_validation_folds']==[1,2,3,4,0],'FOLD_DESIGN_DRIFT')
    need(tuple(d['arms'])==ARM_NAMES and d['epochs_per_arm_fold']==200,'ARM_OR_EPOCH_DRIFT')
    need(d['seed']==42 and d['alpha']==0.125 and d['batch_size']==32,'NUMERIC_DESIGN_DRIFT')
    need(d['outer_test_inference_authorized'] is False and d['bootstrap_authorized'] is False and d['tuning_authorized'] is False,'SCOPE_DRIFT')
    need(d['historical_checkpoint_reuse_authorized'] is False and d['early_stopping'] is False,'TRAINING_REUSE_OR_EARLYSTOP_DRIFT')
    need(d['scheduled_training_runs']==20 and d['learning_rate']==0.0007723 and d['weight_decay']==0.002453 and d['explicit_fc_l2']==0.01,'BUDGET_OR_OPTIMIZER_DRIFT')

def validate_sources(root=HERE):
    m=read(root/SOURCE_MANIFEST)
    need(m['schema']=='W25_A2_FORMAL_SOURCE_MANIFEST_V1','SOURCE_MANIFEST_SCHEMA')
    for name,h in m['files'].items():
        need(name==Path(name).name and re.fullmatch('[0-9a-f]{64}',h) is not None,'INVALID_SOURCE_ENTRY')
        need((root/name).is_file() and not (root/name).is_symlink() and sha(root/name)==h,'SOURCE_SHA_DRIFT_'+name)
    d=read(root/DESIGN_NAME);validate_design(d)
    need(m['files'][DESIGN_NAME]==sha(root/DESIGN_NAME),'DESIGN_HASH_MISSING')
    return m,d

def validate_smoke(s):
    need(s['schema']=='W25_A1_MATCHED_AUX_SMOKE_RESULT_V1' and s['status']=='PASS_IMPLEMENTATION_SMOKE_ONLY','A1_NOT_PASS')
    need(len(s['arms'])==4 and tuple(a['arm'] for a in s['arms'])==ARM_NAMES,'A1_ARM_SET_DRIFT')
    for a in s['arms']:
        need(a['epochs']==2 and a['optimizer_steps']==112 and a['train_rows_per_epoch']==1764 and a['validation_rows']==596,'A1_COUNTS_DRIFT')
        need(all(a[k] is True for k in ('shared_initial_state_matches','gradient_gates_pass','checkpoint_roundtrip')),'A1_ARM_GATE_NOT_PASS')
        need(a['outer_test_evaluated'] is False,'A1_OUTER_TEST_ALREADY_OPENED')
    need(s['pairing_gates']['sampled_minibatch_order_equal_all_arms'] is True and s['pairing_gates']['augmented_input_tensors_equal_all_arms'] is True,'A1_PAIRING_NOT_PASS')
    need(s['legacy_one_step_parity']['loss_bit_equal'] is True and s['legacy_one_step_parity']['one_adamw_update_state_bit_equal'] is True,'A1_LEGACY_PARITY_NOT_PASS')
    need(s['source_and_training_validation_inputs_unchanged'] is True and s['historical_A3_checkpoint_reuse_authorized'] is False,'A1_REUSE_SCOPE_DRIFT')

def core_load():
    spec=importlib.util.spec_from_file_location('a1_exact_parent',HERE/'01_remote_matched_aux_smoke.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def paired_epoch_gate(records, reference, epoch):
    """Input/order fingerprints remain server-only; result exports boolean parity."""
    need(records['epoch']==epoch and records['seen_rows']>0,'EPOCH_RECORD_INVALID')
    if reference is not None:
        need(reference['epoch']==epoch and reference['seen_rows']==records['seen_rows'],'EPOCH_ROW_PAIRING_FAILED')
        need(reference['sample_sha256']==records['sample_sha256'],'SAMPLED_MINIBATCH_ORDER_NOT_MATCHED')
        need(reference['augmented_sha256']==records['augmented_sha256'],'AUGMENTED_INPUT_STREAM_NOT_MATCHED')

def safe_progress(run,state,**kwargs):
    p={'schema':'W25_A2_FORMAL5_PROGRESS_V1','state':state,'updated_unix':time.time(),
       'worker_pid':os.getpid(),'stage':STAGE,'outer_test_inference':False,'bootstrap_executed':False,
       'scheduled_runs':20,'epochs_per_run':200,'completed_runs':0}
    p.update(kwargs);dump(run/'00_PROGRESS.json',p)

def torch_save_atomic(torch,obj,path):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    torch.save(obj,tmp);tmp.replace(path)

def train_one_arm(core,common,s54,prog,obj,legacy,tr,va,run,outer,vf,arm,initial,device,reference,completed):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,WeightedRandomSampler
    common.set_all_seeds(42)
    # These are the same constructions/order as A1. No outer-test Dataset exists.
    train_ds=common.SelectedPreprocDataset(tr,'coral_msaux',True,s54,42)
    val_ds=common.SelectedPreprocDataset(va,'coral_msaux',False,s54,42)
    labels=tr[common.label_col(tr)].astype(int).to_numpy()
    counts=np.bincount(labels,minlength=4)
    cw=np.divide(1.0,counts,out=np.zeros_like(counts,dtype=float),where=counts>0)
    sw=torch.as_tensor(cw[labels],dtype=torch.double)
    gen=torch.Generator();gen.manual_seed(42)
    sampler=WeightedRandomSampler(sw,num_samples=len(sw),replacement=True,generator=gen)
    train_loader=DataLoader(train_ds,batch_size=common.CNN_BATCH,shuffle=False,sampler=sampler,num_workers=0,pin_memory=True)
    val_loader=DataLoader(val_ds,batch_size=common.CNN_BATCH,shuffle=False,num_workers=0,pin_memory=True)
    model=core.create_model(common,s54,prog,device)
    need(core.all_same_state(model.state_dict(),initial),'FORMAL_ARM_INITIAL_RANDOM_STREAM_DRIFT')
    obj.configure_arm(model,arm)
    optimizer=torch.optim.AdamW(model.parameters(),lr=common.CNN_LR,weight_decay=common.CNN_WD)
    ad=run/'models'/f'outer_{outer}'/arm;ad.mkdir(parents=True,exist_ok=False)
    ck=ad/'BEST_MIN_FIXED_INNER_ORDINAL_OBJECTIVE.pth'
    best=float('inf');best_epoch=None;history=[];stream=[];steps=0
    for epoch in range(1,201):
        model.train();seen=0;loss_sum=0.0;h1=hashlib.sha256();h2=hashlib.sha256()
        for x,y,r,idx in train_loader:
            h1.update(idx.numpy().astype('<i8').tobytes());h2.update(x.contiguous().numpy().tobytes())
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True);logits,risk=model(x)
            joint,_,_=obj.objective(model,logits,risk,y,arm,s54,prog,common.CNN_FC_L2)
            need(bool(torch.isfinite(joint).item()),'NONFINITE_TRAIN_LOSS')
            joint.backward()
            need(all(p.grad is None for p in model.risk_head.parameters()),'LEGACY_RISK_HEAD_RECEIVED_GRADIENT')
            ag=[p.grad for p in model.progression_aux.parameters()]
            if arm in ('A0_NATURAL','A1_SCALED'):
                need(all(g is None for g in ag),'DISABLED_AUX_RECEIVED_GRADIENT')
            else:
                need(all(g is not None and bool(torch.isfinite(g).all()) for g in ag),'AUX_GRADIENT_MISSING_OR_NONFINITE')
            need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'NONFINITE_GRADIENT')
            optimizer.step();n=y.numel();seen+=n;steps+=1;loss_sum+=float(joint.detach().cpu())*n
        need(seen==len(tr),'TRAIN_EPOCH_ROW_COUNT_DRIFT')
        rec={'epoch':epoch,'seen_rows':seen,'sample_sha256':h1.hexdigest(),'augmented_sha256':h2.hexdigest()}
        paired_epoch_gate(rec,None if reference is None else reference[epoch-1],epoch);stream.append(rec)
        vloss,vy,vprob,vpred=legacy.evaluate(model,val_loader,device,s54)
        need(np.isfinite(vloss) and len(vy)==len(va) and np.isfinite(vprob).all(),'VALIDATION_FORWARD_FAILED')
        need(np.allclose(vprob.sum(1),1,atol=2e-6),'VALIDATION_PROBABILITIES_INVALID')
        history.append({'epoch':epoch,'train_objective':loss_sum/seen,'validation_ordinal_objective':float(vloss)})
        if vloss<best:
            best=float(vloss);best_epoch=epoch
            torch_save_atomic(torch,{'model_state_dict':copy.deepcopy(model.state_dict()),'best_epoch':epoch,
              'best_validation_ordinal_objective':best,'outer_fold':outer,'fixed_validation_fold':vf,
              'arm':arm,'alpha':.125,'seed':42,'design_sha256':sha(HERE/DESIGN_NAME),
              'training_budget_epochs':200,'outer_test_opened':False,'smoke_checkpoint_reused':False},ck)
        # Do not publish per-arm performance rankings while training; local history
        # is solely the locked checkpoint-selection record.
        dump(ad/'SERVER_ONLY_TRAINING_HISTORY.json',history)
        dump(ad/'SERVER_ONLY_MATCHED_INPUT_HASHES.json',stream)
        safe_progress(run,'TRAINING',outer_fold=outer,validation_fold=vf,arm=arm,epoch=epoch,
                      completed_runs=completed,completed_epoch_units=completed*200+epoch,
                      total_epoch_units=4000,paired_inputs_verified_to_epoch=epoch,latest_loss_finite=True)
        if epoch==1 or epoch%10==0:
            print(f'W25_A2_TRAIN|fold={outer}|arm={arm}|epoch={epoch}/200|FINITE=True|PAIRING=PASS',flush=True)
    need(steps==math.ceil(len(tr)/32)*200,'OPTIMIZER_STEP_COUNT_DRIFT')
    saved=common.torch_load(ck,'cpu');model.load_state_dict(saved['model_state_dict'],strict=True)
    reloaded,_,probs,_=legacy.evaluate(model,val_loader,device,s54)
    need(abs(reloaded-best)<=1e-8 and np.isfinite(probs).all(),'CHECKPOINT_ROUNDTRIP_FAILED')
    summary={'outer_fold':outer,'fixed_validation_fold':vf,'arm':arm,'epochs_completed':200,
      'train_rois':len(tr),'validation_rois':len(va),'optimizer_steps':steps,
      'shared_initialization_verified':True,'all_epoch_sampling_and_augmented_inputs_paired':True,
      'gradient_gates_pass':True,'checkpoint_roundtrip':True,'best_epoch':best_epoch,
      'best_validation_ordinal_objective':best,'checkpoint_relative':str(ck.relative_to(run)),
      'checkpoint_sha256':sha(ck),'initial_state_sha256':core.state_digest(initial),
      'outer_test_inference':False,'efficacy_comparison_completed':False}
    dump(ad/'00_TRAINED_AND_FROZEN.json',summary)
    del model,optimizer,train_loader,val_loader,train_ds,val_ds;torch.cuda.empty_cache()
    return summary,stream

def train_formal(run,report):
    global STAGE
    import numpy as np
    import pandas as pd
    import torch
    core=core_load();core.validate_source_bundle()
    smoke=read(HERE/SMOKE_NAME);validate_smoke(smoke)
    smoke_run=Path(smoke['new_run_directory'])
    need(smoke_run.parent==RUN_PARENT and smoke_run.name=='paper2_w25_a1_matched_aux_smoke_20261022_c861a7af5d14','SMOKE_RUN_IDENTITY')
    core.gate(smoke_run/'00_SMOKE_REVIEW.json',sha(HERE/SMOKE_NAME),'SUCCESSFUL_A1_SERVER_REPORT')
    report['gpu_gate']=core.gpu_gate()
    need(torch.cuda.is_available(),'CUDA_REQUIRED')
    device=torch.device('cuda:0');free,_=torch.cuda.mem_get_info(device);need(free>=1024**3,'GPU_FREE_MEMORY_GATE')
    runtime={'python':sys.version.split()[0],'torch':str(torch.__version__),'cuda':torch.version.cuda,
             'cudnn':torch.backends.cudnn.version(),'gpu_name':torch.cuda.get_device_name(0)}
    need(runtime==smoke['runtime'],'SMOKE_TO_FORMAL_RUNTIME_CHANGED_REVIEW_REQUIRED')
    report['runtime']=runtime
    common=core.source_load(HERE/'common_step04a.py','common_step04a')
    prog=core.source_load(HERE/'ornet_progression_aux.py','ornet_progression_aux')
    s54=core.source_load(HERE/'source54.py','a2_source54')
    obj=core.source_load(HERE/'matched_aux_objectives.py','matched_aux_objectives')
    common.SCI=run
    legacy=core.source_load(HERE/'parent_r09_trainer.py.txt','a2_legacy_r09')
    need(tuple(obj.ARM_NAMES)==ARM_NAMES and prog.SOURCE_ALPHA==obj.ALPHA==.125,'OBJECTIVE_DESIGN_DRIFT')
    need((common.CNN_EPOCHS,common.CNN_BATCH,common.CNN_LR,common.CNN_WD,common.CNN_FC_L2,common.CNN_DROPOUT)==(200,32,.0007723,.002453,.01,.5),'TRAINING_CONSTANT_DRIFT')
    core.gate(common.FORMAL_WRAPPER,common.SHA_WRAPPER,'EXACT_LEGACY_ROLE_AUDITOR')
    fw=common.load_module(common.FORMAL_WRAPPER,'a2_original_role_auditor')
    manifest=common.CTRL/'02_RESOLVED_EXPERIMENT_MANIFEST.csv'
    core.gate(manifest,smoke['control_manifest_sha256_observed_and_identity_checked'],'A1_MATCHED_CONTROL_MANIFEST')
    core.protect_observed(common.CTRL/'00_PREFLIGHT_PASS.json','ORIGINAL_CONTROL_PREFLIGHT')
    core.gate(core.CANON_PATH,core.CANON_SHA,'CANONICAL_IDENTITY')
    core.gate(core.R11_PATH,core.R11_SHA,'R11_IDENTITY')
    with zipfile.ZipFile(core.R11_PATH) as z:
        b=z.read('01_SELECTED_WEIGHT_SYSTEM_POOLED_OOF.csv')
        need(hashlib.sha256(b).hexdigest()==core.R11_MEMBER_SHA,'R11_MEMBER_SHA_DRIFT')
        authority=core.csv_bytes(b)
    df=pd.read_csv(manifest,encoding='utf-8-sig');canonical=pd.read_csv(core.CANON_PATH,encoding='utf-8-sig')
    report['training_manifest_identity']=core.validate_rows(df,canonical,authority)
    del canonical,authority
    report['folds']=[];report['trained_models']=[];report['model_training_started']=True
    for outer in range(5):
        STAGE='FOLD_ROLE_AND_INPUT_GATES'
        vf=(outer+1)%5;tr,va,te,audit=fw.role_audit(df,outer)
        need(core.integer(audit['fixed_validation_fold'])==vf,'FIXED_INNER_FOLD_DRIFT')
        expected=[df[~df.unified_outer_fold.astype(int).isin([outer,vf])],df[df.unified_outer_fold.astype(int)==vf],df[df.unified_outer_fold.astype(int)==outer]]
        for actual,exp in zip((tr,va,te),expected):
            need(set(map(core.integer,actual.protocol_v4_row_id))==set(map(core.integer,exp.protocol_v4_row_id)),'ROLE_IDENTITY_DRIFT')
        need((len(tr),len(va),len(te))==(2978-TEST_COUNTS[outer]-TEST_COUNTS[vf],TEST_COUNTS[vf],TEST_COUNTS[outer]),'FOLD_ROLE_COUNT_DRIFT')
        sets=[set(map(core.pid,part.patient_id)) for part in (tr,va,te)]
        need(not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'PATIENT_ROLE_LEAKAGE')
        role={'outer_fold':outer,'fixed_validation_fold':vf,'train_rois':len(tr),'validation_rois':len(va),
              'outer_rois_not_inferred':len(te),'train_patients':len(sets[0]),'validation_patients':len(sets[1]),
              'outer_patients_not_inferred':len(sets[2]),'patient_roles_disjoint':True,'outer_test_dataset_created':False}
        # No outer-test pixel dataset or forward pass. Other folds legitimately use
        # these patients as train/validation; this is not a globally held-out cohort.
        del te,expected,sets
        for p in dict.fromkeys(str(x) for x in list(tr.iq8_path)+list(va.iq8_path)):
            if p in core.PROTECTED:need(core.sha(p)==core.PROTECTED[p],'INPUT_CHANGED_BETWEEN_FOLDS')
            else:core.protect_observed(p,'TRAIN_VAL_IQ8_FILE')
        common.set_all_seeds(42);m=core.create_model(common,s54,prog,device)
        initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};del m
        need(core.state_digest(initial)==smoke['initialization']['full_model_state_sha256'],'SMOKE_TO_FORMAL_INITIAL_STATE_DRIFT')
        fd=run/'models'/f'outer_{outer}';fd.mkdir(parents=True,exist_ok=False)
        torch_save_atomic(torch,{'model_state_dict':initial,'outer_fold':outer,'seed':42},fd/'MATCHED_INITIAL_STATE.pth')
        STAGE='LEGACY_SINGLE_STEP_PARITY'
        probe=common.SelectedPreprocDataset(tr,'coral_msaux',False,s54,42)
        items=[probe[i] for i in range(min(8,len(probe)))];x=torch.stack([v[0] for v in items]).to(device);y=torch.stack([v[1] for v in items]).to(device)
        role['legacy_one_step_parity']=core.step_parity(initial,x,y,common,s54,prog,obj,device)
        del probe,items,x,y
        STAGE='MATCHED_FORMAL_TRAINING';ref=None
        for arm in ARM_NAMES:
            summ,stream=train_one_arm(core,common,s54,prog,obj,legacy,tr,va,run,outer,vf,arm,initial,device,ref,len(report['trained_models']))
            if ref is None:ref=stream
            report['trained_models'].append(summ)
            dump(run/'SERVER_ONLY_ACCUMULATING_REPORT.json',report)
        role['all_four_arms_matched_for_200_epochs']=True;report['folds'].append(role)
        print(f'W25_A2_FOLD_FREEZE|fold={outer}|MODELS=4/4|INPUT_STREAM_PAIRING=PASS',flush=True)
    STAGE='FINAL_FREEZE_AND_POST_SOURCE_CHECK'
    need(len(report['trained_models'])==20 and len(report['folds'])==5,'FORMAL_RUN_COUNT_INCOMPLETE')
    for m in report['trained_models']:
        need(sha(run/m['checkpoint_relative'])==m['checkpoint_sha256'],'FROZEN_CHECKPOINT_DRIFT')
    validate_sources(HERE)
    need(sha(HERE/SOURCE_MANIFEST)==report['source_manifest_sha256'],'FORMAL_SOURCE_MANIFEST_CHANGED_DURING_TRAINING')
    core.verify_post(report)
    dump(run/'SERVER_ONLY_PROTECTED_INPUT_HASHES.json',core.PROTECTED)
    report['source_and_training_validation_inputs_unchanged']=True
    report['all_20_checkpoints_frozen_before_any_outer_inference']=True
    report['all_4000_training_epochs_completed']=True
    report['all_five_folds_four_arms_input_pairing_verified']=True
    report['historical_checkpoints_reused']=False
    report['formal_training_complete']=True

def result_export(run,report):
    # Whitelist safe outputs only: no IDs, images, row hashes, predictions or model tensors.
    dump(run/'01_FORMAL_TRAIN_FREEZE_REVIEW.json',report)
    payload={'01_FORMAL_TRAIN_FREEZE_REVIEW.json':(run/'01_FORMAL_TRAIN_FREEZE_REVIEW.json').read_bytes(),
             DESIGN_NAME:(HERE/DESIGN_NAME).read_bytes(),SOURCE_MANIFEST:(HERE/SOURCE_MANIFEST).read_bytes()}
    payload['98_MEMBER_SHA256.json']=(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in payload.items()},indent=2)+'\n').encode()
    tmp=run/'02_FORMAL_SAFE_RESULT.zip.partial';target=run/'02_FORMAL_SAFE_RESULT.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
    tmp.replace(target)
    dump(run/'99_RESULT_IDENTITY.json',{'report_sha256':sha(run/'01_FORMAL_TRAIN_FREEZE_REVIEW.json'),'bundle_sha256':sha(target)})

def process_identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();rest=text[text.rfind(')')+2:].split()
        return {'state':rest[0],'start_ticks':rest[19]}
    except (OSError,IndexError):return None

def worker_live(run):
    p=run/'WORKER_PID.json'
    if not p.exists():return False
    a=read(p);cur=process_identity(a['pid'])
    if not cur or cur['state']=='Z' or cur['start_ticks']!=a['start_ticks']:return False
    try:cmd=Path('/proc')/str(a['pid'])/'cmdline';b=cmd.read_bytes()
    except OSError:return False
    return str(run/'source'/Path(__file__).name).encode() in b and b'worker' in b

def validate_run_sources(run,manifest_sha):
    need(not (run/'source').is_symlink(),'PERSISTENT_SOURCE_SYMLINK_NOT_ALLOWED')
    launch=read(run/'00_LAUNCH.json')
    need(launch['source_manifest_sha256']==manifest_sha and launch['design_sha256']==sha(HERE/DESIGN_NAME),'EXISTING_JOB_IDENTITY_MISMATCH')
    need(sha(run/'source'/SOURCE_MANIFEST)==manifest_sha,'INSTALLED_SOURCE_MANIFEST_CHANGED')
    m,_=validate_sources(run/'source')
    return launch

def status_snapshot(run,manifest_sha):
    base={'schema':'W25_A2_FORMAL5_STATUS_V1','run_directory':str(run),'source_manifest_sha256':manifest_sha,
          'design_sha256':sha(HERE/DESIGN_NAME),'outer_test_inference':False,'bootstrap_executed':False,
          'completed_efficacy_ablation':False,'restart_permitted':False}
    if not run.exists():base.update(state='NOT_STARTED',worker_live=False);return base
    validate_run_sources(run,manifest_sha)
    p=read(run/'00_PROGRESS.json') if (run/'00_PROGRESS.json').is_file() else {'state':'STARTING'}
    live=worker_live(run);base.update(progress=p,worker_live=live)
    ident=run/'99_RESULT_IDENTITY.json'
    if ident.exists():
        h=read(ident);rr=run/'01_FORMAL_TRAIN_FREEZE_REVIEW.json';zz=run/'02_FORMAL_SAFE_RESULT.zip'
        need(rr.is_file() and zz.is_file() and sha(rr)==h['report_sha256'] and sha(zz)==h['bundle_sha256'],'FINISHED_RESULT_SHA_DRIFT')
        r=read(rr);need(r['outer_test_inference'] is False and r['bootstrap_executed'] is False,'FINISHED_RESULT_SCOPE_DRIFT')
        base.update(state=r['status'],report_sha256=h['report_sha256'],bundle_sha256=h['bundle_sha256'],safe_result_available=True)
    elif live:base.update(state=p['state'],safe_result_available=False)
    else:base.update(state='STOPPED_OR_STARTUP_UNCONFIRMED_NO_AUTORESTART',safe_result_available=False)
    return base

def worker(run,manifest_sha):
    global STAGE
    require_inside_run(run);validate_run_sources(run,manifest_sha)
    need(HERE==run/'source','WORKER_MUST_RUN_PERSISTENT_SNAPSHOT')
    with (run/'WORKER.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise GateError('FORMAL_WORKER_ALREADY_ACTIVE')
        cur=process_identity(os.getpid());need(cur is not None,'WORKER_PID_IDENTITY_UNAVAILABLE')
        dump(run/'WORKER_PID.json',{'pid':os.getpid(),'start_ticks':cur['start_ticks']})
        safe_progress(run,'PREPARING')
        report={'schema':'W25_A2_FORMAL5_TRAIN_FREEZE_REVIEW_V1','status':'RUNNING','run_directory':str(run),
                'source_manifest_sha256':manifest_sha,'design_sha256':sha(HERE/DESIGN_NAME),
                'a1_report_sha256':sha(HERE/SMOKE_NAME),'created_unix':time.time(),
                'formal_training_complete':False,'outer_test_inference':False,'bootstrap_executed':False,
                'hyperparameter_search':False,'fusion_or_weight_selection_executed':False,
                'completed_efficacy_ablation':False,'model_performance_claim_authorized':False,
                'single_seed_internal_design':True,'model_training_started':False,'trained_models':[]}
        rc=0
        try:
            train_formal(run,report);report['status']=COMPLETE
        except (Exception,SystemExit) as e:
            rc=2;report['status']=FAILED;report['failed_stage']=STAGE
            report['error_code']=str(e) if isinstance(e,GateError) or e.__class__.__name__=='GateError' else 'RUNTIME_'+type(e).__name__
            (run/'SERVER_ONLY_EXCEPTION.log').write_text(traceback.format_exc(),encoding='utf-8')
            core=sys.modules.get('a1_exact_parent')
            if core is not None:
                try:core.verify_post(report);dump(run/'SERVER_ONLY_PROTECTED_INPUT_HASHES.json',core.PROTECTED)
                except Exception:report['source_and_training_validation_inputs_unchanged']=False
        report['finished_unix']=time.time();result_export(run,report)
        safe_progress(run,report['status'],completed_runs=len(report['trained_models']),completed_epoch_units=200*len(report['trained_models']),
                      safe_result_available=True,error_code=report.get('error_code'))
        print('W25_A2_FINAL_STATUS='+report['status'],flush=True)
        return rc

def start_job(run,manifest_sha):
    require_inside_run(run)
    lock_path=RUN_PARENT/(run.name+'.START.lock')
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise GateError('ANOTHER_START_REQUEST_ACTIVE_USE_STATUS')
        if run.exists():
            s=status_snapshot(run,manifest_sha);s['start_action']='EXISTING_JOB_NOT_RESTARTED';return s
        m,d=validate_sources();s=read(HERE/SMOKE_NAME);validate_smoke(s)
        need(Path(s['new_run_directory']).parent==RUN_PARENT,'INVALID_A1_RUN_ROOT')
        need(sha(Path(s['new_run_directory'])/'00_SMOKE_REVIEW.json')==sha(HERE/SMOKE_NAME),'SUCCESSFUL_A1_SERVER_REPORT_DRIFT')
        # Refuse contention before scheduling. A worker repeats this gate before CUDA work.
        core=core_load();core.gpu_gate()
        run.mkdir();source=run/'source';source.mkdir()
        for n in m['files']:
            shutil.copyfile(HERE/n,source/n)
        shutil.copyfile(HERE/SOURCE_MANIFEST,source/SOURCE_MANIFEST)
        validate_sources(source)
        dump(run/'00_LAUNCH.json',{'schema':'W25_A2_DURABLE_LAUNCH_V1','created_unix':time.time(),
             'design_sha256':sha(HERE/DESIGN_NAME),'source_manifest_sha256':manifest_sha,
             'successful_a1_report_sha256':sha(HERE/SMOKE_NAME),'source_copy_complete':True,
             'outer_test_inference_authorized':False,'scheduled_runs':20,'epochs_per_run':200,
             'automatic_restart_or_resume':False})
        safe_progress(run,'STARTING')
        command=[sys.executable,'-B',str(source/Path(__file__).name),'--mode','worker','--manifest-sha',manifest_sha]
        with (run/'SERVER_ONLY_WORKER.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                  cwd=source,start_new_session=True,close_fds=True)
        # A bounded handshake verifies that the worker took ownership. Never wait for training.
        for _ in range(100):
            if (run/'WORKER_PID.json').exists() or proc.poll() is not None:break
            time.sleep(.1)
        st=status_snapshot(run,manifest_sha);st['start_action']='DURABLE_NEW_JOB_CREATED'
        st['submitted_worker_pid']=proc.pid
        return st

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['start','status','worker'],required=True)
    ap.add_argument('--out',type=Path);ap.add_argument('--bundle',type=Path);ap.add_argument('--manifest-sha',required=True)
    a=ap.parse_args()
    need(sha(HERE/SOURCE_MANIFEST)==a.manifest_sha,'EXPECTED_SOURCE_MANIFEST_SHA_MISMATCH')
    validate_sources();run=run_for_design(sha(HERE/DESIGN_NAME));require_inside_run(run)
    if a.mode=='worker':return worker(run,a.manifest_sha)
    need(a.out is not None and a.bundle is not None,'CLIENT_OUTPUT_ARGUMENTS_REQUIRED')
    for p in (a.out,a.bundle):
        need(re.fullmatch(r'/PATH/TO/TEMP/lss_w25a2_[A-Za-z0-9]{8}',str(p.parent)) is not None,'CLIENT_OUTPUT_NOT_OWN_TEMP_STAGE')
    s=start_job(run,a.manifest_sha) if a.mode=='start' else status_snapshot(run,a.manifest_sha)
    dump(a.out,s)
    if s.get('safe_result_available'):
        shutil.copyfile(run/'02_FORMAL_SAFE_RESULT.zip',a.bundle)
        need(sha(a.bundle)==s['bundle_sha256'],'CLIENT_BUNDLE_COPY_SHA')
    print('W25_A2_JOB_STATE='+s['state'],flush=True)
    print('W25_A2_RUN_DIRECTORY='+str(run),flush=True)
    if s.get('progress'):
        q=s['progress'];print('W25_A2_PROGRESS|completed='+str(q.get('completed_runs',0))+'/20|fold='+str(q.get('outer_fold','-'))+'|arm='+str(q.get('arm','-'))+'|epoch='+str(q.get('epoch','-'))+'/200',flush=True)
    print('W25_A2_OUTER_TEST_INFERENCE=FALSE',flush=True)
    print('W25_A2_AUTOMATIC_RESTART=FALSE',flush=True)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except GateError as e:print('W25_A2_CONTROLLER_STOP='+str(e),flush=True);sys.exit(3)
