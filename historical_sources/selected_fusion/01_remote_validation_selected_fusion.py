#!/usr/bin/env python3
"""A6: bounded foreground execution, inherited archive/transport safeguards.
Evaluate creates one design-addressed run; repeated calls only read its state.
Status never starts or resumes computation. Original scientific assets read-only.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,re,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent
RUN_PARENT=Path('/PATH/TO/RESEARCH_WORKSPACE/server_runs')
PREFIX='paper2_w25_a6_matched_val_selection_20261026_'
MANIFEST='10_SELECTION_SOURCE_SHA256.json';DESIGN='03_SELECTION_DESIGN.json'
COMPLETE='PASS_MATCHED_VALIDATION_SELECTION_AND_PAIRED_ANALYSIS'
FAILED='FAIL_A6_DIAGNOSTIC_SAVED'
SCHEMA='W25_A6_VALIDATION_SELECTION_STATUS_V1'

def need(ok,code):
    if not bool(ok):raise RuntimeError(code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,obj):
    p=Path(path);q=p.with_name(p.name+'.partial')
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)

def validate(root,manifest_sha):
    need(sha(root/MANIFEST)==manifest_sha,'A6_SOURCE_MANIFEST_DRIFT');m=read(root/MANIFEST)
    need(m['schema']=='W25_A6_SELECTION_SOURCE_MANIFEST_V1','A6_MANIFEST_SCHEMA')
    for n,h in m['files'].items():
        need(Path(n).name==n and not (root/n).is_symlink(),'A6_SOURCE_PATH')
        need(sha(root/n)==h,'A6_SOURCE_BYTE_CHANGED_'+n)
    d=read(root/DESIGN);need(d['schema']=='W25_A6_MATCHED_VALIDATION_SELECTION_DESIGN_V1','A6_DESIGN_SCHEMA')
    for k in ('new_training','new_model_inference','raw_probability_mutation','outer_driven_search','new_fixed_weight_search'):need(d[k] is False,'A6_SCOPE_'+k)
    need(d['new_validation_weight_selection'] is True and d['new_bootstrap'] is True,'A6_SELECTION_SCOPE')
    return m,d

def process_identity(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text();p=s[s.rfind(')')+2:].split();return {'state':p[0],'start_ticks':p[19]}
    except (OSError,IndexError):return None

def live(run):
    p=run/'WORKER_PID.json'
    if not p.is_file():return False
    w=read(p);now=process_identity(w['pid'])
    if not now or now['state']=='Z' or now['start_ticks']!=w['start_ticks']:return False
    try:return str(run/'source'/Path(__file__).name).encode() in Path('/proc',str(w['pid']),'cmdline').read_bytes()
    except OSError:return False

def run_path(dsha):
    need(re.fullmatch('[a-f0-9]{64}',dsha) is not None,'A6_DESIGN_SHA');return RUN_PARENT/(PREFIX+dsha[:12])

def snapshot(run,msha,dsha):
    base={'schema':SCHEMA,'source_manifest_sha256':msha,'design_sha256':dsha,'run_directory':str(run),
      'new_training':False,'new_model_inference':False,'outer_driven_search':False,'probability_mutation':False,'old_jobs_restarted':False,'restart_permitted':False,'safe_result_available':False,'worker_live':False}
    if not run.exists():return dict(base,execution_status='NOT_STARTED',selection_completed=False,bootstrap_executed=False)
    need(not run.is_symlink() and run.parent==RUN_PARENT,'A6_RUN_PATH_CHANGED')
    launch=read(run/'00_LAUNCH.json')
    need(launch['source_manifest_sha256']==msha and launch['design_sha256']==dsha,'A6_EXISTING_RUN_IDENTITY')
    validate(run/'source',msha);base['worker_live']=live(run)
    idfile=run/'99_RESULT_IDENTITY.json'
    if idfile.is_file():
        ident=read(idfile)
        need(sha(run/'02_SELECTED_FUSION_REVIEW.json')==ident['report_sha256'] and sha(run/'03_SELECTED_FUSION_SAFE_RESULT.zip')==ident['bundle_sha256'],'A6_RESULT_IDENTITY_DRIFT')
        rr=read(run/'02_SELECTED_FUSION_REVIEW.json')
        base.update(execution_status=rr['status'],report_sha256=ident['report_sha256'],bundle_sha256=ident['bundle_sha256'],safe_result_available=True,
          selection_completed=rr.get('selection_completed',False),bootstrap_executed=rr.get('bootstrap_executed',False),selection_counts=rr.get('selection_counts',0),completed_systems=len(rr.get('pooled_results',{})),
          sources_unchanged=rr.get('read_source_files_unchanged',False),error_code=rr.get('error_code'),failed_stage=rr.get('failed_stage'),selection_freeze_sha256=rr.get('selection_freeze_sha256'))
    else:
        p=read(run/'00_PROGRESS.json') if (run/'00_PROGRESS.json').is_file() else {'state':'PREPARING'}
        base.update(execution_status=p['state'] if base['worker_live'] else 'STOPPED_OR_STARTUP_UNCONFIRMED_NO_AUTORESTART',progress=p)
    return base

def export(run,report):
    dump(run/'02_SELECTED_FUSION_REVIEW.json',report)
    names=['02_SELECTED_FUSION_REVIEW.json']
    if (run/'01_SELECTION_FREEZE.json').is_file():names.append('01_SELECTION_FREEZE.json')
    payload={n:(run/n).read_bytes() for n in names}
    for n in [DESIGN,MANIFEST,'18_A5_R2_REVIEWED_REPORT.json']:payload[n]=(run/'source'/n).read_bytes()
    payload['98_MEMBER_SHA256.json']=(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in payload.items()},indent=2)+'\n').encode()
    temp=run/'03_SELECTED_FUSION_SAFE_RESULT.zip.partial'
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
    temp.replace(run/'03_SELECTED_FUSION_SAFE_RESULT.zip')
    dump(run/'99_RESULT_IDENTITY.json',{'report_sha256':sha(run/'02_SELECTED_FUSION_REVIEW.json'),'bundle_sha256':sha(run/'03_SELECTED_FUSION_SAFE_RESULT.zip')})

def worker(run,msha):
    need(HERE==run/'source','A6_WORKER_NOT_PERSISTENT_SOURCE');_,d=validate(HERE,msha)
    latch=os.open(run/'WORKER_ONCE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(latch)
    pid=process_identity(os.getpid());need(pid is not None,'A6_PID_IDENTITY')
    dump(run/'WORKER_PID.json',{'pid':os.getpid(),'start_ticks':pid['start_ticks']})
    report={'schema':'W25_A6_SELECTED_FUSION_REVIEW_V1','status':'RUNNING','started_unix':time.time(),'source_manifest_sha256':msha,'design_sha256':sha(HERE/DESIGN),
      'new_training':False,'new_model_inference':False,'raw_probabilities_modified':False,'outer_driven_search':False,'new_validation_selection':True,
      'selection_completed':False,'outer_prediction_complete':False,'bootstrap_executed':False,'old_jobs_restarted':False,'single_seed_internal_retrospective':True,'multiplicity_adjusted':False,
      'bootstrap_reselects_weights':False,'author_interpretation_required':True}
    stage=['INITIALIZING'];module=None
    def progress(state,**extra):
        stage[0]=state;dump(run/'00_PROGRESS.json',{'state':state,'updated_unix':time.time(),**extra});print('W25_A6_PHASE='+state+'|'+json.dumps(extra,sort_keys=True),flush=True)
    try:
        spec=importlib.util.spec_from_file_location('a6_analysis',HERE/'26_selected_fusion_analysis.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        module.run_analysis(run,report,d,progress);report['status']=COMPLETE
    except Exception as e:
        report['status']=FAILED;report['failed_stage']=stage[0]
        text=str(e)
        report['error_code']=text if re.fullmatch('[A-Za-z0-9_+.:/-]{1,180}',text or '') else 'UNEXPECTED_'+type(e).__name__
        (run/'SERVER_ONLY_ERROR_TRACE.txt').write_text(traceback.format_exc(),encoding='utf-8')
    try:
        report['protected_source_files_unchanged']=module.A4.postcheck() if module is not None else 0
        report['read_source_files_unchanged']=True
    except Exception:
        report['status']=FAILED;report['read_source_files_unchanged']=False;report['error_code']='SOURCE_PROTECTION_CHANGED'
    report['finished_unix']=time.time();export(run,report);progress(report['status'])
    return 0

def request(mode,msha,out,bundle):
    m,d=validate(HERE,msha);dsha=sha(HERE/DESIGN);run=run_path(dsha)
    need(RUN_PARENT.is_dir() and not RUN_PARENT.is_symlink(),'A6_RUN_PARENT_UNAVAILABLE')
    if mode=='evaluate':
        try:run.mkdir();created=True
        except FileExistsError:created=False
        if created:
            # Atomic directory creation prevents a second launch of this design.
            (run/'source').mkdir()
            for n,h in m['files'].items():
                shutil.copyfile(HERE/n,run/'source'/n);need(sha(run/'source'/n)==h,'A6_INSTALL_SOURCE_COPY')
            shutil.copyfile(HERE/MANIFEST,run/'source'/MANIFEST)
            dump(run/'00_LAUNCH.json',{'source_manifest_sha256':msha,'design_sha256':dsha,'created_unix':time.time(),'automatic_restart':False,'foreground':True})
            dump(run/'00_PROGRESS.json',{'state':'PREPARING','updated_unix':time.time()})
            # Synchronous child only; no detached launch and no retry/resume branch.
            cp=subprocess.run([sys.executable,'-B',str(run/'source'/Path(__file__).name),'--mode','worker','--manifest-sha',msha],check=False)
            print('A6_FOREGROUND_WORKER_EXIT_CODE='+str(cp.returncode),flush=True)
        else:print('A6_EXISTING_DESIGN_READ_ONLY_NO_RESTART=TRUE',flush=True)
    state=snapshot(run,msha,dsha)
    if state['safe_result_available']:
        shutil.copyfile(run/'03_SELECTED_FUSION_SAFE_RESULT.zip',bundle);need(sha(bundle)==state['bundle_sha256'],'A6_DOWNLOAD_COPY_SHA')
    dump(out,state);print('W25_A6_STATE='+state['execution_status'],flush=True)
    print('W25_A6_OLD_A2_A3_A4_A5_RESTARTED=FALSE',flush=True)
    return 0

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['evaluate','status','worker'],required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--out',type=Path);p.add_argument('--bundle',type=Path);a=p.parse_args()
    if a.mode=='worker':return worker(run_path(sha(HERE/DESIGN)),a.manifest_sha)
    need(a.out is not None and a.bundle is not None,'A6_OUTPUT_ARGS')
    return request(a.mode,a.manifest_sha,a.out,a.bundle)

if __name__=='__main__':raise SystemExit(main())
