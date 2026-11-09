"""A6 matched inner selection: exact A5-R2 readers and A4 output/metric lineage.
New scientific operation is inner-only weight selection followed by frozen
outer comparison. No training, inference, probability mutation, or grid changes.
"""
from pathlib import Path
import ast,csv,hashlib,importlib.util,io,json,sys,time,zipfile
import numpy as np
HERE=Path(__file__).resolve().parent

def load(name,file):
    s=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

A4=load('a6_exact_a4','20_A4_FIXED_FUSION_PARENT.py')
A5=load('a6_exact_a5','21_A5_FORMAT_AUDIT_PARENT.py')
F=load('a6_exact_formats','17_exact_validation_formats.py')
S=load('a6_inner_selector','25_matched_inner_selection.py')
M=load('a6_exact_metrics','02_fixed_fusion_metrics.py')
A5.PROTECTED=A4.PROTECTED
need=A4.need;gate=A4.gate;read=A4.read;sha=A4.sha;dump=A4.dump
ARMS=list(A4.ARMS)+['PHADKE_P4C','R09_SELECTED_ALPHA']

def source_review_gate(d,sd,review,report):
    need(review['execution_status']=='COMPLETE_EXACT_VALIDATION_FORMAT_REVIEW_PENDING' and review['read_source_files_unchanged'],'A5_R2_REVIEW_STATE')
    need(review['format_completion_counts']=={'a3_current_sha_matched':20,'dual_validation_role_matches':5,'p4c_validation_matches':5,'p4c_test_archive_matches':5,'rejected_outer_test_candidates':5,'new_selection':False,'new_model_inference':False},'A5_R2_REVIEW_COUNTS')
    # Parse text only: historic generators have top-level inference/training code.
    for x in d['producer_files']:
        h=x.get('sha256',x.get('sha256_observed_now'));gate(x['path'],h,'REVIEWED_PRODUCER_TEXT')
        ast.parse(Path(x['path']).read_text(encoding='utf-8-sig'))
    text=(HERE/'23_LEGACY_SELECTOR.py.txt').read_text();tree=ast.parse(text)
    grid=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GRID' for t in n.targets))
    need(grid==d['weights']==S.GRID,'ORIGINAL_SELECTOR_GRID_DRIFT')
    need('["qwk","mae","balanced_accuracy","distance_from_fixed_040","coral_weight"]' in text and 'ascending=[False,True,False,True,True]' in text,'ORIGINAL_SELECTOR_ORDER_DRIFT')
    for x in sd['format_completion']['files']:
        gate(x['path'],x['sha256'],'EXACT_A5_R2_INPUT')
    for x in sd['format_completion']['a3_current_validation_locks']:
        gate(x['path'],x['sha256'],'EXACT_A3_VALIDATION')
    report['source_review']={'accepted_evidence':d['source_review'],'original_generators_executed':False,'old_jobs_restarted':False,'historical_per_file_freeze_claimed':False}

def verify_deit_checkpoint_manifest(sd,review,ids,parent,summary):
    mat=[x for x in review['historical_dual_metadata'] if 'deit_checkpoint_sha256' in x]
    need(len(mat)==5,'DUAL_MATERIALIZED_CHECKPOINT_COUNT')
    root=Path('/PATH/TO/RESEARCH_WORKSPACE')
    lock=root/'server_packages/step06a7_fix5_fix2_exact_sha_final_lock_20260815'
    rb=root/'server_runs/paper2_source_g1_corrected_fixed_inner_formal_p3_l4_20260917'
    r09=root/'server_runs/paper2_p10_c3_r5_selected_alpha_outer_eval_20261005/final_oof/coral_msaux'
    out=[]
    for f in range(5):
        m=next(x for x in mat if x['outer']==f)
        actual=next(x for x in summary['materialized'] if x['outer']==f)
        need(m==actual,'HISTORICAL_DUAL_MATERIALIZATION_RECORD_CHANGED')
        gate(lock/f'deit_inputs/outer_{f}/best_val_loss.pth',m['deit_checkpoint_sha256'],'FROZEN_DEIT_CHECKPOINT')
        gate(r09/f'outer_{f}/best_checkpoint_min_validation_ordinal_objective.pth',m['coral_checkpoint_sha256'],'FROZEN_R09_CHECKPOINT')
        man=rb/f'training/g0/deit/outer_{f}/fixed_inner_trval_manifest.csv'
        gate(man,m['deit_manifest_sha256'],'DEIT_HISTORICAL_ROLE_MANIFEST')
        rows=A5.csvrows(man.read_bytes());vf=(f+1)%5
        val=[x for x in rows if A5.integer(x['protocol_v3_cv_fold'])==vf]
        ii=[A5.integer(x['protocol_v4_row_id']) for x in val]
        F.exact_role(ii,ids,f,'validation')
        if 'patient_id' in val[0]:need(all(parent.normalize_pid(x['patient_id'])==ids[i][0] for x,i in zip(val,ii)),'DEIT_MANIFEST_PATIENT_MISMATCH')
        out.append({'outer_fold':f,'inner_fold':vf,'deit_checkpoint_sha256':m['deit_checkpoint_sha256'],'r09_checkpoint_sha256':m['coral_checkpoint_sha256'],'manifest_sha256':m['deit_manifest_sha256'],'validation_rows':len(ii)})
    return out

def load_validations(sd,review,ids,bridge,parent,report):
    cfg=sd['format_completion'];duals={};arms={};nprofiles={};role_audit={}
    archive=Path(cfg['historical_dual_archive_path']);gate(archive,cfg['historical_dual_archive_sha256'],'ORIGINAL_DUAL_ARCHIVE')
    with zipfile.ZipFile(archive) as z:
        summary=json.loads(z.read('00_P10_C3_R6_R4_R2_SUMMARY.json'))
        need(summary['outer_test_opened'] is False and summary['outer_oof_used_for_weight_selection'] is False,'HISTORICAL_DUAL_SCOPE')
        report['deit_and_r09_checkpoint_manifest_reverification']=verify_deit_checkpoint_manifest(sd,review,ids,parent,summary)
        for f in range(5):
            ref=next(x for x in review['historical_dual_validation_files'] if x['model_outer_fold']==f)
            raw=z.read(ref['archive_member']);need(hashlib.sha256(raw).hexdigest()==ref['sha256'],'DUAL_MEMBER_CHANGED')
            rows=A5.csvrows(raw);by={A5.integer(x['_id']):x for x in rows};ii=sorted(by)
            need(len(by)==len(rows),'DUAL_DUPLICATE_ID');ordered=[by[i] for i in ii]
            yy=np.asarray([A5.integer(x['y_true_infer_coral']) for x in ordered],int)
            need(np.array_equal(yy,[A5.integer(x['y_true_infer_deit']) for x in ordered]),'DUAL_LABELS_DISAGREE')
            F.exact_role(ii,ids,f,'validation',yy)
            need(all(A5.integer(x['outer'])==f and A5.integer(x['inner_fold'])==(f+1)%5 for x in ordered),'DUAL_MODEL_DATA_ROLE')
            need(sorted({float(x['selected_alpha']) for x in ordered})==ref['selected_alpha_values'],'DUAL_ALPHA_CHANGED')
            cp=S.check_probability([[float(x['coral_p'+str(k)]) for k in range(4)] for x in ordered],'R09_VAL')
            dp=S.check_probability([[float(x['deit_p'+str(k)]) for k in range(4)] for x in ordered],'DEIT_VAL')
            duals[f]={'ids':ii,'y':yy,'deit':dp};arms[('R09_SELECTED_ALPHA',f)]=cp
            role_audit[f]={'model_outer_fold':f,'data_fold':(f+1)%5,'rows':len(ii),'patient_overlap_with_outer':0}
    # Reuse exact A3 producer/checkpoint checks, not filename-only acceptance.
    a3report={};A5.audit_a3_validation(sd,ids,bridge,parent,a3report)
    need(a3report['a3_validation_role_match_count']==20,'A3_ROLE_COUNT')
    report['a3_validation_reverification']={'count':20,'current_file_hashes_reverified':20,'checkpoint_and_objective_links_verified':True,'historical_per_file_hash_lock':False}
    for rec in review['a3_twenty_validation_files']:
        f=rec['model_outer_fold'];a=rec['arm'];p=Path(rec['path']);gate(p,rec['sha256_observed_now'],'A5_R2_A3_VAL_SHA')
        rows=A5.csvrows(p.read_bytes());by={A5.integer(x['protocol_v4_row_id']):x for x in rows};ii=duals[f]['ids']
        need(len(rows)==len(by) and set(by)==set(ii),'A3_VAL_EXACT_JOIN')
        ordered=[by[i] for i in ii];yy=[A5.integer(x['y_true']) for x in ordered]
        F.exact_role(ii,ids,f,'validation',yy,[x['patient_id'] for x in ordered],parent.normalize_pid)
        need(np.array_equal(yy,duals[f]['y']),'A3_DUAL_VAL_LABEL_PARITY')
        cp=S.check_probability([[float(x['p'+str(k)]) for k in range(4)] for x in ordered],'A3_VAL')
        need(np.array_equal(cp.argmax(1),[A5.integer(x['prediction']) for x in ordered]),'A3_VAL_NATIVE_CLASS')
        arms[(a,f)]=cp
    p4root=Path(sd['p4c_archive']['path']).parent
    for f in range(5):
        spec=next(x for x in review['p4c_prediction_archives'] if x['model_outer_fold']==f and x['role']=='validation')
        arr,schemas,errors=F.load_npz_members(Path(spec['path']).read_bytes());need(not errors,'P4C_VAL_MEMBER_ERRORS')
        need(set(arr)=={'foramen_id','patient','y','P','fold'},'P4C_VAL_EXACT_FIELDS')
        ii0=[bridge[str(x).strip()] for x in arr['foramen_id']];F.exact_role(ii0,ids,f,'validation',arr['y'],arr['patient'],parent.normalize_pid)
        need(np.equal(arr['fold'],(f+1)%5).all(),'P4C_VAL_DATA_FOLD')
        done=read(p4root/f'fold{f}/COMPLETE.json');hist=A5.csvrows((p4root/f'fold{f}/history.csv').read_bytes())
        need(len(hist)==done['epochs_ran'] and [A5.integer(x['epoch']) for x in hist]==list(range(1,len(hist)+1)),'P4C_HISTORY_LENGTH')
        firstbest=max(hist,key=lambda x:float(x['val_qwk']))
        need(A5.integer(firstbest['epoch'])==done['best_epoch'] and abs(float(firstbest['val_qwk'])-done['best_val_qwk'])<1e-12,'P4C_BEST_EPOCH_HISTORY')
        valq=F.qwk(arr['y'],arr['P'].argmax(1));need(abs(valq-done['best_val_qwk'])<=1e-10,'P4C_BEST_EMA_VAL_PARITY')
        need(done['test_used_for_selection'] is False and done['test_inference_after_checkpoint_freeze'] is True,'P4C_TRAIN_TEST_SCOPE')
        pos={i:j for j,i in enumerate(ii0)};ii=duals[f]['ids'];need(set(pos)==set(ii),'P4C_VAL_JOIN')
        cp=np.asarray(arr['P'],float)[[pos[i] for i in ii]]
        S.check_probability(cp,'P4C_VAL_LOCKED',spec['probability_profile'])
        need(np.array_equal(np.asarray(arr['y'])[[pos[i] for i in ii]],duals[f]['y']),'P4C_VAL_LABEL_PARITY')
        arms[('PHADKE_P4C',f)]=cp;nprofiles[f]=spec['probability_profile']
    report['validation_roles']=list(role_audit.values())
    report['p4c_validation_profiles']=nprofiles
    return duals,arms

def freeze_selections(run,d,duals,arms,report):
    records=[];parities=[]
    for f in range(5):
        for a in ARMS:
            cp=arms[(a,f)];dp=duals[f]['deit']
            parity=S.prediction_parity(cp,dp,d['weights'],p4c=a=='PHADKE_P4C')
            parities.append({'outer_fold':f,'arm':a,**parity})
    # All 30 candidate-grid parity gates precede any new validation scoring.
    for f in range(5):
        for a in ARMS:
            cp=arms[(a,f)];dp=duals[f]['deit']
            rec=S.select_inner(duals[f]['y'],cp,dp,d['weights']);rec.update(outer_fold=f,validation_fold=(f+1)%5,arm=a,validation_rows=len(cp))
            records.append(rec)
    original=[next(x['selected_image_weight'] for x in records if x['arm']=='R09_SELECTED_ALPHA' and x['outer_fold']==f) for f in range(5)]
    need(original==d['historical_r11_weights'],'HISTORICAL_R11_VALIDATION_WEIGHTS_DO_NOT_REPRODUCE_NO_RETUNING')
    frozen={'schema':'W25_A6_THIRTY_SELECTION_FREEZE_V1','frozen_unix':time.time(),'design_sha256':sha(HERE/'03_SELECTION_DESIGN.json'),
      'records':records,'source_hashes_observed_before_new_outer_predictions':dict(A4.PROTECTED),
      'selection_data':'corresponding outer model inner validation only','outer_metric_used':False,'validation_grid_decision_parity':parities,
      'original_R11_weights_reproduced':original,'historical_results_replaced':False}
    path=run/'01_SELECTION_FREEZE.json';need(not path.exists(),'SELECTION_FREEZE_ALREADY_EXISTS_NO_OVERWRITE');dump(path,frozen)
    report['selection_freeze_sha256']=sha(path);report['selection_completed']=True
    report['selection_counts']=len(records);report['selection_records']=records
    report['validation_numeric_representation_parity']={'combinations':30,'grid_candidates_each':8,'rows_across_branches_per_grid':17868,'all_candidates_invariant':True,'observed_changes':0}
    return records

def outer_evaluation(run,d,sd,review,ids,parent,records,report,progress):
    ad=read(HERE/'04_A4_COMPLETED_DESIGN.json');ar=read(HERE/'04_A4_COMPLETED_REPORT.json')
    a3,r,ph=A4.verify_a3(ad);auth,aid=A4.authoritative_rows(ad,parent);need(aid==ids,'OUTER_IDENTITY_CHANGED_AFTER_FREEZE')
    vals,audit=A4.align_a3(a3,ad,ids,parent,M,r)
    deit,da=A4.align_deit(ad,ids,auth,parent,M);p4,pa=A4.align_phadke(ad,ids,parent,M)
    old=[auth[i] for i in range(2978)];r09=A4.probabilities(old,['tuned_progression_coral_prob_'+str(k) for k in range(4)],'R09_OUTER')
    branches=vals+[p4,r09];fold=np.asarray([ids[i][1] for i in range(2978)],int)
    y=np.asarray([ids[i][2] for i in range(2978)],int);patients=np.asarray([ids[i][0] for i in range(2978)],str)
    fixed=[.40*p+.60*deit for p in branches];selected=[];outerparity=[]
    for j,a in enumerate(ARMS):
        ww={f:next(x['selected_image_weight'] for x in records if x['arm']==a and x['outer_fold']==f) for f in range(5)}
        w=np.asarray([ww[int(f)] for f in fold],float)
        outerparity.append({'arm':a,**S.prediction_parity(branches[j],deit,[w],p4c=a=='PHADKE_P4C')})
        selected.append(w[:,None]*branches[j]+(1.-w[:,None])*deit)
    # Existing fixed results must remain exactly as reported; no report substitution.
    allp=branches+[deit]+fixed+selected;pred=np.stack([p.argmax(1) for p in allp],axis=1).astype(int)
    need(d['systems']==ARMS+['ORIGINAL_DEIT']+[a+'_F040' for a in ARMS]+[a+'_VSEL' for a in ARMS],'SYSTEM_ORDER')
    for name,oldp in ar['pooled_results'].items():
        j=d['systems'].index(name);need(A4.points_equal(M.point(y,pred[:,j]),oldp),'A4_FROZEN_RESULT_DRIFT')
    j=d['systems'].index('R09_SELECTED_ALPHA_VSEL')
    need(np.array_equal(pred[:,j],[A4.integer(x['pred_selected_weight_fusion']) for x in old]),'R11_OUTER_CLASS_PARITY')
    hp=A4.probabilities(old,['selected_weight_fusion_prob_'+str(k) for k in range(4)],'HISTORICAL_R11')
    err=float(np.abs(selected[-1]-hp).max());need(err<=1e-7,'R11_OUTER_PROBABILITY_PARITY')
    need(sha(run/'01_SELECTION_FREEZE.json')==report['selection_freeze_sha256'],'FROZEN_WEIGHTS_CHANGED')
    report['original_R11_forward_parity']={'prediction_mismatches':0,'probability_max_abs_error':err,'weights_reproduced':d['historical_r11_weights']}
    report['a4_all_11_original_points_reproduced']=True;report['outer_numeric_representation_parity']=outerparity
    report['pooled_results']={n:M.point(y,pred[:,k]) for k,n in enumerate(d['systems'])}
    report['fold_results']=[{'outer_fold':f,'patients':len(set(patients[fold==f])),'rois':int((fold==f).sum()),
      'systems':{n:M.point(y[fold==f],pred[fold==f,k]) for k,n in enumerate(d['systems'])}} for f in range(5)]
    with (run/'SERVER_ONLY_SELECTED_AND_REFERENCE_PREDICTIONS.csv').open('w',newline='',encoding='utf-8') as fh:
        wr=csv.writer(fh);wr.writerow(['protocol_v4_row_id','patient_id','outer_fold','y_true']+d['systems'])
        for i in range(2978):wr.writerow([i,patients[i],int(fold[i]),int(y[i]),*pred[i].tolist()])
    report['server_only_predictions_sha256']=sha(run/'SERVER_ONLY_SELECTED_AND_REFERENCE_PREDICTIONS.csv')
    report['outer_prediction_complete']=True
    def callback(n,total):
        if n%1024==0 or n==total:progress('BOOTSTRAPPING',completed_systems=19,bootstrap_completed=n,bootstrap_total=total)
    report['paired_bootstrap']=M.paired_fusion_bootstrap(y,pred,patients,d['systems'],d['contrasts'],iterations=d['bootstrap_repeats'],seed=d['bootstrap_seed'],callback=callback)
    need(all(m['valid_replicates']==d['bootstrap_repeats'] for c in report['paired_bootstrap']['contrasts'] for m in c['metrics'].values()),'UNDEFINED_BOOTSTRAP_REPLICATES')
    report['bootstrap_executed']=True;report['matched_fusion_selection_opportunity_evaluation_complete']=True
    report['equal_architecture_training_alpha_search_budget_claimed']=False

def run_analysis(run,report,d,progress):
    sd=read(HERE/'03_DIAGNOSTIC_DESIGN.json');review=read(HERE/'18_A5_R2_REVIEWED_REPORT.json')
    for n,key in [('18_A5_R2_REVIEWED_REPORT.json','a5_report_sha256'),('03_DIAGNOSTIC_DESIGN.json','a5_design_sha256'),('04_A4_COMPLETED_DESIGN.json','a4_design_sha256'),('04_A4_COMPLETED_REPORT.json','a4_report_sha256'),('04_A2_FREEZE_REPORT.json','a2_freeze_sha256'),('04_A3_COMPLETED_REPORT.json','a3_report_sha256'),('23_LEGACY_SELECTOR.py.txt','original_selector_sha256')]:gate(HERE/n,d[key],'PACKAGED_'+n)
    need(d['weights']==S.GRID and d['arms']==ARMS and d['bootstrap_repeats']==20000 and d['bootstrap_seed']==42 and len(d['contrasts'])==12,'SELECTION_DESIGN_DRIFT')
    progress('VERIFYING_REVIEWED_SOURCES')
    A5.completed_gate(sd,report);source_review_gate(d,sd,review,report)
    parent=A5.safe_source_module();ids,bridge=A5.build_identity(sd,parent)
    progress('READING_EXACT_INNER_VALIDATION')
    duals,arms=load_validations(sd,review,ids,bridge,parent,report)
    progress('SELECTING_INNER_ONLY')
    records=freeze_selections(run,d,duals,arms,report)
    print('W25_A6_THIRTY_INNER_SELECTIONS_FROZEN=TRUE',flush=True)
    # No new outer predictions or outer performance ranking before this boundary.
    progress('EVALUATING_FROZEN_WEIGHTS')
    outer_evaluation(run,d,sd,review,ids,parent,records,report,progress)
    report['protected_source_files_unchanged']=A4.postcheck()
    need(sha(run/'01_SELECTION_FREEZE.json')==report['selection_freeze_sha256'],'FINAL_SELECTION_FREEZE_CHANGED')
    report['raw_probabilities_modified']=False;report['old_jobs_restarted']=False
