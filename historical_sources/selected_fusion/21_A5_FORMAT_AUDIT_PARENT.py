#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W25-A5-R2: exact-format, source-read-only validation provenance audit.
D1 I/O helpers and P5R2 identity bridge are inherited byte-for-byte. No model
code is imported from discovered files; no training, inference, fusion or search.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, importlib.util, io, json, os, re, sys, time, zipfile
from pathlib import Path
from collections import Counter, defaultdict, deque
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROTECTED={}
SOURCE_EXPORTS={}
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

def csvrows(b):
    reader=csv.DictReader(io.StringIO(b.decode('utf-8-sig')))
    need(reader.fieldnames is not None and len(set(reader.fieldnames))==len(reader.fieldnames),'CSV_HEADER_INVALID')
    return list(reader)

def integer(v):
    s=str(v).strip();need(re.fullmatch(r'-?[0-9]+(?:\.0+)?',s) is not None,'INTEGER_ID_OR_LABEL_INVALID')
    return int(s.split('.')[0])

def require_columns(rows,columns,label):
    need(len(rows)>0 and all(set(columns)<=set(r) for r in rows),label+'_SCHEMA')

def process_identity(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text();parts=s[s.rfind(')')+2:].split()
        return {'state':parts[0],'start_ticks':parts[19]}
    except (OSError,IndexError):return None

# The following validation-asset logic is NEW; it is not a changed evaluator.
SCHEMA='W25_A5_R2_EXACT_FORMAT_REVIEW_V1'
COMPLETE='COMPLETE_EXACT_VALIDATION_FORMAT_REVIEW_PENDING'
FAILED='FAIL_EXACT_VALIDATION_FORMAT_DIAGNOSTIC_SAVED'
ROW_ID_COLUMNS=('protocol_v4_row_id','_id','row_index')
PID_COLUMNS=('patient_id','patient','pid')
LABEL_COLUMNS=('y_true','y_true_auth','grade','severity','label')
COHORT_FOLD_COLUMNS=('unified_outer_fold','protocol_v3_cv_fold')
PROB_GROUPS=[['p0','p1','p2','p3'],['prob_0_normal','prob_1_mild','prob_2_moderate','prob_3_severe'],['prob_0','prob_1','prob_2','prob_3'],['prob0','prob1','prob2','prob3']]


def bsha(b):return hashlib.sha256(b).hexdigest()

def observe(path):
    p=Path(path)
    need(p.is_file() and not p.is_symlink(),'OBSERVED_FILE_MISSING_OR_SYMLINK')
    h=sha(p)
    need(str(p) not in PROTECTED or PROTECTED[str(p)]==h,'SOURCE_CHANGED_DURING_READ')
    PROTECTED[str(p)]=h
    return h

def safe_source_module():
    path=HERE/'05_P5R2_IDENTITY_PARENT.py'
    spec=importlib.util.spec_from_file_location('a5_exact_identity_parent',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def archive_table(cfg,member_key='member',hash_key='member_sha256'):
    gate(cfg['path'],cfg['sha256'],'LOCKED_ARCHIVE')
    with zipfile.ZipFile(cfg['path']) as z:
        name=cfg[member_key];need(z.namelist().count(name)==1,'ARCHIVE_MEMBER_NOT_UNIQUE')
        raw=z.read(name);need(bsha(raw)==cfg[hash_key],'LOCKED_ARCHIVE_MEMBER_SHA')
    return csvrows(raw)

def build_identity(d,parent):
    rows=archive_table(d['r11_archive']);ids={}
    for r in rows:
        i=integer(r['_id']);need(i not in ids,'AUTHORITY_DUPLICATE_ROW_ID')
        pid=parent.normalize_pid(r['patient_id']);need(pid is not None,'AUTHORITY_PATIENT_UNRECOGNIZED')
        ids[i]=(pid,integer(r['_outer_eval']),integer(r['y_true_auth']))
    need(set(ids)==set(range(2978)),'AUTHORITY_ROW_SET')
    need(len({x[0] for x in ids.values()})==468,'AUTHORITY_PATIENT_COUNT')
    folds=defaultdict(set)
    for pid,f,y in ids.values():
        need(f in range(5) and y in range(4),'AUTHORITY_FOLD_OR_LABEL_RANGE');folds[pid].add(f)
    need(all(len(v)==1 for v in folds.values()),'AUTHORITY_PATIENT_MULTIFOLD')
    need([sum(t[1]==f for t in ids.values()) for f in range(5)]==d['outer_counts'],'AUTHORITY_FOLD_COUNTS')
    idx=archive_table(d['p4c_archive'],'index_member','index_sha256')
    gate(d['r2_manifest']['path'],d['r2_manifest']['sha256'],'LOCKED_R2_BRIDGE')
    r2=csvrows(Path(d['r2_manifest']['path']).read_bytes())
    cross=parent.try_exact_r2_roi_crosswalk(r2,idx)
    need(cross['status']=='EXACT_2978_ROI_IMAGE_BBOX_CROSSWALK','P4C_BRIDGE_NOT_EXACT')
    c=cross['successful_key']
    def key(row,p4):
        pid=parent.normalize_pid(row['patient' if p4 else 'patient_id'])
        name=parent.filename_part(row.get(c['p4c' if p4 else 'r2'],''),c['mode']=='stem')
        bbox=parent.bbox_from(row,('xmin','ymin','xmax','ymax') if p4 else ('x0','y0','x1','y1'))
        need(pid and name and bbox is not None,'BRIDGE_INCOMPLETE')
        return pid,name,bbox
    ix={key(r,True):r for r in idx};need(len(ix)==2978,'BRIDGE_DUPLICATE_KEY')
    bridge={}
    for r in r2:
        q=ix[key(r,False)];fid=str(q['foramen_id']).strip();i=integer(r['row_index'])
        need(fid not in bridge and i in ids,'BRIDGE_DUPLICATE_OR_MISSING')
        need(ids[i]==(parent.normalize_pid(r['patient_id']),integer(r['outer_fold']),integer(q['grade'])),'BRIDGE_VS_AUTHORITY_IDENTITY')
        bridge[fid]=i
    need(len(bridge)==2978 and set(bridge.values())==set(ids),'BRIDGE_NOT_BIJECTIVE')
    return ids,bridge

def probability_profile(rows,columns):
    """Diagnostic only: no clipping, normalization, or new acceptance policy."""
    try:p=np.asarray([[float(r[c]) for c in columns] for r in rows],dtype=float)
    except (ValueError,TypeError,KeyError):return {'parsed':False}
    finite=np.isfinite(p)
    result={'parsed':True,'columns':columns,'rows':len(rows),'nonfinite_values':int((~finite).sum())}
    if not finite.all():return result
    sums=p.sum(1)
    result.update(negative_values=int((p<0).sum()),values_above_one=int((p>1).sum()),
                  min_value=float(p.min()),max_value=float(p.max()),
                  min_row_sum=float(sums.min()),max_row_sum=float(sums.max()),
                  max_abs_row_sum_error=float(np.abs(sums-1).max()),
                  rows_exceeding_2e_6=int((np.abs(sums-1)>2e-6).sum()))
    return result

def table_profile(rows,ids,bridge,parent):
    """Cohort membership is verified; filenames do not prove producer identity."""
    need(bool(rows),'EMPTY_TABLE')
    columns=list(rows[0]);out={'rows':len(rows),'columns':columns,'row_values_exported':False,
        'model_checkpoint_link_proved_by_this_profile':False,
        'ready_for_weight_selection':False}
    groups=[g for g in PROB_GROUPS if set(g)<=set(columns)]
    for c in columns:
        if c.endswith('_prob_0'):
            g=[c[:-1]+str(i) for i in range(4)]
            if set(g)<=set(columns) and g not in groups:groups.append(g)
    out['probability_profiles']=[probability_profile(rows,g) for g in groups]
    idcol=next((c for c in ROW_ID_COLUMNS if c in columns),None)
    fid='foramen_id' if 'foramen_id' in columns else None
    pc=next((c for c in PID_COLUMNS if c in columns),None)
    yc=next((c for c in LABEL_COLUMNS if c in columns),None)
    if not groups or (idcol is None and fid is None) or pc is None or yc is None:
        out['identity_status']='SCHEMA_INSUFFICIENT_FOR_ROLE_VALIDATION';return out
    try:
        ii=[integer(r[idcol]) if idcol else bridge[str(r[fid]).strip()] for r in rows]
        mismatches={'patient':0,'label':0,'cohort_fold':0,'alternative_id_columns':0}
        seen=set();duplicates=0;cohort=Counter()
        for i,r in zip(ii,rows):
            if i in seen:duplicates+=1
            seen.add(i);pid,f,y=ids[i];cohort[f]+=1
            mismatches['patient']+=parent.normalize_pid(r[pc])!=pid
            mismatches['label']+=integer(r[yc])!=y
            for fc in COHORT_FOLD_COLUMNS:
                if fc in columns:mismatches['cohort_fold']+=integer(r[fc])!=f
            if idcol and fid:mismatches['alternative_id_columns']+=bridge[str(r[fid]).strip()]!=i
        out.update(identity_mismatch_counts={k:int(v) for k,v in mismatches.items()},duplicate_row_ids=duplicates,
                   patients=len({ids[i][0] for i in seen}),cohort_fold_counts={str(k):v for k,v in sorted(cohort.items())})
        exact=[f for f in range(5) if seen=={i for i,x in ids.items() if x[1]==f} and len(ii)==len(seen)]
        out['exact_full_data_fold_match']=exact
        out['compatible_model_outer_if_this_is_inner_validation']=[(f-1)%5 for f in exact]
        out['outer_oof_like_full_cohort']=len(ii)==len(ids) and seen==set(ids)
        out['identity_status']='ROW_PATIENT_LABEL_MATCH' if not any(mismatches.values()) and duplicates==0 else 'IDENTITY_CONFLICT_OR_REPEATED_ROWS'
        if out['outer_oof_like_full_cohort']:out['selection_warning']='DO_NOT_REPURPOSE_POOLED_OUTER_OOF_AS_INNER_VALIDATION'
        # These are observations, not a claim that a column is a model-fold field.
        fields={}
        for c in ('outer','outer_fold','_outer_eval','model_outer_fold','fixed_val','selected_inner_validation_fold','fixed_validation_fold'):
            if c in columns:
                try:
                    vals=sorted({integer(r[c]) for r in rows})
                    fields[c]=vals if set(vals)<=set(range(5)) else 'VALUES_OUTSIDE_FIVE_FOLDS'
                except GateError:fields[c]='NONINTEGER_VALUES'
        out['observed_fold_columns_not_assumed_model_roles']=fields
        hashes={c:sorted({r[c].lower() for r in rows}) for c in columns if ('sha' in c.lower() and all(re.fullmatch('[a-fA-F0-9]{64}',r[c]) for r in rows))}
        out['checkpoint_or_source_hash_columns']=hashes
    except (KeyError,ValueError,TypeError,GateError):out['identity_status']='IDENTITY_PARSE_OR_LOOKUP_FAILED'
    return out

def audit_deit_reference(d,ids,bridge,parent,report):
    cfg=d['deit'];paths=[Path(x) for x in cfg['paths'] if Path(x).is_file()]
    need(bool(paths),'EXACT_FROZEN_DEIT_REFERENCE_MISSING')
    for p in paths:gate(p,cfg['sha256'],'EXACT_FROZEN_DEIT_REFERENCE')
    rows=csvrows(paths[0].read_bytes());profile=table_profile(rows,ids,bridge,parent)
    need(profile.get('identity_status')=='ROW_PATIENT_LABEL_MATCH' and profile.get('outer_oof_like_full_cohort'),'FROZEN_DEIT_AUTHORITY_IDENTITY')
    counts=Counter();epochs=defaultdict(set)
    for r in rows:
        i=integer(r[cfg['id_column']]);outer=ids[i][1]
        need(integer(r[cfg['fold_column']])==outer and integer(r[cfg['inner_fold_column']])==(outer+1)%5,'FROZEN_DEIT_MODEL_ROLE_DRIFT')
        counts[outer]+=1
        if 'checkpoint_epoch' in r:epochs[outer].add(integer(r['checkpoint_epoch']))
    report['same_frozen_deit_reference']={'sha256':cfg['sha256'],'rows':len(rows),'paths_verified':len(paths),
        'fold_rois':[counts[f] for f in range(5)],'checkpoint_epochs_by_outer':{str(f):sorted(v) for f,v in epochs.items()},
        'row_patient_label_and_model_roles_match':True,'used_as_validation':False,
        'warning':'This is a reference outer OOF only; it cannot supply inner-validation probabilities for the same outer model.'}

def audit_a3_validation(d,ids,bridge,parent,report):
    root=Path(d['a3']['run']);r=read(HERE/'04_A3_COMPLETED_REPORT.json');freeze=read(HERE/'04_A2_FREEZE_REPORT.json')
    a2=Path(d['a2']['run']);records=[]
    gate(a2/'01_FORMAL_TRAIN_FREEZE_REVIEW.json',d['a2']['freeze_report_sha256'],'REMOTE_A2_FREEZE_REPORT')
    for f in range(5):
        vf=(f+1)%5
        for arm in d['arms']:
            cp=next(x for x in freeze['trained_models'] if x['outer_fold']==f and x['arm']==arm)
            p=a2/cp['checkpoint_relative'];gate(p,cp['checkpoint_sha256'],'FROZEN_A2_MODEL')
            evidence=next(x for x in r['inference_checks'] if x['outer_fold']==f and x['arm']==arm)
            need(evidence['checkpoint_sha256']==cp['checkpoint_sha256'] and evidence['validation_loss_reproduction_max_abs']==0,'A3_VALIDATION_PRODUCER_LINK_DRIFT')
            v=root/f'SERVER_ONLY_VAL_outer{f}_{arm}.csv'
            rec={'arm':arm,'model_outer_fold':f,'expected_validation_data_fold':vf,'expected_rois':d['outer_counts'][vf],
                 'checkpoint_sha256':cp['checkpoint_sha256'],'checkpoint_actual_bytes_match':True,
                 'producer_source_sha256':d['a3']['controller_sha256'],'original_report_validation_objective_parity':True,
                 'path':str(v),'file_existed_before_audit':v.is_file(),'historically_exported_validation_file_sha':False}
            if v.is_file():
                rec['sha256_observed_now']=observe(v);rows=csvrows(v.read_bytes());profile=table_profile(rows,ids,bridge,parent)
                rec['profile']=profile
                good=(profile.get('identity_status')=='ROW_PATIENT_LABEL_MATCH' and profile.get('exact_full_data_fold_match')==[vf] and profile['rows']==d['outer_counts'][vf] and len(profile['probability_profiles'])==1)
                pprof=profile['probability_profiles'][0] if profile['probability_profiles'] else {}
                good=good and pprof.get('nonfinite_values',1)==0 and pprof.get('negative_values',1)==0 and pprof.get('max_value',2)<=1+1e-6 and pprof.get('rows_exceeding_2e_6',1)==0
                if 'prediction' in rows[0]:
                    pp=np.asarray([[float(x['p'+str(i)]) for i in range(4)] for x in rows])
                    bad=int(np.count_nonzero(pp.argmax(1)!=np.asarray([integer(x['prediction']) for x in rows])))
                    rec['stored_prediction_argmax_mismatches']=bad;good=good and bad==0
                rec['status']='EXACT_ROLE_MATCH_A3_PRODUCER_LINK_NEW_FILE_HASH' if good else 'VALIDATION_CONTENT_REQUIRES_REVIEW'
                rec['no_readiness_from_filename_only']=True
            else:rec['status']='MISSING_VALIDATION_FILE_NO_AUTO_INFERENCE'
            records.append(rec)
        print(f'A5_KNOWN_VALIDATION_AUDIT|outer={f}|collected={len(records)}/20',flush=True)
    report['a3_twenty_validation_files']=records
    report['a3_validation_role_match_count']=sum(x['status']=='EXACT_ROLE_MATCH_A3_PRODUCER_LINK_NEW_FILE_HASH' for x in records)
    report['a3_validation_hash_boundary']='A3 source writes validation CSVs and reports checkpoint/objective parity; their individual bytes were not in the historical outer-prediction SHA manifest. Current SHA observations do not retroactively prove an earlier per-file freeze.'


def metadata_summary(obj):
    """Return artifact references/hashes, never patient or per-row JSON content."""
    hashes=set();paths=set();links=[];count=0
    def safe_key(k):
        s=str(k)
        return s if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]{0,100}',s) and not re.search(r'patient[_-]?\d|case[_-]?\d',s,re.I) else '<redacted-key>'
    def visit(x,key='',depth=0):
        nonlocal count
        if depth>10 or count>8000:return
        count+=1
        if isinstance(x,dict):
            # Keep links already present in the same metadata object; do not infer
            # lineage merely from similar filenames or matching cohort sizes.
            link={}
            for k,v in x.items():
                kk=safe_key(k)
                if kk=='<redacted-key>':continue
                if isinstance(v,str) and re.fullmatch('[a-fA-F0-9]{64}',v):link[kk]=v.lower()
                elif isinstance(v,str) and v.startswith('/PATH/TO/RESEARCH_WORKSPACE/') and Path(v).suffix.lower() in ('.py','.json','.csv','.pt','.pth','.zip') and not re.search(r'patient[_-]?\d|case[_-]?\d|/\d{3,}/',v,re.I):link[kk]=v
                elif k in ('outer_fold','fixed_validation_fold','selected_inner_validation_fold','best_epoch','checkpoint_epoch') and isinstance(v,int) and 0<=v<=10000:link[kk]=v
            if len(link)>=2 and any(isinstance(v,str) and re.fullmatch('[a-f0-9]{64}',v) for v in link.values()):links.append(link)
            for k,v in x.items():
                if re.search(r'patient|foramen|(^|_)row(_|$)|label|prediction|probabilit|^p[0-3]$',str(k),re.I):
                    # A per-file SHA entry is safe even if its key contains prediction.
                    if isinstance(v,str) and re.fullmatch('[a-fA-F0-9]{64}',v):hashes.add(v.lower())
                    continue
                visit(v,str(k),depth+1)
        elif isinstance(x,list):
            for v in x[:100]:
                if isinstance(v,dict):visit(v,key,depth+1)
        elif isinstance(x,str):
            if re.fullmatch('[a-fA-F0-9]{64}',x):hashes.add(x.lower())
            elif x.startswith('/PATH/TO/RESEARCH_WORKSPACE/') and Path(x).suffix.lower() in ('.py','.json','.csv','.pt','.pth','.zip'):
                if not re.search(r'patient[_-]?\d|case[_-]?\d|/\d{3,}/',x,re.I):paths.add(x)
    visit(obj)
    return {'top_level_keys':[safe_key(k) for k in list(obj)[:80]] if isinstance(obj,dict) else ['<list>' if isinstance(obj,list) else '<scalar>'],
            'artifact_paths':sorted(paths)[:100],'declared_hashes':sorted(hashes)[:150], 'same_object_artifact_links':links[:100],
            'values_and_patient_records_exported':False,'traversal_limited':count>8000}

def source_summary(raw,d):
    h=bsha(raw);roles=[k for k,v in d['known_source_hashes'].items() if h==v]
    out={'sha256':h,'exact_prior_source_roles':roles,'executed':False}
    try:text=raw.decode('utf-8-sig');tree=ast.parse(text)
    except (SyntaxError,UnicodeError):out['parse_status']='NOT_VALID_PYTHON';return out
    out['parse_status']='AST_ONLY_NOT_EXECUTED'
    out['functions']=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    # A small non-row metadata extraction: path literals and declared identifiers.
    out['artifact_path_literals']=sorted({n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('/PATH/TO/RESEARCH_WORKSPACE/') and len(n.value)<350 and not re.search(r'patient[_-]?\d|case[_-]?\d',n.value,re.I)})[:80]
    if roles and not re.search(r'(?i)(?:password|private[_-]?key|api[_-]?key)\s*=\s*[\'\"][^\'\"]+[\'\"]',text):
        name='sources/verified_'+h+'.py.txt';SOURCE_EXPORTS[name]=raw;out['exact_source_export_member']=name
    return out


def candidate_content(raw,name,d,ids,bridge,parent):
    suffix=Path(name).suffix.lower()
    out={'sha256':bsha(raw),'bytes':len(raw)}
    try:
        if suffix=='.csv':
            rows=csvrows(raw)
            if not rows:out['type']='EMPTY_CSV'
            else:out['type']='CSV';out['table_profile']=table_profile(rows,ids,bridge,parent)
        elif suffix=='.json':out['type']='JSON';out['metadata']=metadata_summary(json.loads(raw.decode('utf-8-sig')))
        elif suffix=='.py':out['type']='PYTHON';out['source']=source_summary(raw,d)
        else:out['type']='UNREAD_FORMAT'
    except (ValueError,TypeError,UnicodeError,GateError):out['parse_status']='UNRECOGNIZED_OR_UNSAFE_METADATA_NOT_ASSUMED_AVAILABLE'
    return out


def targeted_roots(d):
    cfg=d['discovery'];roots=[dict(x) for x in cfg['direct_roots']];scope=[]
    for spec in cfg['top_level_prefix_scopes']:
        base=Path(spec['parent']);matches=[]
        if base.is_dir():
            # List names at ONE level only; descend only into the locked task prefixes.
            for p in sorted(base.iterdir()):
                if p.name.startswith(spec['prefix']) and p.is_dir() and not p.is_symlink():matches.append(p)
        scope.append({'parent':str(base),'prefix':spec['prefix'],'matching_root_count':len(matches)})
        roots.extend({'role':spec['role'],'path':str(p)} for p in matches)
    need(len(roots)<=cfg['max_roots'],'TARGET_ROOT_SCOPE_LIMIT')
    return roots,scope


def inspect_candidates(d,ids,bridge,parent,report):
    cfg=d['discovery'];roots,scope=targeted_roots(d);records=[];walks=[];seen=set();stats=Counter();start=time.monotonic();limits=[]
    def budget():
        if time.monotonic()-start>cfg['max_elapsed_seconds']:limits.append('ELAPSED_SCAN_LIMIT');return False
        if stats['entries']>=cfg['max_entries']:limits.append('DIRECTORY_ENTRY_LIMIT');return False
        if stats['read_bytes']>=cfg['max_read_total_bytes']:limits.append('TEXT_READ_BUDGET');return False
        if len(records)>=cfg['max_files']:limits.append('FILE_RECORD_LIMIT');return False
        return True
    for spec in roots:
        base=Path(spec['path']);walk={'role':spec['role'],'root':str(base),'exists':base.is_dir(),'visited_entries':0};walks.append(walk)
        if not base.is_dir() or base.is_symlink():continue
        q=deque([(base,0)])
        while q and budget():
            directory,depth=q.popleft()
            try:entries=sorted(directory.iterdir())
            except OSError:walk.setdefault('issues',[]).append('UNREADABLE_SUBDIRECTORY');continue
            for p in entries:
                if not budget():break
                stats['entries']+=1;walk['visited_entries']+=1
                if p.is_symlink():stats['symlink_entries_skipped']+=1;continue
                if p.is_dir():
                    if p.name in ('__pycache__','.git','images','rois','cases','cache','tensorboard'):continue
                    if depth<cfg['max_depth']:q.append((p,depth+1))
                    else:stats['directories_at_depth_limit']+=1
                    continue
                if not p.is_file() or str(p) in seen:continue
                seen.add(str(p));suffix=p.suffix.lower()
                if suffix not in ('.csv','.json','.py','.zip','.pt','.pth','.npz'):continue
                rel=p.relative_to(base).as_posix()
                if re.search(r'patient[_-]?\d|case[_-]?\d',rel,re.I):stats['patient_named_files_skipped']+=1;continue
                size=p.stat().st_size
                rec={'scope_role':spec['role'],'path':str(p),'relative':rel,'bytes':size,'not_selected_by_performance':True}
                if suffix in ('.pt','.pth'):
                    stats['checkpoint_candidates']+=1
                    if stats['checkpoint_hashes']>=cfg['max_checkpoints'] or stats['checkpoint_hashed_bytes']+size>cfg['max_checkpoint_hash_bytes']:
                        rec['status']='CHECKPOINT_FOUND_NOT_HASHED_BUDGET';records.append(rec);continue
                    rec['sha256_observed_now']=observe(p);rec['status']='CHECKPOINT_HASH_ONLY_NOT_DESERIALIZED'
                    stats['checkpoint_hashes']+=1;stats['checkpoint_hashed_bytes']+=size;records.append(rec);continue
                if suffix=='.zip':
                    if size>cfg['max_text_file_bytes']:
                        rec['status']='ARCHIVE_FOUND_NOT_OPENED_SIZE_LIMIT';records.append(rec);continue
                    if stats['read_bytes']+size>cfg['max_read_total_bytes']:limits.append('TEXT_READ_BUDGET');break
                    rec['sha256_observed_now']=observe(p);stats['read_bytes']+=size;members=[]
                    try:
                        with zipfile.ZipFile(p) as z:
                            candidates=[i for i in z.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() in ('.csv','.json','.py')]
                            if len(candidates)>cfg['max_archive_members']:
                                rec['status']='ARCHIVE_MEMBER_LIMIT_NOT_FULLY_INSPECTED';records.append(rec);continue
                            for inf in candidates:
                                if not budget():break
                                n=inf.filename.replace('\\','/')
                                if '..' in Path(n).parts or n.startswith('/') or re.search(r'patient[_-]?\d|case[_-]?\d',n,re.I):continue
                                if inf.file_size>cfg['max_archive_member_bytes'] or stats['read_bytes']+inf.file_size>cfg['max_read_total_bytes']:
                                    members.append({'member':n,'bytes':inf.file_size,'status':'NOT_READ_SIZE_LIMIT'});continue
                                raw=z.read(inf);stats['read_bytes']+=len(raw)
                                sub=candidate_content(raw,n,d,ids,bridge,parent);sub['member']=n;members.append(sub)
                        rec.update(status='ARCHIVE_METADATA_INSPECTED_NO_EXTRACTION',members=members)
                    except (OSError,zipfile.BadZipFile,RuntimeError):rec['status']='ARCHIVE_READ_FAILED_NO_ASSUMPTION'
                    records.append(rec);continue
                if size>cfg['max_text_file_bytes'] or stats['read_bytes']+size>cfg['max_read_total_bytes']:
                    rec['status']='FILE_FOUND_NOT_READ_SIZE_LIMIT';records.append(rec);continue
                if suffix=='.npz':
                    # No object-array unpickling. Record array schema, never rows.
                    rec['sha256_observed_now']=observe(p);stats['read_bytes']+=size
                    try:
                        with zipfile.ZipFile(p) as zz:
                            need(sum(x.file_size for x in zz.infolist())<=64_000_000,'NUMPY_EXPANSION_LIMIT')
                        with np.load(p,allow_pickle=False) as npz:
                            rec['arrays']={k:{'shape':list(npz[k].shape),'dtype':str(npz[k].dtype)} for k in npz.files[:30] if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,100}',k) and not re.search(r'patient[_-]?\d|case[_-]?\d',k,re.I)}
                        rec['status']='NUMPY_SCHEMA_ONLY_ROLE_REQUIRES_SOURCE_REVIEW'
                    except (ValueError,OSError,GateError,zipfile.BadZipFile):rec['status']='NUMPY_OBJECT_OR_UNREADABLE_OR_TOO_LARGE_NO_PICKLE_ALLOWED'
                    records.append(rec);continue
                rec['sha256_observed_now']=observe(p);raw=p.read_bytes();stats['read_bytes']+=size
                rec.update(candidate_content(raw,p.name,d,ids,bridge,parent));records.append(rec)
                if len(records)%40==0:print('A5_TARGETED_ASSET_PROGRESS|files='+str(len(records))+'|NO_SELECTION=True',flush=True)
        if not budget():break
    if stats['directories_at_depth_limit']>0:limits.append('DIRECTORY_DEPTH_LIMIT')
    report['targeted_scan']={'roots':walks,'top_level_prefix_discovery':scope,'stats':dict(stats),'limits_reached':sorted(set(limits)),
                            'scope_exhausted_within_limits':not limits and stats['directories_at_depth_limit']==0,'records':records}
    ready=[];source_hits=[]
    for rec in records:
        entries=rec.get('members',[rec])
        for sub in entries:
            pr=sub.get('table_profile',{})
            if pr.get('identity_status')=='ROW_PATIENT_LABEL_MATCH' and len(pr.get('exact_full_data_fold_match',[]))==1:
                ready.append({'scope_role':rec['scope_role'],'path':rec['path'],'member':sub.get('member'),
                              'sha256':sub['sha256'],'data_fold':pr['exact_full_data_fold_match'][0],
                              'compatible_model_outer':pr['compatible_model_outer_if_this_is_inner_validation'][0],
                              'status':'COHORT_ROLE_COMPATIBLE_MODEL_PROVENANCE_NOT_YET_APPROVED'})
            if sub.get('source',{}).get('exact_prior_source_roles'):
                source_hits.append({'path':rec['path'],'member':sub.get('member'),**sub['source']})
    report['legacy_validation_role_candidates']=ready
    report['exact_legacy_selector_source_hits']=source_hits
    report['selection_authorized']=False
    report['not_found_meaning']='Not found within the recorded bounded source scopes, not proof of absence on the server.'
    report['outcome']='COLLECTED_VALIDATION_PROVENANCE_REQUIRES_REVIEW' if not limits else 'PARTIAL_BOUNDED_COLLECTION_REVIEW_LIMITS'


def completed_gate(d,report):
    for k,rf,df,mf,expected in [
        ('a4','01_FIXED_FUSION_REVIEW.json','03_FUSION_DESIGN.json','10_FUSION_SOURCE_SHA256.json','PASS_FIXED_DEIT_MATCHED_FUSION_AND_PAIRED_ANALYSIS'),
        ('a3','01_OUTER_EVALUATION_REVIEW.json','03_EVALUATION_DESIGN.json','10_EVAL_SOURCE_SHA256.json','PASS_FROZEN_OUTER_EVALUATION_AND_PAIRED_ANALYSIS')]:
        cfg=d[k];root=Path(cfg['run']);need(root.is_dir() and not root.is_symlink(),'COMPLETED_ROOT_MISSING')
        gate(root/rf,cfg['report_sha256'],'EXACT_COMPLETED_'+k.upper()+'_REPORT')
        gate(root/'source'/mf,cfg['source_manifest_sha256'],'EXACT_COMPLETED_'+k.upper()+'_SOURCE')
        rr=read(root/rf);need(rr['status']==expected,'UPSTREAM_NOT_COMPLETE')
        worker=root/'WORKER_PID.json'
        if worker.is_file():
            observe(worker);w=read(worker);st=process_identity(int(w.get('pid',-1)))
            if st and st['state']!='Z' and str(st['start_ticks'])==str(w.get('start_ticks')):
                try:cmd=Path('/proc',str(w['pid']),'cmdline').read_bytes()
                except OSError:cmd=b''
                need(str(root).encode() not in cmd,'UPSTREAM_WORKER_ACTIVE_NO_RESTART')
        manifest=read(root/'source'/mf)
        for name,h in manifest['files'].items():
            need(Path(name).name==name,'UPSTREAM_SOURCE_NAME_UNSAFE');gate(root/'source'/name,h,'UPSTREAM_SOURCE')
    a3=Path(d['a3']['run']);gate(a3/'source/01_remote_evaluate_frozen_aux.py',d['a3']['controller_sha256'],'A3_VALIDATION_PRODUCER')
    for n,h in d['a3']['prediction_files'].items():gate(a3/n,h,'A3_OUTER_FILES_PROTECTED_NOT_TUNED')
    report['completed_a3_a4_verified_not_restarted']=True


def audit(d,report):
    completed_gate(d,report)
    parent=safe_source_module();ids,bridge=build_identity(d,parent)
    report['identity_authority']={'rows':len(ids),'patients':468,'fold_rois':d['outer_counts'],'p4c_bridge_bijective':True}
    audit_deit_reference(d,ids,bridge,parent,report)
    audit_a3_validation(d,ids,bridge,parent,report)
    # R2: no repeated generic discovery. Reuse A5 exact observed hashes.
    import importlib
    formats=importlib.import_module('17_exact_validation_formats')
    formats.audit_formats(d,report,sys.modules[__name__],ids,bridge,parent)
    print('A5_KNOWN_A3_VALIDATION_ROLE_MATCH='+str(report['a3_validation_role_match_count'])+'/20',flush=True)
    print('A5_LEGACY_ROLE_CANDIDATES='+str(len(report['legacy_validation_role_candidates'])),flush=True)
    print('A5_EXACT_SELECTOR_SOURCE_HITS='+str(len(report['exact_legacy_selector_source_hits'])),flush=True)
    print('A5_R2_FORMAT_COUNTS='+json.dumps(report.get('format_completion_counts',{}),sort_keys=True),flush=True)
    print('A5_SELECTION_EXECUTED=False|NO_OUTER_WEIGHT_SEARCH=True',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['audit'],required=True)
    ap.add_argument('--manifest-sha',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--bundle',type=Path,required=True)
    args=ap.parse_args()
    for p in (args.out,args.bundle):need(p.parent==HERE and re.fullmatch(r'/PATH/TO/TEMP/lss_w25a5r2_[A-Za-z0-9]{8}',str(HERE)) is not None,'OUTPUT_NOT_OWN_TEMP')
    gate(HERE/'10_DIAGNOSTIC_SOURCE_SHA256.json',args.manifest_sha,'A5_SOURCE_MANIFEST');m=read(HERE/'10_DIAGNOSTIC_SOURCE_SHA256.json')
    need(m['schema']=='W25_A5_SOURCE_MANIFEST_V1','A5_SOURCE_SCHEMA')
    for n,h in m['files'].items():need(Path(n).name==n,'SOURCE_NAME_UNSAFE');gate(HERE/n,h,'A5_SOURCE')
    d=read(HERE/'03_DIAGNOSTIC_DESIGN.json');need(d['schema']=='W25_A5_INNER_VALIDATION_ASSET_AUDIT_V1','A5_DESIGN_SCHEMA')
    flags=['new_training','new_model_inference','new_fusion','new_bootstrap','weight_search','probability_mutation','old_run_restart']
    for k in flags:need(d[k] is False,'FORBIDDEN_SCOPE_CHANGE')
    report={'schema':SCHEMA,'execution_status':'STARTED','started_unix':time.time(),'source_manifest_sha256':args.manifest_sha,
            'design_sha256':sha(HERE/'03_DIAGNOSTIC_DESIGN.json'),**{k:False for k in flags}}
    try:audit(d,report);report['execution_status']=COMPLETE
    except Exception as e:
        report['execution_status']=FAILED;report['error_code']=str(e) if isinstance(e,GateError) else 'UNEXPECTED_'+type(e).__name__
        print('A5_AUDIT_ERROR='+report['error_code'],flush=True)
    try:
        for p,h in PROTECTED.items():need(sha(p)==h,'READ_SOURCE_CHANGED')
        report['read_source_files_unchanged']=True;report['protected_file_count']=len(PROTECTED)
    except Exception:report['execution_status']=FAILED;report['read_source_files_unchanged']=False;report['protection_error']='READ_SOURCE_CHANGED'
    report['finished_unix']=time.time();report['selection_authorized']=False
    payload={'00_INNER_VALIDATION_ASSET_REVIEW.json':(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode(),
             '03_DIAGNOSTIC_DESIGN.json':(HERE/'03_DIAGNOSTIC_DESIGN.json').read_bytes(),
             '10_DIAGNOSTIC_SOURCE_SHA256.json':(HERE/'10_DIAGNOSTIC_SOURCE_SHA256.json').read_bytes(),**SOURCE_EXPORTS}
    payload['98_MEMBER_SHA256.json']=(json.dumps({n:bsha(b) for n,b in payload.items()},indent=2)+'\n').encode()
    with zipfile.ZipFile(args.bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
    status={k:report[k] for k in ['schema','execution_status','source_manifest_sha256','design_sha256','read_source_files_unchanged',*flags]}
    status.update(diagnostic_outcome=report.get('outcome','DIAGNOSTIC_STOPPED'),selection_authorized=False,
                  a3_validation_role_match_count=report.get('a3_validation_role_match_count',0),
                  legacy_role_candidate_count=len(report.get('legacy_validation_role_candidates',[])),
                  selector_source_hit_count=len(report.get('exact_legacy_selector_source_hits',[])),
                  bundle_sha256=sha(args.bundle),report_sha256=bsha(payload['00_INNER_VALIDATION_ASSET_REVIEW.json']))
    status['format_completion_counts']=report.get('format_completion_counts',{})
    if 'error_code' in report:status['error_code']=report['error_code']
    dump(args.out,status);print('W25_A5_AUDIT='+report['execution_status'],flush=True)
    print('NO_TRAINING_INFERENCE_WEIGHT_SELECTION_OR_BOOTSTRAP=True',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
