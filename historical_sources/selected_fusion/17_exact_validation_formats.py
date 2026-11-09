#!/usr/bin/env python3
"""A5-R2 exact-format completion. No training, inference, selection or pickle execution.
New adapter code; identity, SHA and workspace protection inherited from A5-R1.
"""
from __future__ import annotations
import ast, io, json, math, pickletools, re, zipfile
from pathlib import Path
from collections import Counter
import numpy as np

class FormatError(ValueError): pass

def require(ok, code):
    if not ok: raise FormatError(code)

class Symbol:
    def __init__(self, module, name): self.module,self.name=module,name

class Inert:
    def __init__(self, kind, args): self.kind,self.args,self.state=kind,args,None

# These symbols are represented as data. None of their functions/classes are called.
GLOBALS={('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),
         ('numpy','ndarray'),('numpy','dtype'),('_codecs','encode')}

def object_string_payload(data, expected_shape):
    """Interpret a narrow NumPy object-string pickle as inert data, never unpickle.
    Unsupported opcodes/symbols or non-string array values are rejected. This is
    intentionally not a general pickle interpreter or a checkpoint reader.
    """
    require(len(data)<=2_000_000,'OBJECT_PAYLOAD_TOO_LARGE')
    require(len(expected_shape)==1 and 0<=expected_shape[0]<=10000,'OBJECT_SHAPE_LIMIT')
    stack=[]; memo={}; mark=object(); count=0; stop=None
    def pop():
        require(bool(stack),'PICKLE_STACK_UNDERFLOW');return stack.pop()
    def take_mark():
        vals=[]
        while stack:
            x=pop()
            if x is mark:return vals[::-1]
            vals.append(x)
        raise FormatError('PICKLE_MARK_MISSING')
    for op,arg,pos in pickletools.genops(data):
        count+=1;require(count<=100000 and len(stack)<=30000 and len(memo)<=30000,'PICKLE_RESOURCE_LIMIT')
        name=op.name
        if name=='PROTO': require(2<=arg<=5,'PICKLE_PROTOCOL_NOT_SUPPORTED')
        elif name=='FRAME':require(0<=arg<=len(data),'PICKLE_FRAME_LIMIT')
        elif name=='MARK':stack.append(mark)
        elif name=='NONE':stack.append(None)
        elif name=='NEWTRUE':stack.append(True)
        elif name=='NEWFALSE':stack.append(False)
        elif name in ('BININT','BININT1','BININT2','INT','LONG','LONG1','LONG4'):
            require(type(arg) is int and abs(arg)<=10_000_000,'PICKLE_INT_LIMIT');stack.append(arg)
        elif name in ('BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','UNICODE'):
            require(isinstance(arg,str) and len(arg)<=512,'PICKLE_STRING_LIMIT');stack.append(arg)
        elif name in ('BINBYTES','SHORT_BINBYTES','BINBYTES8','BINSTRING','SHORT_BINSTRING'):
            require(isinstance(arg,(str,bytes)) and len(arg)<=512,'PICKLE_BYTES_LIMIT');stack.append(arg)
        elif name=='EMPTY_TUPLE':stack.append(())
        elif name=='EMPTY_LIST':stack.append([])
        elif name=='TUPLE':stack.append(tuple(take_mark()))
        elif name in ('TUPLE1','TUPLE2','TUPLE3'):
            n=int(name[-1]);require(len(stack)>=n,'PICKLE_TUPLE_UNDERFLOW');v=stack[-n:];del stack[-n:];stack.append(tuple(v))
        elif name=='LIST':stack.append(take_mark())
        elif name=='APPEND':
            v=pop();require(stack and type(stack[-1]) is list,'PICKLE_APPEND_TARGET');stack[-1].append(v)
        elif name=='APPENDS':
            vals=take_mark();require(stack and type(stack[-1]) is list,'PICKLE_APPENDS_TARGET');stack[-1].extend(vals)
        elif name in ('BINPUT','LONG_BINPUT','PUT'):
            require(stack and 0<=int(arg)<30000,'PICKLE_MEMO_LIMIT');memo[int(arg)]=stack[-1]
        elif name=='MEMOIZE':
            require(stack,'PICKLE_MEMO_EMPTY');memo[len(memo)]=stack[-1]
        elif name in ('BINGET','LONG_BINGET','GET'):
            require(int(arg) in memo,'PICKLE_MEMO_MISSING');stack.append(memo[int(arg)])
        elif name in ('GLOBAL','STACK_GLOBAL'):
            if name=='GLOBAL': mod,cls=arg.split(' ',1)
            else: cls=pop();mod=pop()
            require(type(mod) is str and type(cls) is str and (mod,cls) in GLOBALS,'PICKLE_GLOBAL_NOT_ALLOWED')
            stack.append(Symbol(mod,cls))
        elif name=='REDUCE':
            args=pop();sym=pop();require(isinstance(sym,Symbol) and type(args) is tuple,'PICKLE_REDUCE_INVALID')
            if sym.module=='_codecs':
                require(len(args)==2 and isinstance(args[0],str) and args[1]=='latin1','PICKLE_ENCODING_INVALID')
                stack.append(args[0].encode('latin1'))
            else:
                require(sym.name in ('_reconstruct','dtype'),'PICKLE_REDUCE_SYMBOL_NOT_ALLOWED')
                stack.append(Inert(sym.name,args))
        elif name=='BUILD':
            state=pop();obj=pop();require(isinstance(obj,Inert) and obj.state is None,'PICKLE_BUILD_INVALID');obj.state=state;stack.append(obj)
        elif name=='STOP':
            require(len(stack)==1,'PICKLE_FINAL_STACK_INVALID');stop=pos;break
        else: raise FormatError('PICKLE_OPCODE_NOT_ALLOWED_'+name)
    require(stop is not None and stop+1==len(data),'PICKLE_TRAILING_OR_MISSING_STOP')
    obj=stack[0];require(isinstance(obj,Inert) and obj.kind=='_reconstruct','OBJECT_NOT_NUMPY_ARRAY')
    require(len(obj.args)==3 and isinstance(obj.args[0],Symbol) and obj.args[0].name=='ndarray' and obj.args[1]==(0,) and obj.args[2]==b'b','OBJECT_RECONSTRUCT_SIGNATURE')
    st=obj.state;require(type(st) is tuple and len(st)==5 and st[0]==1 and st[1]==tuple(expected_shape) and type(st[3]) is bool,'OBJECT_STATE_INVALID')
    dt=st[2];require(isinstance(dt,Inert) and dt.kind=='dtype' and dt.args in (('O8',False,True),('O4',False,True),('O',False,True)),'OBJECT_DTYPE_NOT_STRINGS_CONTAINER')
    require(type(st[4]) is list and len(st[4])==expected_shape[0],'OBJECT_LIST_SHAPE_MISMATCH')
    vals=[]
    for v in st[4]:
        require(type(v) in (str,bytes),'OBJECT_ARRAY_NONSTRING_VALUE')
        if isinstance(v,bytes):v=v.decode('utf-8')
        require(len(v)<=160 and '\x00' not in v,'OBJECT_ID_STRING_INVALID');vals.append(v)
    return np.asarray(vals,dtype=str)


def npy_header(raw):
    require(len(raw)<64_000_000,'NPY_SIZE_LIMIT');s=io.BytesIO(raw);ver=np.lib.format.read_magic(s)
    if ver==(1,0):shape,order,dt=np.lib.format.read_array_header_1_0(s)
    elif ver==(2,0):shape,order,dt=np.lib.format.read_array_header_2_0(s)
    else:raise FormatError('NPY_VERSION_NOT_SUPPORTED')
    require(len(shape)<=3 and all(type(n) is int and 0<=n<=20000 for n in shape),'NPY_SHAPE_LIMIT')
    require(math.prod(shape)<=200000,'NPY_ELEMENT_LIMIT');return shape,order,dt,s.tell()


def load_npz_members(raw):
    """A rejected member does not discard independent numeric members."""
    require(len(raw)<=8_000_000,'NPZ_SIZE_LIMIT');arrays={};schemas={};errors={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        inf=z.infolist();require(len(inf)<=32 and len({x.filename for x in inf})==len(inf),'NPZ_MEMBER_COUNT_OR_DUPLICATE')
        require(sum(x.file_size for x in inf)<=64_000_000,'NPZ_EXPANSION_LIMIT')
        for x in inf:
            require(re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,100}\\.npy',x.filename) is not None,'NPZ_MEMBER_NAME_INVALID')
            key=x.filename[:-4];b=z.read(x)
            try:
                shape,order,dt,offset=npy_header(b);schemas[key]={'shape':list(shape),'dtype':str(dt),'object_dtype':bool(dt.hasobject)}
                if dt.hasobject:
                    require(key in ('foramen_id','foramen_ids','patient','patient_id','ids','fid'),'OBJECT_MEMBER_NOT_IDENTITY')
                    arrays[key]=object_string_payload(b[offset:],shape);schemas[key]['reader']='INERT_PICKLE_OPCODE_INTERPRETER_NO_EXECUTION'
                else:
                    require(dt.fields is None and dt.kind in 'biufUS','NPY_DTYPE_NOT_ALLOWED')
                    arrays[key]=np.load(io.BytesIO(b),allow_pickle=False);schemas[key]['reader']='NUMPY_ALLOW_PICKLE_FALSE'
            except (FormatError,ValueError,UnicodeError,EOFError,OverflowError) as e:
                errors[key]=str(e) if isinstance(e,FormatError) else 'ARRAY_FORMAT_'+type(e).__name__
    return arrays,schemas,errors


def exact_role(indices, ids, f, role, labels=None, patients=None, normalize=None):
    ii=[int(i) for i in indices];vf=(f+1)%5;expected=vf if role=='validation' else f
    require(role in ('validation','test'),'ROLE_UNKNOWN')
    expected_set={i for i,v in ids.items() if v[1]==expected}
    require(len(ii)==len(set(ii)) and set(ii)==expected_set,'EXACT_ROLE_ROW_SET_MISMATCH')
    if labels is not None:
        yy=np.asarray(labels);require(yy.ndim==1 and len(yy)==len(ii),'LABEL_SHAPE')
        require(np.isfinite(yy.astype(float)).all() and np.equal(yy.astype(float),yy.astype(int)).all(),'LABEL_INTEGER')
        require(all(int(y)==ids[i][2] for i,y in zip(ii,yy)),'LABEL_MISMATCH')
    if patients is not None:
        require(len(patients)==len(ii) and normalize is not None,'PATIENT_SHAPE')
        require(all(normalize(str(p))==ids[i][0] for i,p in zip(ii,patients)),'PATIENT_MISMATCH')
    ps={ids[i][0] for i in ii};outerps={v[0] for v in ids.values() if v[1]==f}
    if role=='validation':require(not ps&outerps,'VALIDATION_TEST_PATIENT_OVERLAP')
    return {'row_count':len(ii),'patient_count':len(ps),'model_outer_fold':f,'data_fold':expected,
            'role':role,'exact_authority_rows':True,'labels_checked':labels is not None,
            'patient_column_checked':patients is not None,'outer_patient_overlap_count':len(ps&outerps) if role=='validation' else None}


def numeric_profile(p):
    a=np.asarray(p,dtype=float);require(a.ndim==2 and a.shape[1]==4 and len(a)>0,'PROBABILITY_SHAPE')
    finite=np.isfinite(a);out={'shape':list(a.shape),'finite':bool(finite.all())}
    if not finite.all():return out
    sums=a.sum(1);out.update(negative_values=int((a<0).sum()),above_one=int((a>1).sum()),
        max_abs_mass_error=float(np.max(np.abs(sums-1))),min_sum=float(sums.min()),max_sum=float(sums.max()),
        rows_exceeding_original_2e_6=int((np.abs(sums-1)>2e-6).sum()))
    return out


def qwk(y,p):
    # Deterministic validation checkpoint reproduction ONLY, not model selection.
    y=np.asarray(y,int);p=np.asarray(p,int);labels=np.union1d(y,p)
    require(len(labels)>1,'QWK_UNDEFINED')
    a=np.searchsorted(labels,y);b=np.searchsorted(labels,p);n=len(labels)
    cm=np.bincount(a*n+b,minlength=n*n).reshape(n,n).astype(float)
    w=(np.arange(n)[:,None]-np.arange(n)[None,:])**2
    den=(w*np.outer(cm.sum(1),cm.sum(0))/cm.sum()).sum();require(den>0,'QWK_UNDEFINED')
    return float(1-(w*cm).sum()/den)


def report_metadata(obj):
    """Additional scalar provenance fields only; never return patient-level rows."""
    keys={'outer','outer_fold','inner','inner_fold','fixed_val','fixed_validation_fold','selected_inner_validation_fold',
      'rows','n_rows','train_rows','val_rows','test_rows','train_patients','val_patients','test_patients',
      'selected_alpha','alpha','epoch','best_epoch','checkpoint_epoch','epochs_ran','seed','run_seed',
      'outer_test_opened','outer_oof_used_for_weight_selection','training_performed','inference_performed',
      'test_used_for_selection','test_inference_after_checkpoint_freeze','checkpoint_selection','status',
      'coral_checkpoint_sha256','deit_checkpoint_sha256','deit_manifest_sha256','checkpoint_sha256',
      'checkpoint_path','deit_checkpoint','coral_checkpoint','deit_manifest','fixed_oof_sha256'}
    results=[]
    def walk(x,level=0):
        if level>9:return
        if isinstance(x,dict):
            row={k:v for k,v in x.items() if k in keys and (type(v) in (str,int,float,bool) or v is None)}
            if row:results.append(row)
            for k,v in x.items():
                if k not in ('patients','patient_ids','predictions','probabilities','row_ids','cases'):walk(v,level+1)
        elif isinstance(x,list):
            for v in x[:100]:
                if isinstance(v,(dict,list)):walk(v,level+1)
    walk(obj);return results[:100]


def audit_formats(d,report,A,ids,bridge,parent):
    """Exact SHA-locked files obtained in A5. No repeated root-wide CSV scan."""
    cfg=d['format_completion'];original=A.read(A.HERE/'16_REVIEWED_A5_REPORT.json')
    A.gate(A.HERE/'16_REVIEWED_A5_REPORT.json',cfg['reviewed_a5_report_sha256'],'REVIEWED_A5_R1')
    locks={x['path']:x for x in cfg['files']}
    for x in cfg['files']:A.gate(x['path'],x['sha256'],'A5_OBSERVED_EXACT_FILE')
    for x in cfg['a3_current_validation_locks']:A.gate(x['path'],x['sha256'],'A5_CURRENT_VALIDATION_LOCK')
    report['a3_file_hashes_now_locked_against_a5']=20
    report['rejected_outer_test_false_candidates']=cfg['legacy_false_role_candidates']
    report['legacy_validation_role_candidates']=[]
    report['exact_legacy_selector_source_hits']=original['exact_legacy_selector_source_hits']
    # Historical summary and CSVs are anchored by the archive used by the recovered selector.
    archive=Path(cfg['historical_dual_archive_path']);summary_name='00_P10_C3_R6_R4_R2_SUMMARY.json'
    A.gate(archive,cfg['historical_dual_archive_sha256'],'ORIGINAL_SELECTOR_INPUT_ARCHIVE')
    dual=[]
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist())==len(set(z.namelist())),'DUAL_ARCHIVE_DUPLICATE')
        summary=json.loads(z.read(summary_name).decode('utf-8-sig'))
        require(summary.get('outer_test_opened') is False and summary.get('outer_oof_used_for_weight_selection') is False,'HISTORICAL_DUAL_SCOPE_CONFLICT')
        report['historical_dual_metadata']=report_metadata(summary)
        for f in range(5):
            name=f'inner_probs/outer_{f}_dual_inner_validation_probs.csv';raw=z.read(name)
            loose=archive.parent/name;A.gate(loose,locks[str(loose)]['sha256'],'LOOSE_DUAL_LOCK')
            require(raw==loose.read_bytes(),'ARCHIVE_LOOSE_DUAL_DIFFERENCE')
            rows=A.csvrows(raw);cols=['_id','y_true_infer_coral','y_true_infer_deit','outer','inner_fold','selected_alpha']+[b+str(i) for b in ('coral_p','deit_p') for i in range(4)]
            A.require_columns(rows,cols,'EXACT_DUAL')
            ii=[A.integer(r['_id']) for r in rows]
            require(all(A.integer(r['outer'])==f and A.integer(r['inner_fold'])==(f+1)%5 for r in rows),'DUAL_EXPLICIT_MODEL_ROLE')
            rc=exact_role(ii,ids,f,'validation',[A.integer(r['y_true_infer_coral']) for r in rows])
            exact_role(ii,ids,f,'validation',[A.integer(r['y_true_infer_deit']) for r in rows])
            cp=np.asarray([[float(r['coral_p'+str(k)]) for k in range(4)] for r in rows]);dp=np.asarray([[float(r['deit_p'+str(k)]) for k in range(4)] for r in rows])
            pc,pd=numeric_profile(cp),numeric_profile(dp)
            require(pc['finite'] and pd['finite'] and pc['negative_values']==pd['negative_values']==0 and pc['above_one']==pd['above_one']==0,'DUAL_PROBABILITY_INVALID')
            # Report, do not silently normalize or broaden the actual selection policy.
            rc.update(archive_member=name,sha256=A.bsha(raw),archive_loose_equal=True,
                      coral_probability_profile=pc,deit_probability_profile=pd,
                      selected_alpha_values=sorted({float(r['selected_alpha']) for r in rows}),
                      patient_identity_recovered_from_authoritative_id_map=True,
                      model_probability_provenance_review_pending=True)
            dual.append(rc)
            print(f'A5_R2_DUAL_VALIDATION_ROLE|outer={f}|rows={len(rows)}|PASS',flush=True)
    report['historical_dual_validation_files']=dual
    # Individual NPZ members: safe numeric load; object strings use an inert grammar.
    p4_index=A.archive_table(d['p4c_archive'],'index_member','index_sha256')
    p4p=Path(d['p4c_archive']['path']).parent
    full=A.archive_table(d['p4c_archive'],'prediction_member','prediction_sha256') if 'prediction_member' in d['p4c_archive'] else None
    # Read original CSV from locked P4C archive, not a different protocol.
    if full is None:
        with zipfile.ZipFile(d['p4c_archive']['path']) as z:full=A.csvrows(z.read('01_OOF_PREDICTIONS.csv'))
    oof={str(r['foramen_id']).strip():r for r in full};require(len(oof)==2978,'P4C_OOF_DUPLICATES')
    npz_results=[];metadata=[]
    for f in range(5):
        fd=p4p/f'fold{f}';done=A.read(fd/'COMPLETE.json')
        metadata.append({'outer':f,'source_sha256':A.sha(fd/'COMPLETE.json'),'data':report_metadata(done)})
        require(int(done.get('outer',-1))==f and int(done.get('fixed_val',-1))==(f+1)%5,'P4C_COMPLETE_ROLE_MISMATCH')
        require(done.get('test_used_for_selection') is False and done.get('test_inference_after_checkpoint_freeze') is True,'P4C_COMPLETE_SCOPE_MISMATCH')
        for role,name in [('validation','val_predictions.npz'),('test','test_predictions.npz')]:
            path=fd/name;rec={'model_outer_fold':f,'role':role,'path':str(path),'sha256':A.sha(path),'checkpoint_sha256':A.sha(fd/'best_ema_state.pt'),'model_origin_from_directory_only':True,'ready_for_selection':False}
            try:
                arr,schemas,errors=load_npz_members(path.read_bytes());rec.update(array_schema=schemas,member_errors=errors)
                fidnames=[k for k in ('foramen_id','foramen_ids','fid','ids') if k in arr]
                require(len(fidnames)==1,'NPZ_FORAMEN_ID_FIELD_AMBIGUOUS_OR_UNREADABLE')
                fids=np.asarray(arr[fidnames[0]]);require(fids.ndim==1,'NPZ_ID_SHAPE')
                values=[str(v).strip() for v in fids];require(all(x in bridge for x in values),'NPZ_ID_NOT_IN_EXACT_FORAMEN_BRIDGE')
                ii=[bridge[x] for x in values];pn=[k for k in ('P','probs','probabilities','p') if k in arr]
                require(len(pn)==1,'NPZ_PROBABILITY_FIELD_AMBIGUOUS_OR_UNREADABLE');P=np.asarray(arr[pn[0]],float)
                require(P.shape==(len(ii),4),'NPZ_PROBABILITY_COUNT');prof=numeric_profile(P);rec['probability_profile']=prof
                require(prof['finite'] and prof['negative_values']==0 and prof['above_one']==0,'NPZ_PROBABILITY_INVALID')
                yn=[k for k in ('y','y_true','labels') if k in arr];require(len(yn)<=1,'NPZ_LABEL_FIELD_AMBIGUOUS')
                ynval=arr[yn[0]] if yn else None
                if role=='validation':require(ynval is not None,'NPZ_VALIDATION_LABEL_MISSING')
                pidnames=[k for k in ('patient','patient_id') if k in arr];require(len(pidnames)<=1,'NPZ_PATIENT_FIELD_AMBIGUOUS')
                role_check=exact_role(ii,ids,f,role,ynval,arr[pidnames[0]] if pidnames else None,parent.normalize_pid)
                rec['role_check']=role_check
                # Explicit model fold fields, when present, must match the producer fold.
                for k in ('outer','outer_fold','model_outer_fold'):
                    if k in arr:
                        vv=np.asarray(arr[k]);require(vv.size in (1,len(ii)) and np.equal(vv,f).all(),'NPZ_MODEL_FOLD_MISMATCH')
                if 'source_fold' in arr:rec['source_fold_values_observed_not_assumed']=[int(x) for x in np.unique(arr['source_fold'])]
                if role=='test':
                    ref=np.asarray([[float(oof[x]['p'+str(i)]) for i in range(4)] for x in values]);maxerr=float(np.abs(P-ref).max())
                    rec['outer_archive_probability_max_abs_error']=maxerr
                    rec['outer_archive_argmax_mismatches']=int(np.count_nonzero(P.argmax(1)!=ref.argmax(1)))
                    # Serialization-level tolerance used for lineage only, not changing arrays.
                    require(maxerr<=1e-7 and rec['outer_archive_argmax_mismatches']==0,'NPZ_TEST_VS_ARCHIVE_MISMATCH')
                else:
                    valq=qwk(np.asarray(ynval,int),P.argmax(1));recordq=float(done['best_val_qwk']);gap=abs(valq-recordq)
                    rec['validation_checkpoint_qwk_reproduction']={'computed':valq,'archived':recordq,'max_abs_error':gap}
                    require(gap<=1e-10,'P4C_VALIDATION_SELECTED_EPOCH_QWK_MISMATCH')
                rec['status']='EXACT_CONTENT_AND_ROLE_MATCH_PRODUCER_REVIEW_PENDING'
            except (FormatError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as e:
                rec['status']='FORMAT_OR_ROLE_REVIEW_REQUIRED';rec['error_code']=str(e) if isinstance(e,FormatError) else 'FORMAT_'+type(e).__name__
            npz_results.append(rec)
        print(f'A5_R2_P4C_NPZ|outer={f}|read_without_pickle_execution=True',flush=True)
    report['p4c_prediction_archives']=npz_results;report['p4c_training_role_metadata']=metadata
    # Recover exact producer text. Read/parse only, never import or execute.
    exact_sources=[]
    for spec in cfg['original_selector_sources']:
        raw=Path(spec['path']).read_bytes();h=A.bsha(raw);require(h==spec['sha256'],'EXACT_PRODUCER_SOURCE_DRIFT')
        text=raw.decode('utf-8-sig');ast.parse(text)
        require(not re.search(r'(?i)(?:password|private[_-]?key|api[_-]?key)\s*=\s*[\'\"][^\'\"]+[\'\"]',text),'SOURCE_SECRET_LITERAL_STOP')
        member='sources/producer_'+h+'.py.txt';A.SOURCE_EXPORTS[member]=raw
        exact_sources.append({'path':spec['path'],'sha256':h,'export_member':member,'executed':False,'locked_to_a5_current_observation':True})
    report['exact_producer_sources']=exact_sources
    # Optional narrow P4C code scope not visited by the first A5 scan.
    spec=cfg['optional_p4c_source_lookup'];base=Path(spec['parent']);roots=[];sources=[];skipped=[];visited=0
    if base.is_dir():roots=[p for p in sorted(base.iterdir()) if p.name.startswith(spec['prefix']) and p.is_dir() and not p.is_symlink()]
    require(len(roots)<=spec['max_roots'],'P4C_SOURCE_ROOT_LIMIT')
    for root in roots:
        queue=[(root,0)]
        while queue:
            path,depth=queue.pop(0)
            for p in sorted(path.iterdir()):
                visited+=1;require(visited<=spec['max_entries'],'P4C_SOURCE_ENTRY_LIMIT')
                if p.is_symlink():continue
                if p.is_dir():
                    if depth<spec['max_depth'] and p.name not in ('__pycache__','.git','images','rois'):queue.append((p,depth+1))
                    continue
                if p.suffix not in spec['extensions'] or not p.is_file():continue
                if p.stat().st_size>spec['max_file_bytes']:skipped.append({'name':p.name,'reason':'SIZE_LIMIT'});continue
                raw=p.read_bytes();h=A.observe(p)
                if p.suffix=='.py':
                    try:text=raw.decode('utf-8-sig');tree=ast.parse(text)
                    except (UnicodeError,SyntaxError):continue
                    if not any(t in text for t in ('val_predictions.npz','best_ema_state.pt','roi_source40_index.csv')):continue
                    require(not re.search(r'(?i)(?:password|private[_-]?key|api[_-]?key)\s*=\s*[\'\"][^\'\"]+[\'\"]',text),'SOURCE_SECRET_LITERAL_STOP')
                    name='sources/p4c_candidate_'+h+'.py.txt';A.SOURCE_EXPORTS[name]=raw
                    sources.append({'path':str(p),'sha256_observed_now':h,'export_member':name,'functions':[n.name for n in tree.body if isinstance(n,ast.FunctionDef)],'executed':False,'generation_lineage_not_proved_by_name':True})
                elif p.suffix=='.json':
                    try:meta=A.metadata_summary(json.loads(raw.decode('utf-8-sig')))
                    except (ValueError,UnicodeError):continue
                    sources.append({'path':str(p),'sha256_observed_now':h,'metadata':meta})
                require(len(sources)<=spec['max_files'],'P4C_SOURCE_FILE_LIMIT')
    report['p4c_producer_source_scope']={'parent':str(base),'prefix':spec['prefix'],'root_count':len(roots),'entries':visited,'records':sources,'skipped':skipped,'not_found_is_not_absence':True}
    report['format_completion_counts']={'a3_current_sha_matched':20,'dual_validation_role_matches':len(dual),
        'p4c_validation_matches':sum(x['role']=='validation' and x['status']=='EXACT_CONTENT_AND_ROLE_MATCH_PRODUCER_REVIEW_PENDING' for x in npz_results),
        'p4c_test_archive_matches':sum(x['role']=='test' and x['status']=='EXACT_CONTENT_AND_ROLE_MATCH_PRODUCER_REVIEW_PENDING' for x in npz_results),
        'rejected_outer_test_candidates':len(cfg['legacy_false_role_candidates']),'new_selection':False,'new_model_inference':False}
    report['selection_authorized']=False
    report['outcome']='EXACT_FORMAT_COMPLETION_PRODUCER_REVIEW_PENDING'
    report['not_found_meaning']='No replacement model or OOF-based tuning. Unresolved format/provenance is reported, not converted into permission to select weights.'
