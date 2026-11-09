#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W24-P5: strictly read-only provenance and split audit before any R11 374/94 training.

Patient-level raw data is read locally on the scientific server, never exported.
No training, inference, hyperparameter search, model loading, bootstrap or editing.
Historical assets are opened only after exact SHA-256 validation.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import re
import sys
import zipfile
from collections import Counter

SCHEMA='LSS_PAPER2_W24P5_R2_R11_SPLIT_GRADE_CROSSWALK_V1'
ROOT=pathlib.Path('/PATH/TO/RESEARCH_WORKSPACE')
SCRIPT_ROOT=pathlib.Path('/PATH/TO/PROJECT/experiments/baseline_v1/scripts')

# Exactly 94 *published dataset-internal* IDs, transcribed from the authors'
# public data/splits.csv test rows, NOT user-derived IDs or medical identifiers.
# Public snapshot: exact original split must be supplied separately; no IDs bundled.
def _load_public_test_ids_from_source_csv():
    split_path = os.environ.get('PAPER2_SOURCE_SPLITS_CSV')
    if not split_path:
        raise SystemExit('PUBLIC_SNAPSHOT_REQUIRES_PAPER2_SOURCE_SPLITS_CSV: supply the original published data/splits.csv; no substitute split is generated')
    split_bytes = pathlib.Path(split_path).expanduser().read_bytes()
    # Exact official source CSV bytes measured 2026-11-09; not a retroactive historical freeze.
    if hashlib.sha256(split_bytes).hexdigest() != '1dab827011428f783b51624a32c52ca063e67a608375bfe41ee52cb6e0f5380f':
        raise SystemExit('PUBLIC_SOURCE_SPLIT_CSV_DIFFERS_FROM_OFFICIAL_SNAPSHOT_20261109')
    with io.StringIO(split_bytes.decode('utf-8-sig'), newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['patient', 'split']:
            raise SystemExit('PUBLIC_SOURCE_SPLIT_COLUMNS_MUST_BE_patient_split')
        test_ids = [row['patient'].strip() for row in reader if row['split'] == 'test']
    if len(test_ids) != 94 or len(set(test_ids)) != 94 or not all(re.fullmatch(r'\d{4}', value) for value in test_ids):
        raise SystemExit('PUBLIC_SOURCE_SPLIT_TEST_IDS_INVALID_OR_COUNT_NOT_94')
    canonical = '\n'.join(sorted(test_ids)).encode('utf-8')
    if hashlib.sha256(canonical).hexdigest() != '9d2cc0d11364f06c8c1f777cc1f37b77a6cef28a64a8134dae453d8fd5034f45':
        raise SystemExit('PUBLIC_SOURCE_SPLIT_TEST_ID_SET_DIFFERS_FROM_FROZEN_ORIGINAL')
    return set(test_ids)

PUBLIC_TEST_IDS = _load_public_test_ids_from_source_csv()
EXPECTED_PUBLIC_TEST_COUNT=94
if len(PUBLIC_TEST_IDS)!=94 or len(set(PUBLIC_TEST_IDS))!=94:
    raise SystemExit('PUBLIC_TEST_ID_TRANSCRIPTION_COUNT_GATE_FAILED')

ASSETS={
 'P4_1C_COMMON_RESULT':(
   ROOT/'server_runs/paper2_p4_1c_source_reconstructed_resnet18_corn_formal_20260918/PAPER2_P4_1C_SOURCE_RECONSTRUCTED_RESNET18_CORN_FORMAL_RESULT.zip',
   '4089ed9d6e446abd76b16433f54a41a38c263a612bacefdee250d8f61e21804f'),
 'P4_1_SR1_374_94_RESULT':(
   ROOT/'server_runs/paper2_p4_1_sr1_source_protocol_reconstruction_20260919/PAPER2_P4_1_SR1_SOURCE_PROTOCOL_RECONSTRUCTION_RESULT.zip',
   '28cbc2e75dbbaacd92c7a87691c480bc522d218cb30ec99002ab4513603401a7'),
 'R2_468_PATIENT_AUTHORITATIVE_MANIFEST':(
   ROOT/'server_runs/secondgen_step01f_r2_patient_split_403link_20260820/01_FROZEN_SOURCE_MANIFEST_WITH_PATIENT_SPLIT.csv',
   '39371acae04df30197c19be73a2a3441379f4596f494a6651d93255ad7c8d907'),
 'R11_R6_R6_FROZEN_FINAL':(
   ROOT/'server_runs/paper2_p10_c3_r6_r6_selected_weight_outer_oof_eval_20261005/P10_C3_R6_R6_SELECTED_WEIGHT_OUTER_OOF_EVAL_RESULT.zip',
   '617076d01c99104f81bf91f1e89628d4813df1a0944d93fc0437ab25c87cfb2e'),
 'R09_R4_ALPHA_SELECTION':(
   ROOT/'server_runs/paper2_p10_c3_r4_innerval_alpha_search_20261005/P10_C3_R4_INNERVAL_ALPHA_SEARCH_RESULT.zip',
   'ca30bf25c1b4f99872c8c1cdcbc8d929269454d13b67ecc66c6ca59d4f1d977f'),
}
SOURCE_EXPECTED={
 'R09_CORE_54':('b8e5360657fa467e74db846c4d03e7f5f80901aa2108e66587b04a58e89cb259',SCRIPT_ROOT/'54_train_coral_risk_protocol_v3_full468_oof.py'),
 'DEIT_CORE_52':('dc8c0b4b1aafe640d7ce764c3efff984e5154430621c946ea7025c0b4d3887ae',SCRIPT_ROOT/'52_train_official_deit_protocol_v3_full468_oof.py'),
 'COMMON_04A':('6032f392ea409a116d1ef5619c6b33f9fc37bc297786f0a4a58d70311a7b44c6',None),
 'R09_TRAINER':('2529afd31e91dcc7327381de640651605c25a2eafc79cc846cd5dc6763f2c3bf',None),
 'PHADKE_SR1_TRAINER':('9ff0d2a05d49ab884d9381ca396127f28d9759c02916eb8515af004f8cf47cae',None),
 'PHADKE_SR1_AGGREGATOR':('b97c2aaa45805cb80990b1e4efa38a8915a2a12a2fadd33c8600aeb0fea2a530',None),
}

BAD_PATH=re.compile(r'(^/|\\|(^|/)\.\.(/|$)|\x00)')

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for buf in iter(lambda:f.read(1024*1024),b''): h.update(buf)
 return h.hexdigest()
def fail_or_pass_asset(name,path,expected):
 r={'tag':name,'path':str(path),'expected_sha256':expected,'status':'MISSING'}
 if path.is_file():
  h=sha(path);r['observed_sha256']=h
  r['status']='PASS_EXACT_SHA' if h==expected else 'SHA_MISMATCH'
 print('ASSET_SHA|'+name+'|'+r['status'],flush=True)
 return r

def read_zip_inventory(path):
 arr=[]
 with zipfile.ZipFile(path) as z:
  for it in z.infolist():
   if it.is_dir():continue
   # Member name, SHA, bytes, and at most column NAMES; no rows or IDs are exported.
   rec={'name':it.filename,'size_bytes':it.file_size,'sha256':sha_bytes(z.read(it.filename)) if it.file_size<50_000_000 else 'SIZE_LIMITED'}
   if it.filename.lower().endswith('.csv'):
    with z.open(it) as f:
     line=f.readline(10000).decode('utf-8-sig',errors='replace').strip()
     rec['columns']=next(csv.reader([line]),[])[:90]
   elif it.filename.lower().endswith('.json') and it.file_size<=300000:
    try:
     obj=json.loads(z.read(it.filename).decode('utf-8-sig'))
     rec['top_level_keys']=list(obj.keys())[:60] if isinstance(obj,dict) else ['<list>',f'length={len(obj)}'] if isinstance(obj,list) else ['<scalar>']
    except Exception:rec['top_level_keys']=['<parse_fail>']
   arr.append(rec)
 return arr

def normalize_pid(v):
 s=str(v).strip()
 if s.endswith('.0') and re.fullmatch(r'\d+\.0',s):s=s[:-2]
 if re.fullmatch(r'\d{1,4}',s):return s.zfill(4)
 if re.fullmatch(r'(?i)(?:patient[_ -]?|pid[_ -]?)[0-9]{1,4}',s):
  return re.search(r'([0-9]{1,4})$',s).group(1).zfill(4)
 return None

def parse_grade(v):
 # Equivalent exact integer or integer-valued decimal encodings, never rounding.
 s=str(v).strip()
 if re.fullmatch(r'[0-3]',s):return int(s)
 if re.fullmatch(r'[0-3]\.0+',s):return int(s[0])
 return None

def numeric_grades(rows, columns):
 # Strict four-grade labels only; do not coerce missing/ambiguous values.
 for col in columns:
  values=[]
  for row in rows:
   parsed=parse_grade(row.get(col,''))
   if parsed is None:
    values=[];break
   values.append(parsed)
  if len(values)==len(rows):
   return col,values
 return None,None

def inspect_manifest(path):
  # W24P5-R2: the R2 source-image manifest deliberately contains NO grade column.
  # Patient membership and ROI counts are verified here; grade/ROI identity comes
  # independently from the exact-SHA P4.1-C source-index and P4.1-SR1 archives.
  res={'status':'UNRESOLVED','privacy':'only aggregate counts, hashes and column names',
       'parser':'stdlib_csv_strict_v2_with_independent_grade_authority'}
  try:
   with path.open('r',encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f)
    if not reader.fieldnames or 'patient_id' not in reader.fieldnames:
     raise ValueError('R2_PATIENT_ID_COLUMN_MISSING')
    rows=list(reader)
    columns=list(reader.fieldnames)
   pids=[normalize_pid(r['patient_id']) for r in rows]
   n_bad=sum(v is None for v in pids)
   counts=Counter(pids)
   tests=set(PUBLIC_TEST_IDS)
   rtest={p:n for p,n in counts.items() if p in tests}
   rdev={p:n for p,n in counts.items() if p not in tests}
   res.update({'columns':columns[:90],'n_rows':len(rows),
               'grade_column_present':any(c.lower() in {'grade','y_true','severity'} for c in columns),
               'unique_patients':len(counts),'invalid_patient_ids':n_bad,
               'source_test_patients':len(rtest),'source_test_roi_rows':sum(rtest.values()),
               'development_patients':len(rdev),'development_roi_rows':sum(rdev.values()),
               'patient_overlap_train_test':0,
               'source_test_exact_public_patient_id_membership':set(rtest)==tests,
               'source_test_membership_sha256':sha_bytes(('\n'.join(sorted(rtest))+'\n').encode())})
   if (len(rows)==2978 and n_bad==0 and len(counts)==468 and
       len(rtest)==94 and sum(rtest.values())==580 and
       len(rdev)==374 and sum(rdev.values())==2398 and set(rtest)==tests):
    res['status']='EXACT_PUBLIC_SOURCE_PATIENT_SPLIT_374_94_AND_2398_580'
    res['_private_r2_rows']=rows
    res['_private_r2_counts']=counts
   else:res['status']='R2_SOURCE_PATIENT_OR_ROI_COUNT_MISMATCH'
   print('PATIENT_SOURCE_SPLIT_AUDIT|'+res['status'],flush=True)
  except Exception as e:
   res['status']='ERROR_READ_MANIFEST'
   res['error_type']=type(e).__name__
   print('PATIENT_SOURCE_SPLIT_AUDIT|ERROR_'+type(e).__name__,flush=True)
  return res


def read_csv_zip(z,member,required):
  with z.open(member) as raw:
   r=csv.DictReader(io.TextIOWrapper(raw,encoding='utf-8-sig',newline=''))
   if not required.issubset(set(r.fieldnames or [])):
    raise ValueError('CSV_REQUIRED_COLUMNS_MISSING_'+member)
   return list(r)


def grade_rows(rows,patient_col,grade_col,foramen_col):
  out={};count=Counter();grades=Counter();pids=set()
  for row in rows:
   pid=normalize_pid(row.get(patient_col,''));grade=parse_grade(row.get(grade_col,''));fid=str(row.get(foramen_col,'')).strip()
   if pid is None or grade is None or not fid or fid in out:
    raise ValueError('GRADE_ID_MISSING_OR_DUPLICATE')
   out[fid]=(pid,grade);count[pid]+=1;grades[grade]+=1;pids.add(pid)
  return out,count,grades,pids


def bbox_from(row,cols):
  try:
   from decimal import Decimal
   vals=[str(row.get(c,'')).strip() for c in cols]
   if any(not v for v in vals):return None
   return tuple(Decimal(v).normalize() for v in vals)
  except Exception:return None


def filename_part(val,stem):
  val=str(val or '').strip().replace('\\','/')
  if not val:return ''
  b=val.rsplit('/',1)[-1].casefold()
  if stem and '.' in b:b=b.rsplit('.',1)[0]
  return b


def try_exact_r2_roi_crosswalk(r2_rows,index_rows):
  # Test only exact *unique* structural keys, no fuzzy patient or ROI matching.
  # A source-image+box match can prove per-ROI lineage. Patient counts alone cannot.
  info={'status':'UNPROVEN_NO_EXACT_ROI_KEY_BRIDGE','successful_key':None,
        'candidate_results':[],'no_fuzzy_matching':True}
  for sourceR,sourceP in [('source_roi_png','png_path'),('image_name','png_path'),
                          ('source_roi_png','roi_path'),('IPNG_path','png_path')]:
   for is_stem in [False,True]:
    mode=('stem' if is_stem else 'basename')
    def k_r(row):
     pid=normalize_pid(row.get('patient_id',''))
     bb=bbox_from(row,('x0','y0','x1','y1'))
     name=filename_part(row.get(sourceR),is_stem)
     return (pid,name,bb) if pid and bb and name else None
    def k_p(row):
     pid=normalize_pid(row.get('patient',''))
     bb=bbox_from(row,('xmin','ymin','xmax','ymax'))
     name=filename_part(row.get(sourceP),is_stem)
     return (pid,name,bb) if pid and bb and name else None
    a=[k_r(r) for r in r2_rows];b=[k_p(r) for r in index_rows]
    rec={'r2':sourceR,'p4c':sourceP,'mode':mode,'n_unique_r2_keys':len(set(a)),
         'n_unique_p4c_keys':len(set(b)),
         'n_identical_unique_keys':len(set(a)&set(b)),
         'all_rows_nonempty':None not in a and None not in b}
    verified=(len(a)==2978 and len(b)==2978 and None not in a and None not in b
              and len(set(a))==2978 and len(set(b))==2978 and set(a)==set(b))
    rec['exact_2978_one_to_one_match']=verified
    info['candidate_results'].append(rec)
    if verified and info['successful_key'] is None:
     info['successful_key']={'r2':sourceR,'p4c':sourceP,'mode':mode}
  if info['successful_key'] is not None: info['status']='EXACT_2978_ROI_IMAGE_BBOX_CROSSWALK'
  return info


def check_source_sr1_cohort(p4c_result_path,sr1_result_path,manifest_audit):
  result={'status':'NOT_CHECKED','privacy':'only aggregates and verification statuses exported',
          'no_patient_ids_exported':True}
  if manifest_audit.get('status')!='EXACT_PUBLIC_SOURCE_PATIENT_SPLIT_374_94_AND_2398_580':
   result['status']='SKIP_R2_SOURCE_PATIENT_SPLIT_NOT_VERIFIED';return result
  try:
   # Member checksums independently retained from the exact-SHA archive inventory.
   members_p4c={'roi_source40_index.csv':'4254cb0e46e14311fc2a418a8382ffee91d5e0ca785e77ec60b49d08e9b2e4ae',
                '01_OOF_PREDICTIONS.csv':'610aef7113fb921ab071c84c9823e85b9b3c56af389cc6672d112c975dc58803'}
   members_sr1={'01_DEVELOPMENT_OOF.csv':'97e9cd9ea539adaad49df9661472162ad24eab5ef4ef6f823f5cc3f9fd4594df',
                '02_LOCKED_TEST_PREDICTIONS.csv':'4e9353bf8092818329a1d70e12c8cc8833d80b4bd1add05ee968a8ce460f4d79'}
   with zipfile.ZipFile(p4c_result_path) as z:
    for n,h in members_p4c.items():
     if sha_bytes(z.read(n))!=h: raise ValueError('P4C_MEMBER_SHA_MISMATCH')
    index=read_csv_zip(z,'roi_source40_index.csv',{'foramen_id','patient','grade'})
    common=read_csv_zip(z,'01_OOF_PREDICTIONS.csv',{'foramen_id','patient','y_true','outer_fold'})
   with zipfile.ZipFile(sr1_result_path) as z:
    for n,h in members_sr1.items():
     if sha_bytes(z.read(n))!=h: raise ValueError('SR1_MEMBER_SHA_MISMATCH')
    dev=read_csv_zip(z,'01_DEVELOPMENT_OOF.csv',{'foramen_id','patient','y_true'})
    test=read_csv_zip(z,'02_LOCKED_TEST_PREDICTIONS.csv',{'foramen_id','patient','y_true'})
   i,ic,ig,ip=grade_rows(index,'patient','grade','foramen_id')
   c,cc,cg,cp=grade_rows(common,'patient','y_true','foramen_id')
   d,dc,dg,dp=grade_rows(dev,'patient','y_true','foramen_id')
   t,tc,tg,tp=grade_rows(test,'patient','y_true','foramen_id')
   master_count=manifest_audit['_private_r2_counts']
   expected={'0':2009,'1':507,'2':254,'3':208}
   test_ids=set(PUBLIC_TEST_IDS)
   fold_map={};fold_values=set();fold_valid=True
   for row in common:
    fid=str(row['foramen_id']).strip();f=str(row['outer_fold']).strip()
    if f not in {'0','1','2','3','4'}:fold_valid=False
    fold_values.add(f)
    pid=normalize_pid(row['patient'])
    if pid in fold_map and fold_map[pid]!=f:fold_valid=False
    fold_map[pid]=f
   result.update({
    'r2_patient_rows_exactly_match_p4c_by_patient':master_count==ic,
    'index_rows':len(index),'index_patients':len(ip),
    'common_oof_rows':len(common),'common_oof_patients':len(cp),
    'common_vs_source_index_exact_foramen_id_patient_grade':c==i,
    'common_folds_exact_patient_disjoint':fold_valid and fold_values=={'0','1','2','3','4'},
    'source_index_global_four_grade_counts':{str(k):ig[k] for k in range(4)},
    'source_index_grade_counts_authority_match':{str(k):ig[k] for k in range(4)}==expected,
    'sr1_dev_rows':len(dev),'sr1_dev_patients':len(dp),
    'sr1_test_rows':len(test),'sr1_test_patients':len(tp),
    'sr1_dev_test_patient_disjoint':not bool(dp&tp),
    'sr1_test_exact_public_patient_membership':tp==test_ids,
    'sr1_dev_exact_master_non_test_membership':dp==set(master_count)-test_ids,
    'sr1_test_exact_foramen_ids_patient_and_grade':all(i.get(fid)==pair for fid,pair in t.items()) and len(t)==580,
    'sr1_dev_exact_foramen_ids_patient_and_grade':all(i.get(fid)==pair for fid,pair in d.items()) and len(d)==2398,
    'sr1_dev_test_union_equals_common_and_index':set(d).isdisjoint(t) and set(d)|set(t)==set(i),
    'sr1_test_grade_counts':{str(k):tg[k] for k in range(4)},
    'sr1_dev_grade_counts':{str(k):dg[k] for k in range(4)},
    'sr1_per_patient_grade_counts_exactly_equal_p4c':
      Counter((p,g) for p,g in d.values())+Counter((p,g) for p,g in t.values())==
      Counter((p,g) for p,g in i.values())
   })
   must_all=('r2_patient_rows_exactly_match_p4c_by_patient',
             'common_vs_source_index_exact_foramen_id_patient_grade',
             'common_folds_exact_patient_disjoint',
             'source_index_grade_counts_authority_match',
             'sr1_dev_test_patient_disjoint','sr1_test_exact_public_patient_membership',
             'sr1_dev_exact_master_non_test_membership',
             'sr1_test_exact_foramen_ids_patient_and_grade',
             'sr1_dev_exact_foramen_ids_patient_and_grade',
             'sr1_dev_test_union_equals_common_and_index',
             'sr1_per_patient_grade_counts_exactly_equal_p4c')
   passed=(len(index)==len(common)==2978 and len(ip)==len(cp)==468 and
           len(dev)==2398 and len(dp)==374 and len(test)==580 and len(tp)==94 and
           all(result[k] is True for k in must_all))
   roi=try_exact_r2_roi_crosswalk(manifest_audit['_private_r2_rows'],index)
   result['r2_p4c_image_roi_bridge']=roi
   if passed:
    result['status']='EXACT_SOURCE_374_94_PATIENT_AND_GRADE_IDENTITIES_P4C_SR1'
   else:result['status']='SOURCE_ARCHIVE_OR_R2_COHORT_MISMATCH'
   print('PATIENT_SOURCE_SR1_CROSSCHECK|'+result['status'],flush=True)
   print('R2_P4C_ROW_LEVEL_ROI_CROSSWALK|'+roi['status'],flush=True)
  except Exception as e:
   result['status']='ERROR_SOURCE_COHORT_READ'
   result['error_type']=type(e).__name__
   print('PATIENT_SOURCE_SR1_CROSSCHECK|ERROR_'+type(e).__name__,flush=True)
  return result

def discover_sources():
 # All recovered files must have *one of the known historical SHA256 values*;
 # no unspecified source is exported. Search constrained to known project directories.
 candidates=set()
 for name,(_digest,direct) in SOURCE_EXPECTED.items():
  if direct is not None:candidates.add(direct)
 # Historical run scopes and bounded package scopes, no global /data recursive walking.
 dlist=[
  ROOT/'server_runs/paper2_p4_1_sr1_source_protocol_reconstruction_20260919',
  ROOT/'server_runs/paper2_p4_1c_source_reconstructed_resnet18_corn_formal_20260918',
  ROOT/'server_runs/secondgen_step04a_firstgenbest_coralmsaux_fixeddeit_iq8_nohe_g1_20260821',
  ROOT/'server_runs/secondgen_step04a_firstgenbest_control_20260821',
  ROOT/'server_runs/paper2_p10_c3_r4_innerval_alpha_search_20261005',
  ROOT/'server_runs/paper2_p10_c3_r5_selected_alpha_outer_20261005',
  ROOT/'server_runs/paper2_p10_c3_r6_r6_selected_weight_outer_oof_eval_20261005',
  ROOT/'server_packages/secondgen_step04a_firstgenbest_iq8_nohe_g1_20260821',
  SCRIPT_ROOT,
 ]
 for d in dlist:
  if not d.is_dir():continue
  # Never scan patient data: only actual source code .py files.
  for i,p in enumerate(d.rglob('*.py')):
   if i>=600:break
   candidates.add(p)
 pkg=ROOT/'server_packages'
 if pkg.is_dir():
  for globpat in ('paper2_p4*','paper2_p10_c3_r4*','paper2_p10_c3_r5*','paper2_p10_c3_r6*'):
   for d in list(pkg.glob(globpat))[:30]:
    if d.is_dir():
     for i,p in enumerate(d.rglob('*.py')):
      if i>=150:break
      candidates.add(p)
 matched={k:[] for k in SOURCE_EXPECTED}
 sha_to_name={v[0]:k for k,v in SOURCE_EXPECTED.items()}
 inspected=0
 for p in sorted(candidates):
  if not p.is_file():continue
  try:
   if p.stat().st_size>600000:continue
   h=sha(p);inspected+=1
   if h in sha_to_name:matched[sha_to_name[h]].append(str(p))
  except (OSError,PermissionError):pass
 return matched,inspected

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
 out=pathlib.Path(args.output)
 if not re.fullmatch(r'/PATH/TO/TEMP/lss_w24p5r2_[0-9a-f]{10}/r11_preflight_evidence\.zip',str(out)):
  raise SystemExit('NO_WRITES_OUTSIDE_TMP_STAGE')
 out.parent.mkdir(parents=True,exist_ok=True)
 status={k:fail_or_pass_asset(k,*v) for k,v in ASSETS.items()}
 all_real=all(status[k]['status']=='PASS_EXACT_SHA' for k in
              ['P4_1C_COMMON_RESULT','P4_1_SR1_374_94_RESULT','R2_468_PATIENT_AUTHORITATIVE_MANIFEST','R11_R6_R6_FROZEN_FINAL','R09_R4_ALPHA_SELECTION'])
 inventories={}
 for name in ('P4_1C_COMMON_RESULT','P4_1_SR1_374_94_RESULT'):
  if status[name]['status']=='PASS_EXACT_SHA':
   inventories[name]=read_zip_inventory(ASSETS[name][0])
 print('HISTORICAL_ARCHIVE_MEMBER_INVENTORIES='+str({k:len(v) for k,v in inventories.items()}),flush=True)
 split={'status':'SKIP_R2_MANIFEST_SHA_NOT_VERIFIED'}
 if status['R2_468_PATIENT_AUTHORITATIVE_MANIFEST']['status']=='PASS_EXACT_SHA':
  split=inspect_manifest(ASSETS['R2_468_PATIENT_AUTHORITATIVE_MANIFEST'][0])
 sr1_check={'status':'SKIP_HISTORICAL_SR1_ARCHIVE_SHA_NOT_VERIFIED'}
 if (status['P4_1_SR1_374_94_RESULT']['status']=='PASS_EXACT_SHA' and
     status['P4_1C_COMMON_RESULT']['status']=='PASS_EXACT_SHA'):
  sr1_check=check_source_sr1_cohort(ASSETS['P4_1C_COMMON_RESULT'][0],ASSETS['P4_1_SR1_374_94_RESULT'][0],split)
 # Never serialize R2 patient identifiers, raw file paths or any patient-level rows.
 split.pop('_private_r2_rows',None)
 split.pop('_private_r2_counts',None)
 found,count=discover_sources()
 src={}
 for name,(expect,path) in SOURCE_EXPECTED.items():
  src[name]={'expected_sha256':expect,'status':'PASS_EXACT_SHA' if found[name] else 'NOT_FOUND',
             'matching_server_paths':found[name]}
  print('SOURCE_RECOVERED|'+name+'|'+src[name]['status'],flush=True)
 ready=(all_real and split['status']=='EXACT_PUBLIC_SOURCE_PATIENT_SPLIT_374_94_AND_2398_580' and
       sr1_check['status']=='EXACT_SOURCE_374_94_PATIENT_AND_GRADE_IDENTITIES_P4C_SR1' and all(
   src[k]['status']=='PASS_EXACT_SHA' for k in ('R09_CORE_54','DEIT_CORE_52','COMMON_04A','R09_TRAINER')))
 output={'schema':SCHEMA,'verified_assets':status,'phadke_archive_inventories':inventories,
         'public_split_test_patient_count':EXPECTED_PUBLIC_TEST_COUNT,'split_audit':split,
         'source_sr1_cohort_crosscheck':sr1_check,
         'historical_source_recovery':src,'source_files_examined':count,
         'ready_for_design_and_implementation':ready,
         'must_retrain':'Both R09-CORAL and DeiT using development-only, all alpha/weight choices development-only',
         'interpretation':'Retrospective matched 374/94 holdout, NOT new untouched independent external test',
         'science_files_mutated':False,'training':False,'inference':False,'bootstrap':False,'tuning':False,
         'patient_level_rows_exported':False,'patient_ids_exported':False,
          'r2_p4c_per_roi_identity_proven':sr1_check.get('r2_p4c_image_roi_bridge',{}).get('status')=='EXACT_2978_ROI_IMAGE_BBOX_CROSSWALK',
          'ready_to_start_training':False,
         'source_protocol_claim':'source protocol not confirmed solely from archive identity; training readiness requires independent implementation checks'}
 raw=json.dumps(output,ensure_ascii=False,sort_keys=True,indent=2).encode('utf-8')
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
  z.writestr('00_R11_374_94_READONLY_PREFLIGHT.json',raw)
  z.writestr('01_EVIDENCE_SHA256.json',json.dumps({'00_R11_374_94_READONLY_PREFLIGHT.json':sha_bytes(raw)},indent=2))
 print('W24P5_REMOTE_PREFLIGHT=COMPLETE',flush=True)
 print('R11_374_94_COHORT_EVIDENCE_READY='+str(ready),flush=True)
 print('READY_TO_TRAIN_R11_374_94=False',flush=True)
 print('NO_TRAINING_NO_INFERENCE_NO_BOOTSTRAP_NO_PATIENT_EXPORT=PASS',flush=True)
 print('REMOTE_RESULT_SHA256='+sha(out),flush=True)
 return 0
if __name__=='__main__':sys.exit(main())
