#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P9P1 SR1-matched 4-train/1-validation source-role data and code gate.  No model imports or test inference.

Derived from the exact SHA-locked P5-R2 patient/foramen/bbox audit.  All
patient-level manifests remain on the user's research server; only aggregate
pass records may be copied to Windows.  Every input is closed by SHA256.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile
from collections import Counter

HERE = Path(__file__).resolve().parent
ROOT = Path('/PATH/TO/RESEARCH_WORKSPACE')
P4C = ROOT/'server_runs/paper2_p4_1c_source_reconstructed_resnet18_corn_formal_20260918/PAPER2_P4_1C_SOURCE_RECONSTRUCTED_RESNET18_CORN_FORMAL_RESULT.zip'
SR1 = ROOT/'server_runs/paper2_p4_1_sr1_source_protocol_reconstruction_20260919/PAPER2_P4_1_SR1_SOURCE_PROTOCOL_RECONSTRUCTION_RESULT.zip'
R2 = ROOT/'server_runs/secondgen_step01f_r2_patient_split_403link_20260820/01_FROZEN_SOURCE_MANIFEST_WITH_PATIENT_SPLIT.csv'
EXPECTED = {
 'P4C': '4089ed9d6e446abd76b16433f54a41a38c263a612bacefdee250d8f61e21804f',
 'SR1': '28cbc2e75dbbaacd92c7a87691c480bc522d218cb30ec99002ab4513603401a7',
 'R2': '39371acae04df30197c19be73a2a3441379f4596f494a6651d93255ad7c8d907',
}
SOURCES={
 'COMMON_04A.py':'6032f392ea409a116d1ef5619c6b33f9fc37bc297786f0a4a58d70311a7b44c6',
 'R09_CORE_54.py':'b8e5360657fa467e74db846c4d03e7f5f80901aa2108e66587b04a58e89cb259',
 'R09_TRAINER.py':'2529afd31e91dcc7327381de640651605c25a2eafc79cc846cd5dc6763f2c3bf',
 'DEIT_CORE_52.py':'dc8c0b4b1aafe640d7ce764c3efff984e5154430621c946ea7025c0b4d3887ae',
 'ORNET_ALPHA0125.py':'f0f7aa3fdfdf00fa1b6ced5eb8d00bec478399212fdcaff7a81252e82c3283ca',
 'PARENT_P5R2_SPLIT_AUDITOR.py':'a3a6d4b638f3b992b469c8fdf28fb4ccf02e6d068ffc63a8e5088adbdf054631',
 'PHADKE_SR1_TRAINER.py':'9ff0d2a05d49ab884d9381ca396127f28d9759c02916eb8515af004f8cf47cae',
 'PHADKE_SR1_AGGREGATOR.py':'b97c2aaa45805cb80990b1e4efa38a8915a2a12a2fadd33c8600aeb0fea2a530',
}

def req(ok,msg):
 if not ok: raise RuntimeError(msg)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def shagate(p,expected,label):
 req(p.is_file(),label+'_MISSING')
 got=sha(p);req(got==expected,label+'_SHA_MISMATCH:'+got)
 print('SHA_GATE|'+label+'|PASS',flush=True)

def atomic_json(p,obj):
 b=(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode('utf8')
 tmp=p.with_name(p.name+'.part'); tmp.write_bytes(b);tmp.replace(p)

def source_index_key(row,previous=False):
 # This is exactly the key that passed the actual 2978/2978 P5-R2 check:
 # (normalized patient ID, casefolded PNG basename, decimal bbox).
 if previous:
  pid=parent.normalize_pid(row.get('patient',''))
  name=parent.filename_part(row.get('png_path',''),False)
  bbox=parent.bbox_from(row,('xmin','ymin','xmax','ymax'))
 else:
  pid=parent.normalize_pid(row.get('patient_id',''))
  name=parent.filename_part(row.get('image_name',''),False)
  bbox=parent.bbox_from(row,('x0','y0','x1','y1'))
 req(pid and name and bbox, 'ROI_IDENTITY_EMPTY')
 return (pid,name,bbox)

ap=argparse.ArgumentParser()
ap.add_argument('--run-dir',required=True)
a=ap.parse_args()
RUN=Path(a.run_dir).resolve()
req(str(RUN).startswith(str(ROOT/'server_runs/paper2_w24p9p1_sr1_roles_devonly_20261012_')),'RUN_DIR_NOT_W24P9P1_ISOLATED')
req(RUN.is_dir() and (RUN/'source').is_dir(),'RUN_DIR_NOT_MATERIALIZED')
# Never overwrite a completed prepared dev manifest.  New run dir required.
req(not(RUN/'00_PREPARE_PASS.json').exists(),'P9P1_PREPARE_ALREADY_FROZEN')
for k,expected in SOURCES.items():shagate(RUN/'source'/k,expected,'SOURCE_'+k)
for label,path in [('P4C',P4C),('SR1',SR1),('R2',R2)]:shagate(path,EXPECTED[label],label+'_ARCHIVE')

spec=importlib.util.spec_from_file_location('locked_parent_p5r2',str(RUN/'source/PARENT_P5R2_SPLIT_AUDITOR.py'))
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
# Repeat exact P5R2 authority & triple mapping, no changed/easier gates.
m=parent.inspect_manifest(R2)
req(m['status']=='EXACT_PUBLIC_SOURCE_PATIENT_SPLIT_374_94_AND_2398_580','P5R2_PATIENT_GATE_FAILED')
c=parent.check_source_sr1_cohort(P4C,SR1,m)
req(c['status']=='EXACT_SOURCE_374_94_PATIENT_AND_GRADE_IDENTITIES_P4C_SR1','P5R2_GRADE_GATE_FAILED')
req(c['r2_p4c_image_roi_bridge']['status']=='EXACT_2978_ROI_IMAGE_BBOX_CROSSWALK','P5R2_ROI_GATE_FAILED')

with zipfile.ZipFile(P4C) as z:
 index=parent.read_csv_zip(z,'roi_source40_index.csv',{'foramen_id','patient','grade'})
with zipfile.ZipFile(SR1) as z:
 dev=parent.read_csv_zip(z,'01_DEVELOPMENT_OOF.csv',{'foramen_id','patient','y_true','source_fold'})
 test=parent.read_csv_zip(z,'02_LOCKED_TEST_PREDICTIONS.csv',{'foramen_id','patient','y_true'})
# The locked test is touched HERE for identity audit, but its rows and labels
# MUST NOT be serialized to any training-readable artifact, nor read by training.
req(len(test)==580 and len(dev)==2398,'SR1_DEV_TEST_SIZE_MISMATCH')
test_set={str(r['foramen_id']).strip() for r in test}
req(len(test_set)==580,'SR1_TEST_FID_DUPLICATE')
index_lookup={source_index_key(r,True):r for r in index}
req(len(index_lookup)==2978,'P4C_SOURCE_INDEX_KEY_NOT_UNIQUE')
fold_by_fid={}
for r in dev:
 fid=str(r['foramen_id']).strip();v=str(r['source_fold']).strip()
 req(v in {'0','1','2','3','4'},'SR1_SOURCE_FOLD_NOT_0_4:'+v)
 req(fid not in fold_by_fid,'SR1_DEV_FOLD_FID_DUPLICATE')
 fold_by_fid[fid]=int(v)
req(len(fold_by_fid)==2398 and not(set(fold_by_fid)&test_set),'SR1_DEV_FOLD_LEAK')

raw=m['_private_r2_rows']
records=[];roi_ids=[];n_test=0;pt_fold={};label_counts=Counter();img_sources=Counter()
for r in raw:
 k=source_index_key(r)
 req(k in index_lookup,'R2_P4C_EXACT_KEY_MISSING')
 item=index_lookup[k]
 fid=str(item['foramen_id']).strip();pid=k[0]
 req(fid not in roi_ids,'ROW_ID_DUPLICATE')
 roi_ids.append(fid)
 if fid in test_set:
  n_test+=1
  continue  # HARD STOP: do not persist test rows or paths
 req(fid in fold_by_fid,'NONTEST_FID_MISSING_SOURCE_FOLD')
 f=fold_by_fid[fid]
 if pid in pt_fold:req(pt_fold[pid]==f,'PATIENT_ACROSS_DEV_FOLDS')
 pt_fold[pid]=f
 grade=parent.parse_grade(item['grade']);req(grade is not None,'GRADE_INVALID')
 label_counts[grade]+=1
 iq=str(r.get('IQ8_path','')).strip();req(iq,'R2_IQ8_MISSING')
 # Source original DeiT and original compact CNN both use the ROI crop files.
 # For DeiT select exact original source ROI, not P4C 40% expanded image.
 deit_path=str(r.get('source_roi_png','')).strip();req(deit_path,'R2_SOURCE_ROI_PNG_MISSING')
 # No new re-extraction, no fallback to P4C ROI or source metadata.
 for name,path in [('IQ8',iq),('SOURCE_ROI',deit_path)]:
  if not Path(path).is_file():
   raise RuntimeError(f'DEV_INPUT_FILE_UNAVAILABLE|{name}|foramen_sha={hashlib.sha256(fid.encode()).hexdigest()[:12]}')
 img_sources['IQ8']+=1;img_sources['SOURCE_ROI']+=1
 records.append({'foramen_id':fid,'patient_id':pid,'severity':grade,'iq8_path':iq,
                 'roi_path':deit_path,'dev_fold':f})
req(n_test==580 and len(records)==2398,'DEV_TEST_DISCARD_COUNT_FAIL')
req(len(pt_fold)==374 and len(set(roi_ids))==2978,'P9P1_FOLD_OR_ROW_GATE_FAIL')
req(dict(label_counts)=={0:1631,1:403,2:201,3:163},'DEVELOPMENT_GRADE_COUNTS_FAIL')
req(set(pt_fold.values())==set(range(5)),'SOURCE_DEV_FIVEFOLD_GATE_FAIL')
counts={str(f):{'patients':sum(1 for x in pt_fold.values() if x==f),
                'rois':sum(1 for x in records if x['dev_fold']==f)} for f in range(5)}
req(all(v['patients']>=50 and v['rois']>=350 for v in counts.values()),'DEV_FOLD_DEGENERATE')

# Authenticated prior W24-P9P0-R1 role audit, exact bytes from user-verified result.
ROLE_AUDIT_SHA = '6851a9df17efa0a1ded8ac9db0715bdc511471a45507597e3e5b7ebf93487018'
role_path=RUN/'03_PARENT_P9P0R1_ROLE_AUDIT.json'
shagate(role_path,ROLE_AUDIT_SHA,'AUTHENTIC_P9P0R1_ROLE_AUDIT')
role=json.loads(role_path.read_text(encoding='utf8'))
req(role['schema']=='LSS_PAPER2_W24P9P0_SINGLEFOLD_ROLE_AUDIT_V1','ROLE_AUDIT_SCHEMA')
req(role['historical_sr1_fold_training_summary_exactly_verified'] is True and
    role['sr1_fold_by_fold_probability_evidence_complete'] is True and
    role['new_r11_proposed_fold_roles_equal_sr1'] is True,'SR1_AUDIT_NOT_STRICT')
req(len(role['folds'])==5,'SR1_ROLE_COUNT_INVALID')
role_counts={}
for a in role['folds']:
 f=int(a['fold'])
 req(f in range(5) and f not in role_counts,'SR1_FOLD_ID_INVALID')
 val= [r for r in records if r['dev_fold']==f]
 train=[r for r in records if r['dev_fold']!=f]
 vp={r['patient_id'] for r in val};tp={r['patient_id'] for r in train}
 req(vp.isdisjoint(tp),'SR1_ROLE_PATIENT_LEAK')
 req(len(val)==int(a['sr1_val_rois']) and len(vp)==int(a['sr1_val_patients']),'SR1_VAL_FOLD_COUNT_MISMATCH')
 req(len(train)==int(a['sr1_train_rois']) and len(tp)==int(a['sr1_train_patients']),'SR1_TRAIN_FOLD_COUNT_MISMATCH')
 req(int(a['sr1_val_source_fold'])==f and
     list(a['sr1_train_source_folds'])==[j for j in range(5) if j!=f], 'SR1_ROLE_MAPPING_MISMATCH')
 role_counts[str(f)]={'train_patients':len(tp),'train_rois':len(train),
                      'val_patients':len(vp),'val_rois':len(val),
                      'train_source_folds':[j for j in range(5) if j!=f],
                      'val_source_fold':f}
req(set(role_counts)==set(map(str,range(5))),'SR1_ROLE_FIVEFOLD_INCOMPLETE')
print('W24P9P1_SOURCE_SINGLEFOLD_ROLE_IDENTITY=PASS_EXACT_5_OF_5',flush=True)

out=RUN/'_PRIVATE_DEVONLY_2398.csv'
with out.open('x',newline='',encoding='utf8') as fi:
 w=csv.DictWriter(fi,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
os.chmod(out,0o600)
contract={'schema':'LSS_PAPER2_W24P9P1_SR1_MATCHED_DEV_INPUT_FREEZE_V1',
          'source_protocol':'SR1 matching 4 train source folds + 1 identical validation fold, per fold, same 94/580 only after freeze',
          'dev_patients':len(pt_fold),'dev_rows':len(records),
          'test_patients':94,'test_rows':n_test,
          'dev_grade_counts':{str(k):label_counts[k] for k in range(4)},
          'dev_source_fold_distribution':counts,'development_source_roi_paths_present':img_sources['SOURCE_ROI'],
          'development_iq8_paths_present':img_sources['IQ8'],
          'private_dev_manifest_sha256':sha(out),
          'authentic_p9p0r1_role_audit_sha256':ROLE_AUDIT_SHA,
          'source_singlefold_train_val_roles':role_counts,
          'historical_input_sha256':EXPECTED,
          'historical_source_sha256':SOURCES,
          'test_rows_written_to_dev_manifest':False,
          'source_split_p5r2_gates':'EXACT_374_94_PATIENT_GRADE_AND_2978_ROI',
          'locking':'94-person target not read by training runner; test inference separate; validation used for checkpoint alpha and weight choice, not independent development OOF',
          'warning':'Retrospective same dataset holdout, already accessed in prior P8; not independent; no fresh blind test',
          'train_started':False,'test_inference_started':False,'bootstrap_started':False}
atomic_json(RUN/'00_PREPARE_PASS.json',contract)
print('W24P9P1_DATA_SPLIT=PASS_EXACT_DEV374_TEST94',flush=True)
print('W24P9P1_DEV_IMAGES_PRESENT=2398/2398 IQ8+SOURCE_ROI',flush=True)
print('W24P9P1_DEVONLY_INPUT_SHA256='+contract['private_dev_manifest_sha256'],flush=True)
print('W24P9P1_PREPARE_DEVONLY=PASS',flush=True)
