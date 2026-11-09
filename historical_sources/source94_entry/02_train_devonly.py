#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W24-P9P1 SR1-matched 4-train / 1-validation R11: DEVELOPMENT-ONLY training.

15 original-style R09 candidate fits, 5 original-style DeiT fits.
For each source validation fold f, train on the same other four source folds
as the authenticated Phadke SR1 trainer. Select checkpoint, alpha, fusion
weight using validation fold f; DO NOT claim independent development OOF.
94-person source test is NOT read or predicted here. Its one-time inference
requires a distinct frozen-evaluation package.

Scientific ancestry (exact source files under source/; original bytes protected):
 - 03_train_coral_progression_aux_fold.py (R09_TRAINER)
 - 54_train_coral_risk_protocol_v3_full468_oof.py (R09_CORE_54)
 - common_step04a.py (COMMON_04A)
 - ORNet alpha .125 exact source implementation (ORNET_ALPHA0125)
 - 52_train_official_deit_protocol_v3_full468_oof.py (DEIT_CORE_52)
 - R09 alpha selection and R11 fusion weight lexicographic criteria.
Important planned deviations: SR1-exact 4-train/1-validation role mapping and
fresh per-fold dev-only training; 94 person test excluded entirely.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
os.environ['HF_HUB_OFFLINE']='1'  # REQUIRE cached pretrained weights; no network download
os.environ['TRANSFORMERS_OFFLINE']='1'

HERE=Path(__file__).resolve().parent
ALPHAS=(0.10,0.125,0.15)
INITIAL_WEIGHTS=(0.30,0.40,0.50,0.60,0.70)
EXTENDED_WEIGHTS=(0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00)

# NumPy, scipy, and PyTorch must come from the historical GPU environment.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,cohen_kappa_score,f1_score,mean_absolute_error


def req(ok,msg):
 if not ok:raise RuntimeError(msg)

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def write_json(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.temp')
 t.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,default=str)+'\n',encoding='utf8');t.replace(p)

def module(filename, name):
 path=HERE/'source'/filename
 s=importlib.util.spec_from_file_location(name,str(path))
 req(s is not None and s.loader is not None,'MODULE_SPEC_FAIL:'+name)
 m=importlib.util.module_from_spec(s)
 s.loader.exec_module(m)
 return m


def get_modules():
 source54=module('R09_CORE_54.py','w24p7_r09_original')
 common=module('COMMON_04A.py','w24p7_common_original')
 prog=module('ORNET_ALPHA0125.py','w24p7_ornet_original')
 deit=module('DEIT_CORE_52.py','w24p7_deit_original')
 return source54,common,prog,deit


def metrics(y,p):
 y=np.asarray(y,int);p=np.asarray(p,int)
 return {'accuracy':float(accuracy_score(y,p)),
         'balanced_accuracy':float(balanced_accuracy_score(y,p)),
         'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
         'weighted_f1':float(f1_score(y,p,average='weighted',zero_division=0)),
         'qwk':float(cohen_kappa_score(y,p,weights='quadratic',labels=[0,1,2,3])),
         'mae':float(mean_absolute_error(y,p)),
         'g1_to0':float(((y==1)&(p==0)).sum()/max(1,(y==1).sum()))}


def freeze_preflight(run,modules,probe_pretrain=False):
 import timm
 source54,common,prog,deit=modules
 p=run/'00_PREPARE_PASS.json'
 req(p.is_file(),'PREPARE_GATE_MISSING')
 contract=json.loads(p.read_text(encoding='utf8'))
 req(contract['schema']=='LSS_PAPER2_W24P9P1_SR1_MATCHED_DEV_INPUT_FREEZE_V1','PREPARE_SCHEMA')
 req(contract['dev_patients']==374 and contract['dev_rows']==2398,'PREPARE_COHORT_COUNT')
 req(contract['test_rows_written_to_dev_manifest'] is False,'TEST_COUPLING_GATE')
 manifest=run/'_PRIVATE_DEVONLY_2398.csv'
 req(sha(manifest)==contract['private_dev_manifest_sha256'],'PRIVATE_DEV_SHA_CHANGED')
 for k,expect in contract['historical_source_sha256'].items():
  req(sha(run/'source'/k)==expect,'HISTORICAL_SOURCE_SHA_CHANGED:'+k)
 df=pd.read_csv(manifest,encoding='utf8',dtype={'foramen_id':str,'patient_id':str})
 req(len(df)==2398 and df.foramen_id.nunique()==2398 and df.patient_id.nunique()==374,'DEV_FILE_IDENTITY_MISMATCH')
 req(set(df['dev_fold'])==set(range(5)),'DEV_FOLD_DOMAIN_FAIL')
 roles=contract['source_singlefold_train_val_roles']
 for f in range(5):
  part=df[df.dev_fold==f]; other=df[df.dev_fold!=f]
  vr=roles[str(f)]
  req(len(part)==vr['val_rois'] and part.patient_id.nunique()==vr['val_patients'],f'REAL_VAL_ROLE_{f}_COUNT_MISMATCH')
  req(len(other)==vr['train_rois'] and other.patient_id.nunique()==vr['train_patients'],f'REAL_TRAIN_ROLE_{f}_COUNT_MISMATCH')
  req(vr['val_source_fold']==f and vr['train_source_folds']==[i for i in range(5) if i!=f],f'SR1_ROLE_MAPPING_{f}_MISMATCH')
 req(sha(run/'03_PARENT_P9P0R1_ROLE_AUDIT.json')==contract['authentic_p9p0r1_role_audit_sha256'],'P9_ROLE_AUDIT_DRIFT')
 req(df.groupby('patient_id')['dev_fold'].nunique().max()==1,'DEV_FOLD_PATIENT_LEAK')
 req(all(df['severity'].isin([0,1,2,3])),'DEV_LABEL_INVALID')
 expected=contract['dev_grade_counts']
 req({str(k):int(v) for k,v in df.severity.value_counts().sort_index().items()}==expected,'DEV_LABEL_COUNTS_FAIL')
 req(all(df['roi_path'].map(lambda p:Path(p).is_file())) and
     all(df['iq8_path'].map(lambda p:Path(p).is_file())),'DEV_IMAGES_MISSING')
 req(torch.cuda.is_available(),'CUDA_REQUIRED_FOR_DEV_TRAIN')
 req(torch.cuda.device_count()>=1,'CUDA_DEVICE_COUNT_ZERO')
 # Test the exact two read-only data pipelines and small forward/loss calculation.
 s54,comm,pg,di=modules
 one=df.iloc[:2]
 x,yy,_,_=comm.SelectedPreprocDataset(one,'coral_msaux',False,s54)[0]
 req(tuple(x.shape)==(1,64,64),'R09_SOURCE_PREPROC_SHAPE')
 deit_tf=di.build_transforms(42)[1]
 xx,dy,_=di.ROIDataset(one,'roi_path','severity',deit_tf)[0]
 req(tuple(xx.shape)==(3,224,224),'DEIT_SOURCE_PREPROC_SHAPE')
 m=s54.RiskAwareOrdinalCNN('coral',dropout=.5)
 m.progression_aux=pg.ORNetProgressionAux(4)
 r=m(x[None])[0]; q=pg.ORNetStyleFusionLoss().regloss(m.progression_aux(r),pg.progression_targets(yy[None]).float())
 req(torch.isfinite(q).item() and r.shape==(1,3),'R09_AUX_FORWARD_FAIL')
 del m
 # The original model MUST be ImageNet pretrained, not randomly initialized.
 # In offline mode, failure to resolve the pretrain cache blocks all training.
 if probe_pretrain:
  dt=timm.create_model('deit_tiny_patch16_224.fb_in1k',pretrained=True,num_classes=4,drop_rate=.25)
  req(dt is not None and sum(p.numel() for p in dt.parameters())>1000000,'DEIT_PRETRAIN_LOAD_FAILED')
  del dt
 print('W24P9P1_PRETRAIN_AND_GPU_GATE=PASS' if probe_pretrain else 'W24P9P1_DEV_PRECHECK=PASS',flush=True)
 return df,contract

@torch.no_grad()
def eval_coral(model,loader,device,s54):
 model.eval();ys=[];pp=[];n=0;loss=0.0
 for x,y,_,_ in loader:
  x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
  logits,_=model(x)
  obj=s54.ordinal_loss(logits,y,'coral')+0.01*s54.ordinal_regularization(model)
  count=y.numel();n+=count;loss+=float(obj.detach().cpu())*count
  _,p=s54.cumulative_and_class_probabilities(logits,'coral')
  ys.append(y.cpu().numpy());pp.append(p.cpu().numpy())
 Y=np.concatenate(ys);P=np.concatenate(pp)
 return loss/n,Y,P

@torch.no_grad()
def eval_deit(model,loader,device,crit):
 model.eval();ys=[];pp=[];loss=0.;n=0
 for x,y,_ in loader:
  x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
  logits=model(x)
  ll=crit(logits,y)
  count=y.numel();n+=count;loss+=float(ll.detach().cpu())*count
  ys.append(y.cpu().numpy());pp.append(torch.softmax(logits,1).cpu().numpy())
 Y=np.concatenate(ys);P=np.concatenate(pp)
 return loss/n,Y,P


def init_coral(s54,prog,device):
 model=s54.RiskAwareOrdinalCNN('coral',dropout=.5).to(device)
 model.progression_aux=prog.ORNetProgressionAux(4).to(device)
 for p in model.risk_head.parameters():p.requires_grad_(False)
 return model


def load_ckpt(path):
 try:return torch.load(path,map_location='cpu',weights_only=False)
 except TypeError:return torch.load(path,map_location='cpu')


def start_folder(folder,finish):
 if (folder/finish).is_file():return True
 if folder.exists():
  n=0
  while folder.with_name(folder.name+('_PARTIAL_RESIDUE' if not n else f'_PARTIAL_RESIDUE_{n}')).exists():n+=1
  folder.rename(folder.with_name(folder.name+('_PARTIAL_RESIDUE' if not n else f'_PARTIAL_RESIDUE_{n}')))
 folder.mkdir(parents=True)
 return False


def train_coral(run,df,fold,alpha,modules,device):
 s54,common,prog,_=modules
 tag=str(alpha).replace('.','p')
 fd=run/'models'/f'fold_{fold}'/f'coral_alpha_{tag}'
 if start_folder(fd,'PASS_CANDIDATE.json'):
  info=json.loads((fd/'PASS_CANDIDATE.json').read_text());req(info['checkpoint_sha256']==sha(fd/'best_checkpoint.pth'),'R09_RESUME_SHA_BAD')
  req(info['alpha']==alpha and info['validation_source_fold']==fold and info['train_rows']==int((df.dev_fold!=fold).sum()),'R09_RESUME_ROLE_BAD')
  req(info['validation_probability_sha256']==sha(fd/'inner_val_probs.npz'),'R09_RESUME_VALIDATION_SHA_BAD')
  return info
 val_fold=fold
 tr=df.loc[df.dev_fold!=fold].reset_index(drop=True)
 va=df.loc[df.dev_fold==val_fold].reset_index(drop=True)
 req(tr.patient_id.nunique()>=299 and va.patient_id.nunique()>=74,'SR1_MATCHED_R09_ROLE_COUNTS_INVALID')
 req(set(tr.patient_id).isdisjoint(va.patient_id),'CORAL_PATIENT_LEAK')
 common.set_all_seeds(42)
 train_ds=common.SelectedPreprocDataset(tr,'coral_msaux',True,s54,42)
 val_ds=common.SelectedPreprocDataset(va,'coral_msaux',False,s54,42)
 cnt=np.bincount(tr.severity.astype(int).to_numpy(),minlength=4)
 req(np.all(cnt>0),'R09_CLASS_EMPTY')
 cw=1/cnt.astype(float)
 w=torch.as_tensor(cw[tr.severity.astype(int).to_numpy()],dtype=torch.double)
 gen=torch.Generator().manual_seed(42)
 sampler=WeightedRandomSampler(w,num_samples=len(w),replacement=True,generator=gen)
 train_loader=DataLoader(train_ds,batch_size=32,shuffle=False,sampler=sampler,num_workers=0,pin_memory=True)
 val_loader=DataLoader(val_ds,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
 model=init_coral(s54,prog,device)
 opt=torch.optim.AdamW(model.parameters(),lr=.0007723,weight_decay=.002453)
 best=math.inf;history=[];best_ep=None
 out=fd/'best_checkpoint.pth'
 print(f'R09_DEV_START|fold={fold}|alpha={alpha}|train={len(tr)}|val={len(va)}',flush=True)
 for epoch in range(1,201):
  model.train();train_loss=0.;seen=0
  for x,y,_,_ in train_loader:
   x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
   opt.zero_grad(set_to_none=True)
   logits,_=model(x)
   ord_loss=s54.ordinal_loss(logits,y,'coral')
   target=prog.progression_targets(y).type_as(logits)
   aux=prog.ORNetStyleFusionLoss().regloss(model.progression_aux(logits),target)
   joint=(1.-alpha)*ord_loss+alpha*aux+.01*(s54.ordinal_regularization(model)+model.progression_aux.reg.weight.square().sum())
   joint.backward();opt.step()
   seen+=y.numel();train_loss+=float(joint.detach().cpu())*y.numel()
  vl,Y,P=eval_coral(model,val_loader,device,s54)
  vm=metrics(Y,P.argmax(1))
  history.append({'epoch':epoch,'train_joint_loss':train_loss/seen,'val_ordinal_objective':vl,'val_qwk':vm['qwk'],'val_macro_f1':vm['macro_f1']})
  if vl<best:
   best=vl;best_ep=epoch
   torch.save({'model_state_dict':copy.deepcopy(model.state_dict()),'best_epoch':epoch,
               'best_validation_ordinal_objective':vl,'validation_source_fold':fold,
               'fixed_validation_fold':val_fold,'alpha':alpha},out)
  if epoch==1 or epoch%10==0 or epoch==200:
   print(f'R09_DEV_PROGRESS|fold={fold}|alpha={alpha}|ep={epoch}/200|val_ord={vl:.6f}|best_ep={best_ep}',flush=True)
 req(best_ep is not None,'R09_NO_BEST')
 obj=load_ckpt(out);model.load_state_dict(obj['model_state_dict'],strict=True)
 val_obj,vy,vprob=eval_coral(model,val_loader,device,s54)
 chosen=metrics(vy,vprob.argmax(1))
 pd.DataFrame(history).to_csv(fd/'training_history.csv',index=False)
 np.savez_compressed(fd/'inner_val_probs.npz',Y=vy,P=vprob)
 info={'schema':'W24P9P1_CORAL_CANDIDATE_PASS','alpha':alpha,'validation_source_fold':fold,'inner_val_fold':val_fold,
       'train_rows':len(tr),'val_rows':len(va),'best_epoch':best_ep,
       'best_val_ordinal_objective':best,'checkpoint_sha256':sha(out),'validation_probability_sha256':sha(fd/'inner_val_probs.npz'),
       'val_qwk':chosen['qwk'],'val_macro_f1':chosen['macro_f1'],
       'test_inference':False,'test_used_in_training_or_selection':False}
 write_json(fd/'PASS_CANDIDATE.json',info)
 del model,opt,train_loader,val_loader;torch.cuda.empty_cache()
 return info


def train_deit(run,df,fold,modules,device):
 *_,deit=modules
 fd=run/'models'/f'fold_{fold}'/'deit'
 if start_folder(fd,'PASS_DEIT.json'):
  info=json.loads((fd/'PASS_DEIT.json').read_text());req(info['checkpoint_sha256']==sha(fd/'best_val_loss.pth'),'DEIT_RESUME_SHA_BAD')
  req(info['validation_source_fold']==fold and info['train_rows']==int((df.dev_fold!=fold).sum()),'DEIT_RESUME_ROLE_BAD')
  req(info['validation_probability_sha256']==sha(fd/'inner_val_probs.npz'),'DEIT_RESUME_VALIDATION_SHA_BAD')
  return info
 val_fold=fold
 tr=df.loc[df.dev_fold!=fold].reset_index(drop=True)
 va=df.loc[df.dev_fold==val_fold].reset_index(drop=True)
 deit.set_seed(42)
 trtf,vatf=deit.build_transforms(42)
 train_ds=deit.ROIDataset(tr,'roi_path','severity',trtf)
 val_ds=deit.ROIDataset(va,'roi_path','severity',vatf)
 gen=torch.Generator().manual_seed(42)
 tl=DataLoader(train_ds,batch_size=16,shuffle=True,num_workers=0,pin_memory=True,generator=gen)
 vl=DataLoader(val_ds,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
 cnt=np.bincount(tr.severity.astype(int).to_numpy(),minlength=4)
 req(np.all(cnt>0),'DEIT_CLASS_EMPTY')
 weights=1/cnt.astype(float);weights=weights/weights.sum()*4.
 import timm
 model=timm.create_model('deit_tiny_patch16_224.fb_in1k',pretrained=True,num_classes=4,drop_rate=.25).to(device)
 crit=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
 opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
 scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-5,epochs=20,steps_per_epoch=len(tl),pct_start=.25,anneal_strategy='cos')
 scaler=torch.amp.GradScaler('cuda',enabled=True)
 hist=[];best=math.inf;best_ep=None
 out=fd/'best_val_loss.pth'
 print(f'DEIT_DEV_START|fold={fold}|train={len(tr)}|val={len(va)}',flush=True)
 for epoch in range(1,21):
  model.train();los=0.;n=0
  for x,y,_ in tl:
   x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
   opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',enabled=True):
    logit=model(x);loss=crit(logit,y)
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();scheduler.step()
   los+=float(loss.detach().cpu())*y.numel();n+=y.numel()
  vobj,Y,P=eval_deit(model,vl,device,crit)
  met=metrics(Y,P.argmax(1))
  hist.append({'epoch':epoch,'train_loss':los/n,'val_loss':vobj,'val_qwk':met['qwk'],'val_macro_f1':met['macro_f1']})
  if vobj<best:
   best=vobj;best_ep=epoch
   torch.save({'model_state_dict':copy.deepcopy(model.state_dict()),'epoch':epoch,'val_loss':vobj,
               'model_name':'deit_tiny_patch16_224.fb_in1k','validation_source_fold':fold,
               'inner_val_fold':val_fold},out)
  if epoch==1 or epoch%5==0 or epoch==20:
   print(f'DEIT_DEV_PROGRESS|fold={fold}|ep={epoch}/20|val_loss={vobj:.6f}|best_ep={best_ep}',flush=True)
 req(best_ep is not None,'DEIT_NO_BEST')
 model.load_state_dict(load_ckpt(out)['model_state_dict'],strict=True)
 obj,vy,vp=eval_deit(model,vl,device,crit)
 pd.DataFrame(hist).to_csv(fd/'training_history.csv',index=False)
 np.savez_compressed(fd/'inner_val_probs.npz',Y=vy,P=vp)
 meta={'schema':'W24P9P1_DEIT_PASS','validation_source_fold':fold,'inner_val_fold':val_fold,
       'train_rows':len(tr),'val_rows':len(va),'best_epoch':best_ep,
       'best_val_loss':best,'checkpoint_sha256':sha(out),'validation_probability_sha256':sha(fd/'inner_val_probs.npz'),
       'test_inference':False,'test_used_in_training_or_selection':False}
 write_json(fd/'PASS_DEIT.json',meta)
 del model,tl,vl,train_ds,val_ds;torch.cuda.empty_cache()
 return meta


def select_alpha_fold(run,fold):
 choices=[]
 for a in ALPHAS:
  tag=str(a).replace('.','p');folder=run/'models'/f'fold_{fold}'/f'coral_alpha_{tag}'
  data=json.loads((folder/'PASS_CANDIDATE.json').read_text())
  choices.append(data)
 # Exactly old tie order. Checkpoint selection has already finished.
 choices.sort(key=lambda v:(-v['val_qwk'],v['best_val_ordinal_objective'],
                            -v['val_macro_f1'],abs(v['alpha']-.125),v['alpha']))
 return choices[0],choices

def select_weight(Pc,Pd,Y,grid):
 cands=[]
 for w in grid:
  p=(w*Pc+(1.-w)*Pd).argmax(1)
  v=metrics(Y,p)
  cands.append({'coral_weight':float(w),'deit_weight':float(1.-w),**v})
 cands.sort(key=lambda x:(-x['qwk'],x['mae'],-x['balanced_accuracy'],abs(x['coral_weight']-.4),x['coral_weight']))
 return cands[0],cands


def finalize_dev(run,df,modules,device):
 """Freeze five SR1-matched fold models and validation-only choices.

 Validation fold f selects native checkpoints, alpha, and fusion weight.
 No development OOF estimate exists here; the same validation labels were
 involved in model selection, so observed validation metrics are optimistic.
 """
 candidates=[]
 for fold in range(5):
  top,all_a=select_alpha_fold(run,fold)
  va=df[df.dev_fold==fold].reset_index(drop=True)
  base=run/'models'/f'fold_{fold}'
  rpath=base/f"coral_alpha_{str(top['alpha']).replace('.','p')}"/'inner_val_probs.npz'
  dpath=base/'deit'/'inner_val_probs.npz'
  with np.load(rpath,allow_pickle=False) as x: CY=x['Y'].copy();CP=x['P'].copy()
  with np.load(dpath,allow_pickle=False) as x: DY=x['Y'].copy();DP=x['P'].copy()
  req(np.array_equal(CY,DY) and np.array_equal(CY,va.severity.astype(int).to_numpy()),'SR1_MATCHED_VALIDATION_LABEL_ORDER_FAIL')
  req(CP.shape==DP.shape==(len(va),4),'FUSION_VALIDATION_PROB_SHAPE_FAIL')
  req(np.all(np.isfinite(CP)) and np.all(np.isfinite(DP)),'FUSION_PROB_NAN')
  req(np.max(np.abs(CP.sum(1)-1))<1e-4 and np.max(np.abs(DP.sum(1)-1))<1e-4,'FUSION_PROB_NOT_NORMALIZED')
  first,table=select_weight(CP,DP,CY,INITIAL_WEIGHTS)
  candidates.append((fold,top,all_a,CY,CP,DP,first))
 boundary=sum(1 for item in candidates if abs(item[-1]['coral_weight']-.70)<1e-9)
 grid=EXTENDED_WEIGHTS if boundary>=3 else INITIAL_WEIGHTS
 results=[]; all_checkpoints={}
 for fold,top,all_a,Yv,CP,DP,first in candidates:
  sel,weighttab=select_weight(CP,DP,Yv,grid)
  base=run/'models'/f'fold_{fold}'
  coral_dir=base/f"coral_alpha_{str(top['alpha']).replace('.','p')}"
  ddir=base/'deit'
  va=df[df.dev_fold==fold].reset_index(drop=True)
  tr=df[df.dev_fold!=fold].reset_index(drop=True)
  req(len(tr)+len(va)==2398 and set(tr.patient_id).isdisjoint(set(va.patient_id)), 'SR1_FOLD_ROLE_LEAK')
  for a in ALPHAS:
   atag=str(a).replace('.','p')
   folder=base/f'coral_alpha_{atag}'
   record=json.loads((folder/'PASS_CANDIDATE.json').read_text(encoding='utf8'))
   req(record['validation_source_fold']==fold and record['train_rows']==len(tr) and record['val_rows']==len(va),'CORAL_CANDIDATE_ROLE_DRIFT')
   req(record['checkpoint_sha256']==sha(folder/'best_checkpoint.pth'),'CORAL_FROZEN_CHECKPOINT_DRIFT')
   req(record['validation_probability_sha256']==sha(folder/'inner_val_probs.npz'),'CORAL_FROZEN_VALPROB_DRIFT')
   all_checkpoints[f'CORAL_F{fold}_A{atag}']=record['checkpoint_sha256']
  drecord=json.loads((ddir/'PASS_DEIT.json').read_text(encoding='utf8'))
  req(drecord['validation_source_fold']==fold and drecord['train_rows']==len(tr) and drecord['val_rows']==len(va),'DEIT_ROLE_DRIFT')
  req(drecord['checkpoint_sha256']==sha(ddir/'best_val_loss.pth'),'DEIT_CK_DRIFT')
  req(drecord['validation_probability_sha256']==sha(ddir/'inner_val_probs.npz'),'DEIT_PROBS_DRIFT')
  all_checkpoints[f'DEIT_F{fold}']=drecord['checkpoint_sha256']
  results.append({'fold':fold,'validation_source_fold':fold,
                  'training_source_folds':[j for j in range(5) if j!=fold],
                  'train_patients':tr.patient_id.nunique(),'train_rows':len(tr),
                  'validation_patients':va.patient_id.nunique(),'validation_rows':len(va),
                  'alpha':top['alpha'],'alpha_ranking':all_a,
                  'selected_coral_checkpoint_sha256':top['checkpoint_sha256'],
                  'selected_coral_validation_probability_sha256':sha(coral_dir/'inner_val_probs.npz'),
                  'selected_weight':sel['coral_weight'],'initial_weight_winner':first['coral_weight'],
                  'fusion_weight_grid':list(grid),'fusion_weight_table':weighttab,
                  'deit_checkpoint_sha256':drecord['checkpoint_sha256'],
                  'deit_validation_probability_sha256':drecord['validation_probability_sha256'],
                  'validation_fusion_metrics_NOT_OOF':{k:sel[k] for k in ('qwk','mae','accuracy','balanced_accuracy','macro_f1','g1_to0')}})
 req(len(results)==5 and len(all_checkpoints)==20,'P9P1_FOLD_OR_CHECKPOINT_COUNT_FAIL')
 frozen={'schema':'LSS_PAPER2_W24P9P1_SR1_MATCHED_4TRAIN_1VAL_FROZEN_V1',
         'source_dev_patients':374,'source_dev_rois':2398,
         'source_locked_test_patients':94,'source_locked_test_rois':580,
         'parent_p9p0r1_role_audit_sha256':sha(run/'03_PARENT_P9P0R1_ROLE_AUDIT.json'),
         'historical_p7_input_sha256':'7de0762eb4ea7ea8f2970d6ee602f36e8bd30870316b339826b41b94b4419947',
         'alpha_grid':list(ALPHAS),'initial_weight_grid':list(INITIAL_WEIGHTS),
         'weight_grid_expansion_boundary_winners':boundary,'expanded_weight_grid':list(grid),
         'folds':results,'all_20_checkpoint_sha256':all_checkpoints,
         'devonly_input_sha256':sha(run/'_PRIVATE_DEVONLY_2398.csv'),
         'fold_predictions_after_freeze':'FIVE_FOLD_PROBABILITY_EQUAL_MEAN_WITH_FOLD_NATIVE_SELECTED_ALPHA_AND_WEIGHT',
         'default_prediction_decision_rule':'FOUR_CLASS_ARGMAX_AFTER_FIVE_FOLD_PROBABILITY_AVERAGE',
         'validation_metrics_status':'FOR_SELECTION_ONLY_NOT_INDEPENDENT_OOF',
         'test_inference_executed':False,
         'test_labels_or_probabilities_read_by_trainer':False,
         'retrospective_caveat':'94 source patients already seen by earlier research P8; cannot claim untouched validation',
         'input_and_budget_differences_from_phadke_sr1_remain':True,
         'training_complete':True,'bootstrap_executed':False}
 write_json(run/'00_MATCHED_DEVFREEZE_PASS.json',frozen)
 print('W24P9P1_DEVONLY_MATCHED_FREEZE=PASS',flush=True)
 print('W24P9P1_SELECTED_ALPHA_BY_SOURCE_FOLD='+json.dumps([r['alpha'] for r in results]),flush=True)
 print('W24P9P1_SELECTED_WEIGHT_BY_SOURCE_FOLD='+json.dumps([r['selected_weight'] for r in results]),flush=True)
 print('W24P9P1_NO_INDEPENDENT_DEV_OOF=TRUE',flush=True)
 print('LOCKED_94_TEST_OPENED_BY_TRAINER=False',flush=True)


def run_training(run,mods):
 df,config=freeze_preflight(run,mods,probe_pretrain=True)
 device=torch.device('cuda')
 for f in range(5):
  for a in ALPHAS:train_coral(run,df,f,a,mods,device)
  train_deit(run,df,f,mods,device)
  selected,_=select_alpha_fold(run,f)
  print(f'W24P9P1_SR1_FOLD_TRAIN_COMPLETE|fold={f}|alpha={selected["alpha"]}',flush=True)
 finalize_dev(run,df,mods,device)


def main():
 pa=argparse.ArgumentParser()
 pa.add_argument('--run-dir',required=True)
 pa.add_argument('--mode',choices=['check','train'],default='check')
 args=pa.parse_args()
 run=Path(args.run_dir).resolve()
 req(str(run).startswith('/PATH/TO/RESEARCH_WORKSPACE/server_runs/paper2_w24p9p1_sr1_roles_devonly_20261012_'),'RUN_DIR_NOT_ALLOWED')
 mods=get_modules()
 if args.mode=='check':freeze_preflight(run,mods,probe_pretrain=True)
 else:
  req(not(run/'00_MATCHED_DEVFREEZE_PASS.json').exists(),'ALREADY_FROZEN_NO_RETRAIN')
  run_training(run,mods)
 return 0

if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:
  print('W24P9P1_DEVONLY_FAIL|'+type(e).__name__+'|'+str(e),file=sys.stderr,flush=True)
  raise
