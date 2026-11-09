"""A6 inner-only selector derived from the exact archived R11 selector.
The five sort keys and raw float calculations are inherited. No outer input.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,cohen_kappa_score

def selection_metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int)
    g1=y==1; den=max(int(g1.sum()),1)
    return {
      "qwk":float(cohen_kappa_score(y,p,weights="quadratic")),
      "mae":float(np.mean(np.abs(y-p))),
      "balanced_accuracy":float(balanced_accuracy_score(y,p)),
      "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
      "accuracy":float(accuracy_score(y,p)),
      "g1_to0_rate":float(np.sum(g1&(p==0))/den),
    }

GRID=[0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]
ORDER=['qwk','mae','balanced_accuracy','distance_from_fixed_040','coral_weight']

class SelectionError(RuntimeError):pass

def need(ok,code):
    if not bool(ok):raise SelectionError(code)

def check_probability(p,label='PROB',profile=None):
    p=np.asarray(p,dtype=np.float64)
    need(p.ndim==2 and p.shape[1]==4 and len(p)>0 and np.isfinite(p).all(),label+'_SHAPE_FINITE')
    need((p>=0).all() and (p<=1).all(),label+'_RANGE')
    err=p.sum(1)-1.0
    if profile is None:
        need((np.abs(err)<=2e-6).all(),label+'_MASS')
    else:
        # Only caller's SHA-locked P4C validation file is permitted here.
        need(profile['finite'] and profile['shape']==list(p.shape) and profile['negative_values']==0 and profile['above_one']==0,label+'_REVIEW_PROFILE')
        need(int((np.abs(err)>2e-6).sum())==profile['rows_exceeding_original_2e_6'],label+'_REVIEW_COUNT')
        for got,key in [(float(np.abs(err).max()),'max_abs_mass_error'),(float(p.sum(1).min()),'min_sum'),(float(p.sum(1).max()),'max_sum')]:
            need(abs(got-profile[key])<=1e-12,label+'_REVIEW_'+key)
        floor=1e-6;slack=8*np.finfo(np.float32).eps
        hits=np.isclose(p,floor,rtol=0,atol=1e-12).sum(1)
        need((p>=floor-1e-12).all() and (hits<=3).all(),label+'_FLOOR_PROFILE')
        need((err>=-slack).all() and (err<=hits*floor+slack).all(),label+'_MASS_ENVELOPE')
    return p

def prediction_parity(cp,dp,weights,p4c=False):
    """No label argument: representation checks cannot use outcome scores."""
    cp=np.asarray(cp,dtype=np.float64);dp=np.asarray(dp,dtype=np.float64)
    need(cp.shape==dp.shape and cp.ndim==2 and cp.shape[1]==4,'PARITY_SHAPE')
    need(np.isfinite(cp).all() and np.isfinite(dp).all() and (cp>=0).all() and (dp>=0).all(),'PARITY_NUMERIC')
    need((cp.sum(1)>0).all() and (dp.sum(1)>0).all(),'PARITY_ZERO_MASS')
    cn=cp/cp.sum(1,keepdims=True);dn=dp/dp.sum(1,keepdims=True)
    counts=[]
    for w in weights:
        v=np.asarray(w,dtype=float)
        need(v.ndim==0 or v.shape==(len(cp),),'WEIGHT_SHAPE')
        v=float(v) if v.ndim==0 else v[:,None]
        raw=(v*cp+(1-v)*dp).argmax(1)
        ponly=(v*(cn if p4c else cp)+(1-v)*dp).argmax(1)
        common=(v*cn+(1-v)*dn).argmax(1)
        a=int(np.count_nonzero(raw!=ponly));b=int(np.count_nonzero(raw!=common))
        counts.append({'p4c_only_mismatches':a,'all_branches_mismatches':b})
        need(a==0 and b==0,'NORMALIZATION_CHANGES_DECISION_STOP_NO_ALTERNATE_ANALYSIS')
    return {'weights_checked':len(weights),'decisions_checked':len(weights)*len(cp),'two_diagnostic_representations_unchanged':True,'counts':counts,'primary_arrays_modified':False}

def select_inner(y,cp,dp,weights=GRID):
    """No outer predictions, outer labels, or outer performance arguments."""
    y=np.asarray(y)
    need(y.ndim==1 and np.isin(y,[0,1,2,3]).all() and len(y)==len(cp)==len(dp),'INNER_LABEL_OR_ROW_COUNT')
    need(list(weights)==GRID,'GRID_CHANGED')
    rows=[]
    for w in weights:
        pred=(w*cp+(1.0-w)*dp).argmax(1)
        row={'coral_weight':w,'deit_weight':1.0-w,**selection_metrics(y,pred)}
        need(all(np.isfinite(v) for v in row.values()),'SELECTION_NONFINITE_METRIC')
        rows.append(row)
    sdf=pd.DataFrame(rows);sdf['distance_from_fixed_040']=(sdf['coral_weight']-0.40).abs()
    ranked=sdf.sort_values(ORDER,ascending=[False,True,False,True,True],kind='mergesort')
    winner=ranked.iloc[0]
    return {'selected_image_weight':float(winner['coral_weight']),'selected_deit_weight':float(winner['deit_weight']),
            'grid_results':sdf.to_dict(orient='records'),'sort_order':ORDER,'validation_metrics_are_not_test_results':True}
