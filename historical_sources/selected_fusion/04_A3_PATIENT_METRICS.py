"""A3 metrics: extends P9P3 patient-confusion bootstrap to four fixed arms.
No model/threshold/weight selection. sklearn default active-label semantics for
QWK and F1; MAE always uses the actual integer grades 0,1,2,3. Undefined QWK
is retained as NaN internally and counted explicitly, never replaced by zero.
"""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, mean_absolute_error)
METRICS=('accuracy','balanced_accuracy','macro_f1','weighted_f1','qwk','mae')
ARMS=('A0_NATURAL','A1_SCALED','A2_BINARY_P','A3_GRADE_FIXED')
CONTRASTS=((3,1),(3,0),(3,2),(1,0))

def need(ok,code):
    if not ok:raise ValueError(code)

def cm(y,p):
    y=np.asarray(y);p=np.asarray(p)
    need(y.ndim==p.ndim==1 and y.shape==p.shape and len(y)>0,'METRIC_SHAPE')
    need(np.issubdtype(y.dtype,np.integer) and np.issubdtype(p.dtype,np.integer),'METRIC_INTEGER')
    need(((y>=0)&(y<4)&(p>=0)&(p<4)).all(),'METRIC_RANGE')
    out=np.zeros((4,4),dtype=np.int64);np.add.at(out,(y,p),1);return out

def batch_metrics(cms):
    x=np.asarray(cms,dtype=np.float64)
    need(x.ndim==3 and x.shape[1:]==(4,4),'CM_SHAPE')
    need(np.isfinite(x).all() and (x>=0).all(),'CM_VALUES')
    n=x.sum((1,2));need((n>0).all(),'EMPTY_CM')
    ty=x.sum(2);py=x.sum(1);tp=np.diagonal(x,axis1=1,axis2=2)
    active=(ty+py)>0;true_present=ty>0
    rec=np.divide(tp,ty,out=np.zeros_like(tp),where=true_present)
    f1=np.divide(2*tp,ty+py,out=np.zeros_like(tp),where=active)
    # sklearn compresses absent categories in its default-label confusion matrix.
    # Compute the active union independently for each replicate and each arm.
    ranks=np.cumsum(active,axis=1)-1
    w=(ranks[:,:,None]-ranks[:,None,:])**2
    obs=(x*w).sum((1,2))
    exp=(ty[:,:,None]*py[:,None,:]*w).sum((1,2))/n
    ratio=np.divide(obs,exp,out=np.full(len(x),np.nan),where=exp>0)
    distance=np.abs(np.arange(4)[:,None]-np.arange(4)[None,:])
    values=np.column_stack((tp.sum(1)/n,rec.sum(1)/true_present.sum(1),
        f1.sum(1)/active.sum(1),(f1*ty).sum(1)/n,1-ratio,(x*distance).sum((1,2))/n))
    return values

def reference(y,p):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.array([accuracy_score(y,p),balanced_accuracy_score(y,p),
            f1_score(y,p,average='macro',zero_division=0),
            f1_score(y,p,average='weighted',zero_division=0),
            cohen_kappa_score(y,p,weights='quadratic'),mean_absolute_error(y,p)],dtype=float)

def check_equivalence(y,p):
    a=batch_metrics(cm(y,p)[None])[0];b=reference(y,p)
    need(np.allclose(a,b,rtol=0,atol=1e-12,equal_nan=True),'SKLEARN_EQUIVALENCE_FAILED')
    finite=np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0

def safe_value(x):return float(x) if np.isfinite(x) else None

def point(y,p):
    confusion=cm(y,p);err=check_equivalence(y,p);vals=batch_metrics(confusion[None])[0]
    recalls=np.divide(np.diag(confusion),confusion.sum(1),out=np.full(4,np.nan),where=confusion.sum(1)>0)
    return {'rois':len(y),'metrics':{k:safe_value(v) for k,v in zip(METRICS,vals)},
        'confusion_matrix':confusion.tolist(),'grade_counts':confusion.sum(1).tolist(),
        'per_grade_recall':[safe_value(v) for v in recalls],
        'g1_to0_count':int(confusion[1,0]),'g1_total':int(confusion[1].sum()),
        'sklearn_max_abs_error':err}

def paired_patient_bootstrap(y,preds,patients,iterations=20000,seed=42,chunk=64,callback=None):
    """All four contrasts share the very same sampled patient multiplicities."""
    y=np.asarray(y);preds=np.asarray(preds);patients=np.asarray(patients,dtype=str)
    need(preds.shape==(len(y),4) and patients.shape==y.shape,'BOOTSTRAP_ALIGNMENT')
    need(iterations>0 and chunk>0,'BOOTSTRAP_BUDGET')
    u,inv=np.unique(patients,return_inverse=True)
    pcms=np.zeros((len(u),4,4,4),dtype=np.int32)
    for a in range(4):
        cm(y,preds[:,a]);np.add.at(pcms,(inv,np.full(len(y),a),y,preds[:,a]),1)
    rng=np.random.default_rng(seed);draws=np.empty((iterations,4,6),dtype=np.float64)
    equivalent=0;maximum=0.0
    for start in range(0,iterations,chunk):
        b=min(chunk,iterations-start)
        sampled=rng.integers(0,len(u),size=(b,len(u)),dtype=np.int32)
        totals=pcms[sampled].sum(axis=1)
        vals=np.stack([batch_metrics(totals[:,a]) for a in range(4)],axis=1)
        if start==0:
            # Independent direct sklearn check of the actual first patient draws.
            for j in range(min(8,b)):
                indices=np.concatenate([np.flatnonzero(inv==k) for k in sampled[j]])
                for a in range(4):
                    r=reference(y[indices],preds[indices,a]);v=vals[j,a]
                    need(np.allclose(v,r,rtol=0,atol=1e-12,equal_nan=True),'BOOTSTRAP_REPLICATE_EQUIVALENCE_FAILED')
                    f=np.isfinite(v)&np.isfinite(r)
                    if f.any():maximum=max(maximum,float(np.max(np.abs(v[f]-r[f]))))
                    equivalent+=1
        for c,(a,b0) in enumerate(CONTRASTS):draws[start:start+b,c]=vals[:,a]-vals[:,b0]
        if callback:callback(min(start+b,iterations),iterations)
    base=np.stack([batch_metrics(cm(y,preds[:,a])[None])[0] for a in range(4)])
    results=[]
    for c,(a,b) in enumerate(CONTRASTS):
        row={'candidate':ARMS[a],'reference':ARMS[b],
             'planned_role':'primary_contrast' if c<3 else 'secondary_scale_contrast','metrics':{}}
        for k,name in enumerate(METRICS):
            x=draws[:,c,k];f=np.isfinite(x);valid=int(f.sum())
            row['metrics'][name]={'point_delta':safe_value(base[a,k]-base[b,k]),
                'percentile_95ci':[float(v) for v in np.percentile(x[f],[2.5,97.5])] if valid else None,
                'valid_replicates':valid,'undefined_replicates':iterations-valid,
                'direction':'negative_favors_candidate' if name=='mae' else 'positive_favors_candidate'}
        results.append(row)
    return {'patients':len(u),'roi_count':len(y),'repeats':iterations,'seed':seed,
       'resampling_unit':'patient; all ROIs retained with each sampled patient',
       'same_resample_for_all_arms_and_contrasts':True,'interval_method':'percentile 2.5 and 97.5',
       'multiplicity_adjusted':False,'interpretation':'internal retrospective exploratory; conditional on frozen fitted models; not training-seed uncertainty',
       'replicate_sklearn_checks':equivalent,'replicate_sklearn_max_abs_error':maximum,'contrasts':results}
