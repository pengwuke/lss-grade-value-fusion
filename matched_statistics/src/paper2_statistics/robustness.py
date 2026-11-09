"""Post-hoc fixed-prediction robustness; no fitting, selection or inference."""
import hashlib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error
from .kernels import fusion as kernel

OUTCOMES=tuple(kernel.METRICS)+tuple('recall_grade'+str(g) for g in range(4))

def values(cms):
    cms=np.asarray(cms)
    core=np.stack([kernel.batch_metrics(cms[:,s]) for s in range(2)],axis=1)
    truth=cms.sum(axis=3)
    diagonal=np.diagonal(cms,axis1=2,axis2=3)
    recalls=np.divide(diagonal,truth,out=np.full(diagonal.shape,np.nan),where=truth>0)
    return np.concatenate((core[:,0]-core[:,1],recalls[:,0]-recalls[:,1]),axis=1)

def weighted_reference(y,p,w):
    return np.asarray([accuracy_score(y,p,sample_weight=w),balanced_accuracy_score(y,p,sample_weight=w),
        f1_score(y,p,average='macro',sample_weight=w,zero_division=0),f1_score(y,p,average='weighted',sample_weight=w,zero_division=0),
        cohen_kappa_score(y,p,weights='quadratic',sample_weight=w),mean_absolute_error(y,p,sample_weight=w)])

def run_robustness(y,preds,patients,folds,iterations=100000,seed=42,chunk=32,callback=None):
    y=np.asarray(y);preds=np.asarray(preds);patients=np.asarray(patients,dtype=str);folds=np.asarray(folds)
    kernel.need(preds.shape==(len(y),2) and patients.shape==folds.shape==y.shape,'ROBUSTNESS_ALIGNMENT')
    kernel.need(type(iterations) is int and iterations>0 and type(chunk) is int and chunk>0,'ROBUSTNESS_BUDGET')
    u,inv=np.unique(patients,return_inverse=True);pcms=np.zeros((len(u),2,4,4),dtype=np.int32)
    for s in range(2):
        kernel.cm(y,preds[:,s]);np.add.at(pcms,(inv,np.full(len(y),s),y,preds[:,s]),1)
    for k in range(len(u)):kernel.need(len(np.unique(folds[inv==k]))==1,'PATIENT_CROSSES_FOLDS')
    totals=pcms.sum(0);points=values(totals[None])[0];draws=np.empty((iterations,10),dtype=np.float64)
    rng=np.random.default_rng(seed);maxerr=0.;checked=0;recall_checks=0
    for start in range(0,iterations,chunk):
        n=min(chunk,iterations-start);sampled=rng.integers(0,len(u),size=(n,len(u)),dtype=np.int32)
        cms=pcms[sampled].sum(1);draws[start:start+n]=values(cms)
        if start==0:
            for j in range(min(8,n)):
                ii=np.concatenate([np.flatnonzero(inv==k) for k in sampled[j]])
                for s in range(2):
                    a=kernel.batch_metrics(cms[j:j+1,s])[0];b=kernel.reference(y[ii],preds[ii,s])
                    kernel.need(np.allclose(a,b,rtol=0,atol=1e-12,equal_nan=True),'ROBUSTNESS_SKLEARN_PARITY')
                    finite=np.isfinite(a)&np.isfinite(b)
                    if finite.any():maxerr=max(maxerr,float(np.max(np.abs(a[finite]-b[finite]))))
                    checked+=1
                for g in range(4):
                    mask=y[ii]==g
                    expected=(np.mean(preds[ii,0][mask]==g)-np.mean(preds[ii,1][mask]==g)) if mask.any() else np.nan
                    kernel.need(np.isclose(draws[j,6+g],expected,rtol=0,atol=1e-12,equal_nan=True),'ROBUSTNESS_RECALL_PARITY')
                    recall_checks+=1
        if callback:callback(start+n,iterations)
    outcomes={}
    for k,name in enumerate(OUTCOMES):
        x=draws[:,k];good=np.isfinite(x);valid=int(good.sum())
        outcomes[name]={'point_delta':kernel.safe_value(points[k]),'percentile_95ci':np.percentile(x[good],[2.5,97.5]).tolist() if valid else None,
          'bonferroni_type_99_5ci':np.percentile(x[good],[.25,99.75]).tolist() if valid else None,
          'valid_replicates':valid,'undefined_replicates':iterations-valid,'direction':'negative_favorable' if name=='mae' else 'positive_favorable'}
    prefix=None
    if iterations>=20000:
        prefix={name:np.percentile(draws[:20000,k][np.isfinite(draws[:20000,k])],[2.5,97.5]).tolist() if np.isfinite(draws[:20000,k]).any() else None for k,name in enumerate(OUTCOMES[:6])}
    delete_patient=values(totals[None]-pcms)[:,:6];influence={}
    for k,name in enumerate(OUTCOMES[:6]):
        x=delete_patient[:,k];good=np.isfinite(x)
        influence[name]={'point_delta':kernel.safe_value(points[k]),'delete_one_patient_min':float(x[good].min()) if good.any() else None,
          'delete_one_patient_max':float(x[good].max()) if good.any() else None,'maximum_absolute_point_change':float(np.max(np.abs(x[good]-points[k]))) if good.any() else None,
          'positive_differences':int((x[good]>0).sum()),'zero_differences':int((x[good]==0).sum()),'negative_differences':int((x[good]<0).sum()),'undefined_deletions':int((~good).sum())}
    delete_fold=[]
    for fold in np.unique(folds):
        keep=folds!=fold
        cm=np.stack([kernel.cm(y[keep],preds[keep,s]) for s in range(2)])
        vv=values(cm[None])[0]
        delete_fold.append({'omitted_outer_fold':int(fold),'remaining_patients':len(np.unique(patients[keep])),'remaining_rois':int(keep.sum()),'deltas':{name:kernel.safe_value(vv[k]) for k,name in enumerate(OUTCOMES[:6])}})
    counts=pcms[:,0].sum((1,2));weighted_cms=(pcms/counts[:,None,None,None]).sum(0)
    weighted_points=np.stack([kernel.batch_metrics(weighted_cms[s:s+1])[0] for s in range(2)])
    weights=1/counts[inv];weighted_maxerr=0.
    for s in range(2):
        ref=weighted_reference(y,preds[:,s],weights)
        kernel.need(np.allclose(ref,weighted_points[s],rtol=0,atol=1e-12,equal_nan=True),'EQUAL_PATIENT_SKLEARN_PARITY')
        finite=np.isfinite(ref)&np.isfinite(weighted_points[s]);weighted_maxerr=max(weighted_maxerr,float(np.max(np.abs(ref[finite]-weighted_points[s,finite]))))
    equal_patient={name:{'R11':kernel.safe_value(weighted_points[0,k]),'natural_selected':kernel.safe_value(weighted_points[1,k]),'delta':kernel.safe_value(weighted_points[0,k]-weighted_points[1,k])} for k,name in enumerate(OUTCOMES[:6])}
    return {'patients':len(u),'roi_count':len(y),'iterations':iterations,'seed':seed,'chunk':chunk,
      'resampling_unit':'patient retaining all ROIs and multiplicities; paired',
      'family_size':10,'family_alpha':.05,'individual_confidence':.995,'family_percentile_bounds':[.25,99.75],
      'coverage_note':'Approximate percentile-bootstrap Bonferroni-type sensitivity for this ten-outcome family only; not correction of the earlier paper-wide contrasts',
      'outcomes':outcomes,'first_20000_core_95ci':prefix,'draws_sha256':hashlib.sha256(draws.tobytes()).hexdigest(),
      'replicate_sklearn_checks':checked,'replicate_sklearn_max_abs_error':maxerr,'raw_grade_recall_checks':recall_checks,
      'equal_patient_weighted_sklearn_max_abs_error':weighted_maxerr,'patient_roi_count_summary':{'minimum':int(counts.min()),'median':float(np.median(counts)),'maximum':int(counts.max())},
      'delete_one_patient':influence,'delete_one_outer_fold':delete_fold,'equal_patient_weighted_pooled_metrics':equal_patient,
      'influence_note':'All deletions and equal-patient metrics are descriptive changes to the evaluation set/estimand; no retraining; not leave-one-out cross-validation',
      'model_training_or_selection_performed':False,'patient_identifiers_in_result':False}
