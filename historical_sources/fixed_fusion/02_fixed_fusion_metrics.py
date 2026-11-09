"""A4 extension of exact A3 metrics to fixed fusion and linear contrasts.
The six metric definitions, active-label QWK and undefined-value handling are
imported unchanged from the reviewed A3 module. No new model selection.
"""
from pathlib import Path
import importlib.util
import numpy as np
p=Path(__file__).with_name('04_A3_PATIENT_METRICS.py')
spec=importlib.util.spec_from_file_location('a4_exact_a3_metrics',p)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
METRICS=base.METRICS
point=base.point
cm=base.cm
batch_metrics=base.batch_metrics
reference=base.reference
need=base.need
safe_value=base.safe_value

def validate_contrasts(contrasts,nsystems):
    need(len(contrasts)>0,'NO_CONTRASTS')
    need(len({c['name'] for c in contrasts})==len(contrasts),'DUPLICATE_CONTRAST')
    for c in contrasts:
        terms=c['terms']
        need(len(terms)>=2 and len({t[0] for t in terms})==len(terms),'CONTRAST_TERM_DUPLICATE')
        need(all(isinstance(i,int) and 0<=i<nsystems and coef in (-1,1) for i,coef in terms),'CONTRAST_TERM_INVALID')
        need(sum(coef for _,coef in terms)==0,'CONTRAST_NOT_ZERO_SUM')

def linear_effect(values,terms):
    return sum(coef*values[...,i,:] for i,coef in terms)

def paired_fusion_bootstrap(y,preds,patients,systems,contrasts,iterations=20000,seed=42,chunk=32,callback=None):
    """Same patient resample for every single/fused system and interaction."""
    y=np.asarray(y);preds=np.asarray(preds);patients=np.asarray(patients,dtype=str)
    ns=len(systems);nc=len(contrasts)
    need(preds.shape==(len(y),ns) and patients.shape==y.shape,'BOOTSTRAP_ALIGNMENT')
    need(ns>0 and len(set(systems))==ns and iterations>0 and chunk>0,'BOOTSTRAP_CONFIG')
    validate_contrasts(contrasts,ns)
    u,inv=np.unique(patients,return_inverse=True)
    pcms=np.zeros((len(u),ns,4,4),dtype=np.int32)
    for a in range(ns):
        cm(y,preds[:,a]);np.add.at(pcms,(inv,np.full(len(y),a),y,preds[:,a]),1)
    rng=np.random.default_rng(seed);draws=np.empty((iterations,nc,6),dtype=np.float64)
    equivalents=0;maximum=0.0
    for start in range(0,iterations,chunk):
        n=min(chunk,iterations-start)
        sampled=rng.integers(0,len(u),size=(n,len(u)),dtype=np.int32)
        totals=pcms[sampled].sum(axis=1)
        vals=np.stack([batch_metrics(totals[:,a]) for a in range(ns)],axis=1)
        if start==0:
            for j in range(min(8,n)):
                indices=np.concatenate([np.flatnonzero(inv==k) for k in sampled[j]])
                for a in range(ns):
                    v=vals[j,a];r=reference(y[indices],preds[indices,a])
                    need(np.allclose(v,r,rtol=0,atol=1e-12,equal_nan=True),'BOOTSTRAP_SKLEARN_PARITY')
                    f=np.isfinite(v)&np.isfinite(r)
                    if f.any():maximum=max(maximum,float(np.max(np.abs(v[f]-r[f]))))
                    equivalents+=1
        for c,definition in enumerate(contrasts):draws[start:start+n,c]=linear_effect(vals,definition['terms'])
        if callback:callback(start+n,iterations)
    base_points=np.stack([batch_metrics(cm(y,preds[:,a])[None])[0] for a in range(ns)])
    out=[]
    for c,definition in enumerate(contrasts):
        estimate=linear_effect(base_points,definition['terms'])
        row={'name':definition['name'],'role':definition['role'],
             'terms':[{'system':systems[i],'coefficient':coef} for i,coef in definition['terms']],
             'metrics':{}}
        for j,name in enumerate(METRICS):
            x=draws[:,c,j];good=np.isfinite(x);n=int(good.sum())
            row['metrics'][name]={'point_delta':safe_value(estimate[j]),
                'percentile_95ci':[float(v) for v in np.percentile(x[good],[2.5,97.5])] if n else None,
                'valid_replicates':n,'undefined_replicates':iterations-n,
                'direction':'negative_favorable' if name=='mae' else 'positive_favorable'}
        out.append(row)
    return {'patients':len(u),'roi_count':len(y),'repeats':iterations,'seed':seed,
            'same_resample_for_all_systems_and_contrasts':True,'resampling_unit':'patient, retaining all ROIs and duplicate multiplicities',
            'interval_method':'percentile 2.5 and 97.5','multiplicity_adjusted':False,
            'replicate_sklearn_checks':equivalents,'replicate_sklearn_max_abs_error':maximum,
            'interpretation':'internal retrospective exploratory; conditional on frozen models; no training-seed uncertainty',
            'contrasts':out}
