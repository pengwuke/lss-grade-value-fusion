"""Frozen-prediction descriptive metrics for the independent reporting utility.

Numerical scoring derives from the study's locked full-five-fold reporting core.
The CSV wrapper validates structure and caller declarations; neither this module
nor a successful report can establish training/selection independence.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error, confusion_matrix, precision_recall_fscore_support, log_loss)


def ECE(y, pred, p):
    conf=p.max(axis=1)
    bins=np.minimum(np.floor(conf*10).astype(int),9)
    total=0.
    for k in range(10):
        mask=bins==k
        if mask.any(): total+=float(mask.mean()*abs(np.mean(pred[mask]==y[mask])-conf[mask].mean()))
    return total

METRIC_NAMES=('accuracy','balanced_accuracy','macro_f1','weighted_f1','qwk','mae')

GRADE_NAMES=('normal','mild','moderate','severe')

def score(y,pred,prob=None):
    cm=confusion_matrix(y,pred,labels=[0,1,2,3])
    vals=[accuracy_score(y,pred),balanced_accuracy_score(y,pred),
        f1_score(y,pred,average='macro',zero_division=0),
        f1_score(y,pred,average='weighted',zero_division=0),
        cohen_kappa_score(y,pred,labels=[0,1,2,3],weights='quadratic'),
        mean_absolute_error(y,pred)]
    pre,rec,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1,2,3],zero_division=0)
    rates={}
    for name,n,den in (
        ('normal_to_mild',cm[0,1],cm[0].sum()),
        ('mild_to_normal',cm[1,0],cm[1].sum()),
        ('moderate_correct',cm[2,2],cm[2].sum()),
        ('severe_underestimated',cm[3,:3].sum(),cm[3].sum()),
        ('severe_predicted_normal_or_mild',cm[3,:2].sum(),cm[3].sum())):
        rates[name]={'count':int(n),'denominator':int(den),'rate':float(n/den) if den else None}
    # Probability origin and hard-class parity must be checked by the wrapper.
    # No probability score is imputed when original OOF probabilities have not
    # been supplied and explicitly declared by the caller.
    pm={'multiclass_log_loss':None,
        'multiclass_brier_sum_over_4_classes':None,
        'top_label_ECE_10_equal_width_bins':None}
    if prob is not None:
        pn=prob/prob.sum(axis=1,keepdims=True)
        onehot=np.eye(4)[y]
        eps=np.finfo(np.float64).eps
        ll=float(-np.log(np.clip(pn[np.arange(len(y)),y],eps,1-eps)).mean())
        assert abs(ll-log_loss(y,pn,labels=[0,1,2,3]))<1e-12
        pm={'multiclass_log_loss':ll,
            'multiclass_brier_sum_over_4_classes':float(((pn-onehot)**2).sum(axis=1).mean()),
            'top_label_ECE_10_equal_width_bins':ECE(y,pred,pn)}
    return {'n_ROIs':int(len(y)), 'metrics':dict(zip(METRIC_NAMES,map(float,vals))),
        'per_grade':{str(g):{'grade':GRADE_NAMES[g],'precision':float(pre[g]),'recall':float(rec[g]),'f1':float(f1[g]),'support':int(support[g])} for g in range(4)},
        'confusion_matrix_true_rows_predicted_columns':cm.tolist(),
        'normal_mild_true_rows_all_4_predicted_columns':cm[:2].tolist(),
        'targeted_errors':rates,'probability_metrics':pm}

def mean_sd(items):
    def pair(xs):
        finite=[x for x in xs if x is not None and np.isfinite(x)]
        complete=len(xs)==5 and len(finite)==5
        return {'mean':float(np.mean(xs)) if complete else None,
                'sample_SD_ddof1':float(np.std(xs,ddof=1)) if complete else None,
                'n_fold_values':len(xs),'n_finite_fold_values':len(finite)}
    return {
        'metrics':{k:pair([x['metrics'][k] for x in items]) for k in METRIC_NAMES},
        'per_grade':{str(g):{k:pair([x['per_grade'][str(g)][k] for x in items]) for k in ('precision','recall','f1')} for g in range(4)},
        'targeted_error_rates':{k:pair([x['targeted_errors'][k]['rate'] for x in items]) for k in items[0]['targeted_errors']},
        'probability_metrics':{k:pair([x['probability_metrics'][k] for x in items]) for k in items[0]['probability_metrics']},
    }
