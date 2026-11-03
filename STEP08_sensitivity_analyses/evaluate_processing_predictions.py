"""STEP08 evaluation: all 28 fixed processing comparisons, no classifier fit."""
from pathlib import Path
import argparse
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output,load_model_input
from pet_vns_prediction import sensitivity
from STEP07_prediction_models.evaluate_predictions import predictions_result


def source_auc(y,p):
    positives=p[y==1];negatives=p[y==0]
    return float(np.mean((positives[:,None]>negatives)+(positives[:,None]==negatives)*.5))


def paired_comparison(y,a,b,*,seed=20260910,draws=2000):
    y,a,b=np.asarray(y),np.asarray(a,float),np.asarray(b,float)
    if a.shape!=y.shape or b.shape!=y.shape or not np.isfinite([a,b]).all():
        raise ValueError('Aligned finite patient-average probabilities required')
    rng=np.random.default_rng(seed);values=[]
    for _ in range(draws):
        ix=rng.integers(0,len(y),len(y));yy=y[ix]
        if len(np.unique(yy))<2:continue
        values.append([source_auc(yy,a[ix])-source_auc(yy,b[ix]),
                       np.mean((yy-a[ix])**2)-np.mean((yy-b[ix])**2)])
    if not values:raise ValueError('No two-class resampling support')
    ci=np.quantile(values,[.025,.975],axis=0)
    if max(abs(source_auc(y,p)-roc_auc_score(y,p)) for p in (a,b))>1e-12:
        raise ArithmeticError('Pair-count and independent ROC AUC disagree')
    return {'delta_auc':source_auc(y,a)-source_auc(y,b),
            'delta_brier':float(np.mean((y-a)**2)-np.mean((y-b)**2)),
            'delta_auc_conditional95':ci[:,0].tolist(),
            'delta_brier_conditional95':ci[:,1].tolist(),'valid_draws':len(values),'seed':seed}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--primary-predictions',type=Path,required=True)
    ap.add_argument('--processing-results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    data,x,y,s=load_model_input(a.input)
    primary=read_json(a.primary_predictions)
    source=predictions_result(primary,list(x.index),s.repeats)
    processing=read_json(a.processing_results)
    names=set(sensitivity.processing_model_definitions(s))
    if processing.get('status')!='PASS' or processing.get('participant_ids')!=list(x.index):
        raise ValueError('Complete successful processing predictions in identical cohort order required')
    if set(processing.get('predictions',{}))!=names:
        raise ValueError('All 14 processing predictions are required; no selected-model evaluation')
    if not primary.get('split_digest') or primary['split_digest']!=processing.get('split_digest'):
        raise ValueError('Primary and processing predictions must declare identical split digest')
    matrices={}
    for name,value in processing['predictions'].items():
        matrix=np.asarray(value,float)
        if matrix.shape!=(s.repeats,len(y)) or not np.isfinite(matrix).all() or ((matrix<0)|(matrix>1)).any():
            raise ValueError('Invalid processing probability matrix')
        matrices[name]=matrix
    means={name:matrix.mean(0) for name,matrix in matrices.items()}
    means.update({'CORE_'+name:matrix.mean(0) for name,matrix in source['predictions'].items()})
    pairs=sensitivity.processing_prediction_comparisons()
    comparisons=[{**pair,**paired_comparison(y.to_numpy(),means[pair['a']],means[pair['b']],seed=pair['seed'])} for pair in pairs]
    models={name:{'auc':source_auc(y.to_numpy(),p),'brier':float(np.mean((y.to_numpy()-p)**2))} for name,p in means.items()}
    out=new_output(a.output)
    write_json(out/'processing_comparisons.json',{'models':models,'comparisons':comparisons,
        'comparison_count':len(comparisons),'difference_sign':'both metrics are a minus b',
        'model_fitting_performed':False,'intervals':'conditional fixed patient-average OOF, 2000 paired draws',
        'prediction_training_provenance':'supplied predictions; not reverified by summary-only calculation'})
    print('All 28 processing comparisons saved:',out)


if __name__=='__main__':main()
