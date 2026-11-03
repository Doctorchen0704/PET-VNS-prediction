"""Assemble plotting inputs from numerical results; never refit a model."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score,brier_score_loss
from pet_vns_prediction.figures import fixed_bin_calibration


def ablation_plot_rows(payload):
    if isinstance(payload,list):rows=payload
    elif 'ablation' in payload:rows=payload['ablation']
    else:
        metrics=payload.get('ablation_metrics',payload)
        order=['CP_without_thalamus','CP_without_striatopallidal','CP_without_both']
        if set(metrics.get('comparisons',{}))!={'CP_minus_'+name for name in order}:
            raise ValueError('All three prespecified retuned ablation comparisons are required')
        rows=[]
        for name in order:
            row=metrics['comparisons']['CP_minus_'+name]
            rows.append({'model_removed':name,'delta_auc_CP_minus_reduced':row['delta_auc'],
                'conditional95':row['delta_auc_conditional95'],'reduced_model_auc':metrics['models'][name]['auc']})
    if len(rows)!=3:raise ValueError('All three ordered retuned-ablation rows required')
    for row in rows:
        if not np.isfinite(row['delta_auc_CP_minus_reduced']) or np.asarray(row['conditional95']).shape!=(2,):
            raise ValueError('Finite ablation point and interval required')
    return rows


def figure3_payload(y,predictions,metrics,ablation,intervals=None):
    y=np.asarray(y)
    if y.ndim!=1 or len(y)<4 or not np.isin(y,[0,1]).all() or len(set(y.tolist()))!=2:
        raise ValueError('Both binary outcome groups in the frozen participant order required')
    models={};source_models=metrics.get('models',metrics)
    cis=intervals.get('intervals',intervals) if intervals is not None else None
    for name in ('C','CP','CPT'):
        pred=np.asarray(predictions[name],float)
        if pred.ndim!=2 or pred.shape[1]!=len(y) or not np.isfinite(pred).all() or np.any((pred<0)|(pred>1)):
            raise ValueError('Full repeats-by-participants probabilities required; no row filtering')
        mean=pred.mean(axis=0); record=source_models[name]
        auc=float(roc_auc_score(y,mean)); brier=float(brier_score_loss(y,mean))
        if abs(auc-record['auc'])>1e-12 or abs(brier-record['brier'])>1e-12:
            raise ValueError('Predictions and supplied metric version do not agree')
        fpr,tpr,_=roc_curve(y,mean)
        models[name]={key:record[key] for key in ('auc','brier')}
        for key in ('auc','brier'):
            if cis is not None:
                models[name][key+'_conditional95']=cis[name+'_'+key]
            elif key+'_conditional95' in record:
                models[name][key+'_conditional95']=record[key+'_conditional95']
            else:raise ValueError('Supply STEP07 conditional_intervals.json explicitly; no interval is inferred from point metrics')
        models[name].update(roc_fpr=fpr.tolist(),roc_tpr=tpr.tolist())
        if name=='CP':models[name]['frozen_calibration']=record['calibration']
    rows=ablation_plot_rows(ablation)
    if isinstance(ablation,dict) and 'ablation_metrics' in ablation:
        ref=ablation['ablation_metrics']['models']['CP']
        if abs(ref['auc']-models['CP']['auc'])>1e-12 or abs(ref['brier']-models['CP']['brier'])>1e-12:
            raise ValueError('Ablation full-CP fit does not match the primary model version')
    return {'models':models,'calibration':fixed_bin_calibration(y,np.asarray(predictions['CP']).mean(0)),
            'ablation':rows,'n':len(y),'summary':'Mean held-out probability across repeats'}


def figure4_payload(aggregate,arrays,labels,limits):
    if len(labels)!=68 or len(set(labels))!=68:raise ValueError('68 ordered cortical parcel names required')
    scatter=[]
    for condition in ('baseline','DAT'):
        x=np.asarray(arrays[condition+'_target_residuals'])[:,0]
        y=np.asarray(arrays[condition+'_PC1_residuals'])[0]
        effect=float(np.asarray(arrays[condition+'_effects'])[0,0])
        if x.shape!=(68,) or y.shape!=(68,) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Complete saved rank-residual pairs required')
        if abs(np.corrcoef(x,y)[0,1]-effect)>1e-12:raise ValueError('Saved residuals disagree with saved effect')
        scatter.append({'condition':condition,'saved_effect':effect,
            'points':[{'parcel':lab,'x':float(xx),'y':float(yy)} for lab,xx,yy in zip(labels,x,y)]})
    return {**aggregate,**limits,'scatter_data':scatter,
            'cohort_order':aggregate.get('cohort_order',limits.get('cohort_order',['VNS','TLE','Healthy'])),
            'interval_plot_rows':aggregate.get('interval_plot_rows',aggregate.get('SERT_correspondence'))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figure',choices=['Figure3','Figure4'],required=True)
    p.add_argument('--input',type=Path,required=True,help='Figure3: model input JSON; Figure4: aggregate interval JSON')
    p.add_argument('--predictions',type=Path,help='Figure3: JSON with predictions mapping')
    p.add_argument('--metrics',type=Path,help='Figure3: complete performance JSON with conditional intervals')
    p.add_argument('--intervals',type=Path,help='Figure3: STEP07 conditional_intervals.json; required if CIs are not embedded')
    p.add_argument('--ablation',type=Path,help='Figure3: JSON list or object containing ablation')
    p.add_argument('--arrays',type=Path,help='Figure4: conditioning arrays NPZ')
    p.add_argument('--labels',type=Path,help='Figure4: JSON list or object containing labels')
    p.add_argument('--color-limits',type=Path,help='Figure4: JSON PC1_color_limits and SERT_color_limits')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.figure=='Figure3':
        if not all((a.predictions,a.metrics,a.ablation)):p.error('Figure3 requires predictions, metrics and ablation')
        data=read_json(a.input);pred=read_json(a.predictions);abl=read_json(a.ablation)
        if 'participant_ids' in pred and pred['participant_ids']!=data['study_ids']:
            raise ValueError('Model-input and prediction participant order differ')
        result=figure3_payload(data['y'],pred.get('predictions',pred),read_json(a.metrics),abl,
            read_json(a.intervals) if a.intervals else None)
    else:
        if not all((a.arrays,a.labels,a.color_limits)):p.error('Figure4 requires arrays, labels and color-limits')
        labels=read_json(a.labels);labels=labels.get('labels',labels) if isinstance(labels,dict) else labels
        with np.load(a.arrays,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        result=figure4_payload(read_json(a.input),arrays,labels,read_json(a.color_limits))
    out=new_output(a.output);write_json(out/(a.figure+'_plot_data.json'),result)


if __name__=='__main__':main()
