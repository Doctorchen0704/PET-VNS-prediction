"""Join new STEP06/08 numerical outputs into Figure2, S1 or S2 plotting inputs."""
from pathlib import Path
import argparse
import hashlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output,load_model_input
import numpy as np
from sklearn.metrics import roc_auc_score
from pet_vns_prediction import sensitivity
from pet_vns_prediction.figures import PROCESSING_ORDER,PROCESSING_LABELS


def family_rows(payload,number):
    rows=payload['rows']
    if len(rows)!=number or payload.get('family_tests',number)!=number or any(r.get('family_tests')!=number for r in rows):
        raise ValueError('Complete original '+str(number)+'-test family required; do not supply only plotted rows')
    keys=[(r.get('region',r.get('pathway')),r['adjustment'],r.get('variant','original')) for r in rows]
    if len(set(keys))!=number:raise ValueError('Duplicated association-family row')
    return rows


def matching_provenance(payload,data,settings,input_sha=None):
    if payload.get('participant_ids')!=data['study_ids'] or payload.get('region_order')!=list(settings.pet):
        raise ValueError('Association cohort/feature order must match the exact STEP05 source')
    if input_sha is not None and payload.get('source_input_sha256')!=input_sha:
        raise ValueError('Association source-input fingerprint differs from this model-input version')


def figure2_payload(data,x,y,settings,primary,extended,limits,input_sha=None):
    matching_provenance(primary,data,settings,input_sha);matching_provenance(extended,data,settings,input_sha)
    primary=family_rows(primary,6);extended=family_rows(extended,18)
    pp=[];ee=[]
    for name in settings.pet:
        p=[r for r in primary if r.get('region',r.get('pathway'))==name and r['adjustment']=='clinical5_scanner']
        e=[r for r in extended if r.get('region',r.get('pathway'))==name and r['adjustment']=='base_plus_epilepsy_type_and_etiology3']
        if len(p)!=1 or len(e)!=1:raise ValueError('One fixed primary/combined-extended association per regional column required')
        row=p[0];ci=row['standardized_CI95']
        pp.append({**row,'pathway':name,'difference_pooled_sd':row['standardized_difference'],'ci_low_pooled_sd':ci[0],'ci_high_pooled_sd':ci[1]});ee.append({**e[0],'pathway':name})
    if len(limits)!=2 or not np.isfinite(limits).all() or limits[0]>=limits[1]:raise ValueError('Explicit fixed shared distribution limits required')
    raw=x.loc[:,list(settings.pet)].to_numpy(float)
    if raw[:,[0,4]].min()<limits[0] or raw[:,[0,4]].max()>limits[1]:raise ValueError('Fixed distribution limits would clip an observation')
    return {'raw_relative_values':raw,'y':y.to_numpy(),'primary':pp,'extended':ee,'primary_family_size':6,
        'extended_family_size':18,'shared_distribution_xlim':list(limits),'source':'STEP06 complete 6/18 families; STEP05 exact cohort/feature order'}


def figure_s1_payload(scanner,development):
    if scanner.get('status')!='PASS' or development.get('status')!='PASS' or len(scanner.get('directions',[]))!=2:
        raise ValueError('Completed bidirectional scanner and development-bootstrap outputs required')
    rows=[]
    for direction,r in enumerate(scanner['directions']):
        for model in ('C','CP'):
            m=r['models'][model]['metrics'];ci=r['conditional_intervals']['models'][model]
            if m['auc'] is None or ci['auc'] is None:raise ValueError('Scanner AUC not estimable; retain numerical result, do not fabricate a plotted interval')
            rows.append({'direction':direction,'model':model,'training_device':r['training_scanner'],'test_device':r['heldout_scanner'],
                'n_train':r['n_train'],'n_test':r['n_test'],'auc':m['auc'],'brier':m['brier'],
                'auc_conditional95':ci['auc'],'brier_conditional95':ci['brier'],
                'calibration_offset_intercept':m['calibration']['intercept_offset_slope_fixed_1']})
    pct=np.asarray(development['probability_stability']['CP']['bootstrap_percentiles_2p5_50_97p5'],float)
    n=len(development['participant_ids'])
    if pct.shape!=(3,n) or not np.isfinite(pct).all() or np.any(np.diff(pct,axis=0)<0):raise ValueError('All participant-wise development percentiles required')
    order=np.argsort(pct[1],kind='stable')
    ranked=[{'rank':j+1,'p2p5':float(pct[0,i]),'p50':float(pct[1,i]),'p97p5':float(pct[2,i])} for j,i in enumerate(order)]
    return {'cross_scanner':rows,'ranked_saved_CP_percentiles':ranked,
        'source':'STEP08 scanner and full development-bootstrap outputs; ranges are not validated prediction intervals'}


def conditional_metric_summaries(y,probabilities,draws=2000,seed=20260910):
    y=np.asarray(y,int);means={k:np.asarray(v,float).mean(axis=0) for k,v in probabilities.items()}
    if len(set(y))!=2 or any(v.shape!=y.shape or not np.isfinite(v).all() or np.any((v<0)|(v>1)) for v in means.values()):
        raise ValueError('Aligned full finite repeated held-out probabilities required')
    def score(yy,p):
        a=p[yy==1,None];b=p[yy==0];return [float(np.mean((a>b)+.5*(a==b))),float(np.mean((yy-p)**2))]
    point={k:score(y,p) for k,p in means.items()};samples={k:[] for k in means};rng=np.random.default_rng(seed)
    for _ in range(draws):
        ix=rng.integers(0,len(y),len(y))
        if len(set(y[ix]))<2:continue
        for k,p in means.items():samples[k].append(score(y[ix],p[ix]))
    result={}
    for k,p in means.items():
        if abs(point[k][0]-roc_auc_score(y,p))>1e-12:raise ArithmeticError('Independent AUC calculation differs')
        if not samples[k]:raise ValueError('No evaluable draw; no replacement sampling')
        ci=np.quantile(samples[k],[.025,.975],axis=0)
        result[k]={'auc':point[k][0],'brier':point[k][1],'auc_conditional95':ci[:,0].tolist(),'brier_conditional95':ci[:,1].tolist()}
    return result


def figure_s2_payload(data,y,settings,associations,main,processing,input_sha=None):
    matching_provenance(associations,data,settings,input_sha)
    rows=family_rows(associations,96)
    ids=data['study_ids']
    if main.get('participant_ids')!=ids or processing.get('participant_ids')!=ids or processing.get('status')!='PASS':
        raise ValueError('Same fixed participant order and completed predictions required')
    if not main.get('split_digest') or main['split_digest']!=processing.get('split_digest'):raise ValueError('Primary/processing split digest differs')
    if set(processing['predictions'])!=set(sensitivity.processing_model_definitions(settings)):
        raise ValueError('All fourteen processing models must remain available')
    probabilities={'original':np.asarray(main['predictions']['CP'],float)}
    probabilities.update({v:np.asarray(processing['predictions'][v+'_CP'],float) for v in sensitivity.PREDICTION_VARIANTS})
    if any(v.shape!=(settings.repeats,len(y)) for v in probabilities.values()):raise ValueError('Full repeats-by-participants prediction matrices required')
    models=conditional_metric_summaries(y,probabilities)
    regions=[settings.pet[0],settings.pet[4]];shown={}
    for region in regions:
        shown[region]=[]
        for variant in PROCESSING_ORDER:
            match=[r for r in rows if r.get('region',r.get('pathway'))==region and r['variant']==variant and r['adjustment']=='base']
            if len(match)!=1:raise ValueError('One fixed base-adjusted processing association per displayed cell required')
            shown[region].append(match[0])
    return {'order':list(PROCESSING_ORDER),'labels':list(PROCESSING_LABELS),'regions':regions,'association_rows':shown,
        'models':models,'q_family_size':96,'not_displayed_but_retained_in_family':['pvc4','pvc6'],
        'conditional_intervals':{'draws':2000,'seed':20260910,'scope':'Fixed patient-average held-out predictions, not model-development bootstrap'},
        'source':'STEP06 all96; STEP07 originalCP; STEP08 all14 processing fits; no model selection or fitting during assembly'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--figure',choices=['Figure2','FigureS1','FigureS2'],required=True)
    for name in ['model-input','primary','extended','scanner','development','processing-associations','primary-predictions','processing-predictions']:
        p.add_argument('--'+name,type=Path)
    p.add_argument('--distribution-limits',type=float,nargs=2);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.figure=='FigureS1':
        if not(a.scanner and a.development):p.error('S1 requires scanner and development outputs')
        payload=figure_s1_payload(read_json(a.scanner),read_json(a.development))
    else:
        if not a.model_input:p.error('--model-input required')
        data,x,y,settings=load_model_input(a.model_input)
        if a.figure=='Figure2':
            if not(a.primary and a.extended and a.distribution_limits):p.error('Figure2 requires primary,extended,distribution-limits')
            payload=figure2_payload(data,x,y,settings,read_json(a.primary),read_json(a.extended),a.distribution_limits,
                hashlib.sha256(a.model_input.read_bytes()).hexdigest())
        else:
            if not(a.processing_associations and a.primary_predictions and a.processing_predictions):p.error('S2 requires all96 associations and both prediction sources')
            payload=figure_s2_payload(data,y.to_numpy(),settings,read_json(a.processing_associations),read_json(a.primary_predictions),read_json(a.processing_predictions),
                hashlib.sha256(a.model_input.read_bytes()).hexdigest())
    out=new_output(a.output);write_json(out/(a.figure+'_plot_data.json'),payload)


if __name__=='__main__':main()
