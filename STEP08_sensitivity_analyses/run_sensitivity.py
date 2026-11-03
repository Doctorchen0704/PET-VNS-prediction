"""STEP08: manually run one sensitivity/validation analysis; never chain branches."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output,load_model_input
from STEP05_clinical_data_assembly.table_io import keyed_csv,align
from pet_vns_prediction import validation,sensitivity


def processing_tables(payload,x,settings):
    if payload.get('study_ids')!=list(x.index) or set(payload.get('variants',{}))!=set(sensitivity.PREDICTION_VARIANTS):
        raise ValueError('Exactly seven alternative PET matrices in frozen participant order required')
    tables={}
    for key,values in payload['variants'].items():
        a=np.asarray(values,float)
        if a.shape!=(len(x),6) or not np.isfinite(a).all():
            raise ValueError('Each variant requires a finite participants-by-six matrix')
        tables[key]=pd.DataFrame(a,index=x.index,columns=settings.pet)
    return tables


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('analysis',choices=['ablation','scanner','development-bootstrap','processing-prediction'])
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--metadata',type=Path,help='scanner analysis: CSV containing study_id and scanner string labels')
    ap.add_argument('--processing-features',type=Path,help='processing-prediction: seven ordered alternative matrices')
    ap.add_argument('--timeout-seconds',type=float,default=7200)
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args()
    data,x,y,s=load_model_input(a.input)
    scanner=None;variants=None
    if a.analysis=='scanner':
        if a.metadata is None:ap.error('--metadata is required for scanner')
        table=align(keyed_csv(a.metadata),list(x.index))
        if 'scanner' not in table or table['scanner'].isna().any() or len(set(table['scanner']))!=2:
            raise ValueError('Two explicit nonmissing scanner string labels required')
        scanner=table['scanner'].astype(str)
    if a.analysis=='processing-prediction':
        if a.processing_features is None:ap.error('--processing-features is required')
        variants=processing_tables(read_json(a.processing_features),x,s)
    print('Analysis:',a.analysis,'participants:',len(y),'full paper defaults; no other branch is started')
    if not a.execute:
        print('PLAN ONLY. Add --execute to run this analysis.');return
    out=new_output(a.output)
    write_json(out/'run_status.json',{'status':'RUNNING','analysis':a.analysis})
    try:
        if a.analysis=='ablation':
            result=validation.run_ablations(x,y,data['plans'],s,interval_draws=2000,
                interval_seed=20260910,timeout_seconds=a.timeout_seconds)
        elif a.analysis=='scanner':
            result=validation.run_cross_scanner(x,y,scanner,s,interval_draws=2000,
                interval_seed=20260910,timeout_seconds=a.timeout_seconds)
        elif a.analysis=='development-bootstrap':
            result=validation.run_development_bootstrap(x,y,s,draws=500,seed=20260921,
                reference_inner_seed=20260910,timeout_seconds=a.timeout_seconds,
                progress=lambda p:print(p,flush=True))
        else:
            result=sensitivity.run_processing_predictions(x,variants,y,s,plans=data['plans'],
                timeout_seconds_per_model=a.timeout_seconds,progress=lambda p:print(p,flush=True))
            result['comparison_definitions']=sensitivity.processing_prediction_comparisons()
            result['comparison_intervals_computed']=False
        write_json(out/'results_PRIVATE.json',result)
        write_json(out/'run_status.json',{'status':'PASS','analysis':a.analysis,'next_step_started':False})
    except BaseException:
        write_json(out/'run_status.json',{'status':'FAILED_RETAINED','analysis':a.analysis})
        raise
    print('STEP08 complete:',out)


if __name__=='__main__':main()
