"""Prepare the original descriptive Table1 fields from keyed analysis inputs."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import load_model_input,new_output,write_json
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-input',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True,help='study_id,epilepsy_type,etiology_detailed,scanner_source_key')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data,x,y,_=load_model_input(a.model_input)
    meta=pd.read_csv(a.metadata,dtype=str,keep_default_na=False)
    required={'study_id','epilepsy_type','etiology_detailed','scanner_source_key'}
    if not required<=set(meta.columns) or meta.study_id.duplicated().any():
        raise ValueError('Unique keyed metadata with all declared categorical fields required')
    meta=meta.set_index('study_id')
    if not set(x.index)<=set(meta.index):raise ValueError('Missing cohort member; do not delete rows')
    rows=[]
    for sid in x.index:
        m=meta.loc[sid]
        if any(not m[k] for k in ('epilepsy_type','etiology_detailed','scanner_source_key')):
            raise ValueError('Missing categorical source value')
        r=x.loc[sid].to_dict()
        r.update(study_id=sid,epilepsy_classification=m['epilepsy_type'],
                 etiology_group=m['etiology_detailed'],scanner=m['scanner_source_key'],
                 baseline_monthly_frequency_reconstructed=float(np.expm1(r['log1p_baseline_monthly_frequency'])))
        rows.append(r)
    out=new_output(a.output)
    write_json(out/'Table1_input.json',{'rows':rows,'y':y.tolist(),
        'continuous':['age_at_implant_years','epilepsy_duration_reported_years','baseline_monthly_frequency_reconstructed'],
        'categorical':['sex','mri_lesional_status','epilepsy_classification','etiology_group','scanner'],
        'frequency_definition':'Original descriptive-table arithmetic: expm1 of the frozen log1p monthly history frequency. Not a new postoperative count.',
        'scanner_presentation':'Source grouping retained; use separately verified scanner display labels in publication.'})

if __name__=='__main__':main()
