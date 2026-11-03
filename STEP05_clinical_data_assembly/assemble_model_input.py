"""STEP05: join manually verified clinical/PET/T1 tables in frozen cohort order."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output
from STEP05_clinical_data_assembly.table_io import keyed_csv,assemble


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('cohort','clinical','pet','t1','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    split=ap.add_mutually_exclusive_group(required=True)
    split.add_argument('--split-plan',type=Path,help='Exact frozen outer/inner splits for paper reproduction')
    split.add_argument('--generate-splits',action='store_true',help='For a new dataset only; not a claim of historical split reproduction')
    ap.add_argument('--preserve-frozen-log',action='store_true',help='Retain explicitly supplied frozen log1p column only after agreement with raw frequency within 1e-12')
    a=ap.parse_args()
    plans=None if a.split_plan is None else read_json(a.split_plan)
    if isinstance(plans,dict):
        plans=plans['plans']
    payload,x,y=assemble(*[keyed_csv(p) for p in (a.cohort,a.clinical,a.pet,a.t1)],
                         plans=plans,generate_splits=a.generate_splits,preserve_frozen_log=a.preserve_frozen_log)
    out=new_output(a.output)
    write_json(out/'model_input.json',payload)
    x.rename_axis('study_id').to_csv(out/'predictors.csv')
    y.rename('responder_12m').rename_axis('study_id').to_csv(out/'outcomes.csv')
    write_json(out/'assembly_summary.json',{'participants':len(y),'responders':int(y.sum()),
        'row_order':'cohort.csv order preserved','identity_matching':'supplied verified key only',
        'clinical_frequency':'reported mean monthly frequency; log1p only, no 28/30-day rescaling',
        'log_frequency_source':payload['log_frequency_source'],
        'eligibility_or_outcome_changes':False,'next_step_started':False})
    print('STEP05 complete:',out)


if __name__=='__main__':main()
