"""STEP07A: refit the fixed C/CP/CPT repeated nested-CV specification."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import load_model_input,new_output,write_json
from pet_vns_prediction.modeling import run_nested_models
from pet_vns_prediction import core


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--execute',action='store_true',help='Fit only this step after reviewing inputs and plan')
    ap.add_argument('--timeout-seconds',type=float,default=7200)
    a=ap.parse_args()
    data,x,y,s=load_model_input(a.input)
    print('C/CP/CPT; participants:',len(y),'repeats:',s.repeats,'outer splits:',len(data['plans']),
          'inner folds:',s.inner_folds,'grid:',len(s.c_grid),'x',len(s.l1_grid))
    if not a.execute:
        print('PLAN ONLY. Add --execute to fit. No output written.');return
    out=new_output(a.output)
    write_json(out/'run_status.json',{'status':'RUNNING','method':'full_nested_refit'})
    try:
        result=run_nested_models(x,y,s,model_columns=data['columns'],plans=data['plans'],
            timeout_seconds=a.timeout_seconds,progress=lambda p:print(p,flush=True))
        write_json(out/'predictions.json',{'status':'PASS','participant_ids':list(x.index),
            'predictions':result['predictions'],'model_columns':result['model_columns'],
            'split_digest':result['split_digest']})
        write_json(out/'performance.json',core.summarize_predictions(result,y,s))
        write_json(out/'conditional_intervals.json',core.paired_bootstrap(result,y,draws=2000,seed=20260910))
        write_json(out/'fit_details_PRIVATE.json',{'audit':result['audit'],'tuning':result['tuning'],'plans':result['plans']})
        write_json(out/'run_status.json',{'status':'PASS','method':'full_nested_refit','next_step_started':False})
    except BaseException:
        write_json(out/'run_status.json',{'status':'FAILED_RETAINED','method':'full_nested_refit'})
        raise
    print('STEP07A complete:',out)


if __name__=='__main__':main()
