"""STEP07B: summaries from supplied predictions only; never refits a model."""
from pathlib import Path
import argparse
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output,load_model_input
from pet_vns_prediction import core


def predictions_result(payload,ids,repeats):
    if payload.get('participant_ids')!=ids or payload.get('status')!='PASS':
        raise ValueError('Explicit successful predictions in identical participant order required')
    if set(payload.get('predictions',{}))!={'C','CP','CPT'}:
        raise ValueError('All three C/CP/CPT prediction matrices required')
    matrices={key:np.asarray(value,float) for key,value in payload['predictions'].items()}
    if any(m.shape!=(repeats,len(ids)) or not np.isfinite(m).all() or ((m<0)|(m>1)).any() for m in matrices.values()):
        raise ValueError('Complete repeats-by-participants probability matrices required')
    return {'status':'PASS','participant_ids':ids,'predictions':matrices}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True,help='Exact model_input.json supplies outcome and order')
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    _,x,y,s=load_model_input(a.input)
    result=predictions_result(read_json(a.predictions),list(x.index),s.repeats)
    out=new_output(a.output)
    write_json(out/'performance.json',core.summarize_predictions(result,y,s))
    write_json(out/'conditional_intervals.json',core.paired_bootstrap(result,y,draws=2000,seed=20260910))
    write_json(out/'scope.json',{'model_refitted':False,'saved_prediction_training_provenance':'not_reverified',
        'intervals':'conditional participant bootstrap of fixed predictions; not redevelopment'})
    print('STEP07B summary-only complete:',out)


if __name__=='__main__':main()
