"""Gather STEP04 single-subject PET/T1 JSONs into explicit keyed CSVs."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,new_output
from pet_vns_prediction import paper_settings
import pandas as pd


def aggregate(paths, kind):
    settings=paper_settings()
    columns=settings.pet if kind=='pet' else settings.t1
    records=[]
    for path in paths:
        data=read_json(path)
        sid=data.get('study_id',data.get('subject','').removeprefix('sub-'))
        if not isinstance(sid,str) or not sid:
            raise ValueError('Explicit study_id or subject required')
        fields=data.get('canonical_features',data.get('features',data.get('primary_features',{})))
        if set(columns).issubset(fields):
            selected={key:fields[key] for key in columns}
        else:
            prefix='pet_supraGM__' if kind=='pet' else 't1__'
            selected={key:fields[key.removeprefix(prefix)] for key in columns}
        if kind=='pet' and data.get('reference')!='supraGM':
            raise ValueError('Primary PET table must use supraGM, not another reference')
        records.append({'study_id':sid,**selected})
    table=pd.DataFrame(records,columns=['study_id',*columns])
    if not records or table['study_id'].duplicated().any():
        raise ValueError('Nonempty uniquely identified feature JSON files required')
    return table


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind',choices=['pet','t1'],required=True)
    ap.add_argument('--inputs',type=Path,nargs='+',required=True,help='Explicit STEP04 JSON files; no source discovery or patient inference')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    table=aggregate(a.inputs,a.kind)
    out=new_output(a.output)
    table.to_csv(out/(a.kind+'_features.csv'),index=False,na_rep='')
    print('Feature table saved:',out)


if __name__=='__main__':main()
