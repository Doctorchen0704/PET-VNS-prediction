"""Generate descriptive Table1 data and a plain CSV, not groupwise P values."""
from pathlib import Path
import argparse
import csv
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output
from pet_vns_prediction.tables import cohort_descriptives


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='JSON rows,y,continuous,categorical; rows already keyed/aligned and units explicit')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=read_json(a.input)
    r=cohort_descriptives(d['rows'],d['y'],continuous=d['continuous'],categorical=d['categorical'])
    out=new_output(a.output);write_json(out/'Table1_data.json',r)
    with (out/'Table1_descriptive.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['field','level_or_statistic','all','responder','nonresponder'])
        for field in d['continuous']:
            for statistic in ('mean','sd','median','q1','q3','min','max'):
                w.writerow([field,statistic,*[r['groups'][g]['continuous'][field][statistic] for g in ('all','responder','nonresponder')]])
        for field in d['categorical']:
            for level in r['groups']['all']['categorical'][field]:
                w.writerow([field,str(level),*[f"{r['groups'][g]['categorical'][field][level]['count']} ({r['groups'][g]['categorical'][field][level]['percent']:.1f}%)" for g in ('all','responder','nonresponder')]])


if __name__=='__main__':main()
