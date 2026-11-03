"""Alternative-SERT/DAT controls of supplied fixed PC1 and saved uncertainty maps."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output
import numpy as np
from pet_vns_prediction import atlas_specificity,dat_conditioning


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['DAT','atlas','holm4'],required=True)
    p.add_argument('--input',type=Path,required=True,help='NPZ for DAT/atlas, JSON containing exactly four ordered pvalues for holm4')
    p.add_argument('--n-participants',type=int)
    p.add_argument('--atlas-use-authorized',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='holm4':
        d=read_json(a.input); values=d['pvalues']; labels=d['labels']
        if len(labels)!=4 or len(set(labels))!=4:raise ValueError('Four uniquely identified tests across the fixed two cohorts required')
        result={'labels':labels,'pvalues':values,'holm4':atlas_specificity.holm(values),'family_size':4};arrays={}
    else:
        with np.load(a.input,allow_pickle=False) as f:d={k:f[k] for k in f.files}
        x=d['observed68'];boot=d['bootstrap68'];maps=d['maps68x3'];z=d['z68x4']
        if a.mode=='DAT':result,arrays=dat_conditioning.compute(x,boot,maps,z,a.n_participants)
        else:result,arrays=atlas_specificity.analyze(x,boot,d['raw_null68'],d['conditional_null68'],maps,z)
    out=new_output(a.output);write_json(out/'specificity_results.json',result)
    if arrays:np.savez_compressed(out/'specificity_arrays.npz',**arrays)


if __name__=='__main__':main()
