"""Fit one externally specified healthy or adult-TLE cohort; no downloads."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json, write_json, new_output
import numpy as np
from pet_vns_prediction import spatial, spatial_tle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='NPZ arrays; schema is in README')
    p.add_argument('--metadata',type=Path,required=True,help='JSON: feature_names and receptor_names; fixed QC membership is supplied separately')
    p.add_argument('--branch',choices=['healthy','adult_tle'],required=True)
    p.add_argument('--effect-only',action='store_true',help='TLE sensitivity only: no spatial P values')
    p.add_argument('--atlas-use-authorized',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); meta=read_json(a.metadata)
    with np.load(a.input,allow_pickle=False) as f: d={k:f[k] for k in f.files}
    if a.branch=='healthy':
        if a.effect_only: raise ValueError('Healthy production branch has fixed full spatial tests')
        r=spatial.analyze_cohort(d['raw_roi_means'],meta['feature_names'],d['cortical_indices'],
            d['receptors68x18'],meta['receptor_names'],d['sphere_coords68x3'],d['ap68'],d['g168'],d.get('original_loading82'))
        result,arrays=r['results'],r['arrays']
    else:
        names=meta['feature_names']; ix=np.asarray(d['cortical_indices'],int)
        spatial._validate_mapping(names,ix,list(spatial.TARGETS))
        receptor=d['receptors68x2']; original=d['original_loading82']
        if receptor.shape!=(68,2) or original.shape!=(82,):raise ValueError('TLE: two fixed target maps and original 82D loading required')
        z=spatial.nuisance_design(d['ap68'],d['g168']);w=spatial.common_graph(d['sphere_coords68x3'])
        targets=np.column_stack([original[ix],receptor])
        result,arrays=spatial_tle.analyze(d['raw_roi_means'],names,ix,receptor,z,w,original,targets,not a.effect_only)
    out=new_output(a.output)
    write_json(out/'external_results.json',result)
    np.savez_compressed(out/'external_arrays.npz',**arrays)


if __name__=='__main__':main()
