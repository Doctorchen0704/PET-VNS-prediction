"""STEP06: run one declared complete HC3/BH association family at a time."""
from pathlib import Path
import argparse
import hashlib
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_support import read_json,write_json,new_output,load_model_input
from STEP05_clinical_data_assembly.table_io import clinical_design,metadata_arrays
from pet_vns_prediction import associations,sensitivity


def run_family(data,x,y,settings,metadata,family,processing=None):
    extended=family in ('clinical18','followup114')
    _,scanner,types,etiologies=metadata_arrays(metadata,list(x.index),extended=extended)
    clinical=clinical_design(x)
    values=x.loc[:,list(settings.pet)].to_numpy(float)
    t1=x.loc[:,list(settings.t1)].to_numpy(float)
    names=list(settings.pet)
    if family=='primary6':
        scaled=clinical.copy()
        for j in (0,2,3):
            sd=scaled[:,j].std(ddof=1)
            if not np.isfinite(sd) or sd<=0:raise ValueError('Nonpositive clinical SD')
            scaled[:,j]=(scaled[:,j]-scaled[:,j].mean())/sd
        return {'rows':associations.association_family(values,y.to_numpy(),
            {'clinical5_scanner':np.c_[scaled,scanner]},region_names=names,expected_tests=6),
            'family_tests':6}
    if family=='morphometric12':
        return {'rows':sensitivity.clinical_morphometric_family(values,y,clinical,scanner,t1,region_names=names),
                'family_tests':12}
    if family=='clinical18':
        return {'rows':sensitivity.extended_clinical_family(values,y,clinical,scanner,types,etiologies,region_names=names),
                'family_tests':18}
    if processing is None or processing.get('study_ids')!=list(x.index):
        raise ValueError('Complete processing payload with identical study_ids order required')
    variants=processing['variants']
    if not np.array_equal(np.asarray(variants.get('original'),float),values):
        raise ValueError('Original processing matrix must exactly match primary model PET values')
    if family=='processing96':
        return sensitivity.processing_association_family(variants,y,clinical,scanner,t1,region_names=names)
    return sensitivity.followup_association_families(variants,y,clinical,scanner,t1,types,etiologies,region_names=names)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True,help='STEP05 model_input.json')
    ap.add_argument('--metadata',type=Path,required=True)
    ap.add_argument('--family',choices=['primary6','morphometric12','clinical18','processing96','followup114'],required=True)
    ap.add_argument('--processing-features',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    data,x,y,settings=load_model_input(a.input)
    payload=None if a.processing_features is None else read_json(a.processing_features)
    result=run_family(data,x,y,settings,a.metadata,a.family,payload)
    result.update(participant_ids=list(x.index),region_order=list(settings.pet),
                  source_input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest())
    out=new_output(a.output)
    write_json(out/'associations.json',result)
    import pandas as pd
    pd.DataFrame(result['rows']).to_csv(out/'associations.csv',index=False)
    print('STEP06 complete:',out)


if __name__=='__main__':main()
