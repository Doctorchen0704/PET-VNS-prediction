"""PC1 association with epilepsy type, six PET composites and response; no AUC."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from step_support import write_json,new_output
from clinical_bridge_methods import type_result,response_result,correlations,ols,logit_fit,REGIONS
import numpy as np


def exact_duplicate_scores(clr,idx,loading,score_sd):
    # Project each original person once, then index. Repeated occurrences of the
    # same person have identical values before rankdata; distinct people are not rounded.
    return ((clr-clr[idx].mean(axis=0))@loading/score_sd)[idx]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='NPZ with fixed clinical design matrices and original PC1_z')
    p.add_argument('--bootstrap-input',type=Path,help='NPZ clr82,row_indices,pc1_loadings,score_sd; explicit optional 2000-draw association replay')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    score=d['PC1_z'];y=d['y'];n=len(score)
    if score.shape!=(n,) or y.shape!=(n,) or not np.isin(y,[0,1]).all() or not np.isclose(score.std(ddof=1),1):
        raise ValueError('Binary outcomes and original sample-SD-standardized PC1 score required')
    tm={'primary':d['type_primary'],'MRI_ASM_sensitivity':d['type_MRI_ASM']}
    rm={'primary':d['response_primary'],'plus_type_sensitivity':d['response_plus_type'],
        'original_clinical5_sensitivity':d['response_original5']}
    for x in [*tm.values(),*rm.values()]:
        if x.ndim!=2 or len(x)!=n or not np.isfinite(x).all():raise ValueError('Complete fixed design matrices required')
    if d['pet6'].shape!=(n,6) or not np.isfinite(d['pet6']).all():raise ValueError('Fixed 6-region input required')
    for x in rm.values():
        if not np.array_equal(x[:,4],score):raise ValueError('Response design PC1 column/order differs')
    types={k:type_result(score,x) for k,x in tm.items()};responses={k:response_result(y,x) for k,x in rm.items()}
    ps=np.array([types['primary']['p_raw'],responses['primary']['p_raw']]);order=np.argsort(ps)
    q=np.empty(2);q[order]=np.minimum(1,np.maximum.accumulate(ps[order]*[2,1]))
    types['primary']['p_Holm_two_primary']=float(q[0]);responses['primary']['p_Holm_two_primary']=float(q[1])
    rho=correlations(score,d['pet6'],d['age'],d['female'],d['scanner'])
    corrs=[{'region':name,'raw_Spearman':float(rho[0,j]),'adjusted_partial_Spearman':float(rho[1,j])} for j,name in enumerate(REGIONS)]
    arrays={};reasons={}
    if a.bootstrap_input:
        with np.load(a.bootstrap_input,allow_pickle=False) as f:b={k:f[k] for k in f.files}
        idxs=b['row_indices'];loads=b['pc1_loadings'];clr=b['clr82'];sd=float(b['score_sd'])
        if idxs.shape!=(2000,n) or loads.shape!=(2000,82) or clr.shape!=(n,82) or sd<=0:
            raise ValueError('All fixed 2000 participant/PCA resamples and original score SD required')
        if not np.issubdtype(idxs.dtype,np.integer) or np.any((idxs<0)|(idxs>=n)):raise ValueError('Bootstrap participant indices invalid')
        bt=np.full((2000,2,3),np.nan);br=np.full((2000,3),np.nan);bc=np.full((2000,2,6),np.nan)
        for k,(idx,loading) in enumerate(zip(idxs,loads)):
            s=(clr[idx]-clr[idx].mean(axis=0))@loading/sd
            for j,x in enumerate(tm.values()):
                try:
                    beta=ols(s,x[idx])[0];bt[k,j]=[beta[4],beta[5],beta[4]-beta[5]]
                except (ValueError,np.linalg.LinAlgError) as e:reasons['type'+str(j)+':'+str(e)]=reasons.get('type'+str(j)+':'+str(e),0)+1
            for j,x in enumerate(rm.values()):
                xx=x[idx].copy();xx[:,4]=s
                try:br[k,j]=logit_fit(xx,y[idx])[0][4]
                except (ValueError,np.linalg.LinAlgError) as e:reasons['response'+str(j)+':'+str(e)]=reasons.get('response'+str(j)+':'+str(e),0)+1
            exact=exact_duplicate_scores(clr,idx,loading,sd)
            bc[k]=correlations(exact,d['pet6'][idx],d['age'][idx],d['female'][idx],d['scanner'][idx])
        for j,result in enumerate(types.values()):
            valid=np.isfinite(bt[:,j]).all(axis=1)
            if not valid.any():raise ValueError('No estimable type bootstrap draw; retain failed output for review')
            result['PCA_bootstrap']={'valid_n':int(valid.sum()),'total_n':2000,
                'interval_scope':'percentile conditional on valid fits' if valid.mean()<.95 else 'percentile participant resampling with reestimated PCA',
                'contrast_ci95':np.quantile(bt[valid,j],[.025,.975],axis=0).T.tolist()}
        for j,result in enumerate(responses.values()):
            valid=np.isfinite(br[:,j])
            if not valid.any():raise ValueError('No estimable response bootstrap draw')
            ci=np.quantile(br[valid,j],[.025,.975]);result['PCA_bootstrap']={'valid_n':int(valid.sum()),'total_n':2000,'log_OR_ci95':ci.tolist(),'OR_ci95':np.exp(ci).tolist(),
                'interval_scope':'diagnostic only; >5% invalid fits' if valid.mean()<.95 else 'percentile participant resampling with reestimated PCA'}
        for j,row in enumerate(corrs):
            row['raw_PCA_bootstrap_ci95']=np.quantile(bc[:,0,j],[.025,.975]).tolist()
            row['adjusted_PCA_bootstrap_ci95']=np.quantile(bc[:,1,j],[.025,.975]).tolist()
        arrays={'type_contrasts':bt,'response_logOR':br,'six_region_correlations_exact_ties':bc}
    out=new_output(a.output);write_json(out/'clinical_bridge_results.json',{'n':n,'epilepsy_type':types,'response':responses,
        'six_region_bridge':corrs,'bootstrap_failures':reasons,'multiplicity':'Holm two primary clinical tests; six-region associations descriptive without P',
        'no_auc_computed':True,'exact_duplicate_rank_tie_correction':True})
    if arrays:np.savez_compressed(out/'clinical_bridge_bootstrap.npz',**arrays)


if __name__=='__main__':main()
