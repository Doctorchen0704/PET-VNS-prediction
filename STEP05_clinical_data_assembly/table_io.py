"""Explicit keyed table assembly; no eligibility decisions or patient matching."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from pet_vns_prediction import core, paper_settings


def keyed_csv(path):
    path = Path(path)
    table = pd.read_csv(path, sep='\t' if path.suffix.lower()=='.tsv' else ',',
                        dtype={'study_id':str}, keep_default_na=True,
                        float_precision='round_trip')
    if 'study_id' not in table or table['study_id'].isna().any():
        raise ValueError('A complete study_id column is required: '+str(path))
    if table['study_id'].duplicated().any() or not table['study_id'].str.fullmatch(r'[A-Za-z0-9]+').all():
        raise ValueError('Unique simple study_id keys required')
    return table.set_index('study_id', verify_integrity=True)


def align(table, ids):
    if len(ids)!=len(set(ids)) or not ids:
        raise ValueError('Frozen cohort order must be nonempty and unique')
    missing = set(ids)-set(table.index)
    if missing:
        raise ValueError('Required cohort rows missing; do not drop them: '+str(sorted(missing)))
    return table.loc[ids].copy()


def binary(values, name):
    numbers = pd.to_numeric(values, errors='raise')
    if numbers.isna().any() or not numbers.isin([0,1]).all():
        raise ValueError(name+' must contain complete explicit 0/1 values')
    return numbers.astype(int)


def assemble(cohort, clinical, pet, t1, plans=None, *, generate_splits=False, preserve_frozen_log=False):
    settings = paper_settings()
    ids = list(cohort.index)
    c,p,t = [align(frame,ids) for frame in (clinical,pet,t1)]
    required = ['age_at_implant_years','sex','epilepsy_duration_reported_years',
                'baseline_monthly_frequency','mri_lesional_status','responder_12m']
    if not set(required).issubset(c):
        raise ValueError('Clinical columns missing: '+str(sorted(set(required)-set(c))))
    if not set(settings.pet).issubset(p) or not set(settings.t1).issubset(t):
        raise ValueError('All six canonical PET and six canonical T1 columns are required')
    x = pd.DataFrame(index=ids)
    for field in ('age_at_implant_years','epilepsy_duration_reported_years'):
        x[field] = pd.to_numeric(c[field],errors='raise')
    for field in ('sex','mri_lesional_status'):
        x[field] = c[field]
    frequency = pd.to_numeric(c['baseline_monthly_frequency'],errors='raise')
    if (frequency.dropna()<0).any() or np.isinf(frequency.to_numpy(dtype=float)).any():
        raise ValueError('Baseline monthly frequency must be nonnegative and finite or explicitly missing')
    # The executed clinical input preparation used Python math.log1p per row.
    x['log1p_baseline_monthly_frequency'] = [math.log1p(float(v)) if pd.notna(v) else np.nan for v in frequency]
    if preserve_frozen_log:
        name='log1p_baseline_monthly_frequency'
        if name not in c:
            raise ValueError('Preserving frozen log requires an explicitly supplied derived column')
        frozen=pd.to_numeric(c[name],errors='raise')
        derived=x[name].to_numpy(float)
        values=frozen.to_numpy(float)
        if not np.array_equal(np.isnan(derived),np.isnan(values)) or np.isinf(values).any():
            raise ValueError('Frozen log and raw frequency missingness differs')
        observed=np.isfinite(derived)
        if np.any(np.abs(derived[observed]-values[observed])>1e-12):
            raise ValueError('Frozen log conflicts with raw monthly frequency')
        x[name]=frozen
    for frame,fields in ((p,settings.pet),(t,settings.t1)):
        for field in fields:
            x[field] = pd.to_numeric(frame[field],errors='raise')
    x = x.loc[:,list(settings.model_columns()['CPT'])]
    y = binary(c['responder_12m'],'responder_12m')
    y.index = x.index
    core.validate_inputs(x,y,settings,settings.model_columns()['CPT'])
    if generate_splits and plans is not None:
        raise ValueError('Supply saved splits OR explicitly generate new splits, not both')
    if plans is None:
        if not generate_splits:
            raise ValueError('Exact saved split plan required for reproducing the study')
        plans = core.make_split_plan(y,settings)
    core.validate_split_plan(plans,y,settings.repeats,settings.inner_folds)
    rows = x.astype(object).where(pd.notna(x),None).to_dict(orient='records')
    payload={'study_ids':ids,'x':rows,'y':y.tolist(),
            'columns':{k:list(v) for k,v in settings.model_columns().items()},
            'plans':plans,
            'split_source':'newly_generated_not_claimed_historical' if generate_splits else 'supplied_frozen_plan',
            'log_frequency_source':'preserved_frozen_derived_precision_after_raw_check' if preserve_frozen_log else 'math.log1p_of_reported_monthly_frequency'}
    # Apply the same strict integer/range/fold-key checks used by STEP07 before
    # writing a new analysis input. This performs no model fit or file writes.
    from pet_vns_prediction.cli import _validated_input
    _validated_input(payload,['C','CP','CPT'],settings)
    return payload,x,y


def clinical_design(x):
    settings = paper_settings()
    for field,allowed in settings.categories.items():
        if x[field].isna().any() or not set(x[field]).issubset(allowed):
            raise ValueError('Complete declared categories required for association: '+field)
    result=np.c_[x['age_at_implant_years'].to_numpy(float),
                 (x['sex']=='male').to_numpy(float),
                 x['epilepsy_duration_reported_years'].to_numpy(float),
                 x['log1p_baseline_monthly_frequency'].to_numpy(float),
                 (x['mri_lesional_status']=='lesional').to_numpy(float)]
    if not np.isfinite(result).all():
        raise ValueError('Association design must be complete; no automatic imputation or case deletion')
    return result


def metadata_arrays(path, ids, *, extended=False):
    table=align(keyed_csv(path),ids)
    if 'scanner_indicator' not in table:
        raise ValueError('Explicit scanner_indicator and its reference documentation required')
    scanner=binary(table['scanner_indicator'],'scanner_indicator').to_numpy()
    if not extended:
        return table,scanner,None,None
    allowed_type={'generalized','focal','combined_generalized_and_focal'}
    allowed_etiology={'unknown','structural','other_known'}
    if 'epilepsy_type' not in table or table['epilepsy_type'].isna().any() or not set(table['epilepsy_type']).issubset(allowed_type):
        raise ValueError('Fixed three-level epilepsy_type is incomplete or invalid')
    if 'etiology_group' not in table or table['etiology_group'].isna().any() or not set(table['etiology_group']).issubset(allowed_etiology):
        raise ValueError('Fixed three-level etiology_group is incomplete or invalid')
    types=np.c_[(table['epilepsy_type']=='focal').to_numpy(float),
                (table['epilepsy_type']=='combined_generalized_and_focal').to_numpy(float)]
    etiologies=np.c_[(table['etiology_group']=='structural').to_numpy(float),
                     (table['etiology_group']=='other_known').to_numpy(float)]
    return table,scanner,types,etiologies
