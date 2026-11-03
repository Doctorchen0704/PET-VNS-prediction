"""Explicit local-input command line; never downloads or discovers study data."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
from . import core, paper_settings, run_nested_models
from .intake import convert_dicom, promote_selected, new_output, sha256_file, load_json


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serial, allow_nan=False)+'\n',encoding='utf-8')


def environment():
    packages={}
    for name in ('numpy','pandas','scipy','scikit-learn','threadpoolctl','nibabel','matplotlib','Pillow'):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    return {'python':sys.version,'platform':platform.platform(),'packages':packages}


def _validated_input(data, models, settings):
    """Validate input even for saved-prediction summaries, before creating output."""
    if not isinstance(data, dict) or not isinstance(models, (list, tuple)) or not models:
        raise ValueError('Input object and nonempty model list required')
    if len(set(models)) != len(models):
        raise ValueError('Do not request the same model more than once')
    ids = data['study_ids']
    if not isinstance(ids, list) or len(ids) < 2 or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError('Ordered nonempty string study IDs required')
    if len(set(ids)) != len(ids):
        raise ValueError('Unique ordered study IDs required')
    outcome = data['y']
    if not isinstance(outcome, list) or len(outcome) != len(ids) or any(type(v) not in (int, float) or v not in (0, 1) for v in outcome):
        raise ValueError('One explicit numeric 0/1 outcome per ordered study ID required')
    if not isinstance(data['x'], (dict, list)):
        raise ValueError('Predictors must be an explicit column mapping or row records')
    x = pd.DataFrame(data['x'], index=ids)
    y = pd.Series(outcome, index=ids)
    expected = settings.model_columns()
    for name in models:
        if name not in expected or list(data['columns'][name]) != list(expected[name]):
            raise ValueError('Frozen model specification differs from paper settings')
        core.validate_inputs(x, y, settings, expected[name])
    outer_folds = core.choose_outer_folds(y)
    plans = data['plans']
    if not isinstance(plans, list) or len(plans) != settings.repeats * outer_folds:
        raise ValueError('Complete frozen repeated outer split plan required')
    seen = set()
    for plan in plans:
        if not isinstance(plan, dict):
            raise ValueError('Split plan must contain objects')
        repeat, fold = plan['repeat'], plan['fold']
        if type(repeat) is not int or not 0 <= repeat < settings.repeats or type(fold) is not int or not 0 <= fold < outer_folds:
            raise ValueError('Repeat/fold must be nonnegative in-range integer indices')
        if (repeat, fold) in seen:
            raise ValueError('Repeated outer repeat/fold identifier')
        seen.add((repeat, fold))
        index_lists = [plan['train'], plan['test']]
        if not isinstance(plan['inner'], list):
            raise ValueError('Inner splits must be a list')
        for pair in plan['inner']:
            index_lists.extend([pair['train'], pair['validation']])
        for rows in index_lists:
            if not isinstance(rows, list) or not rows or any(type(i) is not int or not 0 <= i < len(ids) for i in rows):
                raise ValueError('Split rows must be nonempty in-range integer lists')
    core.validate_split_plan(plans, y, settings.repeats, settings.inner_folds)
    return ids, x, y, expected, plans


def predict(input_path, output_path, models=('C','CP','CPT'), timeout=7200, summary_only=False):
    """Final frozen model-input JSON schema; uses its saved folds and row order.

    --summary-only reads saved predictions supplied inside the input JSON and
    does NOT fit. A refit is the default and has its own new output directory.
    """
    source=Path(input_path).resolve()
    data=load_json(source)
    settings=paper_settings()
    ids,x,y,expected,plans=_validated_input(data,models,settings)
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('Positive finite timeout required')
    preds = None
    if summary_only:
        saved = data.get('saved_predictions')
        if not isinstance(saved, dict):
            raise ValueError('Saved prediction arrays must be supplied for summary-only')
        preds={m:np.asarray(saved[m],float) for m in models}
        if any(p.shape!=(settings.repeats,len(ids)) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) for p in preds.values()):
            raise ValueError('Saved prediction shape/range mismatch')
    out=new_output(output_path)
    if source==out or out in source.parents: raise ValueError('Output cannot contain the source')
    out.mkdir(parents=True)
    status={'status':'RUNNING','input_sha256':sha256_file(source),
            'method':'saved_prediction_summary' if summary_only else 'full_refit',
            'model_fit_performed':False,
            'saved_predictions_training_provenance':'NOT_REVERIFIED' if summary_only else 'FIT_AUDIT_WILL_BE_RECORDED',
            'clinical_or_imaging_QC_performed':False,
            'environment':environment(),'models':list(models)}
    write(out/'status.json',status)
    try:
        if summary_only:
            result={'status':'PASS','predictions':preds,'participant_ids':ids,'plans':plans}
        else:
            def progress(p):
                status.update(p); write(out/'status.json',status)
                print(json.dumps(p),flush=True)
            result=run_nested_models(x,y,settings,model_columns={m:expected[m] for m in models},
                                     plans=plans,timeout_seconds=timeout,progress=progress)
            status['model_fit_performed']=True
            write(out/'fit_audit_PRIVATE.json',{'audit':result['audit'],'tuning':result['tuning'],
                  'split_digest':result['split_digest'],'model_columns':result['model_columns']})
        write(out/'predictions_PRIVATE.json',{'study_ids':ids,'predictions':result['predictions']})
        performance=core.summarize_predictions(result,y,settings)
        write(out/'performance.json',performance)
        if len(models)>1:
            # The original API requires the standard comparisons; keep exact
            # primary bootstrap only when all three primary models are present.
            if set(models)=={'C','CP','CPT'}:
                write(out/'conditional_bootstrap.json',core.paired_bootstrap(result,y))
        status.update(status='COMPLETE',n=len(ids),responders=int(y.sum()))
        write(out/'status.json',status)
        return performance
    except BaseException as exc:
        status.update(status='FAILED_PRESERVED',error_type=type(exc).__name__)
        write(out/'status.json',status)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('environment')
    p=sub.add_parser('predict');p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--models',nargs='+',choices=['C','CP','CPT'],default=['C','CP','CPT'])
    p.add_argument('--timeout-seconds',type=float,default=7200);p.add_argument('--summary-only',action='store_true')
    p=sub.add_parser('convert-dicom');p.add_argument('--manifest',required=True);p.add_argument('--raw-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--converter',required=True);p.add_argument('--execute',action='store_true')
    p=sub.add_parser('promote-bids');p.add_argument('--manifest',required=True);p.add_argument('--staging-root',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.action=='environment': result=environment()
    elif args.action=='predict': result=predict(args.input,args.output,args.models,args.timeout_seconds,args.summary_only)
    else:
        manifest=load_json(args.manifest)
        if args.action=='convert-dicom': result=convert_dicom(manifest,args.raw_root,args.output,args.converter,execute=args.execute)
        else: result={'promoted_files':len(promote_selected(manifest,args.staging_root,args.output))}
    print(json.dumps(result,default=serial,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__': main()
