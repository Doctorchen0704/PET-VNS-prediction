"""Final deterministic nested-CV numerical hook, with explicit in-memory inputs.

The historical filesystem runner and import-time global monkeypatch are omitted.
core.fit_logistic delegates here explicitly. Fits remain serial, as in the study.
"""
import numpy as np
from scipy.special import expit
from . import core, solver

ACTIVE_FITS = None


def corrected_fit(x,y,settings,c,l1,seed):
    try:
        fitted=solver.fit(x,np.asarray(y,int),c,l1)
    except solver.NumericalFailure as exc:
        error=core.FitFailure(str(exc))
        error.failed_candidate={'C':float(c),'l1_ratio':float(l1),'training_n':len(y),'preserved_seed_argument':int(seed),
                                'solver_diagnostics':exc.diagnostics}
        raise error from exc
    if ACTIVE_FITS is not None:
        ACTIVE_FITS.append({'fit_slot':len(ACTIVE_FITS),'preserved_seed_argument':int(seed),
                            'iterations':int(fitted.n_iter_[0]),**fitted.numerical_})
    return fitted,{'retried':False,'iterations':int(fitted.n_iter_[0])}


def corrected_outer(x,y,columns,settings,plan,model,audit,deadline=lambda:None):
    global ACTIVE_FITS
    if ACTIVE_FITS is not None:
        raise RuntimeError('Nested numerical collection is not permitted.')
    fits=[]
    ACTIVE_FITS=fits
    try:
        prediction,info=core.run_outer_model(x,y,columns,settings,plan,model,audit,deadline)
    except Exception as exc:
        exc.partial_numerical_fits=fits
        if hasattr(exc,'failed_candidate'):
            exc.failed_candidate['fit_slot']=len(fits)
            inner_fit_count=len(settings.c_grid)*len(settings.l1_grid)*settings.inner_folds
            exc.failed_candidate['stage']='inner' if len(fits)<inner_fit_count else 'outer_refit'
            exc.failed_candidate['inner_fold_zero_based']=len(fits)%settings.inner_folds if len(fits)<inner_fit_count else None
        raise
    finally:
        ACTIVE_FITS=None
    expected=len(settings.c_grid)*len(settings.l1_grid)*settings.inner_folds+1
    if len(fits)!=expected:
        raise AssertionError('Incomplete or reordered numerical candidate grid.')
    selected=info['selected']
    transformer=core.FoldPreprocessor(columns,settings.categories).fit(x.iloc[plan['train']])
    train_x=transformer.transform(x.iloc[plan['train']])
    test_x=transformer.transform(x.iloc[plan['test']])
    train_y=np.asarray(y.iloc[plan['train']],int)
    deadline()
    repeated=solver.fit(train_x,train_y,selected['C'],selected['l1_ratio'])
    last=fits[-1]
    exact=(np.array_equal(repeated.coef_[0],last['coefficients'])
           and repeated.intercept_[0]==last['intercept']
           and np.array_equal(repeated.predict_proba(test_x)[:,1],prediction))
    if not exact:
        raise core.FitFailure('Selected outer fit did not reproduce exactly.')
    w,independent=solver.independent_slsqp(train_x,train_y,selected['C'],selected['l1_ratio'])
    independent_p=expit(w[0]+test_x@w[1:])
    training_difference=float(np.max(np.abs(expit(w[0]+train_x@w[1:])-repeated.predict_proba(train_x)[:,1])))
    probability_difference=float(np.max(np.abs(independent_p-prediction)))
    objective_difference=abs(independent['objective_mean']-last['objective_mean'])
    if (not independent['success'] or max(probability_difference,training_difference)>1e-5 or objective_difference>1e-10):
        raise core.FitFailure(f'Independent outer-fit agreement failed: {independent}; prediction difference {probability_difference}; objective difference {objective_difference}')
    info.update(numerical_fits=fits,exact_selected_fit_repetition=True,
                independent_selected={'solver':'independent_epigraph_SLSQP','intercept':float(w[0]),
                    'coefficients':w[1:].tolist(),'training_probability_max_difference':training_difference,
                    'held_out_probability_max_difference':probability_difference,
                    'objective_max_difference':objective_difference,**independent})
    return prediction,info
