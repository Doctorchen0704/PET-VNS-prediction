"""Source-retained PC1 clinical-association numerical functions; no file discovery."""
import numpy as np
from scipy import stats, optimize
from scipy.special import expit
GROUPS = ["generalized", "focal", "combined_generalized_and_focal"]
REGIONS = ["thalamus", "cingulate", "insula_frontal_operculum", "limbic_medial_temporal", "striatopallidal", "sensorimotor"]

def ols(y, x):
    n, p = x.shape
    if np.linalg.matrix_rank(x) < p:
        raise ValueError('rank deficient')
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    inv = np.linalg.inv(x.T@x)
    residual = y-x@b
    leverage = np.einsum('ij,jk,ik->i', x, inv, x)
    if leverage.max() >= 1-1e-10:
        raise ValueError('unit leverage')
    cov = inv @ (x.T @ (x*((residual/(1-leverage))**2)[:, None])) @ inv
    return b, cov, residual, leverage

def type_result(y, x):
    b, cov, residual, lev = ols(y, x)
    df = len(y)-x.shape[1]
    effect = b[4:6]; c = cov[4:6, 4:6]
    f = float(effect@np.linalg.solve(c, effect)/2)
    reduced = np.delete(x, [4, 5], axis=1)
    resid0 = y-reduced@np.linalg.lstsq(reduced, y, rcond=None)[0]
    eta = float(1-(residual@residual)/(resid0@resid0))
    critical = stats.t.ppf(.975, df)
    means, contrasts = [], []
    designs = []
    for i, g in enumerate(GROUPS):
        xx = x.copy(); xx[:, 4:6] = 0
        if i: xx[:, 3+i] = 1
        v = xx.mean(axis=0); designs.append(v)
        se = np.sqrt(v@cov@v)
        means.append({'group': g, 'n': int((x[:, 4+i-1] == 1).sum()) if i else int(((x[:, 4]+x[:, 5]) == 0).sum()),
                      'adjusted_mean_pc1_sd': float(v@b),
                      'hc3_ci95': [float(v@b-critical*se), float(v@b+critical*se)]})
    for left, right in [(1,0), (2,0), (1,2)]:
        v = designs[left]-designs[right]; est = float(v@b); se = float(np.sqrt(v@cov@v))
        contrasts.append({'contrast': GROUPS[left]+' minus '+GROUPS[right], 'difference_pc1_sd': est,
                          'hc3_nominal_ci95': [est-critical*se, est+critical*se]})
    return {'overall_HC3_F': f, 'df': [2, df], 'p_raw': float(stats.f.sf(f, 2, df)),
            'partial_R2_type': eta, 'adjusted_means': means, 'contrasts': contrasts,
            'diagnostics': {'rank': int(np.linalg.matrix_rank(x)), 'condition_number': float(np.linalg.cond(x)),
                            'max_leverage': float(lev.max()), 'residual_skewness': float(stats.skew(residual))},
            'coefficients': b.tolist(), 'HC3_covariance': cov.tolist()}

def nll(beta, x, y, offset=0.):
    eta = x@beta+offset
    return float(np.sum(np.logaddexp(0., eta)-y*eta))

def logit_fit(x, y, offset=0., start=None):
    p = x.shape[1]
    if np.linalg.matrix_rank(x) < p or len(np.unique(y)) < 2:
        raise ValueError('rank or one outcome class')
    b = np.zeros(p) if start is None else np.array(start, float).copy()
    for iteration in range(100):
        mu = expit(x@b+offset)
        gradient = x.T@(mu-y)
        w = mu*(1-mu)
        info = x.T@(x*w[:, None])
        if not np.isfinite(info).all() or np.linalg.cond(info) > 1e12 or np.max(abs(b)) > 40:
            raise ValueError('separation or ill-conditioned information')
        if np.max(abs(gradient)) < 1e-8:
            return b, nll(b, x, y, offset), info, iteration, float(np.max(abs(gradient)))
        step = np.linalg.solve(info, gradient)
        old = nll(b, x, y, offset); scale = 1.
        while scale > 2**-25 and nll(b-scale*step, x, y, offset) > old-1e-4*scale*(gradient@step):
            scale *= .5
        if scale <= 2**-25:
            # Floating-point plateau is acceptable only with a strict score check.
            if np.max(abs(gradient)) < 1e-6:
                return b, old, info, iteration, float(np.max(abs(gradient)))
            raise ValueError('line search stalled')
        b -= scale*step
    raise ValueError('maximum iterations')

def separation(x, y):
    signed = x*(2*y-1)[:, None]
    lp = optimize.linprog(-signed.sum(axis=0), A_ub=-signed, b_ub=np.zeros(len(y)),
                          bounds=[(-1, 1)]*x.shape[1], method='highs')
    assert lp.success
    return bool(-lp.fun > 1e-7), float(-lp.fun)

def response_result(y, x):
    separated, value = separation(x, y)
    if separated:
        return {'status': 'nonestimable_separation', 'separation_lp_objective': value, 'p_raw': 1.}
    b, loss, info, iterations, grad = logit_fit(x, y)
    reduced = np.delete(x, 4, axis=1)
    rb, loss0, _, _, _ = logit_fit(reduced, y)
    lr = max(0., 2*(loss0-loss)); cutoff = stats.chi2.ppf(.95, 1)
    nuisance = np.delete(b, 4)
    def equation(beta):
        _, constrained, _, _, _ = logit_fit(reduced, y, offset=x[:,4]*beta, start=nuisance)
        return 2*(constrained-loss)-cutoff
    bounds = []
    for direction in [-1, 1]:
        distance = .5
        for _ in range(14):
            end = b[4]+direction*distance
            if equation(end) >= 0:
                bounds.append(float(optimize.brentq(equation, min(b[4],end), max(b[4],end), xtol=1e-10)))
                break
            distance *= 2
        else:
            raise RuntimeError('profile endpoint not bracketed')
    return {'status': 'estimated', 'log_OR_per_original_PC1_SD': float(b[4]),
            'OR_per_original_PC1_SD': float(np.exp(b[4])), 'profile_logOR_ci95': bounds,
            'profile_OR_ci95': np.exp(bounds).tolist(), 'LR_chi2': float(lr),
            'p_raw': float(stats.chi2.sf(lr, 1)), 'coefficients': b.tolist(),
            'information': info.tolist(), 'negative_loglikelihood': loss,
            'diagnostics': {'separation_lp_objective': value, 'rank': int(np.linalg.matrix_rank(x)),
                            'condition_number': float(np.linalg.cond(x)), 'information_condition': float(np.linalg.cond(info)),
                            'max_abs_score': grad, 'iterations': iterations,
                            'fitted_probability_range': [float(expit(x@b).min()), float(expit(x@b).max())]}}

def correlations(score, pet, age, female, scanner):
    ranks = stats.rankdata(np.column_stack([score, pet]), axis=0)
    design = np.column_stack([np.ones(len(score)), stats.rankdata(age), female, scanner])
    residual = ranks-design@np.linalg.lstsq(design, ranks, rcond=None)[0]
    raw = np.corrcoef(ranks.T)[0,1:]
    partial = np.corrcoef(residual.T)[0,1:]
    return np.array([raw, partial])


