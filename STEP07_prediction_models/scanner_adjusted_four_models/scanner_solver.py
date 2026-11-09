"""Isolated extension of the frozen elastic-net solver: unpenalized nuisance.

For binary penalty_mask m (scanner=0, all original predictors=1), minimize
  mean(binary logistic loss) + a * sum(m*abs(beta))
                            + b/2 * sum(m*beta**2),
where a=l1_ratio/(C*n), b=(1-l1_ratio)/(C*n). Intercept is unpenalized.
No feature penalty rescaling, patient I/O, model selection or randomness.
Original numerical acceptance thresholds are retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, xlogy

TARGET_KKT = 1e-11
ACCEPT_KKT = 1e-8
ACCEPT_GAP = 1e-10
MAX_LBFGS = 5000
MAX_POLISH = 80
NUISANCE_EQUALITY_TOL = 1e-12


class NumericalFailure(RuntimeError):
    def __init__(self, message, **diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def soft(v, threshold):
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.)


def objective(x, y, w, l1, l2):
    """l1/l2 are per-column penalty vectors, not globally rescaled factors."""
    z = w[0] + x @ w[1:]
    loss = np.mean(np.logaddexp(0., np.where(y == 1, -z, z)))
    return float(loss + np.dot(l1, np.abs(w[1:])) + .5*np.dot(l2, w[1:]**2))


def gradient(x, y, w, l2):
    error = expit(w[0] + x @ w[1:]) - y
    return np.r_[error.mean(), x.T @ error / len(y) + l2*w[1:]]


def residual(g, w, l1):
    r = np.r_[g[0], np.where(w[1:] != 0, g[1:] + l1*np.sign(w[1:]), soft(g[1:], l1))]
    return float(np.max(np.abs(r)))


def optimize_nuisance(x, y, w, penalty_mask):
    """Newton refinement of all zero-penalty coefficients and the intercept.

    It solves nuisance score equations, retaining the penalized coefficients.
    Fixed damping provides monotonic logistic loss, not a different model.
    """
    free = np.r_[True, penalty_mask == 0]
    design = np.column_stack((np.ones(len(y)), x))
    zfree = design[:, free]
    w = w.copy()
    for iteration in range(30):
        eta = design @ w
        p = expit(eta)
        score = zfree.T @ (p-y) / len(y)
        if float(np.max(np.abs(score))) < 2e-16:
            return w
        h = zfree.T @ ((p*(1-p))[:, None]*zfree) / len(y)
        try:
            direction = -np.linalg.solve(h, score)
        except np.linalg.LinAlgError as exc:
            raise NumericalFailure('Degenerate nuisance information.', iteration=iteration) from exc
        if not np.isfinite(direction).all():
            raise NumericalFailure('Nonfinite nuisance Newton direction.')
        before = float(np.mean(np.logaddexp(0., np.where(y == 1, -eta, eta))))
        step = 1.
        for _ in range(60):
            candidate = w.copy()
            candidate[free] += step*direction
            eta_new = design @ candidate
            after = float(np.mean(np.logaddexp(0., np.where(y == 1, -eta_new, eta_new))))
            new_score = zfree.T @ (expit(eta_new)-y) / len(y)
            if (after <= before + 1e-4*step*float(score@direction)
                    or (after <= before+2e-15 and np.max(np.abs(new_score)) < np.max(np.abs(score)))):
                w = candidate
                break
            step *= .5
        else:
            if np.max(np.abs(score)) <= NUISANCE_EQUALITY_TOL/10:
                return w
            raise NumericalFailure('Nuisance Newton line search stalled.', score=score.tolist())
    final_score = zfree.T @ (expit(design@w)-y) / len(y)
    if np.max(np.abs(final_score)) > NUISANCE_EQUALITY_TOL/10:
        raise NumericalFailure('Nuisance Newton limit exhausted.', score=final_score.tolist())
    return w


def certificate(x, y, w, l1, l2, penalty_mask):
    """Convex KKT/Fenchel certificate, with explicit nuisance dual equalities.

    Fenchel feasibility for zero penalties requires X_free.T@(q-y)=0, in
    addition to the intercept equality. Those equations are solved by nuisance
    refinement and independently required within 1e-12, not silently ignored.
    As in the frozen solver, the certificate is numerical (not symbolic).
    """
    g = gradient(x, y, w, l2)
    q = expit(w[0] + x @ w[1:])
    dual_score = x.T @ (y-q) / len(y)
    penalized = penalty_mask == 1
    free = ~penalized
    scale = 1.
    if np.any(l2 > 0):
        positive = l2 > 0
        conjugate = float(np.sum(soft(dual_score[positive], l1[positive])**2/(2*l2[positive])))
    else:
        positive_scores = np.abs(dual_score[penalized])
        nz = positive_scores > 0
        if np.any(nz):
            scale = min(1., float(np.min(l1[penalized][nz]/positive_scores[nz])))
        if scale < 1:
            scale *= 1.-8*np.finfo(float).eps
        q = y-scale*(y-q)
        conjugate = 0.
    entropy = -float(np.mean(xlogy(q,q) + xlogy(1-q,1-q)))
    primal = objective(x,y,w,l1,l2)
    eq = np.r_[np.mean(q-y), x[:,free].T@(q-y)/len(y)]
    return {'objective_mean':primal, 'dual_bound_mean':entropy-conjugate,
            'duality_gap_mean':primal-(entropy-conjugate), 'kkt_inf':residual(g,w,l1),
            'intercept_gradient':float(g[0]), 'dual_equality_error':float(np.max(np.abs(eq))),
            'nuisance_gradient_inf':float(np.max(np.abs(g[np.r_[True,free]]))),
            'dual_scale':float(scale), 'nonzero_coefficients':int(np.count_nonzero(w[1:]))}


def polish(x,y,w,l1,l2,penalty_mask):
    design = np.column_stack((np.ones(len(y)),x))
    full_l1 = np.r_[0.,l1]
    nonsmooth = full_l1 > 0
    for iteration in range(MAX_POLISH):
        w = optimize_nuisance(x,y,w,penalty_mask)
        g = gradient(x,y,w,l2)
        if nonsmooth.any():
            tiny = (np.abs(w[1:])<1e-12)&(np.abs(g[1:])<=l1)&(l1>0)
            w[1:][tiny] = 0.
            g = gradient(x,y,w,l2)
        r = residual(g,w,l1)
        if r <= TARGET_KKT:
            proof = certificate(x,y,w,l1,l2,penalty_mask)
            if (-1e-12 <= proof['duality_gap_mean'] <= ACCEPT_GAP
                    and proof['dual_equality_error'] <= NUISANCE_EQUALITY_TOL):
                return w,iteration
        p = expit(design@w)
        h = design.T@((p*(1-p))[:,None]*design)/len(y)
        h.flat[::len(w)+1] += np.r_[0.,l2]
        active = np.r_[True,(w[1:]!=0)|(np.abs(g[1:])>l1)|(l1==0)]
        signs = np.sign(w)
        signs[(w==0)&active] = -np.sign(g[(w==0)&active])
        direction = np.zeros(len(w))
        for _ in range(len(w)+1):
            restricted = h[np.ix_(active,active)]
            rhs = -(g+full_l1*signs)[active]
            try:
                step = np.linalg.solve(restricted,rhs)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(restricted,rhs,rcond=1e-14)[0]
            direction[:] = 0.; direction[active] = step
            wrong = (w==0)&active&(direction*signs<=0)&nonsmooth
            if not wrong.any():
                break
            active[wrong] = False
        directional = float(g@direction + np.sum(full_l1*np.where(w!=0,np.sign(w)*direction,np.abs(direction))))
        if not np.isfinite(direction).all() or directional >= 0:
            sub = np.r_[g[0],np.where(w[1:]!=0,g[1:]+l1*np.sign(w[1:]),soft(g[1:],l1))]
            j = int(np.argmax(np.abs(sub)))
            if h[j,j] <= 0:
                raise NumericalFailure('No valid coordinate curvature.',iteration=iteration,kkt=r)
            direction[:] = 0.
            direction[j] = float(soft(w[j]-g[j]/h[j,j],full_l1[j]/h[j,j]))-w[j]
        bound = 1.
        crossing = np.zeros(len(w),dtype=bool)
        candidates = np.flatnonzero((w!=0)&(w*direction<0)&nonsmooth)
        if candidates.size:
            ratios = -w[candidates]/direction[candidates]
            bound = min(1.,float(ratios.min()))
            crossing[candidates[ratios==bound]] = True
        target = w+bound*direction
        target[crossing] = 0.
        direction = target-w
        predicted = float(g@direction + np.dot(full_l1,np.abs(target)-np.abs(w)))
        old = objective(x,y,w,l1,l2)
        step_size = 1.
        for _ in range(60):
            candidate = target.copy() if step_size==1 else w+step_size*direction
            value = objective(x,y,candidate,l1,l2)
            nr = residual(gradient(x,y,candidate,l2),candidate,l1)
            if value <= old+1e-4*step_size*predicted or (value <= old+2e-15 and nr<r):
                w = candidate
                break
            step_size *= .5
        else:
            raise NumericalFailure('Newton polishing line search stalled.',iteration=iteration,kkt=r)
    raise NumericalFailure('Newton polishing exhausted fixed limit.',iteration=MAX_POLISH,kkt=r,
                           last_certificate=certificate(x,y,w,l1,l2,penalty_mask))


@dataclass
class FittedLogistic:
    coef_:np.ndarray
    intercept_:np.ndarray
    n_iter_:np.ndarray
    numerical_:dict

    @property
    def classes_(self):
        return np.array([0,1])

    def predict_proba(self,x):
        p = expit(self.intercept_[0]+np.asarray(x)@self.coef_[0])
        return np.column_stack((1-p,p))


def _validate(x,y,c,l1_ratio,penalty_mask):
    x,y = np.asarray(x,float),np.asarray(y,float)
    mask = np.asarray(penalty_mask,float)
    if (x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or set(np.unique(y))!={0.,1.}
            or not np.isfinite(c) or c<=0 or not 0<=l1_ratio<=1
            or mask.shape!=(x.shape[1],) or not np.isin(mask,[0.,1.]).all()):
        raise ValueError('Invalid input or binary per-feature penalty mask.')
    free = np.column_stack((np.ones(len(y)),x[:,mask==0]))
    if np.linalg.matrix_rank(free) != free.shape[1]:
        raise ValueError('Unpenalized design must have full rank; check scanner presence/encoding.')
    # For the approved binary scanner adjustment, a single-class scanner group
    # makes the intercept+scanner logistic optimum nonfinite. A tiny numerical
    # gradient along a diverging coefficient is NOT evidence of a finite MLE.
    # Reject before optimization; do not add a ridge term or clip coefficients.
    if np.count_nonzero(mask==0) == 1:
        nuisance = x[:,mask==0][:,0]
        levels = np.unique(nuisance)
        if len(levels) == 2:
            group_classes = [len(np.unique(y[nuisance==level])) for level in levels]
            if min(group_classes) < 2:
                raise NumericalFailure('Nonfinite optimum: binary unpenalized scanner has a single-class training group.',
                                       group_class_counts=group_classes,
                                       failure_type='unpenalized_scanner_separation')
    return x,y,mask


def fit(x,y,c,l1_ratio,penalty_mask):
    x,y,mask = _validate(x,y,c,l1_ratio,penalty_mask)
    n,d = x.shape
    original_x = x
    shift = x.mean(axis=0)
    x = x-shift
    a,b = l1_ratio/(c*n),(1-l1_ratio)/(c*n)
    l1,l2 = a*mask,b*mask
    prevalence = float(y.mean())
    start = np.r_[np.log(prevalence/(1-prevalence)),np.zeros(d)]
    start = optimize_nuisance(x,y,start,mask)
    g0 = gradient(x,y,start,l2)
    zero = bool(np.max(np.abs(g0[1:][mask==1])-l1[mask==1],initial=0.) <= 0)
    lbfgs_status = None
    iterations = polish_iterations = 0
    if zero:
        w = start
        method = 'analytical_nuisance_only_KKT'
    else:
        if a == 0:
            def fun(w):
                return objective(x,y,w,l1,l2),gradient(x,y,w,l2)
            result = minimize(fun,start,jac=True,method='L-BFGS-B',
                              options={'gtol':TARGET_KKT,'ftol':1e-15,'maxiter':MAX_LBFGS,'maxls':50,'maxcor':30})
            w = result.x
        else:
            initial = np.r_[start[0],np.maximum(start[1:],0.),np.maximum(-start[1:],0.)]
            def fun(v):
                w = np.r_[v[0],v[1:d+1]-v[d+1:]]
                smooth = objective(x,y,w,np.zeros(d),l2)
                g = gradient(x,y,w,l2)
                return smooth+np.dot(np.r_[l1,l1],v[1:]),np.r_[g[0],g[1:]+l1,-g[1:]+l1]
            result = minimize(fun,initial,jac=True,method='L-BFGS-B',
                              bounds=[(None,None)]+[(0.,None)]*(2*d),
                              options={'gtol':TARGET_KKT,'ftol':1e-15,'maxiter':MAX_LBFGS,'maxls':50,'maxcor':30})
            w = np.r_[result.x[0],result.x[1:d+1]-result.x[d+1:]]
        iterations = int(result.nit)
        lbfgs_status = {'status':int(result.status),'success':bool(result.success),'message':str(result.message)}
        w,polish_iterations = polish(x,y,w,l1,l2,mask)
        method = 'masked_split_LBFGSB_with_KKT_Newton_polish' if a else 'masked_LBFGSB_with_KKT_Newton_polish'
    w = optimize_nuisance(x,y,w,mask)
    w[0] -= shift@w[1:]
    cert = certificate(original_x,y,w,l1,l2,mask)
    if (not all(np.isfinite(v) for v in cert.values()) or cert['kkt_inf']>ACCEPT_KKT
            or abs(cert['intercept_gradient'])>1e-12 or cert['dual_equality_error']>NUISANCE_EQUALITY_TOL
            or cert['nuisance_gradient_inf']>NUISANCE_EQUALITY_TOL
            or cert['duality_gap_mean']>ACCEPT_GAP or cert['duality_gap_mean'] < -1e-12):
        raise NumericalFailure('Numerical certificate failed.',certificate=cert)
    cert.update(method=method,C=float(c),l1_ratio=float(l1_ratio),lambda1=a,lambda2=b,
                penalty_mask=mask.astype(int).tolist(),penalty_rescaling=False,
                training_n=n,training_prevalence=prevalence,analytical_zero=zero,
                lbfgs=lbfgs_status,polish_iterations=polish_iterations,
                coefficients=w[1:].tolist(),intercept=float(w[0]),
                internal_conditioning='training-X centering reversed in saved intercept; penalty unchanged')
    return FittedLogistic(w[None,1:],w[:1],np.array([iterations+polish_iterations]),cert)


def independent_slsqp(x,y,c,l1_ratio,penalty_mask):
    """Independent epigraph optimization, starts at zero; never a fallback fit."""
    x,y,mask = _validate(x,y,c,l1_ratio,penalty_mask)
    n,d = x.shape
    a,b = l1_ratio/(c*n),(1-l1_ratio)/(c*n)
    objective_scale = max(1.,a,b)
    # Only positive-L1 coefficients need epigraph auxiliaries. Adding a free
    # scanner t-variable (zero objective weight) creates redundant directions.
    penalized = np.flatnonzero((mask == 1) & (a > 0))
    k = len(penalized)
    init = np.r_[np.log(y.mean()/(1-y.mean())),np.zeros(d+k)]
    def fun(v):
        beta = v[1:1+d]
        z = v[0]+x@beta
        error = expit(z)-y
        loss = np.mean(np.logaddexp(0.,np.where(y==1,-z,z)))+.5*b*np.dot(mask,beta**2)+a*np.sum(v[1+d:])
        jac = np.r_[error.mean(),x.T@error/n+b*mask*beta,np.full(k,a)]
        return float(loss/objective_scale),jac/objective_scale
    matrix = np.zeros((2*k,1+d+k))
    matrix[np.arange(k),1+penalized] = -1.
    matrix[k+np.arange(k),1+penalized] = 1.
    matrix[:k,1+d:] = matrix[k:,1+d:] = np.eye(k)
    constraints = [{'type':'ineq','fun':lambda v:matrix@v,'jac':lambda v:matrix}] if k else []
    result = minimize(fun,init,jac=True,method='SLSQP',
                      constraints=constraints,
                      options={'ftol':1e-14,'maxiter':3000})
    w = result.x[:1+d]
    return w,{'success':bool(result.success),'status':int(result.status),'message':str(result.message),
              'iterations':int(result.nit),'objective_mean':objective(x,y,w,a*mask,b*mask)}
