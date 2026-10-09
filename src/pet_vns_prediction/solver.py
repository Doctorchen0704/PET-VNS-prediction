"""Deterministic convex elastic-net logistic solver, with an unpenalized intercept.

Same objective as sklearn 1.8 LogisticRegression/SAGA, divided by sample count:
 mean(logaddexp(0,z)-y*z) + l1/(C*n)*|beta|_1
                            + (1-l1)/(2*C*n)*|beta|_2^2.
No participant data, model selection, filesystem IO or randomization here.
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


class NumericalFailure(RuntimeError):
    def __init__(self,message,**diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics


def soft(v, threshold):
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.)


def objective(x, y, w, l1, l2):
    z = w[0] + x @ w[1:]
    # Class-specific softplus avoids cancellation for correctly classified tails.
    loss = np.mean(np.logaddexp(0., np.where(y == 1, -z, z)))
    return float(loss + l1 * np.sum(np.abs(w[1:])) + .5*l2*np.dot(w[1:],w[1:]))


def gradient(x, y, w, l2):
    error = expit(w[0]+x@w[1:])-y
    return np.r_[error.mean(), x.T@error/len(y)+l2*w[1:]]


def residual(g, w, l1):
    r = np.r_[g[0], np.where(w[1:] != 0, g[1:]+l1*np.sign(w[1:]), soft(g[1:],l1))]
    return float(np.max(np.abs(r)))


def optimize_intercept(x,y,w):
    """Scalar Newton update; it never penalizes the intercept."""
    w=w.copy()
    for _ in range(12):
        p=expit(w[0]+x@w[1:])
        error=float(np.mean(p-y))
        if abs(error) < 2e-16:
            break
        information=float(np.mean(p*(1-p)))
        if information <= 0:
            raise NumericalFailure('Degenerate intercept information.')
        w[0]-=error/information
    return w


def certificate(x,y,w,l1,l2):
    """Independent convex KKT + feasible Fenchel-dual bound, mean-loss units.

    For pure L1, shrink the dual residual to satisfy the infinity-norm constraint.
    The intercept first-order equation enforces mean(q-y)=0. Its floating-point
    tolerance is explicitly retained, rather than claiming symbolic arithmetic.
    """
    g=gradient(x,y,w,l2)
    q=expit(w[0]+x@w[1:])
    dual_score=x.T@(y-q)/len(y)
    scale=1.
    if l2 == 0:
        norm=float(np.max(np.abs(dual_score))) if dual_score.size else 0.
        scale=min(1.,l1/norm) if norm else 1.
        if scale < 1:
            scale*=1.-8*np.finfo(float).eps
        q=y-scale*(y-q)
        conjugate=0.
    else:
        conjugate=float(np.sum(soft(dual_score,l1)**2)/(2*l2))
    entropy=-float(np.mean(xlogy(q,q)+xlogy(1-q,1-q)))
    primal=objective(x,y,w,l1,l2)
    gap=primal-(entropy-conjugate)
    return {'objective_mean':primal,'dual_bound_mean':entropy-conjugate,
            'duality_gap_mean':gap,'kkt_inf':residual(g,w,l1),
            'intercept_gradient':float(g[0]),'dual_equality_error':float(np.mean(q-y)),
            'dual_scale':float(scale),'nonzero_coefficients':int(np.count_nonzero(w[1:]))}


def polish(x,y,w,l1,l2):
    """Damped orthant active-set Newton with exact coefficient-boundary steps.

    A coefficient crossing zero is put exactly on that boundary; the active
    set is then reconsidered. This removes the ill-conditioned, slowly converging
    cyclic quadratic subproblem in v3. Objective and acceptance are unchanged.
    A single-coordinate descent direction is a fixed numerical safeguard when
    an active Newton direction is unusable, never another predictive model.
    """
    design=np.column_stack((np.ones(len(y)),x))
    for iteration in range(MAX_POLISH):
        w=optimize_intercept(x,y,w)
        g=gradient(x,y,w,l2)
        if l1:
            tiny=(np.abs(w[1:])<1e-12)&(np.abs(g[1:])<=l1)
            w[1:][tiny]=0.
            g=gradient(x,y,w,l2)
        r=residual(g,w,l1)
        if r<=TARGET_KKT:
            proof=certificate(x,y,w,l1,l2)
            if -1e-12<=proof['duality_gap_mean']<=ACCEPT_GAP:
                return w,iteration
        p=expit(design@w)
        h=design.T@((p*(1-p))[:,None]*design)/len(y)
        h.flat[::len(w)+1]+=np.r_[0.,np.full(x.shape[1],l2)]
        active=np.r_[True,(w[1:]!=0)|(np.abs(g[1:])>l1)]
        signs=np.sign(w)
        signs[(w==0)&active]=-np.sign(g[(w==0)&active])
        direction=np.zeros(len(w))
        for _ in range(len(w)+1):
            restricted=h[np.ix_(active,active)]
            rhs=-(g+np.r_[0.,l1*signs[1:]])[active]
            try:
                step=np.linalg.solve(restricted,rhs)
            except np.linalg.LinAlgError:
                step=np.linalg.lstsq(restricted,rhs,rcond=1e-14)[0]
            direction[:]=0.;direction[active]=step
            wrong=(w==0)&active&(direction*signs<=0)
            wrong[0]=False
            if not l1 or not wrong.any():
                break
            active[wrong]=False
        directional=float(g@direction+l1*np.sum(np.where(w[1:]!=0,np.sign(w[1:])*direction[1:],np.abs(direction[1:]))))
        if not np.isfinite(direction).all() or directional>=0:
            sub=np.r_[g[0],np.where(w[1:]!=0,g[1:]+l1*np.sign(w[1:]),soft(g[1:],l1))]
            j=int(np.argmax(np.abs(sub)))
            if h[j,j]<=0:
                raise NumericalFailure('No valid coordinate curvature.',iteration=iteration,kkt=r)
            direction[:]=0.
            new=w[j]-g[j]/h[j,j]
            direction[j]=(float(soft(new,l1/h[j,j])) if j else new)-w[j]
        bound=1.
        crossing=np.zeros(len(w),dtype=bool)
        if l1:
            candidates=np.flatnonzero((w[1:]!=0)&(w[1:]*direction[1:]<0))+1
            if candidates.size:
                ratios=-w[candidates]/direction[candidates]
                bound=min(1.,float(ratios.min()))
                crossing[candidates[ratios==bound]]=True
        direction*=bound
        target=w+direction
        target[crossing]=0.
        direction=target-w
        predicted=float(g@direction+l1*(np.sum(np.abs(target[1:]))-np.sum(np.abs(w[1:]))))
        old=objective(x,y,w,l1,l2)
        step_size=1.
        for _ in range(60):
            candidate=w+step_size*direction
            # Preserve exact active-set zeros when taking the full Newton step.
            if step_size==1:
                candidate=target.copy()
            value=objective(x,y,candidate,l1,l2)
            nr=residual(gradient(x,y,candidate,l2),candidate,l1)
            if value <= old+1e-4*step_size*predicted or (value<=old+2e-15 and nr<r):
                w=candidate
                break
            step_size*=.5
        else:
            raise NumericalFailure('Newton polishing line search stalled.',iteration=iteration,kkt=r,
                                   hessian_condition=float(np.linalg.cond(h)),active_coefficients=int(np.count_nonzero(w[1:])))
    raise NumericalFailure('Newton polishing exhausted its fixed limit.',iteration=MAX_POLISH,kkt=r,
                           hessian_condition=float(np.linalg.cond(h)),last_certificate=certificate(x,y,w,l1,l2))


@dataclass
class FittedLogistic:
    coef_: np.ndarray
    intercept_: np.ndarray
    n_iter_: np.ndarray
    numerical_: dict

    @property
    def classes_(self):
        return np.array([0,1])

    def predict_proba(self,x):
        p=expit(self.intercept_[0]+np.asarray(x)@self.coef_[0])
        return np.column_stack((1-p,p))


def fit(x,y,c,l1_ratio):
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    if (x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all()
        or not np.isfinite(y).all() or set(np.unique(y))!={0.,1.}
        or not np.isfinite(c) or c<=0 or not 0<=l1_ratio<=1):
        raise ValueError('Invalid binary logistic input or penalty.')
    n,d=x.shape
    original_x=x
    shift=x.mean(axis=0)
    x=x-shift
    l1,l2=l1_ratio/(c*n),(1-l1_ratio)/(c*n)
    prevalence=float(y.mean())
    start=np.r_[np.log(prevalence/(1-prevalence)),np.zeros(d)]
    g0=gradient(x,y,start,l2)
    zero=bool(np.max(np.abs(g0[1:]),initial=0.)<=l1)
    lbfgs_status=None
    iterations=0
    polish_iterations=0
    if zero:
        w=start
        method='analytical_zero_KKT'
    else:
        if l1==0:
            def fun(w):
                return objective(x,y,w,0.,l2),gradient(x,y,w,l2)
            result=minimize(fun,start,jac=True,method='L-BFGS-B',
                            options={'gtol':TARGET_KKT,'ftol':1e-15,'maxiter':MAX_LBFGS,'maxls':50,'maxcor':30})
            w=result.x
        else:
            initial=np.r_[start[0],np.zeros(2*d)]
            def fun(v):
                beta=v[1:d+1]-v[d+1:]
                w=np.r_[v[0],beta]
                smooth=objective(x,y,w,0.,l2)
                g=gradient(x,y,w,l2)
                return smooth+l1*np.sum(v[1:]),np.r_[g[0],g[1:]+l1,-g[1:]+l1]
            result=minimize(fun,initial,jac=True,method='L-BFGS-B',
                            bounds=[(None,None)]+[(0.,None)]*(2*d),
                            options={'gtol':TARGET_KKT,'ftol':1e-15,'maxiter':MAX_LBFGS,'maxls':50,'maxcor':30})
            w=np.r_[result.x[0],result.x[1:d+1]-result.x[d+1:]]
        iterations=int(result.nit)
        lbfgs_status={'status':int(result.status),'success':bool(result.success),'message':str(result.message)}
        w,polish_iterations=polish(x,y,w,l1,l2)
        method='split_LBFGSB_with_KKT_Newton_polish' if l1 else 'LBFGSB_with_KKT_Newton_polish'
    w=optimize_intercept(x,y,w)
    w[0]-=shift@w[1:]
    cert=certificate(original_x,y,w,l1,l2)
    if (not all(np.isfinite(v) for v in cert.values()) or cert['kkt_inf']>ACCEPT_KKT
        or abs(cert['intercept_gradient'])>1e-12 or abs(cert['dual_equality_error'])>1e-12
        or cert['duality_gap_mean']>ACCEPT_GAP or cert['duality_gap_mean'] < -1e-12):
        raise NumericalFailure(f'Numerical certificate failed: {cert}')
    cert.update(method=method,C=float(c),l1_ratio=float(l1_ratio),lambda1=l1,lambda2=l2,
                training_n=n,training_prevalence=prevalence,analytical_zero=zero,
                lbfgs=lbfgs_status,polish_iterations=polish_iterations,
                coefficients=w[1:].tolist(),intercept=float(w[0]),
                internal_conditioning='training-X centering, exactly reversed in saved intercept; beta penalty unchanged')
    return FittedLogistic(w[None,1:],w[:1],np.array([iterations+polish_iterations]),cert)


def independent_slsqp(x,y,c,l1_ratio):
    """Separate constrained epigraph formulation; never used to select a model.

    Independently starts at beta=0; no primary-solver warm start or active set.
    The epigraph uses beta and t>=|beta|, not the primary +/- coefficient split.
    """
    x,y=np.asarray(x,float),np.asarray(y,float)
    n,d=x.shape
    a,b=l1_ratio/(c*n),(1-l1_ratio)/(c*n)
    objective_scale=max(1.,a,b)
    init=np.r_[np.log(y.mean()/(1-y.mean())),np.zeros(2*d)]
    def fun(v):
        z=v[0]+x@v[1:1+d]
        error=expit(z)-y
        loss=np.mean(np.logaddexp(0.,np.where(y==1,-z,z)))+.5*b*np.sum(v[1:1+d]**2)+a*np.sum(v[1+d:])
        jac=np.r_[error.mean(),x.T@error/n+b*v[1:1+d],np.full(d,a)]
        return float(loss/objective_scale),jac/objective_scale
    matrix=np.zeros((2*d,1+2*d))
    matrix[:d,1:1+d]=-np.eye(d)
    matrix[d:,1:1+d]=np.eye(d)
    matrix[:d,1+d:]=matrix[d:,1+d:]=np.eye(d)
    constraints={'type':'ineq','fun':lambda v:matrix@v,'jac':lambda v:matrix}
    result=minimize(fun,init,jac=True,method='SLSQP',constraints=constraints,
                    options={'ftol':1e-14,'maxiter':3000})
    w=result.x[:1+d]
    return w,{'success':bool(result.success),'status':int(result.status),'message':str(result.message),
              'iterations':int(result.nit),'objective_mean':objective(x,y,w,a,b)}
