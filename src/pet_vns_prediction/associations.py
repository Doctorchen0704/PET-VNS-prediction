"""HC3 regional association calculations with explicit multiplicity families.

Inputs are in-memory arrays. Covariates must be encoded by the caller, with the
paper's declared reference categories and no automatic row or variable removal.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from scipy import linalg, stats


def hc3_fit(design, values, *, check=True):
    """OLS via QR, HC3 sandwich covariance and residual-df t inference.

Numerical expressions follow the study implementation. The independent SVD
sandwich check is retained; filesystem logging and elapsed-time globals are not.
    """
    if not __debug__:
        raise RuntimeError("Do not disable the numerical checks with Python -O.")
    a, y = np.asarray(design, dtype=float), np.asarray(values, dtype=float)
    if a.ndim != 2 or y.shape != (len(a),) or not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError("Finite design matrix and matching outcome vector required.")
    n, p = a.shape
    if np.linalg.matrix_rank(a) != p or n <= p:
        raise ValueError("The complete-case design must have full rank and positive residual df.")
    q, r = np.linalg.qr(a, mode='reduced')
    beta = linalg.solve_triangular(r, q.T @ y)
    invr = linalg.solve_triangular(r, np.eye(p))
    bread = invr @ invr.T
    res = y - a @ beta
    h = (q*q).sum(axis=1)
    if max(h) >= 1 - 1e-8:
        raise ValueError("Leverage is too close to one for the fixed HC3 rule.")
    meat = a.T @ (((res / (1-h))**2)[:, None] * a)
    cov = bread @ meat @ bread
    diagnostics = None
    if check:
        ap = np.linalg.pinv(a)
        b2 = ap @ y
        e2 = y - a @ b2
        h2 = np.einsum('ij,ji->i', a, ap)
        c2 = (ap * ((e2/(1-h2))**2)[None, :]) @ ap.T
        db, dc = float(np.max(abs(beta-b2))), float(np.max(abs(cov-c2)))
        assert np.allclose(beta, b2, rtol=1e-9, atol=1e-10)
        assert np.allclose(cov, c2, rtol=1e-8, atol=1e-10)
        diagnostics = {'coefficient_max_abs_diff': db, 'HC3_cov_max_abs_diff': dc}
    se = np.sqrt(np.diag(cov))
    crit = stats.t.ppf(.975, n-p)
    mse = np.dot(res, res)/(n-p)
    cooks = res**2/(p*mse)*h/(1-h)**2
    return {'beta': beta, 'se': se, 'lo': beta-crit*se, 'hi': beta+crit*se,
            'p': 2*stats.t.sf(abs(beta/se), n-p), 'res': res,
            'fitted': a @ beta, 'h': h, 'cooks': cooks, 'df': n-p,
            'independent_check': diagnostics}


def bh_adjust(pvalues, *, expected_tests: int):
    """Adjust the entire declared family; no selection of only plotted results."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) != expected_tests or expected_tests < 1:
        raise ValueError("P-value count does not match the complete declared family.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("All family p-values must be finite and between zero and one.")
    q = np.asarray(stats.false_discovery_control(p, method="bh"))
    order = np.argsort(p)
    manual = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1].clip(0, 1)
    if np.max(abs(q[order]-manual)) >= 1e-14:
        raise ArithmeticError("Independent Benjamini-Hochberg calculation disagrees.")
    return q


def association_family(
    regional_values,
    response,
    adjustment_covariates: Mapping[str, np.ndarray],
    *,
    region_names: Sequence[str] | None = None,
    expected_tests: int,
) -> list[dict]:
    """Fit a complete six-region family for one or more adjustment designs.

Each design contains covariates only: intercept and response are added here.
Use one adjustment with expected_tests=6 for the primary family, or three
predeclared adjustments with expected_tests=18 for the joint extended family.
Processing/morphometry families with region-specific design columns require a
separate explicit orchestration; do not pass a reduced plotted subset here.
    """
    values = np.asarray(regional_values, dtype=float)
    y = np.asarray(response, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6 or y.shape != (len(values),):
        raise ValueError("Exactly six regional columns and a matching response vector are required.")
    if not np.isfinite(values).all() or not np.array_equal(np.unique(y), [0., 1.]):
        raise ValueError("Finite measurements and two explicit 0/1 response groups required.")
    if min(np.sum(y == 0), np.sum(y == 1)) < 2:
        raise ValueError("Pooled within-group SD requires at least two observations per group.")
    names = tuple(region_names) if region_names is not None else tuple(f"region_{i+1}" for i in range(6))
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError("Six unique region names required.")
    if not adjustment_covariates or len(adjustment_covariates) * 6 != expected_tests:
        raise ValueError("Adjustment count differs from the declared family size.")
    rows = []
    for label, covariates in adjustment_covariates.items():
        covariates = np.asarray(covariates, dtype=float)
        if covariates.ndim != 2 or len(covariates) != len(y):
            raise ValueError("Covariates must be an n-by-p numeric array; use n-by-0 for no covariates.")
        design = np.c_[np.ones(len(y)), y, covariates]
        for index, region in enumerate(names):
            z = values[:, index]
            positive, negative = z[y == 1], z[y == 0]
            pooled_sd = np.sqrt(((len(positive)-1)*positive.var(ddof=1) + (len(negative)-1)*negative.var(ddof=1)) / (len(y)-2))
            if not np.isfinite(pooled_sd) or pooled_sd <= 0:
                raise ValueError("Positive pooled within-group SD required for every region.")
            fit = hc3_fit(design, z)
            rows.append({"region": region, "adjustment": str(label), "n": len(y),
                         "difference": float(fit['beta'][1]), "HC3_se": float(fit['se'][1]),
                         "df": fit['df'], "p": float(fit['p'][1]),
                         "pooled_within_group_sd": float(pooled_sd),
                         "standardized_difference": float(fit['beta'][1]/pooled_sd),
                         "standardized_CI95": [float(fit['lo'][1]/pooled_sd), float(fit['hi'][1]/pooled_sd)]})
    qvalues = bh_adjust([row['p'] for row in rows], expected_tests=expected_tests)
    for row, qvalue in zip(rows, qvalues):
        row['q_family'] = float(qvalue)
        row['family_tests'] = expected_tests
    return rows
