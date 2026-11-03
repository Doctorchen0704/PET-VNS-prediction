"""Original VNS PCA conventions, distinct from external-cohort sign anchoring."""
import numpy as np
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment

def clr(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("CLR requires finite positive inputs; no epsilon substitution")
    z = np.log(x)
    return z - z.mean(axis=1, keepdims=True)

def fit(z, design):
    beta = np.linalg.lstsq(design, z, rcond=None)[0]
    r = z - design @ beta
    # Intercept belongs to every design, but explicitly remove round-off mean.
    r = r - r.mean(axis=0, keepdims=True)
    _, s, vt = svd(r, full_matrices=False, check_finite=True, lapack_driver="gesdd")
    v = vt.T
    for k in range(v.shape[1]):
        if v[np.argmax(np.abs(v[:, k])), k] < 0:
            v[:, k] *= -1
    eigen = s * s / (len(z) - 1)
    return {"beta": beta, "residual": r, "loadings": v, "eigen": eigen,
            "scores": r @ v, "fraction": eigen / eigen.sum()}

def similarity(a, b, k=3):
    cross = a[:, :k].T @ b[:, :k]
    same = np.abs(np.diag(cross))
    ii, jj = linear_sum_assignment(-np.abs(cross))
    matched = np.zeros(k)
    matched[ii] = np.abs(cross[ii, jj])
    subspace = [float(svd(a[:, :j].T @ b[:, :j], compute_uv=False)[-1])
                for j in range(1, k + 1)]
    return same, matched, np.asarray(subspace)

def quantiles(a):
    a = np.asarray(a)
    return {"min": a.min(axis=0).tolist(), "q10": np.quantile(a, .1, axis=0).tolist(),
            "median": np.median(a, axis=0).tolist(), "q90": np.quantile(a, .9, axis=0).tolist(),
            "max": a.max(axis=0).tolist()}

def pc1(z):
    # Identical intercept residualization and double centering as original fit.
    d = np.ones((len(z), 1))
    r = z - d @ np.linalg.lstsq(d, z, rcond=None)[0]
    r -= r.mean(axis=0, keepdims=True)
    _, s, vt = svd(r, full_matrices=False, check_finite=True, lapack_driver='gesdd')
    e = s * s / (len(z) - 1)
    return vt[0].copy(), e

def partial(x, y, d):
    xr = x - d @ np.linalg.lstsq(d, x, rcond=None)[0]
    yr = y - d @ np.linalg.lstsq(d, y, rcond=None)[0]
    assert np.max(np.abs(d.T @ xr)) < 1e-8
    assert np.max(np.abs(d.T @ yr)) < 1e-8
    denom = np.linalg.norm(xr) * np.linalg.norm(yr)
    if denom < 1e-10:
        raise ValueError('Degenerate residual maps')
    r = float(xr @ yr / denom)
    r2x = float(1 - (xr @ xr) / np.sum((x - x.mean()) ** 2))
    r2y = float(1 - (yr @ yr) / np.sum((y - y.mean()) ** 2))
    return r, r2x, r2y


def scanner_stratified_bootstrap(raw_roi_means, scanner_groups, reference_loading,
                                 n_boot=2000, seed=20261025):
    """Original VNS participant bootstrap, with explicit local input arrays.

    Preserves group sizes, sorted scanner-level ordering and within-draw
    concatenation. Every draw refits column centering and PC1 using ``pc1``.
    Signs align with the supplied frozen 82D reference, never with a receptor
    target. Any degenerate draw stops this implementation; it is not redrawn.
    The caller must supply the original participant row order for exact replay.
    """
    x = np.asarray(raw_roi_means, dtype=float)
    groups = np.asarray(scanner_groups)
    ref = np.asarray(reference_loading, dtype=float)
    if x.ndim != 2 or x.shape[1] != 82 or len(x) < 3:
        raise ValueError("Expected at least three participants by 82 regions")
    if groups.shape != (len(x),) or ref.shape != (82,) or not np.isfinite(ref).all():
        raise ValueError("Scanner/reference dimensions or finite values")
    if n_boot < 1 or not np.isclose(np.linalg.norm(ref), 1., atol=1e-12):
        raise ValueError("Positive bootstrap count and unit reference required")
    z = clr(x)
    observed, _ = pc1(z)
    observed *= -1. if observed @ ref < 0 else 1.
    if not np.allclose(observed, ref, atol=1e-13, rtol=0):
        raise ValueError("Reference does not reproduce from supplied raw inputs")
    strata = [np.flatnonzero(groups == level) for level in sorted(set(groups.tolist()))]
    rng = np.random.default_rng(seed)
    indices = np.empty((n_boot, len(x)), dtype=np.int64)
    loadings = np.empty((n_boot, 82))
    eigen12 = np.empty((n_boot, 2))
    fractions = np.empty(n_boot)
    cosines = np.empty(n_boot)
    flips = np.zeros(n_boot, bool)
    for b in range(n_boot):
        ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        v, eb = pc1(z[ix])
        if not np.isfinite(eb).all() or eb.sum() <= np.finfo(float).tiny:
            raise ValueError(f"Degenerate bootstrap draw {b}; not replaced")
        flips[b] = v @ ref < 0
        if flips[b]:
            v *= -1
        indices[b] = ix
        loadings[b] = v
        eigen12[b] = eb[:2]
        fractions[b] = eb[0] / eb.sum()
        cosines[b] = v @ ref
        if not np.isfinite(v).all() or eb[0] < eb[1] or abs(np.linalg.norm(v) - 1) >= 1e-12:
            raise ArithmeticError(f"Bootstrap invariant failed at draw {b}")
    return {"row_indices": indices, "pc1_loadings": loadings,
            "eigenvalues_first2": eigen12, "explained_fraction": fractions,
            "cosine_to_reference": cosines, "sign_flipped": flips}
