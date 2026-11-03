"""Original VNS conditional-spatial specificity functions; explicit array inputs."""
import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

def norm(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)

def project(a, z):
    return a - z @ np.linalg.lstsq(z, a, rcond=None)[0]

def zscore(a):
    return (a - a.mean(axis=0)) / a.std(axis=0)

def design(ap, g1, hemi=None):
    parts = [np.ones(len(ap))]
    if hemi is not None and len(set(hemi)) > 1:
        parts.append(hemi)
    parts += [zscore(rankdata(ap)), zscore(rankdata(g1))]
    z = np.column_stack(parts)
    assert np.linalg.matrix_rank(z) == z.shape[1]
    return z

def coefficients(xrows, y, z):
    a = project(np.atleast_2d(xrows).T, z).T
    b = project(np.atleast_2d(y).T if y.ndim == 1 else y, z)
    return norm(a) @ norm(b.T).T

def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(q, 1)
    return out

def summary(a):
    return dict(zip(["min", "q025", "median", "q975", "max"],
                    np.quantile(a, [0, .025, .5, .975, 1], axis=0).tolist()))

def graph(coords, k):
    w = np.zeros((34, 34))
    for c in (coords[:34], coords[34:]):
        distances = cdist(c, c)
        adj = np.zeros_like(w)
        for i, ns in enumerate(np.argsort(distances, axis=1)[:, 1:k+1]):
            adj[i, ns] = 1
        w += np.maximum(adj, adj.T) / 2
    return w

def moran(rows, w):
    a = np.atleast_2d(rows)
    a = a - a.mean(axis=1, keepdims=True)
    return len(w) / w.sum() * np.einsum("bi,ij,bj->b", a, w, a) / np.sum(a*a, axis=1)

def spatial_diagnostics(e, sim, d):
    result = {"original_norm": float(np.linalg.norm(e)),
              "null_norm": summary(np.linalg.norm(sim, axis=1)),
              "max_nuisance_dot": float(np.max(abs(sim @ d["z"]))),
              "original_cross_hemisphere_cosine": float(norm(e[:34]) @ norm(e[34:])),
              "null_cross_hemisphere_cosine": summary(np.sum(norm(sim[:, :34])*norm(sim[:, 34:]), axis=1))}
    for k in (3, 5, 8):
        w34 = graph(d["coords"], k); w68 = block_diag(w34, w34)
        result[f"k{k}_global"] = {"original": float(moran(e, w68)[0]), "null": summary(moran(sim, w68))}
        for h, sl in (("R", slice(0, 34)), ("L", slice(34, 68))):
            result[f"k{k}_{h}"] = {"original": float(moran(e[sl], w34)[0]), "null": summary(moran(sim[:, sl], w34))}
    return result

def test_family(x, d, basis, signs, method):
    e = project(rankdata(x), d["z"])
    yr = project(d["yr"], d["z"])
    obs = norm(e) @ norm(yr.T).T
    if method == "conditional_MSR":
        sim = (signs * (basis.T @ e)) @ basis.T
        assert np.max(abs(sim @ d["z"])) < 1e-9
        assert np.max(abs(np.sum(sim*sim, axis=1) - e@e)) < 1e-8
        assert np.max(abs(moran(sim, d["W"]) - moran(e, d["W"]))) < 1e-12
    else:
        sim = project(e[d["spins"]].T, d["z"]).T
    correlations = norm(sim) @ norm(yr.T).T
    p = (1 + np.sum(abs(correlations) >= abs(obs) - 1e-12, axis=0)) / (len(sim) + 1)
    q = bh(p)
    rows = [{"target": n, "partial_rho": float(obs[i]), "p_spatial": float(p[i]),
             "q_BH18": float(q[i]), "null_rho": summary(correlations[:, i])}
            for i, n in enumerate(d["names"])]
    return rows, sim, correlations, spatial_diagnostics(e, sim, d)
