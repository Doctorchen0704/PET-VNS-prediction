"""Original rank-based molecular annotation and numerical tie convention."""
import numpy as np
from scipy.stats import rankdata, spearmanr, false_discovery_control

def normalize(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=0)
    return x / np.sqrt((x*x).sum(axis=0))

def bh(p):
    p = np.asarray(p, float)
    ix = np.argsort(p)
    ordered = np.minimum.accumulate((p[ix] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    q = np.empty(len(p))
    q[ix] = np.clip(ordered, 0, 1)
    assert np.allclose(q, false_discovery_control(p), atol=1e-12)
    return q

def summarize(x):
    return dict(zip(['min', 'q025', 'median', 'q975', 'max'], map(float, np.quantile(x, [0, .025, .5, .975, 1]))))

def statistics(x, y, spins):
    xr, yr = normalize(rankdata(x)), normalize(rankdata(y))
    rho = float(xr @ yr)
    null = xr[spins] @ yr
    # Exact/repeated rank permutations can differ at the last floating-point bit.
    tolerance = 1e-12
    p = float((1 + np.count_nonzero(np.abs(null) >= abs(rho)-tolerance)) / (len(null)+1))
    mean = float(null.mean())
    pc = float((1+np.count_nonzero(np.abs(null-mean) >= abs(rho-mean)-tolerance)) / (len(null)+1))
    assert abs(rho-float(spearmanr(x, y).statistic)) < 1e-12
    return dict(rho=rho, p_spin=p, p_spin_null_centered=pc,
                pearson_r=float(normalize(x) @ normalize(y)), null_mean=mean,
                null_rho_quantiles=summarize(null)), null
