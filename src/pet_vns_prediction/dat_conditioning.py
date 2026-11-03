"""Bounded DAT conditioning of caller-supplied fixed PC1 and bootstrap maps.

Derived from crosscohort_sert_specificity_20261026_v1/code/atlas_kernel.py
rank5 effect-only section; no spatial-null, clinical, PCA, or file operations.
"""
import numpy as np
from scipy.stats import rankdata

MAPS = ['SERT_primary_Beliveau', 'SERT_alternative_Savli', 'DAT_Dukart_FP-CIT_SPECT']

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)

def finite(a):
    a = np.asarray(a, dtype=np.float64)
    require(np.isfinite(a).all(), 'Nonfinite values; no filtering allowed')
    return a

def residual(a, z):
    return a - z @ np.linalg.lstsq(z, a, rcond=None)[0]

def correlations(x, y):
    x = np.atleast_2d(finite(x))
    y = finite(y)
    if y.ndim == 1:
        y = y[:, None]
    require(x.shape[1] == y.shape[0], 'Correlation dimensions')
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=0)
    require(np.all(nx > 1e-12) and np.all(ny > 1e-12), 'Degenerate correlation; preserve failure, do not drop draws')
    return np.clip((x / nx[:, None]) @ (y / ny), -1., 1.)

def describe(values, observed):
    a = finite(values)
    require(a.ndim == 1, 'Full 1D bootstrap expected')
    return {'n': len(a), 'ci95': np.quantile(a, [.025, .975]).tolist(),
            'median': float(np.median(a)), 'min': float(a.min()), 'max': float(a.max()),
            'positive_fraction': float(np.mean(a > 0)),
            'negative_fraction': float(np.mean(a < 0)),
            'zero_fraction': float(np.mean(a == 0)),
            'same_sign_fraction': float(np.mean(np.sign(a) == np.sign(observed))),
            'interval_scope': 'Pointwise percentile95, fixed maps and all saved participant/PCA resamples; not simultaneous or spatial-null inference'}

def compute(observed, boot, maps, z, n_participants=None):
    observed, boot, maps, z = map(finite, (observed, boot, maps, z))
    require(observed.shape == (68,) and boot.ndim == 2 and boot.shape[1] == 68, 'PC1 shape')
    require(maps.shape == (68, 3) and z.shape == (68, 4), 'Maps/Z shape')
    require(np.linalg.matrix_rank(z) == 4, 'Original rank4 design required')
    ranks_x = rankdata(np.vstack([observed, boot]), axis=1, method='average')
    ranks_y = rankdata(maps, axis=0, method='average')
    base_x, base_y = residual(ranks_x.T, z).T, residual(ranks_y, z)
    base = correlations(base_x, base_y)
    dat = ranks_y[:, 2]
    require(dat.std(ddof=0) > 1e-12, 'Constant DAT')
    zd = np.column_stack([z, (dat - dat.mean()) / dat.std(ddof=0)])
    require(np.linalg.matrix_rank(zd) == 5, 'DAT design must have rank5')
    dat_x, dat_y = residual(ranks_x.T, zd).T, residual(ranks_y[:, :2], zd)
    effect = correlations(dat_x, dat_y)
    retained = np.sum(dat_y ** 2, axis=0) / np.sum(base_y[:, :2] ** 2, axis=0)
    require(np.all(retained > 1e-12), 'SERT targets degenerate after DAT')
    result = {
        'n_participants': n_participants, 'n_cortical_parcels': 68, 'bootstrap_draws': len(boot),
        'all_saved_draws_retained': True, 'PCA_refitted': False, 'spatial_P': 'NOT_COMPUTED_RANK5_EFFECT_ONLY',
        'parametric_P': 'NOT_COMPUTED', 'baseline_rank4_reproduction': [],
        'DAT_conditioned': {'design_rank': 5, 'condition_number': float(np.linalg.cond(zd)),
            'PC1_residual_variance_fraction_retained': float(np.sum(dat_x[0] ** 2) / np.sum(base_x[0] ** 2)),
            'rows': []},
        'scope': 'Two fixed SERT maps, descriptive rank5 partial Spearman plus saved-participant uncertainty; no new nulls, clinical tests, AUC or cohort contrasts.'}
    for i, name in enumerate(MAPS):
        result['baseline_rank4_reproduction'].append({'map': name, 'partial_Spearman': float(base[0, i]),
                                                     'participant_bootstrap': describe(base[1:, i], base[0, i])})
    for i, name in enumerate(MAPS[:2]):
        result['DAT_conditioned']['rows'].append({'map': name, 'partial_Spearman': float(effect[0, i]),
            'participant_bootstrap': describe(effect[1:, i], effect[0, i]),
            'target_residual_variance_fraction_retained': float(retained[i]),
            'target_VIF_from_DAT': float(1. / retained[i]),
            'low_residual_variance_warning': bool(retained[i] < .1)})
    arrays = {'baseline_effects': base, 'DAT_effects': effect, 'DAT_design': zd,
              'baseline_PC1_residuals': base_x, 'baseline_target_residuals': base_y,
              'DAT_PC1_residuals': dat_x, 'DAT_target_residuals': dat_y,
              'target_residual_fraction': retained}
    return result, arrays

def synthetic_tests():
    rng = np.random.default_rng(2026102608)
    z = np.column_stack([np.ones(68), np.r_[np.ones(34), np.zeros(34)], rng.normal(size=(68, 2))])
    maps = rng.normal(size=(68, 3)); x = rng.normal(size=68); boot = rng.normal(size=(12, 68))
    result, arrays = compute(x, boot, maps, z)
    checks = []
    def check(name, good):
        require(good, name); checks.append(name)
    q = np.linalg.qr(arrays['DAT_design'], mode='reduced')[0]
    xr, yr = rankdata(x), rankdata(maps[:, 0])
    ex, ey = xr - q @ (q.T @ xr), yr - q @ (q.T @ yr)
    check('rank5 vs independent QR', abs(np.corrcoef(ex, ey)[0, 1] - arrays['DAT_effects'][0, 0]) < 1e-12)
    check('rank5 residual orthogonality', np.max(np.abs(arrays['DAT_PC1_residuals'] @ arrays['DAT_design'])) < 1e-9)
    check('variance fraction within bounds', np.all((arrays['target_residual_fraction'] > 0) & (arrays['target_residual_fraction'] <= 1 + 1e-12)))
    transformed, _ = compute(x * 2 + 3, boot * 5 - 1, maps * 3 + 8, z)
    check('positive affine transform rank invariance', abs(result['DAT_conditioned']['rows'][0]['partial_Spearman'] - transformed['DAT_conditioned']['rows'][0]['partial_Spearman']) < 1e-12)
    try:
        correlations(np.ones(68), maps)
    except ValueError:
        checks.append('constant correlation rejected')
    else:
        raise ValueError('constant was not rejected')
    return {'status': 'PASS', 'checks': checks, 'real_data_used': False}
