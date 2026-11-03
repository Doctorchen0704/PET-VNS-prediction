"""Three-target external TLE PC1 analysis, using caller-supplied arrays."""
import numpy as np
from scipy.linalg import block_diag, null_space
from scipy.stats import rankdata
from . import spatial as engine
N_BOOT, N_NULL = 2000, 9999
SEEDS = {'bootstrap': 2026102401, 'msr': 2026102402, 'conditional': 2026102404}
TARGETS = ['original_VNS_PC1', '5-HT2A', 'SERT']
GRADIENT_TARGETS = ['5-HT2A', 'SERT']

def expected_direction_fraction(values, direction):
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.mean(finite * direction > 0)) if finite.size else None


def holm(pvalues):
    p = engine._array(pvalues, 1, 'P values')
    engine._require(len(p) in (2, 3) and np.all((p >= 0) & (p <= 1)), 'Holm family must be exactly 2 or 3')
    order = np.argsort(p, kind='stable')
    corrected = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    answer = np.empty_like(p)
    answer[order] = corrected
    return answer


def family(observed, null_effects, names):
    obs = engine._array(observed, 1, 'Observed')
    null = engine._array(null_effects, 2, 'Null')
    engine._require(len(names) == len(obs) and len(obs) in (2, 3) and null.shape[1] == len(obs), 'Wrong test family')
    p = (1 + np.sum(np.abs(null) >= np.abs(obs)[None, :] - 1e-12, axis=0)) / (len(null) + 1)
    adj = holm(p)
    return {'multiplicity': 'Holm' + str(len(obs)), 'n_null': len(null), 'two_sided': True,
            'tail_rule': '(1+count(abs(null)>=abs(observed)-1e-12))/(n_null+1)',
            'rows': [{'target': name, 'effect': float(obs[j]), 'p_spatial': float(p[j]),
                      'p_Holm': float(adj[j])} for j, name in enumerate(names)]}


def conditional_two(x, receptor, z, W, n_null=N_NULL, seed=SEEDS['conditional']):
    x = engine._array(x, 1, 'conditional map')
    receptor = engine._array(receptor, 2, 'conditional receptor matrix')
    z = engine._array(z, 2, 'conditional design')
    W = engine._array(W, 2, 'conditional graph')
    engine._require(x.shape == (68,) and receptor.shape == (68, 2)
                    and z.shape == (68, 4) and W.shape == (68, 68), 'conditional dimensions')
    engine._require(np.linalg.matrix_rank(z) == 4, 'conditional design rank')
    e = engine.residualize(rankdata(x, method='average'), z)
    yr = engine.residualize(rankdata(receptor, axis=0, method='average'), z)
    engine._require(np.linalg.norm(e) > 1e-10 and np.all(np.linalg.norm(yr, axis=0) > 1e-10), 'degenerate conditional residual')
    q = null_space(z.T)
    engine._require(q.shape == (68, 64), 'conditional residual rank must be 64')
    eigenvalues, eigenvectors = np.linalg.eigh(q.T @ W @ q)
    basis = q @ eigenvectors
    coefficients = basis.T @ e
    signs = np.random.Generator(np.random.PCG64(seed)).choice([-1., 1.], size=(n_null, 64))
    sim = (signs * coefficients) @ basis.T
    qa = {
        'orthogonality_error': engine._check_close(basis.T @ basis, np.eye(64), 'conditional basis'),
        'reconstruction_error': engine._check_close(basis @ coefficients, e, 'conditional reconstruction'),
        'max_nuisance_dot_error': engine._check_close(sim @ z, 0., 'conditional nuisance orthogonality', atol=1e-8),
        'max_norm2_error': engine._check_close(np.sum(sim * sim, axis=1), e @ e, 'conditional norm', atol=1e-8),
        'max_Moran_error': engine._check_close(engine.moran(sim, W), engine.moran(e, W)[0], 'conditional global Moran'),
        'observed_Moran': float(engine.moran(e, W)[0]), 'nuisance_rank': 4, 'residual_dimension': 64,
        'status': 'PASS'}
    return {'maps': sim, 'basis': basis, 'eigenvalues': eigenvalues, 'coefficients': coefficients,
            'signs': signs, 'x_residual': e, 'receptor_residual': yr,
            'observed': engine.correlations(e, yr)[0], 'null_effects': engine.correlations(sim, yr), 'qa': qa}


def mapping(axis, cortex):
    names = axis['features']
    rows = cortex['rows']
    indices = np.array([r['project_feature_index'] for r in rows], dtype=int)
    engine._require(len(names) == len(set(names)) == 82 and len(rows) == 68, 'Feature/parcel counts')
    engine._require([r['public_cortex_row'] for r in rows] == list(range(68)), 'Public order')
    engine._require(len(set(indices)) == 68 and np.all((0 <= indices) & (indices < 82)), 'Unique indices')
    engine._require([names[i] for i in indices] == [r['project_feature'] for r in rows], 'Name/index crosswalk')
    mapped = [names[i] for i in indices]
    engine._require(all(n.startswith('R_') for n in mapped[:34]) and all(n.startswith('L_') for n in mapped[34:]), 'Hemisphere order')
    engine._require([n[2:] for n in mapped[:34]] == [n[2:] for n in mapped[34:]], 'Homologue order')
    coords = np.array([r['sphere_xyz'] for r in rows])
    receptor = np.array([[r['receptor_5HT2A'], r['receptor_SERT']] for r in rows])
    z = engine.nuisance_design([r['pial_y_mm'] for r in rows], [r['functional_G1'] for r in rows])
    w = engine.common_graph(coords, k=5)
    original = np.array(axis['original_PC1_loading'], dtype=float)
    targets = np.column_stack([original[indices], receptor])
    return indices, coords, receptor, z, w, original, targets


def analyze(raw, names, indices, receptors, z, w, original, targets, inferential):
    clr = engine.clr_transform(raw)
    fit = engine.orient_pallidum(engine.fit_pc1(clr), names)
    loading = fit['loading']
    cortical = loading[indices]
    observed = engine.correlations(cortical, targets)[0]
    partial_x = engine.residualize(rankdata(cortical, method='average'), z)
    partial_y = engine.residualize(rankdata(receptors, axis=0, method='average'), z)
    partial = engine.correlations(partial_x, partial_y)[0]
    meta, boot = engine.bootstrap_pc1(clr, loading, n_boot=N_BOOT, seed=SEEDS['bootstrap'])
    valid = ~boot['degenerate']
    boot_effect = np.full((N_BOOT, 3), np.nan)
    boot_partial = np.full((N_BOOT, 2), np.nan)
    boot_cosine_original = np.full(N_BOOT, np.nan)
    boot_effect[valid] = engine.correlations(boot['loadings'][valid][:, indices], targets)
    boot_cosine_original[valid] = boot['loadings'][valid] @ original / np.linalg.norm(original)
    for i in np.flatnonzero(valid):
        e = engine.residualize(rankdata(boot['loadings'][i, indices], method='average'), z)
        boot_partial[i] = engine.correlations(e, partial_y)[0]
    result = {
        'n': len(raw), 'pca': {k: fit[k] for k in ['center', 'loading', 'scores', 'eigenvalues', 'eigen1', 'eigen2',
             'gap', 'relative_gap', 'variance_explained', 'near_tied_first_two', 'sign_anchor']},
        'primary_effects': [{'target': name, 'pearson_r': float(observed[i]),
             'bootstrap': engine._effect_interval(boot_effect[:, i], observed[i]),
             'expected_direction_fraction_of_finite': expected_direction_fraction(boot_effect[:, i], [1, -1, 1][i])}
             for i, name in enumerate(TARGETS)],
        'gradient_effects': [{'target': name, 'partial_Spearman_rho': float(partial[i]),
             'bootstrap': engine._effect_interval(boot_partial[:, i], partial[i])} for i, name in enumerate(GRADIENT_TARGETS)],
        'original_82_cosine': {'observed': float(loading @ original / np.linalg.norm(original)),
             'bootstrap': engine._effect_interval(boot_cosine_original, loading @ original / np.linalg.norm(original)),
             'role': 'Descriptive, no separate significance test'},
        'bootstrap': meta, 'inferential_branch': inferential}
    arrays = {'raw': raw, 'clr': clr, 'cortical_indices': indices, 'center': fit['center'],
              'loading': loading, 'eigenvalues': fit['eigenvalues'], 'cortical_targets': targets,
              'receptors': receptors, 'nuisance_design': z, 'w34': w, 'W68': block_diag(w, w),
              'original_loading': original, 'bootstrap_primary_effects': boot_effect,
              'bootstrap_partial_effects': boot_partial, 'bootstrap_original_cosine': boot_cosine_original,
              'primary_observed': observed, 'gradient_observed': partial}
    arrays.update({'bootstrap_' + k: v for k, v in boot.items()})
    if inferential:
        null = engine.nonconditional_msr(cortical, w, n_null=N_NULL, seed=SEEDS['msr'])
        null_effects = engine.correlations(null['maps'], targets)
        result['primary_spatial_family'] = family(observed, null_effects, TARGETS)
        result['primary_spatial_QA'] = null['qa']
        arrays.update({'msr_' + k: null[k] for k in ['maps', 'basis', 'coefficients', 'signs', 'eigenvalues']})
        arrays['msr_null_effects'] = null_effects
        cond = conditional_two(cortical, receptors, z, block_diag(w, w))
        engine._check_close(partial, cond['observed'], 'Partial effect agreement')
        result['gradient_spatial_family'] = family(cond['observed'], cond['null_effects'], GRADIENT_TARGETS)
        result['gradient_spatial_QA'] = cond['qa']
        arrays.update({'conditional_' + k: cond[k] for k in ['maps', 'basis', 'coefficients', 'signs',
                      'eigenvalues', 'x_residual', 'receptor_residual', 'null_effects']})
    else:
        result['spatial_significance'] = 'NOT_RUN_SENSITIVITY_EFFECT_AND_STABILITY_ONLY'
    return result, arrays


def synthetic_tests():
    rng = np.random.default_rng(20261026)
    checks = []
    def test(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    test('Holm3 exact fixture', np.allclose(holm([.03, .001, .02]), [.04, .003, .04]))
    test('Holm2 exact fixture', np.allclose(holm([.01, .04]), [.02, .04]))
    test('Holm ties and cap', np.allclose(holm([.5, .5, 1]), [1, 1, 1]))
    test('Direction fraction uses finite-effect denominator', expected_direction_fraction([1., -1., np.nan], 1) == .5)
    test('No finite effect returns explicit missing', expected_direction_fraction([np.nan], 1) is None)
    f = family([.5, -.3, 0.], np.array([[.5-5e-13, -.1, 0.], [.4, .3, .1], [-.6, .2, -.2]]), TARGETS)
    test('Two-sided tie tolerance and plus one', np.allclose([r['p_spatial'] for r in f['rows']], [.75, .5, 1.]))
    raw = np.exp(rng.normal(size=(19, 82)))
    x = engine.clr_transform(raw)
    test('CLR row sum', np.allclose(x.sum(axis=1), 0, atol=1e-12))
    test('CLR intensity scaling', np.allclose(x, engine.clr_transform(raw * np.arange(1, 20)[:, None]), atol=1e-12))
    names = ['L_pallidum', 'R_pallidum'] + [f'roi{i}' for i in range(80)]
    observed = engine.orient_pallidum(engine.fit_pc1(x), names)
    cov = np.cov(x, rowvar=False)
    ev, vec = np.linalg.eigh(cov)
    test('SVD agrees with independent covariance eigensystem', abs(observed['loading'] @ vec[:, -1]) > 1-1e-10)
    test('Observed sign anchored without references', observed['loading'][:2].mean() >= 0)
    meta, boot = engine.bootstrap_pc1(x, observed['loading'], n_boot=31)
    test('Bootstrap indices deterministic', np.array_equal(boot['indices'], np.random.default_rng(SEEDS['bootstrap']).integers(0, 19, size=(31, 19))))
    test('Bootstrap retention and alignment', len(boot['indices']) == 31 and np.all(boot['loadings'] @ observed['loading'] >= -1e-12))
    coords = rng.normal(size=(68, 3)); coords /= np.linalg.norm(coords, axis=1)[:, None]
    w = engine.common_graph(coords, 5)
    cortical = rng.normal(size=68)
    null = engine.nonconditional_msr(cortical, w, n_null=127)
    test('Nonconditional all invariants', null['qa']['status'] == 'PASS')
    z = engine.nuisance_design(rng.normal(size=68), rng.normal(size=68))
    receptor18 = rng.normal(size=(68, 18))
    two = conditional_two(cortical, receptor18[:, :2], z, block_diag(w, w), n_null=127)
    old = engine.conditional_msr(cortical, receptor18, z, block_diag(w, w), n_null=127)
    test('Two-target adaptation matches frozen synthetic 18-target kernel', np.allclose(two['null_effects'], old['null_effects'][:, :2], atol=1e-12))
    test('Conditional maps unchanged by target count', np.array_equal(two['maps'], old['maps']))
    test('Conditional observed partial effects unchanged', np.allclose(two['observed'], old['observed'][:2], atol=1e-12))
    test('Conditional all invariants', two['qa']['status'] == 'PASS')
    negatives = []
    for label, fn in [('zero_ROI', lambda: engine.clr_transform(np.zeros((3, 82)))),
                      ('constant_PCA', lambda: engine.fit_pc1(np.ones((19, 82)))),
                      ('wrong_family_size', lambda: holm([.1]*18)),
                      ('wrong_conditional_targets', lambda: conditional_two(cortical, receptor18, z, block_diag(w, w), n_null=2))]:
        try:
            fn()
        except (ValueError, ArithmeticError):
            negatives.append(label)
        else:
            raise AssertionError('Failed to reject: ' + label)
    return {'status': 'PASS', 'checks': checks, 'negative_controls': negatives,
            'scope': 'Synthetic only; no participant data or real associations computed',
            'production_n_boot': N_BOOT, 'production_n_null': N_NULL, 'seeds': SEEDS}
