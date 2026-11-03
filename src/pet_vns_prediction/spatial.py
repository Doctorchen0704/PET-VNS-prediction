"""Source-adapted numerical engine for the fixed external-cohort PC1 check.

No filesystem/network operations, implicit dataset loading, result-file writes,
component selection, clinical-outcome analysis, or classical parcel-IID P tests.
``analyze_cohort`` fixes production counts/seeds. Lower-level count arguments
exist for synthetic tests; they do not alter the production entry point.
Sphere/atlas order is right34 then left34, matched homologues within each half.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag, helmert, null_space
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata


TARGETS = ("5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT",
           "A4B2", "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1",
           "mGluR5", "MOR", "NET", "VAChT")
N_BOOT = 2000
N_NULL = 9999
SEEDS = {"bootstrap": 2026102401, "msr": 2026102402,
         "spin": 2026102403, "conditional": 2026102404}


def _array(value, ndim=None, name="array"):
    a = np.asarray(value, dtype=np.float64)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: nonfinite input")
    return a


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _check_close(actual, expected, name, rtol=1e-9, atol=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ArithmeticError(f"{name}: nonfinite invariant")
    err = float(np.max(np.abs(a - b)))
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise ArithmeticError(f"{name}: invariant failed; max error={err}")
    return err


def _unit_rows(a):
    a = np.asarray(a, dtype=np.float64)
    size = np.linalg.norm(a, axis=-1, keepdims=True)
    return np.divide(a, size, out=np.full_like(a, np.nan), where=size > 1e-14)


def correlations(xrows, y):
    """Pearson effect only: x=(draws,regions), y=(regions,targets)."""
    x = np.atleast_2d(np.asarray(xrows, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    _require(x.shape[1] == y.shape[0], "correlation dimensions mismatch")
    return np.clip(_unit_rows(x - x.mean(axis=1, keepdims=True)) @
                   _unit_rows((y - y.mean(axis=0)).T).T, -1., 1.)


def bh(pvalues):
    p = _array(pvalues, 1, "P values")
    _require(len(p) > 0 and np.all((p >= 0) & (p <= 1)), "invalid P values")
    order = np.argsort(p, kind="stable")
    q = np.minimum.accumulate((p[order] * len(p) /
                               np.arange(1, len(p) + 1))[::-1])[::-1]
    answer = np.empty_like(p)
    answer[order] = np.minimum(q, 1.)
    return answer


def _distribution(a):
    values = np.asarray(a, dtype=np.float64).ravel()
    valid = values[np.isfinite(values)]
    out = {"n_total": int(values.size), "n_finite": int(valid.size),
           "n_nonfinite": int(values.size - valid.size)}
    labels = ("min", "q025", "median", "q975", "max")
    out.update(dict(zip(labels, np.quantile(valid, [0, .025, .5, .975, 1]).tolist()))
               if valid.size else {k: None for k in labels})
    return out


def _effect_interval(a, observed):
    out = _distribution(a)
    finite = np.asarray(a)[np.isfinite(a)]
    out["same_sign_fraction_of_finite"] = (float(np.mean(np.sign(finite) == np.sign(observed)))
                                          if finite.size else None)
    out["positive_fraction_of_finite"] = float(np.mean(finite > 0)) if finite.size else None
    out["negative_fraction_of_finite"] = float(np.mean(finite < 0)) if finite.size else None
    out["interval_method"] = "participant-bootstrap percentile 95%; no draw filtering by cosine"
    return out


def clr_transform(raw_roi_means):
    a = _array(raw_roi_means, 2, "raw ROI means")
    _require(a.shape[0] >= 3 and a.shape[1] == 82, "need at least 3 participants and 82 ROIs")
    _require(np.all(a > 0), "ROI means must be strictly positive; no imputation")
    logs = np.log(a)
    return logs - logs.mean(axis=1, keepdims=True)


def fit_pc1(clr_matrix):
    """Covariance PCA via SVD; no feature scaling, no sign orientation."""
    x = _array(clr_matrix, 2, "CLR matrix")
    _require(x.shape[0] >= 2 and x.shape[1] >= 2, "PCA dimensions too small")
    center = x.mean(axis=0)
    centered = x - center
    total = float(np.sum(centered * centered))
    if total <= np.finfo(float).eps * max(float(np.sum(x * x)), np.finfo(float).tiny):
        raise ValueError("degenerate PCA: no between-participant variance")
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = s * s / (len(x) - 1)
    load = vt[0].copy()
    ev1, ev2 = float(eigenvalues[0]), float(eigenvalues[1])
    return {"center": center, "centered": centered, "loading": load,
            "scores": centered @ load, "eigenvalues": eigenvalues,
            "eigen1": ev1, "eigen2": ev2, "gap": ev1 - ev2,
            "relative_gap": (ev1 - ev2) / ev1,
            "variance_explained": float(s[0] ** 2 / np.sum(s * s)),
            "near_tied_first_two": bool((ev1 - ev2) <= 1e-10 * ev1)}


def orient_pallidum(pca, feature_names):
    """Observed-PC sign only; fallback order fixed before any associations."""
    names = list(feature_names)
    load = pca["loading"].copy()
    _require(len(names) == len(load) and len(set(names)) == len(names), "invalid feature names")
    _require("L_pallidum" in names and "R_pallidum" in names, "pallidum anchors missing")
    left, right = names.index("L_pallidum"), names.index("R_pallidum")
    value = float((load[left] + load[right]) / 2.)
    rule = "mean_bilateral_pallidum"
    fallback = abs(value) <= 1e-12
    if fallback:
        for rule, ix in (("L_pallidum", left), ("R_pallidum", right)):
            value = float(load[ix])
            if abs(value) > 1e-12:
                break
        else:
            candidates = np.flatnonzero(np.abs(load) == np.max(np.abs(load)))
            ix = min(candidates, key=lambda i: names[i])
            rule, value = "lexicographic_max_abs:" + names[ix], float(load[ix])
    flip = -1. if value < 0 else 1.
    answer = dict(pca)
    answer["loading"] = load * flip
    answer["scores"] = pca["scores"] * flip
    answer["sign_anchor"] = {"rule": rule, "fallback": fallback,
                             "bilateral_pallidum_mean_after": float(answer["loading"][[left, right]].mean())}
    return answer


def bootstrap_pc1(clr_matrix, observed_loading, n_boot=N_BOOT, seed=SEEDS["bootstrap"]):
    """Refit sample center and PC1 per draw; never swap/drop low-cosine PCs."""
    x = _array(clr_matrix, 2, "CLR bootstrap matrix")
    obs = _array(observed_loading, 1, "observed loading")
    _require(x.shape[1] == len(obs) and n_boot > 0, "bootstrap dimensions/count")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(x), size=(n_boot, len(x)), dtype=np.int64)
    arrays = {"indices": indices, "loadings": np.full((n_boot, x.shape[1]), np.nan),
              "center": np.full((n_boot, x.shape[1]), np.nan)}
    for name in ("eigen1", "eigen2", "gap", "relative_gap", "variance_explained", "cosine"):
        arrays[name] = np.full(n_boot, np.nan)
    arrays["degenerate"] = np.zeros(n_boot, dtype=bool)
    arrays["near_tied_first_two"] = np.zeros(n_boot, dtype=bool)
    arrays["sign_alignment_ambiguous"] = np.zeros(n_boot, dtype=bool)
    reasons = []
    for b, ix in enumerate(indices):
        try:
            fit = fit_pc1(x[ix])
        except ValueError as err:
            if not str(err).startswith("degenerate PCA"):
                raise
            arrays["degenerate"][b] = True
            reasons.append({"draw": b, "reason": str(err)})
            continue
        dot = float(fit["loading"] @ obs)
        arrays["sign_alignment_ambiguous"][b] = abs(dot) <= 1e-12
        arrays["loadings"][b] = fit["loading"] * (-1. if dot < 0 else 1.)
        arrays["center"][b] = fit["center"]
        arrays["cosine"][b] = abs(dot) / np.linalg.norm(obs)
        arrays["near_tied_first_two"][b] = fit["near_tied_first_two"]
        for name in ("eigen1", "eigen2", "gap", "relative_gap", "variance_explained"):
            arrays[name][b] = fit[name]
    meta = {"draws": int(n_boot), "seed": int(seed), "rng": "numpy.Generator(PCG64)",
            "n_degenerate": int(arrays["degenerate"].sum()), "degenerate_draws": reasons,
            "n_near_tied": int(arrays["near_tied_first_two"].sum()),
            "n_ambiguous_alignment": int(arrays["sign_alignment_ambiguous"].sum()),
            "n_cosine_below_0_5": int(np.sum(arrays["cosine"] < .5)),
            "n_cosine_below_0_8": int(np.sum(arrays["cosine"] < .8)),
            "retention": "All draws retained in arrays; degenerate draws flagged NaN, not replaced; no low-cosine filtering",
            "distributions": {k: _distribution(arrays[k]) for k in
                              ("eigen1", "eigen2", "gap", "relative_gap", "variance_explained", "cosine")}}
    return meta, arrays


def common_graph(sphere_coords, k=5):
    coords = _array(sphere_coords, 2, "sphere coordinates")
    _require(coords.shape == (68, 3) and 1 <= k < 34, "expected paired DK68 sphere coordinates")
    graphs = []
    for c in (coords[:34], coords[34:]):
        distances = cdist(c, c)
        np.fill_diagonal(distances, np.inf)
        _require(np.min(distances) > 0, "duplicate spherical centroids")
        near = np.argsort(distances, axis=1, kind="stable")[:, :k]
        adjacency = np.zeros((34, 34), dtype=np.float64)
        adjacency[np.arange(34)[:, None], near] = 1.
        graphs.append(np.maximum(adjacency, adjacency.T))
    w = (graphs[0] + graphs[1]) / 2.
    _check_close(w, w.T, "symmetric common graph")
    _require(np.all(np.diag(w) == 0) and w.sum() > 0, "invalid graph")
    return w


def moran(rows, weights):
    x = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    w = np.asarray(weights, dtype=np.float64)
    centered = x - x.mean(axis=1, keepdims=True)
    den = np.sum(centered * centered, axis=1)
    num = len(w) / w.sum() * np.einsum("bi,ij,bj->b", centered, w, centered, optimize=True)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-28)


def nonconditional_msr(x, w34, n_null=N_NULL, seed=SEEDS["msr"]):
    x = _array(x, 1, "cortical PC1 map")
    w = _array(w34, 2, "common graph")
    _require(x.shape == (68,) and w.shape == (34, 34), "MSR dimensions")
    h = helmert(34).T
    eigenvalues, eigenvectors = np.linalg.eigh(h.T @ w @ h)
    basis = h @ eigenvectors
    paired = np.column_stack((x[:34], x[34:]))
    means = paired.mean(axis=0)
    centered = paired - means
    _require(np.all(np.linalg.norm(centered, axis=0) > 1e-14), "constant hemisphere PC1 map")
    coefficients = basis.T @ centered
    signs = np.random.Generator(np.random.PCG64(seed)).choice([-1., 1.], size=(n_null, 33))
    sim = np.einsum("ij,bjk->bik", basis, signs[:, :, None] * coefficients[None, :, :], optimize=True) + means
    sc = sim - means
    qa = {"orthogonality_error": _check_close(basis.T @ basis, np.eye(33), "MSR basis orthogonality"),
          "zero_mean_basis_error": _check_close(basis.sum(axis=0), np.zeros(33), "MSR basis mean"),
          "reconstruction_error": _check_close(basis @ coefficients, centered, "MSR reconstruction"),
          "max_hemisphere_mean_error": _check_close(sim.mean(axis=1), means, "MSR hemisphere means"),
          "max_hemisphere_centered_norm2_error": _check_close(np.sum(sc * sc, axis=1), np.sum(centered * centered, axis=0), "MSR hemisphere norm"),
          "max_hemisphere_raw_norm2_error": _check_close(np.sum(sim * sim, axis=1), np.sum(paired * paired, axis=0), "MSR raw norm"),
          "max_bilateral_centered_dot_error": _check_close(np.sum(sc[:, :, 0] * sc[:, :, 1], axis=1), centered[:, 0] @ centered[:, 1], "MSR bilateral centered dot"),
          "max_bilateral_raw_dot_error": _check_close(np.sum(sim[:, :, 0] * sim[:, :, 1], axis=1), paired[:, 0] @ paired[:, 1], "MSR bilateral raw dot")}
    for hi, label in enumerate(("R", "L")):
        observed = float(moran(centered[:, hi], w)[0])
        values = moran(sc[:, :, hi], w)
        qa[f"max_Moran_{label}_error"] = _check_close(values, observed, f"MSR {label} Moran")
        qa[f"observed_Moran_{label}"] = observed
    sim68 = np.concatenate((sim[:, :, 0], sim[:, :, 1]), axis=1)
    W = block_diag(w, w)
    qa["global_Moran_observed"] = float(moran(x, W)[0])
    qa["global_Moran_null"] = _distribution(moran(sim68, W))
    qa["global_Moran_is_exact_invariant"] = False
    qa["global_Moran_note"] = "Hemisphere means differ and irregular graph degrees need not preserve globally centered Moran; diagnostic only"
    qa["status"] = "PASS"
    return {"maps": sim68, "basis": basis, "eigenvalues": eigenvalues,
            "coefficients": coefficients, "signs": signs, "qa": qa}


def spin_generator(sphere_coords, n_null=N_NULL, seed=SEEDS["spin"]):
    """Original convention: LH uses R, RH uses F R F; paired rotations.

    Both hemispheres have bijective Hungarian assignments. Corresponding
    homologues share a geometric rotation, not necessarily an identical index
    permutation, because the empirical hemisphere centroids differ.
    """
    coords = _array(sphere_coords, 2, "spin coordinates")
    _require(coords.shape == (68, 3), "spin requires right34/left34 coordinates")
    rng = np.random.Generator(np.random.PCG64(seed))
    rotations = Rotation.random(n_null, random_state=rng).as_matrix()
    reflection = np.diag([-1., 1., 1.])
    indices = np.empty((n_null, 68), dtype=np.int16)
    costs = np.empty((n_null, 2))
    for b, rot in enumerate(rotations):
        for hi, ids in enumerate((np.arange(34, 68), np.arange(34))):
            actual_rot = rot if hi == 0 else reflection @ rot @ reflection
            distances = cdist(coords[ids], coords[ids] @ actual_rot)
            rows, cols = linear_sum_assignment(distances)
            indices[b, ids[rows]] = ids[cols]
            costs[b, hi] = distances[rows, cols].sum()
    _require(np.all(np.sort(indices, axis=1) == np.arange(68)), "spin nonbijective")
    _require(np.all(indices[:, :34] < 34) and np.all(indices[:, 34:] >= 34), "spin hemisphere exchange")
    _check_close(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), "spin orthogonality")
    _check_close(np.linalg.det(rotations), 1., "spin proper rotations")
    # Independently verify geometric coupling on the first and last actual draw.
    coupling_error = 0.
    for b in sorted(set((0, n_null - 1))):
        rot = rotations[b]
        for ids, actual_rot in ((np.arange(34, 68), rot),
                                (np.arange(34), reflection @ rot @ reflection)):
            rows, cols = linear_sum_assignment(cdist(coords[ids], coords[ids] @ actual_rot))
            _require(np.array_equal(indices[b, ids[rows]], ids[cols]), "spin rotation-assignment mismatch")
        coupling_error = max(coupling_error, _check_close(
            (coords[:34] @ reflection) @ rot @ reflection,
            coords[:34] @ (reflection @ rot @ reflection), "coupled hemisphere geometry"))
    return {"indices": indices, "rotations": rotations, "assignment_cost_L_R": costs,
            "qa": {"status": "PASS", "draws": int(n_null), "seed": int(seed),
                   "unique_bilateral_permutations": int(len(np.unique(indices, axis=0))),
                   "identity_draws": int(np.sum(np.all(indices == np.arange(68), axis=1))),
                   "coupled_geometry_error": coupling_error,
                   "convention": "LH rotation; RH F@rotation@F, F=diag(-1,1,1); rows original, columns rotated",
                   "homologue_index_identity_required": False}}


def nuisance_design(ap, g1):
    ap, g1 = _array(ap, 1, "AP"), _array(g1, 1, "G1")
    _require(len(ap) == len(g1) == 68, "gradient dimensions")
    parts = [np.ones(68), np.r_[np.ones(34), np.zeros(34)]]
    for gradient in (ap, g1):
        ranks = rankdata(gradient, method="average")
        _require(np.std(ranks) > 0, "constant gradient")
        parts.append((ranks - ranks.mean()) / ranks.std())
    z = np.column_stack(parts)
    _require(np.linalg.matrix_rank(z) == 4, "conditional design must have full rank 4")
    return z


def residualize(values, z):
    values = np.asarray(values, dtype=np.float64)
    return values - z @ np.linalg.lstsq(z, values, rcond=None)[0]


def conditional_msr(x, receptor, z, W, n_null=N_NULL, seed=SEEDS["conditional"]):
    x = _array(x, 1, "conditional map")
    receptor = _array(receptor, 2, "conditional receptor matrix")
    z = _array(z, 2, "conditional design")
    W = _array(W, 2, "conditional graph")
    _require(x.shape == (68,) and receptor.shape == (68, 18) and z.shape == (68, 4)
             and W.shape == (68, 68), "conditional dimensions")
    e = residualize(rankdata(x, method="average"), z)
    yr = residualize(rankdata(receptor, axis=0, method="average"), z)
    _require(np.linalg.norm(e) > 1e-10 and np.all(np.linalg.norm(yr, axis=0) > 1e-10), "degenerate conditional residual")
    q = null_space(z.T)
    _require(q.shape == (68, 64), "conditional residual rank must be 64")
    eigenvalues, eigenvectors = np.linalg.eigh(q.T @ W @ q)
    basis = q @ eigenvectors
    coefficients = basis.T @ e
    signs = np.random.Generator(np.random.PCG64(seed)).choice([-1., 1.], size=(n_null, 64))
    sim = (signs * coefficients) @ basis.T
    qa = {"orthogonality_error": _check_close(basis.T @ basis, np.eye(64), "conditional basis"),
          "reconstruction_error": _check_close(basis @ coefficients, e, "conditional reconstruction"),
          "max_nuisance_dot_error": _check_close(sim @ z, 0., "conditional nuisance orthogonality", atol=1e-8),
          "max_norm2_error": _check_close(np.sum(sim * sim, axis=1), e @ e, "conditional norm", atol=1e-8),
          "max_Moran_error": _check_close(moran(sim, W), moran(e, W)[0], "conditional global Moran"),
          "observed_Moran": float(moran(e, W)[0]), "nuisance_rank": 4,
          "residual_dimension": 64, "status": "PASS"}
    # Do not rerank residuals. Pearson of ranked-map residuals is partial rho.
    observed = correlations(e, yr)[0]
    null_effects = correlations(sim, yr)
    return {"maps": sim, "basis": basis, "eigenvalues": eigenvalues, "coefficients": coefficients,
            "signs": signs, "x_residual": e, "receptor_residual": yr,
            "observed": observed, "null_effects": null_effects, "qa": qa}


def spatial_family(observed, null_effects, names=TARGETS, effect_name="pearson_r"):
    obs = _array(observed, 1, "observed effects")
    null = _array(null_effects, 2, "null effects")
    _require(len(obs) == len(names) == 18 and null.shape[1] == 18, "family must contain exactly 18 tests")
    p = (1 + np.sum(np.abs(null) >= np.abs(obs)[None, :] - 1e-12, axis=0)) / (len(null) + 1)
    q = bh(p)
    rows = [{"target": str(name), effect_name: float(obs[j]), "p_spatial": float(p[j]),
             "q_BH18": float(q[j]), "null_effects": _distribution(null[:, j])}
            for j, name in enumerate(names)]
    return {"all18": rows, "n_null": len(null), "multiplicity": "BH over all 18 targets",
            "p_rule": "(1+count(abs(null)>=abs(observed)-1e-12))/(n_null+1)"}


def _validate_mapping(feature_names, cortical_indices, receptor_names):
    names = list(feature_names)
    raw_indices = np.asarray(cortical_indices)
    ix = np.asarray(cortical_indices, dtype=np.int64)
    _require(len(names) == 82 and len(set(names)) == 82, "need 82 unique feature names")
    _require(ix.shape == (68,) and np.array_equal(raw_indices, ix) and len(set(ix)) == 68
             and np.all((ix >= 0) & (ix < 82)), "invalid cortical mapping")
    cortex_names = [names[i] for i in ix]
    _require(all(n.startswith("R_") for n in cortex_names[:34]) and
             all(n.startswith("L_") for n in cortex_names[34:]), "atlas order must be right34 then left34")
    _require([n[2:] for n in cortex_names[:34]] == [n[2:] for n in cortex_names[34:]], "homologue order mismatch")
    _require(tuple(receptor_names) == TARGETS, "receptor family/order differs from frozen plan")
    return names, ix


def analyze_cohort(raw_roi_means, feature_names, cortical_indices, receptors,
                   receptor_names, sphere_coords, ap, g1, vns_loadings=None):
    """Production entry point: 2,000 bootstrap + 9,999 draws per spatial model.

    Call for each authorized frozen cohort separately. Identical fixed seeds are used
    independently in each call; no paired branch-difference inference is made.
    Returns ``{'results': JSON-safe dict, 'arrays': dict[str,numpy.ndarray]}``.
    Caller alone handles source validation/freezing, serialization, and reports.
    """
    names, ix = _validate_mapping(feature_names, cortical_indices, receptor_names)
    receptor = _array(receptors, 2, "receptors")
    _require(receptor.shape == (68, 18), "receptors must be 68 by 18")
    _require(np.all(np.std(receptor, axis=0) > 1e-14), "constant receptor map")
    clr = clr_transform(raw_roi_means)
    observed = orient_pallidum(fit_pc1(clr), names)
    x = observed["loading"][ix]
    boot_meta, boot = bootstrap_pc1(clr, observed["loading"])
    xb = boot["loadings"][:, ix]
    w = common_graph(sphere_coords)
    W = block_diag(w, w)
    z = nuisance_design(ap, g1)
    msr = nonconditional_msr(x, w)
    primary_obs = correlations(x, receptor)[0]
    primary_null = correlations(msr["maps"], receptor)
    primary = spatial_family(primary_obs, primary_null)
    primary.update({"seed": SEEDS["msr"], "qa": msr["qa"], "method": "nonconditional paired-hemisphere singleton MSR"})
    spins = spin_generator(sphere_coords)
    xr, yr = rankdata(x, method="average"), rankdata(receptor, axis=0, method="average")
    spin_maps = xr[spins["indices"]]
    spin_obs, spin_null = correlations(xr, yr)[0], correlations(spin_maps, yr)
    secondary = spatial_family(spin_obs, spin_null, effect_name="spearman_rho")
    secondary.update({"seed": SEEDS["spin"], "qa": spins["qa"], "method": "secondary hemisphere-coupled Hungarian spins",
                      "Moran_raw_observed": float(moran(x, W)[0]),
                      "Moran_raw_null": _distribution(moran(x[spins["indices"]], W)),
                      "Moran_rank_observed": float(moran(xr, W)[0]),
                      "Moran_rank_null": _distribution(moran(spin_maps, W))})
    conditional = conditional_msr(x, receptor, z, W)
    cond_results = spatial_family(conditional["observed"], conditional["null_effects"], effect_name="partial_rank_r")
    cond_results.update({"seed": SEEDS["conditional"], "qa": conditional["qa"],
                         "method": "rank maps residualized on intercept, hemisphere, rank AP and rank G1; conditional MSR; no residual reranking",
                         "interpretation": "Changes estimand and null together; not a rescue of primary non-support"})
    finite_boot = ~boot["degenerate"]
    boot_pearson = np.full((N_BOOT, 18), np.nan)
    boot_spearman = np.full((N_BOOT, 18), np.nan)
    boot_partial = np.full((N_BOOT, 18), np.nan)
    boot_pearson[finite_boot] = correlations(xb[finite_boot], receptor)
    ranked_boot = rankdata(xb[finite_boot], axis=1, method="average")
    boot_spearman[finite_boot] = correlations(ranked_boot, yr)
    boot_partial[finite_boot] = correlations(residualize(ranked_boot.T, z).T,
                                            conditional["receptor_residual"])
    for result, values, obs in ((primary, boot_pearson, primary_obs),
                               (secondary, boot_spearman, spin_obs),
                               (cond_results, boot_partial, conditional["observed"])):
        for j, row in enumerate(result["all18"]):
            row["participant_bootstrap"] = _effect_interval(values[:, j], obs[j])
    focus = {}
    for label, sign in (("5HT2a", -1), ("5HTT", 1)):
        j = TARGETS.index(label)
        focus[label] = {"prespecified_direction": "negative" if sign < 0 else "positive",
                        "primary_direction_and_q_support": bool(sign * primary_obs[j] > 0 and primary["all18"][j]["q_BH18"] < .05),
                        "participant_uncertainty": primary["all18"][j]["participant_bootstrap"]}
    results = {"engine_scope": "One fixed PC1 branch; no patient outcomes; no classical parcel-IID P values",
               "n_participants": len(clr), "n_features": 82, "n_cortex": 68, "n_targets": 18,
               "seeds": dict(SEEDS), "branch_seed_policy": "Same fixed seeds in each separately fitted branch; no paired difference inference",
               "pca": {k: observed[k] for k in ("eigen1", "eigen2", "gap", "relative_gap", "variance_explained", "near_tied_first_two", "sign_anchor")},
               "bootstrap": boot_meta, "primary_msr": primary, "secondary_spin": secondary,
               "conditional_msr": cond_results, "focus_decisions": focus,
               "VNS_comparison": None,
               "limitations": ["Participant bootstrap does not include receptor-atlas or historical exploration uncertainty",
                               "MSR assumes spectral sign symmetry, not distribution-free parcel permutation",
                               "Healthy shared topology is not a VNS outcome validation or disease contrast"]}
    arrays = {"clr82": clr, "centered82": observed["centered"], "feature_center82": observed["center"],
              "pc1_loading82": observed["loading"], "pc1_scores": observed["scores"],
              "pca_eigenvalues": observed["eigenvalues"], "cortical_indices": ix, "pc1_cortex68": x,
              "W34": w, "W68": W, "primary_null_r": primary_null,
              "bootstrap_receptor_pearson": boot_pearson, "bootstrap_receptor_spearman": boot_spearman,
              "bootstrap_receptor_partial_rank": boot_partial,
              "spin_indices": spins["indices"], "spin_rotations": spins["rotations"],
              "spin_assignment_cost_L_R": spins["assignment_cost_L_R"], "spin_null_rho": spin_null,
              "conditional_design": z}
    arrays.update({"bootstrap_" + k: v for k, v in boot.items()})
    arrays.update({"msr_" + k: msr[k] for k in ("maps", "basis", "eigenvalues", "coefficients", "signs")})
    arrays.update({"conditional_" + k: conditional[k] for k in
                   ("maps", "basis", "eigenvalues", "coefficients", "signs", "x_residual", "receptor_residual", "null_effects")})
    if vns_loadings is not None:
        vns = _array(vns_loadings, 1, "frozen VNS loading")
        _require(vns.shape == (82,) and np.linalg.norm(vns) > 1e-14, "VNS map dimensions/norm")
        # Use exactly supplied historical orientation; never choose sign for similarity.
        cosine = float(observed["loading"] @ vns / np.linalg.norm(vns))
        boot_cos = boot["loadings"] @ vns / np.linalg.norm(vns)
        pearson = float(correlations(x, vns[ix])[0, 0])
        spearman = float(correlations(rankdata(x), rankdata(vns[ix]))[0, 0])
        bp = correlations(xb, vns[ix])[:, 0]
        bs = np.full(N_BOOT, np.nan)
        bs[finite_boot] = correlations(rankdata(xb[finite_boot], axis=1), rankdata(vns[ix]))[:, 0]
        results["VNS_comparison"] = {"cosine82": cosine, "cortical_pearson": pearson, "cortical_spearman": spearman,
                                     "cosine82_bootstrap": _effect_interval(boot_cos, cosine),
                                     "cortical_pearson_bootstrap": _effect_interval(bp, pearson),
                                     "cortical_spearman_bootstrap": _effect_interval(bs, spearman),
                                     "VNS_orientation": "Unchanged supplied frozen VNS map; no similarity-driven sign alignment",
                                     "VNS_bilateral_pallidum_mean": float(vns[[names.index("L_pallidum"), names.index("R_pallidum")]].mean()),
                                     "interpretation": "HC participant uncertainty conditional on fixed VNS map, not a disease comparison"}
        arrays.update({"vns_loading82": vns, "bootstrap_vns_cosine82": boot_cos,
                       "bootstrap_vns_cortical_pearson": bp, "bootstrap_vns_cortical_spearman": bs})
    # Only result metadata must be JSON-compatible; arrays retain explicit NaNs.
    import json
    json.dumps(results, allow_nan=False)
    return {"results": results, "arrays": arrays}


def synthetic_tests():
    """No real-data reads. Numerical and negative controls; returns JSON-safe QA."""
    rng = np.random.Generator(np.random.PCG64(417))
    checks = {}
    a = np.exp(rng.normal(size=(20, 82)))
    clr = clr_transform(a)
    checks["CLR_row_sum"] = _check_close(clr.sum(axis=1), 0., "synthetic CLR rows")
    checks["CLR_participant_scale_invariance"] = _check_close(clr_transform(a * np.exp(rng.normal(size=(20, 1)))), clr, "synthetic CLR scaling")
    pca = fit_pc1(clr)
    eig = np.linalg.eigvalsh(np.cov(clr, rowvar=False))[::-1]
    checks["PCA_covariance_eigen1"] = _check_close(pca["eigen1"], eig[0], "synthetic PCA eigen1")
    checks["PCA_covariance_eigen2"] = _check_close(pca["eigen2"], eig[1], "synthetic PCA eigen2")
    checks["PCA_covariance_score_variance"] = _check_close(np.var(pca["scores"], ddof=1), pca["eigen1"], "synthetic score variance")
    names = [f"feature_{i:02d}" for i in range(82)]
    names[0], names[1] = "L_pallidum", "R_pallidum"
    anchored = orient_pallidum(pca, names)
    _require(anchored["loading"][:2].mean() > 0, "synthetic anchor")
    fake = dict(pca, loading=np.r_[0., 0., 1., np.zeros(79)], scores=np.zeros(20))
    _require(orient_pallidum(fake, names)["sign_anchor"]["fallback"], "anchor fallback not detected")
    meta, boot = bootstrap_pc1(clr, anchored["loading"], n_boot=40)
    meta2, boot2 = bootstrap_pc1(clr, anchored["loading"], n_boot=40)
    _require(np.array_equal(boot["indices"], boot2["indices"]) and
             np.array_equal(boot["loadings"], boot2["loadings"]), "bootstrap seed nondeterministic")
    refit = fit_pc1(clr[boot["indices"][0]])
    checks["bootstrap_refit_center"] = _check_close(boot["center"][0], refit["center"], "bootstrap refit center")
    checks["bootstrap_refit_eigen1"] = _check_close(boot["eigen1"][0], refit["eigen1"], "bootstrap refit eigen")
    checks["bootstrap_retained_draws"] = int(len(boot["loadings"]))
    degenerate_input = np.vstack((np.zeros(82), np.ones(82), np.ones(82)))
    dmeta, db = bootstrap_pc1(degenerate_input, np.ones(82) / np.sqrt(82), n_boot=100)
    _require(dmeta["n_degenerate"] > 0 and len(db["loadings"]) == 100, "degenerate draws not retained")
    checks["degenerate_draws_retained"] = dmeta["n_degenerate"]
    checks["BH_known_vector"] = _check_close(bh([.01, .04, .03, .8]), [.04, .05333333333333334, .05333333333333334, .8], "BH control")
    right = _unit_rows(rng.normal(size=(34, 3)))
    reflection = np.diag([-1., 1., 1.])
    left = right @ reflection
    coords = np.vstack((right, left))
    w = common_graph(coords)
    W = block_diag(w, w)
    x = rng.normal(size=68) + np.r_[np.full(34, 2.), np.full(34, -3.)]
    msr = nonconditional_msr(x, w, n_null=81)
    checks["MSR_invariants"] = msr["qa"]
    sp = spin_generator(coords, n_null=47)
    _require(np.array_equal(sp["indices"][:, :34], sp["indices"][:, 34:] - 34), "exact mirrored homologue spin control failed")
    checks["spin_exact_mirror_homologue_control"] = "PASS"
    checks["spin_QA"] = sp["qa"]
    receptor = rng.normal(size=(68, 18))
    z = nuisance_design(rng.normal(size=68), rng.normal(size=68))
    cond = conditional_msr(x, receptor, z, W, n_null=81)
    checks["conditional_invariants"] = cond["qa"]
    obs = correlations(x, receptor)[0]
    fam = spatial_family(obs, correlations(msr["maps"], receptor))
    _require(all(1/82 <= v["p_spatial"] <= 1 for v in fam["all18"]), "add-one bounds failed")
    checks["spatial_P_add_one_and_BH18"] = "PASS"
    negative = {}
    for label, operation in (
        ("zero_ROI", lambda: clr_transform(np.zeros((4, 82)))),
        ("NaN_ROI", lambda: clr_transform(np.full((4, 82), np.nan))),
        ("constant_PCA", lambda: fit_pc1(np.ones((4, 82)))),
        ("rank_deficient_conditional_design", lambda: nuisance_design(np.ones(68), np.ones(68))),
        ("wrong_atlas_order", lambda: _validate_mapping(names, np.arange(68), TARGETS)),
        ("invalid_invariant", lambda: _check_close(1., 2., "deliberate corruption"))):
        try:
            operation()
        except (ValueError, ArithmeticError):
            negative[label] = "REJECTED_AS_EXPECTED"
        else:
            raise AssertionError("negative control unexpectedly accepted: " + label)
    return {"status": "PASS", "scope": "Synthetic inputs only; no file/network/data access",
            "production_n_boot": N_BOOT, "production_n_null": N_NULL,
            "production_seeds": dict(SEEDS), "checks": checks, "negative_controls": negative}
