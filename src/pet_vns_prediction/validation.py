"""In-memory ports of the executed ablation and C/CP validation methods.

No private input, filesystem I/O, or fitting occurs on import. Returned records
can contain participant-level information: do not publish real-data results.
These routines are serial because the certified solver collects a global audit.
"""
from __future__ import annotations

from collections import Counter
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from . import core, workflow
from .modeling import paper_settings, run_nested_models


def _require(ok, message):
    if not ok:
        raise core.InputRejected(message)


def _deadline(timeout_seconds):
    _require(np.isfinite(timeout_seconds) and timeout_seconds > 0, "Positive finite timeout required.")
    started = time.monotonic()

    def check():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("Validation time limit exceeded; no automatic retry.")

    return check


def _settings(settings):
    result = paper_settings() if settings is None else settings
    _require(__debug__, "Required assertions disabled; do not use Python -O.")
    _require(result.inner_folds == 3, "Executed validation requires three inner folds.")
    return result


def _draws(value):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0,
             "Number of draws must be a positive integer.")


def _cohort(x, y, settings):
    core.validate_inputs(x, y, settings, settings.model_columns()["CP"])
    _require(all(isinstance(v, str) and v.strip() for v in x.index),
             "Participant indices must be nonempty strings.")
    _require(set(y.unique()) == {0, 1}, "Both outcome classes required.")


def grouped_inner_plan(y, groups, seed):
    """Three stratified folds over unique people, expanded to all sampled copies.

    Sorting unique group keys, fold construction, multiplicity expansion, and
    class-support rejection are identical to the original ``inner_plan``.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    _require(y.ndim == groups.ndim == 1 and len(y) == len(groups) > 0,
             "Aligned nonempty labels and groups required.")
    _require(not pd.isna(groups).any() and set(y.tolist()).issubset({0, 1}),
             "Missing groups or nonbinary outcomes.")
    unique = np.unique(groups)
    labels = np.asarray([y[np.flatnonzero(groups == value)[0]] for value in unique], int)
    _require(all(len(np.unique(y[groups == value])) == 1 for value in unique), "Conflicting copied label.")
    if len(np.unique(labels)) != 2 or min(np.bincount(labels)) < 3:
        raise core.ClassSupportError("Fewer than three distinct sampled patients in an outcome class")
    folds, coverage = [], np.zeros(len(y), int)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for a, b in splitter.split(np.zeros(len(unique)), labels):
        tr = np.flatnonzero(np.isin(groups, unique[a])).tolist()
        va = np.flatnonzero(np.isin(groups, unique[b])).tolist()
        _require(not set(groups[tr]) & set(groups[va]), "Copied-patient leakage.")
        _require(set(tr) | set(va) == set(range(len(y))), "Inner coverage.")
        _require(len(np.unique(y[tr])) == len(np.unique(y[va])) == 2, "Inner class support.")
        coverage[va] += 1
        folds.append({"train": tr, "validation": va})
    _require(np.all(coverage == 1), "Inner validation count.")
    return folds


def develop(xtrain, ytrain, groups, xtest, model, seed, settings=None, *, deadline=None):
    """Retune and refit C/CP using training data only; no test outcomes accepted.

    Duplicate training index values are allowed only with identical group keys.
    ``groups`` identifies original people, not the sampled row occurrences.
    The calling cohort wrapper applies the baseline missingness gate; it is not
    reapplied to bootstrap samples, preserving the executed method.
    """
    settings = _settings(settings)
    _require(model in ("C", "CP"), "Development supports C and CP only.")
    _require(isinstance(xtrain, pd.DataFrame) and isinstance(xtest, pd.DataFrame)
             and isinstance(ytrain, pd.Series) and xtrain.index.equals(ytrain.index),
             "Aligned training frame/series and evaluation frame required.")
    _require(len(xtest) > 0, "At least one evaluation row required.")
    groups = np.asarray(groups)
    inner = grouped_inner_plan(ytrain, groups, seed)
    # Prevent callers assigning different groups to copies of the same index.
    identity_groups = {}
    group_identities = {}
    for identity, group in zip(xtrain.index, groups.tolist()):
        _require(not pd.isna(identity), "Missing training identity.")
        _require(identity not in identity_groups or identity_groups[identity] == group,
                 "Copies of one participant must retain one group.")
        _require(group not in group_identities or group_identities[group] == identity,
                 "A group must identify one participant.")
        identity_groups[identity], group_identities[group] = group, identity
    fields = settings.model_columns()[model]
    for frame in (xtrain, xtest):
        _require(frame.columns.is_unique and set(fields).issubset(frame.columns), "Missing or duplicate predictors.")
        _require(not (set(frame.columns) - settings.allowed_columns()), "Unexpected predictor columns.")
        for field in fields:
            values = frame[field]
            if field in settings.categories:
                _require(not (set(values.dropna()) - set(settings.categories[field])), "Unknown category.")
            else:
                _require(not np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any(), "Infinite predictor.")
    check = _deadline(1800) if deadline is None else deadline
    check()
    n = len(xtrain)
    joined = pd.concat([xtrain, xtest], axis=0)
    yy = pd.Series(np.r_[np.asarray(ytrain, int), np.zeros(len(xtest), int)], index=joined.index)
    plan = {"repeat": 0, "fold": 0, "train": list(range(n)), "test": list(range(n, len(joined))),
            "inner_seed": int(seed), "inner": inner}
    audit = []
    with threadpool_limits(limits=1):
        p, info = workflow.corrected_outer(joined, yy, fields, settings, plan, model, audit, check)
    check()
    _require(all(not set(a["fit_rows"]) & set(plan["test"]) for a in audit), "Preprocessing test-row leakage.")
    return {"model": model, "seed": int(seed), "inner": inner, "training_patient_groups": groups.tolist(),
            "evaluation_predictions": p.tolist(), "preprocessing_audit": audit, **info}


def measures(y, p, clip=1e-6):
    """Executed AUC, Brier and calibration, retaining single-class failures."""
    y, p = np.asarray(y), np.asarray(p, float)
    _require(y.ndim == p.ndim == 1 and len(y) == len(p) and set(y.tolist()).issubset({0, 1}), "Invalid labels.")
    _require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid probability.")
    if len(np.unique(y)) != 2:
        return {"n": len(y), "responders": int(y.sum()), "auc": None,
                "brier": float(np.mean((y - p)**2)) if len(y) else None,
                "calibration": {"status": "NOT_ESTIMABLE_SINGLE_CLASS"}}
    return {"n": len(y), "responders": int(y.sum()), "auc": float(roc_auc_score(y, p)),
            "brier": float(np.mean((y - p)**2)), "calibration": core.calibration(y, p, clip)}


def conditional_intervals(y, predictions, *, draws=2000, seed=20260910, deadline=None):
    """Paired C/CP bootstrap of fixed held-out probabilities, with no refitting."""
    _draws(draws)
    y = np.asarray(y)
    _require(len(y) > 0 and set(predictions) == {"C", "CP"}, "C/CP probabilities required.")
    for p in predictions.values():
        measures(y, p)
    rng = np.random.default_rng(seed)
    values = {m: {"auc": [], "brier": []} for m in ("C", "CP")}
    delta_auc, delta_brier, invalid = [], [], 0
    for _ in range(draws):
        if deadline is not None:
            deadline()
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            invalid += 1
            continue
        for m in values:
            p = np.asarray(predictions[m])[ix]
            values[m]["auc"].append(float(roc_auc_score(y[ix], p)))
            values[m]["brier"].append(float(np.mean((y[ix] - p)**2)))
        delta_auc.append(values["CP"]["auc"][-1] - values["C"]["auc"][-1])
        delta_brier.append(values["C"]["brier"][-1] - values["CP"]["brier"][-1])
    quantile = lambda v: np.quantile(v, [.025, .975]).tolist() if len(v) else None
    return {"scope": "Conditional on these fixed heldout predictions, no retraining", "valid": len(delta_auc),
            "one_class_draws": invalid, "models": {m: {k: quantile(v) for k, v in vv.items()} for m, vv in values.items()},
            "delta_auc_CP_minus_C": quantile(delta_auc), "delta_brier_C_minus_CP": quantile(delta_brier)}


def run_cross_scanner(x, y, scanner, settings=None, *, interval_draws=2000,
                      interval_seed=20260910, timeout_seconds=1800):
    """Two same-center scanner holdouts, ordered lexicographically; C and CP paired."""
    settings, check = _settings(settings), _deadline(timeout_seconds)
    _cohort(x, y, settings)
    _draws(interval_draws)
    _require(isinstance(scanner, pd.Series) and scanner.index.equals(x.index), "Aligned scanner Series required.")
    _require(not scanner.isna().any() and all(isinstance(v, str) and v for v in scanner), "Scanner labels must be strings.")
    scanners, yy = np.asarray(scanner), np.asarray(y, int)
    _require(len(set(scanners)) == 2, "Exactly two scanners required for bidirectional validation.")
    records = []
    for direction, held in enumerate(sorted(set(scanners))):
        check()
        tr, te = np.flatnonzero(scanners != held), np.flatnonzero(scanners == held)
        rec = {"heldout_scanner": str(held), "training_scanner": str(scanners[tr[0]]),
               "train_rows": tr.tolist(), "test_rows": te.tolist(), "n_train": len(tr), "n_test": len(te),
               "train_responders": int(yy[tr].sum()), "test_responders": int(yy[te].sum()), "models": {}}
        predictions = {}
        for model in ("C", "CP"):
            fit = develop(x.iloc[tr], y.iloc[tr], tr, x.iloc[te], model,
                          settings.seed + 7000 + direction, settings, deadline=check)
            predictions[model] = fit["evaluation_predictions"]
            fit["metrics"] = measures(yy[te], predictions[model], settings.clip)
            rec["models"][model] = fit
        rec["conditional_intervals"] = conditional_intervals(yy[te], predictions, draws=interval_draws,
                                                               seed=interval_seed, deadline=check)
        c, cp = (rec["models"][m]["metrics"] for m in ("C", "CP"))
        rec["delta_auc_CP_minus_C"] = cp["auc"] - c["auc"] if c["auc"] is not None else None
        rec["delta_brier_C_minus_CP"] = c["brier"] - cp["brier"]
        rec["training_prevalence_brier"] = float(np.mean((yy[te] - yy[tr].mean())**2))
        records.append(rec)
    check()
    return {"status": "PASS", "directions": records,
            "scope": "Same-center cross-scanner stress test, not independent external validation"}


def ablation_columns(settings=None):
    """The three executed pathway removals, retaining all clinical predictors."""
    settings = _settings(settings)
    _require(len(settings.pet) == 6 and settings.pet[0].endswith("__vns_thalamus")
             and settings.pet[4].endswith("__vns_striatopallidal"), "Expected frozen six-pathway order.")
    return {"CP_without_thalamus": list(settings.clinical) + [p for i, p in enumerate(settings.pet) if i != 0],
            "CP_without_striatopallidal": list(settings.clinical) + [p for i, p in enumerate(settings.pet) if i != 4],
            "CP_without_both": list(settings.clinical) + [p for i, p in enumerate(settings.pet) if i not in (0, 4)]}


def _pair_auc(y, p):
    diff = p[y == 1, None] - p[y == 0][None, :]
    return float(np.mean((diff > 0) + .5 * (diff == 0)))


def evaluate_ablations(result, y, *, draws=2000, seed=20260910, clip=1e-6, deadline=None):
    """Original CP-minus-ablation paired fixed-OOF resampling (both signs CP minus)."""
    _draws(draws)
    _require(result["status"] == "PASS" and isinstance(y, pd.Series)
             and list(y.index) == result["participant_ids"], "Completed aligned model result required.")
    expected = {"CP", "CP_without_thalamus", "CP_without_striatopallidal", "CP_without_both"}
    matrices = {m: np.asarray(p, float) for m, p in result["predictions"].items()}
    _require(set(matrices) == expected and set(result["model_split_digests"]) == expected
             and len(set(result["model_split_digests"].values())) == 1,
             "Exact ablation set on identical supplied partitions required.")
    yy = np.asarray(y, int)
    _require(set(np.asarray(y).tolist()) == {0, 1}, "Both outcome classes required.")
    _require(len({p.shape for p in matrices.values()}) == 1, "Model probability matrices must have identical shape.")
    for p in matrices.values():
        _require(p.ndim == 2 and p.shape[1] == len(y) and p.shape[0] > 0
                 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid OOF probabilities.")
    means = {name: mat.mean(0) for name, mat in matrices.items()}
    rng = np.random.default_rng(seed)
    indices = []
    for _ in range(draws):
        if deadline is not None:
            deadline()
        ix = rng.integers(0, len(y), len(y))
        if len(set(yy[ix])) == 2:
            indices.append(ix)
    _require(bool(indices), "No evaluable paired bootstrap draws; no replacement draws attempted.")
    models, bs = {}, {}
    for name, mat in matrices.items():
        p, a = means[name], _pair_auc(yy, means[name])
        _require(abs(a - roc_auc_score(yy, p)) < 1e-12, "Pair-count AUC disagreement.")
        scores = []
        for ix in indices:
            if deadline is not None:
                deadline()
            scores.append([_pair_auc(yy[ix], p[ix]), np.mean((yy[ix] - p[ix])**2)])
        bs[name] = np.asarray(scores)
        q, ra = np.quantile(bs[name], [.025, .975], axis=0), [_pair_auc(yy, row) for row in mat]
        models[name] = {"auc": a, "brier": float(np.mean((yy - p)**2)), "auc_conditional95": q[:, 0].tolist(),
                        "brier_conditional95": q[:, 1].tolist(), "repeat_auc": ra, "repeat_auc_mean": float(np.mean(ra)),
                        "repeat_auc_range": [min(ra), max(ra)], "calibration": core.calibration(yy, p, clip),
                        "mean_predictions": p.tolist()}
    comparisons = {}
    for name in matrices:
        if name == "CP":
            continue
        q = np.quantile(bs["CP"] - bs[name], [.025, .975], axis=0)
        comparisons["CP_minus_" + name] = {"delta_auc": models["CP"]["auc"] - models[name]["auc"],
            "delta_brier": models["CP"]["brier"] - models[name]["brier"], "delta_auc_conditional95": q[:, 0].tolist(),
            "delta_brier_conditional95": q[:, 1].tolist(),
            "repeat_auc_differences": (np.array(models["CP"]["repeat_auc"]) - models[name]["repeat_auc"]).tolist()}
    return {"models": models, "comparisons": comparisons, "bootstrap_valid": len(indices),
            "bootstrap_seed": seed, "bootstrap_requested": draws, "one_class_draws": draws - len(indices),
            "uncertainty": "Fixed-OOF paired patient bootstrap, conditional descriptive only. Not selection/history-adjusted.",
            "positive_delta_auc": "Retaining the removed pathway(s) improves full CP AUC",
            "negative_delta_brier": "Retaining the removed pathway(s) improves full CP Brier", "no_best_model_promotion": True}


def run_ablations(x, y, plans, settings=None, *, interval_draws=2000, interval_seed=20260910, timeout_seconds=1800):
    """Retune full CP and all three ablations on the same explicitly supplied splits.

    Unlike the historical runner, full CP is refitted instead of reading its old
    private saved predictions. The supplied plans are never regenerated.
    """
    settings, check = _settings(settings), _deadline(timeout_seconds)
    _require(plans is not None, "Explicit shared split plan required.")
    _draws(interval_draws)
    columns = {"CP": settings.model_columns()["CP"], **ablation_columns(settings)}
    result = run_nested_models(x, y, settings, model_columns=columns, plans=plans, timeout_seconds=timeout_seconds)
    check()
    result["ablation_metrics"] = evaluate_ablations(result, y, draws=interval_draws, seed=interval_seed,
                                                   clip=settings.clip, deadline=check)
    check()
    return result


def _distribution(values):
    v = np.asarray(values, float)
    if not len(v):
        return {"n": 0, "mean": None, "sd": None, "percentiles_2p5_50_97p5": None, "min": None, "max": None}
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
            "percentiles_2p5_50_97p5": np.quantile(v, [.025, .5, .975]).tolist(), "min": float(v.min()), "max": float(v.max())}


def _mcse(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None


def run_development_bootstrap(x, y, settings=None, *, draws=500, seed=20260921,
                              reference_inner_seed=20260910, timeout_seconds=1800, progress=None):
    """Full-development ordinary paired-person bootstrap, with no redraw on rejection.

    The default reference/draw/inner seeds and optimism arithmetic match the
    executed source. Each draw tunes again with duplicate-person grouped inner
    folds. Returned probability ranges mix in-bag and OOB predictions and are
    stability descriptions, not prediction intervals or corrected-OOF intervals.
    """
    settings, check = _settings(settings), _deadline(timeout_seconds)
    _draws(draws)
    _cohort(x, y, settings)
    yy, n = np.asarray(y, int), len(y)
    columns = settings.model_columns()
    ref, records, invalid = {}, [], []
    coefficients = {m: [] for m in ("C", "CP")}
    for model in ("C", "CP"):
        ref[model] = develop(x, y, np.arange(n), x, model, reference_inner_seed, settings, deadline=check)
        ref[model]["metrics"] = measures(yy, ref[model]["evaluation_predictions"], settings.clip)
    rng = np.random.default_rng(seed)
    for serial in range(draws):
        check()
        sample = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), sample)
        inner_seed = settings.seed + 200000 + serial
        try:
            grouped_inner_plan(yy[sample], sample, inner_seed)
        except core.ClassSupportError as exc:
            invalid.append({"draw": serial, "sample_rows": sample.tolist(), "reason": str(exc), "no_replacement_draw": True})
            continue
        row = {"draw": serial, "sample_rows": sample.tolist(), "oob_rows": oob.tolist(),
               "unique_training_n": len(set(sample.tolist())), "models": {}}
        for model in ("C", "CP"):
            fit = develop(x.iloc[sample], y.iloc[sample], sample, x, model, inner_seed, settings, deadline=check)
            pp = np.asarray(fit["evaluation_predictions"])
            fit["inbag"] = measures(yy[sample], pp[sample], settings.clip)
            fit["original"] = measures(yy, pp, settings.clip)
            fit["oob"] = measures(yy[oob], pp[oob], settings.clip)
            fit["auc_optimism"] = fit["inbag"]["auc"] - fit["original"]["auc"]
            fit["brier_optimism"] = fit["original"]["brier"] - fit["inbag"]["brier"]
            row["models"][model] = fit
            kept = [c for c in columns[model] if c not in fit["preprocessing_audit"][-1]["dropped"]]
            vals = dict(zip(kept, fit["numerical_fits"][-1]["coefficients"]))
            coefficients[model].append([vals.get(c, 0.) for c in columns[model]])
        records.append(row)
        if progress is not None:
            progress({"draws_attempted": serial + 1, "draws_requested": draws, "draws_valid": len(records)})
    if not records:
        error = core.ClassSupportError("No valid bootstrap draws; no automatic retry.")
        error.invalid_draws = invalid
        error.draws_requested = draws
        raise error
    summary = {"n": n, "responders": int(yy.sum()), "draws_requested": draws, "draws_valid": len(records),
               "invalid_draws": invalid, "bootstrap": {}, "full_data_reference": {}, "paired_increment": {}}
    stability = {}
    for model in ("C", "CP"):
        rr = [row["models"][model] for row in records]
        optim, boptim = (np.asarray([v[key] for v in rr]) for key in ("auc_optimism", "brier_optimism"))
        allp = np.asarray([v["evaluation_predictions"] for v in rr])
        basep = np.asarray(ref[model]["evaluation_predictions"])
        co = np.asarray(coefficients[model])
        width = np.quantile(allp, .975, axis=0) - np.quantile(allp, .025, axis=0)
        b = {"auc_optimism": _distribution(optim), "auc_optimism_mcse": _mcse(optim),
             "brier_optimism": _distribution(boptim), "brier_optimism_mcse": _mcse(boptim),
             "optimism_corrected_auc": ref[model]["metrics"]["auc"] - float(optim.mean()),
             "optimism_corrected_brier": ref[model]["metrics"]["brier"] + float(boptim.mean()),
             "parameter_frequencies": dict(Counter(f"C={v['selected']['C']};l1={v['selected']['l1_ratio']}" for v in rr)),
             "OOB_auc_distribution": _distribution([v["oob"]["auc"] for v in rr if v["oob"]["auc"] is not None]),
             "OOB_auc_nonestimable": sum(v["oob"]["auc"] is None for v in rr),
             "unique_training_n": _distribution([row["unique_training_n"] for row in records]),
             "probability_stability_scope": "All original-cohort predictions from each bootstrap-trained model, mix of inbag/OOB; not predictive confidence intervals",
             "mean_absolute_probability_change_from_reference": _distribution(np.abs(allp - basep).mean(1)),
             "per_patient_95percentile_width": _distribution(width),
             "coefficient_stability": {c: {"negative_fraction": float(np.mean(co[:, j] < 0)),
                 "zero_fraction": float(np.mean(co[:, j] == 0)), "positive_fraction": float(np.mean(co[:, j] > 0))}
                 for j, c in enumerate(columns[model])}}
        for where in ("inbag", "original", "oob"):
            b[where + "_calibration_status"] = dict(Counter(v[where]["calibration"]["status"] for v in rr))
        summary["bootstrap"][model] = b
        summary["full_data_reference"][model] = {"selected": ref[model]["selected"], "metrics": ref[model]["metrics"]}
        stability[model] = {"reference_predictions": basep.tolist(),
            "bootstrap_percentiles_2p5_50_97p5": np.quantile(allp, [.025, .5, .975], axis=0).tolist(),
            "not_external_validation_or_prediction_intervals": True}
    optdiff = np.asarray([row["models"]["CP"]["auc_optimism"] - row["models"]["C"]["auc_optimism"] for row in records])
    summary["paired_increment"] = {
        "apparent_delta_auc_CP_minus_C": ref["CP"]["metrics"]["auc"] - ref["C"]["metrics"]["auc"],
        "optimism_corrected_delta_auc_CP_minus_C": summary["bootstrap"]["CP"]["optimism_corrected_auc"] - summary["bootstrap"]["C"]["optimism_corrected_auc"],
        "optimism_corrected_delta_brier_C_minus_CP": summary["bootstrap"]["C"]["optimism_corrected_brier"] - summary["bootstrap"]["CP"]["optimism_corrected_brier"],
        "paired_auc_optimism_difference_mcse": _mcse(optdiff)}
    check()
    return {"status": "PASS", "summary": summary, "reference_fits": ref, "records": records,
            "probability_stability": stability, "participant_ids": list(x.index),
            "seed": seed, "reference_inner_seed": reference_inner_seed,
            "interpretation_limits": ["Corrects full-data apparent development performance, not original OOF performance",
                "Monte Carlo SE measures simulation precision, not clinical sampling uncertainty",
                "Resampling ranges are not validated confidence intervals for corrected performance",
                "Historical feature, method and label-review choices are not replayed",
                "OOB performance reflects fewer distinct training people, not independent external validation"]}
