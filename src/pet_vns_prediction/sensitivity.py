"""Explicit association families and the historical processing-prediction engine.

Only in-memory inputs and outputs are used. The caller owns row alignment,
eligibility, feature extraction, and secure storage of person-level diagnostics.
No image processing, outcome-dependent family reduction, or case removal occurs.
"""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.metrics import brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from . import core, solver
from .associations import bh_adjust, hc3_fit
from .modeling import paper_settings


PROCESSING_VARIANTS = (
    "original", "wholeBrain", "cerebellarCortex", "same_seg_noPVC",
    "pvc54", "pvc4", "pvc6", "petprep",
)
PREDICTION_VARIANTS = ("same_seg_noPVC", "pvc54", "pvc4", "pvc6", "wholeBrain", "cerebellarCortex", "petprep")
MORPHOMETRIC_ADJUSTMENTS = ("clinical5", "clinical5_scanner_t1")
PROCESSING_ADJUSTMENTS = ("base", "base_plus_corresponding_T1")
CLINICAL_ADJUSTMENTS = (
    "base_plus_epilepsy_type", "base_plus_etiology3",
    "base_plus_epilepsy_type_and_etiology3",
)
CLINICAL_NAMES = (
    "age_at_implant_years", "male", "epilepsy_duration_reported_years",
    "log1p_baseline_monthly_frequency", "mri_lesional", "scanner_indicator",
)


def _finite_matrix(value, n, width, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (n, width) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite n-by-{width} matrix; no implicit row removal.")
    return array


def _response(value):
    y = np.asarray(value, dtype=float)
    if y.ndim != 1 or not np.array_equal(np.unique(y), [0., 1.]):
        raise ValueError("Response must be a complete explicit 0/1 vector with both groups.")
    if min(np.sum(y == 0), np.sum(y == 1)) < 2:
        raise ValueError("Each group needs at least two observations for pooled SD.")
    return y


def _names(region_names):
    names = tuple(region_names) if region_names is not None else tuple(paper_settings().pet)
    if len(names) != 6 or len(set(names)) != 6 or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Exactly six unique named regions are required in fixed PET/T1 order.")
    return names


def _designs(response, clinical5, scanner):
    y = _response(response)
    clinical = _finite_matrix(clinical5, len(y), 5, "Clinical covariates").copy()
    indicator = np.asarray(scanner, dtype=float)
    if indicator.shape != y.shape or not np.isin(indicator, [0., 1.]).all():
        raise ValueError("Scanner must be an explicitly encoded 0/1 indicator.")
    if not np.isin(clinical[:, [1, 4]], [0., 1.]).all():
        raise ValueError("Clinical columns 1 and 4 must encode male and lesional indicators.")
    # Full complete-case association design, not prediction-fold preprocessing.
    for index in (0, 2, 3):
        sd = clinical[:, index].std(ddof=1)
        if sd <= 0:
            raise ValueError("Continuous clinical covariates must have positive sample SD.")
        clinical[:, index] = (clinical[:, index] - clinical[:, index].mean()) / sd
    clinical_design = np.c_[np.ones(len(y)), y, clinical]
    return y, clinical_design, np.c_[clinical_design, indicator]


def _standardized_t1(t1, n):
    values = _finite_matrix(t1, n, 6, "Corresponding T1 features")
    if np.any(values.std(axis=0, ddof=1) <= 0):
        raise ValueError("Every corresponding T1 feature requires positive sample SD.")
    return stats.zscore(values, axis=0, ddof=1)


def _pooled_sd(value, response):
    positive, negative = value[response == 1], value[response == 0]
    sd = np.sqrt(((len(positive)-1)*positive.var(ddof=1)
                  + (len(negative)-1)*negative.var(ddof=1)) / (len(response)-2))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Each regional measurement requires positive pooled within-group SD.")
    return float(sd)


def leave_one_person_influence(design, values, response, *, covariate_names=None):
    """Source-formula OLS influence diagnostics; never recommend exclusions.

Column 0 is the intercept and column 1 is response. The full-sample pooled
within-group SD and already-scaled design are retained in every leave-one-out
fit. LOO coefficients use lstsq even if a deletion reduces rank (reported).
Returned row positions, residuals and fitted values can be sensitive outputs.
    """
    y = _response(response)
    a, value = np.asarray(design, float), np.asarray(values, float)
    if a.ndim != 2 or len(a) != len(y) or a.shape[1] < 2 or value.shape != y.shape:
        raise ValueError("Aligned design, measurement and response are required.")
    if not np.array_equal(a[:, 0], np.ones(len(y))) or not np.array_equal(a[:, 1], y):
        raise ValueError("Design must start with intercept and response columns.")
    f, sd = hc3_fit(a, value), _pooled_sd(value, y)
    loo, ranks = [], []
    for k in range(len(y)):
        aa, vv = np.delete(a, k, 0), np.delete(value, k)
        loo.append(float(np.linalg.lstsq(aa, vv, rcond=None)[0][1] / sd))
        ranks.append(int(np.linalg.matrix_rank(aa)))
    names = tuple(covariate_names) if covariate_names is not None else tuple(f"column_{j}" for j in range(a.shape[1]))
    if len(names) != a.shape[1] or len(set(names)) != len(names):
        raise ValueError("Unique covariate names must match the entire design including intercept.")
    vif = {}
    for j in range(1, a.shape[1]):
        rest, z = np.delete(a, j, axis=1), a[:, j]
        error = z - rest @ np.linalg.lstsq(rest, z, rcond=None)[0]
        vif[names[j]] = float(np.sum((z-z.mean())**2) / np.dot(error, error))
    return {
        "rank": int(np.linalg.matrix_rank(a)), "columns": a.shape[1],
        "condition": float(np.linalg.cond(a)), "max_leverage": float(max(f["h"])),
        "VIF": vif, "max_Cooks_distance": float(max(f["cooks"])),
        "Cooks_gt_4_over_n_row_positions": np.flatnonzero(f["cooks"] > 4/len(y)).tolist(),
        "residual_skew": float(stats.skew(f["res"], bias=False)),
        "residual_excess_kurtosis": float(stats.kurtosis(f["res"], bias=False)),
        "fitted": f["fitted"].tolist(), "residuals": f["res"].tolist(),
        "full_sample_pooled_SD": sd,
        "leave_one_out_standardized_differences": loo,
        "leave_one_out_standardized_range": [min(loo), max(loo)],
        "leave_one_out_sign_changes": int(sum(np.sign(v) != np.sign(f["beta"][1]) for v in loo)),
        "leave_one_out_largest_change_row_position": int(np.argmax(abs(np.asarray(loo)-f["beta"][1]/sd))),
        "leave_one_out_design_ranks": ranks,
        "case_exclusions": [], "interpretation": "Descriptive influence only; no exclusion recommendation.",
    }


def _association_row(value, y, design, *, region, family, variant, adjustment):
    sd, f = _pooled_sd(value, y), hc3_fit(design, value)
    if not np.isfinite(f["p"]).all():
        raise ValueError("Nonfinite inference; the complete family cannot be reported.")
    return {"region": region, "family": family, "variant": variant, "adjustment": adjustment,
            "n": len(y), "responders": int(y.sum()), "nonresponders": int((1-y).sum()),
            "df": f["df"], "responder_mean": float(value[y == 1].mean()),
            "nonresponder_mean": float(value[y == 0].mean()), "pooled_SD": sd,
            "difference": float(f["beta"][1]), "HC3_SE": float(f["se"][1]),
            "CI95": [float(f["lo"][1]), float(f["hi"][1])],
            "standardized_difference": float(f["beta"][1]/sd),
            "standardized_CI95": [float(f["lo"][1]/sd), float(f["hi"][1]/sd)],
            "p": float(f["p"][1])}


def _adjust(rows, count, field="q_family"):
    q = bh_adjust([row["p"] for row in rows], expected_tests=count)
    for row, value in zip(rows, q):
        row[field] = float(value)
        if field == "q_family":
            row["family_tests"] = count
    return rows


def clinical_morphometric_family(regional_values, response, clinical5, scanner, t1, *, region_names=None):
    """Joint 12 tests: six clinical-only and six clinical+scanner+matching-T1.

This is not the six-test primary clinical+scanner family. Clinical columns are
age, male indicator, duration, log1p frequency, and lesional indicator, in order.
    """
    y, clinical, base = _designs(response, clinical5, scanner)
    values, morphometry = _finite_matrix(regional_values, len(y), 6, "PET"), _standardized_t1(t1, len(y))
    rows = []
    for i, name in enumerate(_names(region_names)):
        for adjustment, design in ((MORPHOMETRIC_ADJUSTMENTS[0], clinical),
                                   (MORPHOMETRIC_ADJUSTMENTS[1], np.c_[base, morphometry[:, i]])):
            rows.append(_association_row(values[:, i], y, design, region=name,
                                         family="clinical_morphometric", variant="original", adjustment=adjustment))
    return _adjust(rows, 12)


def extended_clinical_family(regional_values, response, clinical5, scanner, epilepsy_type, etiology3, *, region_names=None):
    """All 18 tests, with two fixed dummy columns for each three-level category.

Epilepsy columns encode focal and combined-generalized-and-focal, with generalized
reference. Etiology columns encode structural and other-known, with unknown
reference. Category grouping is the caller's explicit upstream responsibility.
    """
    y, _, base = _designs(response, clinical5, scanner)
    values = _finite_matrix(regional_values, len(y), 6, "PET")
    typecols = _finite_matrix(epilepsy_type, len(y), 2, "Epilepsy category dummies")
    eticols = _finite_matrix(etiology3, len(y), 2, "Etiology category dummies")
    for array in (typecols, eticols):
        if not np.isin(array, [0., 1.]).all() or np.any(array.sum(axis=1) > 1):
            raise ValueError("Two mutually exclusive 0/1 dummies are required per category.")
    designs = (np.c_[base, typecols], np.c_[base, eticols], np.c_[base, typecols, eticols])
    rows = [_association_row(values[:, i], y, design, region=name, family="clinical",
                             variant="original", adjustment=adjustment)
            for adjustment, design in zip(CLINICAL_ADJUSTMENTS, designs)
            for i, name in enumerate(_names(region_names))]
    return _adjust(rows, 18)


def processing_association_family(regional_variants: Mapping, response, clinical5, scanner, t1, *, region_names=None, influence=True):
    """All 96 tests: eight fixed variants x six regions x two adjustments.

Every variant is mandatory, including PVC 4 and 6. BH always uses all 96 tests,
regardless of which rows are later displayed. Each variant has its own pooled SD.
    """
    if set(regional_variants) != set(PROCESSING_VARIANTS):
        raise ValueError("Exactly all eight declared processing variants are required.")
    y, _, base = _designs(response, clinical5, scanner)
    morphometry, names = _standardized_t1(t1, len(y)), _names(region_names)
    rows, diagnostics = [], []
    for variant in PROCESSING_VARIANTS:
        values = _finite_matrix(regional_variants[variant], len(y), 6, variant)
        for i, name in enumerate(names):
            for adjustment, design in ((PROCESSING_ADJUSTMENTS[0], base),
                                       (PROCESSING_ADJUSTMENTS[1], np.c_[base, morphometry[:, i]])):
                row = _association_row(values[:, i], y, design, region=name, family="processing",
                                       variant=variant, adjustment=adjustment)
                rows.append(row)
                if influence:
                    cn = ("intercept", "response") + CLINICAL_NAMES
                    if adjustment == PROCESSING_ADJUSTMENTS[1]:
                        cn += ("corresponding_T1",)
                    diagnostics.append({key: row[key] for key in ("family", "variant", "adjustment", "region")} |
                                       leave_one_person_influence(design, values[:, i], y, covariate_names=cn))
    return {"rows": _adjust(rows, 96), "diagnostics": diagnostics,
            "family_tests": 96, "processing_variants": list(PROCESSING_VARIANTS)}


def followup_association_families(regional_variants, response, clinical5, scanner, t1, epilepsy_type, etiology3, *, region_names=None, influence=True):
    """Separate 18/96-test BH families plus the source's secondary all-114 BH.

The 114-test column does not replace either family-specific correction.
    """
    processing = processing_association_family(regional_variants, response, clinical5, scanner, t1,
                                                region_names=region_names, influence=influence)
    clinical = extended_clinical_family(regional_variants["original"], response, clinical5, scanner,
                                        epilepsy_type, etiology3, region_names=region_names)
    rows = _adjust(clinical + processing["rows"], 114, "q_all114")
    diagnostics = []
    if influence:
        y, _, base = _designs(response, clinical5, scanner)
        values = np.asarray(regional_variants["original"], float)
        typecols, eticols = np.asarray(epilepsy_type, float), np.asarray(etiology3, float)
        types = ("type_focal", "type_combined")
        etiologies = ("etiology_structural", "etiology_other_known")
        designs = ((np.c_[base, typecols], types), (np.c_[base, eticols], etiologies),
                   (np.c_[base, typecols, eticols], types + etiologies))
        for adjustment, (design, extra_names) in zip(CLINICAL_ADJUSTMENTS, designs):
            for i, name in enumerate(_names(region_names)):
                diagnostics.append({"family": "clinical", "variant": "original", "adjustment": adjustment, "region": name} |
                                   leave_one_person_influence(design, values[:, i], y,
                                       covariate_names=("intercept", "response") + CLINICAL_NAMES + extra_names))
        diagnostics.extend(processing["diagnostics"])
    return {"rows": rows, "diagnostics": diagnostics,
            "family_test_counts": {"clinical": 18, "processing": 96}, "all_tests": 114}


class PlainColumnsTransformer:
    """Only the plain-column branch of the historical processing transformer."""

    def __init__(self, definition, settings):
        if set(definition) != {"columns", "tuning"} or definition["tuning"] != "brier":
            raise ValueError("Processing definitions require only columns and Brier tuning; PCA/AUC branches are not enabled.")
        self.definition, self.settings = definition, settings

    def fit(self, x):
        self.plain = core.FoldPreprocessor(self.definition["columns"], self.settings.categories).fit(x)
        return self

    def transform(self, x):
        return self.plain.transform(x)

    def state(self):
        return {"kind": "original_training_only_preprocessing", **self.plain.state()}


def select_processing_candidate(candidates):
    """Complete finite Brier/AUC grid; Brier tie then smaller C, then l1 ratio."""
    if not candidates or any(not np.isfinite(v["mean_brier"]) or not np.isfinite(v["mean_auc"]) for v in candidates):
        raise core.FitFailure("Incomplete/nonfinite tuning candidate.")
    best = min(v["mean_brier"] for v in candidates)
    pool = [v for v in candidates if v["mean_brier"] <= best + 1e-8]
    return min(pool, key=lambda v: (v["C"], v["l1_ratio"]))


def run_processing_outer(x, y, definition, settings, plan, deadline=lambda: None):
    """Exact plain-column processing branch; distinct from corrected_outer.

Both inner metrics are retained. The selected solution is repeated exactly and
checked independently on train and held-out probabilities, as in the source.
The caller validates the complete shared split plan before using this primitive.
    """
    if not __debug__:
        raise RuntimeError("Required numerical checks must not be disabled with Python -O.")
    y = np.asarray(y)
    prepared, audits = [], []
    for j, pair in enumerate(plan["inner"]):
        transformer = PlainColumnsTransformer(definition, settings).fit(x.iloc[pair["train"]])
        audits.append({"stage": "inner", "inner_fold": j, "fit_rows": pair["train"], **transformer.state()})
        prepared.append((transformer.transform(x.iloc[pair["train"]]), y[pair["train"]],
                         transformer.transform(x.iloc[pair["validation"]]), y[pair["validation"]]))
    candidates, fits = [], []
    for c in settings.c_grid:
        for l1 in settings.l1_grid:
            briers, aucs = [], []
            for j, (tx, ty, vx, vy) in enumerate(prepared):
                deadline()
                fitted = solver.fit(tx, ty, c, l1)
                p = fitted.predict_proba(vx)[:, 1]
                briers.append(float(brier_score_loss(vy, p)))
                aucs.append(float(roc_auc_score(vy, p)))
                fits.append({"stage": "inner", "inner_fold": j, "validation_predictions": p.tolist(), **fitted.numerical_})
            candidates.append({"C": float(c), "l1_ratio": float(l1), "mean_brier": float(np.mean(briers)),
                               "mean_auc": float(np.mean(aucs)), "inner_briers": briers, "inner_aucs": aucs})
    chosen = select_processing_candidate(candidates)
    transformer = PlainColumnsTransformer(definition, settings).fit(x.iloc[plan["train"]])
    audits.append({"stage": "outer_refit", "fit_rows": plan["train"], **transformer.state()})
    tx, ty, vx = transformer.transform(x.iloc[plan["train"]]), y[plan["train"]], transformer.transform(x.iloc[plan["test"]])
    deadline()
    fitted = solver.fit(tx, ty, chosen["C"], chosen["l1_ratio"])
    p = fitted.predict_proba(vx)[:, 1]
    fits.append({"stage": "outer_refit", **fitted.numerical_})
    repeated = solver.fit(tx, ty, chosen["C"], chosen["l1_ratio"])
    assert np.array_equal(repeated.coef_, fitted.coef_) and np.array_equal(repeated.intercept_, fitted.intercept_)
    assert np.array_equal(repeated.predict_proba(vx)[:, 1], p)
    w, independent = solver.independent_slsqp(tx, ty, chosen["C"], chosen["l1_ratio"])
    train_diff = float(np.max(np.abs(expit(w[0] + tx @ w[1:]) - fitted.predict_proba(tx)[:, 1])))
    test_diff = float(np.max(np.abs(expit(w[0] + vx @ w[1:]) - p)))
    obj_diff = abs(independent["objective_mean"] - fitted.numerical_["objective_mean"])
    if not independent["success"] or max(train_diff, test_diff) > 1e-5 or obj_diff > 1e-10:
        raise core.FitFailure(f"Independent selected-fit agreement failed: {train_diff}, {test_diff}, {obj_diff}.")
    return {"selected": chosen, "candidates": candidates, "audit": audits, "numerical_fits": fits,
            "predictions": p.tolist(), "test_rows": plan["test"], "exact_selected_repetition": True,
            "independent_selected": {**independent, "intercept": float(w[0]), "coefficients": w[1:].tolist(),
                                     "train_probability_max_difference": train_diff, "test_probability_max_difference": test_diff,
                                     "objective_abs_difference": obj_diff}}


def processing_model_definitions(settings=None):
    """The complete 14 models, with canonical six PET names in every frame."""
    settings = paper_settings() if settings is None else settings
    return {variant + "_" + model: {"variant": variant, "model": model,
                                    "definition": {"columns": list(settings.model_columns()[model]), "tuning": "brier"}}
            for variant in PREDICTION_VARIANTS for model in ("CP", "CPT")}


def processing_prediction_comparisons():
    """All 28 source processing comparisons, including unplotted PVC 4/6."""
    pairs = []
    for variant in PREDICTION_VARIANTS[:4]:
        for model in ("CP", "CPT"):
            pairs.append({"a": variant + "_" + model, "b": "CORE_" + model, "seed": 20260910})
            if variant != "same_seg_noPVC":
                pairs.append({"a": variant + "_" + model, "b": "same_seg_noPVC_" + model, "seed": 20260910})
    for variant in PREDICTION_VARIANTS:
        pairs.append({"a": variant + "_CP", "b": "CORE_C", "seed": 20260910})
        pairs.append({"a": variant + "_CPT", "b": variant + "_CP", "seed": 20260910})
    return pairs


def run_processing_predictions(original, processing_pet: Mapping, response, settings=None, *, plans=None, timeout_seconds_per_model=3600, progress=None):
    """Run all 14 processing models on aligned in-memory PET variant tables.

Each table must have exactly the six canonical settings.pet columns and the
original index/order. Only PET is replaced; clinical and T1 values stay fixed.
Original CORE_C/CP/CPT predictions use their separate original engine and are
not refit by this call. No partial family or automatic fallback is accepted.
    """
    settings = paper_settings() if settings is None else settings
    if not isinstance(original, pd.DataFrame) or not isinstance(response, pd.Series):
        raise core.InputRejected("Indexed predictor table and response series required.")
    if set(processing_pet) != set(PREDICTION_VARIANTS):
        raise core.InputRejected("All seven alternative processing PET tables are required.")
    if timeout_seconds_per_model <= 0:
        raise ValueError("Positive per-model time limit required.")
    if any(not isinstance(value, str) or not value for value in original.index):
        raise core.InputRejected("Nonempty string row identifiers required; none are written automatically.")
    core.validate_inputs(original, response, settings, settings.model_columns()["CPT"])
    frames = {}
    for variant in PREDICTION_VARIANTS:
        table = processing_pet[variant]
        if not isinstance(table, pd.DataFrame) or not table.index.equals(original.index) or tuple(table.columns) != settings.pet:
            raise core.InputRejected("Variant index/order and canonical PET column order must exactly match.")
        frames[variant] = original.copy()
        frames[variant].loc[:, list(settings.pet)] = table.to_numpy()
        core.validate_inputs(frames[variant], response, settings, settings.model_columns()["CPT"])
    plans = core.make_split_plan(response, settings) if plans is None else plans
    core.validate_split_plan(plans, response, settings.repeats, settings.inner_folds)
    definitions = processing_model_definitions(settings)
    predictions, fold_records, audits = {}, [], []
    with threadpool_limits(limits=1):
        for name, route in definitions.items():
            started = time.monotonic()
            matrix, seen = np.full((settings.repeats, len(original)), np.nan), np.zeros((settings.repeats, len(original)), int)
            def deadline():
                if time.monotonic() - started >= timeout_seconds_per_model:
                    raise TimeoutError("Processing model time limit exceeded; no automatic retry.")
            for serial, plan in enumerate(plans):
                deadline()
                record = run_processing_outer(frames[route["variant"]], response.to_numpy(), route["definition"], settings, plan, deadline)
                p = np.asarray(record["predictions"])
                if p.shape != (len(plan["test"]),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise core.FitFailure("Invalid held-out probabilities.")
                matrix[plan["repeat"], plan["test"]] = p
                seen[plan["repeat"], plan["test"]] += 1
                record.update({"model": name, "repeat": plan["repeat"], "fold": plan["fold"], "shared_train_rows": plan["train"]})
                fold_records.append(record)
                audits.extend({"model": name, "repeat": plan["repeat"], "fold": plan["fold"], **audit} for audit in record["audit"])
                if progress is not None:
                    progress({"model": name, "completed_outer_splits": serial + 1, "total_outer_splits": len(plans)})
            if not np.all(seen == 1) or not np.isfinite(matrix).all():
                raise core.FitFailure("Incomplete or duplicate held-out prediction coverage.")
            predictions[name] = matrix
    digest = sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()
    result = {"status": "PASS", "profile": settings.profile, "predictions": predictions,
              "participant_ids": list(original.index), "plans": plans, "split_digest": digest,
              "model_split_digests": {name: digest for name in definitions}, "audit": audits,
              "fold_records": fold_records, "model_definitions": definitions, "failures": [],
              "comparisons": processing_prediction_comparisons(),
              "engine": "historical_plain_columns_processing_with_train_and_test_independent_check"}
    core.assert_training_isolation(result)
    return result
