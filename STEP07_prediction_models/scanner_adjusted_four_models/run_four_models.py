"""Manually run the PET-scanner-adjusted C/CT/CP/CPT comparison.

Each command performs only its named step. Inputs and fit records are private.
The original three-model STEP07 remains a separate, unchanged analysis.
"""
from __future__ import annotations

if not __debug__:
    raise RuntimeError("Required numerical assertions are disabled; do not use Python -O or -OO")

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
import traceback

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE))
import step_support
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from pet_vns_prediction import core
from pet_vns_prediction.modeling import paper_settings
from pet_vns_prediction.cli import _validated_input, environment

SCANNER = "pet_scanner"
LEVELS = ("Biograph HiRes", "Biograph Vision 600")
MODELS = ("C", "CT", "CP", "CPT")
COMPARISONS = (("CT", "CPT"), ("C", "CP"), ("CP", "CPT"), ("C", "CT"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, value, *, boundary):
    path, boundary = Path(path).resolve(), Path(boundary).resolve()
    if path == boundary or boundary not in path.parents:
        raise ValueError("Output must remain within the explicitly selected directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=serial, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def plan_digest(plans):
    return sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()


def new_private_output(path):
    path = Path(path).resolve()
    if path == PACKAGE or PACKAGE in path.parents:
        raise ValueError("Keep participant inputs/results outside the public code package")
    if path.exists():
        raise FileExistsError("Use a fresh output directory; no implicit resume or overwrite")
    path.mkdir(parents=True)
    return path


def code_fingerprints():
    files = sorted(set((PACKAGE / "src").rglob("*.py")) |
                   set(Path(__file__).resolve().parent.glob("*.py")) |
                   {PACKAGE / "step_support.py"},
                   key=lambda path: path.relative_to(PACKAGE).as_posix())
    return [{"path": str(path.relative_to(PACKAGE)).replace("\\", "/"),
             "sha256": digest(path)} for path in files]


def read_scanner_metadata(path):
    metadata = pd.read_csv(path, dtype={"study_id": str}, keep_default_na=False)
    required = {"study_id", "scanner"}
    if not required.issubset(metadata.columns):
        raise ValueError("Scanner CSV requires study_id and scanner columns")
    if not metadata.study_id.is_unique or metadata.study_id.eq("").any():
        raise ValueError("Scanner CSV needs exactly one nonempty row per study ID")
    # Extra source columns are not copied into the prepared private input.
    allowed = ["study_id", "scanner"] + (["scanner_indicator"] if "scanner_indicator" in metadata else [])
    return metadata[allowed]


def assemble(payload, metadata):
    """Validate original inputs; amend inner folds once, never outer partitions."""
    old = paper_settings()
    ids, x, y, _, original_plans = _validated_input(payload, ["C", "CP", "CPT"], old)
    if not metadata.study_id.is_unique or set(ids) != set(metadata.study_id):
        raise ValueError("Scanner CSV must match the selected cohort exactly by study_id")
    metadata = metadata.set_index("study_id").loc[ids]
    if metadata["scanner"].isna().any() or set(metadata["scanner"]) != set(LEVELS):
        raise ValueError("Scanner values must be exactly Biograph HiRes or Biograph Vision 600")
    scanner_indicator = (metadata["scanner"] == LEVELS[1]).astype(int).to_numpy()
    if "scanner_indicator" in metadata:
        supplied = pd.to_numeric(metadata["scanner_indicator"], errors="raise").to_numpy()
        if not np.array_equal(supplied, scanner_indicator):
            raise ValueError("Scanner text and scanner_indicator disagree")
    x[SCANNER] = metadata["scanner"].to_numpy()
    plans = deepcopy(original_plans)
    joint = 2 * scanner_indicator + y.to_numpy(dtype=int)
    for plan, original_plan in zip(plans, original_plans):
        train = np.asarray(plan["train"], dtype=int)
        if min(np.bincount(joint[train], minlength=4)) < 3:
            raise ValueError("Every outer training set needs at least three people per scanner/outcome cell")
        if type(plan.get("inner_seed")) is not int:
            raise ValueError("Every original split must retain its integer inner_seed")
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=plan["inner_seed"])
        plan["inner"] = [{"train": train[a].tolist(), "validation": train[b].tolist()}
                         for a, b in splitter.split(np.zeros(len(train)), joint[train])]
        if any(plan[k] != original_plan[k] for k in ("repeat", "fold", "train", "test", "inner_seed")):
            raise ValueError("Outer partitions or original inner seed changed")
    core.validate_split_plan(plans, y, old.repeats, old.inner_folds)
    settings = replace(old, clinical=old.clinical + (SCANNER,),
                       categories={**old.categories, SCANNER: LEVELS},
                       profile="authorized_four_models_unpenalized_PET_scanner_joint_inner_20261109")
    columns = {"C": settings.clinical, "CT": settings.clinical + settings.t1,
               "CP": settings.clinical + settings.pet,
               "CPT": settings.clinical + settings.pet + settings.t1}
    for fields in columns.values():
        core.validate_inputs(x, y, settings, fields)
    cells = []
    for plan in plans:
        for stage, ii, rows in [("outer_train", None, plan["train"])] + [
                ("inner_train", ii, pair["train"]) for ii, pair in enumerate(plan["inner"])]:
            counts = {level: {str(outcome): sum(x.iloc[r][SCANNER] == level and y.iloc[r] == outcome for r in rows)
                             for outcome in (0, 1)} for level in LEVELS}
            if any(counts[level][str(outcome)] == 0 for level in LEVELS for outcome in (0, 1)):
                raise ValueError("A training scanner group has only one outcome class; no finite nuisance optimum")
            cells.append({"repeat": plan["repeat"], "fold": plan["fold"], "inner_fold": ii,
                          "stage": stage, "scanner_outcome_cells": counts})
    return ids, x, y, settings, columns, plans, cells


def prepare(input_path, scanner_path, output):
    payload = read(input_path)
    metadata = read_scanner_metadata(scanner_path)
    ids, x, y, settings, columns, plans, cells = assemble(payload, metadata)
    out = new_private_output(output)
    try:
        write(out / "inputs/model_input_PRIVATE.json", payload, boundary=out)
        write(out / "inputs/scanner_metadata_PRIVATE.json", metadata.to_dict(orient="records"), boundary=out)
        write(out / "frozen_splits_joint_inner_PRIVATE.json", plans, boundary=out)
        protocol = {
            "status": "PREPARED_NOT_FITTED", "time_utc": stamp(), "settings": asdict(settings),
            "n": len(ids), "responders": int(y.sum()), "nonresponders": int(len(y) - y.sum()),
            "model_columns": columns, "scanner_reference": LEVELS[0], "scanner_comparison": LEVELS[1],
            "scanner_penalty_factor": 0, "other_slope_penalty_factor": 1, "intercept_penalty_factor": 0,
            "objective": "mean(binary logistic loss) + l1_ratio/(C*n)*sum(abs(beta_pen)) + (1-l1_ratio)/(2*C*n)*sum(beta_pen**2); intercept and PET scanner unpenalized; no penalty-factor renormalization",
            "same_saved_outer_plans": True, "same_saved_inner_plans": False, "outer_splits": len(plans),
            "original_split_digest": plan_digest(payload["plans"]), "split_digest": plan_digest(plans),
            "inner_amendment": "3-fold StratifiedKFold; shuffle=True; original inner_seed; strata=2*scanner_indicator+outcome; one construction without performance-based selection",
            "primary_comparison": ["CT", "CPT"], "other_descriptive_comparisons": list(COMPARISONS[1:]),
            "tuning": "mean inner 3-fold Brier; full 6x5 grid; tie tolerance 1e-8; smaller C then smaller l1_ratio",
            "evaluation": "AUC/Brier/calibration/decision curves from each person's average of 20 held-out predictions; plus repeat-specific metrics",
            "bootstrap": {"draws": 2000, "seed": 20260910, "unit": "participant", "paired": True,
                          "fixed_predictions": True, "refit": False, "interval": "2.5th and 97.5th percentiles; conditional descriptive"},
            "numerical_acceptance": {"KKT_inf_max": 1e-8, "duality_gap_max": 1e-10,
                                     "selected_fit_exact_repeat": True,
                                     "independent_prediction_max_difference": 1e-5,
                                     "independent_objective_max_difference": 1e-10},
            "incomplete_policy": "Stop affected model; preserve diagnostics and completed folds; no fallback or implicit resume",
            "environment": environment(),
        }
        write(out / "protocol.json", protocol, boundary=out)
        write(out / "training_cell_audit_PRIVATE.json", {"status": "PASS", "partitions": cells}, boundary=out)
        frozen_names = ["inputs/model_input_PRIVATE.json", "inputs/scanner_metadata_PRIVATE.json",
                        "frozen_splits_joint_inner_PRIVATE.json", "protocol.json", "training_cell_audit_PRIVATE.json"]
        manifest = {"work_files": [{"path": name, "sha256": digest(out / name)} for name in frozen_names],
                    "code_files": code_fingerprints(),
                    "original_input_sha256": digest(input_path), "original_scanner_csv_sha256": digest(scanner_path)}
        write(out / "freeze_manifest.json", manifest, boundary=out)
        write(out / "preflight.json", {"status": "PASS", "model_fit_performed": False,
              "n": len(y), "responders": int(y.sum()), "models": MODELS,
              "same_saved_outer_plans": True, "joint_inner_plans_frozen": True,
              "all_training_scanner_outcome_cells_nonempty": True,
              "participant_records_private": True}, boundary=out)
    except BaseException as exc:
        write(out / "preparation_failure.json", {"status": "FAILED_PRESERVED", "error": str(exc),
              "error_type": type(exc).__name__}, boundary=out)
        raise
    return out


def verify_prepared(workdir):
    root = Path(workdir).resolve()
    if read(root / "preflight.json").get("status") != "PASS":
        raise ValueError("A completed preparation is required")
    manifest = read(root / "freeze_manifest.json")
    for row in manifest["work_files"]:
        path = (root / row["path"]).resolve()
        if root not in path.parents or digest(path) != row["sha256"]:
            raise ValueError("Frozen prepared input or protocol changed")
    if manifest["code_files"] != code_fingerprints():
        raise ValueError("Analysis code changed after preparation")
    payload = read(root / "inputs/model_input_PRIVATE.json")
    metadata = pd.DataFrame(read(root / "inputs/scanner_metadata_PRIVATE.json"))
    ids, x, y, settings, columns, plans, _ = assemble(payload, metadata)
    if plans != read(root / "frozen_splits_joint_inner_PRIVATE.json"):
        raise ValueError("Prepared joint inner splits differ from their deterministic construction")
    if read(root / "protocol.json")["split_digest"] != plan_digest(plans):
        raise ValueError("Prepared split fingerprint changed")
    return root, ids, x, y, settings, columns, plans


def mask_for(transformer):
    kept = [f for f, keep in zip(transformer.columns, transformer.keep) if keep]
    assert SCANNER in kept
    return np.asarray([0. if f == SCANNER else 1. for f in kept])


def fit_fold(name, x, y, fields, settings, plan, deadline):
    import scanner_solver as solver
    prepared, audit, fits = [], [], []
    for ii, pair in enumerate(plan["inner"]):
        tr, va = pair["train"], pair["validation"]
        prep = core.FoldPreprocessor(fields, settings.categories).fit(x.iloc[tr])
        mask = mask_for(prep)
        audit.append({"model": name, "repeat": plan["repeat"], "fold": plan["fold"], "stage": "inner",
                      "inner_fold": ii, "fit_rows": tr, "penalty_mask": mask, **prep.state()})
        prepared.append((prep.transform(x.iloc[tr]), np.asarray(y.iloc[tr], int), prep.transform(x.iloc[va]), y.iloc[va], mask))
    candidates = []
    for c in settings.c_grid:
        for l1 in settings.l1_grid:
            scores = []
            for ii, (tx, ty, vx, vy, mask) in enumerate(prepared):
                deadline()
                fitted = solver.fit(tx, ty, c, l1, mask)
                scores.append(float(brier_score_loss(vy, fitted.predict_proba(vx)[:, 1])))
                fits.append({"stage": "inner", "inner_fold": ii, "iterations": int(fitted.n_iter_[0]), **fitted.numerical_})
            candidates.append({"C": float(c), "l1_ratio": float(l1), "mean_brier": float(np.mean(scores))})
    chosen = core.select_candidate(candidates)
    tr, te = plan["train"], plan["test"]
    prep = core.FoldPreprocessor(fields, settings.categories).fit(x.iloc[tr])
    mask = mask_for(prep)
    audit.append({"model": name, "repeat": plan["repeat"], "fold": plan["fold"], "stage": "outer_refit",
                  "fit_rows": tr, "penalty_mask": mask, **prep.state()})
    tx, ty, vx = prep.transform(x.iloc[tr]), np.asarray(y.iloc[tr], int), prep.transform(x.iloc[te])
    deadline()
    fitted = solver.fit(tx, ty, chosen["C"], chosen["l1_ratio"], mask)
    prediction = fitted.predict_proba(vx)[:, 1]
    fits.append({"stage": "outer_refit", "iterations": int(fitted.n_iter_[0]), **fitted.numerical_})
    repeated = solver.fit(tx, ty, chosen["C"], chosen["l1_ratio"], mask)
    assert np.array_equal(fitted.coef_, repeated.coef_) and np.array_equal(fitted.intercept_, repeated.intercept_)
    assert np.array_equal(prediction, repeated.predict_proba(vx)[:, 1])
    deadline()
    w, independent = solver.independent_slsqp(tx, ty, chosen["C"], chosen["l1_ratio"], mask)
    td = float(np.max(np.abs(expit(w[0] + tx @ w[1:]) - fitted.predict_proba(tx)[:, 1])))
    vd = float(np.max(np.abs(expit(w[0] + vx @ w[1:]) - prediction)))
    od = abs(independent["objective_mean"] - fitted.numerical_["objective_mean"])
    if not independent["success"] or max(td, vd) > 1e-5 or od > 1e-10:
        raise RuntimeError(f"Independent selected-fit verification failed: training={td}, test={vd}, objective={od}, diagnostics={independent}")
    assert len(fits) == 91 and np.isfinite(prediction).all()
    return {"model": name, "repeat": plan["repeat"], "fold": plan["fold"], "test": te,
            "prediction": prediction, "audit": audit, "selected": chosen, "candidates": candidates,
            "numerical_fits": fits, "exact_selected_fit_repetition": True,
            "independent_selected": {**independent, "training_probability_max_difference": td,
                                     "held_out_probability_max_difference": vd, "objective_max_difference": od}}


def fit_one(workdir, name, *, execute=False, timeout=10800):
    if name not in MODELS:
        raise ValueError("Choose one of C, CT, CP or CPT")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("A positive finite timeout is required")
    root, _, x, y, settings, columns, plans = verify_prepared(workdir)
    if not execute:
        return {"status": "PLAN_ONLY_NOT_FITTED", "model": name, "outer_splits": len(plans),
                "inner_candidate_fits_per_outer_split": 90, "selected_refit_per_outer_split": 1,
                "selected_refit_exact_repeat_and_independent_check": True,
                "scanner_penalty_factor": 0, "output": str(root / "models" / name)}
    target = root / "models" / name
    if target.exists():
        raise FileExistsError("Model output already exists; no implicit restart")
    target.mkdir(parents=True)
    started = time.monotonic()
    status = {"status": "RUNNING", "model": name, "pid": os.getpid(), "started_utc": stamp(),
              "completed_outer_splits": 0, "total_outer_splits": len(plans), "timeout_seconds": timeout,
              "split_digest": plan_digest(plans), "participant_records_private": True}
    write(target / "status.json", status, boundary=root)
    def deadline():
        if time.monotonic() - started > timeout:
            raise TimeoutError("Fixed model runtime limit reached; completed records retained")
    try:
        with threadpool_limits(limits=1):
            for i, plan in enumerate(plans):
                result = fit_fold(name, x, y, columns[name], settings, plan, deadline)
                write(target / f"fold_{i:03d}_PRIVATE.json", result, boundary=root)
                status.update(completed_outer_splits=i + 1, elapsed_seconds=time.monotonic() - started, updated_utc=stamp())
                write(target / "status.json", status, boundary=root)
                if (i + 1) % 5 == 0:
                    print(f"{name}: {i+1}/{len(plans)} outer folds complete", flush=True)
        verify_prepared(root)
        status.update(status="PASS", finished_utc=stamp(),
                      fold_sha256=[digest(target / f"fold_{i:03d}_PRIVATE.json") for i in range(len(plans))])
        write(target / "status.json", status, boundary=root)
        return status
    except BaseException as exc:
        status.update(status="FAILED_PRESERVED", error_type=type(exc).__name__, error=str(exc),
                      failed_outer_split=status["completed_outer_splits"], traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic() - started)
        if hasattr(exc, "diagnostics"):
            status["numerical_diagnostics"] = exc.diagnostics
        write(target / "status.json", status, boundary=root)
        raise


def bootstrap_all(result, y):
    # Same draws and original comparisons, with CT comparisons added using those
    # very same participant draws. No model fitting or tuning in the bootstrap.
    probabilities = {m: v.mean(axis=0) for m, v in result["predictions"].items()}
    yy = np.asarray(y)
    rng = np.random.default_rng(20260910)
    records, invalid = [], 0
    for _ in range(2000):
        rows = rng.integers(0, len(yy), size=len(yy))
        if len(np.unique(yy[rows])) < 2:
            invalid += 1
            continue
        scores = {m: (float(roc_auc_score(yy[rows], p[rows])), float(brier_score_loss(yy[rows], p[rows]))) for m, p in probabilities.items()}
        record = {f"{m}_{metric}": val[j] for m, val in scores.items() for j, metric in enumerate(("auc", "brier"))}
        for base, added in COMPARISONS:
            record[f"{added}_minus_{base}_auc"] = scores[added][0] - scores[base][0]
            record[f"{base}_minus_{added}_brier"] = scores[base][1] - scores[added][1]
        records.append(record)
    intervals = {key: np.quantile([r[key] for r in records], [.025, .975]).tolist() for key in records[0]}
    original = core.paired_bootstrap(result, y, draws=2000, seed=20260910)
    assert all(np.array_equal(v, intervals[k]) for k, v in original["intervals"].items())
    return {**{k: v for k, v in original.items() if k != "intervals"}, "intervals": intervals,
            "original_C_CP_CPT_bootstrap_exactly_matched": True, "primary_comparison": ["CT", "CPT"]}


def comparisons_for(performance, intervals):
    comparisons = []
    for base, added in COMPARISONS:
        comparisons.append({"base": base, "added": added, "primary": (base, added) == ("CT", "CPT"),
                            "delta_auc_added_minus_base": performance[added]["auc"] - performance[base]["auc"],
                            "delta_auc_conditional_95_interval": intervals["intervals"][f"{added}_minus_{base}_auc"],
                            "brier_improvement_base_minus_added": performance[base]["brier"] - performance[added]["brier"],
                            "brier_improvement_conditional_95_interval": intervals["intervals"][f"{base}_minus_{added}_brier"],
                            "repeat_delta_auc": (np.array(performance[added]["repeat_auc"]) - performance[base]["repeat_auc"]).tolist()})
    return comparisons


def collect_completed(root, ids, x, y, settings, columns, plans):
    predictions = {m: np.full((settings.repeats, len(y)), np.nan) for m in MODELS}
    seen = {m: np.zeros((settings.repeats, len(y)), int) for m in MODELS}
    audit, checks = [], []
    for name in MODELS:
        target = root / "models" / name
        status = read(target / "status.json")
        if (status.get("status") != "PASS" or status.get("model") != name or
                status.get("completed_outer_splits") != len(plans) or
                status.get("split_digest") != plan_digest(plans)):
            raise ValueError("All four models require completed, matching fit records")
        if len(status.get("fold_sha256", [])) != len(plans):
            raise ValueError("Completed model is missing frozen fold hashes")
        for i, plan in enumerate(plans):
            path = target / f"fold_{i:03d}_PRIVATE.json"
            if digest(path) != status["fold_sha256"][i]:
                raise ValueError("A completed fold record changed")
            row = read(path)
            if (row["model"], row["repeat"], row["fold"], row["test"]) != (name, plan["repeat"], plan["fold"], plan["test"]):
                raise ValueError("Fold record does not match its prepared split")
            prediction = np.asarray(row["prediction"], float)
            if (prediction.shape != (len(plan["test"]),) or not np.isfinite(prediction).all() or
                    np.any((prediction < 0) | (prediction > 1))):
                raise ValueError("Invalid held-out probability array")
            if len(row["numerical_fits"]) != 91 or row["exact_selected_fit_repetition"] is not True:
                raise ValueError("Incomplete selected-fit numerical audit")
            maximum_kkt = max(f["kkt_inf"] for f in row["numerical_fits"])
            maximum_gap = max(f["duality_gap_mean"] for f in row["numerical_fits"])
            independent = row["independent_selected"]
            if (maximum_kkt > 1e-8 or maximum_gap > 1e-10 or not independent["success"] or
                    max(independent["training_probability_max_difference"], independent["held_out_probability_max_difference"]) > 1e-5 or
                    independent["objective_max_difference"] > 1e-10):
                raise ValueError("A saved fit fails the fixed numerical acceptance thresholds")
            for item in row["audit"]:
                kept = [field for field in columns[name] if field not in item["dropped"]]
                expected_mask = [0. if field == SCANNER else 1. for field in kept]
                if item["penalty_mask"] != expected_mask:
                    raise ValueError("A saved fit does not retain exactly the unpenalized scanner contrast")
            predictions[name][plan["repeat"], plan["test"]] = prediction
            seen[name][plan["repeat"], plan["test"]] += 1
            audit += row["audit"]
            checks.append({"model": name, "repeat": row["repeat"], "fold": row["fold"],
                           "selected": row["selected"], "fits": len(row["numerical_fits"]),
                           "max_kkt": maximum_kkt, "max_duality_gap": maximum_gap,
                           "independent_selected": independent})
    if any(not np.all(seen[name] == 1) or not np.isfinite(predictions[name]).all() for name in MODELS):
        raise ValueError("Held-out prediction coverage must be exactly once per person per repeat")
    sd = plan_digest(plans)
    result = {"status": "PASS", "participant_ids": ids, "predictions": predictions,
              "plans": plans, "audit": audit, "model_columns": columns, "split_digest": sd,
              "model_split_digests": {name: sd for name in MODELS}}
    core.assert_training_isolation(result)
    return result, checks


def summarize(workdir, output):
    root, ids, x, y, settings, columns, plans = verify_prepared(workdir)
    result, checks = collect_completed(root, ids, x, y, settings, columns, plans)
    out = new_private_output(output)
    try:
        with threadpool_limits(limits=1):
            performance = core.summarize_predictions(result, y, settings)
            intervals = bootstrap_all(result, y)
            comparisons = comparisons_for(performance, intervals)
        write(out / "predictions_PRIVATE.json", {key: value for key, value in result.items()
              if key not in ("audit", "plans")}, boundary=out)
        write(out / "performance.json", performance, boundary=out)
        write(out / "conditional_intervals.json", intervals, boundary=out)
        write(out / "paired_comparisons.json", comparisons, boundary=out)
        write(out / "numerical_QA_PRIVATE.json", {"status": "PASS", "outer_model_tasks": len(checks),
              "fits": sum(row["fits"] for row in checks), "selected_fit_repeats": len(checks),
              "independent_selected_checks": len(checks), "training_isolation": "PASS",
              "same_original_outer_splits": True, "same_joint_inner_splits_for_all_models": True,
              "complete_held_out_coverage": True, "scanner_zero_penalty_all_models": True,
              "checks": checks}, boundary=out)
        verify_prepared(root)
        write(out / "summary_status.json", {"status": "PASS", "time_utc": stamp(),
              "model_fit_performed": False, "input_workdir": str(root),
              "conditional_fixed_prediction_intervals": True,
              "clinical_or_imaging_QC_performed": False}, boundary=out)
        return performance
    except BaseException as exc:
        write(out / "summary_status.json", {"status": "FAILED_PRESERVED", "error_type": type(exc).__name__,
              "error": str(exc), "traceback": traceback.format_exc()}, boundary=out)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    command = actions.add_parser("prepare", help="Freeze explicit inputs and one joint-stratified inner plan; no fitting")
    command.add_argument("--input", required=True, help="STEP05 model_input.json with original complete frozen plans")
    command.add_argument("--scanner-metadata", required=True, help="Private CSV with study_id, scanner; optional scanner_indicator")
    command.add_argument("--output", required=True, help="New private working directory outside this code package")
    command = actions.add_parser("fit", help="Fit only one named model; execution is opt-in")
    command.add_argument("--workdir", required=True)
    command.add_argument("--model", required=True, choices=MODELS)
    command.add_argument("--execute", action="store_true")
    command.add_argument("--timeout-seconds", type=float, default=10800)
    command = actions.add_parser("summarize", help="Summarize all four completed fits; does not refit")
    command.add_argument("--workdir", required=True)
    command.add_argument("--output", required=True, help="New private summary directory")
    args = parser.parse_args()
    if args.action == "prepare":
        result = {"prepared_workdir": prepare(args.input, args.scanner_metadata, args.output), "model_fit_performed": False}
    elif args.action == "fit":
        result = fit_one(args.workdir, args.model, execute=args.execute, timeout=args.timeout_seconds)
    else:
        result = summarize(args.workdir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=serial, allow_nan=False))


if __name__ == "__main__":
    main()
