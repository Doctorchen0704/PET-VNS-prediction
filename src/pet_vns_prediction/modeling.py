"""In-memory C/CP/CPT and reduced-feature model interface.

Inputs and returned fold records may contain private participant information.
This package performs no automatic filesystem I/O. Store real-data outputs only
in an appropriately controlled location, never in the public repository.
"""
from __future__ import annotations

from hashlib import sha256
import json
import time
from collections.abc import Mapping

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import core, workflow


def paper_settings() -> core.Settings:
    """Public parameter-only defaults, matching the paper's locked specification.

The old SAGA tolerance/iteration fields are retained for Settings compatibility;
the active solver uses its own fixed KKT and duality-gap acceptance thresholds.
No outcomes, source paths, participant exclusions, or labels are encoded here.
"""
    pet_names = (
        "vns_thalamus", "vns_cingulate", "vns_insula_frontal_operculum",
        "vns_limbic_medial_temporal", "vns_striatopallidal", "vns_sensorimotor",
    )
    t1_names = (
        "t1_thalamus_volume", "t1_cingulate_thickness", "t1_insula_operculum_thickness",
        "t1_medial_temporal_volume", "t1_striatopallidal_volume", "t1_sensorimotor_thickness",
    )
    return core.Settings(
        clinical=("age_at_implant_years", "sex", "epilepsy_duration_reported_years",
                  "log1p_baseline_monthly_frequency", "mri_lesional_status"),
        categories={"sex": ("female", "male"), "mri_lesional_status": ("nonlesional", "lesional")},
        pet=tuple("pet_supraGM__" + name for name in pet_names),
        t1=tuple("t1__" + name for name in t1_names),
        c_grid=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
        l1_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        seed=20260909, repeats=20, inner_folds=3,
        tol=1e-5, max_iter=10000, retry_iter=50000,
        clip=1e-6, thresholds=tuple(i / 10 for i in range(1, 10)),
        max_missing=0.2, profile="paper_defaults",
    )


def run_nested_models(
    x: pd.DataFrame,
    y: pd.Series,
    settings: core.Settings | None = None,
    *,
    model_columns: Mapping[str, tuple[str, ...] | list[str]] | None = None,
    plans: list[dict] | None = None,
    timeout_seconds: float = 3600,
    progress=None,
) -> dict:
    """Tune/refit all requested models on shared repeated nested-CV partitions.

Defaults are C/CP/CPT. Pass an explicit reduced-feature mapping for ablations or
an alternative PET representation using the same numerical engine. No automatic
selection among models, cohort changes, or fallback solver is performed. Any
fit failure stops this call; incomplete predictions never become a summary.
    """
    if not __debug__:
        raise RuntimeError("Required numerical assertions are disabled; do not use Python -O.")
    if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.Series):
        raise core.InputRejected("Predictors and outcome must be indexed tables/series.")
    if any(not isinstance(value, str) or not value.strip() for value in x.index) or any(
        not isinstance(value, str) or not value.strip() for value in y.index
    ):
        raise core.InputRejected("Predictor and outcome indices must contain nonempty strings only.")
    settings = paper_settings() if settings is None else settings
    columns = dict(settings.model_columns() if model_columns is None else model_columns)
    if not columns or any(not isinstance(name, str) or not fields for name, fields in columns.items()):
        raise core.InputRejected("At least one named nonempty model is required.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    for fields in columns.values():
        if len(fields) != len(set(fields)):
            raise core.InputRejected("Duplicate predictor in model definition.")
        core.validate_inputs(x, y, settings, fields)
    plans = core.make_split_plan(y, settings) if plans is None else plans
    core.validate_split_plan(plans, y, settings.repeats, settings.inner_folds)
    predictions = {name: np.full((settings.repeats, len(x)), np.nan) for name in columns}
    seen = {name: np.zeros((settings.repeats, len(x)), dtype=int) for name in columns}
    audit, tuning = [], []
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("Nested CV exceeded the caller's time limit; no automatic retry.")

    with threadpool_limits(limits=1):
        for serial, plan in enumerate(plans):
            for name, fields in columns.items():
                prediction, info = workflow.corrected_outer(
                    x, y, fields, settings, plan, name, audit, deadline,
                )
                predictions[name][plan["repeat"], plan["test"]] = prediction
                seen[name][plan["repeat"], plan["test"]] += 1
                tuning.append({"model": name, "repeat": plan["repeat"], "fold": plan["fold"], **info})
            if progress is not None:
                progress({"completed_outer_splits": serial + 1, "total_outer_splits": len(plans)})
    if not all(np.all(seen[name] == 1) and np.isfinite(values).all() for name, values in predictions.items()):
        raise core.FitFailure("Incomplete or duplicate held-out prediction coverage.")
    digest = sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()
    result = {
        "status": "PASS", "profile": settings.profile, "predictions": predictions,
        "participant_ids": list(x.index), "plans": plans, "split_digest": digest,
        "model_split_digests": {name: digest for name in columns}, "audit": audit,
        "tuning": tuning, "failures": [], "model_columns": columns,
        "elapsed_seconds": time.monotonic() - started,
        "solver": "deterministic_convex_elastic_net_with_independent_selected_fit_check",
    }
    core.assert_training_isolation(result)
    return result
