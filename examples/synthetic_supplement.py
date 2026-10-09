"""Artificial-only supplemental smoke run: reduced settings, counts, no files.

This demonstrates software interfaces, not manuscript results or validation
evidence. No actual participant, image, clinical file, or private source is read.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

# The example itself and imports should not create output or bytecode files.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pet_vns_prediction import core, sensitivity, validation
from pet_vns_prediction.modeling import paper_settings


def run_demo():
    settings = replace(
        paper_settings(), repeats=1, c_grid=(0.01,), l1_grid=(0.0,),
        profile="artificial_supplement_smoke_not_paper_defaults",
    )
    x, y = core.synthetic_fixture(settings, n=48, seed=1703)
    rng = np.random.default_rng(1704)
    # Both artificial devices contain both artificial outcome classes.
    scanner = pd.Series("synthetic_device_a", index=x.index)
    for label in (0, 1):
        scanner.iloc[np.flatnonzero(y.to_numpy() == label)[::2]] = "synthetic_device_b"
    scanner_indicator = (scanner == "synthetic_device_b").to_numpy(dtype=float)
    clinical5 = np.c_[
        x[settings.clinical[0]], x.sex == "male", x[settings.clinical[2]],
        x[settings.clinical[3]], x.mri_lesional_status == "lesional",
    ].astype(float)
    pet, t1 = x[list(settings.pet)].to_numpy(), x[list(settings.t1)].to_numpy()
    variants = {
        name: pet.copy() if name == "original" else pet + rng.normal(0, .005, pet.shape)
        for name in sensitivity.PROCESSING_VARIANTS
    }
    types, etiologies = rng.integers(0, 3, len(x)), rng.integers(0, 3, len(x))
    twelve = sensitivity.clinical_morphometric_family(pet, y, clinical5, scanner_indicator, t1)
    followup = sensitivity.followup_association_families(
        variants, y, clinical5, scanner_indicator, t1,
        np.c_[types == 1, types == 2], np.c_[etiologies == 1, etiologies == 2],
    )
    plans = core.make_split_plan(y, settings)
    processing_pet = {
        name: pd.DataFrame(variants[name], index=x.index, columns=settings.pet)
        for name in sensitivity.PREDICTION_VARIANTS
    }
    processing = sensitivity.run_processing_predictions(
        x, processing_pet, y, settings, plans=plans, timeout_seconds_per_model=60,
    )
    cross_scanner = validation.run_cross_scanner(
        x, y, scanner, settings, interval_draws=12, timeout_seconds=60,
    )
    ablations = validation.run_ablations(
        x, y, plans, settings, interval_draws=12, timeout_seconds=60,
    )
    bootstrap = validation.run_development_bootstrap(
        x, y, settings, draws=3, timeout_seconds=60,
    )
    return {
        "synthetic_only": True,
        "not_paper_defaults": True,
        "profile": settings.profile,
        "interpretation": "Counts demonstrate software execution only; no scientific performance claims.",
        "files_written": 0,
        "counts": {
            "artificial_rows": len(x),
            "repeats": settings.repeats,
            "tuning_candidates": len(settings.c_grid)*len(settings.l1_grid),
            "outer_splits": len(plans),
            "clinical_morphometric_tests": len(twelve),
            "extended_clinical_tests": followup["family_test_counts"]["clinical"],
            "processing_association_tests": followup["family_test_counts"]["processing"],
            "joint_followup_tests": len(followup["rows"]),
            "influence_diagnostics": len(followup["diagnostics"]),
            "processing_prediction_models": len(processing["predictions"]),
            "processing_comparison_definitions": len(processing["comparisons"]),
            "cross_scanner_directions": len(cross_scanner["directions"]),
            "ablation_models_including_full_CP": len(ablations["predictions"]),
            "conditional_bootstrap_draws_requested": 12,
            "development_bootstrap_draws_requested": bootstrap["summary"]["draws_requested"],
            "development_bootstrap_draws_valid": bootstrap["summary"]["draws_valid"],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, allow_nan=False))
