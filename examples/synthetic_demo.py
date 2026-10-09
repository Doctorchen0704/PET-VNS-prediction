"""Small software demonstration, not a recreation of the study cohort."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from pet_vns_prediction import association_family, paper_settings, run_nested_models
from pet_vns_prediction import core


def run_demo():
    # Deliberately reduced runtime. The library's paper_settings() defaults are
    # unchanged; this example uses neither real patients nor the full grid.
    settings = replace(
        paper_settings(), repeats=1, c_grid=(0.1,), l1_grid=(0.0,),
        profile="synthetic_smoke_not_manuscript_reproduction",
    )
    x, y = core.synthetic_fixture(settings, n=40, seed=1701)
    result = run_nested_models(x, y, settings, timeout_seconds=300)
    # Scanner adjustment belongs to association analyses, not the five-variable
    # clinical prediction model. This is an artificial, balanced scanner code.
    scanner = np.random.default_rng(1702).permutation(np.tile([0.0, 1.0], 20))
    covariates = np.column_stack([
        x.age_at_implant_years,
        (x.sex == "male").astype(float),
        x.epilepsy_duration_reported_years,
        x.log1p_baseline_monthly_frequency,
        (x.mri_lesional_status == "lesional").astype(float),
        scanner,
    ])
    associations = association_family(
        x[list(settings.pet)].to_numpy(), y.to_numpy(),
        {"primary": covariates}, expected_tests=6,
        region_names=settings.pet,
    )
    return {
        "synthetic_only": True,
        "interpretation": "Software demonstration; not manuscript estimates or validation evidence",
        "profile": settings.profile,
        "performance": core.summarize_predictions(result, y, settings),
        "regional_associations": associations,
        "outer_splits": len(result["plans"]),
        "observed_data_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are never replaced")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new file.")
    text = json.dumps(run_demo(), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
