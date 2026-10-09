# Release scope: initial core methods package

Version 0.1.0 is a source-adapted, independently usable methods package. It is
not the complete manuscript analysis archive, a frozen cohort release, or an
end-to-end reproduction of the paper. The historical source filenames and
source-code SHA-256 values are listed in [source_provenance.json](source_provenance.json).
Those original source files are not required to import or use this package.

## Included

- Primary PET reference normalization and six-region feature calculations from
  supplied arrays or already aligned NIfTI files; six structural features from
  supplied FreeSurfer statistics. See [preprocessing.md](preprocessing.md) for
  exact definitions, assumptions, and the preprocessing/QC boundary.
- The study's deterministic elastic-net solver and independent selected-fit
  numerical checks, not a substitute generic logistic-regression estimator.
- Training-fold-only imputation, encoding, standardization and constant-feature
  handling; shared repeated nested-CV splits; inner Brier-score tuning and the
  original tie-breaking rule; C/CP/CPT fitting and held-out predictions.
- Parameter-only defaults for 20 outer repeats, three inner folds, the
  30-candidate grid, and clinical/PET/T1 predictor definitions. Outer-fold count
  follows the retained class-support rule. Explicit reduced-feature mappings
  can use the same model engine but do not reproduce the whole ablation report.
- Prediction-summary primitives, calibration, decision curves, and paired
  bootstrap summaries of existing held-out predictions. This bootstrap is not
  the separate development-bootstrap refitting procedure.
- Six-region HC3 association estimates, standardized effects and explicit
  complete-family BH adjustment; generic six-test primary and 18-test extended
  adjustment interfaces with caller-supplied covariates.
- Synthetic-only examples and tests. Inputs must follow the documented schema;
  model row indices must be unique, aligned, nonempty strings, never silently coerced.

## Not included in this release

- Raw or derived participant data, real clinical/outcome tables, private cohort
  rules or record links, study row ordering/splits, individual predictions, or
  pretrained clinical prediction weights.
- DICOM conversion, FreeSurfer execution, registration/resampling workflows,
  manual image/eligibility review, atlas images, external software licenses,
  or an automated image-to-paper pipeline.
- PVC or PetPrep execution and the dedicated original processing-sensitivity
  runner. Supplying alternative feature columns to the generic model interface
  is not verification of the complete processing-sensitivity analysis.
- Explicit corresponding-T1 12-test and processing 96-test family orchestration,
  complete ablation comparison reports and influence analyses, the separate
  development-bootstrap refitting analysis, or cross-scanner evaluation.
- Final manuscript figure renderers and a one-command reproduction of all main
  and supplementary tables and figures. No claim that this package covers every
  analysis reported in the manuscript should be made.

## What has been checked

Source-adaptation checks retained the complete solver and the pure core
definitions, and compared the original and public corrected workflow on one
synthetic CP outer fold with all 30 candidates and three inner folds. Predicted
probabilities and saved fit information were exactly equal for that comparison.
The retained HC3 calculations and primary PET/T1 feature calculations were also
compared with original functions on synthetic inputs. Public tests exercise the
interfaces and small synthetic model runs; no real participant analysis was
rerun to prepare this release.

Passing these checks establishes the stated source parity and synthetic tests,
not validation of a clinical device, manuscript-result reproduction, or support
for every dependency version allowed by the package metadata. Review
[reproducibility.md](reproducibility.md) before describing what was reproduced.
Outputs generated with a caller's real data remain potentially sensitive and
must not be committed to this public repository.
