# Release notes

## 1.1.0 — 2026-11-09

- Added an independently invoked PET-scanner-adjusted C/CT/CP/CPT analysis under
  `STEP07_prediction_models/scanner_adjusted_four_models`.
- Preserved the completed analysis's standalone masked elastic-net solver, training-only
  preprocessing, 30-candidate tuning grid, numerical checks and paired result summaries.
- Added explicit input preparation from saved outer partitions and keyed scanner metadata.
  Original outer partitions and inner seeds are retained; inner folds are reconstructed
  once using joint scanner-by-outcome stratification. All four models share those folds.
- Scanner and intercept coefficients are unpenalized. No feature selection, new outer
  partitions, imaging reprocessing or outcome editing is part of this release.
- All earlier scripts remain available, including the original three-model workflow.
  New numerical code is an additional analysis, not an update of historical predictions.
- Patient data, saved research folds, fitted coefficients and person-level predictions
  are not published. Validation scope is recorded separately from clinical interpretation.

## 1.0.0 — 2026-11-03

Introduced manually invoked STEP01–10 scripts using the verified 0.3.0 numerical library.
The fixed earlier snapshot is commit `ac25aaf0fd36b7d4d258e26154ee395ba0bc3b36`.
