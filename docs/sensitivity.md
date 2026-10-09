# Supplemental associations and processing predictions

`pet_vns_prediction.sensitivity` is an in-memory numerical adaptation, not a
loader for the private study. It does not read clinical or image files, perform
registration/segmentation/PVC, choose a cohort, remove observations, or publish
outputs. Returned fit records and influence diagnostics can contain sensitive
row-level information: keep actual-data results outside the public repository.

These are exploratory sensitivity analyses, not independently validated models
or estimates of a VNS-specific treatment effect. The manuscript's primary PVC
setting is 5.4 mm; 4 and 6 mm are retained processing checks. Software alone does
not establish the applicability of an effective-resolution value.

## Fixed association families

| Function | Entire correction family |
| --- | --- |
| `clinical_morphometric_family` | 12 tests: six clinical-only adjustments and six clinical + scanner + corresponding T1 adjustments |
| `extended_clinical_family` | 18 tests: six regions in each of epilepsy-type, grouped-etiology, and both-block adjustments, added to clinical + scanner |
| `processing_association_family` | 96 tests: eight processing variants × six regions × two adjustment designs |
| `followup_association_families` | Separate 18- and 96-test BH corrections, plus the source's additional all-114 correction |

The primary six-test clinical + scanner family is separate; use the public
`associations.association_family` with that explicit design. It must not be
substituted by a selected two-region family. The all-114 column does not replace
the two family-specific corrections and is not a correction for all historical
exploration.

All six PET columns and their corresponding T1 columns must have identical,
caller-verified participant and region order. Association inputs must be finite
and complete; these functions neither impute nor silently remove rows.
The numeric clinical block is ordered as:

1. Age at implantation.
2. Male indicator, with female as reference.
3. Reported epilepsy duration.
4. `log1p` baseline monthly seizure frequency (already transformed by the caller).
5. Lesional MRI indicator, with nonlesional as reference.

Age, duration, and log-frequency are standardized with sample SD (`ddof=1`)
across the complete association sample. Scanner is a caller-supplied 0/1
indicator with a documented reference. Corresponding T1 is standardized with
sample SD, one T1 feature per region. This full-sample association scaling is
distinct from training-only prediction preprocessing.

The two epilepsy-type dummies encode focal and combined-generalized-and-focal,
with generalized as reference. The two etiology dummies encode structural and
other-known, with unknown as reference. The source's other-known category groups
infectious, immune, and metabolic etiology. Grouping must be declared before
fitting, never inferred from results; it does not eliminate detailed etiologic
residual confounding. The functions accept these fixed dummies and do not edit
the caller's clinical classifications.

The complete processing-association order is `original`, `wholeBrain`,
`cerebellarCortex`, `same_seg_noPVC`, `pvc54`, `pvc4`, `pvc6`, `petprep`.
All eight are required, including PVC 4 and 6 even when absent from a plot.
Each variant has clinical + scanner (`base`) and clinical + scanner + matching
T1 (`base_plus_corresponding_T1`) adjustments. Every effect uses that variant's
full-sample pooled within-group SD, not the original variant's SD. HC3 QR/SVD
checks, residual-df t inference, and complete-family BH are retained.

`leave_one_person_influence` preserves the full-sample design scaling and
pooled SD while deleting one row at a time from OLS. It returns all coefficients,
their range, sign-change count, rank diagnostics, leverage, Cook's distance,
VIF, fitted values and residuals. Rank-deficient leave-one-out designs retain the
source's minimum-norm `lstsq` calculation and are flagged by their returned rank;
they are not silently treated as full-rank. No case is removed and no exclusion
recommendation is made. Full-design HC3 still requires full rank and leverage
below the fixed threshold. The combined follow-up function returns all 114
diagnostic records by default; `influence=False` omits diagnostics only, never
tests or p-value correction.

## Processing prediction engine

`run_processing_predictions(original, processing_pet, response, settings=None)`
requires seven alternative PET tables, in addition to the original predictor
table. Each alternative contains exactly the six canonical `settings.pet`
columns with identical participant index and order. Only PET values are replaced;
clinical and T1 columns remain those in `original`. The variants are
`same_seg_noPVC`, `pvc54`, `pvc4`, `pvc6`, `wholeBrain`, `cerebellarCortex`, and
`petprep`. For each variant, both CP and CPT are fitted: 14 models in total.

The engine is an adaptation of the source's `common.py` plain-column
`Transformer`, Brier branch of `select`, and `run_outer`. It is deliberately not
an alias for the primary C/CP/CPT engine. It preserves:

- Fold-only median/mode imputation and continuous scaling, constant-column
  removal, category references, and training-row audits.
- The complete C × l1-ratio grid, both inner-fold Brier and AUC records, and
  minimum mean Brier selection; ties within `1e-8` favor smaller C then smaller
  l1 ratio. AUC is recorded, not optimized for this branch.
- Deterministic certified elastic-net fits, exact repetition of the selected
  fit, and independent SLSQP checks on both training and held-out probabilities
  (maximum difference `1e-5`) and objective (`1e-10`).
- Shared repeated nested partitions, exactly one held-out prediction per person
  per repeat, per-model time limits, and failure without fallback or incomplete
  summaries.

Defaults come from `paper_settings`: 20 repeats, source class-support outer-fold
rule, three inner folds, six C values and five l1 ratios. Smaller test settings
are explicitly labeled synthetic. The primitive `run_processing_outer` is also
available; it assumes the caller has validated the full shared split plan.

`processing_model_definitions()` returns every model recipe.
`processing_prediction_comparisons()` returns the 28 inherited processing
comparisons, each with seed 20260910:

- Four same-segmentation/PVC settings × CP/CPT against the corresponding
  original `CORE_CP`/`CORE_CPT` (8 comparisons).
- Three PVC settings × CP/CPT against `same_seg_noPVC` (6 comparisons).
- Seven settings' CP against `CORE_C`, and CPT against matching CP (14).

Original `CORE_C`, `CORE_CP`, and `CORE_CPT` predictions must be supplied from
their own primary engine when evaluating these comparisons; this module does
not refit or relabel them. The comparison definitions are not inferential
results. Fixed out-of-fold paired bootstrap intervals, if subsequently computed,
are conditional descriptive uncertainty, not full model-development uncertainty.

## Verification and limits

The public synthetic test file is `tests/test_sensitivity.py`; run it with:

```text
python -m unittest discover -s tests -p test_sensitivity.py -v
```

`python examples/synthetic_supplement.py` provides an additional no-file,
artificial-data smoke demonstration spanning sensitivity and validation
interfaces. It prints counts only, with one repeat, one tuning candidate,
12 conditional-bootstrap draws and three development-bootstrap draws. These are
deliberately reduced software-smoke settings, not the paper defaults.

All 11 tests passed during adaptation. Tests cover full 12/18/96/114 families,
all 114 influence diagnostics, fixed-SD leave-one-out calculations, rejection
of missing variants/rows/invalid dummy encoding, Brier tie behavior, held-out
label invariance, rejected abandoned transformer branches, full 14-model runs
on shared splits, and no fallback into the primary model engine.

An additional maintainer-only parity check extracted function/class definitions
with Python AST from the read-only sources below. It did not import their
top-level scripts. File reads/writes and cohort loading were replaced with
in-memory artificial fixtures. The check found:

- 12-test coefficient/p/q/standardized-effect maximum absolute difference:
  `3.3306690738754696e-16`.
- 114-test coefficient/HC3-SE/p/q/standardized-effect maximum difference: `0.0`.
- All 114 leave-one-out ranges maximum difference: `0.0`.
- Exact equality of the complete selected processing outer-fold record,
  including preprocessing, tuning, numerical certificates and predictions.
- Exact AST equality of the source/public solver function and class definitions.

This checks synthetic numerical parity, not reproduction of actual study results
or robustness to other cohorts. The maintainer check is not part of the public
test dependency: no private source directory or study data is required to run
the public tests.

Not ported here: raw-image preprocessing or PVC generation; private input joins,
eligibility/exclusion lists, frozen study splits, source/cache lock files,
execution launchers and saved study outputs; stratified descriptive tables,
Hedges-g bootstrap descriptives, balance/correlation reports and figure styling;
historical-q lookup against prior saved results; the processing route's saved
fold replay/evaluation file reader; abandoned PCA/AUC-selection, asymmetry,
network, voxel and other exploratory prediction engines. The explicit comparison
plan is ported, but not the private 34-model historical-results evaluator.

## Read-only source identification

Only source basenames/module-relative names and hashes are included; no private
participant keys, local paths, source data, configuration payloads, or cohort
results are redistributed.

The matching `subcortical_phenotype_followup_20260921_v1.json` was consulted for
fixed family/category definitions only. Its configuration payload and hash are
not redistributed; hashes below identify code sources only.

| Source | SHA-256 |
| --- | --- |
| `gate5_source_revision_20260920_v4.py` | `0dcb6be2d3da36579cb272b3ca49294ea9ba0816f703adb03e4ed3ab857ee091` |
| `subcortical_phenotype_followup_20260921_v1.py` | `e5c5b0cf38b671e240e31f959da6fb75e7fae2500846e864935cfaaa58e97d4e` |
| `gate5_method_exploration_20260910_v1/common.py` | `0239d40534d63412598bf86bddb1856c19c809b7d8abbcdbd6502047ff5fbe11` |
| `full_cohort_20260921_other_routes.py` | `82d9acc8b020e89b16ed4bfa3f774bf6823427732b59caabe654d4e448103401` |
| `full_cohort_20260921_other_routes_evaluate.py` | `8a4bd55209e824b6aedace9269d3ab1bb0373ed34641010e0bd568aafc20e256` |
| `gate5_six_pathway_effects_20260912_v1.py` | `5e8581b4466b2be1c45ceb62e25250bcf9519b6112921f6eed445ee96192fb3d` |
| `gate5_modeling_20260910_v1/core.py` | `679c6e539643eb2493ff3d0d3f466176904b08de634993a09ec1a4461826c051` |
| `gate5_numerical_correction_20260910_v4/solver.py` | `f310b6e8d4742ade3da2538f974deb091a095b1177068e9596f3153c8e3f2f2f` |
