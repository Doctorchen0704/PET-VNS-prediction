# Validation and retuned ablations

`pet_vns_prediction.validation` ports the executed numerical methods into an
explicit, in-memory interface. Importing it does not load files or fit models.
The library does not contain participant data, saved study predictions, study
outcomes, scanner assignments, private paths, or hashes of data files.

## Available interfaces

| Function | Executed method retained |
| --- | --- |
| `run_ablations(x, y, plans, settings)` | Full CP and three separately retuned removals: thalamus, striatopallidal, and both; identical caller-supplied nested partitions |
| `evaluate_ablations(result, y)` | Average repeated OOF probabilities per person, then paired ordinary person bootstrap of fixed predictions; pair-count AUC; 2.5/97.5 percentiles |
| `run_cross_scanner(x, y, scanner, settings)` | Two lexicographically ordered scanner holdouts; training-only C/CP preprocessing, inner tuning and refit; paired conditional held-out intervals |
| `run_development_bootstrap(x, y, settings)` | Ordinary paired-person bootstrap with full redevelopment on every draw; unique-person grouped inner folds; apparent/test optimism and individual probability stability |
| `develop(...)`, `grouped_inner_plan(...)` | Auditable lower-level training and duplicate-person fold construction; no test-outcome argument |

The original clinical/PET definitions and certified deterministic numerical
solver are reused. Three inner folds, mean validation Brier selection, the
`1e-8` tie rule, training-only imputation/scaling/constant-column removal,
exact selected-fit repetition and independent SLSQP checks are unchanged.
No feature/grid search or automatic best-model promotion is introduced.

## Shared resampling and signs

Ablations use the same supplied outer/inner plans for every model. Each
ablation is retuned, not produced by zeroing coefficients in a fitted CP model.
Both ablation differences are **CP minus reduced CP**: positive AUC difference
and negative Brier difference favor retaining the removed pathways. Fixed-OOF
bootstrap indices are generated once and reused across all four models.

Cross-scanner comparisons use **CP minus C AUC**, but **C minus CP Brier**;
positive values favor CP for both. C and CP use identical training rows,
grouped inner folds, and held-out bootstrap indices. Scanner labels are not
predictors. No harmonization or held-out calibration is performed. This is a
same-center device transport stress test, not independent external validation.

## Development bootstrap

Defaults retain 500 development draws (seed `20260921`), 2,000 conditional
interval draws (seed `20260910`), and a full-data reference inner seed of
`20260910`. Scanner inner seeds are `settings.seed + 7000 + direction`.
Development draw inner seeds are `settings.seed + 200000 + draw`.

Each development sample has the original cohort size, sampled with replacement.
The three stratified inner folds are built on sorted **distinct original
person groups** and expanded to include every sampled copy in the same fold.
Both preprocessing and validation Brier retain sample multiplicities. Fewer
than three distinct people in either class makes a draw invalid: record it and
continue to the next planned draw, without replacement draws. Any numerical fit
failure aborts the call; it is not silently removed from the summaries.

- AUC optimism = in-bag AUC minus original-cohort AUC.
- Corrected AUC = full-data apparent AUC minus mean AUC optimism.
- Brier optimism = original-cohort Brier minus in-bag Brier.
- Corrected Brier = full-data apparent Brier plus mean Brier optimism.

The same development draws are used for C/CP incremental comparisons. Original
cohort probabilities from every bootstrap-trained model are summarized with
2.5/50/97.5 percentiles, per-person widths, and mean absolute change from the
full-data reference. These mix in-bag and OOB predictions and are **stability
descriptions, not predictive confidence intervals**. The bootstrap corrects
full-data apparent development performance, not a previously reported OOF AUC.
Monte Carlo SE describes simulation precision, not clinical uncertainty.
Historical method/feature/outcome review is not replayed. OOB AUC is retained
as a smaller-effective-training-set stress test, not a replacement main result.

## Safe usage and deliberate adaptations

Use indexed predictor frames and binary outcome Series in matching order.
`scanner` must be a matching Series containing exactly two nonmissing string
labels. The bootstrap starts from unique participant indices; repeated sampled
copies are introduced internally. Direct `develop` callers must assign the same
group to every copy of an identity, and one identity per group. Returned audit,
prediction and bootstrap records may contain sensitive information; keep actual
research results out of this public repository.

The adaptation accepts arbitrary eligible cohort size instead of a hardcoded
study cohort. `run_ablations` refits full CP on supplied plans rather than opening
historical private baseline files. It does not claim those historical files have
been reproduced. Original file locks, restart/output-size policy, original OOF
file comparisons, and the filesystem-oriented independent verifier are not
public runtime interfaces. The original verifier's logic informed the synthetic
tests; no real-data verification is asserted.

No writes are performed. Serial execution and one computational thread mirror
the executed runner; do not call these wrappers concurrently in one process.
The default cooperative deadline is 1,800 seconds, checked around fits and
resampling loops. It cannot interrupt a single already-running solver call;
an external process limit is needed for a strict wall-clock kill.

Empty conditional-bootstrap support returns explicit null intervals. Empty OOB
distributions and one-draw Monte Carlo SE are also explicit nulls, instead of
the original script's undefined empty quantiles or NaN standard errors. These
are edge-case safeguards, not changes to estimable-case arithmetic. All-invalid
development draws fail without returning a completed result; the exception
retains `invalid_draws` and `draws_requested` for audit.

## Source-code provenance

These are **Python source-code hashes only**, not hashes of participant data:

| Original relative source | SHA-256 |
| --- | --- |
| `code/cp_validation_20260921_v1.py` | `9292c6cae327522f8965821e2fdaf8f959c8f200647ed58d30ca0231ff430427` |
| `code/verify_cp_validation_20260921_v1.py` | `506997a4cc76cd5c7c5b0e554a9cbcf5a7ca2b2edb2dd2d96284bc3426b3126b` |
| `code/subcortical_phenotype_followup_20260921_v1.py` | `e5c5b0cf38b671e240e31f959da6fb75e7fae2500846e864935cfaaa58e97d4e` |

The first supplies `inner_plan`, `develop`, `measures`,
`conditional_intervals`, `dist`, and the development-summary arithmetic. The
second supplies independent pair-count AUC/Brier and reconstruction checks.
The third supplies `columns`, `ablations`, `auc`, and `evaluate` semantics.
The existing public core/workflow/solver retain their own documented provenance.

## Artificial-only tests

`tests/test_validation.py` uses at most two development draws and ten
conditional draws, one outer repeat, and a small penalty grid. It checks exact
sample identities, grouped-copy isolation, train-only preprocessing, held-out
predictor invariance, retuned ablations on shared plans, paired quantile signs,
optimism arithmetic, per-person probability percentiles, failure propagation,
and deadline guards. These tests establish implementation behavior, not study
performance or clinical validity.

Optional `PET_VNS_ORIGINAL_CODE` enables source-only parity tests against an
authorized local original-code directory. They AST-select named pure function
definitions and inject synthetic arrays and the public numerical engine. They
never import the original scripts or execute their top-level private I/O.
Comparisons cover exact `develop` records, grouped folds, fixed held-out
intervals, measures, distributions, ablation definitions and pair-count AUC.
The original ablation `evaluate` function is also compared exactly using
in-memory read/save substitutes; only its hardcoded cohort dimension and draw
count are adapted to the tiny artificial fixture. Independent verifier AUC/Brier
are also checked. The monolithic original
filesystem runner and saved historical study results are deliberately not run.

Dependencies are unchanged: NumPy, pandas, SciPy, scikit-learn and threadpoolctl.
