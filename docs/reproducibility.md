# Reproducibility and release boundary

## What this repository provides

This is an adaptation of the study's executed implementation into reusable modules. Private file discovery, source-workbook processing, participant-specific exclusions and release-stage administrative checks are not part of the public interfaces. A public adaptation is not represented as the untouched historical execution directory.

The numerical source lineage is important: the final study workflow used a corrected deterministic elastic-net solver. The earlier modeling module's logistic-regression routine was replaced by that solver in the executed workflow. Publishing only the earlier module, without this correction, would not represent the final implementation.

Source-derived modules retain their relative source references and code-provenance records. Machine-local absolute paths and confidential input hashes are retained only in the private research archive, not in this repository.

## Scope of verification

Software preparation and synthetic numerical tests do not rerun the real study. In particular, they do not verify all scanner reconstruction steps, visual quality-control decisions, clinical adjudication, or the complete image-to-paper workflow. The original research code and outputs are kept separately and have not been overwritten by release preparation.

For an authorized numerical reproduction, the required materials include the frozen predictor and outcome tables, their exact participant ordering, the saved outer and inner splits, the analysis configuration, and the execution environment. Matching random seeds alone is not enough if row order or library behavior differs.

## Statistical conventions that must be preserved

The original main prediction specification used 20 repetitions of stratified outer cross-validation (five folds in the analyzed cohort) with three-fold inner cross-validation. The split seed was 20260909. The penalty grid comprised C = 0.001, 0.01, 0.1, 1, 10, 100 and L1 ratios = 0, 0.25, 0.5, 0.75, 1. Selection minimized mean validation Brier score, breaking ties within 1e-8 by the smallest C and then the smallest L1 ratio. The intercept was unpenalized.

Conditional intervals used 2,000 paired participant bootstrap draws of the mean held-out predictions with seed 20260910. Those draws are not independent modeling experiments. Repeated predictions must not be treated as additional participants.

Multiple-testing families must not be narrowed after examining results. The primary six-region family, extended clinical-adjustment family, morphometric-adjustment family and processing-sensitivity family are distinct. The processing family includes tested variants even when not all variants are displayed in the paper.

The supplemental implementation retains 12-test clinical/morphometric, 18-test extended-clinical and 96-test processing families, plus the source's secondary joint-114 correction. Processing-prediction fits use their own original plain-column engine. They are not inferred from an association p-value or replaced with the main modeling wrapper.

Retuned ablations share nested splits. Same-center scanner holdouts fit only on the training device. The separate 500-draw development bootstrap groups duplicate copies of each sampled person in inner folds and corrects full-data apparent performance for optimism. It does not retrospectively correct the reported mean-OOF AUC. See [validation.md](validation.md) and [sensitivity.md](sensitivity.md).

## Public tests and figure checks

Tests and examples use only artificial arrays, artificial clinical fields and synthetic plot payloads. Optional original-source parity tests select function definitions through AST without importing historical scripts or accessing study inputs. Source-code hashes are in [source_provenance.json](source_provenance.json); additional source-specific checks are described in the module documents.

Figure renderers require explicit inputs. Synthetic previews check layout and data handling, not the paper's values. Licensed reference anatomy must be supplied separately; it is not replaced with artificial images and presented as real anatomy. Export support does not establish that a particular journal's final size, typography or accessibility requirements have been met.

## Not included as public data

- Clinical source records, identity crosswalks and outcome-adjudication records.
- DICOM, MRI/PET derivatives, segmentation images and reference-anatomy assets.
- Real participant-level predictors, labels, split membership and held-out predictions.
- Private work manifests, logs, absolute machine paths and environment credentials.

These exclusions protect participant confidentiality; they do not convert this internal-validation study into an independent validation study. Data may be requested from the corresponding author under the conditions described in the manuscript.
