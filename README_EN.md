# Manual PET–VNS analysis steps

This repository provides ordinary, independently invoked research scripts. It does not run
the whole study automatically, make eligibility decisions, or approve image QC. Run one
step, inspect its output, and start the next step yourself.

The manual-script edition is 1.1.0. The previous STEP01–10 scripts and 0.3.0 numerical
library are retained. This edition adds a separate PET-scanner-adjusted four-model
comparison; it does not replace the earlier three-model analysis.

1. **STEP01**: dcm2niix conversion, explicit human series selection, BIDS organization.
2. **STEP02**: single-subject FreeSurfer segmentation and manual review.
3. **STEP03**: rigid PET–T1 registration, then the individual 2-mm analysis grid.
4. **STEP04**: PET regional/82-parcel and T1 features; separately invoked PVC substeps.
5. **STEP05**: keyed clinical/outcome/feature joins with the saved cohort and folds.
6. **STEP06**: prespecified regional associations and complete multiplicity families.
7. **STEP07**: original C/CP/CPT fitting and a separate scanner-adjusted C/CT/CP/CPT branch.
8. **STEP08**: independently invoked ablation, scanner, development-bootstrap and processing analyses.
9. **STEP09**: PC1, molecular annotation, spatial controls, external patterns and clinical bridge.
10. **STEP10**: assemble plot inputs, render individual figures, produce descriptive tables,
    and optionally regenerate standard-brain assets.

### Scanner-adjusted four-model comparison

See [the four-model guide](STEP07_prediction_models/scanner_adjusted_four_models/README.md).
All four models include an unpenalized PET-scanner contrast. CT contains the clinical
and T1 features; CPT additionally contains PET features. Saved outer partitions are
preserved. Within each outer training set, the original inner seed is reused to construct
three inner folds stratified jointly by scanner and outcome. The same amended inner
partitions are shared by all four models; they are not the earlier inner partitions.

Input preparation, fitting each model and result summarization are separate manual
commands. Participant-linked inputs, saved folds, predictions and fitted coefficients
remain private. See [CHANGELOG.md](CHANGELOG.md) for release scope.

Install Python dependencies from `environment/requirements.txt`, then use `python -m pip
install -e . --no-deps` from the repository root. Image processing separately requires
licensed FreeSurfer 7.4.1 on Linux/WSL and dcm2niix v1.0.20250505. Standard-brain 3-D
rendering additionally needs PyVista/VTK; approved methods-SVG export optionally uses
ordinary Node.js and Sharp. See the environment guide and each step's `--help`.

Private clinical records, identified Excel files, real predictors/outcomes, raw images,
saved folds, licensed anatomy and normative maps are **not** distributed here. Authorized
reviewers receive a separate input companion; public users must supply their own permitted
data and reviewed QC decisions. Templates/examples are artificial.

The external pattern steps start from frozen ROI measurements; this repository does not
claim a fresh reproduction of every external image-processing decision. Six historical
PetPrep image intermediates were unavailable, although their saved features support
numerical sensitivity replay. Neither software tests nor external spatial comparisons
constitute independent validation of the VNS response prediction model.

For the approved Figure1, export its editable SVG rather than substituting the generic
redrawing option. Re-rendered anatomy and all final layouts require visual review.

Use the exact commit for citation. The authors' code uses the MIT license; external
software/data retain their own terms. The detailed operator guide is [README.md](README.md).
