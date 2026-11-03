# Manual PET–VNS analysis steps

This repository provides ordinary, independently invoked research scripts. It does not run
the whole study automatically, make eligibility decisions, or approve image QC. Run one
step, inspect its output, and start the next step yourself.

The manual-script edition is 1.0.0. It retains the previously verified 0.3.0 numerical
library; reorganizing the entry points does not change the study models or results.

1. **STEP01**: dcm2niix conversion, explicit human series selection, BIDS organization.
2. **STEP02**: single-subject FreeSurfer segmentation and manual review.
3. **STEP03**: rigid PET–T1 registration, then the individual 2-mm analysis grid.
4. **STEP04**: PET regional/82-parcel and T1 features; separately invoked PVC substeps.
5. **STEP05**: keyed clinical/outcome/feature joins with the saved cohort and folds.
6. **STEP06**: prespecified regional associations and complete multiplicity families.
7. **STEP07**: repeated nested C/CP/CPT fitting; saved-prediction summaries are separate.
8. **STEP08**: independently invoked ablation, scanner, development-bootstrap and processing analyses.
9. **STEP09**: PC1, molecular annotation, spatial controls, external patterns and clinical bridge.
10. **STEP10**: assemble plot inputs, render individual figures, produce descriptive tables,
    and optionally regenerate standard-brain assets.

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
