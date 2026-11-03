# Ordered reproduction workflow

This guide separates portable code from the study's controlled data and human
decisions. Follow the final analysis configuration, not old trial filenames or
whichever model has the largest AUC. The controlled review package includes a
stage registry, final row order/splits, result comparators, and source snapshots.
It must never be uploaded to this repository.

## Environments

| Layer | Executed/tested environment | Required separately |
|---|---|---|
| DICOM conversion | Windows dcm2niix v1.0.20250505, package folder labeled 1.0.20250506 | Original DICOM, reviewed patient/series mapping; converter binary |
| Original VNS image processing | Linux/WSL FreeSurfer 7.4.1; recon-all 8 threads; registration 4 threads | FreeSurfer license, raw T1/PET, accepted QC, per-subject `-cw256` decisions |
| Numerical analyses | Windows Python 3.12.4; exact tested versions in requirements-tested.txt; BLAS 1 thread | Frozen predictors, outcome labels, splits, phenotype designs and analysis-specific inputs |
| Public external-image processing | Distinct from VNS recon-all: supplied DEPICT SynthSeg 2.2.0; ds004054 SynthSeg-robust 2.0 with --parc | Permitted external images/derivatives, frozen technical QC and cohort lists |
| Spatial annotation | Python/NumPy/SciPy numerical layer, no GPU needed for supplied arrays | Licensed/approved receptor maps, fsaverage-DK parcel crosswalk, sphere coordinates, AP coordinates and functional gradient |
| Figures | Matplotlib/Pillow for portable plots; original 3D assets additionally use PyVista/VTK and Node/sharp | Reviewed numerical payloads, fonts and reference-anatomy assets |

Today's installed package version is not proof of the version used for every
historical stage. Unrecorded exact environment details are identified in the
controlled registry, not silently replaced with the current defaults. Installation
and numerical validation on a new machine are separate from imaging validation.

## 1. Source identity and DICOM conversion

Use the confidential source crosswalk. Folder order is not an identity mapping.
The conversion manifest has `schema_version: 1` and a `rows` array. Each row
contains `subject` (alphanumeric label without `sub-`), `modality` (`T1` or `PET`),
`source_relative`, `identity_reviewed: true`, and `series_reviewed: true`.
Those booleans record an existing review; this code does not perform it.

```sh
python -m pet_vns_prediction.cli convert-dicom --manifest /private/conversion.json --raw-root /private/DICOM --output /private/new_staging --converter /tools/dcm2niix
# Review the printed plan. Then repeat with --execute only in a fresh output directory.
```

The converter uses `-b y -ba y -z y -w 0 -f series-%s_desc-%p` as in the original
workflow. Logs remain private. Patient metadata, dates or facial anatomy can
remain even when the converter's anonymization option is enabled; conversion
does not authorize publication of its outputs.

## 2. Review and promote selected series

The promotion manifest lists exact `source_nii`, `source_json`, `nii_sha256`,
`json_sha256`, `target_nii`, `review_pass: true` and any explicitly reviewed PET
metadata patch with `metadata_patch_reviewed: true`. Recreated gzip bytes may differ across converter/platform
versions; investigate a hash difference and compare voxel/header content rather
than replacing a frozen reference hash to force acceptance.

```sh
python -m pet_vns_prediction.cli promote-bids --manifest /private/selected_series.json --staging-root /private/new_staging --output /private/new_bids
```

Preserve the declared PET units. No injection timing or SUV is inferred from an
approximate departmental uptake protocol. Run a BIDS validator separately and
retain its report. The historical project has internal organization/metadata
checks; do not describe these as an official BIDS-validator pass.

## 3. FreeSurfer, rigid registration and analysis grid

Use `preprocessing.recon_all_command`, `registration_command`, and
`analysis_grid_commands`. They return exact argument lists, not shell strings.
Execute them in a fresh working directory with `SUBJECTS_DIR` and `FS_LICENSE`
set explicitly. Do not run over a completed FreeSurfer subject.

Stop for source/segmentation QC before registration and for visual alignment and
coverage QC before extraction. The final accepted technical decisions are
included in the controlled package. A numerical check is not a visual pass.
Six-DOF registration uses seed 20260731 and saturation 99.9. Native PET is sampled
directly to the individual 2-mm grid, with no extra smoothing/PVC in the primary
regional branch. PET uses trilinear and label/mask images nearest-neighbor
resampling. Reference T1 uses cubic resampling. See preprocessing.md.

## 4. Features and data assembly

Use `features.normalize_pet`, `extract_pet_nifti` and `extract_t1_stats` with
`configs/features.json`. Keep reference-mask zero handling distinct from positive
voxel ROI extraction. Structural features come from FreeSurfer stats/eTIV, not
PET-grid label volumes. Use `preprocessing.extract_sgtm_files` for the separately
specified PVC comparison. Do not mix alternate references or segmentations into
the primary feature table.

Join by the verified study key, never worksheet row order. Final outcomes and
the original clinical source values are separate fields in the controlled
workbook. Missing detailed follow-up counts must remain missing; binary response
does not determine an exact postoperative frequency or reduction percentage.

## 5. Main numerical reproduction

The authorized `model_input` JSON supplies `study_ids`, `x` (record dictionaries),
`y`, exact C/CP/CPT `columns`, and `plans`. Preserve all of them.

```sh
python -m pet_vns_prediction.cli predict --input /private/model_input.json --output /private/new_refit
```

The default performs all 20 repeats with the complete tuning grid. It refuses an
existing output directory. Failures are retained and are not automatically
retried with changed parameters. `--summary-only` is explicitly different: it
requires `saved_predictions` and only recalculates summaries from saved outputs.
The conditional bootstrap is not a full-development bootstrap.

Use the documented `associations`, `validation`, and `sensitivity` interfaces
for the remaining fixed families/ablations. The controlled registry identifies
the exact designs, inputs, comparison results and commands for each stage.

## 6. Spatial pattern, molecular annotation and external cohorts

See spatial_extension.md. Original and external PCA sign/resampling conventions
are intentionally separate. Supply the exact 82-region and 68-cortical orders,
maps and null draws. Do not substitute a clinical outcome vector into the
unsupervised decomposition. External healthy and TLE spatial analyses are not
external validation of the VNS outcome prediction model.

## 7. Tables and figures

Use `tables` and `figures` for parameterized plots, and the controlled final
rendering registry for the exact four-main/two-supplement publication set.
Original renderer dependencies are retained when required by dynamic imports;
their older filenames do not make them additional analyses to run. The new
Figure 4 and final label revisions are not reproduced by an older generic
Figure 1/3 renderer alone. Check the data payloads before checking pixels.

## Known non-automatic steps

Clinical adjudication, source identity, manual technical QC, third-party licenses
and access to department follow-up records cannot be recreated by software.
Six historical PetPrep intermediate PET images were documented as unavailable;
the retained alternative-feature tables support numerical replay, but that
particular historical image-extraction chain has not been verified end to end.
Abandoned trials are excluded from the recommended run order. Scientifically
relevant version decisions and corrections remain in the controlled audit trail.
