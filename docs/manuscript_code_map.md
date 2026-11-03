# Manuscript-to-code map

All modules below use explicit caller inputs. No study data, fitted patient
predictions or reference-anatomy images are bundled.

| Analysis or output | Public module | Required inputs / boundary |
| --- | --- | --- |
| DICOM conversion and selected-series BIDS promotion | `intake`, `cli` | Private reviewed manifests and original sources; fresh output only; no automatic identity or visual pass |
| Image-processing recipes and PVC extraction | `preprocessing` | Explicit image/LTA paths, separately installed FreeSurfer; recipes do not execute commands; manual QC remains external |
| PET normalization and six composites | `features`, `configs/features.json` | Accepted aligned PET, labels and brain mask; relative uptake, not SUV |
| T1 morphometry | `features` | Caller-supplied FreeSurfer statistics and eTIV |
| Primary regional association | `associations` | Six regional values, response and declared covariate design; full six-test correction |
| Extended clinical, T1 and processing adjustments | `sensitivity` | Complete ordered covariates and variant matrices; full 12/18/96 families and secondary joint-114 correction |
| C / CP / CPT prediction | `modeling`, `core`, `workflow`, `solver` | Authorized predictor/outcome tables; saved splits needed to reproduce historical estimates |
| Retuned thalamic / striatopallidal / combined ablation | `validation` | Same original-engine split plans; all reduced models retuned |
| Same-center device holdouts | `validation` | Two device labels and training-only fitting; not external validation |
| Development optimism / stability | `validation` | 500 redevelopment draws by default; duplicate-person grouped inner folds |
| Processing sensitivity prediction | `sensitivity` | Seven alternative six-PET matrices; original C/CP/CPT predictions remain separate; 28 comparison recipes |
| Figure 1 methods overview | `figures` | Explicit flow counts and supplied illustrative assets; no hardcoded study count |
| Figure 2 regional distributions/effects | `figures` | Caller effect/distribution payloads and licensed anatomy assets; not a voxelwise significance map |
| Figure 3 prediction summaries | `figures` | Held-out ROC/AUC, calibration, Brier score and supplied ablation intervals |
| Supplementary validation and processing figures | `figures` | Explicit validation/processing summaries; plotting subsets must retain original full-family q values |
| Descriptive table | `tables` | Authorized tabular input; no baseline significance tests added |
| Original cohort CLR/PCA and participant bootstrap | `vns_pattern` | Frozen 82-region order, original covariance-PCA and scanner-stratified convention |
| Molecular annotation and conditional spatial tests | `molecular_annotation`, `vns_spatial`, `spatial` | Approved receptor maps, cortical crosswalk, coordinates/gradients; fixed full families |
| Healthy and TLE independent spatial analyses | `spatial`, `spatial_tle` | Their distinct frozen lists, input matrices and sign/resampling definitions |
| SERT source and DAT comparator checks | `atlas_specificity`, `dat_conditioning` | Saved observed, bootstrap and null maps, matched atlas ordering |
| Integrated Figure 4 and final publication-label patches | Controlled review package source snapshot | Exact renderer/asset dependencies; not yet wrapped by public `figures` |

## Start with artificial data

Run `examples/synthetic_demo.py` for the main numerical path and
`examples/synthetic_supplement.py` for small supplemental smoke runs. See
[figures.md](figures.md) for rendering examples. These deliberately reduced
demonstrations are not the study's analysis settings or results.

Detailed contracts: [preprocessing](preprocessing.md), [imaging recipes](imaging_recipes.md),
[validation](validation.md), [sensitivity](sensitivity.md), [figures](figures.md).
Output records may contain participant-level information when actual research
inputs are supplied; privacy exclusions apply to generated outputs as well.
