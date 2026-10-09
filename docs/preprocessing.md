# Image preprocessing and feature definitions

## Scope of the public implementation

`src/pet_vns_prediction/features.py` implements the feature calculations derived
from the study's PET and FreeSurfer extraction scripts. It accepts explicitly
provided image arrays, aligned NIfTI files, or three FreeSurfer statistics files.
It contains no participant identifiers, outcomes, images, or cohort selection
rules. The configuration is `configs/features.json`.

This is **not a one-command DICOM-to-paper pipeline**. Image identity checks,
clinical eligibility, segmentation and registration review, and the accepted
transforms belong to the study's controlled preprocessing records. The public
software does not recreate those decisions or supply the underlying data.
Synthetic tests check calculations and file interfaces; they do not establish
full image-processing or manuscript-result reproduction.

## Study preprocessing represented here

The feature sources were generated with FreeSurfer 7.4.1. After segmentation
and registration QC, native PET was resampled using the accepted rigid transform
directly onto an individual T1-aligned, 2-mm isotropic grid. The grid reference
was generated from T1 with cubic resampling. PET used trilinear interpolation;
the `aparc+aseg` labels and brain mask used nearest-neighbor interpolation. This
regional primary-feature branch did not apply additional PET smoothing or PVC.
Separate processing-sensitivity analyses must not be confused with this branch.

The labels correspond to FreeSurfer's Desikan-Killiany `aparc+aseg` parcellation.
FreeSurfer, its license, and its anatomical data are separate dependencies; no
FreeSurfer binaries, license files, or atlas image files are distributed here.
The small label-number/region-name mapping in the configuration describes the
features rather than distributing an atlas image.

The public NIfTI adapter checks matching PET/label geometry and 2-mm voxel size;
it does not estimate a transform or resample mismatched images. Matching headers
alone do not establish correct registration or participant identity.

## Relative PET uptake

The primary reference was supratentorial gray matter. Whole-brain and cerebellar
cortex references were separately constructed for sensitivity analyses; their
features should not be mixed within one input matrix.

The source implementation used these masks, intersected with the brain mask and
finite PET values:

- Supratentorial gray matter: cortical label numbers in the half-open interval
  `[1000, 3000)`, plus bilateral thalamus, caudate, putamen, pallidum, hippocampus,
  amygdala, accumbens, and ventral diencephalon.
- Whole brain: the resampled FreeSurfer brain mask, not a gray-matter-only mask.
- Cerebellar cortex: labels 8 and 47.

Each PET image was divided by the arithmetic mean within the chosen mask and
saved as float32. These are **relative uptake ratios, not SUVs**. The reference
mean includes finite zeros; the subsequent regional means use finite values
strictly greater than zero. These distinct rules are retained from the source.

The original image-level checks also required no negative or nonfinite PET
values, at least 95% positive brain-mask coverage, positive reference means,
and at least 5,000/10,000/300 reference voxels for supratentorial gray matter,
whole brain, and cerebellar cortex, respectively. The public normalization
function reports the count/coverage gate separately from geometry and visual
QC. Tiny synthetic arrays intentionally do not pass the real-image size gate.

## Nineteen bilateral nodes and six PET composites

For each node, left and right voxels were pooled before calculating the mean.
This is not an equal-weight average of the two hemisphere means. Composite
values are then unweighted arithmetic means of their constituent bilateral node
means, so larger nodes do not automatically receive more composite weight.

| PET composite | Constituent bilateral nodes |
|---|---|
| Thalamus | Thalamus |
| Cingulate | Caudal anterior, rostral anterior, posterior, and isthmus cingulate |
| Insula/frontal operculum | Insula, pars opercularis |
| Limbic/medial temporal | Hippocampus, amygdala, entorhinal cortex, parahippocampal cortex, temporal pole |
| Striatopallidal | Caudate, putamen, pallidum, accumbens |
| Sensorimotor | Precentral, postcentral, paracentral |

Left/right means, median uptake, and the signed asymmetry index
`2 * (left - right) / (left + right)` are retained as extraction outputs, not
automatically added to the six-feature prediction model.

ROI values are missing with fewer than 10 label voxels or less than 80% valid
PET coverage. A label count of 10–24 or valid coverage from 80% to below 95%
triggers an attention flag. A composite value is missing if any constituent
bilateral mean is missing. Hemisphere QC flags are retained independently: a
pooled bilateral mean can exist while one hemisphere is flagged missing. Such
flags are not permission to silently use the value. No extraction-stage
imputation, winsorization, or outcome-dependent exclusion is performed.

The original extractor also generated full-atlas and ancillary summaries. This
public feature interface is restricted to the 19 nodes underlying the six PET
composites; it does not turn the full atlas into additional primary predictors.

## Six structural MRI features

The structural features came from `aseg.stats`, `lh.aparc.stats`, and
`rh.aparc.stats`, not from PET-grid voxel counts. Bilateral volumes were summed
and divided by estimated total intracranial volume (eTIV). Cortical thickness
was averaged with surface-area weights across all specified parcels in both
hemispheres.

| Structural feature | Exact definition |
|---|---|
| Thalamic volume | Bilateral thalamus volume / eTIV |
| Cingulate thickness | Area-weighted thickness of the four bilateral cingulate parcels above |
| Insula/operculum thickness | Area-weighted thickness of bilateral insula and pars opercularis |
| Medial temporal volume | Bilateral hippocampus plus amygdala volume / eTIV |
| Striatopallidal volume | Bilateral caudate, putamen, and pallidum volume / eTIV |
| Sensorimotor thickness | Area-weighted thickness of bilateral precentral, postcentral, and paracentral parcels |

The structural composites are not exact copies of PET node membership: medial
temporal **volume** does not incorporate temporal cortical parcels, and
striatopallidal **volume** does not incorporate the accumbens. This preserves
the executed structural-feature specification. A missing, nonfinite, or
nonpositive required component raises an error; no reduced-component substitute
is calculated.

## Explicit-input examples

```python
from pet_vns_prediction.features import (
    load_feature_config, extract_pet_nifti, extract_t1_stats,
)

config = load_feature_config("configs/features.json")
# Supply approved, local input paths. No data are bundled with the repository.
pet_result = extract_pet_nifti(normalized_pet_path, aligned_labels_path, config)
t1_result = extract_t1_stats(aseg_stats_path, left_aparc_stats_path,
                             right_aparc_stats_path, config)
```

`extract_pet_features` and `extract_t1_features` provide the same calculations
for in-memory arrays/statistics. `normalize_pet` requires arrays already on a
common grid and returns the normalized array without saving or modifying input
files. Users remain responsible for permissions and for keeping outputs that
contain individual data outside the public repository.
