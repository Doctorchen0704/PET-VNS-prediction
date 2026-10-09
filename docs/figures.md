# Figure and descriptive-table code

These functions port the existing manuscript plotting logic. They are not new
analyses or alternative figure designs. No real plotting JSON, participant
rows, fitted models, predictions, reference anatomy, images, or fonts are
included. The only supplied inputs are explicitly SYNTHETIC examples.

## Coverage and limits

| Function | Source-derived coverage | Not included |
| --- | --- | --- |
| `figure1_workflow` | Final A/B/C geometry, cohort flow, image stacks, separate T1 morphometry path, model combinations and nested-validation schematic | Template-volume loading, PET-like reference artwork generation, original images and counts |
| `figure2_regional` | Final v7 panel order/positions; supplied anatomy assembly; signed purple/orange scale; primary/extended forest and q table; complete focal raincloud distributions | Actual anatomy generation, surface/slice rendering, atlas downloading, template bundles, real coordinates and observations |
| `figure3_prediction` | Final ROC, five-bin Wilson calibration, Brier and full-minus-reduced AUC panels | Prediction fitting, uncertainty generation, real saved ROC/calibration/ablation values |
| `figure_s1_validation` | Revised heading clearance; cross-device metrics; saved median and percentile ranges | Device labels, cohort sizes, per-person saved ranks/ranges from the study; refitting |
| `figure_s2_processing` | Six displayed pipeline rows, regional effects and prediction metrics; original family declaration | Re-analysis, subset-wise FDR correction, study results |
| `cohort_descriptives` | Table 1 numerical summaries: all/responders/nonresponders; mean, sample SD, median, quartiles, range, category counts/percentages | Publication table formatting, Tables 2/3 manuscript templates, clinical-record extraction or adjudication |

Layouts, colors, marker shapes, line styles, interval semantics and default axis
limits follow the source renderers. Figure 1 accepts counts rather than
embedding them; S1's rank axis derives from the supplied number of ranks. The
source's study-specific assertions and private imports were removed, replaced
with input-shape, bounds and reconciliation checks. These are portable
adaptations, **not a claim of pixel-identical reproduction**. Exact appearance
also depends on installed font, Matplotlib version and caller-owned assets.

The module performs no filesystem discovery. Functions return Matplotlib
figures without writing anything. No function downloads data or fonts.

## Inputs

All numerical inputs must be finite. Supply reviewed summaries whose labels,
row order, population and uncertainty definitions agree; do not use these
renderers to infer provenance. The concrete SYNTHETIC schemas are in
`examples/synthetic_figures.py:make_synthetic_inputs`.

### Figure 1

`figure1_workflow(counts, template_stacks, period=..., exclusions=...)`

- `counts`: integer `vns_implantation`, `imaging_candidates`, `primary`,
  `responders`, `nonresponders`. Nested counts and response totals are checked.
- `exclusions`: three `(label, count)` pairs summing to candidates minus primary.
- `period`: caller-reviewed text; no study dates are supplied.
- `template_stacks`: `pet`, `t1`, `overlay`, each a list of three `AnatomyAsset`
  objects in back-to-front order, already rendered at the chosen levels.
- `feature_counts`: clinical/PET/T1w counts. Defaults `(5, 6, 6)` are method
  specifications, not participant counts. CV defaults are five outer folds,
  twenty repeats and three inner folds. The outcome concept is the original
  12-month, 50% seizure-reduction threshold; this is not a generic outcome-flow
  renderer.

### Figure 2

`figure2_regional(data, anatomy_assets)` requires:

- `raw_relative_values`: N by 6 matrix and aligned binary `y`, in fixed region
  order: thalamus, cingulate, insula/frontal operculum, limbic/medial temporal,
  striatopallidal, sensorimotor. The caller is responsible for row alignment.
- `primary`: six rows with `difference_pooled_sd`, `ci_low_pooled_sd`,
  `ci_high_pooled_sd`, `q_family`.
- `extended`: six aligned rows with `standardized_difference`,
  `standardized_CI95=[lower, upper]`, `q_family`.
- `primary_family_size=6`, `extended_family_size=18`; q values are supplied,
  never recalculated from the displayed rows. These declarations do not verify
  that the caller actually corrected the complete family.
- `shared_distribution_xlim=[lower, upper]`, encompassing both focal regions.
- `anatomy_assets`: exactly `left_lateral`, `left_medial`, `axial`, `coronal`,
  `deep`. All are pre-rendered, pre-cropped `AnatomyAsset` objects. They must use
  the caller-verified effects and source purple/orange scale with limits
  `[-1.3, 1.3]`. The function cannot establish pixel-to-effect consistency.

`AnatomyAsset(image, source, license, annotations={}, orientation=("", ""))`
accepts a caller path, PIL image or array. Source/license strings are required
declarations, **not license validation**. Annotation endpoints, if supplied,
are post-crop axes fractions from real verified geometry, not invented map
coordinates. Orientation labels also come from the caller. The public demo
uses flat typographic placeholders saying **SYNTHETIC / NOT ANATOMY**; they are
not anatomical stand-ins or evidence that anatomy generation is reproduced.

The distributions preserve the source method: a single absolute Gaussian KDE
bandwidth `1.06 * SD(all focal values, ddof=1) * N_focal**(-1/5)`, one common
density-to-height scale for all four distributions, median/IQR boxes, **min–max
whiskers**, all individual points, and deterministic display-only jitter. The
raw distributions are unadjusted; adjusted inference belongs to the forest.
The anatomy/forest encode the same regional estimates, not independent evidence.
Composite estimates do not support separate-nucleus, voxelwise, hemisphere or
connectivity inference.

### Figure 3

`data['models']` has exactly the plotted model roles `C`, `CP`, `CPT`. Each
requires `roc_fpr`, `roc_tpr`, `auc`, `auc_conditional95`, `brier`,
`brier_conditional95`. `CP` also has `frozen_calibration` with
`intercept_offset_slope_fixed_1` and `slope`. Saved ROC points are neither
smoothed nor recomputed. The caller must ensure metric/curve consistency.

`data['calibration']` contains `fixed_bin_edges` and five ordered `bins`.
Each bin has `bin` (1–5), `n`, `mean_probability`, `observed_fraction`,
`wilson95_descriptive`. Empty-bin summaries must be `None`, not zeros; their
line segment is broken. The helper `fixed_bin_calibration(y, probabilities)`
explicitly computes the same five equal-width bins using
`np.digitize(probabilities, np.linspace(0,1,6)[1:-1], right=False)`.
Consequently machine-represented edge values follow NumPy exactly, as in the
original source. These are not quantile bins.

`data['ablation']` contains three ordered rows for removal of thalamus,
striatopallidal, both, each with `delta_auc_CP_minus_reduced` and `conditional95`.
Asymmetric and zero-width intervals are preserved. The renderers do not infer
intervals from point estimates or unpaired comparisons. AUC/Brier/ablation
intervals are the supplied **fixed-prediction conditional** intervals; Wilson
intervals are rough **pointwise descriptive** bin summaries. Neither includes
all uncertainty from dependent cross-validation, retraining or model selection.

### Supplementary figures

S1 `cross_scanner` has four rows: `direction` 0/1 crossed with `model` C/CP,
`training_device`, `test_device`, `n_train`, `n_test`, `auc`,
`auc_conditional95`, `brier`, `brier_conditional95`,
`calibration_offset_intercept`. Use currently verified device labels, including
the revised device naming when reproducing the study. No device names are
hard-coded here. `ranked_saved_CP_percentiles` must already be ordered and has
`rank`, `p2p5`, `p50`, `p97p5`. The plot neither sorts nor computes quantiles.
Saved refitting percentile ranges are not advertised as independent confidence
intervals for an individual's future outcome.

S2 has `order` (six variant keys), `labels`, `regions` (two keys),
`association_rows[region]` (ordered `variant`, `standardized_difference`,
`standardized_CI95`, `q_family`) and `models[variant]` (AUC/Brier and conditional
intervals as above). Require `q_family_size=96` and
`not_displayed_but_retained_in_family=['pvc4','pvc6']`: hidden 4/6 mm rows remain
in the correction family. q is not recalculated or shown as a new subset
analysis. Method defaults include the original 5.4 mm PVC condition; this is
not evidence of a separately measured scanner-specific PSF.

### Descriptive table

`cohort_descriptives(rows, y, continuous=[...], categorical=[...])` returns a
data dictionary. It requires complete requested fields and both groups; missing
values are not silently dropped. SD uses `ddof=1` (`None` for one-row groups),
and quantiles use NumPy's default linear interpolation. Shared category levels
include zero counts in individual groups. Original history-reported monthly
frequency can be reconstructed explicitly from a verified frozen log1p field
with `np.expm1`; it is not reinterpreted as an exact 30-day event count. No
identifier joins or clinical recoding are guessed.

## Rendering and verification

Install the optional `figures` dependencies. These functions use Matplotlib,
NumPy and Pillow; they do not need FreeSurfer, fsaverage or a licensed font
bundle. Arial is the source font; an installed DejaVu Sans fallback is allowed.

```sh
python -m unittest discover -s tests -p test_figures.py -v
python examples/synthetic_figures.py --output /tmp/synthetic-figure-qa --dpi 110
```

On restricted Windows environments, set `TEMP`/`TMP` to an authorized writable
temporary directory before running tests. All exported example files must go
outside the public source tree. Generated examples and PNG QA images are not
part of this code-only release.

`export_figure(fig, path, dpi=...)` refuses overwrite, checks text against the
rendered canvas, preserves SVG text, and renders TIFF directly to native RGB
pixels with lossless LZW compression (not enlargement of a preview). For
high-resolution exports containing anatomy, pass `minimum_asset_dpi=600` as
well as `dpi=600`; insufficient source raster resolution fails explicitly.
The default manuscript axes fail when supplied marks/intervals would be
clipped. Explicit wider limits can be supplied for other datasets; that is a
declared departure from the source axes. Always visually review your output.

Seven synthetic-only tests exercise curve/interval transfer, empty bins,
Wilson arithmetic, point coverage, supplied q values, family guards, layout
bounds, native TIFF/SVG export, refusal to overwrite, asset-resolution checks,
and Table 1 counts/quantiles. Low-resolution synthetic previews were rendered
and inspected for all five figures. This is **code/layout QA, not reproduction
of study results or validation of unshipped anatomy**.

## Original code lineage

Only original source code was inspected for this port. Historical source paths
and data hashes are intentionally omitted. The following SHA-256 values identify
the source code versions, not bundled third-party assets or private data:

| Original source | Code SHA-256 |
| --- | --- |
| `Figure1_text_revised_renderer.py` | `d16d0037e3e41161c46c45f552ce8c9d6a7027516ab84922c32e0d561034208b` |
| `Figure3_text_revised_renderer.py` | `2e3cdcaa55a96411a4e983021b43b9d9c51304b9586d53a806f2ad84307f8f9c` |
| Figure 2 v7 `export_figure.py` | `00ed5dac0c63b1603073498c52066ec821ad5b2ae2144953df12835b54591079` |
| Figure 2 v6 `build_preview.py` | `da2e425074c3155b55f217d94bdd1dfb95c195a0c065c711ac33941a27d7f435` |
| Figure 2 v2 `build_figure2_v2.py` | `52aeb96930606d70df085cdc5cf37e8a014507e119561968f47b741ef36724c9` |
| Figure 2 v1 `build_figure2.py` | `985a5bc01aea35f1ccb182fbef98a620a9ebc0c12c7d11ced8ef098c52542b0f` |
| `revise_figure_s1.py` | `9b3745b9709fba4b70eec71b1f0851a333ab91fbae318d2862c2c8bc943d9a79` |
| Supplement `export_600dpi.py` | `893ce4ef1113081ec86d493f37de01c9341110335140c93acbf6863c2e3ea843` |
| `Validation_build_preview.py` | `9b8861344fea633eabf2cafec394cfa2a3d9d4ba7ca2e941c8c6283f241c41ce` |
| `build_Processing_preview.py` | `6b59e87eacb1ab16cdf690b97bac0f46a34292c1450403b8756613af13379f12` |
| `build_Figure3_prediction.py` | `dcce5bd7aaa5fbdf5730dc0551667da9183fd7f9c6e274467ca2fd7a6cf01381` |
| `full_cohort_20260921_descriptives.py` | `6b27376aa1a7cdb4a32f9377073bb13564f0f6058c3cb32456b1d99825758173` |

`tables.py` is adapted from `full_cohort_20260921_descriptives.py`; the original
script's private source loading, cohort assertions and result writers are not
ported. Other manuscript table formatting templates are not claimed as covered.
