# Cortical metabolic-pattern and molecular-atlas extension

These modules expose the numerical methods used by the integrated manuscript.
Inputs are supplied by an authorized researcher. No patient matrix, outcome,
per-person prediction, external image, receptor map, anatomical template or
licensed spatial resource is distributed here.

## Keep the original and external analyses distinct

| Branch | Numeric entrypoints | Fixed production settings |
|---|---|---|
| Original cohort PC1 | `vns_pattern.clr`, `fit` | 82 positive regional measurements; row CLR; intercept residualization and double column-centering; covariance PCA, no parcel scaling; largest-absolute loading positive in each original component |
| Original participant uncertainty | `vns_pattern.scanner_stratified_bootstrap` | 2,000 whole-person draws within sorted scanner strata; seed 20261025; refit center and PC1; orient to original 82D loading; retain every draw |
| Original cortical spin annotation | `molecular_annotation.statistics`, `bh`; `spatial.spin_generator` | right-34 then left-34 matched parcels; 9,999 coupled Hungarian spins, seed 20261023; Spearman effects; two-sided add-one tails with 1e-12 tie tolerance; BH across 18 receptor targets |
| Original conditional spatial sensitivity | `vns_spatial.design`, `graph`, `test_family` | rank PC1/target maps; intercept, hemisphere, rank AP and rank functional G1; no residual reranking; complete residual-space MSR with 9,999 signs, seed 20261027; BH18; paired-spin residual diagnostic is a separate sensitivity |
| External healthy cohort | `spatial.analyze_cohort` | unscaled 82D covariance PC1; observed sign anchored to bilateral pallidum, not similarity or receptors; 2,000 ordinary participant bootstraps; 9,999 spatial draws; all 18 targets; separate raw Pearson MSR, secondary Spearman spin and conditional rank MSR |
| Adult TLE external cohort | `spatial_tle.analyze` | same external PC1 convention; three primary targets and Holm3; two conditional targets and Holm2; `inferential=False` for effect/stability-only sensitivity |
| Alternative atlas and DAT controls | `atlas_specificity.analyze`, `atlas_specificity.holm`, `dat_conditioning.compute` | reuse saved observed PC1, all 2,000 bootstrap maps and required saved 9,999 nulls; alternative-SERT tests form one four-test Holm family across the two external cohorts; rank5 DAT controls are effect/interval only, with no reused rank4 P value |

The external seed dictionary is fixed in `spatial.SEEDS`: bootstrap 2026102401,
nonconditional MSR 2026102402, spins 2026102403 and conditional MSR 2026102404.
The original and external fits are not interchangeable: they differ in
scanner-stratified resampling, sign anchoring and some original numerical
centering conventions. Do not replace historical saved inputs with whichever
branch gives a preferable result.

## Required array order and preparation

- Original raw means are participant by 82 regions. Exactly reproduce the
  private frozen participant and feature order for a manuscript replay.
- Pass the 68 cortical indices using the verified hemisphere/parcel crosswalk,
  right34 then left34, with matching homologues. Subcortical parcels are not
  included in molecular-atlas inference.
- The 68-by-18 receptor matrix follows `spatial.TARGETS`. `5HTT` is SERT.
- Spherical coordinates are genuine fsaverage sphere centroids; AP coordinates
  come from pial RAS y, not from the sphere. G1 must use the corrected supported
  functional-gradient parcel means. The earlier zero-fill gradient output is
  not an accepted input.
- `atlas_specificity` and `dat_conditioning` expect three maps in order:
  primary SERT (Beliveau), alternative SERT (Savli), DAT (Dukart FP-CIT SPECT).
  DAT is a spatial comparator, not a measured causal patient covariate.
- `vns_spatial.test_family` takes a dictionary with `z`, rank-transformed
  `yr`, `W`, `coords`, `names`, and `spins` for the paired-spin diagnostic.
  Construct the conditional basis as `Q = scipy.linalg.null_space(z.T)`;
  `basis = Q @ numpy.linalg.eigh(Q.T @ W @ Q)[1]`. No modes are discarded.

## Reuse and limitations

The source provenance JSON records the original relative code paths and hashes.
Pure functions were extracted without numerical edits, except explicitly
documented caller-supplied metadata and the portable original-cohort bootstrap
wrapper. Synthetic checks cover original and external PCA conventions,
scanner-stratified resampling, conditional residuals, MSR invariants, coupled
bijective spins, multiplicity, tail ties and DAT conditioning.

Passing these checks is not a real-data reproduction. This addition does not
implement imaging downloads, data-use approval, atlas fetching/parcellation,
manual QC adjudication, the full private participant joins, or every abandoned
exploratory analysis. The original full 21-gene annotation, parcel influence and
other16-receptor-axis descriptive branches are not wrapped by this extension;
preserve their actual scripts in the controlled review package if those results
are requested. Existing clinical bridge and figure orchestration are separate
parts of the project, not silently reproduced by these modules.

## Environment

Numerical modules require Python, NumPy and SciPy already declared by the main
package. They do not require FreeSurfer, GPU, browser, network or source-volume
access. Source runs used single-thread BLAS settings; set
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1` before Python
starts. Preserve exact Python/NumPy/SciPy/BLAS versions in a reproduction log:
SVD/eigendecomposition sign choices or tied eigenspaces can differ by library.
Use the original pinned saved spatial permutations/signs and cohort ordering
when exact numerical tails are required. Portable kernels do not themselves
make an installation on a different operating system an exact replay.

Run `python -m unittest discover -s tests -p test_spatial_extension.py -v`.
Tests use artificial arrays only. The full atlas wrapper test uses 2,000
artificial bootstrap maps and 9,999 artificial null maps to check shape/branch
contracts; it does not read or fit a study dataset.
