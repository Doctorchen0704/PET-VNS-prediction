# Release scope: core and supplemental methods

Version 0.2.0 is a portable adaptation of the executed study methods, not the
private analysis archive or a pretrained prediction service. Source-code
lineage and synthetic checks are recorded in [source_provenance.json](source_provenance.json)
and the module documents. Historical source files are not runtime dependencies.

## Included

- PET normalization, 19 primary bilateral nodes and six regional composites;
  six T1 morphometric features from FreeSurfer statistics.
- Non-executing FreeSurfer segmentation, registration, 2-mm analysis-grid and
  PVC command recipes; explicit LTA geometry checks; sGTM coefficients and
  volume-weighted 84-label reference extraction.
- The final deterministic elastic-net solver, independent selected-fit checks,
  training-fold preprocessing, repeated nested C/CP/CPT prediction, calibration,
  decision curves and paired conditional fixed-OOF bootstrap summaries.
- HC3 regional associations; complete 6-, 12-, 18- and 96-test family interfaces,
  a secondary joint-114 correction, and leave-one-person influence diagnostics.
- Three retuned ablations on identical supplied splits; bidirectional same-center
  cross-scanner evaluation; 500-draw full-development bootstrap defaults with
  duplicate-person grouped inner tuning and optimism/stability summaries.
- The distinct processing-prediction engine: 14 CP/CPT variant models and 28
  comparison definitions. Original-engine C/CP/CPT predictions are separate inputs.
- Parameterized main and supplementary figure renderers, descriptive table
  utilities, synthetic examples and software tests. Reference-anatomy assets
  must be supplied by the caller under the relevant license.

## Outside the public release

- Real images, clinical/outcome records, individual features/predictions,
  original row ordering and split membership, private cohort rules or links,
  pretrained clinical model weights and private execution logs.
- Scanner reconstruction, DICOM conversion/identity reconciliation, automated
  eligibility decisions, manual QC, external tool installation or scheduling.
  The command recipes do not execute FreeSurfer or validate image quality.
- PetPrep execution and redistributed reference-anatomy assets. PetPrep-derived
  alternative feature inputs can be supplied to the sensitivity interface.
- Private historical file joins, source-cache locks, saved-result replay readers,
  all abandoned exploratory models, every historical descriptive report, or a
  single script generating the manuscript from raw scans.

## Verification boundary

Retained numerical functions were compared to original pure definitions on
artificial data. Checks include the primary full-grid outer-fold path, HC3/BH,
feature arithmetic, supplemental family construction, the distinct processing
outer-fold record, grouped development tuning and ablation evaluation. Small
synthetic tests exercise the portable interfaces; synthetic figures are rendered
for layout checks. Details and any edge-case adaptations are documented in the
module pages. No real-cohort fit or imaging rerun was performed for this release.

These checks verify the stated implementation properties. They do not reproduce
the paper's estimates without authorized frozen inputs, establish external
clinical validity, or test every allowed dependency version. Outputs generated
from actual participant data must remain outside this public repository.
