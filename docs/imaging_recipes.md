# Imaging command recipes and partial-volume feature extraction

These are source-derived recipes for the study's **FreeSurfer 7.4.1** processing,
not an installer, a scheduler, or an automatic QC decision. The Python command
builders return argument lists and never execute them. Set up separately licensed
FreeSurfer, `FS_LICENSE`, and `SUBJECTS_DIR` in a compatible Linux environment.
Use a new processing copy and new output targets; do not run over existing results.
Hardware scheduling (`nice`, CPU affinity, cohort queues) is intentionally omitted.

## Native-space primary branch

1. `recon_all_command(t1, subject)` retains `-all -parallel -openmp 8`.
   The exceptional `-cw256` option requires an explicit prior technical decision;
   the original subject-specific list is private and is not reproduced here.
   Check the completion marker, absence of the error marker, completion log and
   segmentation visual QC; a zero exit code alone is insufficient.
2. `registration_command(...)` retains six DOF, four threads, seed 20260731,
   saturation percentile 99.9, reference brain mask and saved cost/parameters.
3. Review the rigid registration. `analysis_grid_commands(...)` then constructs
   a T1-derived 2-mm reference using cubic resampling, samples **native PET directly**
   through the accepted LTA using trilinear interpolation, and samples anatomical
   labels/brainmask using nearest-neighbor interpolation. Do not chain multiple
   PET resamplings or apply unreported smoothing.
4. Use the existing `features.normalize_pet` and `features.extract_pet_features`
   or explicit NIfTI adapter. Check numeric coverage gates and visual overlays
   before accepting measurements. No automatic clinical inclusion/exclusion is made.

## Matched-segmentation no-PVC and sGTM branches

The source workflow ran `gtmseg --usf 2 --no-seg-stats` in a separate FreeSurfer
subject copy. `gtmseg_command` exposes these options; it does not create that copy.
The generated segmentation, color table and LTA are inputs to later steps.

`pvc_geometry_commands` creates PET-to-conformed-anatomy scanner-RAS identity and
the concatenated segmentation-to-PET voxel transform. Identity applies only to
the **already T1-aligned PET grid**, never arbitrary native PET. Parse the LTAs
and apply `validate_pvc_geometry` to the actual image affines. It preserves the
source 1e-6 RAS identity and 1e-4 voxel-geometry checks. Passing these equations
does not replace visual alignment and segmentation review.

`pvc_command` retains `--default-seg-merge --no-rescale --auto-mask 1 .01`, four
threads and `--save-yhat`. Its default is `--psf 5.4`; set `fwhm_mm=None` for the
matched-segmentation `--no-pvc` comparator. The historical 4 and 6 mm runs remain
part of the complete processing multiplicity family. They are not candidates
for selecting whichever analysis produces a higher AUC.

The sGTM extraction **does not use only the six regional predictors as its reference**.
`configs/pvc_regions.json` contains 42 bilateral nodes (84 labels). The reference
is their segmentation-volume-weighted mean of unrounded coefficients. Bilateral
node means are volume weighted, divided by that reference, and combined with
equal node weights into the same six PET composites. This preserves the executed
GTM branch and is distinct from averaging rounded values in `gtm.stats.dat`.

`extract_sgtm_files` requires explicit paths to `gtm.stats.dat`, `gtm.nii.gz`, the
GTM segmentation, `XtX.mat` and `Xty.mat`. It checks coefficient alignment and
rounded-text agreement, solves the normal equations independently, reports the
condition number, and retains/flags nonpositive values rather than clipping them.
An `ATTENTION` flag is not an automatic exclusion. The public neutral success label
`PASS_NUMERICAL_CHECKS` replaces the source's historical assumed-PSF wording; it
does not certify scanner resolution or visual QC.

## Execution and verification boundary

The release tests these command arguments, transform equations and coefficient
calculations with artificial inputs. FreeSurfer reconstruction, registration and
PVC were **not rerun during publication preparation**. DICOM conversion, scanner
reconstruction, visual QC and participant identity matching remain outside this
software adapter. PetPrep is an external processing dependency, not bundled or
reimplemented here; its already extracted measurements can be supplied to the
processing-sensitivity analysis after appropriate version/protocol documentation.

All files produced from real scans, including coefficients and technical
measurements, remain potentially sensitive. Keep them outside the public repository.
