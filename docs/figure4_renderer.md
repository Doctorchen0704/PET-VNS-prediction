# Figure 4 renderer

`pet_vns_prediction.figure4` contains the final four-panel layout, including the
heading **Within-cohort SERT associations**. It draws saved residual pairs and
saved participant-bootstrap intervals; it does not fit a PCA, estimate an
interval, generate a spatial null, or calculate a P value.

Install `pip install -e ".[figures]"`. The implementation uses NumPy, Matplotlib
and Pillow; the core scientific environment supplies NumPy. Matplotlib font
availability and library versions can affect glyph rasterization, so exact TIFF
bytes are not promised across environments.

## Required local inputs

1. A mapping with these fields:
   - `cohort_order`: the three display labels in top-to-bottom order.
   - `interval_plot_rows` (or `SERT_correspondence` from the frozen aggregate JSON):
     twelve rows with `cohort`, `n`, `map`, `adjustment`, `estimate`, and `ci95`.
     Map keys are `SERT_primary_Beliveau` and `SERT_alternative_Savli`; adjustment
     keys are `Spatial covariates` and `Spatial covariates + DAT`. These are
     within-cohort estimates, not a statistical test of differences among cohorts.
   - `scatter_data`: two rows with `condition` (`baseline` or `DAT`),
     `saved_effect`, and 68 `points`, each containing `parcel`, `x`, `y`.
     The points are saved SERT and PC1 rank residuals. The original aggregate
     interval JSON alone does **not** contain these pairs; they come from the
     frozen figure-provenance record / conditioning arrays. Do not substitute
     unadjusted measurements, subject scores, or synthetic points.
   - `PC1_color_limits` and `SERT_color_limits`: two finite ordered limits each.
     Optional `color_ticks` gives each metric's tick values. Optional
     `scatter_xlim`, `scatter_ylim`, `scatter_xticks`, `scatter_yticks` preserve
     author-approved axis settings. The renderer rejects points outside limits.
2. Eight local PNGs keyed as `PC1_lh_lateral`, `PC1_lh_medial`,
   `PC1_rh_lateral`, `PC1_rh_medial`, and the corresponding `SERT_...` keys.
   Supply approved standard-brain renders, not participant screenshots.
3. An explicit `asset_color_limits` mapping documenting the colors already baked
   into the PNGs. This must match the figure colorbars. PC1 uses orange-white-purple
   (`#C17842`, `#F8F6F1`, `#715296`); SERT uses the purple sequential palette in
   the module. The renderer does not recolor or smooth the PNGs.
4. `assets_authorized=True`: caller confirmation of permitted use. It is not a
   substitute for source attribution, applicable atlas/surface licenses, or a
   visual check that parcel mapping, hemispheres and color scales are correct.

Do not publish patient-level arrays or identity tables with these inputs. No
brain geometry, atlas images, receptor arrays or study-specific rows are shipped
in this module. The isolated tests use plainly labelled non-anatomical boxes and
artificial numeric data, only to exercise drawing and input validation.

## Call

```python
from pet_vns_prediction.figure4 import render_figure4

# plot_data, png_paths and color_limits are supplied by the authorized caller.
receipt = render_figure4(
    plot_data, png_paths, "new_figure4_output",
    asset_color_limits=color_limits, assets_authorized=True,
    dpi=600, preview_dpi=150,
)
```

The destination must not already exist. Outputs are a low-resolution PNG,
vector/text PDF, RGB TIFF with lossless compression, and a rendering receipt.
Errors preserve partial files. A successful run is marked
`RENDERED_PENDING_VISUAL_REVIEW`, never automatic visual-QC PASS. The receipt hashes
caller inputs and assets, but does not claim to independently validate atlas
licensing or anatomic mapping. Inspect the completed figure before use.

## Provenance and limits

The layout is adapted from `polish_figure4_preview.py` (the controlled study layout)
with the approved title change in `revise_labels.py`. Private local sources and
their SHA-256 identifiers are recorded in the handoff audit, not embedded as
machine-specific paths. Constants describing layout, marks and colors are
retained; sample sizes, effect estimates, intervals, point arrays and asset paths
are always caller inputs.

This is the **composition renderer**, not the fsaverage surface renderer. The
original licensed-geometry parcellation and PyVista camera/render scripts remain
in the authorized private handoff with their source hashes. A user who has only
the public module must first create the eight lawful, verified anatomical assets.
The public renderer therefore does not by itself provide an end-to-end
atlas-to-brain-image reproduction or establish clinical/technical image QC.
