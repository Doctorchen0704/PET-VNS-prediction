"""Portable final Figure 4 layout; caller-owned aggregate data and PNG assets.

No atlas download, brain mesh generation, cohort data loading or statistical
refitting. Brain color limits must be declared by the asset producer; a PNG
alone cannot prove its anatomical mapping or quantitative color calibration.
See docs/figure4_renderer.md for the input contract and source provenance.
"""
from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
import hashlib
import json

import numpy as np

PURPLE, ORANGE = "#756098", "#D48249"
MAPS = ("SERT_primary_Beliveau", "SERT_alternative_Savli")
CONDITIONS = ("Spatial covariates", "Spatial covariates + DAT")
ASSET_KEYS = tuple(f"{metric}_{hemi}_{view}" for metric in ("PC1", "SERT")
                   for hemi in ("lh", "rh") for view in ("lateral", "medial"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value, label):
    if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{label}: finite numeric value required")
    return float(value)


def _limits(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label}: two ordered numeric limits required")
    lo, hi = [_number(v, label) for v in value]
    if lo >= hi:
        raise ValueError(f"{label}: increasing limits required")
    return lo, hi


def _prepare(plot_data, anatomy_assets, asset_color_limits, assets_authorized):
    from PIL import Image
    if assets_authorized is not True:
        raise ValueError("Caller must confirm authorization to use the supplied assets")
    if not isinstance(plot_data, Mapping) or not isinstance(anatomy_assets, Mapping):
        raise ValueError("Explicit plot data and asset mappings are required")
    # Copy via finite JSON, preserving caller values and rejecting non-JSON data.
    data = json.loads(json.dumps(plot_data, allow_nan=False))
    order = data.get("cohort_order")
    if not isinstance(order, list) or len(order) != 3 or len(set(order)) != 3:
        raise ValueError("This layout requires an explicit order of three cohorts")
    if not all(isinstance(k, str) and k.strip() and len(k) <= 30 for k in order):
        raise ValueError("Cohort labels must be nonempty and at most 30 characters")
    intervals = data.get("interval_plot_rows", data.get("SERT_correspondence"))
    if not isinstance(intervals, list) or len(intervals) != 12:
        raise ValueError("Twelve cohort/map/adjustment interval rows are required")
    indexed, sizes = {}, {}
    for row in intervals:
        key = (row["cohort"], row["map"], row["adjustment"])
        if key in indexed or key[0] not in order or key[1] not in MAPS or key[2] not in CONDITIONS:
            raise ValueError("Duplicate or unexpected interval row")
        estimate = _number(row["estimate"], "correlation estimate")
        lo, hi = _limits(row["ci95"], "95% interval")
        if not (-1 <= estimate <= 1 and -1 <= lo < hi <= 1):
            raise ValueError("Correlations and ordered intervals must lie within [-1, 1]")
        # Percentile bootstrap intervals need not contain the observed estimate.
        n = row.get("n")
        if type(n) is not int or n < 3:
            raise ValueError("Each interval row requires an explicit integer n >= 3")
        if key[0] in sizes and sizes[key[0]] != n:
            raise ValueError("Cohort sample size differs across rows")
        sizes[key[0]], indexed[key] = n, row
    expected = {(c, m, a) for c in order for m in MAPS for a in CONDITIONS}
    if set(indexed) != expected:
        raise ValueError("Interval combinations are incomplete")
    scatter = data.get("scatter_data")
    if not isinstance(scatter, list) or len(scatter) != 2:
        raise ValueError("Two saved rank-residual scatter panels are required")
    scatters = {}
    xlim = _limits(data.get("scatter_xlim", [-40, 40]), "scatter x limits")
    ylim = _limits(data.get("scatter_ylim", [-45, 45]), "scatter y limits")
    for row in scatter:
        condition = row.get("condition")
        if condition not in ("baseline", "DAT") or condition in scatters:
            raise ValueError("Scatter conditions must be baseline and DAT, once each")
        points = row["points"]
        if len(points) != 68 or len({p["parcel"] for p in points}) != 68:
            raise ValueError("Exactly 68 uniquely labelled cortical residual pairs are required")
        if not all(isinstance(p["parcel"], str) and p["parcel"].strip() for p in points):
            raise ValueError("Missing parcel label")
        x = np.array([_number(p["x"], "scatter x") for p in points])
        y = np.array([_number(p["y"], "scatter y") for p in points])
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            raise ValueError("Degenerate scatter residuals")
        if np.any((x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])):
            raise ValueError("Scatter limits would hide supplied points")
        effect = _number(row["saved_effect"], "saved scatter effect")
        if abs(effect - float(np.corrcoef(x, y)[0, 1])) > 1e-9:
            raise ValueError("Saved correlation disagrees with the residual pairs")
        scatters[condition] = row
    if {p["parcel"] for p in scatter[0]["points"]} != {p["parcel"] for p in scatter[1]["points"]}:
        raise ValueError("Scatter panels use different cortical parcels")
    if set(anatomy_assets) != set(ASSET_KEYS):
        raise ValueError("Eight caller-owned PC1/SERT hemisphere/view PNGs are required")
    scales, ticks = {}, {}
    for metric in ("PC1", "SERT"):
        scales[metric] = _limits(data[f"{metric}_color_limits"], metric + " colors")
        if tuple(asset_color_limits[metric]) != scales[metric]:
            raise ValueError("Asset and colorbar limits disagree")
        requested = data.get("color_ticks", {}).get(metric, list(scales[metric]))
        ticks[metric] = [_number(v, "colorbar tick") for v in requested]
        if not ticks[metric] or len(set(ticks[metric])) != len(ticks[metric]) or any(
            v < scales[metric][0] or v > scales[metric][1] for v in ticks[metric]):
            raise ValueError("Invalid colorbar ticks")
    assets, hashes, images = {}, {}, {}
    for key, value in anatomy_assets.items():
        path = Path(value).resolve(strict=True)
        if not path.is_file():
            raise ValueError("Asset must be an existing local PNG file")
        with Image.open(path) as im:
            if im.format != "PNG":
                raise ValueError("Expected PNG asset")
            # Keep native RGB/RGBA so interpolation matches the approved render.
            images[key] = np.array(im if im.mode in ("RGB", "RGBA") else im.convert("RGBA"))
        assets[key], hashes[key] = path, _sha(path)
    return data, order, indexed, sizes, scatters, scales, ticks, assets, hashes, images


def create_figure(plot_data, anatomy_assets, *, asset_color_limits,
                  assets_authorized=False, font_family="Arial"):
    """Create a Figure without writing files; no inferential analysis is run.

    ``assets_authorized`` is a caller attestation, not a licensing or visual-QC
    certificate. The returned receipt records this distinction explicitly.
    """
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    data, order, indexed, sizes, scatters, scales, ticks, assets, hashes, images = _prepare(
        plot_data, anatomy_assets, asset_color_limits, assets_authorized)
    cmaps = {
        "PC1": LinearSegmentedColormap.from_list("pc1", ["#C17842", "#F8F6F1", "#715296"]),
        "SERT": LinearSegmentedColormap.from_list("sert", ["#EFEDE9", "#D8C5E1", "#A48ABC", "#6B528E", "#392A58"])}
    style = {"font.family": font_family, "font.size": 9, "axes.labelsize": 8.5,
             "xtick.labelsize": 8, "ytick.labelsize": 8.5, "axes.linewidth": .7,
             "savefig.facecolor": "white", "pdf.fonttype": 42}
    with mpl.rc_context(style):
        fig = Figure(figsize=(11, 8), facecolor="white")
        FigureCanvasAgg(fig)
        titles = [("A", .025, .073, .965, "Cortical metabolic pattern"),
                  ("B", .525, .573, .965, "Normative SERT distribution"),
                  ("C", .025, .073, .447, "Regional spatial correspondence"),
                  ("D", .525, .573, .447, "Within-cohort SERT associations")]
        for letter, lx, tx, y, title in titles:
            fig.text(lx, y, letter, fontsize=16, fontweight="bold")
            fig.text(tx, y, title, fontsize=11.5, fontweight="bold")
        for metric, left in [("PC1", .055), ("SERT", .555)]:
            for j, hemi in enumerate(("lh", "rh")):
                fig.text(left+j*.215+.1025, .925, "L" if hemi == "lh" else "R",
                         ha="center", fontsize=8.5, color="#444444")
                for i, view in enumerate(("lateral", "medial")):
                    ax = fig.add_axes([left+j*.215, .735-i*.183, .205, .183])
                    ax.imshow(images[f"{metric}_{hemi}_{view}"])
                    ax.axis("off")
            cax = fig.add_axes([left+.080, .537, .260, .012])
            cb = fig.colorbar(ScalarMappable(norm=Normalize(*scales[metric]), cmap=cmaps[metric]),
                              cax=cax, orientation="horizontal")
            cb.outline.set_visible(False)
            cb.set_ticks(ticks[metric])
            cb.set_ticklabels([f"{value:g}".replace("-", "−") for value in ticks[metric]])
            cb.set_label("PC1 loading" if metric == "PC1" else "SERT reference map (z score)",
                         labelpad=3, fontsize=8.5)
            cb.ax.tick_params(length=2, labelsize=7.5)
        for k, (condition, title, color, x0) in enumerate([
                ("baseline", "Spatial adjustment", PURPLE, .075),
                ("DAT", "+ DAT adjustment", ORANGE, .308)]):
            row = scatters[condition]
            ax = fig.add_axes([x0, .126, .178, .257])
            ax.scatter([p["x"] for p in row["points"]], [p["y"] for p in row["points"]],
                       s=17, color=color, alpha=.85, edgecolors="white", linewidths=.35)
            ax.axhline(0, color="#e3e3e3", lw=.6, zorder=0)
            ax.axvline(0, color="#e3e3e3", lw=.6, zorder=0)
            ax.set(xlim=data.get("scatter_xlim", [-40, 40]), ylim=data.get("scatter_ylim", [-45, 45]),
                   xticks=data.get("scatter_xticks", [-30, 0, 30]),
                   yticks=data.get("scatter_yticks", [-40, -20, 0, 20, 40]))
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color("#888888")
            if k == 0:
                ax.set_ylabel("PC1 rank residual", labelpad=6)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("SERT rank residual", labelpad=5)
            ax.set_title(title, fontsize=9.5, pad=10)
            ax.text(.04, .96, f'ρ = {row["saved_effect"]:.3f}', transform=ax.transAxes,
                    va="top", fontsize=9)
        for i, (mapname, title, x0) in enumerate([
                (MAPS[0], "Primary SERT", .635), (MAPS[1], "Alternative SERT", .835)]):
            ax = fig.add_axes([x0, .126, .150, .257])
            ax.axvline(0, color="#a0a0a0", lw=.7, ls=(0, (3, 3)), zorder=0)
            for cohort, yy in zip(order, (2, 1, 0)):
                for condition, offset, color, marker in [
                        (CONDITIONS[0], .13, PURPLE, "o"), (CONDITIONS[1], -.13, ORANGE, "s")]:
                    row = indexed[(cohort, mapname, condition)]
                    ax.hlines(yy+offset, *row["ci95"], color=color, lw=1.45)
                    ax.plot(row["estimate"], yy+offset, marker, ms=4.6, color=color, mec="white", mew=.45)
            ax.set(xlim=(-1, 1), ylim=(-.5, 2.5), xticks=[-1, 0, 1], yticks=[2, 1, 0])
            ax.set_yticklabels([f"{c} (n = {sizes[c]})" for c in order] if i == 0 else [])
            ax.tick_params(axis="y", length=0, pad=7)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#888888")
            ax.set_xlabel("Partial correlation", labelpad=5)
            ax.set_title(title, fontsize=9.5, pad=10)
        handles = [Line2D([0], [0], color=PURPLE, marker="o", lw=1.4, ms=4.6),
                   Line2D([0], [0], color=ORANGE, marker="s", lw=1.4, ms=4.6)]
        fig.legend(handles, ["Spatial adjustment", "+ DAT adjustment"], ncol=2,
                   frameon=False, loc="center", bbox_to_anchor=(.789, .052),
                   fontsize=8.5, columnspacing=1.4)
    if any(_sha(assets[k]) != h for k, h in hashes.items()):
        raise RuntimeError("A supplied asset changed while rendering")
    receipt = {"layout": "final four-panel Figure 4", "new_analysis": False,
               "asset_sha256": hashes, "asset_authorization": "caller attestation",
               "anatomical_mapping_verified_by_renderer": False, "visual_review_status": "NOT_PERFORMED",
               "scatter_panels": 2, "points_per_scatter": 68, "interval_rows": 12,
               "data_sha256": hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    return fig, receipt


def render_figure4(plot_data, anatomy_assets, output_directory, *, asset_color_limits,
                   assets_authorized=False, dpi=600, preview_dpi=150, font_family="Arial"):
    """Write PNG, RGB TIFF and PDF into a NEW directory; never replace files."""
    from PIL import Image
    if any(type(x) is not int or not 1 <= x <= 1200 for x in (dpi, preview_dpi)):
        raise ValueError("dpi and preview_dpi must be integers in [1, 1200]")
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError("A new output directory is required")
    fig, receipt = create_figure(plot_data, anatomy_assets, asset_color_limits=asset_color_limits,
                                 assets_authorized=assets_authorized, font_family=font_family)
    output.mkdir(parents=True, exist_ok=False)
    status = output / "render_status.json"
    status.write_text(json.dumps({"status": "RUNNING", **receipt}, indent=2), encoding="utf-8")
    try:
        fig.savefig(output / "Figure4_preview.png", dpi=preview_dpi)
        fig.savefig(output / "Figure4.pdf")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        buffer.seek(0)
        with Image.open(buffer) as im:
            im.convert("RGB").save(output / "Figure4.tiff", format="TIFF",
                                   compression="tiff_adobe_deflate", dpi=(dpi, dpi))
        files = {}
        for name in ("Figure4_preview.png", "Figure4.pdf", "Figure4.tiff"):
            path = output / name
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
        receipt.update(status="RENDERED_PENDING_VISUAL_REVIEW", dpi=dpi,
                       preview_dpi=preview_dpi, files=files)
        status.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    except Exception as exc:
        status.write_text(json.dumps({"status": "FAILED_PRESERVED", "error_type": type(exc).__name__,
                                      **receipt}, indent=2), encoding="utf-8")
        raise
    finally:
        fig.clear()
    return receipt
