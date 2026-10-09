"""Portable manuscript figure renderers; no bundled study data or anatomy.

Rendering consumes caller-owned, already reviewed summaries. Except for the
explicit descriptive helpers, it does not fit, resample, rebin, or recompute q.
See ``docs/figures.md`` for schemas, source lineage, and reproduction limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from functools import wraps
from pathlib import Path
from typing import Any, Mapping
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

PALETTE = {"C": "#72777D", "CP": "#79529B", "CPT": "#C17C39"}
MODEL_NAMES = {"C": "Clinical", "CP": "Clinical + PET", "CPT": "Clinical + PET + T1w"}
LINESTYLES = {"C": "--", "CP": "-", "CPT": "-."}
MARKERS = {"C": "o", "CP": "s", "CPT": "^"}
REGION_NAMES = ["Thalamus", "Cingulate", "Insula / frontal operculum",
                "Limbic / medial temporal", "Striatopallidal", "Sensorimotor"]
PROCESSING_ORDER = ["original", "wholeBrain", "cerebellarCortex", "petprep", "same_seg_noPVC", "pvc54"]
PROCESSING_LABELS = ["Primary pipeline", "Whole-brain reference", "Cerebellar reference",
                     "PetPrep", "Matched no PVC", "PVC (5.4 mm)"]


def _styled(function):
    @wraps(function)
    def wrapped(*args, font_family="Arial", **kwargs):
        # A locally installed font is used. No third-party font is redistributed.
        with plt.rc_context({"font.family": [font_family, "DejaVu Sans"], "font.size": 10,
                             "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                             "text.color": "#30363A", "axes.labelcolor": "#30363A",
                             "xtick.color": "#555B60", "ytick.color": "#555B60",
                             "axes.edgecolor": "#92969B", "axes.linewidth": .7,
                             "svg.fonttype": "none", "figure.facecolor": "white",
                             "savefig.facecolor": "white"}):
            return function(*args, **kwargs)
    return wrapped


def format_estimate(value):
    """Original three-decimal half-up display rounding, not a new estimate."""
    return str(Decimal(str(round(float(value), 12))).quantize(Decimal(".001"), rounding=ROUND_HALF_UP))


def _finite(values, name="values"):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _interval(value, bounds):
    values = _finite([value, *bounds], "estimate and interval")
    if values.shape != (3,) or not values[1] <= values[0] <= values[2]:
        raise ValueError("interval must contain its estimate and have two ordered bounds")
    return [[float(values[0] - values[1])], [float(values[2] - values[0])]]


def _in_axis(bounds, limits):
    bounds = _finite(bounds)
    _finite(limits, "axis limits")
    if min(bounds) < limits[0] or max(bounds) > limits[1]:
        raise ValueError("a mark would be clipped by the manuscript axis; supply explicit wider limits")


def _clean_axis(ax, grid_axis="both", hide_left=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#E6E7E9", linewidth=.6, zorder=0)
    ax.tick_params(length=3, width=.6)
    ax.set_axisbelow(True)
    if hide_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def _header(fig, letter, title, x, y, size=11.5):
    fig.text(x, y, letter, fontsize=14, weight="bold", va="baseline")
    fig.text(x + .030, y, title, fontsize=size, weight="bold", va="baseline")


def _synthetic(fig, synthetic):
    if synthetic:
        fig.text(.5, .008, "SYNTHETIC DEMONSTRATION — NOT STUDY RESULTS", ha="center",
                 va="bottom", fontsize=8, color="#555B60")
    return fig


def wilson95(k: int, n: int):
    """Descriptive pointwise Wilson interval; empty bins return None."""
    if int(k) != k or int(n) != n or not 0 <= k <= n:
        raise ValueError("require integer 0 <= k <= n")
    if not n:
        return None
    z = 1.959963984540054
    phat = k / n
    den = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return [max(0., center - half), min(1., center + half)]


def fixed_bin_calibration(y, probabilities):
    """Original five equal-width bins, left-closed, last bin right-closed.

    This is an explicit descriptive transformation of caller-supplied predictions,
    not model fitting. Empty bins remain null and break the plotted line.
    """
    y, p = _finite(y, "outcome"), _finite(probabilities, "probabilities")
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("aligned nonempty binary outcomes and probabilities required")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    edges = np.linspace(0, 1, 6)
    membership = np.digitize(p, edges[1:-1], right=False)
    bins = []
    for b in range(5):
        mask = membership == b
        n, k = int(mask.sum()), int(y[mask].sum())
        bins.append({"bin": b + 1, "lower": float(edges[b]), "upper": float(edges[b + 1]),
                     "interval_rule": "[lower, upper)" if b < 4 else "[lower, upper]",
                     "n": n, "responders": k, "mean_probability": float(p[mask].mean()) if n else None,
                     "observed_fraction": k / n if n else None, "wilson95_descriptive": wilson95(k, n)})
    return {"model": "CP", "fixed_bin_edges": edges.tolist(), "bins": bins}


@_styled
def figure3_prediction(data, *, dpi=150, synthetic=False,
                       brier_limits=(0, .305), ablation_limits=(-.075, .52)):
    """Figure 3: saved ROC, five-bin Wilson calibration, Brier, paired ablation."""
    models = data["models"]
    bins = data["calibration"]["bins"]
    if len(bins) != 5 or not np.allclose(data["calibration"]["fixed_bin_edges"], np.linspace(0, 1, 6)):
        raise ValueError("the figure requires five fixed equal-width bins")
    if len(data["ablation"]) != 3:
        raise ValueError("three ordered feature-removal rows required")
    fig = plt.figure(figsize=(9.2, 8.), dpi=dpi)
    ax_a = fig.add_axes([.09, .615, .37, .31])
    ax_b = fig.add_axes([.60, .615, .36, .31])
    ax_c = fig.add_axes([.235, .115, .225, .265])
    ax_d = fig.add_axes([.740, .115, .220, .265])
    for letter, title, x, y in [("A", "Discrimination", .055, .964),
                               ("B", "Calibration: Clinical + PET", .555, .964),
                               ("C", "Prediction error", .055, .416),
                               ("D", "Feature-removal ablation", .555, .416)]:
        _header(fig, letter, title, x, y)
    ax_a.plot([0, 1], [0, 1], color="#B9BDC1", linestyle=":", linewidth=1., zorder=1)
    for model in MODEL_NAMES:
        item = models[model]
        fpr, tpr = _finite(item["roc_fpr"]), _finite(item["roc_tpr"])
        if fpr.ndim != 1 or fpr.shape != tpr.shape or len(fpr) < 2:
            raise ValueError("paired saved ROC coordinates required")
        if np.any(np.diff(fpr) < 0) or np.any(np.diff(tpr) < 0):
            raise ValueError("ROC coordinates must be nondecreasing")
        _in_axis([*fpr, *tpr], (0, 1))
        ax_a.plot(fpr, tpr, color=PALETTE[model], linestyle=LINESTYLES[model],
                  linewidth=1.8 if model == "CP" else 1.55, zorder=3 if model == "CP" else 2)
    ax_a.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False-positive rate", ylabel="True-positive rate")
    for ax in (ax_a, ax_b):
        ax.set_xticks([0, .2, .4, .6, .8, 1]); ax.set_yticks([0, .2, .4, .6, .8, 1])
        _clean_axis(ax)
    for model, yy in zip(MODEL_NAMES, [.509, .481, .453]):
        item = models[model]
        _interval(item["auc"], item["auc_conditional95"])
        fig.add_artist(Line2D([.09, .116], [yy + .005, yy + .005], transform=fig.transFigure,
                             color=PALETTE[model], linestyle=LINESTYLES[model], linewidth=1.8))
        fig.text(.125, yy, MODEL_NAMES[model], fontsize=9.5)
        lo, hi = item["auc_conditional95"]
        fig.text(.46, yy, f"{format_estimate(item['auc'])} ({format_estimate(lo)}–{format_estimate(hi)})",
                 fontsize=9., ha="right")
    xs, ys, lower, upper = [], [], [], []
    for i, b in enumerate(bins):
        if b["bin"] != i + 1 or b["n"] < 0 or int(b["n"]) != b["n"]:
            raise ValueError("calibration bins must be ordered with nonnegative integer counts")
        if not b["n"]:
            if any(b[k] is not None for k in ("mean_probability", "observed_fraction", "wilson95_descriptive")):
                raise ValueError("empty calibration bins must have null summaries")
            xs.append(np.nan); ys.append(np.nan); lower.append(np.nan); upper.append(np.nan)
            continue
        err = _interval(b["observed_fraction"], b["wilson95_descriptive"])
        _in_axis([b["mean_probability"], *b["wilson95_descriptive"]], (0, 1))
        xs.append(b["mean_probability"]); ys.append(b["observed_fraction"])
        lower.append(err[0][0]); upper.append(err[1][0])
    ax_b.plot([0, 1], [0, 1], color="#B9BDC1", linestyle=":", linewidth=1., zorder=1)
    ax_b.errorbar(xs, ys, yerr=[lower, upper], fmt="s-", color=PALETTE["CP"], markersize=5.2,
                  linewidth=1.35, elinewidth=1.05, capsize=3, capthick=1., zorder=3)
    for b in bins:
        if b["n"]:
            near_zero = b["bin"] in (1, 2)
            ax_b.annotate(f"n={b['n']}", (b["mean_probability"], b["wilson95_descriptive"][0]),
                          xytext=(0, 7 if near_zero else -12), textcoords="offset points", ha="center",
                          va="bottom" if near_zero else "top", fontsize=8.5, color=PALETTE["CP"],
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": .3} if near_zero else None)
    ax_b.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Mean predicted response probability",
             ylabel="Observed response proportion")
    cal = models["CP"]["frozen_calibration"]
    _finite([cal["intercept_offset_slope_fixed_1"], cal["slope"]])
    fig.text(.60, .538, f"Calibration intercept: {format_estimate(cal['intercept_offset_slope_fixed_1']).replace('-', '−')}"
             f"     Slope: {format_estimate(cal['slope'])}", fontsize=9.)
    for model, yy in zip(MODEL_NAMES, [2, 1, 0]):
        item = models[model]; x, ci = item["brier"], item["brier_conditional95"]
        _in_axis(ci, brier_limits)
        ax_c.errorbar(x, yy, xerr=_interval(x, ci), fmt=MARKERS[model], color=PALETTE[model],
                      markersize=6., elinewidth=1.5, capsize=3.5, capthick=1.2, zorder=3)
        ax_c.annotate(format_estimate(x), (x, yy), xytext=(0, 11), textcoords="offset points",
                       ha="center", va="bottom", fontsize=9.5, color=PALETTE[model])
    ax_c.set_yticks([2, 1, 0], list(MODEL_NAMES.values()))
    ax_c.set(xlim=brier_limits, ylim=(-.65, 2.65), xlabel="Brier score")
    ax_c.set_xticks([0, .1, .2, .3])
    labels = ["Without thalamus", "Without striatum/pallidum", "Without both"]
    for row, yy in zip(data["ablation"], [2, 1, 0]):
        x, ci = row["delta_auc_CP_minus_reduced"], row["conditional95"]
        _in_axis(ci, ablation_limits)
        ax_d.errorbar(x, yy, xerr=_interval(x, ci), fmt="s", color=PALETTE["CP"], markersize=6.,
                      elinewidth=1.5, capsize=3.5, capthick=1.2, zorder=3)
        label = ("+" if x >= 0 else "") + format_estimate(x)
        ax_d.annotate(label, (x, yy), xytext=(0, 11), textcoords="offset points", ha="center",
                       va="bottom", fontsize=9.5, color=PALETTE["CP"])
    ax_d.axvline(0, color="#858B90", linewidth=.95, linestyle="--", zorder=2)
    ax_d.set_yticks([2, 1, 0], labels)
    ax_d.set(xlim=ablation_limits, ylim=(-.65, 2.65), xlabel="ΔAUC: full − reduced model")
    ax_d.set_xticks([0, .2, .4])
    for ax in (ax_c, ax_d):
        ax.tick_params(axis="y", length=0, labelsize=9.5, pad=9)
        _clean_axis(ax, "x", hide_left=True)
    return _synthetic(fig, synthetic)


@_styled
def figure_s1_validation(data, *, dpi=150, synthetic=False,
                         auc_limits=(.40, 1.04), brier_limits=(.10, .405), offset_limits=(-1.6, 1.6)):
    """Revised S1 layout; caller supplies two device directions and saved ranks."""
    rows, ranked = data["cross_scanner"], data["ranked_saved_CP_percentiles"]
    if len(rows) != 4 or {(r["direction"], r["model"]) for r in rows} != {(d, m) for d in (0, 1) for m in ("C", "CP")}:
        raise ValueError("one C and CP result for each of two scanner directions required")
    if not ranked or [r["rank"] for r in ranked] != list(range(1, len(ranked) + 1)):
        raise ValueError("saved ranks must be contiguous from one")
    if np.any(np.diff([r["p50"] for r in ranked]) < 0):
        raise ValueError("saved rows must already be ordered by median prediction")
    fig = plt.figure(figsize=(9.2, 8.), dpi=dpi)
    axes = [fig.add_axes(b) for b in [[.26, .615, .205, .285], [.51, .615, .205, .285], [.765, .615, .205, .285]]]
    ax_d = fig.add_axes([.10, .115, .87, .295])
    # The final S1 revision moves only the A heading left by .020.
    for letter, title, x in [("A", "Discrimination", .24), ("B", "Prediction error", .51), ("C", "Calibration offset", .765)]:
        fig.text(x - .024, .943, letter, weight="bold", fontsize=13)
        fig.text(x, .943, title, weight="bold", fontsize=10.5)
    for i in (0, 1):
        direction = next(r for r in rows if r["direction"] == i and r["model"] == "C")
        fig.text(.022, [.829, .686][i], f"{direction['training_device']}\n→ {direction['test_device']}\n"
                 f"Train {direction['n_train']} / Test {direction['n_test']}", fontsize=9.5, va="center", linespacing=1.5)
    for row in rows:
        yy = (1 if row["direction"] == 0 else 0) + (.14 if row["model"] == "C" else -.14)
        model = row["model"]; color, marker = PALETTE[model], MARKERS[model]
        for ax, metric, limits in zip(axes[:2], ["auc", "brier"], [auc_limits, brier_limits]):
            x, ci = row[metric], row[metric + "_conditional95"]
            _in_axis(ci, limits)
            ax.errorbar(x, yy, xerr=_interval(x, ci), fmt=marker, color=color, markersize=5.8,
                        elinewidth=1.35, capsize=3, capthick=1., zorder=3)
            ax.annotate(format_estimate(x), (x, yy), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=color)
        x = float(_finite(row["calibration_offset_intercept"]))
        _in_axis([x], offset_limits)
        axes[2].plot(x, yy, marker=marker, color=color, markersize=5.8, linestyle="none", zorder=3)
        axes[2].annotate(format_estimate(x).replace("-", "−"), (x, yy), xytext=(0, 8),
                         textcoords="offset points", ha="center", va="bottom", fontsize=9, color=color)
    for ax in axes:
        ax.set_ylim(-.5, 1.5); ax.set_yticks([]); _clean_axis(ax, "x", hide_left=True)
        ax.axhline(.5, color="#ECEDEF", linewidth=.6)
    axes[0].set(xlim=auc_limits, xlabel="AUC"); axes[0].set_xticks([.5, .75, 1.], ["0.50", "0.75", "1.00"])
    axes[0].axvline(.5, color="#A4A9AE", linewidth=.85, linestyle=":", zorder=1)
    axes[1].set(xlim=brier_limits, xlabel="Brier score"); axes[1].set_xticks([.1, .2, .3, .4])
    axes[2].set(xlim=offset_limits, xlabel="Offset intercept"); axes[2].set_xticks([-1, 0, 1])
    axes[2].axvline(0, color="#A4A9AE", linewidth=.85, linestyle=":", zorder=1)
    handles = [Line2D([], [], linestyle="none", marker=MARKERS[m], color=PALETTE[m], markersize=6,
                     label=MODEL_NAMES[m]) for m in ("C", "CP")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.605, .532), frameon=False, ncol=2,
               fontsize=10, handletextpad=.5, columnspacing=2.)
    fig.text(.045, .463, "D", fontsize=13, weight="bold")
    fig.text(.075, .463, "Refitting variability: Clinical + PET", fontsize=11.5, weight="bold")
    for point in ranked:
        _interval(point["p50"], [point["p2p5"], point["p97p5"]])
        _in_axis([point["p2p5"], point["p97p5"]], (0, 1))
        ax_d.vlines(point["rank"], point["p2p5"], point["p97p5"], colors=PALETTE["CP"], alpha=.35, linewidth=1.4)
    ax_d.plot([p["rank"] for p in ranked], [p["p50"] for p in ranked], linestyle="none", marker="s",
               markersize=3.5, color=PALETTE["CP"])
    n = len(ranked)
    ax_d.set(xlim=(0, n + 1), ylim=(-.015, 1.015), xlabel="Patients ordered by median prediction",
             ylabel="Predicted response probability")
    ax_d.set_xticks(sorted(set([1, *range(10, n, 10), n])))
    ax_d.set_yticks([0, .2, .4, .6, .8, 1]); _clean_axis(ax_d, "y")
    fig.legend(handles=[Line2D([], [], color=PALETTE["CP"], marker="s", linestyle="none", markersize=4,
                               label="Median"), Line2D([], [], color=PALETTE["CP"], alpha=.45, linewidth=1.5,
                                                       label="2.5th–97.5th percentile range")],
               loc="lower left", bbox_to_anchor=(.095, .422), frameon=False, fontsize=8.5, ncol=2,
               borderaxespad=0, columnspacing=1.8)
    return _synthetic(fig, synthetic)


@_styled
def figure_s2_processing(data, *, dpi=150, synthetic=False,
                         effect_limits=(-2.05, .40), auc_limits=(.40, 1.), brier_limits=(.075, .30)):
    """S2 uses supplied full-family q values; hidden hypotheses are not dropped."""
    order = data["order"]
    labels = data["labels"]
    regions = data["regions"]
    if len(order) != 6 or len(set(order)) != 6 or len(labels) != 6 or len(regions) != 2:
        raise ValueError("six ordered processing variants and two regions required")
    if data["q_family_size"] != 96 or set(data["not_displayed_but_retained_in_family"]) != {"pvc4", "pvc6"}:
        raise ValueError("manuscript S2 requires the original 96-test family, retaining PVC 4/6 mm tests")
    fig = plt.figure(figsize=(9.2, 8.5), dpi=dpi)
    axes = [fig.add_axes(r) for r in [[.255, .590, .310, .335], [.665, .590, .310, .335],
                                     [.255, .095, .310, .335], [.665, .095, .310, .335]]]
    ys = [5.5, 4.5, 3.5, 2.5, .8, -.2]
    for i, ax in enumerate(axes):
        _clean_axis(ax, "x", hide_left=True); ax.tick_params(axis="y", length=0, pad=9, labelsize=9.5)
        ax.set_ylim(-.85, 6.15); ax.set_yticks(ys, labels if i in (0, 2) else [""] * 6)
        ax.axhline(1.65, color="#DADCE0", lw=.7)
    for letter, title, x, y in [("A", "Thalamus", .055, .963), ("B", "Striatum / pallidum", .605, .963),
                               ("C", "Discrimination", .055, .475), ("D", "Prediction error", .605, .475)]:
        _header(fig, letter, title, x, y)
    for ax, region, color, marker in zip(axes[:2], regions, [PALETTE["CP"], PALETTE["CPT"]], ["o", "s"]):
        rows = data["association_rows"][region]
        if [r["variant"] for r in rows] != order:
            raise ValueError("association rows must match the declared order")
        for yy, row in zip(ys, rows):
            _in_axis([row["q_family"]], (0, 1))
            x, ci = row["standardized_difference"], row["standardized_CI95"]
            _in_axis(ci, effect_limits)
            ax.errorbar(x, yy, xerr=_interval(x, ci), fmt=marker, color=color, markersize=5.4,
                         elinewidth=1.4, capsize=3, capthick=1)
        ax.axvline(0, color="#92969B", ls="--", lw=.9)
        ax.set_xlim(effect_limits); ax.set_xticks([-2, -1.5, -1, -.5, 0])
        ax.set_xlabel("Adjusted difference / pooled SD", labelpad=7)
    for ax, metric, limits, ticks in [(axes[2], "auc", auc_limits, [.4, .6, .8, 1.]),
                                      (axes[3], "brier", brier_limits, [.1, .15, .2, .25, .3])]:
        for yy, variant in zip(ys, order):
            row = data["models"][variant]; x, ci = row[metric], row[metric + "_conditional95"]
            _in_axis(ci, limits)
            color = PALETTE["CPT"] if variant == "pvc54" else PALETTE["CP"]
            ax.errorbar(x, yy, xerr=_interval(x, ci), fmt="s" if variant == "pvc54" else "o", color=color,
                        markersize=5.4, elinewidth=1.4, capsize=3, capthick=1)
            ax.annotate(format_estimate(x), (x, yy), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.8, color=color)
        ax.set_xlim(limits); ax.set_xticks(ticks)
        ax.set_xlabel("AUC" if metric == "auc" else "Brier score", labelpad=7)
    axes[2].axvline(.5, color="#92969B", ls="--", lw=.9)
    return _synthetic(fig, synthetic)


@dataclass(frozen=True)
class AnatomyAsset:
    """Caller-owned image and post-crop axes-fraction annotation endpoints.

    ``source`` and ``license`` are declarations, not a rights determination.
    Images must already use the appropriate reference geometry and effect colors.
    No image generation, atlas download, anatomy inference, or recoloring occurs.
    """
    image: Any
    source: str
    license: str
    annotations: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    orientation: tuple[str, str] = ("", "")


def _asset_pixels(asset):
    if not asset.source.strip() or not asset.license.strip():
        raise ValueError("asset source and license declarations are required")
    if isinstance(asset.image, (str, Path)):
        with Image.open(asset.image) as image:
            pixels = np.asarray(image.convert("RGB")).copy()
    elif isinstance(asset.image, Image.Image):
        pixels = np.asarray(asset.image.convert("RGB")).copy()
    else:
        pixels = np.asarray(asset.image)
    if pixels.ndim not in (2, 3) or min(pixels.shape[:2]) < 2:
        raise ValueError("an image array or readable image path is required")
    return pixels


def _gaussian_density(values, grid, bandwidth):
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-.5 * z * z).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


@_styled
def figure2_regional(data, anatomy_assets, *, dpi=150, synthetic=False, effect_limits=(-2.05, 1.1)):
    """Final-v7 assembly: licensed anatomy assets + exact forest/raincloud logic."""
    primary, extended = data["primary"], data["extended"]
    raw, y = _finite(data["raw_relative_values"]), _finite(data["y"])
    if len(primary) != 6 or len(extended) != 6 or raw.ndim != 2 or raw.shape != (len(y), 6):
        raise ValueError("six aligned regions and an N by 6 uptake matrix required")
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or min(np.sum(y == 0), np.sum(y == 1)) < 2:
        raise ValueError("both outcome groups require at least two rows")
    if data["primary_family_size"] != 6 or data["extended_family_size"] != 18:
        raise ValueError("retain the primary six-test and extended 18-test q families")
    required_assets = {"left_lateral", "left_medial", "axial", "coronal", "deep"}
    if set(anatomy_assets) != required_assets:
        raise ValueError("five licensed pre-rendered anatomy assets are required")
    limits = tuple(data["shared_distribution_xlim"])
    _in_axis(raw[:, [0, 4]].ravel(), limits)
    fig = plt.figure(figsize=(6.9, 9.), dpi=dpi)
    W, H = 6.9, 9.
    def axis(rect):
        x, yy, w, h = rect
        return fig.add_axes([x / W, yy / H, w / W, h / H])
    def text(x, yy, s, **kw):
        return fig.text(x / W, yy / H, s, **kw)
    def header(x, yy, letter, title):
        text(x, yy, letter, fontsize=12, fontweight="bold", va="top")
        text(x + .23, yy - .01, title, fontsize=10, fontweight="bold", va="top")
    purple, orange, grey, ink = "#655091", "#B86C1E", "#73777B", "#292C30"
    cmap = LinearSegmentedColormap.from_list("purple_orange", matplotlib.colormaps["PuOr_r"](np.linspace(.12, .88, 256)), N=256)
    norm = Normalize(-1.3, 1.3)
    header(.12, 8.91, "A", "Spatial distribution of regional effects")
    specs = [("left_lateral", [.25, 7.43, 1.97, 1.06], "Lateral cortex (L)", {"Sensorimotor": (.03, 1.01)}),
             ("left_medial", [2.49, 7.43, 1.97, 1.06], "Medial cortex (L)", {"Cingulate": (.97, 1.01)}),
             ("axial", [.20, 5.94, 2.06, 1.14], "", {}),
             ("coronal", [2.45, 5.94, 2.06, 1.14], "", {}),
             ("deep", [4.76, 7.29, 1.94, 1.22], "", {"Thalamus": (.01, 1.03), "Caudate": (.99, 1.03),
               "Putamen": (-.04, .60), "Pallidum": (.99, .22), "Hippocampus": (.00, -.10), "Accumbens": (.99, -.10)})]
    for name, rect, title, label_positions in specs:
        asset = anatomy_assets[name]; ax = axis(rect); ax.imshow(_asset_pixels(asset)); ax.axis("off")
        if title:
            text(rect[0] + rect[2] / 2, 8.67, title, ha="center", fontsize=8, color="#666B70")
        for xx, label in zip((.02, .98) if name in ("axial", "coronal") else (-.04, 1.04), asset.orientation):
            if label:
                ax.text(xx, .50, label, transform=ax.transAxes, ha="center", va="center", fontsize=8,
                        color="white" if name in ("axial", "coronal") else "#666B70")
        for label, pos in label_positions.items():
            if label not in asset.annotations:
                continue
            point = _finite(asset.annotations[label], "annotation endpoint")
            if point.shape != (2,) or np.any((point < 0) | (point > 1)):
                raise ValueError("annotation endpoints must be post-crop axes fractions")
            ax.annotate(label, xy=point, xycoords="axes fraction", xytext=pos, textcoords="axes fraction",
                        ha="right" if label == "Putamen" else ("left" if pos[0] < .5 else "right"),
                        va="center", fontsize=8, color=ink,
                        arrowprops={"arrowstyle": "-", "lw": .55, "color": "#6F757B", "shrinkA": 2, "shrinkB": 2})
    text(4.85, 6.98, "Regional composites", fontweight="bold", fontsize=8)
    names = ["Thalamus", "Cingulate", "Insula / operculum", "Limbic / medial temporal", "Striatopallidal", "Sensorimotor"]
    for i, (name, row) in enumerate(zip(names, primary)):
        _in_axis([row["q_family"]], (0, 1))
        xx, yy = 4.85, 6.74 - i * .17
        text(xx + .17, yy, name + (" *" if row["q_family"] < .05 else ""), fontsize=8, va="center")
        fig.add_artist(Rectangle((xx / W, (yy - .045) / H), .095 / W, .09 / H, transform=fig.transFigure,
                                  facecolor=cmap(norm(row["difference_pooled_sd"])), edgecolor="#666B70", lw=.5))
    text(4.85, 5.65, "* FDR-adjusted P < 0.05", fontsize=8, color="#666B70")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=axis([.78, 5.60, 2.9, .08]),
                      orientation="horizontal", ticks=[-1.2, -.6, 0, .6, 1.2])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, length=2, width=.6, pad=2)
    text(.78, 5.25, "Primary-model difference (R − NR), pooled SD units", fontsize=8)
    # Forest and q table: v2 layout translated +2.30 inches in final v7.
    header(.12, 4.96, "B", "Associations before and after further clinical adjustment")
    handles = [Line2D([0], [0], color=purple, marker="o", lw=1, ms=4, label="Primary model"),
               Line2D([0], [0], color=grey, marker="s", mfc="white", lw=1, ms=4, label="Extended model")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(3.47 / W, 4.73 / H), frameon=False,
               ncol=2, handlelength=1.4, columnspacing=1.2, handletextpad=.55, fontsize=8, borderaxespad=0)
    forest = axis([2.15, 2.86, 2.64, 1.51]); qa = axis([5.04, 2.86, 1.66, 1.51])
    forest.axvline(0, color="#A4A7AB", ls=(0, (3, 3)), lw=.7, zorder=0)
    forest.set(xlim=effect_limits, ylim=(5.48, -.48))
    forest.set_yticks(range(6), REGION_NAMES); forest.tick_params(axis="y", length=0, pad=8, labelsize=8)
    forest.set_xticks([-2, -1, 0, 1]); forest.tick_params(axis="x", labelsize=8, length=3, pad=3, width=.6)
    forest.spines[["top", "right", "left"]].set_visible(False)
    forest.spines["bottom"].set_color("#858A8E"); forest.spines["bottom"].set_linewidth(.6)
    forest.set_xlabel("Adjusted difference (R − NR), pooled SD units", fontsize=8.3, labelpad=6)
    qa.set(xlim=(0, 1), ylim=(5.48, -.48)); qa.axis("off")
    positions = [.18, .79]
    text(5.04 + 1.66 * sum(positions) / 2, 4.70, "FDR-adjusted P", ha="center", va="center", fontsize=8.3, fontweight="bold")
    for xx, label, color in zip(positions, ["Primary model", "Extended model"], [purple, grey]):
        text(5.04 + 1.66 * xx, 4.485, label, ha="center", va="center", fontsize=8, color=color)
    for i, (p, e) in enumerate(zip(primary, extended)):
        for row, dy, color, marker, xx, is_primary in [(p, -.16, purple, "o", .18, True), (e, .16, grey, "s", .79, False)]:
            value = row["difference_pooled_sd"] if is_primary else row["standardized_difference"]
            ci = [row["ci_low_pooled_sd"], row["ci_high_pooled_sd"]] if is_primary else row["standardized_CI95"]
            _in_axis(ci, effect_limits); _in_axis([row["q_family"]], (0, 1))
            forest.errorbar(value, i + dy, xerr=_interval(value, ci), fmt=marker, color=color,
                            mfc=color if is_primary else "white", mec=color, ms=4.1, mew=.8, capsize=2, lw=1., zorder=3)
            qa.text(xx, i, f"{row['q_family']:.4f}", ha="center", va="center", fontsize=8, color=ink,
                    fontweight="bold" if row["q_family"] < .05 else "normal")
    # Four distributions retain a single absolute bandwidth and shared density scale.
    pooled = raw[:, [0, 4]].ravel()
    bandwidth = float(1.06 * np.std(pooled, ddof=1) * len(pooled) ** (-1 / 5))
    if bandwidth <= 0:
        raise ValueError("pooled focal values need positive variation for KDE")
    grid = np.linspace(*limits, 500)
    densities = {(j, g): _gaussian_density(raw[y == g, j], grid, bandwidth) for j in (0, 4) for g in (0, 1)}
    peak = max(v.max() for v in densities.values())
    for letter, xx, j, title in [("C", .12, 0, "Thalamus"), ("D", 3.63, 4, "Striatopallidal")]:
        header(xx, 2.26, letter, title); ax = axis([xx + .87, .85, 2.17, 1.05])
        for g, pos, color, marker in [(0, 1., orange, "o"), (1, 0., purple, "^")]:
            vals = raw[y == g, j]; curve = densities[(j, g)] / peak * .30
            ax.fill_between(grid, pos + .10, pos + .10 + curve, color=color, alpha=.25, lw=0)
            ax.plot(grid, pos + .10 + curve, color=color, lw=.75)
            ax.boxplot([vals], positions=[pos + .045], widths=.085, orientation="horizontal", whis=(0, 100),
                       showfliers=False, patch_artist=True, manage_ticks=False,
                       boxprops={"facecolor": "white", "edgecolor": color, "lw": .8},
                       medianprops={"color": ink, "lw": 1.05}, whiskerprops={"color": color, "lw": .7},
                       capprops={"color": color, "lw": .7})
            jitter = np.random.default_rng(20260928 + j * 10 + g).uniform(-.21, -.065, len(vals))
            ax.scatter(vals, pos + jitter, s=12, marker=marker, facecolor="white" if g == 0 else color,
                       edgecolor=color, lw=.65, alpha=.9, zorder=3)
        ax.set(ylim=(-.38, 1.49), xlim=limits)
        ax.set_yticks([1, 0], [f"Nonresponders\n(n = {int(sum(y == 0))})", f"Responders\n(n = {int(sum(y == 1))})"])
        ax.tick_params(axis="y", length=0, pad=7, labelsize=8); ax.tick_params(axis="x", length=3, width=.65, pad=3, labelsize=8)
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#83878B"); ax.spines["bottom"].set_linewidth(.6)
        ax.set_xticks([.7, .9, 1.1, 1.3]); ax.set_xlabel("Relative FDG uptake ratio", fontsize=8.3, labelpad=5)
    return _synthetic(fig, synthetic)


@_styled
def figure1_workflow(counts, template_stacks, *, period, exclusions, feature_counts=(5, 6, 6),
                     outer_folds=5, repeats=20, inner_folds=3, dpi=150, synthetic=False):
    """Final Figure 1 geometry with caller counts and licensed image stacks.

    ``exclusions`` contains three (label, count) pairs for outside-primary cases.
    ``template_stacks`` has pet/t1/overlay, each three already-rendered assets.
    Images are caller-owned reference artwork, never inferred patient scans.
    """
    required = {"vns_implantation", "imaging_candidates", "primary", "responders", "nonresponders"}
    if not required.issubset(counts) or any(int(counts[k]) != counts[k] or counts[k] < 0 for k in required):
        raise ValueError("five nonnegative integer cohort counts required")
    if not counts["vns_implantation"] >= counts["imaging_candidates"] >= counts["primary"]:
        raise ValueError("cohort flow counts must be nested")
    if counts["primary"] != counts["responders"] + counts["nonresponders"]:
        raise ValueError("response counts must sum to primary cohort")
    if len(exclusions) != 3 or any(int(n) != n or n < 0 for _, n in exclusions):
        raise ValueError("three nonnegative integer exclusion counts required")
    if sum(n for _, n in exclusions) != counts["imaging_candidates"] - counts["primary"]:
        raise ValueError("exclusions must reconcile with outside-primary count")
    if set(template_stacks) != {"pet", "t1", "overlay"} or any(len(v) != 3 for v in template_stacks.values()):
        raise ValueError("three licensed image assets per pet/t1/overlay stack required")
    if len(feature_counts) != 3 or any(int(n) != n or n <= 0 for n in [*feature_counts, outer_folds, repeats, inner_folds]):
        raise ValueError("positive integer feature and validation counts required")
    W, H = 12.8, 9.8
    ink, muted, linecolor, purple = "#29262F", "#6E6777", "#C4BFCB", "#79529B"
    orange, grey, blue, blue_pale = "#C17C39", "#9A96A2", "#526D82", "#EBF0F4"
    fig = plt.figure(figsize=(W, H), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, W), ylim=(0, H)); ax.axis("off")
    offset = -.7
    def text(x, yy, s, size=10, bold=False, color=ink, ha="left", **kw):
        return ax.text(x, yy + offset, s, fontsize=size, fontweight="bold" if bold else "normal",
                       color=color, ha=ha, va="center", **kw)
    def line(xs, ys, color=linecolor, lw=.85, **kw):
        ax.plot(xs, np.asarray(ys) + offset, color=color, lw=lw, solid_capstyle="round", **kw)
    def arrow(x1, y1, x2, y2, color=muted, lw=1.2, **kw):
        ax.add_patch(FancyArrowPatch((x1, y1 + offset), (x2, y2 + offset), arrowstyle="-|>",
                     mutation_scale=13, color=color, lw=lw, shrinkA=0, shrinkB=0, **kw))
    def box(x, yy, w, h, fill="white", edge=linecolor, r=.045, lw=.9):
        ax.add_patch(FancyBboxPatch((x, yy + offset), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                   fc=fill, ec=edge, lw=lw))
    def section(letter, title, yy):
        text(.26, yy, letter, 17, True); text(.61, yy, title, 13.5, True)
    def stack(kind, left, bottom, width=2.08, height=2.23):
        for order, (dx, dy) in enumerate([(0, .25), (.13, .125), (.26, 0)]):
            imax = fig.add_axes([(left + dx) / W, (bottom + dy + offset) / H, width / W, height / H], zorder=2 + order)
            pixels = _asset_pixels(template_stacks[kind][order])
            imax.imshow(pixels, interpolation="antialiased"); imax.set_aspect("equal"); imax.axis("off")
            imax.add_patch(Rectangle((-.5, -.5), pixels.shape[1], pixels.shape[0], fill=False, ec="white", lw=.8))
    def tiles(x, yy, color, number):
        for i in range(number):
            ax.add_patch(Rectangle((x + i * 2.52 / number, yy + offset), 1.92 / number, .25, fc=color, ec="white", lw=.5))
    def person(x, yy, scale=1, color="#424952"):
        ax.add_patch(Circle((x, yy + .84 * scale + offset), .082 * scale, fc=color, ec="none"))
        box(x - .10 * scale, yy + .39 * scale, .20 * scale, .33 * scale, fill=color, edge=color, r=.055 * scale, lw=0)
        for sign in (-1, 1):
            line([x + sign * .115 * scale, x + sign * .175 * scale], [yy + .66 * scale, yy + .40 * scale], color=color, lw=4.5 * scale)
            line([x + sign * .06 * scale, x + sign * .085 * scale], [yy + .41 * scale, yy + .08 * scale], color=color, lw=5. * scale)
    section("A", "Patient selection", 10.20)
    for x, key, label in [(1.23, "vns_implantation", "VNS implantation"), (4.12, "imaging_candidates", "Imaging candidates"),
                          (7.10, "primary", "Primary analysis")]:
        person(x - .42, 8.99, .77, "#7798B2"); text(x + .33, 9.36, str(counts[key]), 27, color="#36556D", ha="center")
        text(x, 9.88, label, 11.4, color="#334656", ha="center")
    text(1.23, 8.86, period, 9.8, color="#334656", ha="center")
    arrow(2.11, 9.37, 3.18, 9.37); arrow(5.04, 9.37, 6.19, 9.37)
    arrow(2.62, 9.37, 2.62, 8.94, color=linecolor, lw=.9)
    text(2.49, 8.68, f"Not included (n = {counts['vns_implantation'] - counts['imaging_candidates']})", 10.5, True, ha="center")
    text(2.49, 8.41, "Required PET/MRI not performed,", 10, color=muted, ha="center")
    text(2.49, 8.15, "unavailable or unsuitable", 10, color=muted, ha="center")
    arrow(5.60, 9.37, 5.60, 8.94, color=linecolor, lw=.9)
    text(5.04, 8.72, f"Outside primary analysis (n = {sum(n for _, n in exclusions)})", 10.5, True)
    for yy, (label, n) in zip([8.41, 8.16, 7.91], exclusions):
        text(5.04, yy, f"{label} (n = {n})", 10., color="#334656")
    line([7.92, 8.20, 8.20], [9.37, 9.37, 8.98]); line([8.20, 8.20], [9.37, 9.56])
    arrow(8.20, 9.56, 8.75, 9.56, color=linecolor); arrow(8.20, 8.98, 8.75, 8.98, color=linecolor)
    text(10.58, 9.94, "12-month response", 11, True, ha="center")
    for yy, key, label, criterion, color in [(9.56, "responders", "Responders", "≥50% reduction", purple),
                                             (8.98, "nonresponders", "Nonresponders", "<50% reduction", orange)]:
        person(9.02, yy - .22, .50, color); text(9.76, yy, str(counts[key]), 23, color="#36556D", ha="center")
        text(11.18, yy + .11, label, 10.8, color="#334656", ha="center")
        text(11.18, yy - .15, criterion, 9.6, color=muted, ha="center")
    line([.26, 12.54], [7.68, 7.68], color="#DED9E3", lw=.75)
    offset = .3
    section("B", "Imaging and feature extraction", 6.40)
    for x, label in [(1.54, "FDG-PET"), (4.35, "3D T1-weighted MRI"), (7.15, "PET–T1w overlay"), (10.65, "Regional features")]:
        text(x, 5.98, label, 11.3, True, ha="center")
    stack("pet", .34, 3.34); stack("t1", 3.15, 3.34); stack("overlay", 5.95, 3.34)
    text(2.88, 4.50, "+", 22, color=muted, ha="center"); arrow(5.55, 4.49, 5.91, 4.49, lw=1.3)
    text(10.42, 5.29, f"{feature_counts[1]} PET uptake ratios", 11, True, purple, ha="center")
    tiles(9.28, 4.82, purple, feature_counts[1])
    text(10.42, 4.21, f"{feature_counts[2]} morphometric features", 11, True, ha="center")
    tiles(9.28, 3.74, grey, feature_counts[2])
    line([8.46, 8.88, 8.88], [4.49, 4.49, 4.94], color=muted, lw=.95); arrow(8.88, 4.94, 9.17, 4.94, lw=.95)
    line([4.34, 4.34, 8.95, 8.95], [3.25, 2.98, 2.98, 3.86], color="#A696B6", lw=.95)
    arrow(8.95, 3.86, 9.17, 3.86, color="#A696B6", lw=.95)
    text(6.58, 2.98, "FreeSurfer segmentation and morphometry", 9.5, color=muted, ha="center",
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5})
    text(1.50, 2.98, "Preoperative imaging", 9.5, color=muted, ha="center")
    line([.26, 12.54], [2.70, 2.70], color="#DED9E3", lw=.75)
    offset = 0.
    section("C", "Statistical analyses", 2.72)
    text(.40, 2.32, "Regional associations", 11.4, True, blue)
    text(1.42, 1.92, f"{feature_counts[1]} PET regions", 10.5, ha="center")
    for r in range(2):
        for c in range(3):
            ax.add_patch(Rectangle((.96 + c * .32, 1.39 + r * .20), .24, .13, fc=blue, ec="none"))
    arrow(1.40, 1.29, 1.40, 1.12, color=blue, lw=1)
    text(1.40, .91, "Clinical + scanner", 10.5, ha="center"); text(1.40, .68, "adjustment", 10.5, ha="center")
    arrow(1.40, .52, 1.40, .39, color=blue, lw=1); text(1.40, .21, "Adjusted effects · FDR", 10.5, True, ha="center")
    line([2.73, 2.73], [.15, 2.42], color="#D5DADE", lw=.75)
    text(2.98, 2.32, "Prediction", 11.4, True, blue); text(4.35, 2.32, "Shared cross-validation splits", 10.3, color=blue)
    for xx, label in [(3.79, "Clinical"), (4.43, "PET"), (4.99, "T1w")]:
        text(xx, 1.98, label, 10.2, ha="center")
    for rr, yy in enumerate([1.62, 1.34, 1.06]):
        for cc, xx in enumerate([3.79, 4.43, 4.99]):
            present = cc <= rr
            box(xx - .20, yy - .10, .40, .20, fill=blue if present else "white", edge=blue if present else "#D1D7DC", r=.014, lw=.65)
            if present:
                text(xx, yy, str(feature_counts[cc]), 9.6, bold=True, color="white", ha="center")
    line([5.31, 5.41, 5.41, 5.31], [1.73, 1.73, .95, .95]); arrow(5.41, 1.34, 5.64, 1.34, color=blue, lw=.95)
    text(6.64, 1.98, f"{outer_folds}-fold × {repeats}-repeat CV", 10.5, True, ha="center")
    for j in range(outer_folds):
        box(5.80 + j * 1.675 / outer_folds, 1.51, 1.375 / outer_folds, .24,
            fill=blue_pale if j < outer_folds - 1 else blue, edge=blue, r=.015, lw=.75)
    text(6.30, 1.35, "Train", 9.8, ha="center"); text(7.29, 1.35, "Test", 9.8, color=blue, ha="center")
    text(6.63, 1.04, f"{inner_folds}-fold tuning → refit", 10., ha="center")
    arrow(7.30, 1.87, 5.97, 1.87, color=blue, lw=.7, connectionstyle="arc3,rad=.09")
    arrow(7.53, 1.61, 7.78, 1.61, color=blue, lw=.95)
    text(8.76, 1.98, f"{repeats} predictions per patient", 10.5, True, ha="center")
    columns = math.ceil(repeats / 2)
    for i in range(repeats):
        rr, cc = divmod(i, columns)
        box(7.93 + cc * 1.64 / columns, 1.39 + rr * .20, 1.23 / columns, .14, fill=blue_pale, edge=blue, r=.008, lw=.55)
    arrow(9.64, 1.61, 9.93, 1.61, color=blue, lw=.95)
    text(11.15, 1.98, "Evaluation", 10.5, True, ha="center")
    box(10.15, 1.38, 1.98, .42, fill=blue_pale, edge=blue, r=.035, lw=.7)
    text(11.14, 1.59, "Mean probability", 11.3, True, ha="center", color=blue)
    text(11.14, 1.04, "Observed 12-month response", 10., ha="center")
    line([11.14, 11.14, 4.26], [.83, .68, .68], color="#A8B6C2", lw=.85)
    for xx, title in [(4.26, "AUC"), (7.52, "Brier score"), (10.92, "Calibration")]:
        arrow(xx, .68, xx, .49, color=blue, lw=.85); text(xx, .27, title, 12, True, ha="center")
    return _synthetic(fig, synthetic)


def assert_no_clipped_text(fig, tolerance=1.):
    """Rendered canvas-bound check. It is not a substitute for visual review."""
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); bad = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.width and box.height and (box.x0 < -tolerance or box.y0 < -tolerance or
                    box.x1 > fig.bbox.width + tolerance or box.y1 > fig.bbox.height + tolerance):
                bad.append(text.get_text())
    if bad:
        raise ValueError(f"text outside canvas: {bad}")


def export_figure(fig, path, *, dpi=150, check_text=True, minimum_asset_dpi=None):
    """Refuse overwrite; render PNG/SVG/PDF or native RGB LZW TIFF.

    ``minimum_asset_dpi`` checks embedded raster resolution at displayed size.
    No upsampling is substituted for a high-resolution anatomy asset.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if path.suffix.lower() not in {".png", ".svg", ".pdf", ".tif", ".tiff"}:
        raise ValueError("supported formats: PNG, SVG, PDF, TIFF")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if minimum_asset_dpi is not None:
        for ax in fig.axes:
            for artist in ax.images:
                h, w = artist.get_array().shape[:2]
                box = ax.get_position()
                effective = min(w / (box.width * fig.get_figwidth()), h / (box.height * fig.get_figheight()))
                if effective < minimum_asset_dpi:
                    raise ValueError("embedded image resolution is below the requested effective DPI")
    old_dpi = fig.dpi
    try:
        fig.set_dpi(dpi)
        if check_text:
            assert_no_clipped_text(fig)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            if path.suffix.lower() in {".tif", ".tiff"}:
                fig.canvas.draw()
                rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                Image.fromarray(rgb).save(handle, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
            else:
                # Preserve editable SVG text even after the renderer's rc_context exits.
                with plt.rc_context({"svg.fonttype": "none"}):
                    fig.savefig(handle, format=path.suffix[1:], dpi=dpi, facecolor="white")
    finally:
        fig.set_dpi(old_dpi)
    return path
