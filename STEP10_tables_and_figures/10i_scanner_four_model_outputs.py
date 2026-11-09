"""Create the scanner-adjusted four-model Figure 3 and Table 2 inputs.

This is a display-only step. It reads saved held-out probabilities and summaries,
checks their agreement, and exports aggregate plot data, tables, and a figure.
It never fits models, selects hyperparameters, or performs a new bootstrap.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import new_output, read_json, write_json
import numpy as np

MODELS = ("C", "CT", "CP", "CPT")
NAMES = {"C": "Clinical", "CT": "Clinical + T1w", "CP": "Clinical + PET", "CPT": "Clinical + PET + T1w"}
PALETTE = {"C": "#72777D", "CT": "#526D82", "CP": "#79529B", "CPT": "#C17C39"}
COMPARISONS = (("CT", "CPT"), ("C", "CP"), ("CP", "CPT"), ("C", "CT"))
PLOT_COMPARISONS = (("CT", "CPT"), ("C", "CP"), ("C", "CT"), ("CP", "CPT"))


def formatted(value):
    """Retain the manuscript's decimal half-up display convention."""
    return str(Decimal(str(round(float(value), 12))).quantize(Decimal(".001"), rounding=ROUND_HALF_UP))


def _close(a, b, description):
    if not np.isfinite(a) or not np.isfinite(b) or abs(float(a) - float(b)) > 1e-12:
        raise ValueError(description + " does not agree with the saved predictions")


def _interval(value, description):
    bounds = np.asarray(value, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError("Ordered finite two-sided interval required: " + description)
    return bounds.tolist()


def _roc_and_auc(y, probability):
    """Use the source renderer's stable tie grouping, including all thresholds."""
    order = np.argsort(-probability, kind="stable")
    sy, sp = y[order], probability[order]
    ends = np.r_[np.where(np.diff(sp))[0], len(sp) - 1]
    tp = np.r_[0, np.cumsum(sy)[ends]]
    fp = np.r_[0, np.cumsum(1 - sy)[ends]]
    tpr, fpr = tp / y.sum(), fp / (len(y) - y.sum())
    # Summing trapezoids explicitly also supports the documented NumPy >= 1.26.
    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
    pairwise = float((probability[y == 1, None] > probability[y == 0]).mean()
                     + .5 * (probability[y == 1, None] == probability[y == 0]).mean())
    _close(auc, pairwise, "Trapezoid and pairwise AUC")
    return fpr, tpr, auc


def _calibration_bins(y, probability):
    """Five fixed-width bins; Wilson intervals are descriptive, not bootstrap CIs."""
    bins, z = [], 1.959963984540054
    for i, (lo, hi) in enumerate(zip([0, .2, .4, .6, .8], [.2, .4, .6, .8, 1])):
        mask = (probability >= lo) & ((probability < hi) if i < 4 else (probability <= hi))
        n, successes = int(mask.sum()), int(y[mask].sum())
        if n == 0:
            raise ValueError("The source Figure 3 specification requires all five CP calibration bins to be nonempty")
        fraction = successes / n
        denominator = 1 + z * z / n
        center = (fraction + z * z / (2 * n)) / denominator
        half = z * math.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / denominator
        bins.append({"bin": i + 1, "n": n, "responders": successes,
                     "mean_probability": float(probability[mask].mean()),
                     "observed_fraction": fraction,
                     "wilson95_descriptive": [center - half, center + half]})
    if sum(row["n"] for row in bins) != len(y):
        raise ValueError("Calibration-bin counts do not reconcile with the cohort")
    return bins


def assemble(model_input, predictions, performance, intervals, comparisons):
    """Bind all display values to one exact participant and result version."""
    ids = model_input.get("study_ids")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("Unique ordered study_ids are required")
    if predictions.get("status") != "PASS" or predictions.get("participant_ids") != ids:
        raise ValueError("PASS predictions in exactly the model-input participant order are required")
    y = np.asarray(model_input.get("y"))
    if y.shape != (len(ids),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("One binary outcome per participant, with both groups, is required")
    y = y.astype(int)
    ci = intervals.get("intervals", intervals)
    models, means = {}, {}
    for key in MODELS:
        matrix = np.asarray(predictions.get("predictions", {}).get(key), dtype=float)
        if matrix.shape != (20, len(y)) or not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
            raise ValueError("All four models require complete 20-by-participant probability matrices")
        mean = matrix.mean(axis=0)
        means[key] = mean
        fpr, tpr, auc = _roc_and_auc(y, mean)
        brier = float(np.mean((mean - y) ** 2))
        _close(auc, performance[key]["auc"], key + " AUC")
        _close(brier, performance[key]["brier"], key + " Brier score")
        if int(performance[key]["n_participants"]) != len(y) or int(performance[key]["repeats"]) != 20:
            raise ValueError("Performance denominator or repeat count differs from predictions")
        calibration = performance[key]["calibration"]
        if calibration.get("status") != "ESTIMABLE":
            raise ValueError("The supplied model calibration is not estimable: " + key)
        for field in ("intercept_offset_slope_fixed_1", "joint_intercept", "slope"):
            if not np.isfinite(calibration[field]):
                raise ValueError("A saved finite calibration parameter is required: " + key + " / " + field)
        models[key] = {"auc": performance[key]["auc"], "auc95": _interval(ci[key + "_auc"], key + " AUC"),
                       "brier": performance[key]["brier"], "brier95": _interval(ci[key + "_brier"], key + " Brier"),
                       "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist()}
    if not isinstance(comparisons, list) or len(comparisons) != 4:
        raise ValueError("All four paired comparisons are required")
    indexed = {(row["base"], row["added"]): row for row in comparisons}
    if set(indexed) != set(COMPARISONS):
        raise ValueError("Unexpected, missing, or duplicated model comparison")
    rows = []
    for base, added in PLOT_COMPARISONS:
        row = indexed[(base, added)]
        if row.get("primary") is not ((base, added) == ("CT", "CPT")):
            raise ValueError("Only CT-to-CPT may be flagged as the primary comparison")
        _close(row["delta_auc_added_minus_base"], models[added]["auc"] - models[base]["auc"], "Paired AUC difference")
        _close(row["brier_improvement_base_minus_added"], models[base]["brier"] - models[added]["brier"], "Paired Brier improvement")
        for row_key, ci_key in (("delta_auc_conditional_95_interval", added + "_minus_" + base + "_auc"),
                               ("brier_improvement_conditional_95_interval", base + "_minus_" + added + "_brier")):
            bounds = _interval(row[row_key], row_key)
            if not np.allclose(bounds, _interval(ci[ci_key], ci_key), rtol=0, atol=1e-12):
                raise ValueError("Comparison and conditional-interval versions differ")
        rows.append({key: row[key] for key in ("base", "added", "primary", "delta_auc_added_minus_base",
                    "delta_auc_conditional_95_interval", "brier_improvement_base_minus_added",
                    "brier_improvement_conditional_95_interval")})
    return {"n": len(y), "responders": int(y.sum()), "models": models,
            "calibration": {"model": "CP", "fixed_bin_edges": [0, .2, .4, .6, .8, 1],
                "bins": _calibration_bins(y, means["CP"]), "parameters": performance["CP"]["calibration"]},
            "paired_comparisons": rows, "patient_identifiers_exported": False}


def write_tables(payload, performance, output):
    """Table 2 CSVs retain numeric precision; displayed Delta Brier is added-base."""
    fields_a = ["model", "model_label", "auc", "auc_lower", "auc_upper", "brier", "brier_lower", "brier_upper",
                "calibration_intercept_offset_slope_fixed_1", "calibration_slope"]
    with (output / "Table2_A_performance.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_a)
        writer.writeheader()
        for model in MODELS:
            row, cal = payload["models"][model], performance[model]["calibration"]
            writer.writerow(dict(zip(fields_a, [model, NAMES[model], row["auc"], *row["auc95"], row["brier"],
                *row["brier95"], cal["intercept_offset_slope_fixed_1"], cal["slope"]])))
    fields_b = ["base", "added", "contrast_label", "primary", "delta_auc_added_minus_base", "delta_auc_lower",
                "delta_auc_upper", "delta_brier_added_minus_base", "delta_brier_lower", "delta_brier_upper"]
    indexed = {(r["base"], r["added"]): r for r in payload["paired_comparisons"]}
    with (output / "Table2_B_paired_comparisons.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_b)
        writer.writeheader()
        for base, added in COMPARISONS:
            row = indexed[(base, added)]
            lo, hi = row["brier_improvement_conditional_95_interval"]
            writer.writerow(dict(zip(fields_b, [base, added, NAMES[added] + " - " + NAMES[base], row["primary"],
                row["delta_auc_added_minus_base"], *row["delta_auc_conditional_95_interval"],
                -row["brier_improvement_base_minus_added"], -hi, -lo])))


def render(payload, output, dpi):
    """Preserve the approved four-panel layout and model color/line identities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    styles = {"C": "--", "CT": ":", "CP": "-", "CPT": "-."}
    markers = {"C": "o", "CT": "D", "CP": "s", "CPT": "^"}
    plot, bins = payload["models"], payload["calibration"]["bins"]
    cal = payload["calibration"]["parameters"]
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.edgecolor": "#92969B", "axes.linewidth": .7,
        "text.color": "#30363A", "axes.labelcolor": "#30363A", "xtick.color": "#555B60", "ytick.color": "#555B60",
        "svg.fonttype": "none", "svg.hashsalt": "Figure3_scanner_20261109", "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig = plt.figure(figsize=(9.2, 8.0), dpi=dpi)
    try:
        a = fig.add_axes([.09, .615, .37, .31]); b = fig.add_axes([.60, .615, .36, .31])
        c = fig.add_axes([.235, .10, .225, .265]); d = fig.add_axes([.745, .10, .215, .265])
        def header(letter, title, x, yy):
            fig.text(x, yy, letter, fontsize=14, weight="bold")
            fig.text(x + .030, yy, title, fontsize=11.5, weight="bold")
        def clean(ax, grid="both"):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis=grid, color="#E6E7E9", linewidth=.6, zorder=0)
            ax.tick_params(length=3, width=.6); ax.set_axisbelow(True)
        header("A", "Discrimination", .055, .964); header("B", "Calibration: Clinical + PET", .555, .964)
        header("C", "Prediction error", .055, .404); header("D", "Incremental discrimination", .555, .404)
        a.plot([0, 1], [0, 1], color="#B9BDC1", ls=":", lw=1)
        for key in MODELS:
            a.plot(plot[key]["roc_fpr"], plot[key]["roc_tpr"], color=PALETTE[key], ls=styles[key], lw=1.8 if key == "CP" else 1.55)
        a.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False-positive rate", ylabel="True-positive rate")
        a.set_xticks(np.arange(0, 1.1, .2)); a.set_yticks(np.arange(0, 1.1, .2)); clean(a)
        fig.text(.46, .555, "AUC", fontsize=9, ha="right")
        for key, yy in zip(MODELS, [.527, .499, .471, .443]):
            fig.add_artist(Line2D([.09, .116], [yy + .005, yy + .005], transform=fig.transFigure, color=PALETTE[key], ls=styles[key], lw=1.8))
            fig.text(.125, yy, NAMES[key], fontsize=9.5)
            row = plot[key]
            fig.text(.46, yy, f"{formatted(row['auc'])} ({formatted(row['auc95'][0])}–{formatted(row['auc95'][1])})", fontsize=9, ha="right")
        b.plot([0, 1], [0, 1], color="#B9BDC1", ls=":", lw=1)
        b.errorbar([r["mean_probability"] for r in bins], [r["observed_fraction"] for r in bins],
            yerr=[[r["observed_fraction"] - r["wilson95_descriptive"][0] for r in bins],
                  [r["wilson95_descriptive"][1] - r["observed_fraction"] for r in bins]],
            fmt="s-", color=PALETTE["CP"], ms=5.2, lw=1.35, elinewidth=1.05, capsize=3, capthick=1)
        for row in bins:
            nearzero = row["wilson95_descriptive"][0] < .08
            b.annotate(f"n={row['n']}", (row["mean_probability"], row["wilson95_descriptive"][0]),
                xytext=(0, 7 if nearzero else -12), textcoords="offset points", ha="center", va="bottom" if nearzero else "top",
                fontsize=8.5, color=PALETTE["CP"], bbox={"facecolor": "white", "edgecolor": "none", "pad": .3} if nearzero else None)
        b.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Mean predicted response probability", ylabel="Observed response proportion")
        b.set_xticks(np.arange(0, 1.1, .2)); b.set_yticks(np.arange(0, 1.1, .2)); clean(b)
        fig.text(.60, .538, f"Calibration intercept: {formatted(cal['intercept_offset_slope_fixed_1']).replace('-', '−')}     Slope: {formatted(cal['slope'])}", fontsize=9)
        for key, yy in zip(MODELS, [3, 2, 1, 0]):
            row = plot[key]; x = row["brier"]; lo, hi = row["brier95"]
            c.errorbar(x, yy, xerr=[[x - lo], [hi - x]], fmt=markers[key], color=PALETTE[key], ms=6, elinewidth=1.5, capsize=3.5, capthick=1.2)
            c.annotate(formatted(x), (x, yy), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=PALETTE[key])
        c.set_yticks([3, 2, 1, 0], list(NAMES.values()))
        brier_right = max(.305, math.ceil((max(r["brier95"][1] for r in plot.values()) + .005) / .05) * .05)
        c.set(xlim=(0, brier_right), ylim=(-.65, 3.65), xlabel="Brier score")
        c.set_xticks([0, .1, .2, .3] if brier_right == .305 else np.linspace(0, brier_right, 4))
        clean(c, "x"); c.spines["left"].set_visible(False); c.tick_params(axis="y", length=0, labelsize=9.5, pad=9)
        labels = ["PET added to\nClinical + T1w", "PET added to\nClinical", "T1w added to\nClinical", "T1w added to\nClinical + PET"]
        for row, label, yy in zip(payload["paired_comparisons"], labels, [3, 2, 1, 0]):
            pair = (row["base"], row["added"])
            x = row["delta_auc_added_minus_base"]; lo, hi = row["delta_auc_conditional_95_interval"]
            color = PALETTE["CPT"] if row["primary"] else PALETTE["CP"] if pair == ("C", "CP") else PALETTE["CT"]
            d.errorbar(x, yy, xerr=[[x - lo], [hi - x]], fmt="s" if row["primary"] else "o", color=color, ms=6, elinewidth=1.5, capsize=3.5, capthick=1.2)
            d.annotate(("+" if x >= 0 else "−") + formatted(abs(x)), (x, yy), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=color)
        d.axvline(0, color="#858B90", lw=.95, ls="--"); d.set_yticks([3, 2, 1, 0], labels)
        low = min(-.16, min(r["delta_auc_conditional_95_interval"][0] for r in payload["paired_comparisons"]) - .01)
        high = max(.46, max(r["delta_auc_conditional_95_interval"][1] for r in payload["paired_comparisons"]) + .01)
        d.set(xlim=(low, high), ylim=(-.65, 3.65), xlabel="ΔAUC: added − base model")
        d.set_xticks([-.1, 0, .2, .4]); clean(d, "x"); d.spines["left"].set_visible(False); d.tick_params(axis="y", length=0, labelsize=9.1, pad=9)
        fig.canvas.draw(); renderer = fig.canvas.get_renderer(); outside = []
        for text in fig.findobj(matplotlib.text.Text):
            if text.get_visible() and text.get_text():
                bounds = text.get_window_extent(renderer)
                if bounds.width and bounds.height and (bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.width or bounds.y1 > fig.bbox.height):
                    outside.append(text.get_text())
        if outside:
            raise ValueError("Figure text exceeds the canvas: " + repr(outside))
        fig.savefig(output / "Figure3.png", dpi=dpi)
        fig.savefig(output / "Figure3.svg")
        fig.savefig(output / "Figure3.pdf")
        native = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba().tobytes()).convert("RGB")
        native.save(output / "Figure3.tiff", format="TIFF", compression="tiff_adobe_deflate", dpi=(dpi, dpi))
        return {"dpi": dpi, "pixel_dimensions": list(native.size), "visible_text_outside_canvas": outside,
                "visual_review": "REQUIRED"}
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input", type=Path, required=True, help="Private STEP05 JSON with study_ids and y")
    parser.add_argument("--predictions", type=Path, required=True, help="Private saved four-model held-out probabilities")
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A new directory; existing output is never overwritten")
    parser.add_argument("--dpi", type=int, choices=(150, 600), default=150)
    parser.add_argument("--skip-render", action="store_true", help="Export only aggregate plot data, CSVs and notes")
    args = parser.parse_args()
    inputs = {key: getattr(args, key) for key in ("model_input", "predictions", "performance", "intervals", "comparisons")}
    hashes = {key: sha256(path.read_bytes()).hexdigest() for key, path in inputs.items()}
    performance = read_json(args.performance)
    payload = assemble(read_json(args.model_input), read_json(args.predictions), performance,
                       read_json(args.intervals), read_json(args.comparisons))
    output = new_output(args.output)
    write_json(output / "Figure3_plot_data.json", payload)
    write_tables(payload, performance, output)
    rendered = None if args.skip_render else render(payload, output, args.dpi)
    if hashes != {key: sha256(path.read_bytes()).hexdigest() for key, path in inputs.items()}:
        raise RuntimeError("A source file changed during export")
    write_json(output / "display_provenance.json", {"status": "DISPLAY_EXPORT_COMPLETE", "source_sha256": hashes,
        "models": list(MODELS), "primary_comparison": ["CT", "CPT"], "n": payload["n"],
        "model_fitting": False, "new_bootstrap": False, "sources_unchanged": True,
        "participant_identifiers_exported": False, "render": rendered,
        "intervals": "Saved paired participant-bootstrap conditional 95% intervals; no new resampling",
        "calibration_bins": "Five fixed-width bins; pointwise descriptive Wilson 95% intervals",
        "calibration_intercept": "Offset intercept with slope fixed to 1, not the separately fitted joint intercept",
        "table_delta_brier": "Added minus base; negative indicates lower prediction error",
        "source_notice": "The original three-model/ablation Figure 3 remains a separate historical specification"})
    print(str(output))


if __name__ == "__main__":
    main()
