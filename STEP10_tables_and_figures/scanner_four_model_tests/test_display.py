"""Synthetic-only checks for the scanner-adjusted display step."""
from copy import deepcopy
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "10i_scanner_four_model_outputs.py"
SPEC = importlib.util.spec_from_file_location("scanner_display", SCRIPT)
DISPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISPLAY)


def synthetic_inputs():
    """Artificial data and illustrative bounds, never study observations."""
    n = 30
    y = np.tile([0, 1], n // 2)
    ids = [f"SYNTHETIC_{i:03d}" for i in range(n)]
    model_input = {"study_ids": ids, "y": y.tolist()}
    predictions = {"status": "PASS", "participant_ids": ids, "predictions": {}}
    performance, ci = {}, {}
    for i, key in enumerate(DISPLAY.MODELS):
        p = np.linspace(.01, .99, n)
        if i % 2:
            p = p[::-1]
        predictions["predictions"][key] = np.tile(p, (20, 1)).tolist()
        mean = np.asarray(predictions["predictions"][key]).mean(0)
        _, _, auc = DISPLAY._roc_and_auc(y, mean)
        brier = float(np.mean((mean - y) ** 2))
        performance[key] = {"n_participants": n, "repeats": 20, "auc": auc, "brier": brier,
            "calibration": {"status": "ESTIMABLE", "intercept_offset_slope_fixed_1": -.12,
                            "joint_intercept": .3, "slope": .6}}
        ci[key + "_auc"] = [max(0, auc - .1), min(1, auc + .1)]
        ci[key + "_brier"] = [brier - .05, brier + .05]
    rows = []
    for base, added in DISPLAY.COMPARISONS:
        delta = performance[added]["auc"] - performance[base]["auc"]
        improvement = performance[base]["brier"] - performance[added]["brier"]
        ci[added + "_minus_" + base + "_auc"] = [delta - .12, delta + .13]
        ci[base + "_minus_" + added + "_brier"] = [improvement - .03, improvement + .04]
        rows.append({"base": base, "added": added, "primary": (base, added) == ("CT", "CPT"),
            "delta_auc_added_minus_base": delta,
            "delta_auc_conditional_95_interval": ci[added + "_minus_" + base + "_auc"],
            "brier_improvement_base_minus_added": improvement,
            "brier_improvement_conditional_95_interval": ci[base + "_minus_" + added + "_brier"]})
    return model_input, predictions, performance, {"intervals": ci}, rows


class DisplayTests(unittest.TestCase):
    def test_complete_four_model_payload_and_table_signs(self):
        inputs = synthetic_inputs()
        payload = DISPLAY.assemble(*inputs)
        self.assertEqual(list(payload["models"]), list(DISPLAY.MODELS))
        self.assertEqual([(r["base"], r["added"]) for r in payload["paired_comparisons"]], list(DISPLAY.PLOT_COMPARISONS))
        self.assertEqual(sum(row["n"] for row in payload["calibration"]["bins"]), payload["n"])
        self.assertFalse(payload["patient_identifiers_exported"])
        self.assertNotIn("participant_ids", payload)
        with tempfile.TemporaryDirectory() as temporary:
            DISPLAY.write_tables(payload, inputs[2], Path(temporary))
            with (Path(temporary) / "Table2_A_performance.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows_a = list(csv.DictReader(handle))
            self.assertEqual([r["model"] for r in rows_a], list(DISPLAY.MODELS))
            self.assertEqual(float(rows_a[0]["calibration_intercept_offset_slope_fixed_1"]), -.12)
            with (Path(temporary) / "Table2_B_paired_comparisons.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows_b = list(csv.DictReader(handle))
            original = inputs[4][0]
            self.assertEqual(float(rows_b[0]["delta_brier_added_minus_base"]), -original["brier_improvement_base_minus_added"])
            self.assertEqual(float(rows_b[0]["delta_brier_lower"]), -original["brier_improvement_conditional_95_interval"][1])
            self.assertEqual(float(rows_b[0]["delta_brier_upper"]), -original["brier_improvement_conditional_95_interval"][0])

    def test_rejects_identifier_reordering(self):
        inputs = synthetic_inputs()
        inputs[1]["participant_ids"] = list(reversed(inputs[1]["participant_ids"]))
        with self.assertRaisesRegex(ValueError, "participant order"):
            DISPLAY.assemble(*inputs)

    def test_rejects_metric_version_mismatch(self):
        inputs = synthetic_inputs()
        inputs[2]["CP"]["auc"] += .01
        with self.assertRaisesRegex(ValueError, "CP AUC"):
            DISPLAY.assemble(*inputs)

    def test_rejects_interval_version_mismatch(self):
        inputs = synthetic_inputs()
        inputs[4][0]["delta_auc_conditional_95_interval"] = [.3, .4]
        with self.assertRaisesRegex(ValueError, "interval versions"):
            DISPLAY.assemble(*inputs)

    def test_rejects_prediction_incomplete_or_out_of_range(self):
        for value in (-.1, 1.1, float("nan")):
            inputs = synthetic_inputs()
            inputs[1]["predictions"]["CT"][0][0] = value
            with self.assertRaisesRegex(ValueError, "20-by-participant"):
                DISPLAY.assemble(*inputs)
        inputs = synthetic_inputs()
        inputs[1]["predictions"]["CT"].pop()
        with self.assertRaisesRegex(ValueError, "20-by-participant"):
            DISPLAY.assemble(*inputs)

    def test_ties_and_exact_calibration_edges(self):
        y = np.array([0, 1, 0, 1])
        fpr, tpr, auc = DISPLAY._roc_and_auc(y, np.array([.5, .5, .5, .5]))
        self.assertEqual(auc, .5)
        self.assertEqual(fpr.tolist(), [0, 1])
        self.assertEqual(tpr.tolist(), [0, 1])
        bins = DISPLAY._calibration_bins(np.array([0, 1, 0, 1, 0, 1]), np.array([0, .2, .4, .6, .8, 1]))
        self.assertEqual([r["n"] for r in bins], [1, 1, 1, 1, 2])

    def test_primary_comparison_and_both_outcomes_required(self):
        inputs = synthetic_inputs()
        inputs[4][1]["primary"] = True
        with self.assertRaisesRegex(ValueError, "primary comparison"):
            DISPLAY.assemble(*inputs)
        inputs = synthetic_inputs()
        inputs[0]["y"] = [0] * len(inputs[0]["y"])
        with self.assertRaisesRegex(ValueError, "both groups"):
            DISPLAY.assemble(*inputs)

    def test_original_inputs_not_mutated(self):
        inputs = synthetic_inputs()
        before = deepcopy(inputs)
        DISPLAY.assemble(*inputs)
        self.assertEqual(inputs, before)


if __name__ == "__main__":
    unittest.main()
