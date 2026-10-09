"""Artificial-only validation tests; optional source parity reads Python code only.

Set PET_VNS_ORIGINAL_CODE to a locally authorized original code directory to
enable AST-selected pure-function comparisons. No original module is imported,
and no original configuration, outcomes, images, or results are opened.
"""
import ast
from dataclasses import replace
import inspect
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pet_vns_prediction import core, workflow
from pet_vns_prediction.modeling import paper_settings
from pet_vns_prediction import validation as v


def fixture(n=24):
    settings = replace(paper_settings(), repeats=1, c_grid=(.1, .5), l1_grid=(0., .5),
                       profile="artificial_validation_test")
    x, y = core.synthetic_fixture(settings, n=n, seed=1008)
    return settings, x, y


class ValidationTests(unittest.TestCase):
    def test_default_counts_and_no_test_outcomes(self):
        self.assertEqual(inspect.signature(v.run_development_bootstrap).parameters["draws"].default, 500)
        self.assertEqual(inspect.signature(v.conditional_intervals).parameters["draws"].default, 2000)
        self.assertNotIn("ytest", inspect.signature(v.develop).parameters)

    def test_duplicate_people_never_cross_inner_boundary(self):
        groups = np.array([0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11])
        y = (groups >= 6).astype(int)
        folds = v.grouped_inner_plan(y, groups, 117)
        seen = np.zeros(len(y), int)
        for fold in folds:
            tr, va = fold["train"], fold["validation"]
            self.assertFalse(set(groups[tr]) & set(groups[va]))
            self.assertEqual(set(tr) | set(va), set(range(len(y))))
            seen[va] += 1
        np.testing.assert_array_equal(seen, 1)
        changed = y.copy()
        changed[0] = 1
        with self.assertRaisesRegex(core.InputRejected, "Conflicting"):
            v.grouped_inner_plan(changed, groups, 117)
        with self.assertRaises(core.ClassSupportError):
            v.grouped_inner_plan([0, 0, 1, 1], [0, 1, 2, 3], 117)

    def test_develop_heldout_predictors_cannot_change_fitting(self):
        s, x, y = fixture()
        sample = np.r_[np.arange(len(y)), 0, 0, 3]
        original = v.develop(x.iloc[sample], y.iloc[sample], sample, x.iloc[:4], "CP", 871, s)
        changed = x.iloc[:4].copy()
        for c in s.model_columns()["CP"]:
            if c not in s.categories:
                changed[c] = changed[c] * 5 + 10
        other = v.develop(x.iloc[sample], y.iloc[sample], sample, changed, "CP", 871, s)
        for key in ("selected", "numerical_fits", "preprocessing_audit", "inner"):
            self.assertEqual(original[key], other[key])
        for audit in original["preprocessing_audit"]:
            self.assertTrue(all(row < len(sample) for row in audit["fit_rows"]))
        bad_groups = np.arange(len(sample))
        with self.assertRaisesRegex(core.InputRejected, "Copies of one participant"):
            v.develop(x.iloc[sample], y.iloc[sample], bad_groups, x.iloc[:4], "CP", 871, s)

    def test_retuned_ablations_share_explicit_splits_and_paired_resamples(self):
        s, x, y = fixture()
        plans = core.make_split_plan(y, s)
        result = v.run_ablations(x, y, plans, s, interval_draws=10)
        self.assertIs(result["plans"], plans)
        self.assertEqual(len(set(result["model_split_digests"].values())), 1)
        self.assertEqual(len(result["tuning"]), len(plans) * 4)
        for record in result["tuning"]:
            self.assertEqual(record["fit_count"], 13)
        cols = result["model_columns"]
        self.assertNotIn(s.pet[0], cols["CP_without_thalamus"])
        self.assertNotIn(s.pet[4], cols["CP_without_striatopallidal"])
        self.assertNotIn(s.pet[0], cols["CP_without_both"])
        self.assertNotIn(s.pet[4], cols["CP_without_both"])
        self.assert_paired_quantiles(result, y)

    def assert_paired_quantiles(self, result, y):
        metrics = result["ablation_metrics"]
        rng, yy = np.random.default_rng(20260910), np.asarray(y)
        indices = [rng.integers(0, len(y), len(y)) for _ in range(10)]
        indices = [ix for ix in indices if len(set(yy[ix])) == 2]
        cp = result["predictions"]["CP"].mean(0)
        for name, matrix in result["predictions"].items():
            if name == "CP":
                continue
            pp = matrix.mean(0)
            delta = np.array([[roc_auc_score(yy[ix], cp[ix]) - roc_auc_score(yy[ix], pp[ix]),
                               np.mean((yy[ix] - cp[ix])**2) - np.mean((yy[ix] - pp[ix])**2)] for ix in indices])
            q = np.quantile(delta, [.025, .975], axis=0)
            saved = metrics["comparisons"]["CP_minus_" + name]
            np.testing.assert_allclose(saved["delta_auc_conditional95"], q[:, 0], atol=1e-14)
            np.testing.assert_array_equal(saved["delta_brier_conditional95"], q[:, 1])

    def test_bidirectional_scanner_training_isolation(self):
        s, x, y = fixture()
        # Balance each device deterministically without using real device labels.
        scanner = pd.Series("scanner_a", index=x.index)
        for label in (0, 1):
            rows = np.flatnonzero(np.asarray(y) == label)
            scanner.iloc[rows[::2]] = "scanner_b"
        result = v.run_cross_scanner(x, y, scanner, s, interval_draws=10)
        self.assertEqual([d["heldout_scanner"] for d in result["directions"]], ["scanner_a", "scanner_b"])
        for serial, record in enumerate(result["directions"]):
            self.assertFalse(set(record["train_rows"]) & set(record["test_rows"]))
            self.assertEqual(record["models"]["C"]["inner"], record["models"]["CP"]["inner"])
            for fit in record["models"].values():
                self.assertEqual(fit["seed"], s.seed + 7000 + serial)
                self.assertEqual(fit["training_patient_groups"], record["train_rows"])
                self.assertTrue(all(max(a["fit_rows"]) < record["n_train"] for a in fit["preprocessing_audit"]))

    def test_bootstrap_full_development_and_probability_stability(self):
        s, x, y = fixture()
        result = v.run_development_bootstrap(x, y, s, draws=2)
        self.assertEqual(result["summary"]["draws_valid"], 2)
        rng = np.random.default_rng(20260921)
        for serial, row in enumerate(result["records"]):
            sample = rng.integers(0, len(y), len(y))
            self.assertEqual(row["sample_rows"], sample.tolist())
            self.assertFalse(set(sample) & set(row["oob_rows"]))
            self.assertEqual(row["models"]["C"]["inner"], row["models"]["CP"]["inner"])
            for fit in row["models"].values():
                self.assertEqual(fit["seed"], s.seed + 200000 + serial)
                self.assertEqual(fit["training_patient_groups"], sample.tolist())
                for fold in fit["inner"]:
                    self.assertFalse(set(sample[fold["train"]]) & set(sample[fold["validation"]]))
                pp, yy = np.asarray(fit["evaluation_predictions"]), np.asarray(y)
                self.assertAlmostEqual(fit["auc_optimism"], roc_auc_score(yy[sample], pp[sample]) - roc_auc_score(yy, pp))
                self.assertEqual(fit["brier_optimism"], float(np.mean((yy - pp)**2) - np.mean((yy[sample] - pp[sample])**2)))
        for model in ("C", "CP"):
            fits = [row["models"][model] for row in result["records"]]
            summary = result["summary"]["bootstrap"][model]
            reference = result["reference_fits"][model]["metrics"]
            self.assertEqual(summary["optimism_corrected_auc"], reference["auc"] - np.mean([f["auc_optimism"] for f in fits]))
            self.assertEqual(summary["optimism_corrected_brier"], reference["brier"] + np.mean([f["brier_optimism"] for f in fits]))
            expected = np.quantile([f["evaluation_predictions"] for f in fits], [.025, .5, .975], axis=0)
            np.testing.assert_array_equal(result["probability_stability"][model]["bootstrap_percentiles_2p5_50_97p5"], expected)
        increment = result["summary"]["paired_increment"]
        self.assertEqual(increment["optimism_corrected_delta_auc_CP_minus_C"],
                         result["summary"]["bootstrap"]["CP"]["optimism_corrected_auc"] - result["summary"]["bootstrap"]["C"]["optimism_corrected_auc"])

    def test_invalid_draw_is_not_replaced(self):
        s, x, y = fixture(12)
        # Seed zero's first sample has only two unique members of one class.
        chosen_seed = next(seed for seed in range(100) if min(np.bincount(np.asarray(y)[np.unique(
            np.random.default_rng(seed).integers(0, len(y), len(y)))], minlength=2)) < 3)
        with self.assertRaisesRegex(core.InputRejected, "No valid bootstrap") as raised:
            v.run_development_bootstrap(x, y, s, draws=1, seed=chosen_seed)
        self.assertEqual(raised.exception.draws_requested, 1)
        self.assertEqual(len(raised.exception.invalid_draws), 1)
        self.assertTrue(raised.exception.invalid_draws[0]["no_replacement_draw"])

    def test_timeout_and_numerical_failures_stop_without_retry(self):
        s, x, y = fixture()
        with self.assertRaises(TimeoutError):
            v.run_development_bootstrap(x, y, s, draws=1, timeout_seconds=1e-12)
        with patch.object(workflow, "corrected_outer", side_effect=core.FitFailure("synthetic injected failure")) as fit:
            with self.assertRaises(core.FitFailure):
                v.run_development_bootstrap(x, y, s, draws=1)
            self.assertEqual(fit.call_count, 1)

    def test_empty_intervals_are_explicit_and_bad_input_rejected(self):
        result = v.conditional_intervals([0, 0], {"C": [.1, .2], "CP": [.2, .3]}, draws=10)
        self.assertEqual(result["valid"], 0)
        self.assertEqual(result["one_class_draws"], 10)
        self.assertIsNone(result["delta_auc_CP_minus_C"])
        self.assertIsNone(v.measures([], [])["brier"])
        with self.assertRaises(core.InputRejected):
            v.conditional_intervals([0, 1], {"C": [.1, 1.2], "CP": [.2, .3]}, draws=10)


@unittest.skipUnless(os.environ.get("PET_VNS_ORIGINAL_CODE"), "Optional local original-code-only parity check")
class OriginalSourceParityTests(unittest.TestCase):
    def source_functions(self, filename, names, namespace, constant_overrides=None):
        path = Path(os.environ["PET_VNS_ORIGINAL_CODE"]) / filename
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=filename)
        chosen = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in chosen}, set(names))
        class BoundSyntheticWork(ast.NodeTransformer):
            def visit_Constant(self, node):
                if type(node.value) is int and node.value in (constant_overrides or {}):
                    return ast.copy_location(ast.Constant(value=constant_overrides[node.value]), node)
                return node
        selected = ast.Module(body=chosen, type_ignores=[])
        selected = ast.fix_missing_locations(BoundSyntheticWork().visit(selected))
        # Import statements, module assignments, and main() are never executed.
        exec(compile(selected, filename, "exec"), namespace)
        return namespace

    def test_original_grouped_develop_intervals_and_distribution_exact(self):
        s, x, y = fixture()
        namespace = {"np": np, "pd": pd, "core": core, "w": workflow, "SET": s,
                     "COLS": s.model_columns(), "StratifiedKFold": StratifiedKFold,
                     "roc_auc_score": roc_auc_score, "deadline": lambda: None,
                     "S": {"heldout_intervals": {"draws": 10, "seed": 20260910}}}
        source = self.source_functions("cp_validation_20260921_v1.py",
            ["require", "inner_plan", "develop", "measures", "conditional_intervals", "dist"], namespace)
        sample = np.r_[np.arange(len(y)), 2, 2, 9]
        original = source["develop"](x.iloc[sample], y.iloc[sample], sample, x.iloc[:4], "CP", 887)
        public = v.develop(x.iloc[sample], y.iloc[sample], sample, x.iloc[:4], "CP", 887, s)
        self.assertEqual(public, original)
        self.assertEqual(v.grouped_inner_plan(y.iloc[sample], sample, 887), source["inner_plan"](y.iloc[sample], sample, 887))
        predictions = {"C": np.linspace(.1, .7, len(y)), "CP": np.linspace(.2, .9, len(y))}
        self.assertEqual(v.conditional_intervals(y, predictions, draws=10), source["conditional_intervals"](y, predictions))
        self.assertEqual(v.measures(y, predictions["C"]), source["measures"](y, predictions["C"]))
        self.assertEqual(v._distribution([.1, .9, .2]), source["dist"]([.1, .9, .2]))

    def test_original_ablation_columns_pair_auc_and_independent_metric_verifier(self):
        s, x, y = fixture()
        source = self.source_functions("subcortical_phenotype_followup_20260921_v1.py",
            ["columns", "auc"], {"CLINICAL": list(s.clinical), "PATHWAYS": list(s.pet), "np": np})
        self.assertEqual(v.ablation_columns(s), source["columns"]())
        p, yy = np.linspace(.1, .9, len(y)), np.asarray(y)
        self.assertEqual(v._pair_auc(yy, p), source["auc"](yy, p))
        import math
        verifier = self.source_functions("verify_cp_validation_20260921_v1.py", ["auc", "brier"], {"np": np, "math": math})
        measured = v.measures(yy, p)
        self.assertEqual(measured["auc"], verifier["auc"](yy, p))
        self.assertAlmostEqual(measured["brier"], verifier["brier"](yy, p), places=14)

    def test_original_ablation_evaluation_exact_with_bounded_artificial_inputs(self):
        s, x, y = fixture()
        yy, ids = np.asarray(y), list(x.index)
        rng = np.random.default_rng(113)
        matrices = {name: rng.uniform(.1, .9, size=(2, len(y)))
                    for name in ("CP", *v.ablation_columns(s))}
        result = {"status": "PASS", "participant_ids": ids, "predictions": matrices,
                  "model_split_digests": {name: "artificial-shared-plan" for name in matrices}}
        public = v.evaluate_ablations(result, y, draws=10)
        records = {"baseline": {"predictions": matrices["CP"].tolist()},
                   "metrics": {"models": {"CP": public["models"]["CP"]}}}
        for name in v.ablation_columns(s):
            records[str(Path("artificial") / name / "predictions.json")] = {
                "study_ids": ids, "y": yy.tolist(), "predictions": matrices[name].tolist()}
        saved = {}
        namespace = {"np": np, "core": core, "roc_auc_score": roc_auc_score,
            "CLINICAL": list(s.clinical), "PATHWAYS": list(s.pet), "OUT": Path("artificial"),
            "SOURCES": {"old_CP": "baseline", "old_metrics": "metrics"},
            "read": lambda key: records[str(key)], "save": lambda name, value: saved.update({name: value}),
            "emit": lambda *args, **kwargs: None}
        source = self.source_functions("subcortical_phenotype_followup_20260921_v1.py",
            ["require", "columns", "auc", "evaluate"], namespace,
            # Only cohort dimension and iteration bound are adapted; all
            # arithmetic and paired-resample ordering remains original.
            constant_overrides={57: len(y), 2000: 10})
        source["evaluate"](ids, yy)
        for key, value in saved["prediction_metrics.json"].items():
            self.assertEqual(public[key], value)


if __name__ == "__main__":
    unittest.main()
