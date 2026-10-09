"""Synthetic-only tests of the published numerical methods and interfaces.

No private project, imaging, clinical table, or archived result is read here.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from pet_vns_prediction import association_family, bh_adjust, hc3_fit, paper_settings, run_nested_models
from pet_vns_prediction import core, solver, workflow


class NumericalMethodsTests(unittest.TestCase):
    def test_paper_defaults(self):
        s = paper_settings()
        self.assertEqual(s.seed, 20260909)
        self.assertEqual(s.repeats, 20)
        self.assertEqual(s.inner_folds, 3)
        self.assertEqual(len(s.c_grid) * len(s.l1_grid), 30)
        self.assertEqual([len(s.model_columns()[m]) for m in ("C", "CP", "CPT")], [5, 11, 17])
        self.assertEqual(s.categories["sex"][0], "female")

    def test_solver_exact_repeat_and_independent_solution(self):
        rng = np.random.default_rng(481)
        x = rng.normal(size=(36, 4))
        y = np.r_[np.zeros(18), np.ones(18)]
        rng.shuffle(y)
        with threadpool_limits(limits=1):
            for ratio in (0., .5, 1.):
                with self.subTest(l1_ratio=ratio):
                    fitted = solver.fit(x, y, .5, ratio)
                    repeated = solver.fit(x, y, .5, ratio)
                    np.testing.assert_array_equal(fitted.predict_proba(x), repeated.predict_proba(x))
                    self.assertLessEqual(fitted.numerical_["kkt_inf"], solver.ACCEPT_KKT)
                    self.assertLessEqual(fitted.numerical_["duality_gap_mean"], solver.ACCEPT_GAP)
                    w, independent = solver.independent_slsqp(x, y, .5, ratio)
                    self.assertTrue(independent["success"])
                    self.assertLess(np.max(abs(expit(w[0] + x @ w[1:]) - fitted.predict_proba(x)[:, 1])), 1e-5)
                    self.assertLess(abs(independent["objective_mean"] - fitted.numerical_["objective_mean"]), 1e-10)

    def test_preprocessing_is_training_only(self):
        train = pd.DataFrame({"continuous": [1., 3., np.nan, 7.], "binary": ["no", "yes", "no", "yes"], "constant": [2., 2., 2., 2.]})
        pre = core.FoldPreprocessor(train.columns, {"binary": ("no", "yes")}).fit(train)
        self.assertEqual(pre.fill["continuous"], 3.)
        self.assertEqual(pre.fill["binary"], "no")
        self.assertEqual(pre.mean["continuous"], 3.5)
        self.assertEqual(pre.dropped, ("constant",))
        heldout = pd.DataFrame({"continuous": [9999.], "binary": ["yes"], "constant": [9.]})
        value = pre.transform(heldout)
        self.assertEqual(value.shape, (1, 2))
        self.assertEqual(pre.mean["continuous"], 3.5)

    def test_default_splits_shared_and_complete(self):
        s = paper_settings()
        _, y = core.synthetic_fixture(s, n=60, seed=152)
        plans = core.make_split_plan(y, s)
        self.assertEqual(len(plans), 100)
        self.assertEqual(plans, core.make_split_plan(y, s))
        core.validate_split_plan(plans, y, s.repeats, s.inner_folds)
        broken = [{**p, "inner": [dict(q) for q in p["inner"]]} for p in plans]
        broken[0]["inner"][0]["train"] = broken[0]["inner"][0]["train"] + [broken[0]["test"][0]]
        with self.assertRaises(core.InputRejected):
            core.validate_split_plan(broken, y, s.repeats, s.inner_folds)

    def test_heldout_labels_do_not_change_fit_or_tuning(self):
        s = replace(paper_settings(), repeats=1, c_grid=(.1, .5), l1_grid=(0., .5), profile="synthetic_test")
        x, y = core.synthetic_fixture(s, n=40, seed=885)
        plan = core.make_split_plan(y, s)[0]
        other_y = y.copy()
        other_y.iloc[plan["test"]] = 1 - other_y.iloc[plan["test"]]
        with threadpool_limits(limits=1):
            p, info = workflow.corrected_outer(x, y, s.clinical, s, plan, "C", [])
            q, other = workflow.corrected_outer(x, other_y, s.clinical, s, plan, "C", [])
        np.testing.assert_array_equal(p, q)
        self.assertEqual(info, other)
        self.assertEqual(len(info["numerical_fits"]), 13)
        self.assertTrue(info["exact_selected_fit_repetition"])

    def test_end_to_end_small_synthetic_C_CP_CPT(self):
        s = replace(paper_settings(), repeats=1, c_grid=(.1,), l1_grid=(0.,), profile="synthetic_smoke")
        x, y = core.synthetic_fixture(s, n=40, seed=737)
        result = run_nested_models(x, y, s, timeout_seconds=60)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(set(result["predictions"]), {"C", "CP", "CPT"})
        for matrix in result["predictions"].values():
            self.assertEqual(matrix.shape, (1, 40))
            self.assertTrue(np.isfinite(matrix).all())
        core.assert_training_isolation(result)
        metrics = core.summarize_predictions(result, y, s)
        self.assertTrue(all(0 <= row["auc"] <= 1 for row in metrics.values()))
        intervals = core.paired_bootstrap(result, y, draws=20, seed=19)
        self.assertEqual(intervals["attempted_draws"], 20)
        self.assertGreater(intervals["valid_draws"], 0)
        self.assertIn("CP_minus_C_auc", intervals["intervals"])

    def test_input_rejects_identity_and_misaligned_labels(self):
        s = paper_settings()
        x, y = core.synthetic_fixture(s, n=40, seed=487)
        x["patient_name"] = "not accepted"
        with self.assertRaises(core.InputRejected):
            run_nested_models(x, y, s)
        x = x.drop(columns="patient_name")
        with self.assertRaises(core.InputRejected):
            run_nested_models(x, y.iloc[::-1], s)

    def test_hc3_analytical_two_group_case(self):
        values = np.array([1., 2., 5., 7., 9., 2., 3., 4., 8., 10., 12.])
        group = np.r_[np.zeros(5), np.ones(6)]
        result = hc3_fit(np.c_[np.ones(len(group)), group], values)
        expected_se = np.sqrt(values[group == 0].var(ddof=1)/4 + values[group == 1].var(ddof=1)/5)
        self.assertAlmostEqual(result["beta"][1], values[group == 1].mean() - values[group == 0].mean(), places=12)
        self.assertAlmostEqual(result["se"][1], expected_se, places=12)
        self.assertEqual(result["df"], 9)
        self.assertLess(result["independent_check"]["HC3_cov_max_abs_diff"], 1e-10)

    def test_input_rejects_nonstring_indices_before_fitting(self):
        s = paper_settings()
        x, y = core.synthetic_fixture(s, n=40, seed=487)
        invalid_indices = (
            pd.RangeIndex(len(x)),
            pd.Index(["row"] + list(range(1, len(x)))),
            pd.Index([""] + list(x.index[1:])),
            pd.Index(["   "] + list(x.index[1:])),
        )
        for indices in invalid_indices:
            with self.subTest(index_type=type(indices).__name__):
                numeric_x, numeric_y = x.copy(), y.copy()
                numeric_x.index = numeric_y.index = indices
                with self.assertRaisesRegex(core.InputRejected, "indices must contain nonempty strings only"):
                    run_nested_models(numeric_x, numeric_y, s)
        with self.assertRaises(core.InputRejected):
            run_nested_models(x.to_numpy(), y, s)
        with self.assertRaises(core.InputRejected):
            run_nested_models(x, y.to_numpy(), s)

    def test_bh_complete_family_and_known_case(self):
        np.testing.assert_allclose(bh_adjust([.01, .04, .03, .8], expected_tests=4), [.04, .0533333333333333, .0533333333333333, .8])
        with self.assertRaises(ValueError):
            bh_adjust([.01, .03], expected_tests=6)
        with self.assertRaises(ValueError):
            bh_adjust([.01, np.nan], expected_tests=2)

    def test_six_and_eighteen_test_association_families(self):
        rng = np.random.default_rng(278)
        response = np.r_[np.zeros(20), np.ones(20)]
        features = rng.normal(size=(40, 6)) + response[:, None] * -.3
        cov = rng.normal(size=(40, 4))
        primary = association_family(features, response, {"base": cov[:, :2]}, expected_tests=6)
        self.assertEqual(len(primary), 6)
        self.assertTrue(all(row["family_tests"] == 6 for row in primary))
        extended = association_family(features, response, {"add_a": cov[:, :3], "add_b": cov[:, [0, 1, 3]], "add_both": cov}, expected_tests=18)
        self.assertEqual(len(extended), 18)
        np.testing.assert_allclose([r["q_family"] for r in extended], bh_adjust([r["p"] for r in extended], expected_tests=18))
        with self.assertRaises(ValueError):
            association_family(features, response, {"base": cov}, expected_tests=18)

    def test_candidate_tie_break_preserved(self):
        candidates = [{"C": 1., "l1_ratio": .5, "mean_brier": .2}, {"C": .1, "l1_ratio": 1., "mean_brier": .200000005}, {"C": .1, "l1_ratio": 0., "mean_brier": .200000009}]
        self.assertEqual(core.select_candidate(candidates), candidates[-1])


if __name__ == "__main__":
    unittest.main()
