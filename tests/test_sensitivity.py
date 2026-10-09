"""Artificial-array tests only; no private source module or study data access."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

from pet_vns_prediction import core
from pet_vns_prediction.associations import hc3_fit
from pet_vns_prediction.modeling import paper_settings
from pet_vns_prediction import sensitivity as sensitivity


def artificial_associations(n=72):
    rng = np.random.default_rng(2374)
    y = np.tile([0., 1.], n // 2)
    rng.shuffle(y)
    clinical = rng.normal(size=(n, 5))
    clinical[:, [1, 4]] = rng.integers(0, 2, size=(n, 2))
    scanner = rng.integers(0, 2, n)
    t1 = rng.normal(size=(n, 6))
    original = rng.normal(size=(n, 6)) + y[:, None]*np.linspace(-.3, .2, 6)
    variants = {name: original + (i*.08)*rng.normal(size=original.shape)
                for i, name in enumerate(sensitivity.PROCESSING_VARIANTS)}
    types = rng.integers(0, 3, n)
    etiologies = rng.integers(0, 3, n)
    return variants, y, clinical, scanner, t1, np.c_[types == 1, types == 2], np.c_[etiologies == 1, etiologies == 2]


def source_design(y, clinical, scanner):
    clinical = clinical.copy()
    for j in (0, 2, 3):
        clinical[:, j] = (clinical[:, j]-clinical[:, j].mean())/clinical[:, j].std(ddof=1)
    return np.c_[np.ones(len(y)), y, clinical], np.c_[np.ones(len(y)), y, clinical, scanner]


class SensitivityTests(unittest.TestCase):
    def test_twelve_test_joint_family_matches_independent_design(self):
        variants, y, clinical, scanner, t1, _, _ = artificial_associations()
        rows = sensitivity.clinical_morphometric_family(variants["original"], y, clinical, scanner, t1)
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["adjustment"] for r in rows}, set(sensitivity.MORPHOMETRIC_ADJUSTMENTS))
        clinical_design, base = source_design(y, clinical, scanner)
        pvalues = []
        for i in range(6):
            for j, design in enumerate((clinical_design, np.c_[base, stats.zscore(t1[:, i], ddof=1)])):
                fit = hc3_fit(design, variants["original"][:, i])
                row = rows[2*i+j]
                self.assertAlmostEqual(row["difference"], fit["beta"][1], places=13)
                self.assertAlmostEqual(row["p"], fit["p"][1], places=13)
                pvalues.append(fit["p"][1])
        np.testing.assert_allclose([r["q_family"] for r in rows], stats.false_discovery_control(pvalues))
        self.assertEqual({r["family_tests"] for r in rows}, {12})

    def test_all_96_variants_and_adjustments_are_retained(self):
        variants, y, clinical, scanner, t1, _, _ = artificial_associations()
        result = sensitivity.processing_association_family(variants, y, clinical, scanner, t1, influence=False)
        rows = result["rows"]
        self.assertEqual(len(rows), 96)
        self.assertEqual({r["variant"] for r in rows}, set(sensitivity.PROCESSING_VARIANTS))
        self.assertEqual(sum(r["variant"] in ("pvc4", "pvc6") for r in rows), 24)
        self.assertEqual({r["family_tests"] for r in rows}, {96})
        np.testing.assert_allclose([r["q_family"] for r in rows], stats.false_discovery_control([r["p"] for r in rows]))
        _, base = source_design(y, clinical, scanner)
        for r in rows:
            i = list(paper_settings().pet).index(r["region"])
            design = base if r["adjustment"] == "base" else np.c_[base, stats.zscore(t1[:, i], ddof=1)]
            fit = hc3_fit(design, variants[r["variant"]][:, i])
            self.assertAlmostEqual(r["p"], fit["p"][1], places=13)
        with self.assertRaises(ValueError):
            sensitivity.processing_association_family({k: v for k, v in variants.items() if k != "pvc4"}, y, clinical, scanner, t1)

    def test_followup_keeps_separate_18_96_and_joint_114(self):
        arguments = artificial_associations()
        result = sensitivity.followup_association_families(*arguments, influence=False)
        self.assertEqual(len(result["rows"]), 114)
        self.assertEqual(result["family_test_counts"], {"clinical": 18, "processing": 96})
        self.assertEqual(len(result["diagnostics"]), 0)
        rows = result["rows"]
        for family, size in (("clinical", 18), ("processing", 96)):
            subset = [r for r in rows if r["family"] == family]
            self.assertEqual(len(subset), size)
            np.testing.assert_allclose([r["q_family"] for r in subset], stats.false_discovery_control([r["p"] for r in subset]))
        np.testing.assert_allclose([r["q_all114"] for r in rows], stats.false_discovery_control([r["p"] for r in rows]))

    def test_influence_uses_fixed_full_sample_scale_and_no_exclusions(self):
        variants, y, clinical, scanner, _, _, _ = artificial_associations()
        _, design = source_design(y, clinical, scanner)
        value = variants["original"][:, 0]
        result = sensitivity.leave_one_person_influence(design, value, y)
        positive, negative = value[y == 1], value[y == 0]
        sd = np.sqrt(((len(positive)-1)*positive.var(ddof=1) + (len(negative)-1)*negative.var(ddof=1))/(len(y)-2))
        expected = [np.linalg.lstsq(np.delete(design, k, 0), np.delete(value, k), rcond=None)[0][1]/sd for k in range(len(y))]
        np.testing.assert_array_equal(expected, result["leave_one_out_standardized_differences"])
        self.assertEqual(result["case_exclusions"], [])
        self.assertEqual(result["leave_one_out_standardized_range"], [min(expected), max(expected)])
        self.assertEqual(result["full_sample_pooled_SD"], sd)

    def test_all_114_diagnostics_are_available(self):
        with threadpool_limits(limits=1):
            result = sensitivity.followup_association_families(*artificial_associations())
        self.assertEqual(len(result["diagnostics"]), 114)
        self.assertTrue(all(record["case_exclusions"] == [] for record in result["diagnostics"]))

    def test_no_silent_complete_case_selection_or_dummy_recoding(self):
        variants, y, clinical, scanner, t1, types, etiologies = artificial_associations()
        incomplete = clinical.copy()
        incomplete[2, 0] = np.nan
        with self.assertRaises(ValueError):
            sensitivity.clinical_morphometric_family(variants["original"], y, incomplete, scanner, t1)
        wrong = np.asarray(types, float)
        wrong[0] = [1, 1]
        with self.assertRaises(ValueError):
            sensitivity.extended_clinical_family(variants["original"], y, clinical, scanner, wrong, etiologies)

    def test_processing_selection_requires_complete_two_metric_grid(self):
        rows = [{"C": 1., "l1_ratio": .5, "mean_brier": .21, "mean_auc": .99},
                {"C": .1, "l1_ratio": .5, "mean_brier": .21+1e-10, "mean_auc": .5},
                {"C": .1, "l1_ratio": 0., "mean_brier": .21+2e-10, "mean_auc": .4}]
        self.assertIs(sensitivity.select_processing_candidate(rows), rows[2])
        with self.assertRaises(core.FitFailure):
            sensitivity.select_processing_candidate([{**rows[0], "mean_auc": np.nan}])

    def test_heldout_labels_do_not_change_processing_fits(self):
        s = replace(paper_settings(), repeats=1, c_grid=(.01, .1), l1_grid=(0., .5), profile="synthetic_test")
        x, y = core.synthetic_fixture(s, n=40, seed=119)
        plan = core.make_split_plan(y, s)[0]
        definition = {"columns": list(s.model_columns()["CP"]), "tuning": "brier"}
        alternate = y.to_numpy().copy()
        alternate[plan["test"]] = 1-alternate[plan["test"]]
        with threadpool_limits(limits=1):
            first = sensitivity.run_processing_outer(x, y.to_numpy(), definition, s, plan)
            second = sensitivity.run_processing_outer(x, alternate, definition, s, plan)
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertEqual(first["candidates"], second["candidates"])
        self.assertTrue(first["exact_selected_repetition"])
        self.assertLessEqual(first["independent_selected"]["train_probability_max_difference"], 1e-5)
        self.assertLessEqual(first["independent_selected"]["test_probability_max_difference"], 1e-5)
        self.assertEqual(len(first["numerical_fits"]), 13)
        self.assertTrue(all(not set(a["fit_rows"]) & set(plan["test"]) for a in first["audit"]))

    def test_processing_refuses_abandoned_transformer_branches(self):
        s = paper_settings()
        for definition in ({"pca_columns": list(s.pet), "tuning": "brier"},
                           {"columns": list(s.pet), "tuning": "auc"}):
            with self.assertRaises(ValueError):
                sensitivity.PlainColumnsTransformer(definition, s)

    def test_full_processing_run_uses_distinct_engine_and_shared_splits(self):
        s = replace(paper_settings(), repeats=1, c_grid=(.01,), l1_grid=(0.,), profile="synthetic_test")
        x, y = core.synthetic_fixture(s, n=40, seed=222)
        variants = {name: x.loc[:, list(s.pet)].copy()+i*.001 for i, name in enumerate(sensitivity.PREDICTION_VARIANTS)}
        original = x.copy(deep=True)
        # A call through the primary model engine is an error, not an equivalent fallback.
        with patch("pet_vns_prediction.workflow.corrected_outer", side_effect=AssertionError("wrong engine")):
            result = sensitivity.run_processing_predictions(x, variants, y, s)
        self.assertEqual(set(result["predictions"]), set(sensitivity.processing_model_definitions(s)))
        self.assertEqual(len(result["fold_records"]), 70)
        self.assertTrue(all(v.shape == (1, 40) and np.isfinite(v).all() for v in result["predictions"].values()))
        self.assertEqual(len(set(result["model_split_digests"].values())), 1)
        core.assert_training_isolation(result)
        pd.testing.assert_frame_equal(original, x)
        pairs = result["comparisons"]
        self.assertEqual(len(pairs), 28)
        self.assertEqual(len({(p["a"], p["b"]) for p in pairs}), 28)
        self.assertIn({"a": "pvc4_CP", "b": "same_seg_noPVC_CP", "seed": 20260910}, pairs)
        self.assertIn({"a": "pvc6_CPT", "b": "pvc6_CP", "seed": 20260910}, pairs)

    def test_processing_variant_alignment_is_strict(self):
        s = replace(paper_settings(), repeats=1, c_grid=(.01,), l1_grid=(0.,), profile="synthetic_test")
        x, y = core.synthetic_fixture(s, n=40)
        variants = {name: x.loc[:, list(s.pet)].copy() for name in sensitivity.PREDICTION_VARIANTS}
        variants["pvc4"] = variants["pvc4"].iloc[::-1]
        with self.assertRaises(core.InputRejected):
            sensitivity.run_processing_predictions(x, variants, y, s)
        with self.assertRaises(core.InputRejected):
            sensitivity.run_processing_predictions(x, {k: v for k, v in variants.items() if k != "pvc4"}, y, s)


if __name__ == "__main__":
    unittest.main()
