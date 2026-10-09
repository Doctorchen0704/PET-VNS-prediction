"""SYNTHETIC-only numerical and rendering checks; no study files are accessed."""
import copy
from pathlib import Path
import runpy
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pet_vns_prediction.figures import (
    PALETTE, assert_no_clipped_text, export_figure, figure1_workflow, figure2_regional,
    figure3_prediction, figure_s1_validation, figure_s2_processing, fixed_bin_calibration,
    format_estimate, wilson95,
)
from pet_vns_prediction.tables import cohort_descriptives


class FigureTests(unittest.TestCase):
    def setUp(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "synthetic_figures.py"
        self.synthetic = runpy.run_path(str(example))["make_synthetic_inputs"]()
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-figure-test-")
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(lambda: plt.close("all"))

    def test_wilson_fixed_bins_and_boundary_rule(self):
        edges = np.linspace(0, 1, 6)
        result = fixed_bin_calibration([0, 1, 0, 1, 1, 1], edges)
        self.assertEqual([b["n"] for b in result["bins"]], [1, 1, 1, 1, 2])
        self.assertEqual(result["bins"][-1]["mean_probability"], .9)
        self.assertIsNone(wilson95(0, 0))
        np.testing.assert_allclose(wilson95(2, 4), [.15003898915214947, .8499610108478506])
        self.assertEqual(format_estimate(1.2345), "1.235")
        self.assertEqual(format_estimate(-1.2345), "-1.235")
        with self.assertRaises(ValueError):
            fixed_bin_calibration([0, 2], [.1, .8])
        with self.assertRaises(ValueError):
            fixed_bin_calibration([0, 1], [.1, np.nan])

    def test_figure3_preserves_marks_intervals_and_empty_bins(self):
        data = self.synthetic["figure3"]
        data["calibration"] = fixed_bin_calibration([0, 1, 0, 1], [.05, .08, .84, .96])
        before = copy.deepcopy(data)
        fig = figure3_prediction(data, dpi=80, synthetic=True)
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.lines[2].get_xdata(), data["models"]["CP"]["roc_fpr"])
        np.testing.assert_array_equal(ax.lines[2].get_ydata(), data["models"]["CP"]["roc_tpr"])
        self.assertEqual(ax.lines[2].get_color(), PALETTE["CP"])
        self.assertTrue(np.isnan(fig.axes[1].lines[1].get_xdata(orig=False)[1:4]).all())
        first_ablation_interval = fig.axes[3].collections[0].get_segments()[0]
        np.testing.assert_allclose(first_ablation_interval[:, 0], data["ablation"][0]["conditional95"])
        self.assertEqual(data, before)
        assert_no_clipped_text(fig)

    def test_figure3_does_not_silently_clip(self):
        self.synthetic["figure3"]["ablation"][0]["conditional95"] = [-.2, .3]
        with self.assertRaisesRegex(ValueError, "clipped"):
            figure3_prediction(self.synthetic["figure3"])

    def test_s1_saved_rank_ranges_and_s2_family_guard(self):
        s1 = figure_s1_validation(self.synthetic["s1"], dpi=80, synthetic=True)
        self.assertEqual(len(s1.axes[-1].collections), 24)
        self.assertEqual(s1.axes[-1].get_xlim(), (0, 25))
        assert_no_clipped_text(s1)
        original = copy.deepcopy(self.synthetic["s2"])
        s2 = figure_s2_processing(self.synthetic["s2"], dpi=80, synthetic=True)
        self.assertEqual(len(s2.axes), 4)
        self.assertEqual(self.synthetic["s2"], original)
        assert_no_clipped_text(s2)
        self.synthetic["s2"]["q_family_size"] = 72
        with self.assertRaisesRegex(ValueError, "96-test"):
            figure_s2_processing(self.synthetic["s2"])

    def test_figure2_full_points_shared_density_and_q(self):
        data = self.synthetic["figure2"]
        raw_before = data["raw_relative_values"].copy()
        fig = figure2_regional(data, self.synthetic["assets"], dpi=90, synthetic=True)
        np.testing.assert_allclose(fig.get_size_inches(), [6.9, 9.])
        for ax in fig.axes[-2:]:
            point_collections = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
            self.assertEqual(sum(len(c.get_offsets()) for c in point_collections), len(data["y"]))
            np.testing.assert_allclose(ax.get_xlim(), data["shared_distribution_xlim"])
        self.assertEqual(fig.axes[-3].texts[0].get_text(), f"{data['primary'][0]['q_family']:.4f}")
        np.testing.assert_array_equal(data["raw_relative_values"], raw_before)
        assert_no_clipped_text(fig)
        with self.assertRaisesRegex(ValueError, "five licensed"):
            figure2_regional(data, {})

    def test_figure1_counts_and_native_export(self):
        fig = figure1_workflow(self.synthetic["counts"], self.synthetic["stacks"], period="SYNTHETIC interval",
            exclusions=[("Eligibility / QC", 3), ("Matching", 2), ("Protocol", 1)], synthetic=True, dpi=60)
        self.assertEqual(len(fig.axes), 10)
        assert_no_clipped_text(fig)
        directory = Path(self.temporary.name)
        tif = export_figure(fig, directory / "SYNTHETIC.tif", dpi=60)
        with Image.open(tif) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (768, 588))
            self.assertEqual(image.info["compression"], "tiff_lzw")
            self.assertEqual(image.info["dpi"], (60., 60.))
        svg = export_figure(fig, directory / "SYNTHETIC.svg", dpi=60)
        self.assertIn("<text", svg.read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            export_figure(fig, tif)
        with self.assertRaisesRegex(ValueError, "resolution"):
            export_figure(fig, directory / "too_small.tif", dpi=600, minimum_asset_dpi=600)

    def test_descriptive_table_counts_quantiles_and_missing(self):
        rows = [{"value": i, "category": "A" if i < 2 else "B"} for i in range(4)]
        result = cohort_descriptives(rows, [0, 1, 0, 1], continuous=["value"], categorical=["category"])
        self.assertEqual(result["n"], 4)
        total = result["groups"]["all"]
        self.assertAlmostEqual(total["continuous"]["value"]["sd"], np.std([0, 1, 2, 3], ddof=1))
        self.assertEqual(total["continuous"]["value"]["q1"], .75)
        self.assertEqual(total["categorical"]["category"]["A"], {"count": 2, "percent": 50.})
        rows[0]["value"] = None
        with self.assertRaisesRegex(ValueError, "missing"):
            cohort_descriptives(rows, [0, 1, 0, 1], continuous=["value"], categorical=["category"])


if __name__ == "__main__":
    unittest.main()
