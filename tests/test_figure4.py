"""Artificial aggregate rows and deliberately non-anatomical image fixtures."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw
from pet_vns_prediction import figure4


def artificial_inputs(root):
    rng = np.random.default_rng(7701)
    assets = {}
    for i, key in enumerate(figure4.ASSET_KEYS):
        path = root / (key + ".png")
        image = Image.new("RGB", (260, 180), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((25, 35, 235, 140), fill="#ddd6e8" if i < 4 else "#d5c3e0")
        draw.text((75, 80), "SYNTHETIC", fill="#333333")
        image.save(path)
        assets[key] = path
    data = {"cohort_order": ["Cohort A", "Cohort B", "Cohort C"],
            "PC1_color_limits": [-.4, .4], "SERT_color_limits": [-2, 2],
            "color_ticks": {"PC1": [-.4, 0, .4], "SERT": [-2, 0, 2]},
            "scatter_data": [], "interval_plot_rows": []}
    for condition in ("baseline", "DAT"):
        x = rng.normal(0, 10, 68)
        y = .4*x + rng.normal(0, 8, 68)
        data["scatter_data"].append({"condition": condition,
            "saved_effect": float(np.corrcoef(x,y)[0,1]),
            "points": [{"parcel": f"fake_parcel_{i:02}", "x": float(a), "y": float(b)}
                       for i,(a,b) in enumerate(zip(x,y))]})
    for m in figure4.MAPS:
        for i,c in enumerate(data["cohort_order"]):
            for j,a in enumerate(figure4.CONDITIONS):
                estimate = .35-.2*i-.1*j
                data["interval_plot_rows"].append({"cohort":c,"n":30+10*i,
                    "map":m,"adjustment":a,"estimate":estimate,
                    "ci95":[estimate-.2,estimate+.2]})
    scales = {"PC1":data["PC1_color_limits"], "SERT":data["SERT_color_limits"]}
    return data, assets, scales


class Figure4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data, self.assets, self.scales = artificial_inputs(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def create(self, data=None, **kw):
        return figure4.create_figure(self.data if data is None else data, self.assets,
                    asset_color_limits=self.scales, assets_authorized=True, **kw)

    def test_complete_layout_no_data_mutation(self):
        before = deepcopy(self.data)
        fig, receipt = self.create()
        try:
            self.assertEqual(self.data, before)
            self.assertEqual(len(fig.axes), 14)
            self.assertIn("Within-cohort SERT associations", [t.get_text() for t in fig.texts])
            self.assertEqual(receipt["visual_review_status"], "NOT_PERFORMED")
            self.assertFalse(receipt["anatomical_mapping_verified_by_renderer"])
            np.testing.assert_array_equal(fig.axes[10].collections[0].get_offsets()[:,0],
                [p["x"] for p in self.data["scatter_data"][0]["points"]])
        finally:
            fig.clear()

    def test_no_silent_data_or_asset_substitution(self):
        bad = deepcopy(self.data)
        bad["interval_plot_rows"][1] = deepcopy(bad["interval_plot_rows"][0])
        with self.assertRaises(ValueError): self.create(bad)
        bad = deepcopy(self.data)
        bad["interval_plot_rows"][2]["n"] = 999
        with self.assertRaises(ValueError): self.create(bad)
        bad = deepcopy(self.data)
        bad["scatter_data"][0]["saved_effect"] = -.99
        with self.assertRaises(ValueError): self.create(bad)
        bad = deepcopy(self.data)
        bad["scatter_data"][0]["points"][0]["x"] = 1000
        with self.assertRaises(ValueError): self.create(bad)
        bad = deepcopy(self.data)
        bad["PC1_color_limits"] = [-.8,.8]
        with self.assertRaises(ValueError): self.create(bad)
        with self.assertRaises(ValueError):
            figure4.create_figure(self.data,self.assets,asset_color_limits=self.scales)

    def test_fresh_output_and_rgb_tiff(self):
        output = self.root / "render"
        receipt = figure4.render_figure4(self.data,self.assets,output,
                        asset_color_limits=self.scales,assets_authorized=True,dpi=36,preview_dpi=36)
        self.assertEqual(receipt["status"], "RENDERED_PENDING_VISUAL_REVIEW")
        with Image.open(output / "Figure4.tiff") as im:
            self.assertEqual(im.mode,"RGB")
            self.assertEqual(im.size,(396,288))
            self.assertEqual(tuple(im.info["dpi"]),(36,36))
        with self.assertRaises(FileExistsError):
            figure4.render_figure4(self.data,self.assets,output,asset_color_limits=self.scales,
                                   assets_authorized=True,dpi=36)
        self.assertEqual(json.loads((output/"render_status.json").read_text())["status"],
                         "RENDERED_PENDING_VISUAL_REVIEW")


if __name__ == "__main__":
    unittest.main()
