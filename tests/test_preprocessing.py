"""Artificial geometry/coefficients only; external imaging commands never run."""
from pathlib import Path
import json
from uuid import uuid4
import unittest

import numpy as np
from pet_vns_prediction import preprocessing as p


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parents[1] / "configs" / "pvc_regions.json").read_text())
        self.ids = sorted({v[h] for v in self.config["bilateral_pairs"].values() for h in ("left", "right")})
        self.beta = np.linspace(0.6, 1.8, len(self.ids))
        self.volumes = {k: float(i + 11) for i, k in enumerate(self.ids)}

    def extract(self, beta=None, config=None):
        b = self.beta if beta is None else beta
        return p.sgtm_features(self.ids, b, self.volumes, self.config if config is None else config,
                               xtx=np.eye(len(self.ids)), xty=b)

    def test_source_command_options_and_no_execution(self):
        out = Path(__file__).parents[1] / ("not_created_" + uuid4().hex)
        self.assertFalse(out.exists())
        self.assertEqual(p.recon_all_command("t1.nii.gz", "SIM001", cw256=True),
                         ["recon-all", "-i", "t1.nii.gz", "-s", "SIM001", "-all", "-cw256", "-parallel", "-openmp", "8"])
        reg = p.registration_command("p", "t", "b", "l", "v", "c")
        self.assertEqual(reg[reg.index("--dof") + 1], "6")
        self.assertEqual(reg[reg.index("--seed") + 1], "20260731")
        self.assertEqual(reg[reg.index("--sat") + 1], "99.9")
        grid = p.analysis_grid_commands("native", "t1", "mask", "seg", "accepted", out)
        self.assertEqual(len(grid), 4)
        self.assertEqual(grid[1][2], "native")
        self.assertEqual([c[-1] for c in grid], ["cubic", "--trilin", "--nearest", "--nearest"])
        pvc = p.pvc_command("p", "s", "c", "l", out)
        self.assertEqual(pvc[-2:], ["--psf", "5.4"])
        self.assertIn("--no-rescale", pvc)
        self.assertEqual(p.pvc_command("p", "s", "c", "l", out, fwhm_mm=None)[-1], "--no-pvc")
        self.assertFalse(out.exists())

    def test_argument_validation(self):
        for subject in ("-bad", "../private", "", "a\nb"):
            with self.assertRaises(ValueError):
                p.recon_all_command("t", subject)
        with self.assertRaises(ValueError):
            p.pvc_command("p", "s", "c", "l", "out", fwhm_mm=-1)
        with self.assertRaises(ValueError):
            p.registration_command("p", "t", "b", "l", "v", "c", threads=0)

    def test_geometry_original_equations(self):
        pet = np.diag([2., 2., 2., 1.])
        anat = np.eye(4)
        seg = np.diag([.5, .5, .5, 1.])
        seg[:3, 3] = [3., 4., 5.]
        vox, gmat = np.linalg.inv(pet) @ seg, np.linalg.inv(seg) @ anat
        result = p.validate_pvc_geometry(pet, anat, seg, (1, np.eye(4)), (0, vox), (0, gmat))
        self.assertEqual(result["voxel_error"], 0)
        with self.assertRaises(ValueError):
            p.validate_pvc_geometry(pet, anat, seg, (1, np.eye(4)), (0, np.eye(4)), (0, gmat))
        text = "type = 1 # LINEAR_RAS_TO_RAS\n1 4 4\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
        kind, matrix = p.parse_lta(text)
        self.assertEqual(kind, 1)
        np.testing.assert_array_equal(matrix, np.eye(4))

    def test_gtm_stats_contract(self):
        ids, values = p.parse_gtm_stats("1 10 L 1 2 3 0.75 9\n2 49 R 1 2 3 1.25 9\n")
        self.assertEqual(ids, [10, 49])
        np.testing.assert_array_equal(values, [.75, 1.25])
        for text in ("1 2 3", "1 10 L 1 2 3 0.75 9\n2 10 R 1 2 3 1.25 9"):
            with self.assertRaises(ValueError):
                p.parse_gtm_stats(text)

    def test_full_reference_volume_weighting(self):
        result = self.extract()
        self.assertEqual(len(result["nodes_normalized"]), 42)
        expected = sum(v * self.volumes[k] for k, v in zip(self.ids, self.beta)) / sum(self.volumes.values())
        self.assertEqual(result["reference_raw"], expected)
        pair = self.config["bilateral_pairs"]["thalamus"]
        vals = dict(zip(self.ids, self.beta))
        thalamus = sum(vals[pair[h]] * self.volumes[pair[h]] for h in ("left", "right")) / sum(self.volumes[pair[h]] for h in ("left", "right")) / expected
        self.assertEqual(result["pathways_normalized"]["vns_thalamus"], thalamus)
        self.assertEqual(result["independent_solver_relative_L2_error"], 0)
        self.assertEqual(result["numerical_feature_flag"], "PASS_NUMERICAL_CHECKS")
        primary_only = dict(self.config)
        nodes = {n for names in self.config["primary_features"].values() for n in names}
        primary_only["bilateral_pairs"] = {n: v for n, v in self.config["bilateral_pairs"].items() if n in nodes}
        with self.assertRaises(ValueError):
            self.extract(config=primary_only)

    def test_negative_coefficients_retained_and_flagged(self):
        beta = self.beta.copy()
        beta[0] = -.2
        result = self.extract(beta)
        self.assertEqual(result["region_coefficients"][str(self.ids[0])], -.2)
        self.assertIn(self.ids[0], result["nonpositive_reference_gm_labels"])
        self.assertEqual(result["numerical_feature_flag"], "ATTENTION")

    def test_rounding_and_normal_equation_checks(self):
        with self.assertRaises(ValueError):
            p.sgtm_features(self.ids, self.beta, self.volumes, self.config,
                            xtx=np.eye(84), xty=self.beta, rounded_stats_values=self.beta + .1)
        result = p.sgtm_features(self.ids, self.beta, self.volumes, self.config,
                                 xtx=np.eye(84), xty=self.beta + .1)
        self.assertEqual(result["numerical_feature_flag"], "ATTENTION")

    def test_segmentation_volume_units(self):
        volumes = p.segmentation_volumes(np.array([[[10, 10, 49]]]), [.5, .5, .5])
        self.assertEqual(volumes, {10: .25, 49: .125})


if __name__ == "__main__":
    unittest.main()
