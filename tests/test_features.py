"""Synthetic-only feature tests; no participant images or tables are loaded."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pet_vns_prediction.features import (
    extract_pet_features, extract_pet_nifti, extract_t1_features,
    extract_t1_stats, load_feature_config, normalization_masks, normalize_pet,
    parse_aparc, parse_aseg, roi_stats, worst_status,
)


CONFIG = load_feature_config(Path(__file__).resolve().parents[1] / "configs" / "features.json")


def synthetic_pet():
    labels, values = [], []
    for index, pair in enumerate(CONFIG["bilateral_pairs"].values()):
        labels.extend([pair["left"]] * 25 + [pair["right"]] * 50)
        values.extend([index * 2 + 1.] * 25 + [index * 2 + 4.] * 50)
    return np.array(values).reshape(15, 19, 5), np.array(labels).reshape(15, 19, 5)


def synthetic_t1():
    aseg, aparc = {}, {"lh": {}, "rh": {}}
    for feature in CONFIG["t1_features"]:
        for label in feature.get("aseg_structures", []):
            aseg[label] = 1000.
        for label in feature.get("aparc_parcels", []):
            aparc["lh"][label] = {"surface_area_mm2": 100., "thickness_mm": 2.}
            aparc["rh"][label] = {"surface_area_mm2": 300., "thickness_mm": 4.}
    return aseg, aparc


class FeatureTests(unittest.TestCase):
    def test_config_has_nineteen_unique_primary_nodes(self):
        nodes = [name for group in CONFIG["primary_features"].values() for name in group]
        self.assertEqual(len(nodes), 19)
        self.assertEqual(set(nodes), set(CONFIG["bilateral_pairs"]))
        self.assertEqual(len(set(nodes)), 19)
        self.assertEqual(len(CONFIG["primary_features"]), 6)

    def test_bilateral_voxel_pool_and_equal_node_weights(self):
        pet, labels = synthetic_pet()
        result = extract_pet_features(pet, labels, CONFIG)
        expected = {name: index * 2 + 3. for index, name in enumerate(CONFIG["bilateral_pairs"])}
        for name, value in expected.items():
            self.assertEqual(result["pair_results"][name]["bilateral"]["mean"], value)
        for name, nodes in CONFIG["primary_features"].items():
            self.assertEqual(result["primary_features"][name]["value"], np.mean([expected[x] for x in nodes]))
        self.assertEqual(result["overall_primary_feature_qc_status"], "pass")
        self.assertAlmostEqual(result["pair_results"]["thalamus"]["signed_asymmetry_index"], -1.2)

    def test_roi_validity_and_count_boundaries(self):
        def summary(values):
            array = np.asarray(values, dtype=float)
            return roi_stats(array, np.ones(array.shape, dtype=bool), 10, 24, .95, .8)
        self.assertEqual(summary([1.] * 9)["status"], "missing_label_below_hard_minimum")
        self.assertEqual(summary([1.] * 10)["status"], "attention_low_label_voxels")
        self.assertEqual(summary([1.] * 24)["status"], "attention_low_label_voxels")
        self.assertEqual(summary([1.] * 25)["status"], "pass")
        self.assertEqual(summary([2.] * 19 + [0.])["valid_pet_fraction"], .95)
        self.assertEqual(summary([2.] * 24 + [0.] * 6)["status"], "attention_valid_pet_fraction")
        self.assertEqual(summary([2.] * 23 + [0.] * 7)["mean"], None)
        value = summary([2.] * 26 + [float("nan"), float("inf"), -1., 0.])
        self.assertEqual(value["mean"], 2.)
        self.assertEqual(value["valid_pet_voxels"], 26)

    def test_missing_constituent_not_silently_dropped(self):
        pet, labels = synthetic_pet()
        pet[(labels == 17) | (labels == 53)] = 0
        result = extract_pet_features(pet, labels, CONFIG)
        self.assertIsNone(result["primary_features"]["vns_limbic_medial_temporal"]["value"])
        self.assertEqual(result["overall_primary_feature_qc_status"], "missing")

    def test_hemisphere_flag_is_retained_when_bilateral_mean_exists(self):
        pet, labels = synthetic_pet()
        labels[labels == 10] = 0
        result = extract_pet_features(pet, labels, CONFIG)
        # Source behavior: bilateral value can exist, but hemisphere QC is missing.
        self.assertEqual(result["primary_features"]["vns_thalamus"]["value"], 4.)
        self.assertEqual(result["primary_features"]["vns_thalamus"]["status"], "missing")
        self.assertEqual(worst_status(["pass", "attention_low_label_voxels"]), "attention")

    def test_reference_masks_include_zeros_and_diencephalon(self):
        labels = np.array([1002, 10, 28, 8, 47, 2, 16, 0]).reshape(2, 2, 2)
        pet = np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.float32).reshape(2, 2, 2)
        brain = np.ones_like(pet)
        masks = normalization_masks(pet, labels, brain, CONFIG)
        self.assertEqual(int(masks["supraGM"].sum()), 3)
        self.assertEqual(int(masks["wholeBrain"].sum()), 8)
        result, qc = normalize_pet(pet, labels, brain, "supraGM", CONFIG)
        self.assertEqual(qc["reference_means"]["supraGM"], 2.)
        self.assertEqual(qc["reference_means"]["cerebellarCortex"], 7.)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, pet / 2.)
        self.assertFalse(qc["reference_intensity_gate_pass"])  # deliberately tiny synthetic image

    def test_validation_rejects_bad_inputs(self):
        pet, labels = synthetic_pet()
        with self.assertRaises(ValueError):
            extract_pet_features(pet, labels.astype(float) + .1, CONFIG)
        with self.assertRaises(ValueError):
            extract_pet_features(pet[:, :, 0], labels[:, :, 0], CONFIG)
        with self.assertRaises(ValueError):
            normalize_pet(-pet, labels, np.ones_like(pet), "supraGM", CONFIG)
        with self.assertRaises(ValueError):
            normalize_pet(pet, labels, np.ones_like(pet), "cerebellarCortex", CONFIG)

    def test_t1_actual_volume_and_area_weighting_definitions(self):
        aseg, aparc = synthetic_t1()
        result = extract_t1_features(1_000_000., aseg, aparc, CONFIG)
        self.assertEqual(result["t1_thalamus_volume"], .002)
        self.assertEqual(result["t1_medial_temporal_volume"], .004)
        self.assertEqual(result["t1_striatopallidal_volume"], .006)
        for name in ("t1_cingulate_thickness", "t1_insula_operculum_thickness", "t1_sensorimotor_thickness"):
            self.assertEqual(result[name], 3.5)
        # Extra structures are not silently added to these source-defined composites.
        aseg["Left-Accumbens-area"] = 90000.
        aseg["Right-Accumbens-area"] = 90000.
        self.assertEqual(extract_t1_features(1_000_000., aseg, aparc, CONFIG), result)

    def test_t1_missing_and_nonpositive_components_raise(self):
        aseg, aparc = synthetic_t1()
        missing = dict(aseg)
        del missing["Left-Thalamus"]
        with self.assertRaises(ValueError):
            extract_t1_features(1_000_000., missing, aparc, CONFIG)
        with self.assertRaises(ValueError):
            extract_t1_features(0., aseg, aparc, CONFIG)
        bad = copy.deepcopy(aparc)
        bad["lh"]["insula"]["surface_area_mm2"] = 0
        with self.assertRaises(ValueError):
            extract_t1_features(1_000_000., aseg, bad, CONFIG)

    def test_stats_parsers_on_synthetic_text(self):
        aseg, aparc = synthetic_t1()
        with tempfile.TemporaryDirectory(prefix="synthetic_feature_test_", dir=Path(__file__).resolve().parent) as directory:
            directory = Path(directory)
            aseg_path, left, right = directory / "aseg.stats", directory / "lh.aparc.stats", directory / "rh.aparc.stats"
            aseg_path.write_text(
                "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1000000.0, mm^3\n"
                + "".join(f"{i} {i} 100 {v} {name} 0 0\n" for i, (name, v) in enumerate(aseg.items())),
                encoding="utf-8",
            )
            for hemisphere, path in (("lh", left), ("rh", right)):
                path.write_text("# synthetic aparc only\n" + "".join(
                    f"{name} 100 {item['surface_area_mm2']} 1000 {item['thickness_mm']} 0\n"
                    for name, item in aparc[hemisphere].items()), encoding="utf-8")
            self.assertEqual(parse_aseg(aseg_path), (1_000_000., aseg))
            self.assertEqual(parse_aparc(left), aparc["lh"])
            self.assertEqual(extract_t1_stats(aseg_path, left, right, CONFIG), extract_t1_features(1_000_000., aseg, aparc, CONFIG))

    def test_nifti_adapter_and_geometry_guard(self):
        try:
            import nibabel as nib
        except ImportError:
            self.skipTest("nibabel optional image dependency is unavailable")
        pet, labels = synthetic_pet()
        with tempfile.TemporaryDirectory(prefix="synthetic_feature_test_", dir=Path(__file__).resolve().parent) as directory:
            directory = Path(directory)
            pet_path, seg_path = directory / "synthetic_pet.nii.gz", directory / "synthetic_labels.nii.gz"
            affine = np.diag([2., 2., 2., 1.])
            nib.save(nib.Nifti1Image(pet.astype(np.float32), affine), pet_path)
            nib.save(nib.Nifti1Image(labels.astype(np.int16), affine), seg_path)
            self.assertEqual(extract_pet_nifti(pet_path, seg_path, CONFIG), extract_pet_features(pet, labels, CONFIG))
            affine[0, 3] = 1.
            nib.save(nib.Nifti1Image(labels.astype(np.int16), affine), seg_path)
            with self.assertRaises(ValueError):
                extract_pet_nifti(pet_path, seg_path, CONFIG)


if __name__ == "__main__":
    unittest.main()
