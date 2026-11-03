"""Artificial-data tests only; no study inputs, patient identifiers or downloads."""
import unittest

import numpy as np
from scipy.linalg import block_diag, null_space
from scipy.stats import rankdata

from pet_vns_prediction import spatial, spatial_tle, vns_pattern, vns_spatial
from pet_vns_prediction import atlas_specificity, dat_conditioning, molecular_annotation


class SpatialExtensionTests(unittest.TestCase):
    def test_external_engine_synthetic_controls(self):
        self.assertEqual(spatial.synthetic_tests()["status"], "PASS")

    def test_tle_kernel_synthetic_controls(self):
        self.assertEqual(spatial_tle.synthetic_tests()["status"], "PASS")

    def test_atlas_kernel_synthetic_controls(self):
        self.assertEqual(atlas_specificity.synthetic_tests()["status"], "PASS")

    def test_dat_kernel_synthetic_controls(self):
        self.assertEqual(dat_conditioning.synthetic_tests()["status"], "PASS")

    def test_original_vns_pca_bootstrap(self):
        rng = np.random.default_rng(12)
        raw = np.exp(rng.normal(size=(16, 82)))
        groups = np.array(["scanner_a"] * 10 + ["scanner_b"] * 6)
        z = vns_pattern.clr(raw)
        fit = vns_pattern.fit(z, np.ones((len(z), 1)))
        reference = fit["loadings"][:, 0]
        out = vns_pattern.scanner_stratified_bootstrap(raw, groups, reference, n_boot=11)
        rng = np.random.default_rng(20261025)
        strata = [np.flatnonzero(groups == level) for level in sorted(set(groups))]
        expected = np.array([np.concatenate([rng.choice(s, len(s), replace=True) for s in strata]) for _ in range(11)])
        np.testing.assert_array_equal(out["row_indices"], expected)
        self.assertTrue(np.all(out["cosine_to_reference"] >= -1e-12))
        for b, indices in enumerate(expected):
            self.assertEqual(int(np.sum(groups[indices] == "scanner_a")), 10)
            m = vns_pattern.fit(z[indices], np.ones((16, 1)))
            v = m["loadings"][:, 0]
            v *= -1. if v @ reference < 0 else 1.
            np.testing.assert_allclose(out["pc1_loadings"][b], v, atol=1e-12, rtol=0)
        with self.assertRaises(ValueError):
            vns_pattern.scanner_stratified_bootstrap(raw, groups, -reference[::-1], n_boot=2)

    def test_original_vns_conditional_equivalence(self):
        rng = np.random.default_rng(21)
        coords = rng.normal(size=(68, 3))
        coords /= np.linalg.norm(coords, axis=1, keepdims=True)
        ap, g1 = rng.normal(size=(2, 68))
        hemi = np.r_[np.ones(34), np.zeros(34)]
        z = vns_spatial.design(ap, g1, hemi)
        w34 = vns_spatial.graph(coords, 5)
        W = block_diag(w34, w34)
        x = rng.normal(size=68)
        receptors = rng.normal(size=(68, 18))
        q = null_space(z.T)
        _, vec = np.linalg.eigh(q.T @ W @ q)
        basis = q @ vec
        signs = np.random.default_rng(20261027).choice([-1., 1.], size=(41, 64))
        d = {"z": z, "yr": rankdata(receptors, axis=0), "W": W,
             "coords": coords, "names": list(spatial.TARGETS)}
        rows, sims, null, _ = vns_spatial.test_family(x, d, basis, signs, "conditional_MSR")
        external = spatial.conditional_msr(x, receptors, z, W, n_null=41, seed=20261027)
        np.testing.assert_allclose(null, external["null_effects"], atol=1e-12, rtol=0)
        np.testing.assert_allclose([r["partial_rho"] for r in rows], external["observed"], atol=1e-12)
        np.testing.assert_allclose(sims @ z, 0, atol=1e-10)

    def test_rank_spin_tie_convention(self):
        x = np.arange(68, dtype=float)
        y = x.copy()
        spins = np.vstack([np.arange(68), np.arange(67, -1, -1), np.roll(np.arange(68), 3)])
        result, null = molecular_annotation.statistics(x, y, spins)
        self.assertAlmostEqual(result["rho"], 1.)
        self.assertAlmostEqual(result["p_spin"], .75)
        self.assertEqual(len(null), 3)

    def test_atlas_full_wrapper_and_dat_metadata(self):
        rng = np.random.default_rng(24)
        x = rng.normal(size=68)
        boots = rng.normal(size=(2000, 68))
        maps = rng.normal(size=(68, 3))
        z = spatial.nuisance_design(rng.normal(size=68), rng.normal(size=68))
        raw_null = rng.normal(size=(9999, 68))
        cond_null = spatial.residualize(rng.normal(size=(68, 9999)), z).T
        full, _ = atlas_specificity.analyze(x, boots, raw_null, cond_null, maps, z)
        bounded, arr = dat_conditioning.compute(x, boots, maps, z, n_participants=16)
        self.assertEqual(bounded["n_participants"], 16)
        self.assertEqual(bounded["spatial_P"], "NOT_COMPUTED_RANK5_EFFECT_ONLY")
        np.testing.assert_allclose([r["partial_Spearman"] for r in full["DAT_conditioned"]["rows"]],
                                   arr["DAT_effects"][0], atol=1e-12, rtol=0)
        self.assertEqual(full["bootstrap_draws"], 2000)


if __name__ == "__main__":
    unittest.main()
