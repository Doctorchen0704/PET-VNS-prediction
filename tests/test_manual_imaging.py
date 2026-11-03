"""Artificial unit tests for the independent manual imaging entrypoints."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import step_support


def module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class ManualImagingTests(unittest.TestCase):
    def test_registration_fixed_options_and_native_resampling(self):
        m = module('STEP03_PET_T1_registration/01_register_pet.py', 'manual_register')
        result = m.commands(Path('native.nii.gz'), Path('orig.mgz'), Path('mask.mgz'), Path('fresh'))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][result[0].index('--dof')+1], '6')
        self.assertEqual(result[0][result[0].index('--seed')+1], '20260731')
        self.assertEqual(result[0][result[0].index('--sat')+1], '99.9')
        self.assertEqual(result[1][result[1].index('--mov')+1], 'native.nii.gz')

    def test_roi82_fixed_order_and_values(self):
        import numpy as np
        m = module('STEP04_feature_extraction/01_extract_pet.py', 'manual_extract')
        rc = json.loads((ROOT/'configs/roi82.json').read_text())
        fc = json.loads((ROOT/'configs/features.json').read_text())
        ids = [rc['bilateral_pairs'][p][s] for p in sorted(rc['bilateral_pairs']) for s in ('left','right')]
        labels = np.repeat(ids, 30).reshape(-1,1,1)
        pet = np.repeat(np.arange(1,83,dtype=float),30).reshape(-1,1,1)
        names, values, details = m.extract_roi82(pet, labels, rc, fc)
        self.assertEqual(len(names), 82)
        self.assertEqual(names[:2], ['L_accumbens','R_accumbens'])
        self.assertEqual(values, np.arange(1,83,dtype=float).tolist())
        self.assertTrue(all(v['status']=='pass' for v in details.values()))

    def test_roi82_missing_not_imputed(self):
        import numpy as np
        m = module('STEP04_feature_extraction/01_extract_pet.py', 'manual_extract_missing')
        rc = json.loads((ROOT/'configs/roi82.json').read_text())
        fc = json.loads((ROOT/'configs/features.json').read_text())
        labels = np.zeros((12,12,12), dtype=int)
        _, values, _ = m.extract_roi82(np.ones(labels.shape), labels, rc, fc)
        self.assertTrue(all(v is None for v in values))

    def test_selection_does_not_make_automatic_review_decision(self):
        m = module('STEP01_dicom2bids/02_prepare_selection.py', 'manual_select')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'selected.tsv'
            path.write_text('source_nii\tsource_json\ttarget_nii\treview_pass\nanything\tanything\tanything\tHOLD\n')
            with self.assertRaises(ValueError):
                m.make_selection(path, root)

    def test_selection_hashes_only_explicit_pair(self):
        m = module('STEP01_dicom2bids/02_prepare_selection.py', 'manual_select_hash')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'exact.nii.gz').write_bytes(b'artificial-not-an-image')
            (root/'exact.json').write_text('{}')
            path = root/'selected.tsv'
            path.write_text('source_nii\tsource_json\ttarget_nii\treview_pass\nexact.nii.gz\texact.json\tsub-artificial/anat/sub-artificial_T1w.nii.gz\tPASS_WITH_ATTENTION\n')
            out = m.make_selection(path, root)
            self.assertEqual(len(out['rows']), 1)
            self.assertTrue(out['rows'][0]['review_pass'])
            self.assertEqual(len(out['rows'][0]['nii_sha256']), 64)

    def test_metadata_patch_rejects_duplicate_or_nonfinite(self):
        m = module('STEP01_dicom2bids/02_prepare_selection.py', 'manual_select_json')
        for text in ('{"Units":"x","Units":"y"}', '{"x":NaN}', '[1]'):
            with self.assertRaises(ValueError):
                m.finite_json_object(text)


if __name__ == '__main__':
    unittest.main()
