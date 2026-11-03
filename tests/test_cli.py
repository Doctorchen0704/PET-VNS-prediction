"""Synthetic fixtures for entrypoint safety, not real-study reproduction."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pet_vns_prediction import cli, core, paper_settings
from pet_vns_prediction.intake import safe_relative, conversion_plan, promote_selected, sha256_file, load_json


def fixture():
    settings = replace(paper_settings(), repeats=2)
    ids = [f'artificial{i:02d}' for i in range(24)]
    y = pd.Series([0, 1] * 12, index=ids)
    x = {key: list(np.linspace(1., 2., 24)) for key in settings.model_columns()['CPT']}
    x['sex'] = ['female', 'male'] * 12
    x['mri_lesional_status'] = ['nonlesional'] * 12 + ['lesional'] * 12
    return settings, {'study_ids': ids, 'x': x, 'y': y.tolist(),
        'columns': {k:list(v) for k,v in settings.model_columns().items()},
        'plans': core.make_split_plan(y, settings),
        'saved_predictions': {'C': np.tile(np.linspace(.2, .8, 24), (2, 1)).tolist()}}


class CliTests(unittest.TestCase):
    def test_summary_does_not_fit_and_labels_scope(self):
        settings, data = fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inp = root/'input.json'; inp.write_text(json.dumps(data))
            original = inp.read_bytes()
            with patch.object(cli, 'paper_settings', return_value=settings), patch.object(cli, 'run_nested_models', side_effect=AssertionError('must not fit')):
                result = cli.predict(inp, root/'summary', models=['C'], summary_only=True)
            self.assertIn('auc', result['C'])
            status = load_json(root/'summary/status.json')
            self.assertEqual(status['method'], 'saved_prediction_summary')
            self.assertFalse(status['model_fit_performed'])
            self.assertEqual(status['saved_predictions_training_provenance'], 'NOT_REVERIFIED')
            self.assertFalse((root/'summary/fit_audit_PRIVATE.json').exists())
            self.assertEqual(inp.read_bytes(), original)
            with patch.object(cli, 'paper_settings', return_value=settings):
                with self.assertRaises(FileExistsError):
                    cli.predict(inp, root/'summary', models=['C'], summary_only=True)

    def test_reject_bad_payload_before_writes(self):
        settings, original = fixture()
        cases = []
        bad=copy.deepcopy(original);bad['y'][0]=2;cases.append(bad)
        bad=copy.deepcopy(original);bad['y'][0]=None;cases.append(bad)
        bad=copy.deepcopy(original);bad['study_ids'][0]='';cases.append(bad)
        bad=copy.deepcopy(original);bad['x']['identity_name']=['private']*24;cases.append(bad)
        bad=copy.deepcopy(original);bad['plans'][0]['repeat']=-1;cases.append(bad)
        bad=copy.deepcopy(original);bad['plans'][0]['test'][0]=0.0;cases.append(bad)
        bad=copy.deepcopy(original);bad['plans'][0]['fold']=bad['plans'][1]['fold'];cases.append(bad)
        bad=copy.deepcopy(original);bad['saved_predictions']['C'][0][0]=1.1;cases.append(bad)
        with tempfile.TemporaryDirectory() as d, patch.object(cli, 'paper_settings', return_value=settings):
            root=Path(d)
            for i,data in enumerate(cases):
                inp=root/f'input{i}.json';inp.write_text(json.dumps(data))
                with self.subTest(i=i), self.assertRaises(ValueError):
                    cli.predict(inp,root/f'output{i}',models=['C'],summary_only=True)
                self.assertFalse((root/f'output{i}').exists())
            with self.assertRaises(ValueError):cli._validated_input(original,['C','C'],settings)
            with self.assertRaises(ValueError):cli._validated_input(original,[],settings)

    def test_failed_run_is_preserved_without_overwrite(self):
        settings,data=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=root/'input.json';inp.write_text(json.dumps(data))
            with patch.object(cli,'paper_settings',return_value=settings), patch.object(cli,'run_nested_models',side_effect=RuntimeError('synthetic injected failure')):
                with self.assertRaises(RuntimeError):cli.predict(inp,root/'failed',models=['C'])
            self.assertEqual(load_json(root/'failed/status.json')['status'],'FAILED_PRESERVED')
            self.assertTrue(inp.exists())

    def test_json_duplicate_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.json'
            for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'):
                p.write_text(text)
                with self.assertRaises(ValueError):load_json(p)

    def test_windows_portable_paths_and_case_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'raw';raw.mkdir();(raw/'one').mkdir();(raw/'one/file').write_bytes(b'fixture')
            for bad in ('NUL/file','CON.txt','one./file','one /file','one//file','one/./file','one/*','one\x00'):
                with self.subTest(path=repr(bad)),self.assertRaises(ValueError):safe_relative(raw,bad)
            row={'subject':'Synthetic01','modality':'PET','source_relative':'one','identity_reviewed':True,'series_reviewed':True}
            m={'schema_version':1,'rows':[row,{**row,'subject':'synthetic01'}]}
            with self.assertRaises(ValueError):conversion_plan(m,raw,root/'out','converter')
            self.assertFalse((root/'out').exists())

    def test_promotion_requires_correct_modality_and_reviewed_patch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stage=root/'stage';stage.mkdir()
            n=stage/'image.nii.gz';n.write_bytes(b'not_real_nifti')
            j=stage/'image.json';j.write_text('{"Units":"BQML"}')
            row={'source_nii':n.name,'source_json':j.name,'nii_sha256':sha256_file(n),'json_sha256':sha256_file(j),
                 'target_nii':'sub-artificial01/anat/sub-artificial01_pet.nii.gz','review_pass':True}
            with self.assertRaises(ValueError):promote_selected({'schema_version':1,'rows':[row]},stage,root/'badmodality')
            row['target_nii']='sub-artificial01/pet/sub-artificial01_pet.nii.gz'
            row['pet_metadata_patch']={'Units':'Bq/mL'}
            with self.assertRaises(ValueError):promote_selected({'schema_version':1,'rows':[row]},stage,root/'badpatch')
            row['metadata_patch_reviewed']=True
            promote_selected({'schema_version':1,'rows':[row]},stage,root/'bids')
            receipt=load_json(root/'bids/promotion_receipt_PRIVATE.json')
            self.assertEqual(receipt['visual_QC'],'NOT_PERFORMED')
            self.assertEqual(receipt['BIDS_validator'],'NOT_RUN')
            self.assertEqual(load_json(j)['Units'],'BQML')


if __name__=='__main__':unittest.main()
