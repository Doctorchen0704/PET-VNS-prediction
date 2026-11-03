import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from pet_vns_prediction.intake import safe_relative,new_output,conversion_plan,promote_selected

class IntakeTests(unittest.TestCase):
    def test_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in ('../secret','/outside','C:/secret','a\\b','.'):
                with self.assertRaises(ValueError): safe_relative(d,bad)
    def test_fresh_output(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError): new_output(d)
            with self.assertRaises(ValueError): new_output(Path(d)/'new',(d,))
    def test_no_automatic_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=root/'raw';raw.mkdir();(raw/'one').mkdir();(raw/'one'/'dummy').write_bytes(b'fixture')
            m={'schema_version':1,'rows':[{'subject':'synthetic01','modality':'PET','source_relative':'one','identity_reviewed':False,'series_reviewed':True}]}
            with self.assertRaises(ValueError):conversion_plan(m,raw,root/'out','dcm2niix')
            m['rows'][0]['identity_reviewed']=True
            p=conversion_plan(m,raw,root/'out','dcm2niix')
            self.assertFalse((root/'out').exists());self.assertEqual(p[0]['command'][1:9],['-b','y','-ba','y','-z','y','-w','0'])
    def test_promotion_hash_and_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); stage=root/'stage';stage.mkdir()
            n=stage/'source.nii.gz';j=stage/'source.json'
            # Artificial bytes test file handling only, not NIfTI geometry.
            n.write_bytes(b'not-a-real-image');j.write_text('{"Units":"BQML"}')
            row={'source_nii':n.name,'source_json':j.name,'nii_sha256':hashlib.sha256(n.read_bytes()).hexdigest(),
                 'json_sha256':hashlib.sha256(j.read_bytes()).hexdigest(),'target_nii':'sub-synthetic01/pet/sub-synthetic01_trc-18FFDG_pet.nii.gz',
                 'review_pass':True,'metadata_patch_reviewed':True,'pet_metadata_patch':{'Units':'Bq/mL'}}
            manifest={'schema_version':1,'rows':[row]}
            promote_selected(manifest,stage,root/'bids')
            self.assertEqual(n.read_bytes(),(root/'bids'/row['target_nii']).read_bytes())
            self.assertEqual(json.loads(j.read_text())['Units'],'BQML')
            row['nii_sha256']='0'*64
            with self.assertRaises(ValueError):promote_selected(manifest,stage,root/'bids2')
            self.assertFalse((root/'bids2').exists())

if __name__=='__main__':unittest.main()
