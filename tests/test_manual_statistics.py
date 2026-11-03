"""Artificial input tests for the manually executed statistical steps."""
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import step_support
from pet_vns_prediction import paper_settings,core,sensitivity
from STEP05_clinical_data_assembly.table_io import assemble,align
from STEP05_clinical_data_assembly.aggregate_features import aggregate
from STEP06_regional_associations.run_associations import run_family
from STEP07_prediction_models.evaluate_predictions import predictions_result
from STEP08_sensitivity_analyses.run_sensitivity import processing_tables
from STEP08_sensitivity_analyses.evaluate_processing_predictions import paired_comparison


def fixture(n=30):
    rng=np.random.default_rng(72)
    ids=[f'artificial{i:03}' for i in range(n)]
    settings=paper_settings()
    cohort=pd.DataFrame(index=ids)
    clinical=pd.DataFrame({
        'age_at_implant_years':rng.uniform(10,50,n),'sex':rng.choice(['female','male'],n),
        'epilepsy_duration_reported_years':rng.uniform(1,15,n),
        'baseline_monthly_frequency':rng.uniform(2,60,n),'mri_lesional_status':rng.choice(['lesional','nonlesional'],n),
        'responder_12m':[0,1]*(n//2)},index=ids)
    pet=pd.DataFrame(rng.uniform(.5,1.5,(n,6)),index=ids,columns=settings.pet)
    t1=pd.DataFrame(rng.uniform(.5,1.5,(n,6)),index=ids,columns=settings.t1)
    return cohort,clinical,pet,t1


class ManualStatisticsTests(unittest.TestCase):
    def test_keyed_assembly_preserves_order_and_frequency(self):
        cohort,c,p,t=fixture()
        data,x,y=assemble(cohort,c.iloc[::-1],p.iloc[::-1],t,generate_splits=True)
        self.assertEqual(data['study_ids'],list(cohort.index))
        self.assertEqual(list(x.columns),list(paper_settings().model_columns()['CPT']))
        np.testing.assert_array_equal(x['log1p_baseline_monthly_frequency'],[math.log1p(v) for v in c['baseline_monthly_frequency']])
        self.assertEqual(len(data['plans']),100)
        other,_,_=assemble(cohort,c,p,t,plans=data['plans'])
        self.assertEqual(other['plans'],data['plans'])

    def test_no_case_deletion_or_fractional_binary(self):
        cohort,c,p,t=fixture()
        with self.assertRaises(ValueError):align(c.iloc[:-1],list(cohort.index))
        c['responder_12m']=c['responder_12m'].astype(float)
        c.loc[c.index[0],'responder_12m']=.5
        with self.assertRaises(ValueError):assemble(cohort,c,p,t,generate_splits=True)

    def test_explicit_frozen_log_checks_raw_frequency(self):
        cohort,c,p,t=fixture()
        values=np.array([math.log1p(v) for v in c['baseline_monthly_frequency']])
        values[0]=np.nextafter(values[0],np.inf)
        c['log1p_baseline_monthly_frequency']=values
        data,x,y=assemble(cohort,c,p,t,generate_splits=True,preserve_frozen_log=True)
        np.testing.assert_array_equal(x['log1p_baseline_monthly_frequency'],values)
        self.assertIn('preserved_frozen',data['log_frequency_source'])
        c.loc[c.index[0],'log1p_baseline_monthly_frequency']+=.1
        with self.assertRaises(ValueError):assemble(cohort,c,p,t,generate_splits=True,preserve_frozen_log=True)

    def test_feature_aggregation_preserves_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pet.json';columns=paper_settings().pet
            path.write_text(json.dumps({'study_id':'artificial001','reference':'supraGM',
                'canonical_features':dict.fromkeys(columns,None)}))
            result=aggregate([path],'pet')
            self.assertTrue(result.loc[0,list(columns)].isna().all())
            with self.assertRaises(ValueError):aggregate([path,path],'pet')

    def test_association_primary_and_twelve_counts(self):
        cohort,c,p,t=fixture()
        data,x,y=assemble(cohort,c,p,t,generate_splits=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metadata.csv'
            pd.DataFrame({'study_id':cohort.index,'scanner_indicator':[0,0,1]*10}).to_csv(path,index=False)
            six=run_family(data,x,y,paper_settings(),path,'primary6')
            twelve=run_family(data,x,y,paper_settings(),path,'morphometric12')
            self.assertEqual(len(six['rows']),6)
            self.assertEqual(len(twelve['rows']),12)
            self.assertTrue(all(row['family_tests']==6 for row in six['rows']))

    def test_processing_requires_all_variants_and_frozen_order(self):
        cohort,c,p,t=fixture()
        _,x,y=assemble(cohort,c,p,t,generate_splits=True)
        payload={'study_ids':list(x.index),'variants':{k:p.to_numpy().tolist() for k in sensitivity.PREDICTION_VARIANTS}}
        result=processing_tables(payload,x,paper_settings())
        self.assertEqual(len(result),7)
        del payload['variants']['pvc4']
        with self.assertRaises(ValueError):processing_tables(payload,x,paper_settings())

    def test_saved_predictions_validation(self):
        ids=['artificialA','artificialB']
        payload={'status':'PASS','participant_ids':ids,
            'predictions':{key:[[.2,.8],[.3,.7]] for key in ('C','CP','CPT')}}
        result=predictions_result(payload,ids,2)
        self.assertEqual(result['predictions']['CP'].shape,(2,2))
        with self.assertRaises(ValueError):predictions_result(payload,ids[::-1],2)

    def test_paired_processing_direction_and_same_prediction(self):
        y=np.array([0,1]*12);p=np.linspace(.1,.9,24)
        result=paired_comparison(y,p,p,draws=12)
        self.assertEqual(result['delta_auc'],0)
        self.assertEqual(result['delta_brier'],0)
        self.assertEqual(result['delta_auc_conditional95'],[0.,0.])


if __name__=='__main__':unittest.main()
