"""Artificial fixtures for manual spatial and figure entrypoints."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import step_support


def module(folder,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/folder/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class ManualSpatialFigureTests(unittest.TestCase):
    def test_original_pca_matches_retained_fit(self):
        from pet_vns_prediction import vns_pattern
        m=module('STEP09_spatial_pattern_and_atlas','09a_fit_PC1')
        x=np.random.default_rng(1).lognormal(size=(13,82));names=['region'+str(i) for i in range(82)]
        _,_,_,arrays=m.fit_input({'values':x,'features':names},'original')
        expected=vns_pattern.fit(vns_pattern.clr(x),np.ones((len(x),1)))
        np.testing.assert_array_equal(arrays['loadings'],expected['loadings'])

    def test_cortical_order_rejects_unpaired_hemispheres(self):
        from pet_vns_prediction.spatial import TARGETS
        m=module('STEP09_spatial_pattern_and_atlas','09b_molecular_spatial_tests')
        labels=['R_p'+str(i) for i in range(34)]+['L_p'+str(i) for i in range(34)]
        self.assertEqual(m.validate_order({'cortical_labels':labels,'receptor_names':list(TARGETS)})[0],labels)
        with self.assertRaises(ValueError):m.validate_order({'cortical_labels':labels[::-1],'receptor_names':list(TARGETS)})

    def test_figure4_payload_uses_saved_rank_residuals(self):
        m=module('STEP10_tables_and_figures','10a_prepare_plot_payloads')
        rng=np.random.default_rng(7);arrays={}
        for condition in ['baseline','DAT']:
            x=rng.normal(size=(68,3));y=rng.normal(size=(4,68));effect=np.corrcoef(x[:,0],y[0])[0,1]
            arrays[condition+'_target_residuals']=x;arrays[condition+'_PC1_residuals']=y;arrays[condition+'_effects']=np.full((4,3),effect)
        result=m.figure4_payload({'SERT_correspondence':[]},arrays,['parcel'+str(i) for i in range(68)],{})
        self.assertEqual(len(result['scatter_data']),2)
        self.assertEqual(len(result['scatter_data'][0]['points']),68)
        arrays['DAT_effects'][0,0]+=1
        with self.assertRaises(ValueError):m.figure4_payload({},arrays,['p'+str(i) for i in range(68)],{})

    def test_surface_face_coloring_preserves_mapping(self):
        from matplotlib.colors import Normalize,LinearSegmentedColormap
        m=module('STEP10_tables_and_figures','10c_render_PC1_SERT_surfaces')
        names=['p'+str(i) for i in range(34)];ann=np.repeat(np.arange(34),3);faces=np.arange(102).reshape(34,3)
        labels=['R_'+n for n in names]+['L_'+n for n in names]
        colors,ledger=m.color_faces(ann,faces,names,labels,np.r_[np.arange(34),np.arange(34)],'rh',
            LinearSegmentedColormap.from_list('test',['white','purple']),Normalize(0,33))
        self.assertEqual(colors.shape,(34,3));self.assertEqual(len(ledger),34)
        self.assertTrue(all(row['faces']==1 for row in ledger))

    def test_updated_final_headings(self):
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        m=module('STEP10_tables_and_figures','10b_render_figure')
        f=plt.figure();f.text(.2,.2,'Feature-removal ablation')
        m.final_label_changes(f,'Figure3')
        self.assertIn('Regional feature ablation',[t.get_text() for t in f.texts]);self.assertIn('AUC',[t.get_text() for t in f.texts])
        plt.close(f)

    def test_clinical_bridge_preserves_exact_repeated_person_ties(self):
        m=module('STEP09_spatial_pattern_and_atlas','09e_clinical_bridge')
        rng=np.random.default_rng(19);z=rng.normal(size=(11,82));idx=np.array([3,3,8,1,8,3,0,5,6,7,8])
        loading=rng.normal(size=82);loading/=np.linalg.norm(loading)
        s=m.exact_duplicate_scores(z,idx,loading,1.4)
        self.assertTrue(np.array_equal(s[idx==3],np.full(sum(idx==3),s[idx==3][0])))
        self.assertTrue(np.array_equal(s[idx==8],np.full(sum(idx==8),s[idx==8][0])))

    def test_step07_and_step08_outputs_join_into_figure3(self):
        import pandas as pd
        from pet_vns_prediction import core,paper_settings,validation
        m=module('STEP10_tables_and_figures','10a_prepare_plot_payloads')
        rng=np.random.default_rng(84);ids=['artificial'+str(i) for i in range(40)];y=pd.Series([0,1]*20,index=ids)
        p={name:rng.uniform(.1,.9,(3,len(y))) for name in ('C','CP','CPT')}
        main={'status':'PASS','participant_ids':ids,'predictions':p}
        metrics=core.summarize_predictions(main,y,paper_settings())
        intervals=core.paired_bootstrap(main,y,draws=25,seed=19)
        names=['CP_without_thalamus','CP_without_striatopallidal','CP_without_both']
        ab={'status':'PASS','participant_ids':ids,'predictions':{'CP':p['CP'],**{name:rng.uniform(.1,.9,(3,len(y))) for name in names}},
            'model_split_digests':{name:'same-artificial-splits' for name in ['CP',*names]}}
        ab['ablation_metrics']=validation.evaluate_ablations(ab,y,draws=25,seed=19)
        payload=m.figure3_payload(y,p,metrics,ab,intervals)
        self.assertEqual(payload['models']['CP']['auc_conditional95'],intervals['intervals']['CP_auc'])
        self.assertEqual([r['model_removed'] for r in payload['ablation']],names)
        with self.assertRaises(ValueError):m.figure3_payload(y,p,metrics,ab)

    def test_complete_step06_families_join_into_figure2(self):
        import pandas as pd
        from pet_vns_prediction import associations,paper_settings
        m=module('STEP10_tables_and_figures','10g_prepare_additional_payloads')
        rng=np.random.default_rng(184);s=paper_settings();n=36
        y=pd.Series([0,1]*18,index=['artificial'+str(i) for i in range(n)])
        raw=rng.uniform(.75,1.25,(n,6));x=pd.DataFrame(raw,index=y.index,columns=s.pet)
        primary={'rows':associations.association_family(raw,y,{'clinical5_scanner':rng.normal(size=(n,4))},region_names=s.pet,expected_tests=6),'family_tests':6}
        extended={'rows':associations.association_family(raw,y,{name:rng.normal(size=(n,5)) for name in ('base_plus_epilepsy_type','base_plus_etiology3','base_plus_epilepsy_type_and_etiology3')},region_names=s.pet,expected_tests=18),'family_tests':18}
        for result in (primary,extended):result.update(participant_ids=list(y.index),region_order=list(s.pet),source_input_sha256='artificial-input-fingerprint')
        data={'study_ids':list(y.index)}
        payload=m.figure2_payload(data,x,y,s,primary,extended,[.7,1.3],'artificial-input-fingerprint')
        self.assertEqual(len(payload['primary']),6);self.assertEqual(len(payload['extended']),6)
        np.testing.assert_array_equal(payload['raw_relative_values'],raw)
        surface=module('STEP10_tables_and_figures','10e_render_regional_assets')
        config=step_support.read_json(ROOT/'configs/features.json')
        colors=surface.regional_effects(payload,config)
        self.assertEqual(len(colors),6)
        np.testing.assert_array_equal(list(colors.values()),[r['standardized_difference'] for r in payload['primary']])
        broken={**extended,'rows':extended['rows'][:6]}
        with self.assertRaises(ValueError):m.figure2_payload(data,x,y,s,primary,broken,[.7,1.3])
        with self.assertRaises(ValueError):m.figure2_payload(data,x,y,s,primary,extended,[.7,1.3],'wrong-version')

    def test_scanner_and_development_schema_joins_into_S1(self):
        from pet_vns_prediction import validation
        m=module('STEP10_tables_and_figures','10g_prepare_additional_payloads')
        rng=np.random.default_rng(142);y=np.array([0,1]*12);directions=[]
        for i in range(2):
            p={name:rng.uniform(.1,.9,len(y)) for name in ('C','CP')}
            directions.append({'training_scanner':'artificial-A','heldout_scanner':'artificial-B','n_train':24,'n_test':24,
                'models':{name:{'metrics':validation.measures(y,p[name])} for name in p},
                'conditional_intervals':validation.conditional_intervals(y,p,draws=25,seed=1)})
        pct=np.array([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]])
        development={'status':'PASS','participant_ids':['artificial1','artificial2','artificial3'],
            'probability_stability':{'CP':{'bootstrap_percentiles_2p5_50_97p5':pct}}}
        payload=m.figure_s1_payload({'status':'PASS','directions':directions},development)
        self.assertEqual(len(payload['cross_scanner']),4);self.assertEqual(len(payload['ranked_saved_CP_percentiles']),3)

    def test_metric_interval_summary_retains_probability_ties(self):
        m=module('STEP10_tables_and_figures','10g_prepare_additional_payloads')
        r=m.conditional_metric_summaries([0,1]*10,{'artificial':np.full((3,20),.5)},draws=25,seed=17)
        self.assertEqual(r['artificial']['auc'],.5);self.assertEqual(r['artificial']['auc_conditional95'],[.5,.5])

    def test_all96_associations_and_all14_predictions_join_into_S2(self):
        from pet_vns_prediction import sensitivity,paper_settings
        m=module('STEP10_tables_and_figures','10g_prepare_additional_payloads')
        rng=np.random.default_rng(285);s=paper_settings();n=36;ids=['artificial'+str(i) for i in range(n)];y=np.array([0,1]*18)
        variants={name:rng.uniform(.8,1.3,(n,6)) for name in sensitivity.PROCESSING_VARIANTS}
        clinical=rng.normal(size=(n,5));clinical[:,1]=rng.integers(0,2,n);clinical[:,4]=rng.integers(0,2,n)
        family=sensitivity.processing_association_family(variants,y,clinical,rng.integers(0,2,n),rng.normal(size=(n,6)),region_names=s.pet,influence=False)
        family.update(participant_ids=ids,region_order=list(s.pet),source_input_sha256='artificial')
        main={'participant_ids':ids,'split_digest':'same-artificial-splits','predictions':{'CP':rng.uniform(.1,.9,(s.repeats,n))}}
        processing={'status':'PASS','participant_ids':ids,'split_digest':'same-artificial-splits',
            'predictions':{name:rng.uniform(.1,.9,(s.repeats,n)) for name in sensitivity.processing_model_definitions(s)}}
        payload=m.figure_s2_payload({'study_ids':ids},y,s,family,main,processing,'artificial')
        self.assertEqual(len(payload['models']),8);self.assertEqual(payload['q_family_size'],96)
        self.assertEqual(payload['order'],list(__import__('pet_vns_prediction.figures',fromlist=['PROCESSING_ORDER']).PROCESSING_ORDER))


if __name__=='__main__':unittest.main()
