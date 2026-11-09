"""Synthetic-only tests: no participant inputs or model selection."""
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import step_support
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

import scanner_solver as solver

ORIGINAL = Path(__file__).resolve().parents[2] / 'src' / 'pet_vns_prediction' / 'solver.py'
spec = importlib.util.spec_from_file_location('original_solver',ORIGINAL)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)


class SolverTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261109)
        self.x = rng.normal(size=(120,5))
        self.x[:60,0] = 0.; self.x[60:,0] = 1.
        p = expit(-1.2+1.8*self.x[:,0]+.6*self.x[:,1]-.3*self.x[:,2])
        self.y = (rng.random(120)<p).astype(float)
        self.mask = np.array([0,1,1,1,1])

    def test_all_penalized_matches_frozen_solver(self):
        for c in (.001,.1,10.,100.):
            for l1 in (0.,.25,.5,.75,1.):
                with self.subTest(c=c,l1=l1):
                    a = solver.fit(self.x,self.y,c,l1,np.ones(5))
                    b = old.fit(self.x,self.y,c,l1)
                    np.testing.assert_allclose(a.predict_proba(self.x),b.predict_proba(self.x),rtol=0,atol=1e-8)
                    self.assertLess(abs(a.numerical_['objective_mean']-b.numerical_['objective_mean']),1e-12)

    def test_independent_solver_all_grid_cases(self):
        for c in (.001,.01,.1,1.,10.,100.):
            for l1 in (0.,.25,.5,.75,1.):
                with self.subTest(c=c,l1=l1):
                    fit = solver.fit(self.x,self.y,c,l1,self.mask)
                    w,proof = solver.independent_slsqp(self.x,self.y,c,l1,self.mask)
                    self.assertTrue(proof['success'])
                    self.assertLess(abs(proof['objective_mean']-fit.numerical_['objective_mean']),1e-10)
                    self.assertLess(np.max(np.abs(expit(w[0]+self.x@w[1:])-fit.predict_proba(self.x)[:,1])),1e-5)
                    self.assertLessEqual(fit.numerical_['dual_equality_error'],1e-12)
                    self.assertLessEqual(fit.numerical_['nuisance_gradient_inf'],1e-12)

    def test_strong_penalty_keeps_scanner(self):
        fit = solver.fit(self.x,self.y,.001,1.,self.mask)
        self.assertGreater(abs(fit.coef_[0,0]),.5)
        np.testing.assert_array_equal(fit.coef_[0,1:],np.zeros(4))
        p = fit.predict_proba(self.x)[:,1]
        for group in (0,1):
            np.testing.assert_allclose(p[self.x[:,0]==group],self.y[self.x[:,0]==group].mean(),atol=1e-12)

    def test_categorical_reference_and_scale_invariance(self):
        for l1 in (0.,.5,1.):
            original = solver.fit(self.x,self.y,1.,l1,self.mask)
            changed = self.x.copy(); changed[:,0] = 1.-changed[:,0]
            alternate = solver.fit(changed,self.y,1.,l1,self.mask)
            np.testing.assert_allclose(original.predict_proba(self.x),alternate.predict_proba(changed),rtol=0,atol=1e-9)
            scaled = self.x.copy(); scaled[:,0] = 3.*scaled[:,0]-2.
            alternate = solver.fit(scaled,self.y,1.,l1,self.mask)
            np.testing.assert_allclose(original.predict_proba(self.x),alternate.predict_proba(scaled),rtol=0,atol=1e-9)

    def test_unpenalized_logistic_limit(self):
        mask = np.zeros(5)
        fit = solver.fit(self.x,self.y,.001,1.,mask)
        design = np.column_stack((np.ones(len(self.y)),self.x))
        def fun(w):
            z = design@w
            return np.mean(np.logaddexp(0.,np.where(self.y==1,-z,z))),design.T@(expit(z)-self.y)/len(self.y)
        reference = minimize(fun,np.zeros(6),jac=True,method='BFGS',options={'gtol':1e-11})
        np.testing.assert_allclose(fit.predict_proba(self.x)[:,1],expit(design@reference.x),rtol=0,atol=1e-7)

    def test_deterministic_repetition(self):
        a = solver.fit(self.x,self.y,1.,.5,self.mask)
        b = solver.fit(self.x,self.y,1.,.5,self.mask)
        np.testing.assert_array_equal(a.predict_proba(self.x),b.predict_proba(self.x))

    def test_penalty_excludes_scanner_exactly(self):
        rng = np.random.default_rng(99)
        w = rng.normal(size=6)
        c,l1 = .1,.75
        a,b = l1/(c*len(self.y)),(1-l1)/(c*len(self.y))
        z = w[0]+self.x@w[1:]
        expected = np.mean(np.logaddexp(0.,np.where(self.y==1,-z,z)))+a*np.sum(np.abs(w[2:]))+b/2*np.sum(w[2:]**2)
        self.assertAlmostEqual(solver.objective(self.x,self.y,w,a*self.mask,b*self.mask),expected,places=14)

    def test_reject_constant_unpenalized_scanner(self):
        changed = self.x.copy(); changed[:,0] = 1.
        with self.assertRaises(ValueError):
            solver.fit(changed,self.y,1.,.5,self.mask)

    def test_reject_single_class_unpenalized_scanner_group(self):
        separated = self.y.copy()
        separated[self.x[:,0]==1] = 1.
        for c in (.001,1.,100.):
            for l1 in (0.,.5,1.):
                with self.subTest(c=c,l1=l1):
                    with self.assertRaises(solver.NumericalFailure) as caught:
                        solver.fit(self.x,separated,c,l1,self.mask)
                    self.assertEqual(caught.exception.diagnostics['failure_type'],
                                     'unpenalized_scanner_separation')

    def test_correlated_high_dimension_small_training_sets(self):
        for seed in (21,22,23):
            rng = np.random.default_rng(seed)
            x = rng.normal(size=(30,18))
            x[:,0] = np.arange(30)%2
            x[:,2:7] = .92*x[:,1,None]+.08*x[:,2:7]
            y = np.tile([0,0,1,1,0,1],5).astype(float)
            mask = np.r_[0,np.ones(17)]
            for c in (.001,.1,10.,100.):
                for l1 in (0.,.5,1.):
                    with self.subTest(seed=seed,c=c,l1=l1):
                        fit = solver.fit(x,y,c,l1,mask)
                        w,proof = solver.independent_slsqp(x,y,c,l1,mask)
                        self.assertTrue(proof['success'])
                        self.assertLess(abs(proof['objective_mean']-fit.numerical_['objective_mean']),1e-10)
                        self.assertLess(np.max(np.abs(expit(w[0]+x@w[1:])-fit.predict_proba(x)[:,1])),1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
