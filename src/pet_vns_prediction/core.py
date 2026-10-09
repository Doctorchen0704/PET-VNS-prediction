"""Pure tabular modeling primitives from the study implementation.

No filesystem access, private data, or automatic fitting occurs on import.
The public modeling API uses the final deterministic solver, never SAGA.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


class InputRejected(ValueError):
    pass


class ClassSupportError(InputRejected):
    pass


class FitFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    clinical: tuple[str, ...]
    categories: dict[str, tuple[str, str]]
    pet: tuple[str, ...]
    t1: tuple[str, ...]
    c_grid: tuple[float, ...]
    l1_grid: tuple[float, ...]
    seed: int
    repeats: int
    inner_folds: int
    tol: float
    max_iter: int
    retry_iter: int
    clip: float
    thresholds: tuple[float, ...]
    max_missing: float
    profile: str = "locked_full"

    @classmethod
    def from_documents(cls, model, imaging):
        # A strict interpretation of the frozen specification, not silent defaults.
        if model['version'] != '20260909_v1':
            raise InputRejected('Unexpected model specification version.')
        clinical = tuple(model['clinical_predictors']['primary'])
        refs = model['clinical_predictors']['categorical_reference']
        if refs != {'sex': 'female', 'mri_lesional_status': 'nonlesional'}:
            raise InputRejected('Categorical reference specification changed.')
        mappings = imaging['pathway_mapping']
        if len(mappings) != 6:
            raise InputRejected('Exactly six frozen pathway mappings required.')
        cv, solver = model['validation'], model['models']['convergence']
        return cls(
            clinical=clinical,
            categories={'sex': ('female', 'male'), 'mri_lesional_status': ('nonlesional', 'lesional')},
            pet=tuple('pet_supraGM__' + p['pet'] for p in mappings),
            t1=tuple('t1__' + p['t1'] for p in mappings),
            c_grid=tuple(model['models']['C_grid']), l1_grid=tuple(model['models']['l1_ratio_grid']),
            seed=cv['seed'], repeats=cv['outer_repeats'], inner_folds=cv['inner_folds'],
            tol=solver['tolerance'], max_iter=solver['max_iterations'], retry_iter=solver['retry_once_max_iterations'],
            clip=model['evaluation']['probability_clip_for_logit'],
            thresholds=tuple(model['evaluation']['decision_curve']['thresholds']),
            max_missing=model['preprocessing']['maximum_baseline_missing_fraction'],
        )

    def model_columns(self, pet_variant='supraGM', add_asm=False):
        if pet_variant not in ('supraGM', 'wholeBrain', 'cerebellarCortex', 'petprep'):
            raise InputRejected('Unknown PET variant.')
        prefix = 'petprep_supraGM__' if pet_variant == 'petprep' else f'pet_{pet_variant}__'
        pet = tuple(prefix + p.split('__', 1)[1] for p in self.pet)
        c = self.clinical + (('asm_count_at_implant',) if add_asm else ())
        return {'C': c, 'CP': c + pet, 'CPT': c + pet + self.t1}

    def allowed_columns(self):
        result = set(self.clinical + self.t1 + ('asm_count_at_implant',))
        for variant in ('supraGM', 'wholeBrain', 'cerebellarCortex', 'petprep'):
            result.update(self.model_columns(variant)['CP'])
        return result


def validate_inputs(x, y, settings, columns):
    if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.Series):
        raise InputRejected('Predictors and outcome must be indexed tables/series.')
    if not x.index.is_unique or x.index.hasnans or not y.index.is_unique:
        raise InputRejected('Unique nonmissing participant indices are required.')
    if not x.index.equals(y.index):
        raise InputRejected('Outcome and predictor indices must match exactly in order.')
    if len(x) < 2 or not x.columns.is_unique:
        raise InputRejected('Invalid input dimensions or duplicate columns.')
    if set(x.columns) - settings.allowed_columns():
        raise InputRejected('Unexpected columns; identity/outcome/free-text columns are not allowed.')
    if not set(columns).issubset(x.columns):
        raise InputRejected('Required predictor column missing.')
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise InputRejected('Outcomes must be explicit numeric 0/1; missing outcomes are not imputed.')
    for field in columns:
        values = x[field]
        if field in settings.categories:
            if set(values.dropna().unique()) - set(settings.categories[field]):
                raise InputRejected('Unrecognized categorical code.')
        else:
            try:
                numeric = values.to_numpy(dtype=float, na_value=np.nan)
            except (ValueError, TypeError) as exc:
                raise InputRejected('Non-numeric value in numeric predictor.') from exc
            if np.isinf(numeric).any():
                raise InputRejected('Infinite predictor value.')
        if values.isna().mean() > settings.max_missing:
            raise InputRejected('Missingness exceeds the frozen threshold; new snapshot/QC required.')


def choose_outer_folds(y):
    values, counts = np.unique(np.asarray(y), return_counts=True)
    if not np.array_equal(values, [0, 1]):
        raise ClassSupportError('Two evaluable outcome classes required.')
    minimum = int(counts.min())
    for k in (5, 4, 3):
        if minimum // k >= 2 and minimum - math.ceil(minimum / k) >= 3:
            return k
    raise ClassSupportError('Class support below prespecified nested-validation minimum.')


def make_split_plan(y, settings):
    k = choose_outer_folds(y)
    splitter = RepeatedStratifiedKFold(n_splits=k, n_repeats=settings.repeats, random_state=settings.seed)
    plans = []
    for serial, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
        repeat, fold = divmod(serial, k)
        inner_seed = settings.seed + 1000 * repeat + fold + 1
        inner = StratifiedKFold(n_splits=settings.inner_folds, shuffle=True, random_state=inner_seed)
        inner_pairs = [(train[a], train[b]) for a, b in inner.split(np.zeros(len(train)), y.iloc[train])]
        plans.append({'repeat': repeat, 'fold': fold, 'train': train.tolist(), 'test': test.tolist(),
                      'inner_seed': inner_seed, 'inner': [{'train': a.tolist(), 'validation': b.tolist()} for a, b in inner_pairs]})
    validate_split_plan(plans, y, settings.repeats, settings.inner_folds)
    return plans


def validate_split_plan(plans, y, repeats, inner_folds):
    n = len(y)
    seen = np.zeros((repeats, n), dtype=int)
    for plan in plans:
        train, test = set(plan['train']), set(plan['test'])
        if train & test or train | test != set(range(n)):
            raise InputRejected('Invalid outer partition.')
        if len(plan['train']) != len(train) or len(plan['test']) != len(test):
            raise InputRejected('Duplicate row within a split.')
        if len(plan['inner']) != inner_folds:
            raise InputRejected('Unexpected inner fold count.')
        seen[plan['repeat'], plan['test']] += 1
        inner_seen = {i: 0 for i in train}
        for pair in plan['inner']:
            a, b = set(pair['train']), set(pair['validation'])
            if a & b or a | b != train or (a | b) & test:
                raise InputRejected('Inner split leaks or omits outer-training rows.')
            if len(a) != len(pair['train']) or len(b) != len(pair['validation']):
                raise InputRejected('Duplicate inner participant.')
            if len(np.unique(y.iloc[list(a)])) != 2 or len(np.unique(y.iloc[list(b)])) != 2:
                raise ClassSupportError('Inner fold lacks both classes.')
            for i in b:
                inner_seen[i] += 1
        if any(v != 1 for v in inner_seen.values()):
            raise InputRejected('Inner validation coverage is not exactly once.')
    if not np.all(seen == 1):
        raise InputRejected('Outer prediction coverage is not once per participant per repeat.')


class FoldPreprocessor:
    """Training-only median/mode imputation and continuous standardization."""
    def __init__(self, columns, categories):
        self.columns, self.categories = tuple(columns), categories

    def fit(self, x):
        self.training_ids = tuple(x.index.astype(str))
        self.fill, self.mean, self.scale = {}, {}, {}
        for field in self.columns:
            series = x[field]
            if field in self.categories:
                observed = series.dropna()
                if observed.empty:
                    raise FitFailure('Entirely missing categorical training column; no mode exists.')
                # Ties resolved by the prespecified reference-first code order.
                self.fill[field] = max(self.categories[field], key=lambda v: int((observed == v).sum()))
            else:
                v = series.to_numpy(dtype=float, na_value=np.nan)
                if np.isnan(v).all():
                    raise FitFailure('Entirely missing numeric training column; no median exists.')
                self.fill[field] = float(np.nanmedian(v))
                filled = np.where(np.isnan(v), self.fill[field], v)
                self.mean[field] = float(filled.mean())
                sd = float(filled.std(ddof=0))
                self.scale[field] = sd if sd > 0 else 1.0
        full = self._transform_full(x)
        self.keep = np.ptp(full, axis=0) > 0
        self.dropped = tuple(f for f, keep in zip(self.columns, self.keep) if not keep)
        if not self.keep.any():
            raise FitFailure('No nonconstant training predictor remains.')
        return self

    def _transform_full(self, x):
        values = []
        for field in self.columns:
            if field in self.categories:
                series = x[field].fillna(self.fill[field])
                if set(series.unique()) - set(self.categories[field]):
                    raise InputRejected('Unknown categorical level at transformation.')
                values.append((series == self.categories[field][1]).to_numpy(dtype=float))
            else:
                v = x[field].to_numpy(dtype=float, na_value=np.nan)
                values.append((np.where(np.isnan(v), self.fill[field], v) - self.mean[field]) / self.scale[field])
        result = np.column_stack(values)
        if not np.isfinite(result).all():
            raise InputRejected('Nonfinite transformed predictors.')
        return result

    def transform(self, x):
        return self._transform_full(x)[:, self.keep]

    def state(self):
        return {'training_ids': list(self.training_ids), 'columns': list(self.columns), 'fill': self.fill,
                'mean': self.mean, 'scale': self.scale, 'dropped': list(self.dropped)}


def fit_logistic(x, y, settings, c, l1, seed):
    """Use the final deterministic numerical path through the public adapter."""
    from .workflow import corrected_fit
    return corrected_fit(x, y, settings, c, l1, seed)


def select_candidate(candidates):
    if not candidates or any(not np.isfinite(r['mean_brier']) for r in candidates):
        raise FitFailure('Invalid tuning grid; an incomplete grid is not silently accepted.')
    minimum = min(r['mean_brier'] for r in candidates)
    return min((r for r in candidates if r['mean_brier'] <= minimum + 1e-8), key=lambda r: (r['C'], r['l1_ratio']))


def run_outer_model(x, y, columns, settings, plan, model_name, audit, check_deadline=lambda: None):
    # Reuse only outcome-independent preprocessing within this exact inner split.
    prepared = []
    for inner_index, pair in enumerate(plan['inner']):
        train, valid = pair['train'], pair['validation']
        transformer = FoldPreprocessor(columns, settings.categories).fit(x.iloc[train])
        audit.append({'model': model_name, 'repeat': plan['repeat'], 'fold': plan['fold'],
                      'stage': 'inner', 'inner_fold': inner_index, 'fit_rows': train, **transformer.state()})
        prepared.append((transformer.transform(x.iloc[train]), y.iloc[train], transformer.transform(x.iloc[valid]), y.iloc[valid]))
    candidates, retry_count, fit_count = [], 0, 0
    for c in settings.c_grid:
        for l1 in settings.l1_grid:
            scores = []
            for train_x, train_y, valid_x, valid_y in prepared:
                check_deadline()
                estimator, info = fit_logistic(train_x, train_y, settings, c, l1, plan['inner_seed'])
                retry_count += int(info['retried'])
                fit_count += 1
                scores.append(float(brier_score_loss(valid_y, estimator.predict_proba(valid_x)[:, 1])))
            candidates.append({'C': float(c), 'l1_ratio': float(l1), 'mean_brier': float(np.mean(scores))})
    selected = select_candidate(candidates)
    train, test = plan['train'], plan['test']
    transformer = FoldPreprocessor(columns, settings.categories).fit(x.iloc[train])
    audit.append({'model': model_name, 'repeat': plan['repeat'], 'fold': plan['fold'],
                  'stage': 'outer_refit', 'fit_rows': train, **transformer.state()})
    check_deadline()
    estimator, info = fit_logistic(transformer.transform(x.iloc[train]), y.iloc[train], settings,
                                   selected['C'], selected['l1_ratio'], plan['inner_seed'])
    prediction = estimator.predict_proba(transformer.transform(x.iloc[test]))[:, 1]
    if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)):
        raise FitFailure('Invalid predicted probabilities.')
    return prediction, {'selected': selected, 'candidates': candidates, 'fit_count': fit_count + 1,
                        'convergence_retries': retry_count + int(info['retried'])}


def assert_training_isolation(result):
    by_fold = {(r['repeat'], r['fold']): r for r in result['plans']}
    for record in result['audit']:
        plan = by_fold[(record['repeat'], record['fold'])]
        expected = plan['train'] if record['stage'] == 'outer_refit' else plan['inner'][record['inner_fold']]['train']
        if record['fit_rows'] != expected or set(record['fit_rows']) & set(plan['test']):
            raise AssertionError('Preprocessing fit includes forbidden rows.')
        expected_ids = [result['participant_ids'][i] for i in expected]
        if record['training_ids'] != expected_ids:
            raise AssertionError('Preprocessing training identity audit mismatch.')
    if len(set(result['model_split_digests'].values())) != 1:
        raise AssertionError('Models did not share split plans.')


def calibration(y, p, clip=1e-6):
    y, p = np.asarray(y), np.clip(np.asarray(p), clip, 1 - clip)
    z = np.log(p / (1 - p))
    if len(np.unique(y)) != 2:
        return {'status': 'NOT_ESTIMABLE_SINGLE_CLASS'}
    offset = brentq(lambda a: float(np.mean(expit(z + a)) - np.mean(y)), -50, 50)
    if np.ptp(z) < 1e-10:
        return {'status': 'CONSTANT_PREDICTIONS_SLOPE_NOT_ESTIMABLE', 'intercept_offset_slope_fixed_1': float(offset), 'joint_intercept': None, 'slope': None}
    design = np.column_stack([np.ones(len(y)), z])
    def loss(beta):
        linear = design @ beta
        return float(np.sum(np.logaddexp(0, linear) - y * linear))
    def jac(beta):
        return design.T @ (expit(design @ beta) - y)
    fit = minimize(loss, np.array([0.0, 1.0]), jac=jac, method='BFGS', options={'gtol': 1e-7, 'maxiter': 1000})
    probabilities = expit(design @ fit.x)
    information = design.T @ ((probabilities * (1 - probabilities))[:, None] * design)
    estimable = np.max(np.abs(jac(fit.x))) < 1e-5 and np.linalg.cond(information) < 1e12 and np.max(np.abs(fit.x)) < 50
    return {'status': 'ESTIMABLE' if estimable else 'UNSTABLE_OR_SEPARATED', 'intercept_offset_slope_fixed_1': float(offset),
            'joint_intercept': float(fit.x[0]) if estimable else None, 'slope': float(fit.x[1]) if estimable else None}


def decision_curve(y, p, thresholds):
    y, p = np.asarray(y), np.asarray(p)
    result = []
    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise InputRejected('Decision threshold must be strictly between zero and one.')
        selected = p >= threshold
        weight = threshold / (1 - threshold)
        result.append({'threshold': threshold, 'model_net_benefit': float(np.mean(selected & (y == 1)) - np.mean(selected & (y == 0)) * weight),
                       'treat_all': float(y.mean() - (1 - y.mean()) * weight), 'treat_none': 0.0})
    return result


def summarize_predictions(result, y, settings):
    if isinstance(y, pd.Series) and list(y.index) != result['participant_ids']:
        raise InputRejected('Outcome order changed before performance calculation.')
    if result['status'] != 'PASS':
        raise FitFailure('Failed/incomplete validation cannot produce a performance summary.')
    summaries = {}
    for model, matrix in result['predictions'].items():
        if not np.isfinite(matrix).all():
            raise FitFailure('Missing prediction; no participant can be silently discarded.')
        p = matrix.mean(axis=0)
        summaries[model] = {'n_participants': len(y), 'repeats': matrix.shape[0], 'auc': float(roc_auc_score(y, p)),
                            'brier': float(brier_score_loss(y, p)), 'calibration': calibration(y, p, settings.clip),
                            'repeat_auc': [float(roc_auc_score(y, row)) for row in matrix],
                            'repeat_brier': [float(brier_score_loss(y, row)) for row in matrix],
                            'decision_curve': decision_curve(y, p, settings.thresholds)}
    return summaries


def paired_bootstrap(result, y, draws=2000, seed=20260910):
    if isinstance(y, pd.Series) and list(y.index) != result['participant_ids']:
        raise InputRejected('Outcome order changed before bootstrap.')
    if result['status'] != 'PASS':
        raise FitFailure('Cannot bootstrap an incomplete validation run.')
    p = {m: matrix.mean(axis=0) for m, matrix in result['predictions'].items()}
    y = np.asarray(y)
    if any(len(v) != len(y) or not np.isfinite(v).all() for v in p.values()):
        raise FitFailure('Invalid participant-level predictions.')
    rng, records, invalid = np.random.default_rng(seed), [], 0
    for _ in range(draws):
        rows = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[rows])) < 2:
            invalid += 1
            continue
        scores = {m: (float(roc_auc_score(y[rows], v[rows])), float(brier_score_loss(y[rows], v[rows]))) for m, v in p.items()}
        record = {f'{m}_{metric}': values[j] for m, values in scores.items() for j, metric in enumerate(('auc', 'brier'))}
        for base, added in (('C', 'CP'), ('CP', 'CPT')):
            record[f'{added}_minus_{base}_auc'] = scores[added][0] - scores[base][0]
            record[f'{base}_minus_{added}_brier'] = scores[base][1] - scores[added][1]
        records.append(record)
    intervals = {key: np.quantile([r[key] for r in records], [.025, .975]).tolist() for key in records[0]} if records else {}
    return {'attempted_draws': draws, 'valid_draws': len(records), 'single_class_draws_skipped': invalid,
            'resampling_unit': 'participant; shared draws for all models', 'seed': seed,
            'interpretation': 'conditional descriptive intervals; not full model-development uncertainty', 'intervals': intervals}


def synthetic_fixture(settings, n=60, minority=None, seed=20260910):
    """All values and labels generated here; no participant data is consulted."""
    rng = np.random.default_rng(seed)
    index = pd.Index([f'SIM{i+1:04d}' for i in range(n)], name='synthetic_id')
    data = {
        'age_at_implant_years': rng.uniform(8, 60, n),
        'sex': rng.choice(['female', 'male'], n),
        'epilepsy_duration_reported_years': rng.uniform(1, 8, n),
        'log1p_baseline_monthly_frequency': rng.uniform(0.2, 4.5, n),
        'mri_lesional_status': rng.choice(['nonlesional', 'lesional'], n),
        'asm_count_at_implant': rng.integers(1, 6, n).astype(float),
    }
    for i, field in enumerate(settings.pet):
        value = rng.normal(1.1, .1, n)
        data[field] = value
        for variant in ('wholeBrain', 'cerebellarCortex', 'petprep'):
            name = settings.model_columns(variant)['CP'][len(settings.clinical) + i]
            data[name] = value + rng.normal(0, .01, n)
    for field in settings.t1:
        data[field] = rng.normal(2.5, .15, n)
    x = pd.DataFrame(data, index=index)
    m = n // 2 if minority is None else minority
    if not 0 <= m <= n:
        raise InputRejected('Invalid synthetic class size.')
    labels = np.array([1] * m + [0] * (n - m), dtype=int)
    rng.shuffle(labels)
    return x, pd.Series(labels, index=index, name='synthetic_outcome')
