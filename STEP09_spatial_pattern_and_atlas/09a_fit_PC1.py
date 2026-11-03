"""Fit one prespecified cohort PC1; optionally run its participant bootstrap."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import read_json, write_json, new_output
import numpy as np
from pet_vns_prediction import vns_pattern, spatial


def fit_input(data, branch):
    raw = np.asarray(data.get('values', data.get('raw_roi_means')), dtype=float)
    names = list(data.get('features', data.get('feature_names', [])))
    if raw.ndim != 2 or raw.shape[1] != 82 or len(raw) < 3:
        raise ValueError('Expected at least 3 participants x 82 positive regional means')
    if len(names) != 82 or len(set(names)) != 82:
        raise ValueError('An explicit, unique 82-region order is required')
    if branch == 'original':
        clr = vns_pattern.clr(raw)
        result = vns_pattern.fit(clr, np.ones((len(raw), 1)))
        arrays = {'clr82': clr, 'loadings': result['loadings'], 'scores': result['scores'],
                  'eigenvalues': result['eigen'], 'variance_fractions': result['fraction'],
                  'pc1_loading82': result['loadings'][:, 0]}
    else:
        clr = spatial.clr_transform(raw)
        result = spatial.orient_pallidum(spatial.fit_pc1(clr), names)
        arrays = {'clr82': clr, 'pc1_loading82': result['loading'], 'scores': result['scores'],
                  'eigenvalues': result['eigenvalues'], 'feature_center82': result['center']}
    return raw, names, result, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='JSON: values/features, optional ids/scanner; no outcome fields used')
    p.add_argument('--branch', choices=['original', 'external'], required=True)
    p.add_argument('--bootstrap', action='store_true', help='Also refit all 2,000 participant resamples')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = read_json(a.input)
    raw, names, result, arrays = fit_input(data, a.branch)
    if a.bootstrap:
        if a.branch == 'original':
            if len(data.get('scanner', [])) != len(raw):
                raise ValueError('Original-cohort bootstrap requires the frozen scanner strata in row order')
            b = vns_pattern.scanner_stratified_bootstrap(raw, data['scanner'], arrays['pc1_loading82'])
            arrays.update({'bootstrap_' + k: v for k, v in b.items()})
        else:
            _, b = spatial.bootstrap_pc1(arrays['clr82'], arrays['pc1_loading82'])
            arrays.update({'bootstrap_' + k: v for k, v in b.items()})
    out = new_output(a.output)
    np.savez_compressed(out/'PC1_arrays.npz', **arrays)
    write_json(out/'PC1_results.json', {'branch': a.branch, 'n': len(raw), 'features': names,
        'PC1_loading': arrays['pc1_loading82'], 'pca': result,
        'bootstrap_draws': 2000 if a.bootstrap else 0,
        'outcomes_used': False, 'scope': 'One fixed decomposition, not prediction validation'})


if __name__ == '__main__':
    main()
