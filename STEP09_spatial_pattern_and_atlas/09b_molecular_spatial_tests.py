"""Original-cohort 18-map cortical spin or conditional-MSR annotation."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import read_json, write_json, new_output
import numpy as np
from scipy.linalg import block_diag, null_space
from scipy.stats import rankdata
from pet_vns_prediction import molecular_annotation, spatial, vns_spatial


def validate_order(meta):
    labels = list(meta['cortical_labels'])
    names = list(meta['receptor_names'])
    if len(labels) != 68 or len(set(labels)) != 68:
        raise ValueError('68 unique cortical labels required')
    if not (all(x.startswith('R_') for x in labels[:34]) and all(x.startswith('L_') for x in labels[34:])
            and [x[2:] for x in labels[:34]] == [x[2:] for x in labels[34:]]):
        raise ValueError('Cortical order must be homologous right34 then left34')
    if names != list(spatial.TARGETS):
        raise ValueError('All 18 receptor targets in the original fixed order are required')
    return labels, names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='NPZ: observed68, receptors68x18, sphere_coords68x3, ap68, g168; optional saved spins/basis/signs')
    p.add_argument('--order', type=Path, required=True, help='JSON: cortical_labels, receptor_names')
    p.add_argument('--mode', choices=['spin', 'conditional', 'bootstrap'], required=True)
    p.add_argument('--atlas-use-authorized', action='store_true', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    labels, names = validate_order(read_json(a.order))
    with np.load(a.input, allow_pickle=False) as f:
        inp = {k: f[k] for k in f.files}
    x, y, coords = inp['observed68'], inp['receptors68x18'], inp['sphere_coords68x3']
    if x.shape != (68,) or y.shape != (68,18) or coords.shape != (68,3) or not all(np.isfinite(v).all() for v in (x,y,coords)):
        raise ValueError('Finite PC1, receptor and sphere arrays must follow the declared cortical order')
    spins = np.asarray(inp['spins']) if 'spins' in inp else None
    if a.mode == 'spin' and spins is None:
        spins = spatial.spin_generator(coords, n_null=9999, seed=20261023)['indices']
    if spins is not None and (spins.shape != (9999,68) or not all(np.array_equal(np.sort(row),np.arange(68)) for row in spins)):
        raise ValueError('9,999 full bijective spins required')
    arrays = {'spins': spins} if spins is not None else {}
    if a.mode == 'spin':
        rows, nulls = [], []
        for j, name in enumerate(names):
            row, null = molecular_annotation.statistics(x, y[:,j], spins)
            rows.append({'target': name, **row}); nulls.append(null)
        q = molecular_annotation.bh([r['p_spin'] for r in rows])
        for row, value in zip(rows,q): row['q_BH18'] = float(value)
        arrays['null_rho'] = np.column_stack(nulls)
    elif a.mode == 'conditional':
        z = vns_spatial.design(inp['ap68'], inp['g168'], np.r_[np.ones(34), np.zeros(34)])
        w = block_diag(vns_spatial.graph(coords,5), vns_spatial.graph(coords,5))
        q = null_space(z.T)
        _, vectors = np.linalg.eigh(q.T @ w @ q)
        basis = inp.get('basis', q @ vectors)
        signs = inp.get('signs', np.random.default_rng(20261027).choice([-1.,1.], size=(9999,basis.shape[1])))
        if signs.shape != (9999,basis.shape[1]) or not np.isin(signs,[-1.,1.]).all():
            raise ValueError('Conditional sign matrix dimensions/values')
        d = {'z': z, 'yr': rankdata(y,axis=0), 'W': w, 'coords': coords, 'names': names, 'spins': spins}
        rows, sim, cor, diagnostics = vns_spatial.test_family(x,d,basis,signs,'conditional_MSR')
        arrays.update({'conditional_maps':sim,'null_effects':cor,'basis':basis,'signs':signs,'z':z,'W':w})
    else:
        boot=np.asarray(inp['bootstrap68'],float)
        if boot.shape!=(2000,68) or not np.isfinite(boot).all():
            raise ValueError('All 2,000 already aligned participant-bootstrap maps are required')
        z=vns_spatial.design(inp['ap68'],inp['g168'],np.r_[np.ones(34),np.zeros(34)])
        xr=rankdata(np.vstack([x,boot]),axis=1);yr=rankdata(y,axis=0)
        raw=spatial.correlations(xr,yr)
        partial=spatial.correlations(vns_spatial.project(xr.T,z).T,vns_spatial.project(yr,z))
        rows=[]
        for j,name in enumerate(names):
            rows.append({'target':name,'spearman_rho':float(raw[0,j]),
                'participant_bootstrap95':np.quantile(raw[1:,j],[.025,.975]).tolist(),
                'same_sign_fraction':float(np.mean(np.sign(raw[1:,j])==np.sign(raw[0,j]))),
                'partial_rho':float(partial[0,j]),
                'partial_participant_bootstrap95':np.quantile(partial[1:,j],[.025,.975]).tolist()})
        arrays.update({'bootstrap_spearman':raw[1:],'bootstrap_partial':partial[1:]})
    out = new_output(a.output)
    np.savez_compressed(out/'spatial_arrays.npz', **arrays)
    write_json(out/'spatial_results.json', {'mode':a.mode,'cortical_labels':labels,'family_size':18,
        'rows':rows,'n_null':0 if a.mode=='bootstrap' else 9999,'diagnostics':diagnostics if a.mode=='conditional' else None,
        'scope':'Cortical spatial correspondence; not participant receptor measurement or causal mechanism'})


if __name__ == '__main__': main()
