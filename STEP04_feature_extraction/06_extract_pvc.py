"""Extract full-reference volume-weighted sGTM features without clipping coefficients."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import ROOT, require_review, read_json, write_json, new_output
from pet_vns_prediction.preprocessing import extract_sgtm_files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--study-id', help='Explicit verified analysis key; default removes only the sub- naming prefix')
    ap.add_argument('--variant', choices=['pvc54','pvc4','pvc6','matched_noPVC'], default='pvc54')
    ap.add_argument('--stats', type=Path, required=True, help='gtm.stats.dat')
    ap.add_argument('--coefficients', type=Path, required=True, help='Unrounded gtm.nii.gz')
    ap.add_argument('--segmentation', type=Path, required=True, help='High-resolution gtmseg.mgz')
    ap.add_argument('--xtx', type=Path, required=True)
    ap.add_argument('--xty', type=Path, required=True)
    ap.add_argument('--feature-config', type=Path, default=ROOT/'configs/pvc_regions.json')
    ap.add_argument('--qc-table', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    require_review(a.qc_table, a.subject, 'pvc_segmentation_review')
    require_review(a.qc_table, a.subject, 'pvc_fit_review')
    result = extract_sgtm_files(a.stats, a.coefficients, a.segmentation, a.xtx, a.xty, read_json(a.feature_config))
    out = new_output(a.output)
    write_json(out/'pvc_features.json', {'subject': a.subject, 'study_id': a.study_id if a.study_id is not None else a.subject.removeprefix('sub-'),
        'variant': a.variant, 'features': result['pathways_normalized'],
        'reference': '84-label segmentation-volume-weighted regional coefficients', 'details': result})
    print('Wrote matched-segmentation PVC features; nonpositive coefficients and numerical warnings retained.')


if __name__ == '__main__':
    main()
