"""Extract the six structural features from reviewed FreeSurfer statistics."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import ROOT, read_json, write_json, new_output, require_review
from pet_vns_prediction.features import extract_t1_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--study-id', help='Explicit verified analysis key; default removes only the sub- naming prefix')
    ap.add_argument('--aseg-stats', type=Path, required=True)
    ap.add_argument('--left-aparc-stats', type=Path, required=True)
    ap.add_argument('--right-aparc-stats', type=Path, required=True)
    ap.add_argument('--feature-config', type=Path, default=ROOT/'configs/features.json')
    ap.add_argument('--qc-table', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    require_review(a.qc_table, a.subject, 'segmentation_review')
    values = extract_t1_stats(a.aseg_stats, a.left_aparc_stats, a.right_aparc_stats, read_json(a.feature_config))
    out = new_output(a.output)
    write_json(out/'t1_features.json', {'subject': a.subject, 'study_id': a.study_id if a.study_id is not None else a.subject.removeprefix('sub-'),
        'features': values, 'canonical_features': {'t1__'+key: value for key,value in values.items()},
        'definition': 'aseg structure volume / eTIV, or bilateral parcel-area-weighted thickness in mm'})
    print('Wrote six T1 morphometric features; no PET-grid volume substitution.')


if __name__ == '__main__':
    main()
