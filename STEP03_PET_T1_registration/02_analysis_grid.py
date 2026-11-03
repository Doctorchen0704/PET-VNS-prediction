"""Sample the native PET once to the individual 2-mm grid using a reviewed LTA."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import require_review, run_commands
from pet_vns_prediction.preprocessing import analysis_grid_commands


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--pet', type=Path, required=True, help='Original selected native PET')
    ap.add_argument('--t1', type=Path, required=True)
    ap.add_argument('--brainmask', type=Path, required=True)
    ap.add_argument('--segmentation', type=Path, required=True, help='FreeSurfer aparc+aseg.mgz')
    ap.add_argument('--lta', type=Path, required=True)
    ap.add_argument('--qc-table', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    require_review(a.qc_table, a.subject, 'segmentation_review')
    require_review(a.qc_table, a.subject, 'registration_review')
    paths = [a.pet, a.t1, a.brainmask, a.segmentation, a.lta]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    if a.output.exists():
        raise FileExistsError('Use a fresh analysis-grid directory')
    args = [p.resolve() for p in paths]
    print(json.dumps(run_commands(analysis_grid_commands(*args, a.output.resolve()), a.output, a.execute)))


if __name__ == '__main__':
    main()
