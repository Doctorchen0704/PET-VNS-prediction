"""Estimate one six-DOF native PET-to-T1 transform and make a review image."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import require_review, run_commands
from pet_vns_prediction.preprocessing import registration_command


def commands(pet, t1, brainmask, out):
    out = Path(out)
    lta = out/'pet_to_T1.lta'
    return [registration_command(pet, t1, brainmask, lta, out/'params.txt', out/'final_cost.txt', threads=4),
            ['mri_vol2vol', '--mov', str(pet), '--targ', str(t1), '--lta', str(lta),
             '--o', str(out/'registered_review_pet.nii.gz'), '--trilin']]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--pet', type=Path, required=True, help='Selected native PET, not previously resampled PET')
    ap.add_argument('--t1', type=Path, required=True, help='FreeSurfer anatomical reference from the frozen queue')
    ap.add_argument('--brainmask', type=Path, required=True)
    ap.add_argument('--qc-table', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    require_review(a.qc_table, a.subject, 'source_review')
    require_review(a.qc_table, a.subject, 'segmentation_review')
    for p in (a.pet, a.t1, a.brainmask):
        if not p.is_file():
            raise FileNotFoundError(p)
    if a.output.exists():
        raise FileExistsError('Use a fresh registration directory; no implicit reinitialization')
    print(json.dumps(run_commands(commands(a.pet.resolve(), a.t1.resolve(), a.brainmask.resolve(), a.output.resolve()), a.output, a.execute)))


if __name__ == '__main__':
    main()
