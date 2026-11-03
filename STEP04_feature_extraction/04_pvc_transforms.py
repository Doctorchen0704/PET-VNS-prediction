"""Prepare explicit header-based transforms for already T1-aligned PET; no PVC fit."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import require_review, run_commands
from pet_vns_prediction.preprocessing import pvc_geometry_commands


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--aligned-pet', type=Path, required=True, help='Raw-intensity, reviewed T1-aligned 2-mm PET, not native PET or ratio PET')
    ap.add_argument('--conformed-t1', type=Path, required=True, help='FreeSurfer processing-copy mri/nu.mgz used by the original PVC geometry')
    ap.add_argument('--gtmseg-lta', type=Path, required=True)
    ap.add_argument('--qc-table', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    require_review(a.qc_table, a.subject, 'registration_review')
    require_review(a.qc_table, a.subject, 'coverage_review')
    for p in (a.aligned_pet, a.conformed_t1, a.gtmseg_lta):
        if not p.is_file():
            raise FileNotFoundError(p)
    if a.output.exists():
        raise FileExistsError('A new geometry directory is required')
    out = a.output.resolve()
    commands = pvc_geometry_commands(a.aligned_pet.resolve(), a.conformed_t1.resolve(), a.gtmseg_lta.resolve(), a.subject,
                                    out/'pet_to_anat.lta', out/'seg_to_pet.lta')
    print(json.dumps(run_commands(commands, out, a.execute)))


if __name__ == '__main__':
    main()
