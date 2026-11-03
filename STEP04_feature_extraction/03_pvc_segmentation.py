"""Optional sGTM segmentation, explicitly in a separate FreeSurfer processing copy."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import require_review, run_commands
from pet_vns_prediction.preprocessing import gtmseg_command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--subjects-dir-copy', type=Path, required=True)
    ap.add_argument('--license', type=Path, required=True)
    ap.add_argument('--confirm-processing-copy', action='store_true', help='Confirm this is not the protected original SUBJECTS_DIR')
    ap.add_argument('--qc-table', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='Fresh log and temporary-work directory')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    require_review(a.qc_table, a.subject, 'segmentation_review')
    if not a.confirm_processing_copy:
        raise ValueError('Prepare and review a separate FreeSurfer subject copy first')
    sub = a.subjects_dir_copy.resolve()/a.subject
    if not (sub/'mri/aparc+aseg.mgz').is_file():
        raise FileNotFoundError('The reviewed FreeSurfer copy is incomplete')
    if (sub/'mri/gtmseg.mgz').exists() or a.output.exists():
        raise FileExistsError('Existing gtmseg/output: reuse after checking, do not rerun')
    if a.execute and not a.license.is_file():
        raise FileNotFoundError('FreeSurfer license is missing')
    os.environ['SUBJECTS_DIR'] = str(a.subjects_dir_copy.resolve())
    os.environ['FS_LICENSE'] = str(a.license.resolve())
    print(json.dumps(run_commands([gtmseg_command(a.subject, a.output.resolve()/'temporary')], a.output, a.execute)))


if __name__ == '__main__':
    main()
