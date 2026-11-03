"""Run one reviewed T1 reconstruction with FreeSurfer 7.4.1; no batch scheduler."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import require_review, run_commands
from pet_vns_prediction.preprocessing import recon_all_command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True, help='BIDS identifier including sub-')
    ap.add_argument('--t1', type=Path, required=True)
    ap.add_argument('--subjects-dir', type=Path, required=True)
    ap.add_argument('--license', type=Path, required=True)
    ap.add_argument('--qc-table', type=Path, required=True)
    ap.add_argument('--log-output', type=Path, required=True)
    ap.add_argument('--cw256', action='store_true', help='Only if this subject was preassigned this option')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    require_review(a.qc_table, a.subject, 'source_review')
    if not a.t1.is_file():
        raise FileNotFoundError(a.t1)
    if (a.subjects_dir / a.subject).exists():
        raise FileExistsError('Existing FreeSurfer subject: use completed results or a separately reviewed recovery procedure')
    if a.execute:
        if not a.license.is_file():
            raise FileNotFoundError('FreeSurfer license is not available')
        result = subprocess.run(['recon-all', '-version'], capture_output=True, text=True, check=True)
        if '7.4.1' not in result.stdout + result.stderr:
            raise ValueError('This recipe requires FreeSurfer 7.4.1')
        a.subjects_dir.mkdir(parents=True, exist_ok=True)
    os.environ['SUBJECTS_DIR'] = str(a.subjects_dir.resolve())
    os.environ['FS_LICENSE'] = str(a.license.resolve())
    command = recon_all_command(a.t1.resolve(), a.subject, threads=8, cw256=a.cw256)
    print(json.dumps(run_commands([command], a.log_output, a.execute)))


if __name__ == '__main__':
    main()
