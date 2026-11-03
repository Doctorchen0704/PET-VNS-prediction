"""Copy exact reviewed NIfTI/JSON pairs into a fresh BIDS-organized directory."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import read_json
from pet_vns_prediction.intake import promote_selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--staging', type=Path, required=True)
    ap.add_argument('--bids', type=Path, required=True, help='A new directory')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    data = read_json(a.manifest)
    if a.bids.exists():
        raise FileExistsError('Never promote into existing BIDS data')
    if not a.execute:
        print(json.dumps({'status': 'PLAN_ONLY_NOT_COPIED', 'selected_rows': len(data['rows']),
                          'bids': str(a.bids), 'BIDS_validator': 'NOT_RUN'}))
        return
    promote_selected(data, a.staging, a.bids)
    print('Copied reviewed series. Run a BIDS validator and inspect images separately.')


if __name__ == '__main__':
    main()
