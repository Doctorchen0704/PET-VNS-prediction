"""Convert manually reviewed DICOM folders; does not select series or start STEP02."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import read_json
from pet_vns_prediction.intake import convert_dicom


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--raw-root', type=Path, required=True)
    ap.add_argument('--staging', type=Path, required=True, help='A new conversion directory')
    ap.add_argument('--converter', type=Path, required=True, help='Explicit dcm2niix executable')
    ap.add_argument('--execute', action='store_true', help='Without this flag, print commands only')
    a = ap.parse_args()
    result = convert_dicom(read_json(a.manifest), a.raw_root, a.staging, a.converter, execute=a.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
