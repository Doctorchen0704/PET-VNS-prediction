"""Hash the exact series explicitly selected in a human-reviewed TSV; no heuristic selection."""
from pathlib import Path
import argparse
import csv
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_support import write_json
from pet_vns_prediction.intake import safe_relative, sha256_file, load_json


def finite_json_object(text):
    def invalid(value):
        raise ValueError('Nonfinite metadata constant: '+value)
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('Duplicated metadata key: '+key)
            obj[key] = value
        return obj
    result = json.loads(text, parse_constant=invalid, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError('Metadata patch must be an object')
    return result


def make_selection(tsv, staging):
    with Path(tsv).open(encoding='utf-8-sig', newline='') as f:
        source_rows = list(csv.DictReader(f, delimiter='\t'))
    if not source_rows:
        raise ValueError('A nonempty manually selected-series TSV is required')
    rows = []
    seen = set()
    for row in source_rows:
        if row.get('review_pass', '').upper() not in ('PASS', 'PASS_WITH_ATTENTION'):
            raise ValueError('Do not promote series before source identity/series review')
        nii = safe_relative(staging, row['source_nii'])
        sidecar = safe_relative(staging, row['source_json'])
        if not nii.is_file() or not sidecar.is_file():
            raise FileNotFoundError('A selected NIfTI/JSON pair is missing')
        if not nii.name.endswith('.nii.gz') or sidecar != nii.with_name(nii.name.removesuffix('.nii.gz') + '.json'):
            raise ValueError('NIfTI and JSON must be a paired source basename')
        target = row['target_nii']
        if target.casefold() in seen:
            raise ValueError('Duplicate target; resolve series choice manually')
        seen.add(target.casefold())
        item = {'source_nii': row['source_nii'], 'source_json': row['source_json'],
                'target_nii': target, 'nii_sha256': sha256_file(nii),
                'json_sha256': sha256_file(sidecar), 'review_pass': True,
                'review_decision': row['review_pass'].upper(),
                'attention_note': row.get('attention_note', '')}
        patch = row.get('pet_metadata_patch_json', '').strip()
        if patch:
            if row.get('metadata_patch_reviewed', '').upper() != 'YES':
                raise ValueError('Metadata patches need separate explicit review')
            p = finite_json_object(patch)
            item.update(pet_metadata_patch=p, metadata_patch_reviewed=True)
        load_json(sidecar)
        rows.append(item)
    return {'schema_version': 1, 'rows': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection-tsv', type=Path, required=True)
    ap.add_argument('--staging', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='A new manifest JSON')
    a = ap.parse_args()
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError('Selection manifest already exists')
    result = make_selection(a.selection_tsv, a.staging)
    write_json(out, result)
    print(f"Wrote {len(result['rows'])} explicit selections; no image was copied")


if __name__ == '__main__':
    main()
