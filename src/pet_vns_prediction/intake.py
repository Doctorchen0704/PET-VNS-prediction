"""Explicit-manifest DICOM conversion and reviewed BIDS promotion.

This is an orchestration adapter, not a patient/series classifier. Manifests,
conversion logs and outputs are private. No study-specific mapping is bundled.
The converter flags reproduce the historical conversion recipe. Clinical and
image identity decisions must be supplied from the controlled review record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def load_json(path):
    """Reject duplicate keys/nonfinite JSON rather than silently repairing them."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON constant; use explicit null for missing data')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=unique, parse_constant=invalid)


def _rows(manifest):
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('schema_version=1 manifest required')
    rows = manifest.get('rows')
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError('Nonempty list of manifest row objects required')
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(root, relative):
    root = Path(root).resolve()
    # Explicit forward-slash relative paths work on Windows and Linux.
    if not isinstance(relative, str) or '\\' in relative or ':' in relative or '\x00' in relative:
        raise ValueError('Use forward-slash relative paths, without drive letters')
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Path must stay inside its declared root')
    for part in relative.split('/'):
        if not part or part in ('.', '..') or part.rstrip(' .') != part:
            raise ValueError('Empty, dot, or trailing-dot/space path component')
        if any(ord(char) < 32 for char in part) or any(char in '<>"|?*' for char in part):
            raise ValueError('Invalid portable path component')
        if re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', part.split('.')[0]):
            raise ValueError('Windows device names are not ordinary source paths')
    target = (root / rel).resolve()
    if target == root or root not in target.parents:
        raise ValueError('Path escapes declared root, including via links')
    return target


def new_output(path, protected_roots=()):
    target = Path(path).resolve()
    for value in protected_roots:
        protected = Path(value).resolve()
        if target == protected or protected in target.parents or target in protected.parents:
            raise ValueError('Output and source roots must not overlap')
    if target.exists():
        raise FileExistsError('Use a new output directory; no overwrite or automatic resume')
    return target


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def conversion_plan(manifest, raw_root, output_root, converter):
    """Validate caller-reviewed rows; return commands without executing them.

    Manifest rows: subject (without sub-), modality (T1/PET), source_relative,
    identity_reviewed=true, series_reviewed=true. Source folders may contain
    multiple series; promotion uses a separately reviewed explicit manifest.
    """
    rows = _rows(manifest)
    if not Path(raw_root).is_dir():
        raise ValueError('Raw source root must exist')
    out = new_output(output_root, (raw_root,))
    commands, seen = [], set()
    for row in rows:
        sid, modality = row['subject'], row['modality']
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9]+', sid) or modality not in ('T1', 'PET'):
            raise ValueError('Simple BIDS subject label and T1/PET modality required')
        if row.get('identity_reviewed') is not True or row.get('series_reviewed') is not True:
            raise ValueError('Source identity/series review has not been supplied')
        key = sid.casefold(), modality
        if key in seen:
            raise ValueError('Duplicate subject/modality source row')
        seen.add(key)
        source = safe_relative(raw_root, row['source_relative'])
        if not source.is_dir() or not any(p.is_file() for p in source.rglob('*')):
            raise ValueError('Missing or empty source directory')
        if any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Linked descendants require explicit source inventory outside this converter')
        destination = out / sid / ('t1_all_series' if modality == 'T1' else 'pet_all_series')
        commands.append({'subject': sid, 'modality': modality, 'output': str(destination),
            'command': [str(converter), '-b', 'y', '-ba', 'y', '-z', 'y', '-w', '0',
                        '-f', 'series-%s_desc-%p', '-o', str(destination), str(source)]})
    return commands


def convert_dicom(manifest, raw_root, output_root, converter, *, execute=False):
    """By default return the plan. --execute belongs to an authorized new run.

    Requires the historical dcm2niix v1.0.20250505 build. stdout/stderr are kept
    locally and may contain confidential source metadata. A failed run is never
    retried or deleted. This function does not claim visual or BIDS compliance.
    """
    commands = conversion_plan(manifest, raw_root, output_root, converter)
    if not execute:
        return {'status': 'PLAN_ONLY', 'commands': commands}
    executable = Path(converter).resolve()
    if not executable.is_file():
        raise ValueError('Converter must be an explicit existing executable file')
    version = subprocess.run([str(executable), '--version'], capture_output=True, text=True, check=False)
    version_text = version.stdout + version.stderr
    if version.returncode != 0 or 'v1.0.20250505' not in version_text:
        raise ValueError('Exact historical converter version not confirmed')
    out = new_output(output_root, (raw_root,)); out.mkdir(parents=True)
    receipt = {'status': 'RUNNING', 'converter_sha256': sha256_file(executable),
               'converter_version': version_text.strip(), 'steps': [],
               'identity_and_series_review': 'Caller declaration; not independently verified by code',
               'visual_QC': 'NOT_PERFORMED', 'BIDS_validator': 'NOT_RUN'}
    json_write(out / 'conversion_receipt_PRIVATE.json', receipt)
    try:
        for step in commands:
            destination = Path(step['output']); destination.mkdir(parents=True)
            step['command'][0] = str(executable)
            with (destination / 'converter_PRIVATE.log').open('xb') as log:
                result = subprocess.run(step['command'], stdout=log, stderr=subprocess.STDOUT, check=False)
            outputs = sorted(destination.glob('*.nii.gz'))
            pairs = all(p.with_name(p.name[:-7]+'.json').is_file() for p in outputs)
            receipt['steps'].append({**step, 'exit_code': result.returncode,
                'nifti_count': len(outputs), 'all_json_pairs': pairs,
                'files': [{'name':p.name,'sha256':sha256_file(p)} for p in outputs]})
            json_write(out / 'conversion_receipt_PRIVATE.json', receipt)
            if result.returncode != 0 or not outputs or not pairs:
                raise RuntimeError('Conversion failed or outputs incomplete; see private log')
        receipt['status'] = 'CONVERTED_PENDING_SERIES_AND_IMAGE_QC'
    except BaseException:
        receipt['status'] = 'FAILED_PRESERVED'
        json_write(out / 'conversion_receipt_PRIVATE.json', receipt)
        raise
    json_write(out / 'conversion_receipt_PRIVATE.json', receipt)
    return receipt


def promote_selected(manifest, staging_root, bids_root):
    """Copy exact reviewed series to a fresh BIDS-organized directory.

    Each row supplies source_nii/source_json, both SHA256s, target_nii and
    review_pass=true. Optional pet_metadata_patch is an explicitly reviewed
    dictionary with metadata_patch_reviewed=true (never calculated from assumed injection timing). Source files
    remain byte-identical. BIDS-validator results must be recorded separately.
    """
    rows = _rows(manifest)
    if not Path(staging_root).is_dir():
        raise ValueError('Staging source root must exist')
    out = new_output(bids_root, (staging_root,))
    selected, seen = [], set()
    for row in rows:
        if row.get('review_pass') is not True:
            raise ValueError('Explicit identity and series selection review required')
        nii, sidecar = [safe_relative(staging_root, row[k]) for k in ('source_nii','source_json')]
        target = safe_relative(out, row['target_nii'])
        if not re.fullmatch(r'sub-[A-Za-z0-9]+/(anat|pet)/sub-[A-Za-z0-9]+(?:_[A-Za-z0-9-]+)*_(?:T1w|pet)\.nii\.gz', row['target_nii']):
            raise ValueError('Unexpected anatomical/PET BIDS target')
        if target.parts[-3] != target.name.split('_')[0]:
            raise ValueError('Subject directory and filename disagree')
        if (target.parent.name == 'anat') != target.name.endswith('_T1w.nii.gz'):
            raise ValueError('BIDS modality directory and filename suffix disagree')
        if str(target).casefold() in seen:
            raise ValueError('Duplicate BIDS target')
        seen.add(str(target).casefold())
        if not nii.name.endswith('.nii.gz') or sidecar != nii.with_name(nii.name[:-7] + '.json'):
            raise ValueError('Require the paired NIfTI/JSON source basenames')
        for p,k in ((nii,'nii_sha256'),(sidecar,'json_sha256')):
            if sha256_file(p) != row[k]: raise ValueError('Selected-series source hash mismatch')
        metadata = load_json(sidecar)
        patch = row.get('pet_metadata_patch', {})
        if not isinstance(metadata, dict) or not isinstance(patch, dict):
            raise ValueError('Sidecar and metadata patch must be JSON objects')
        if patch and (target.parent.name != 'pet' or row.get('metadata_patch_reviewed') is not True):
            raise ValueError('PET metadata changes require an explicitly reviewed PET patch')
        metadata.update(patch)
        json.dumps(metadata, allow_nan=False)
        selected.append((nii,target,metadata,row))
    out.mkdir(parents=True)
    receipts=[]
    receipt = {'status':'RUNNING','rows':receipts,
               'selection_review':'Caller declaration; not independently verified by code',
               'visual_QC':'NOT_PERFORMED','BIDS_validator':'NOT_RUN'}
    json_write(out/'promotion_receipt_PRIVATE.json',receipt)
    try:
        for source,target,metadata,row in selected:
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as inp, target.open('xb') as dest:
                shutil.copyfileobj(inp, dest)
            json_write(target.with_name(target.name[:-7]+'.json'),metadata)
            if sha256_file(target) != row['nii_sha256'] or sha256_file(source) != row['nii_sha256']:
                raise ValueError('Source changed or destination copy hash mismatch')
            receipts.append({'target':row['target_nii'],'source_nii_sha256':row['nii_sha256'],
                             'source_json_sha256':row['json_sha256'],'metadata_patch':row.get('pet_metadata_patch',{})})
            json_write(out/'promotion_receipt_PRIVATE.json',receipt)
    except BaseException:
        receipt['status']='FAILED_PRESERVED'
        json_write(out/'promotion_receipt_PRIVATE.json',receipt)
        raise
    json_write(out/'dataset_description.json',{'Name':'Authorized PET/T1 reproduction',
        'BIDSVersion':'1.10.0','DatasetType':'raw','GeneratedBy':[{'Name':'dcm2niix','Version':'v1.0.20250505'}]})
    receipt['status']='COPIED_PENDING_BIDS_AND_VISUAL_QC'
    json_write(out/'promotion_receipt_PRIVATE.json',receipt)
    return receipts
