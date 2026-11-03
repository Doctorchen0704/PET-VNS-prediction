"""Small shared I/O helpers; no scheduler, downloads, or automatic next steps."""
from pathlib import Path
import csv
import json
import math
import os
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from threadpoolctl import threadpool_limits
import numpy as _numpy
import scipy.linalg as _scipy_linalg
_NUMERICAL_THREADS = threadpool_limits(limits=1)


def read_json(path):
    from pet_vns_prediction.intake import load_json
    return load_json(path)


def _json_ready(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, obj):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_ready(obj), ensure_ascii=False, indent=2,
                                 allow_nan=False)+'\n', encoding='utf-8')


def new_output(path):
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError('Use a fresh output directory: '+str(target))
    target.mkdir(parents=True)
    return target


def load_model_input(path):
    from pet_vns_prediction.cli import _validated_input
    from pet_vns_prediction import paper_settings
    data = read_json(path)
    settings = paper_settings()
    _, x, y, _, _ = _validated_input(data, ['C','CP','CPT'], settings)
    return data, x, y, settings


def require_review(csv_path, subject, field):
    path = Path(csv_path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t' if path.suffix.lower()=='.tsv' else ','))
    matched = [r for r in rows if r.get('subject',r.get('study_id',r.get('participant_id'))) == subject]
    if len(matched) != 1:
        raise ValueError('One explicit manual QC row is required for '+subject)
    value = matched[0].get(field, '').strip().upper()
    if value not in ('PASS','PASS_WITH_ATTENTION'):
        raise ValueError('Manual review not accepted for '+subject+' / '+field)
    return matched[0]


def run_commands(commands, output_dir, execute=False):
    """Print a single step's commands; execution is opt-in and never resumes."""
    commands = [[str(arg) for arg in command] for command in commands]
    for command in commands:
        print(subprocess.list2cmdline(command) if os.name=='nt' else shlex.join(command))
    if not execute:
        return {'status':'COMMANDS_ONLY_NOT_EXECUTED','commands':commands}
    out = Path(output_dir).resolve()
    if out.exists():
        # A step may create its fresh directory immediately before this helper.
        # Existing *content* is never accepted as an automatic resume.
        if any(out.iterdir()):
            raise FileExistsError('Step output is not empty; no automatic resume')
    else:
        out.mkdir(parents=True)
    with (out/'commands.log').open('x', encoding='utf-8') as log:
        for command in commands:
            log.write((subprocess.list2cmdline(command) if os.name=='nt' else shlex.join(command))+'\n')
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    return {'status':'COMMANDS_FINISHED_MANUAL_QC_REQUIRED','commands':commands}
