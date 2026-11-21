"""Private Kaggle execution: frozen original benchmark rerun with durable checkpoint."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy
import pandas
import scipy
import sklearn
import torch

ROOT = Path('/kaggle/working')
WORK = ROOT / 'private_work'
HERE = Path(__file__).resolve().parent
STAGES = os.environ.get('B01_STAGES', 'stage4')
MAX_JOBS = int(os.environ.get('B01_MAX_JOBS', '1'))
PREVIOUS = os.environ.get('B01_PREVIOUS', '')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def checkpoint():
    target = ROOT / 'B01_INDEPENDENT_CHECKPOINT.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for source in sorted(WORK.rglob('*')):
            if not source.is_file():
                continue
            rel = source.relative_to(WORK)
            if rel.as_posix() == 'B01/02_DATA/processed/ispy2_core_multimodal.npz':
                continue
            stage7_aggregate_input = STAGES == 'stage7b' and source.name == 'patient_dev_predictions.csv'
            if not stage7_aggregate_input and ('predictions' in rel.parts or source.suffix == '.npz' or 'prediction' in source.name.lower() or source.name.lower().startswith('patient_')):
                continue
            z.write(source, rel.as_posix())
    return target


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    if PREVIOUS:
        prior = Path('/kaggle/input') / PREVIOUS / 'B01_INDEPENDENT_CHECKPOINT.zip'
        if not prior.is_file():
            matches = list(Path('/kaggle/input').rglob('B01_INDEPENDENT_CHECKPOINT.zip')) + list(Path('/kaggle/input').rglob('B01_INDEPENDENT_CHECKPOINT.bin'))
            if len(matches) != 1:
                raise FileNotFoundError(f'Prior checkpoint missing/ambiguous: {matches}')
            prior = matches[0]
        with zipfile.ZipFile(prior) as z:
            z.extractall(WORK)
        print('RESTORED_CHECKPOINT', prior, sha(prior), flush=True)
    matches = [p for p in Path('/kaggle/input').rglob('ispy2_core_multimodal.npz')
               if 'b01-causaltx-mm-core-runtime' in p.parts]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one private processed dataset, found {matches}')
    data = matches[0]
    release = HERE / 'B01_REVIEW_CODE_ARCHIVE.bin'
    command = [sys.executable, str(HERE/'RUN_FROZEN_INDEPENDENT_RERUN.py'),
               '--release', str(release), '--private-data', str(data), '--work', str(WORK),
               '--stages', STAGES, '--max-jobs', str(MAX_JOBS)]
    print('COMMAND', command, flush=True)
    proc = subprocess.run(command, text=True, capture_output=True)
    (WORK/'kernel_driver.log').write_text(proc.stdout+'\nSTDERR\n'+proc.stderr, encoding='utf-8')
    print(proc.stdout, proc.stderr, flush=True)
    comparison = ROOT/'INDEPENDENT_RERUN_COMPARISON.json'
    compare_command = [sys.executable, str(HERE/'COMPARE_INDEPENDENT_RERUN.py'),
                       '--original-code-archive', str(release), '--rerun-results', str(WORK/'B01'/'06_RESULTS'),
                       '--rerun-work', str(WORK), '--stages', STAGES, '--report', str(comparison)]
    if any(not (WORK/'progress'/f'{stage.strip()}_aggregation_COMPLETE.json').is_file() for stage in STAGES.split(',')):
        compare_command.append('--per-run-only')
    compare = subprocess.run(compare_command, text=True, capture_output=True)
    (WORK/'comparison_driver.log').write_text(compare.stdout+'\nSTDERR\n'+compare.stderr, encoding='utf-8')
    print('COMPARISON', compare.returncode, compare.stdout, compare.stderr, flush=True)
    # Kaggle persists every file left in /kaggle/working. Keep participant-level
    # matrices and predictions only for this private training session.
    for private_file in WORK.rglob('*'):
        stage7_aggregate_input = STAGES == 'stage7b' and private_file.name == 'patient_dev_predictions.csv'
        if private_file.is_file() and not stage7_aggregate_input and (private_file.suffix == '.npz' or 'prediction' in private_file.name.lower() or private_file.name.lower().startswith('patient_')):
            private_file.unlink()
    cp = checkpoint()
    manifest = {'utc': utc(), 'status': 'TECHNICAL_COMPLETE' if proc.returncode == 0 else 'FAILED',
                'scientific_comparison_pass': compare.returncode == 0,
                'stage_selection': STAGES, 'max_jobs': MAX_JOBS, 'previous_checkpoint': PREVIOUS,
                'python': sys.version, 'platform': platform.platform(),
                'environment_versions': {'numpy': numpy.__version__, 'pandas': pandas.__version__,
                                         'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__,
                                         'torch': torch.__version__},
                'data_sha256': sha(data), 'release_sha256': sha(release),
                'runner_sha256': sha(HERE/'RUN_FROZEN_INDEPENDENT_RERUN.py'),
                'comparator_sha256': sha(HERE/'COMPARE_INDEPENDENT_RERUN.py'),
                'checkpoint_sha256': sha(cp), 'runner_returncode': proc.returncode,
                'comparator_returncode': compare.returncode}
    (ROOT/'KERNEL_RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('FINAL_MANIFEST', json.dumps(manifest), flush=True)
    if proc.returncode:
        raise RuntimeError('Training failed; checkpoint preserved in output')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
