"""Resume benchmark Stage 4 Kaggle batches from the saved GPU batch 1.

Uses private kernel outputs as durable checkpoints. It never uploads patient data.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
OWNER = 'prinprin'
PREFIX = 'b01-p1-independent-stage4-'
CHAIN = [('batch1', 2), ('batch2', 4), ('batch3', 4), ('batch4', 4), ('aggregate', 0)]
SOURCES = ['prinprin/b01-causaltx-mm-core-runtime', 'prinprin/b01-p1-independent-rerun-code']


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def cli(*args, timeout=120):
    env = os.environ.copy()
    env.update(TEMP=str(BASE), TMP=str(BASE), TMPDIR=str(BASE))
    p = subprocess.run(['kaggle', *args], text=True, capture_output=True, env=env, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def status(slug):
    try:
        code, output = cli('kernels', 'status', f'{OWNER}/{slug}')
    except subprocess.TimeoutExpired:
        return 'UNKNOWN:Kaggle status request timed out; retrying without resubmission'
    if code and any(token in output.lower() for token in ('404', 'wrong kernel slug', 'does not exist', 'permission denied')):
        return 'MISSING'
    match = re.search(r'KernelWorkerStatus\.([A-Z_]+)', output)
    return match.group(1) if match else ('MISSING' if 'not found' in output.lower() or 'could not' in output.lower() else 'UNKNOWN:' + output[-300:])


def prepare(slug, previous, max_jobs, backend):
    directory = BASE / slug
    directory.mkdir(exist_ok=True)
    code = ("import os\nimport runpy\nfrom pathlib import Path\n"
            "os.environ['B01_STAGES']='stage4'\n"
            f"os.environ['B01_MAX_JOBS']='{max_jobs}'\n"
            f"os.environ['B01_PREVIOUS']='{previous}'\n"
            "paths=list(Path('/kaggle/input').rglob('b01_independent_kernel.py'))\n"
            "assert len(paths)==1, paths\n"
            "runpy.run_path(str(paths[0]), run_name='__main__')\n")
    (directory/'run.py').write_text(code, encoding='utf-8')
    metadata = dict(id=f'{OWNER}/{slug}', title=slug.replace('-', ' '), code_file='run.py',
                    language='python', kernel_type='script', is_private=True,
                    enable_gpu=backend == 'gpu', enable_internet=False, dataset_sources=SOURCES,
                    kernel_sources=[f'{OWNER}/{previous}'], competition_sources=[], model_sources=[])
    save(directory/'kernel-metadata.json', metadata)
    return directory


def download_and_validate(slug):
    directory = BASE/'downloads'/slug
    directory.mkdir(parents=True, exist_ok=True)
    code, output = cli('kernels', 'output', f'{OWNER}/{slug}', '-p', str(directory), '-o',
                       '--file-pattern', 'B01_INDEPENDENT_CHECKPOINT.zip|INDEPENDENT_RERUN_COMPARISON.json|KERNEL_RUN_MANIFEST.json', timeout=300)
    if code:
        raise RuntimeError(f'Cannot download persistent output for {slug}: {output[-800:]}')
    cp = directory/'B01_INDEPENDENT_CHECKPOINT.zip'
    manifest = json.loads((directory/'KERNEL_RUN_MANIFEST.json').read_text(encoding='utf-8'))
    comparison = json.loads((directory/'INDEPENDENT_RERUN_COMPARISON.json').read_text(encoding='utf-8'))
    if sha(cp) != manifest['checkpoint_sha256']:
        raise RuntimeError(f'Checkpoint hash mismatch: {slug}')
    with zipfile.ZipFile(cp) as z:
        if z.testzip() is not None or any(n.endswith('.npz') or '/predictions/' in n for n in z.namelist()):
            raise RuntimeError(f'Invalid or private individual-level content in checkpoint: {slug}')
    if manifest['runner_returncode'] != 0:
        raise RuntimeError(f'Training failed in {slug}')
    return dict(slug=slug, completed_utc=manifest['utc'], checkpoint_sha256=manifest['checkpoint_sha256'],
                scientific_comparison_pass=comparison['all_comparisons_pass'], comparison_failures=sum(not c['pass'] for c in comparison['checks']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--backend', choices=('gpu','cpu'), default='gpu')
    args = p.parse_args()
    state_path = BASE/('RUN_STATUS.json' if args.backend == 'gpu' else 'RUN_STATUS_CPU.json')
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'batches': []}
    for index, (name, max_jobs) in enumerate(CHAIN):
        slug = PREFIX + name + ('-cpu' if args.backend == 'cpu' else '')
        previous = PREFIX + CHAIN[index-1][0] + ('-cpu' if args.backend == 'cpu' else '') if index else None
        if index and status(previous) != 'COMPLETE':
            raise RuntimeError(f'Previous batch {previous} is not COMPLETE')
        if index:
            prepare(slug, previous, max_jobs, args.backend)
        retries = 0
        while True:
            current = status(slug)
            print(utc(), slug, current, flush=True)
            if current == 'COMPLETE':
                result = download_and_validate(slug)
                state['batches'] = [x for x in state['batches'] if x['slug'] != slug] + [result]
                state['updated_utc'] = utc()
                save(state_path, state)
                break
            if current in ('MISSING', 'ERROR'):
                if current == 'ERROR':
                    _, logs = cli('kernels', 'logs', f'{OWNER}/{slug}')
                    logpath = BASE/'RUN_LOGS'/f'{slug}_attempt{retries}.json'
                    logpath.parent.mkdir(exist_ok=True)
                    logpath.write_text(logs, encoding='utf-8')
                    retries += 1
                if index == 0 or retries > 2:
                    raise RuntimeError(f'{slug} failed; see RUN_LOGS and Kaggle output')
                code, out = cli('kernels', 'push', '-p', str(BASE/slug), timeout=120)
                if 'Maximum batch' in out and 'session count' in out:
                    print(utc(), 'KAGGLE_QUOTA_WAIT', slug, flush=True)
                    time.sleep(60)
                    continue
                if code or 'successfully pushed' not in out:
                    raise RuntimeError(f'Push failed: {out[-800:]}')
            elif not (current in ('QUEUED', 'RUNNING') or current.startswith('UNKNOWN:')):
                raise RuntimeError(f'Unexpected Kaggle status: {current}')
            time.sleep(args.poll_seconds)
    print('STAGE4 COMPLETE; inspect scientific comparison and PRIMARY_GATE before proceeding.')


if __name__ == '__main__':
    main()
