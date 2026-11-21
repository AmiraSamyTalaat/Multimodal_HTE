"""Resume only scientifically relevant frozen Stage 6A/6B/6C/7B grids.

The 6B branch uses the archived, pre-TEST Stage6A selection as a frozen
dependency. That dependency does not claim independent regeneration of 6A.
"""
import argparse
import json
import time
from pathlib import Path

from orchestrate_stage4 import BASE, OWNER, SOURCES, cli, download_and_validate, save, status, utc

TOTAL = {'stage6a': 10, 'stage6b': 15, 'stage6c': 10, 'stage7b': 12}


def slug(stage, name, backend):
    return f'b01-p1-independent-{stage}-{name}' + ('-cpu' if backend == 'cpu' else '')


def prepare(stage, name, previous, jobs, backend):
    name_slug = slug(stage, name, backend)
    directory = BASE/name_slug
    directory.mkdir(exist_ok=True)
    source = ("import os\nimport runpy\nfrom pathlib import Path\n"
              f"os.environ['B01_STAGES']='{stage}'\n"
              f"os.environ['B01_MAX_JOBS']='{jobs}'\n"
              f"os.environ['B01_PREVIOUS']='{previous}'\n"
              "paths=list(Path('/kaggle/input').rglob('b01_independent_kernel.py'))\n"
              "assert len(paths)==1, paths\n"
              "runpy.run_path(str(paths[0]), run_name='__main__')\n")
    (directory/'run.py').write_text(source, encoding='utf-8')
    metadata = dict(id=f'{OWNER}/{name_slug}', title=name_slug.replace('-', ' '), code_file='run.py',
                    language='python', kernel_type='script', is_private=True,
                    enable_gpu=backend == 'gpu', enable_internet=False,
                    dataset_sources=SOURCES, kernel_sources=[f'{OWNER}/{previous}'],
                    competition_sources=[], model_sources=[])
    save(directory/'kernel-metadata.json', metadata)
    return directory


def watch(stage, backend, poll):
    remaining = TOTAL[stage] - 1
    jobs = []
    while remaining:
        amount = min(3, remaining)
        jobs.append(amount)
        remaining -= amount
    chain = [('smoke', 1)] + [(f'batch{i}', size) for i, size in enumerate(jobs, 1)] + [('aggregate', 0)]
    state_path = BASE/f'RUN_STATUS_{stage.upper()}_{backend.upper()}.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'stage':stage,'backend':backend,'batches':[]}
    for index, (name, count) in enumerate(chain):
        current_slug = slug(stage, name, backend)
        previous = slug(stage, chain[index-1][0], backend) if index else None
        if previous and status(previous) != 'COMPLETE':
            raise RuntimeError(f'Previous stage batch is not complete: {previous}')
        if index:
            prepare(stage, name, previous, count, backend)
        retries = 0
        while True:
            current = status(current_slug)
            print(utc(), current_slug, current, flush=True)
            if current == 'COMPLETE':
                result = download_and_validate(current_slug)
                state['batches'] = [x for x in state['batches'] if x['slug'] != current_slug] + [result]
                state['updated_utc'] = utc()
                save(state_path, state)
                break
            if current == 'ERROR':
                _, logs = cli('kernels','logs',f'{OWNER}/{current_slug}')
                log = BASE/'RUN_LOGS'/f'{current_slug}_attempt{retries}.json'
                log.parent.mkdir(exist_ok=True)
                log.write_text(logs, encoding='utf-8')
                retries += 1
                if not index or retries > 2:
                    raise RuntimeError(f'{current_slug} failed; see {log}')
                current = 'MISSING'
            if current == 'MISSING':
                if not index:
                    raise RuntimeError(f'Smoke kernel not yet pushed: {current_slug}')
                code, output = cli('kernels','push','-p',str(BASE/current_slug))
                if 'Maximum batch' in output and 'session count' in output:
                    print(utc(), 'KAGGLE_QUOTA_WAIT', current_slug, flush=True)
                    time.sleep(60)
                    continue
                if code or 'successfully pushed' not in output:
                    raise RuntimeError(f'Kaggle push failed: {output[-800:]}')
            elif current not in ('QUEUED','RUNNING') and not current.startswith('UNKNOWN:'):
                raise RuntimeError(f'Unexpected Kaggle status {current_slug}: {current}')
            time.sleep(poll)
    print('SELECTED_STAGE_COMPLETE', stage, backend, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=tuple(TOTAL), required=True)
    p.add_argument('--backend', choices=('gpu','cpu'), default='cpu')
    p.add_argument('--poll-seconds', type=int, default=30)
    a = p.parse_args()
    watch(a.stage, a.backend, a.poll_seconds)


if __name__ == '__main__':
    main()
