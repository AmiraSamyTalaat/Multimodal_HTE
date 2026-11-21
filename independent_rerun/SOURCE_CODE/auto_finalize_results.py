"""Watch completed Kaggle downloads, revalidate, and refresh the patient-free ZIP.

The Kaggle orchestrators run separately. This watcher never starts training.
"""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGES = ('stage6a', 'stage6b', 'stage6c')


def stamp(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def run(*args):
    p = subprocess.run([sys.executable, str(BASE/args[0]), *args[1:]], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stamp(f'{args[0]} return={p.returncode} {p.stdout.strip()[-500:]}')
    return p.returncode


def main():
    seen = set()
    while True:
        changed = False
        for stage in STAGES:
            d = BASE/'downloads'/f'b01-p1-independent-{stage}-aggregate-cpu'
            checkpoint = d/'B01_INDEPENDENT_CHECKPOINT.zip'
            manifest = d/'KERNEL_RUN_MANIFEST.json'
            if not checkpoint.is_file() or not manifest.is_file():
                continue
            metadata = json.loads(manifest.read_text(encoding='utf-8'))
            key = (stage, metadata.get('checkpoint_sha256'))
            if key in seen:
                continue
            if metadata.get('runner_returncode') != 0:
                stamp(f'{stage} aggregate has nonzero runner return code')
                continue
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != key[1]:
                stamp(f'{stage} checkpoint SHA mismatch')
                continue
            if run('revalidate_saved_aggregate.py', '--stage', stage, '--backend', 'cpu') == 0:
                seen.add(key)
                changed = True
        if changed:
            run('package_actual_results.py', '--backend', 'cpu')
        if len(seen) == len(STAGES):
            stamp('ALL_SELECTED_STAGE_AGGREGATES_VALIDATED')
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
