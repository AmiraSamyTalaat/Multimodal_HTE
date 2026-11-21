"""Recompare a saved private Kaggle checkpoint after comparator corrections.

This command reads completed training output only. It does not rerun a model.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('stage4','stage6a','stage6b','stage6c','stage7b'), required=True)
    parser.add_argument('--backend', choices=('cpu','gpu'), default='cpu')
    args = parser.parse_args()
    suffix = '-cpu' if args.backend == 'cpu' else ''
    slug = f'b01-p1-independent-{args.stage}-aggregate{suffix}'
    downloaded = BASE / 'downloads' / slug
    checkpoint = downloaded / 'B01_INDEPENDENT_CHECKPOINT.zip'
    manifest_path = downloaded / 'KERNEL_RUN_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['runner_returncode'] != 0 or digest(checkpoint) != manifest['checkpoint_sha256']:
        raise SystemExit('Incomplete or SHA-mismatched Kaggle checkpoint')
    extraction = BASE / f'validation_{args.stage}_{args.backend}'
    extraction.mkdir(exist_ok=True)
    with zipfile.ZipFile(checkpoint) as archive:
        if archive.testzip() is not None:
            raise SystemExit('Checkpoint CRC failure')
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise SystemExit(f'Unsafe checkpoint entry: {info.filename}')
        archive.extractall(extraction)
    comparator = BASE / 'COMPARE_INDEPENDENT_RERUN.py'
    report = downloaded / 'LOCAL_VALIDATED_COMPARISON.json'
    command = [sys.executable, str(comparator), '--original-code-archive',
               str(BASE / 'B01_Paper1_Complete_Reviewer_Code_Updated.zip'),
               '--rerun-results', str(extraction/'B01'/'06_RESULTS'),
               '--rerun-work', str(extraction), '--stages', args.stage,
               '--report', str(report)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode not in (0, 2) or not report.is_file():
        raise SystemExit(f'Comparator error {result.returncode}: {result.stdout}\n{result.stderr}')
    provenance = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': args.stage,
                  'backend': args.backend, 'training_rerun': False,
                  'kaggle_checkpoint_sha256': digest(checkpoint),
                  'comparator_sha256': digest(comparator),
                  'comparison_report_sha256': digest(report),
                  'rtol': 1e-3, 'atol': 1e-6,
                  'comparator_exit_code': result.returncode}
    (downloaded/'LOCAL_VALIDATED_PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(result.stdout.strip(), report)


if __name__ == '__main__':
    main()
