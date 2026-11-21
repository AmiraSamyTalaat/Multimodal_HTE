"""Dispatch the resumable benchmark Kaggle runner to a frozen stage grid."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('stage4', 'stage6a', 'stage6b', 'stage6c', 'stage7b'), default='stage4')
    args, forward = parser.parse_known_args()
    here = Path(__file__).resolve().parent
    if args.stage == 'stage4':
        command = [sys.executable, str(here / 'orchestrate_stage4.py'), *forward]
    else:
        command = [sys.executable, str(here / 'orchestrate_selected_stages.py'), '--stage', args.stage, *forward]
    raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
