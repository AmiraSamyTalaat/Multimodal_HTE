"""Download and validate actual 12-seed post-review neural runs from Kaggle."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import t as student_t

BASE = Path(__file__).resolve().parent
SEEDS = [9001, 9102, 9203, 9304, 9405, 9506, 9607, 9708, 9809, 9910, 10011, 10112]
REGIMES = ['NO_INCREMENTAL_MOLECULAR_SIGNAL', 'JOINT_MOLECULAR_SYNERGY']
NEW_METHODS = {'CM_OSR', 'uniform_same_experts', 'bagged_early_fusion_5fold_capacity_matched', 'bagged_early_fusion_20fit_budget_targeted'}
OLD_METHODS = {'full_X_learner', 'full_T_learner', 'clinical_X_learner', 'clinical_T_learner', 'zero_effect'}
SLUGS = ['b01-p1-fair-neural-20fit-smoke-cpu'] + [f'b01-p1-fair-neural-20fit-batch{i:02d}' for i in range(1, 7)]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def kaggle(*args):
    env = os.environ.copy()
    env.update(TEMP=str(BASE), TMP=str(BASE), TMPDIR=str(BASE))
    p = subprocess.run(['kaggle', *args], capture_output=True, text=True, env=env, timeout=300)
    return p.returncode, p.stdout + p.stderr


def read_csv(text):
    return list(csv.DictReader(io.StringIO(text)))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci(values):
    n = len(values)
    mean = sum(values) / n
    se = (sum((x-mean)**2 for x in values)/(n-1)/n)**0.5
    q = student_t.ppf(.975, n-1)
    return mean, se, mean-q*se, mean+q*se


def collect(watch):
    archive_files = []
    for slug in SLUGS:
        output = BASE/'downloads'/slug
        output.mkdir(parents=True, exist_ok=True)
        target = output/'B01_P1_FAIR_NEURAL_BATCH.zip'
        while not target.exists():
            code, result = kaggle('kernels', 'status', f'prinprin/{slug}')
            if code:
                raise RuntimeError(result[-700:])
            if 'KernelWorkerStatus.ERROR' in result:
                _, logs = kaggle('kernels', 'logs', f'prinprin/{slug}')
                (output/'ERROR_LOG.json').write_text(logs, encoding='utf-8')
                raise RuntimeError(f'Kaggle kernel failed: {slug}')
            if 'KernelWorkerStatus.COMPLETE' in result:
                code, details = kaggle('kernels', 'output', f'prinprin/{slug}', '-p', str(output), '-o',
                                       '--file-pattern', 'B01_P1_FAIR_NEURAL_BATCH.zip')
                if code:
                    raise RuntimeError(details[-700:])
            elif watch:
                print(datetime.now(timezone.utc).isoformat(), slug, 'waiting', flush=True)
                time.sleep(30)
            else:
                raise RuntimeError(f'Incomplete kernel: {slug}: {result[-200:]}')
        archive_files.append((slug, target))
    return archive_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    archives = collect(args.watch)
    old_manifest = {(int(x['dgp_seed']), x['effect_regime']): x for x in json.loads((BASE/'archived_fair_neural_manifests.json').read_text(encoding='utf-8-sig'))}
    old_csv = read_csv((BASE/'archived_fair_neural_per_seed_results.csv').read_text(encoding='utf-8-sig'))
    old_rows = [dict(x, provenance='archived_classical_baseline') for x in old_csv if x['method'] in OLD_METHODS and x['calibration'] == 'raw']
    new_rows, manifests, source = [], {}, []
    for slug, path in archives:
        source.append({'kernel': f'prinprin/{slug}', 'sha256': sha(path), 'file': path.name})
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None or any(n.endswith('.npz') or 'prediction' in n.lower() for n in z.namelist()):
                raise RuntimeError(f'Individual predictions or corrupt ZIP: {slug}')
            for name in z.namelist():
                if name.endswith('/COMPLETE.json'):
                    m = json.loads(z.read(name))
                    key = (int(m['dgp_seed']), m['effect_regime'])
                    if key in manifests:
                        raise RuntimeError(f'Duplicate DGP/regime: {key}')
                    manifests[key] = m
                if name.endswith('/metrics.csv'):
                    for row in read_csv(z.read(name).decode('utf-8-sig')):
                        if row['method'] not in NEW_METHODS:
                            raise RuntimeError(f'Unexpected method: {row["method"]}')
                        new_rows.append(dict(row, provenance=f'new_kaggle:{slug}'))
    expected = {(seed, regime) for seed in SEEDS for regime in REGIMES}
    if set(manifests) != expected:
        raise RuntimeError(f'Missing/extra DGP regimes: {expected ^ set(manifests)}')
    checks = []
    for key in sorted(expected):
        old, new = old_manifest[key], manifests[key]
        details = {'dgp_seed': key[0], 'effect_regime': key[1], 'status': new['status'],
                   'data_sha_match': old['data_sha256'] == new['data_sha256'],
                   'test_indices_sha_match': old['test_indices_sha256'] == new['test_indices_sha256'],
                   'test_cate_mean_difference': float(new['test_cate_mean'])-float(old['test_cate_mean']),
                   'test_cate_sd_difference': float(new['test_cate_sd'])-float(old['test_cate_sd']),
                   'zero_root_pehe_difference': float(new['zero_effect_root_pehe'])-float(old['zero_effect_root_pehe']),
                   'cm_expert_fits': new['cm_expert_fit_count'], 'bag20_fits': new['bag20_fit_count'],
                   'cm_optimizer_steps': new['cm_expert_optimizer_steps'], 'bag20_optimizer_steps': new['bag20_optimizer_steps'],
                   'cm_params': new['cm_deployed_params'], 'bag20_params': new['bag20_deployed_params'],
                   'cm_seconds': new['cm_seconds'], 'bag20_seconds': new['bag20_seconds']}
        details['pass'] = (details['status'] == 'COMPLETE' and details['data_sha_match'] and details['test_indices_sha_match']
                           and max(abs(details[k]) for k in ('test_cate_mean_difference','test_cate_sd_difference','zero_root_pehe_difference')) < 1e-6)
        checks.append(details)
    if not all(x['pass'] for x in checks):
        raise RuntimeError('DGP/test validation failed; inspect checks before pooling')
    rows = new_rows + old_rows
    if len(new_rows) != 24*8 or len(old_rows) != 24*5:
        raise RuntimeError(f'Metric row count mismatch: new={len(new_rows)}, old={len(old_rows)}')
    # Normalize column set across archived and new sources.
    cols = list(dict.fromkeys(k for row in rows for k in row))
    rows = [{k: row.get(k, '') for k in cols} for row in rows]
    write_csv(BASE/'FAIR_NEURAL_COMPARISON.csv', rows)
    write_csv(BASE/'FAIR_NEURAL_TRAINING_BUDGET.csv', checks)
    raw = [r for r in rows if r['calibration'] == 'raw']
    summary, paired = [], []
    for regime in REGIMES:
        methods = sorted({r['method'] for r in raw if r['effect_regime'] == regime})
        lookup = {(int(r['dgp_seed']), r['method']): float(r['root_pehe']) for r in raw if r['effect_regime'] == regime}
        for method in methods:
            vals = [lookup[(seed,method)] for seed in SEEDS]
            mean, se, low, high = ci(vals)
            item = {'effect_regime': regime, 'method': method, 'calibration': 'raw', 'n_independent_dgp_seeds': 12,
                    'mean_root_pehe': mean, 'mc_se': se, 't_ci95_low': low, 't_ci95_high': high}
            for metric in ('policy_regret','clipped_root_pehe','fraction_outside_minus1_plus1'):
                observations = [float(next(r[metric] for r in raw if r['effect_regime']==regime and r['method']==method and int(r['dgp_seed'])==seed)) for seed in SEEDS]
                m, s, lo, hi = ci(observations)
                item.update({f'mean_{metric}': m, f'mc_se_{metric}': s, f't_ci95_low_{metric}': lo, f't_ci95_high_{metric}': hi})
            summary.append(item)
            if method != 'CM_OSR':
                diffs = [lookup[(seed,'CM_OSR')]-lookup[(seed,method)] for seed in SEEDS]
                mean, se, low, high = ci(diffs)
                paired.append({'effect_regime': regime, 'comparison': 'CM_OSR minus '+method,
                               'n_independent_dgp_seeds': 12, 'mean_difference': mean, 'mc_se': se,
                               't_ci95_low': low, 't_ci95_high': high, 'negative_favors_CM': high < 0})
    write_csv(BASE/'FAIR_NEURAL_SUMMARY.csv', summary)
    write_csv(BASE/'FAIR_NEURAL_PAIRED_DGP.csv', paired)
    validation = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(), 'analysis_type': 'post_review',
                  'independence_unit': 'DGP seed; 12 seeds per mechanism, one fixed TEST split and one network seed',
                  'new_metric_rows': len(new_rows), 'archived_classical_rows': len(old_rows), 'sources': source,
                  'checks': checks, 'fairness_scope': 'Training opportunity and parameter budgets approximate, not identical; CM vs uniform shares experts; X/T remain classical archived comparators.'}
    (BASE/'FAIR_NEURAL_VALIDATION.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print('FAIR_NEURAL_COMPLETE', len(expected), len(rows), flush=True)


if __name__ == '__main__':
    main()
