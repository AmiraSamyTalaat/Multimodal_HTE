from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
OUT = REV / 'results' / 'fairness'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [101, 211, 307, 401, 503]
REFERENCE = ['CM_OSR', 'uniform_all_experts_ablation',
             'early_fusion_DR_capacity_matched', 'X_learner']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    rows = []
    manifests = []
    for seed in SEEDS:
        folder = REV / 'runs' / f'kaggle_fairness_dgp{seed}' / 'postreview_results'
        status = folder / 'RUN_STATUS.json'
        metrics = folder / 'per_run_results.csv'
        if not status.exists() or not metrics.exists():
            continue
        state = json.loads(status.read_text(encoding='utf-8'))
        if state.get('status') != 'COMPLETE' or state.get('completed_pairs') != 3:
            continue
        batch = pd.read_csv(metrics)
        batch['source'] = 'private_kaggle_kernel_output'
        batch['source_kernel'] = f'prinprin/b01-p1-postreview-fairness-dgp{seed}'
        batch['output_sha256'] = sha(metrics)
        rows.append(batch)
        manifests.append({'dgp_seed': seed, 'kernel': batch['source_kernel'].iloc[0],
                          'started_utc': state.get('started_utc'),
                          'finished_utc': state.get('finished_utc'),
                          'data_sha256': state.get('data_sha256'),
                          'metrics_sha256': sha(metrics),
                          'mode': state['config']['mode'],
                          'max_epochs': state['config']['max_epochs'],
                          'patience': state['config']['patience'],
                          'completed_pairs': state['completed_pairs']})
    if not rows:
        raise RuntimeError('No completed full Kaggle fairness batches found')
    new = pd.concat(rows, ignore_index=True)
    original = pd.read_csv(REV / 'results' / 'baselines' / 'per_run_results.csv')
    old = original[original.method.isin(REFERENCE)].copy()
    old = old.rename(columns={'run': 'original_run'})
    old['source'] = 'archived_test_predictions'
    old['source_kernel'] = ''
    old['output_sha256'] = ''
    cols = ['dgp_seed','net_seed','method','root_pehe','policy_regret',
            'clipped_root_pehe','fraction_outside_minus1_plus1','source',
            'source_kernel','output_sha256']
    both = pd.concat([new[cols],old[cols]],ignore_index=True)
    both.to_csv(OUT / 'per_run_results.csv',index=False)
    per_seed = both.groupby(['dgp_seed','method','source'],as_index=False).agg(
        root_pehe=('root_pehe','mean'),policy_regret=('policy_regret','mean'),
        clipped_root_pehe=('clipped_root_pehe','mean'),
        fraction_outside_minus1_plus1=('fraction_outside_minus1_plus1','mean'),
        n_network_seeds=('net_seed','nunique'))
    per_seed.to_csv(OUT / 'per_seed_results.csv',index=False)
    summary = per_seed.groupby('method',as_index=False).agg(
        mean_root_pehe=('root_pehe','mean'),sd_across_dgp=('root_pehe','std'),
        mean_policy_regret=('policy_regret','mean'),
        mean_clipped_root_pehe=('clipped_root_pehe','mean'),
        mean_outside_fraction=('fraction_outside_minus1_plus1','mean'),
        n_independent_dgp=('dgp_seed','nunique')).sort_values('mean_root_pehe')
    summary['mcse_root_pehe'] = summary.sd_across_dgp / np.sqrt(summary.n_independent_dgp)
    summary.to_csv(OUT / 'summary_results.csv',index=False)
    wide = per_seed.pivot(index='dgp_seed',columns='method',values='root_pehe')
    comparisons = []
    for candidate in ['bagged_early_fusion','single_early_stopped_early_fusion']:
        for anchor in REFERENCE:
            if candidate not in wide or anchor not in wide:
                continue
            diff=(wide[candidate]-wide[anchor]).dropna()
            comparisons.append({'method':candidate,'comparator':anchor,
                                'mean_paired_root_pehe_difference':diff.mean(),
                                'sd_across_dgp':diff.std(),
                                'mcse':diff.std()/np.sqrt(len(diff)) if len(diff)>1 else np.nan,
                                'n_independent_dgp':len(diff)})
    pd.DataFrame(comparisons).to_csv(OUT/'paired_dgp_comparisons.csv',index=False)
    pd.DataFrame(manifests).to_csv(OUT/'kaggle_batch_manifest.csv',index=False)
    status={'stage':'FAIR_BAGGED_EARLY_FUSION_POSTREVIEW',
            'status':'COMPLETE' if len(manifests)==len(SEEDS) else 'PARTIAL',
            'completed_batches':len(manifests),'expected_batches':len(SEEDS),
            'completed_run_pairs':len(new)//2,'expected_run_pairs':15,
            'independence':'DGP seeds are clusters; 3 network seeds per DGP are not independent',
            'last_updated_utc':datetime.now(timezone.utc).isoformat()}
    (OUT/'RUN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status))

if __name__=='__main__':main()
