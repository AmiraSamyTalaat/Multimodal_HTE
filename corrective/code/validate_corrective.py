from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np,pandas as pd
from corrective_common import COR,REV,DATA,LOCK,now,sha,write

checks=[]
def add(name,pass_check,detail):checks.append(dict(check=name,pass_check=bool(pass_check),detail=str(detail)))
source=pd.read_csv(REV/'source_provenance.csv')
snapshot_mismatch=[];original_mismatch=[];original_missing=[]
for r in source.itertuples():
    snap=Path(r.snapshot_full_path);orig=Path(r.original_full_path)
    if not snap.exists() or sha(snap)!=r.copy_sha256:snapshot_mismatch.append(r.relative_path)
    if not orig.exists():original_missing.append(r.relative_path)
    elif sha(orig)!=r.original_sha256:original_mismatch.append(r.relative_path)
add('original_and_snapshot_immutable',not(snapshot_mismatch or original_mismatch or original_missing),
    f'{len(source)} files; snapshot mismatches {len(snapshot_mismatch)}, original mismatches {len(original_mismatch)}, missing {len(original_missing)}')
kcode=pd.read_csv(COR/'reporting/kaggle_dataset_code_verification.csv')
add('private_kaggle_core_code_matches_snapshot',len(kcode)>=9 and kcode['match'].all(),
    f'{int(kcode["match"].sum())}/{len(kcode)} downloaded core modules match SHA-256')
for analysis,expected in [('clinical_adjustment',5),('nonoracle_tuning',5),('fixed_dgp_splits',72),('fair_neural',24)]:
    p=COR/'results'/analysis/'RUN_STATUS.json'
    info=json.loads(p.read_text()) if p.exists() else {}
    got=info.get('completed_units',info.get('completed_dgp',0))
    add(analysis+'_complete',info.get('status')=='COMPLETE' and got==expected,f'{got}/{expected}')
fixed=pd.read_csv(COR/'results/fixed_dgp_splits/fixed_truth_validation.csv')
add('fixed_dgp_truth_unchanged',len(fixed)==24 and (fixed.completed_splits==3).all() and fixed.fixed_truth_and_test_hashes.all(),
    f'{len(fixed)} seed-regime pairs; three resplits each')
cross=pd.read_csv(COR/'results/fair_neural/cross_platform_validation.csv')
add('kaggle_windows_numerical_match',len(cross)==24 and cross.data_sha_match.all() and cross.test_index_sha_match.all() and
    cross.zero_root_pehe_absolute_difference.max()<1e-6,
    f'{len(cross)} units; max zero-effect Root-PEHE difference {cross.zero_root_pehe_absolute_difference.max():.3g}; exact byte hash match {cross.exact_cate_sha_match.sum()}/{len(cross)}')
budget=pd.read_csv(COR/'results/fair_neural/training_budget.csv')
ratio=budget.bag_deployed_params/budget.cm_deployed_params
add('deployed_parameter_match',len(budget)==24 and ratio.between(.9,1.1).all(),
    f'bag/CM deployed parameter ratio range {ratio.min():.4f}–{ratio.max():.4f}; optimizer steps remain unequal')
for analysis,file in [('clinical','clinical_adjustment/per_seed_results.csv'),
                      ('nonoracle','nonoracle_tuning/per_seed_results.csv'),
                      ('fixed','fixed_dgp_splits/per_split_results.csv'),
                      ('fair','fair_neural/per_seed_results.csv')]:
    d=pd.read_csv(COR/'results'/file)
    vals=d[['root_pehe','policy_regret']].to_numpy(float)
    add(analysis+'_metrics_finite',np.isfinite(vals).all() and (d.root_pehe>=0).all() and (d.policy_regret>=-1e-10).all(),
        f'{len(d)} rows; Root-PEHE range {d.root_pehe.min():.4g}–{d.root_pehe.max():.4g}')
fair=pd.read_csv(COR/'results/fair_neural/summary_results.csv')
add('fair_dgp_cluster_ci',len(fair)==22 and (fair.n_independent_dgp==12).all() and fair.inference_ready.all(),
    f'{len(fair)} summary rows; DGP counts {sorted(fair.n_independent_dgp.unique())}')
old=json.loads((COR/'reporting/reporting_checks.json').read_text())
add('reporting_definitions',old.get('status')=='PASS',old.get('status'))
status='PASS' if all(x['pass_check'] for x in checks) else 'FAIL'
write(COR/'scientific_validation.json',dict(status=status,run_utc=now(),checks=checks,
   source_data_sha256=sha(DATA),interpretation='Validation verifies provenance, arithmetic and protocol invariants; it does not prove causal identification or generalization'))
pd.DataFrame(checks).to_csv(COR/'scientific_validation.csv',index=False)
print(status,pd.DataFrame(checks)[['check','pass_check','detail']].to_string(index=False))
if status!='PASS':raise SystemExit(1)
