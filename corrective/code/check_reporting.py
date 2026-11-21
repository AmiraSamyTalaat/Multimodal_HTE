from __future__ import annotations
import json, math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from corrective_common import COR, REV, SRC, now, sha, write

OUT = COR / 'reporting'; OUT.mkdir(parents=True, exist_ok=True)
checks = []
def check(name, ok, evidence, implication):
    checks.append(dict(check=name, pass_check=bool(ok), evidence=evidence, implication=implication))

rep = pd.read_csv(REV/'results/replication/per_seed_results.csv')
cluster = pd.read_csv(REV/'results/replication/dgp_cluster_means.csv')
old_ci = pd.read_csv(REV/'results/replication/uncertainty_intervals.csv')
new_cluster = rep.groupby(['effect_regime','method','dgp_seed'], as_index=False).agg(
    root_pehe=('root_pehe','mean'), policy_regret=('policy_regret','mean'),
    n_splits=('patient_split_seed','nunique'))
key=['effect_regime','method','dgp_seed']
v=new_cluster.merge(cluster,on=key,suffixes=('_new','_old'),validate='one_to_one')
cluster_error=float(np.max(np.abs(v.root_pehe_new-v.root_pehe_old)))
check('replication_cluster_means',cluster_error<1e-12,f'max absolute error {cluster_error:.3g}',
      'Three patient splits were averaged within each DGP seed.')
rows=[]
for (regime, method), g in new_cluster.groupby(['effect_regime','method']):
    n=g.dgp_seed.nunique(); mean=float(g.root_pehe.mean()); sd=float(g.root_pehe.std(ddof=1))
    se=sd/math.sqrt(n); critical=float(student_t.ppf(.975,n-1))
    rows.append(dict(effect_regime=regime,method=method,mean_root_pehe=mean,n_independent_dgp=n,
                     n_splits_per_dgp=int(g.n_splits.min()),mcse_dgp_cluster=se,
                     ci95_lower=mean-critical*se,ci95_upper=mean+critical*se))
new_ci=pd.DataFrame(rows); new_ci.to_csv(OUT/'verified_dgp_cluster_intervals.csv',index=False)
v=new_ci.merge(old_ci,on=['effect_regime','method'],suffixes=('_new','_old'),validate='one_to_one')
ci_error=max(float(np.max(np.abs(v[f'{c}_new']-v[f'{c}_old']))) for c in ['ci95_lower','ci95_upper'])
check('replication_ci_unit',ci_error<1e-12 and (v.n_independent_dgp_new==30).all(),
      f'max CI endpoint error {ci_error:.3g}; n independent DGP=30 per mechanism, 3 splits each',
      'The 270 DGP×regime×split records are not 270 independent units. CATE changed across splits in this original replication.')

base=pd.read_csv(REV/'results/baselines/per_seed_results.csv')
oracle=base[base.method=='oracle_constant_test_cate'].copy()
oracle_error=float(np.max(np.abs(oracle.root_pehe-oracle.true_cate_sd)))
check('oracle_constant',oracle_error<1e-12,f'max |Root-PEHE minus SD(true TEST CATE)|={oracle_error:.3g}',
      'TEST-CATE mean is the optimal constant under squared error only. It is a nondeployable diagnostic, not a lower bound on all estimators.')
ate=base[base.method=='train_estimated_ATE_constant']
manifest=[json.loads((REV/f'results/baselines/DGP{int(s)}_manifest.json').read_text(encoding='utf-8')) for s in sorted(ate.dgp_seed.unique())]
check('train_ate_definition',all('TRAIN' in x['train_estimated_ate_method'] for x in manifest),
      'Five per-DGP manifests specify mean 5-fold cross-fitted DR pseudo-outcome on TRAIN and propensity clip 0.05–0.95.',
      'This is an estimated constant ATE, distinct from crude TRAIN arm difference and the oracle TEST constant.')

predfiles=sorted((SRC/'04_RUNS').glob('CORE_DGP*_NET*/predictions/test_predictions.npz'))
policy_err=0.0; n_pred=0
from cm_osr.metrics import policy_regret
for f in predfiles:
    with np.load(f,allow_pickle=False) as z:
        tau=np.asarray(z['tau_true'],dtype=float)
        for method in [k for k in z.files if k not in ('tau_true','support')]:
            pred=np.asarray(z[method],dtype=float)
            manual=float(np.maximum(tau,0).mean()-np.mean(tau*(pred>0)))
            policy_err=max(policy_err,abs(manual-policy_regret(pred,tau))); n_pred+=1
check('policy_regret_definition',policy_err<1e-12 and len(predfiles)==15,
      f'{len(predfiles)} archived prediction files; {n_pred} method predictions; max formula discrepancy {policy_err:.3g}',
      'Simulation-only policy regret is oracle positive-CATE value minus value from treating when predicted CATE>0. It is not observed clinical utility.')

shift=pd.read_csv(REV/'results/additional/stage5a_effect_structure_shift.csv')
test_shift=shift[shift.split=='test']
check('stage5a_severe_overlap_cate_shift',len(test_shift)==5 and (test_shift.true_cate_fraction_changed_gt_1e4>0).all(),
      f'mean TEST CATE RMSE primary versus severe {test_shift.true_cate_rmse_primary_vs_severe.mean():.6f}; '
      f'mean TEST fraction changed {test_shift.true_cate_fraction_changed_gt_1e4.mean():.3f}',
      'Original Stage5A severe-overlap comparison changes effect structure as well as propensity; it cannot isolate overlap. Use clean-overlap perturbation for that claim.')

drift_file=COR/'reporting/old_replication_split_truth_drift.csv'
if drift_file.exists():
    drift=pd.read_csv(drift_file)
    drift_ok=(len(drift)==270 and (drift.full_cohort_cate_rmse>0).all() and
              drift.dgp_seed.nunique()==30 and drift.effect_regime.nunique()==3)
    check('old_split_dependent_cate',drift_ok,
          f'270 same-patient split-pair comparisons; mean CATE RMSE by regime '+
          str(drift.groupby('effect_regime').full_cohort_cate_rmse.mean().round(6).to_dict()),
          'The archived split comparison changes true CATE and must not be interpreted as split-only model sensitivity.')

pd.DataFrame(checks).to_csv(OUT/'reporting_checks.csv',index=False)
write(OUT/'reporting_checks.json',{'run_utc':now(),'status':'PASS' if all(x['pass_check'] for x in checks) else 'FAIL',
    'checks':checks,'source_sha256':{str(p.relative_to(REV)):sha(p) for p in [
        REV/'results/replication/per_seed_results.csv',REV/'results/replication/uncertainty_intervals.csv',
        REV/'results/baselines/per_seed_results.csv',REV/'results/additional/stage5a_effect_structure_shift.csv']}})
print(pd.DataFrame(checks)[['check','pass_check','evidence']].to_string(index=False))
