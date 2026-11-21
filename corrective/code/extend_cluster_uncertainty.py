from __future__ import annotations
import math
import numpy as np,pandas as pd
from scipy.stats import t as student_t
from corrective_common import COR,REV,now,sha,write

OUT=COR/'reporting';OUT.mkdir(exist_ok=True)
rep=pd.read_csv(REV/'results/replication/per_seed_results.csv')
cal=pd.read_csv(REV/'results/oracle_fairness/propensity_gof.csv')
rows=[]
def append(df,group,unit,metric,analysis):
    clusters=df.groupby(group+[unit],as_index=False)[metric].mean()
    for keys,g in clusters.groupby(group):
        if not isinstance(keys,tuple):keys=(keys,)
        n=len(g);a=g[metric].to_numpy(float);mean=float(np.mean(a));sd=float(np.std(a,ddof=1));se=sd/math.sqrt(n)
        q=float(student_t.ppf(.975,n-1))
        row=dict(analysis=analysis,metric=metric,mean=mean,sd_dgp=sd,mcse_dgp=se,
                 ci95_lower=mean-q*se,ci95_upper=mean+q*se,n_independent_dgp=n,
                 unit='DGP seed; inner repeats averaged within seed')
        row.update(dict(zip(group,keys)));rows.append(row)
append(rep,['effect_regime','method'],'dgp_seed','policy_regret','old_replication_policy')
test=cal[cal.split=='DEV'].copy()
for m in ['brier','calibration_intercept','calibration_slope','fraction_e_at_005','fraction_e_at_095']:
    append(test,['split'],'dgp_seed',m,'original_5_dgp_propensity_calibration')
fixed_file=COR/'results/fixed_dgp_splits/per_split_results.csv'
if fixed_file.exists():
    fixed=pd.read_csv(fixed_file)
    for m in ['root_pehe','policy_regret']:
        append(fixed,['effect_regime','method'],'dgp_seed',m,'corrective_fixed_dgp_splits')
fair_file=COR/'results/fair_neural/per_seed_results.csv'
if fair_file.exists():
    fair=pd.read_csv(fair_file)
    for m in ['root_pehe','policy_regret']:
        append(fair,['effect_regime','method','calibration'],'dgp_seed',m,'corrective_fair_neural')
pd.DataFrame(rows).to_csv(OUT/'cluster_uncertainty_extension.csv',index=False)
write(OUT/'cluster_uncertainty_extension_manifest.json',dict(run_utc=now(),
  replication_source_sha256=sha(REV/'results/replication/per_seed_results.csv'),
  calibration_source_sha256=sha(REV/'results/oracle_fairness/propensity_gof.csv'),
  caution='Five-DGP calibration CIs are descriptive and wide; old replication policy CIs inherit split-dependent DGP changes'))
print(len(rows),'cluster-aware uncertainty rows')
