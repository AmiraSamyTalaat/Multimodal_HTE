from __future__ import annotations
import json,math,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t as student_t
from corrective_common import COR,LOCK,now,sha,write

OUT=COR/'results/fair_neural';OUT.mkdir(parents=True,exist_ok=True)
SEEDS=LOCK['fair_neural']['dgp_seeds'];REGIMES=LOCK['fair_neural']['effect_regimes']
def interval(x):
    a=np.asarray(x,dtype=float);n=len(a);mean=float(a.mean());se=float(a.std(ddof=1)/math.sqrt(n)) if n>1 else float('nan')
    crit=float(student_t.ppf(.975,n-1)) if n>1 else float('nan')
    return dict(mean=mean,sd_dgp=float(a.std(ddof=1)) if n>1 else float('nan'),mcse_dgp=se,
                ci95_lower=mean-crit*se,ci95_upper=mean+crit*se,n_independent_dgp=n)
rows=[];fits=[];checks=[];seen=set();archive_files=sorted((COR/'kaggle_neural_downloads').glob('batch_*/B01_P1_CORRECTIVE_NEURAL_BATCH.zip'))
for archive in archive_files:
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        for name in names:
            if not name.endswith('/metrics.csv'):continue
            tag=Path(name).parent.name
            if tag in seen:raise RuntimeError('Duplicate completed neural unit '+tag)
            seen.add(tag)
            metric=pd.read_csv(z.open(name));rows.append(metric)
            manifest=json.loads(z.read(str(Path(name).parent/'COMPLETE.json').replace('\\','/')))
            fits.append(dict(dgp_seed=manifest['dgp_seed'],effect_regime=manifest['effect_regime'],
              cm_seconds=manifest['cm_seconds'],bag_seconds=manifest['bag_seconds'],
              cm_expert_fits=manifest['cm_expert_fit_count'],bag_fits=manifest['bag_fit_count'],
              cm_expert_optimizer_steps=sum(x['optimizer_steps'] for x in manifest['cm_expert_fit_details']),
              bag_optimizer_steps=sum(x['optimizer_steps'] for x in manifest['bag_fit_details']),
              router_optimizer_steps=manifest.get('cm_router_optimizer_steps',400),
              cm_deployed_params=manifest['cm_deployed_params'],bag_deployed_params=manifest['bag_deployed_params'],
              source_archive=archive.name,source_archive_sha256=sha(archive)))
            fixed=COR/'results/fixed_dgp_splits/runs'/f'{tag}_SPLIT{LOCK["fair_neural"]["patient_split_seed"]}'
            if (fixed/'metrics.csv').exists():
                f=pd.read_csv(fixed/'metrics.csv')
                zpehe=float(f.loc[f.method=='zero_effect','root_pehe'].iloc[0])
                zmanifest=json.loads((fixed/'COMPLETE.json').read_text(encoding='utf-8'))
                checks.append(dict(dgp_seed=manifest['dgp_seed'],effect_regime=manifest['effect_regime'],
                  data_sha_match=manifest['data_sha256']==zmanifest['data_sha256'],
                  test_index_sha_match=manifest['test_indices_sha256']==zmanifest['test_indices_sha256'],
                  exact_cate_sha_match=manifest['test_cate_sha256']==zmanifest['cate_test_sha256'],
                  zero_root_pehe_absolute_difference=abs(manifest['zero_effect_root_pehe']-zpehe)))
if not rows:raise RuntimeError('No downloaded neural batches')
neural=pd.concat(rows,ignore_index=True)
fixed_file=COR/'results/fixed_dgp_splits/per_split_results.csv'
fixed=pd.read_csv(fixed_file) if fixed_file.exists() else pd.DataFrame()
if not fixed.empty:
    fixed=fixed[(fixed.split_seed==LOCK['fair_neural']['patient_split_seed']) &
                (fixed.method.isin(['full_X_learner','full_T_learner','clinical_X_learner','clinical_T_learner','zero_effect']))].copy()
    fixed['calibration']='raw';fixed['mode']='classical_or_reference';fixed['network_seed']=np.nan
    fixed=fixed[neural.columns.intersection(fixed.columns).tolist()]
    combined=pd.concat([neural,fixed],ignore_index=True,sort=False)
else:combined=neural.copy()
combined.to_csv(OUT/'per_seed_results.csv',index=False)
summary=[]
for (regime,method,cal),g in combined.groupby(['effect_regime','method','calibration']):
    vals=interval(g.root_pehe);vals.update(effect_regime=regime,method=method,calibration=cal,
      mean_policy_regret=float(g.policy_regret.mean()),n_test_per_dgp=int(g.n_test.iloc[0]),
      inference_ready=len(g)==len(SEEDS))
    summary.append(vals)
pd.DataFrame(summary).to_csv(OUT/'summary_results.csv',index=False)
paired=[]
for regime in REGIMES:
    subset=combined[(combined.effect_regime==regime)&(combined.calibration=='raw')]
    pivot=subset.pivot_table(index='dgp_seed',columns='method',values='root_pehe')
    for a,b in [('CM_OSR','uniform_same_experts'),('CM_OSR','bagged_early_fusion_5fold_capacity_matched'),
                ('CM_OSR','full_X_learner'),('CM_OSR','full_T_learner'),('CM_OSR','zero_effect')]:
        if a in pivot and b in pivot:
            z=(pivot[a]-pivot[b]).dropna();v=interval(z);v.update(effect_regime=regime,method_a=a,method_b=b,
                 difference='Root-PEHE(a)-Root-PEHE(b)',inference_ready=len(z)==len(SEEDS))
            paired.append(v)
pd.DataFrame(paired).to_csv(OUT/'paired_dgp_comparisons.csv',index=False)
pd.DataFrame(fits).to_csv(OUT/'training_budget.csv',index=False)
pd.DataFrame(checks).to_csv(OUT/'cross_platform_validation.csv',index=False)
complete=set((int(r.dgp_seed),r.effect_regime) for r in neural.itertuples())
expected={(s,r) for s in SEEDS for r in REGIMES}
write(OUT/'RUN_STATUS.json',dict(status='COMPLETE' if complete==expected else 'PARTIAL',
    completed_units=len(complete),expected_units=len(expected),missing_units=sorted(expected-complete),
    updated_utc=now(),independence='12 DGP seed clusters per effect regime; one TEST cohort, split and learner seed',
    cross_platform_note='Exact CATE byte hashes can differ between Windows and Kaggle PCA numerical stacks; compare fixed TEST index hash and zero Root-PEHE within tolerance, never treat this as independent patient replication'))
print('Neural units',len(complete),'/',len(expected),'archive count',len(archive_files))
