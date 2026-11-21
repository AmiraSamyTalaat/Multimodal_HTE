from __future__ import annotations
import itertools
import numpy as np,pandas as pd
from corrective_common import COR,DATA,SRC,load_raw,now,sha,write
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
from cm_osr.stage7_dgp import generate_stage7

OUT=COR/'reporting';OUT.mkdir(exist_ok=True)
SPLITS=[20260918,20260929,20261013]
REGIMES=['NO_INCREMENTAL_MOLECULAR_SIGNAL','TRANSCRIPTOME_INCREMENTAL','JOINT_MOLECULAR_SYNERGY']
SEEDS=[3001+67*i for i in range(30)]
raw=load_raw();mods={}
for split_seed in SPLITS:
    tr=locked_split(len(raw['clinical']),seed=split_seed)['train']
    prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()})
    mods[split_seed]=(prep.transform(raw),tr)
rows=[]
for seed in SEEDS:
    for regime in REGIMES:
        cate={s:generate_stage7(mods[s][0],seed,regime,reference_idx=mods[s][1])['tau_true'] for s in SPLITS}
        for a,b in itertools.combinations(SPLITS,2):
            delta=np.asarray(cate[a],float)-np.asarray(cate[b],float)
            rows.append(dict(dgp_seed=seed,effect_regime=regime,split_seed_a=a,split_seed_b=b,n_patients=len(delta),
              full_cohort_cate_rmse=float(np.sqrt(np.mean(delta**2))),
              full_cohort_fraction_changed_gt_1e4=float(np.mean(np.abs(delta)>1e-4)),
              full_cohort_cate_max_absolute_change=float(np.max(np.abs(delta)))))
pd.DataFrame(rows).to_csv(OUT/'old_replication_split_truth_drift.csv',index=False)
write(OUT/'old_replication_split_truth_drift_manifest.json',dict(run_utc=now(),n_dgp_seeds=len(SEEDS),
    n_regimes=len(REGIMES),n_split_pairs=3,source_data_sha256=sha(DATA),
    source_dgp_code_sha256=sha(SRC/'01_CODE/cm_osr/stage7_dgp.py'),
    method='Regenerate simulator truth only under archived per-split preprocessing/reference; compare same 736 patients; no learner rerun or TEST selection'))
print('truth drift rows',len(rows),'mean RMSE',pd.DataFrame(rows).groupby('effect_regime').full_cohort_cate_rmse.mean().to_dict())
