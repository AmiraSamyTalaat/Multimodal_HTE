from __future__ import annotations
import json
from collections import defaultdict
import numpy as np,pandas as pd
from corrective_common import COR,LOCK,load_raw,fixed_split,now,sha,write

OUT=COR/'results/fixed_dgp_splits';RUNS=OUT/'runs'
groups=defaultdict(list);raw=load_raw();issues=[];checks=[]
for p in RUNS.glob('*/COMPLETE.json'):
    x=json.loads(p.read_text(encoding='utf-8'))
    key=(x['dgp_seed'],x['effect_regime']);groups[key].append(x)
    split=fixed_split(len(raw['clinical']),x['split_seed'])
    if set(split['train'])&set(split['dev']) or set(split['train'])&set(split['test']) or set(split['dev'])&set(split['test']):
        issues.append(f'overlap {key} split {x["split_seed"]}')
    if (len(split['train']),len(split['dev']),len(split['test']))!=(441,147,148):
        issues.append(f'sizes {key} split {x["split_seed"]}')
for key,items in groups.items():
    attrs=['dgp_npz_sha256','cate_full_sha256','cate_test_sha256','test_indices_sha256']
    consistent=all(len({x[a] for x in items})==1 for a in attrs)
    if not consistent:issues.append('truth or TEST changed across splits '+str(key))
    checks.append(dict(dgp_seed=key[0],effect_regime=key[1],completed_splits=len(items),
      fixed_truth_and_test_hashes=consistent,split_seeds=';'.join(str(x['split_seed']) for x in sorted(items,key=lambda z:z['split_seed']))))
pd.DataFrame(checks).to_csv(OUT/'fixed_truth_validation.csv',index=False)
write(OUT/'fixed_truth_validation.json',dict(status='PASS' if not issues else 'FAIL',checked_utc=now(),
   completed_dgp_regime_pairs=len(groups),expected_pairs=len(LOCK['fixed_dgp']['dgp_seeds'])*len(LOCK['fixed_dgp']['effect_regimes']),
   issues=issues,interpretation='A PASS on partial data validates completed pairs only; inference waits for all 72 units'))
print('PASS' if not issues else 'FAIL',len(groups),'pairs')
