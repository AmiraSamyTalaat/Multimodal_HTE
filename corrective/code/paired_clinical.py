from __future__ import annotations
import math
import numpy as np,pandas as pd
from scipy.stats import t as student_t
from corrective_common import COR,now,write

OUT=COR/'results/clinical_adjustment';d=pd.read_csv(OUT/'per_seed_results.csv')
p=d.pivot(index='dgp_seed',columns='method',values='root_pehe')
target='clinical_R_full_fitted_Rloss_selected'
rows=[]
for other in ['archived_clinical_X','archived_clinical_T','archived_full_X','archived_full_T',
              'archived_zero_effect','clinical_R_clinical_fitted_Rloss_selected',
              'clinical_R_oracle_true_simulation_Rloss_selected']:
    x=(p[target]-p[other]).dropna().to_numpy(float);n=len(x);m=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(n))
    q=float(student_t.ppf(.975,n-1))
    rows.append(dict(method_a=target,method_b=other,difference='Root-PEHE(a)-Root-PEHE(b)',
      mean=m,mcse_dgp=se,ci95_lower=m-q*se,ci95_upper=m+q*se,n_independent_dgp=n))
pd.DataFrame(rows).to_csv(OUT/'paired_dgp_comparisons.csv',index=False)
write(OUT/'paired_dgp_comparisons_manifest.json',dict(run_utc=now(),unit='Five DGP seeds',
  caveat='Descriptive Monte Carlo intervals, fixed patient cohort; no family-wise multiplicity correction'))
print(len(rows),'paired clinical comparisons')
