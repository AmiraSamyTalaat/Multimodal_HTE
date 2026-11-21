from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score

REV=Path(__file__).resolve().parents[1];SRC=REV/'source_snapshot'
sys.path.insert(0,str(SRC/'01_CODE'))
from cm_osr.dgp import generate
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=REV/'results/oracle_fairness';out.mkdir(parents=True,exist_ok=True)
    data=SRC/'02_DATA/processed/ispy2_core_multimodal.npz'
    with np.load(data,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    sp=locked_split(len(raw['clinical']));tr=sp['train']
    mods=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}).transform(raw)
    rows=[];bins=[]
    for seed in [101,211,307,401,503]:
        d=generate(mods,seed,'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
        for name in ['train','dev']:
            ix=sp[name];p=np.asarray(d['e_true'][ix],dtype=float);t=np.asarray(d['t'][ix],dtype=int)
            cal=LogisticRegression(C=1e6,max_iter=2000).fit(logit(p).reshape(-1,1),t)
            rows.append({'dgp_seed':seed,'split':name,'n':len(ix),'treatment_prevalence':float(t.mean()),
                         'mean_specified_e':float(p.mean()),'brier':float(brier_score_loss(t,p)),
                         'log_loss':float(log_loss(t,p,labels=[0,1])),
                         'auc':float(roc_auc_score(t,p)),
                         'calibration_intercept':float(cal.intercept_[0]),
                         'calibration_slope':float(cal.coef_[0,0]),
                         'pearson_chi2_per_patient':float(np.mean((t-p)**2/(p*(1-p))))})
            bucket=pd.qcut(p,10,labels=False,duplicates='drop')
            for b in np.unique(bucket):
                m=bucket==b
                bins.append({'dgp_seed':seed,'split':name,'bin':int(b)+1,'n':int(m.sum()),
                             'mean_specified_e':float(p[m].mean()),'observed_treatment_rate':float(t[m].mean())})
    pd.DataFrame(rows).to_csv(out/'specified_propensity_gof.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'specified_propensity_deciles.csv',index=False)
    (out/'SPECIFIED_PROPENSITY_STATUS.json').write_text(json.dumps({'status':'COMPLETE','date_utc':datetime.now(timezone.utc).isoformat(),
        'code_sha256':sha(__file__),'data_sha256':sha(data),'scope':'Prespecified DGP e(X) calibration against simulated treatment assignments on TRAIN and DEV only'},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby('split')[['brier','calibration_intercept','calibration_slope','pearson_chi2_per_patient']].mean().to_string())
if __name__=='__main__':main()
