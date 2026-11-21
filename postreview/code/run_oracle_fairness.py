from __future__ import annotations
import argparse, hashlib, json, platform, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REV=Path(__file__).resolve().parents[1]
SRC=REV/'source_snapshot'
sys.path.insert(0,str(SRC/'01_CODE'))
from cm_osr.dgp import generate, _proj
from cm_osr.metrics import root_pehe, policy_regret
from cm_osr.nuisance import crossfit_dr, FullNuisance
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
from cm_osr.utils import concat_modalities

DGP_SEEDS=[101,211,307,401,503]
NET_SEEDS=[17,29,43]
ALPHAS=[1.0,10.0,100.0,1000.0,10000.0,100000.0]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def quantiles(x):
    x=np.asarray(x,dtype=float)
    return {'mean':float(x.mean()),'sd':float(x.std(ddof=0)),'rms':float(np.sqrt(np.mean(x*x))),
            'min':float(x.min()),'q01':float(np.quantile(x,.01)),'q05':float(np.quantile(x,.05)),
            'median':float(np.median(x)),'q95':float(np.quantile(x,.95)),
            'q99':float(np.quantile(x,.99)),'max':float(x.max()),
            'fraction_abs_gt_1':float(np.mean(np.abs(x)>1)),
            'fraction_abs_gt_5':float(np.mean(np.abs(x)>5))}
def fit_ridge(X,phi,alpha):return make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(X,phi)
def pseudo(t,y,e,m0,m1):
    return (m1-m0)+t*(y-m1)/e-(1-t)*(y-m0)/(1-e)
def propensity_diagnostic(t,prob,true_prob,split_name,dgp,net):
    prob=np.clip(np.asarray(prob,dtype=float),1e-6,1-1e-6)
    z=logit(prob).reshape(-1,1)
    cal=LogisticRegression(C=1e6,max_iter=2000).fit(z,t)
    return {'dgp_seed':dgp,'net_seed':net,'split':split_name,'n':len(t),
            'prevalence':float(np.mean(t)),'mean_pred_e':float(np.mean(prob)),
            'mean_true_e':float(np.mean(true_prob)),'brier':float(brier_score_loss(t,prob)),
            'oracle_brier':float(brier_score_loss(t,true_prob)),
            'log_loss':float(log_loss(t,prob,labels=[0,1])),
            'oracle_log_loss':float(log_loss(t,true_prob,labels=[0,1])),
            'auc':float(roc_auc_score(t,prob)),'e_vs_true_rmse':float(np.sqrt(np.mean((prob-true_prob)**2))),
            'calibration_intercept':float(cal.intercept_[0]),'calibration_slope':float(cal.coef_[0,0]),
            'fraction_e_at_005':float(np.mean(prob<=0.0500001)),
            'fraction_e_at_095':float(np.mean(prob>=0.9499999))}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dgp-seeds',nargs='*',type=int,default=DGP_SEEDS)
    ap.add_argument('--net-seeds',nargs='*',type=int,default=NET_SEEDS)
    a=ap.parse_args()
    out=REV/'results'/'oracle_fairness';out.mkdir(parents=True,exist_ok=True)
    data_file=SRC/'02_DATA/processed/ispy2_core_multimodal.npz'
    with np.load(data_file,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    split=locked_split(len(raw['clinical']))
    tr,dv,te=split['train'],split['dev'],split['test']
    prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()})
    mods=prep.transform(raw);X=concat_modalities(mods)
    code_hash=sha(__file__);data_hash=sha(data_file)
    status={'stage':'ORACLE_FAIRNESS_POSTREVIEW','started_utc':datetime.now(timezone.utc).isoformat(),
            'code_sha256':code_hash,'data_sha256':data_hash,'requested_pairs':len(a.dgp_seeds)*len(a.net_seeds),
            'completed_pairs':[],'failed_pairs':[],'status':'RUNNING'}
    write(out/'RUN_STATUS.json',status)
    for seed in a.dgp_seeds:
        d=generate(mods,seed,'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
        # Deterministic DGP projections permit direct clipping diagnostics.
        c=_proj(mods['clinical'],1101,tr);g=_proj(mods['transcriptomic'],2203,tr);p=_proj(mods['rppa'],3307,tr)
        tau_unclipped=.10+.16*g*p+.08*c*g-.05*p
        tau_clipped=np.clip(tau_unclipped,-.35,.45)
        before_m1=d['mu0_true']+tau_clipped
        dgp_diag={'dgp_seed':seed,'true_cate_test_sd':float(np.std(d['tau_true'][te])),
                  'true_cate_test_mean':float(np.mean(d['tau_true'][te])),
                  'propensity_true_clip_fraction':float(np.mean((d['e_true']<=.0200001)|(d['e_true']>=.9799999))),
                  'outcome_mu1_clip_fraction':float(np.mean((before_m1<.02)|(before_m1>.98))),
                  'outcome_clipping_sign_reversal_fraction':float(np.mean((np.sign(tau_clipped)!=np.sign(d['tau_true']))&(np.abs(tau_clipped)>1e-12))),
                  'train_treatment_prevalence':float(np.mean(d['t'][tr])),
                  'dev_treatment_prevalence':float(np.mean(d['t'][dv])),
                  'test_treatment_prevalence':float(np.mean(d['t'][te]))}
        write(out/f'DGP{seed}_dgp_diagnostics.json',dgp_diag)
        for net in a.net_seeds:
            name=f'DGP{seed}_NET{net}';directory=out/name;directory.mkdir(exist_ok=True)
            complete=directory/'COMPLETE.json'
            if complete.exists():
                old=json.loads(complete.read_text(encoding='utf-8'))
                if old.get('code_sha256')==code_hash and old.get('data_sha256')==data_hash:
                    status['completed_pairs'].append(name);write(out/'RUN_STATUS.json',status);continue
            try:
                started=datetime.now(timezone.utc).isoformat()
                trainmods={k:v[tr] for k,v in mods.items()}
                devmods={k:v[dv] for k,v in mods.items()}
                fitted=crossfit_dr(trainmods,d['t'][tr],d['y'][tr],n_splits=5,seed=net,clip=(.05,.95))
                full=FullNuisance(seed=net,clip=(.05,.95)).fit(trainmods,d['t'][tr],d['y'][tr])
                devfit=full.predict(devmods)
                phi_oracle=pseudo(d['t'][tr],d['y'][tr],d['e_true'][tr],d['mu0_true'][tr],d['mu1_true'][tr])
                phi_fitted=fitted['phi']
                dist=[]
                for label,values in [('phi_oracle',phi_oracle),('phi_fitted',phi_fitted),('e_true_train',d['e_true'][tr]),('e_fitted_oof_train',fitted['e_hat']),('mu0_true_train',d['mu0_true'][tr]),('mu1_true_train',d['mu1_true'][tr])]:
                    dist.append({'dgp_seed':seed,'net_seed':net,'variable':label,**quantiles(values)})
                pd.DataFrame(dist).to_csv(directory/'distributions.csv',index=False)
                pd.DataFrame([propensity_diagnostic(d['t'][tr],fitted['e_hat'],d['e_true'][tr],'TRAIN_OOF',seed,net),
                              propensity_diagnostic(d['t'][dv],devfit['e_hat'],d['e_true'][dv],'DEV',seed,net)]).to_csv(directory/'propensity_gof.csv',index=False)
                grid=[];preds={}
                for nuisance_label,phi in [('fitted',phi_fitted),('oracle',phi_oracle)]:
                    for alpha in ALPHAS:
                        model=fit_ridge(X[tr],phi,alpha)
                        pdev=model.predict(X[dv]);ptest=model.predict(X[te])
                        grid.append({'dgp_seed':seed,'net_seed':net,'nuisance':nuisance_label,'alpha':alpha,
                                     'dev_root_pehe':root_pehe(pdev,d['tau_true'][dv]),
                                     'test_root_pehe_for_audit_only':root_pehe(ptest,d['tau_true'][te])})
                        preds[(nuisance_label,alpha)]=ptest
                grid_df=pd.DataFrame(grid)
                selected=[]
                for nuisance_label in ['fitted','oracle']:
                    g=grid_df[grid_df.nuisance==nuisance_label]
                    best=g.sort_values(['dev_root_pehe','alpha']).iloc[0]
                    alpha=float(best['alpha']);pred=preds[(nuisance_label,alpha)]
                    selected.append({'dgp_seed':seed,'net_seed':net,'method':f'DR_Ridge_{nuisance_label}_DEV_selected',
                                     'alpha_selected_on_DEV':alpha,'dev_root_pehe':float(best['dev_root_pehe']),
                                     'test_root_pehe':root_pehe(pred,d['tau_true'][te]),
                                     'test_policy_regret':policy_regret(pred,d['tau_true'][te]),
                                     'test_clipped_root_pehe':root_pehe(np.clip(pred,-1,1),d['tau_true'][te]),
                                     'fraction_outside_minus1_plus1':float(np.mean(np.abs(pred)>1))})
                grid_df.to_csv(directory/'ridge_grid.csv',index=False)
                pd.DataFrame(selected).to_csv(directory/'selected_results.csv',index=False)
                write(complete,{'status':'COMPLETE','dgp_seed':seed,'net_seed':net,'started_utc':started,
                                'finished_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':code_hash,
                                'data_sha256':data_hash,'alpha_grid':ALPHAS,
                                'selection':'minimum DEV root PEHE; TEST never used to select alpha',
                                'training':'TRAIN OOF nuisance, 5 folds, 0.05-0.95 e clipping',
                                'oracle':'true DGP e,mu0,mu1 used only in oracle diagnostic',
                                'software':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__}})
                status['completed_pairs'].append(name)
            except Exception as exc:
                (directory/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
                status['failed_pairs'].append({'pair':name,'error':str(exc)})
            write(out/'RUN_STATUS.json',status)
    all_selected=[];all_dist=[];all_gof=[];all_grid=[]
    for name in status['completed_pairs']:
        directory=out/name
        for filename,dest in [('selected_results.csv',all_selected),('distributions.csv',all_dist),('propensity_gof.csv',all_gof),('ridge_grid.csv',all_grid)]:
            dest.append(pd.read_csv(directory/filename))
    for name,parts in [('per_run_results.csv',all_selected),('pseudooutcome_distributions.csv',all_dist),('propensity_gof.csv',all_gof),('ridge_sensitivity.csv',all_grid)]:
        if parts:pd.concat(parts,ignore_index=True).to_csv(out/name,index=False)
    if all_selected:
        s=pd.concat(all_selected,ignore_index=True)
        per_seed=s.groupby(['dgp_seed','method'],as_index=False).agg(test_root_pehe=('test_root_pehe','mean'),alpha_selected_mean=('alpha_selected_on_DEV','mean'))
        per_seed.to_csv(out/'per_seed_results.csv',index=False)
        per_seed.groupby('method',as_index=False).agg(mean_test_root_pehe=('test_root_pehe','mean'),sd_across_dgp=('test_root_pehe','std'),n_dgp=('dgp_seed','nunique')).to_csv(out/'summary_results.csv',index=False)
    status['status']='COMPLETE' if len(status['completed_pairs'])==status['requested_pairs'] and not status['failed_pairs'] else 'PARTIAL'
    status['finished_utc']=datetime.now(timezone.utc).isoformat();write(out/'RUN_STATUS.json',status)
    print(status['status'],len(status['completed_pairs']),'of',status['requested_pairs'])

if __name__=='__main__':main()
