from __future__ import annotations
import argparse,json,traceback
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge

from corrective_common import COR,REV,SRC,DATA,LOCK,load_raw,now,preprocess_model,rloss,score,sha,write
from cm_osr.dgp import generate
from cm_osr.nuisance import crossfit_dr,FullNuisance
from cm_osr.splits import locked_split

OUT=COR/'results/clinical_adjustment';OUT.mkdir(parents=True,exist_ok=True)
ALPHAS=LOCK['clinical_adjustment']['ridge_alpha_grid']
def fit_candidates(Ctr,t,y,oof):
    e=np.asarray(oof['e_hat'],dtype=float)
    m=e*np.asarray(oof['mu1_hat'])+(1-e)*np.asarray(oof['mu0_hat'])
    u=np.asarray(t)-e;v=np.asarray(y)-m
    z=v/u;w=u*u
    models={}
    for a in ALPHAS:
        if a=='constant_limit':models[str(a)]=float(np.sum(w*z)/np.sum(w))
        else:models[str(a)]=Ridge(alpha=float(a)).fit(Ctr,z,sample_weight=w)
    return models,{'train_r_score_mean':float(np.mean(u*v)),
                   'train_u2_mean':float(np.mean(w)),
                   'train_pseudo_rms':float(np.sqrt(np.mean(z*z))),
                   'train_abs_pseudo_gt_5':float(np.mean(np.abs(z)>5))}
def predict(model,C):
    return np.full(len(C),model,dtype=float) if isinstance(model,float) else model.predict(C)
def run_one(raw,seed):
    folder=OUT/f'DGP{seed}';folder.mkdir(exist_ok=True)
    marker=folder/'COMPLETE.json';code_hash=sha(__file__);data_hash=sha(DATA)
    if marker.exists():
        info=json.loads(marker.read_text(encoding='utf-8'))
        if info.get('code_sha256')==code_hash and info.get('data_sha256')==data_hash:
            return pd.read_csv(folder/'selected_results.csv')
    split=locked_split(len(raw['clinical']),seed=LOCK['clinical_adjustment']['original_split_seed'])
    tr,dv,te=split['train'],split['dev'],split['test']
    mods,X,C=preprocess_model(raw,tr)
    d=generate(mods,int(seed),'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
    t=d['t'][tr];y=d['y'][tr]
    nuisance_cases={}
    full_oof=crossfit_dr({k:v[tr] for k,v in mods.items()},t,y,n_splits=5,seed=17,clip=(.05,.95))
    full=FullNuisance(seed=17,clip=(.05,.95)).fit({k:v[tr] for k,v in mods.items()},t,y)
    nuisance_cases['full_fitted']=(full_oof,full.predict({k:v[dv] for k,v in mods.items()}),
                                  full.predict({k:v[te] for k,v in mods.items()}))
    clinical_oof=crossfit_dr({'clinical':C[tr]},t,y,n_splits=5,seed=17,clip=(.05,.95))
    clinical=FullNuisance(seed=17,clip=(.05,.95)).fit({'clinical':C[tr]},t,y)
    nuisance_cases['clinical_fitted']=(clinical_oof,clinical.predict({'clinical':C[dv]}),
                                      clinical.predict({'clinical':C[te]}))
    true_nuis={k:d[k][tr] for k in ['e_true','mu0_true','mu1_true']}
    true_nuis={'e_hat':true_nuis['e_true'],'mu0_hat':true_nuis['mu0_true'],
               'mu1_hat':true_nuis['mu1_true']}
    nuisance_cases['oracle_true_simulation']=(true_nuis,
        {'e_hat':d['e_true'][dv],'mu0_hat':d['mu0_true'][dv],'mu1_hat':d['mu1_true'][dv]},
        {'e_hat':d['e_true'][te],'mu0_hat':d['mu0_true'][te],'mu1_hat':d['mu1_true'][te]})
    grid=[];rows=[];diagnostics=[]
    for label,(oof,ndv,nte) in nuisance_cases.items():
        models,diag=fit_candidates(C[tr],t,y,oof)
        diagnostics.append({'dgp_seed':seed,'nuisance':label,**diag,
            'dev_propensity_brier':float(np.mean((d['t'][dv]-ndv['e_hat'])**2)),
            'dev_e_true_rmse':float(np.sqrt(np.mean((ndv['e_hat']-d['e_true'][dv])**2)))})
        for name,model in models.items():
            pred=predict(model,C[dv])
            grid.append({'dgp_seed':seed,'nuisance':label,'alpha':name,
                'dev_r_loss':rloss(pred,d['t'][dv],d['y'][dv],ndv['e_hat'],ndv['mu0_hat'],ndv['mu1_hat']),
                'dev_true_cate_root_pehe_diagnostic':float(np.sqrt(np.mean((pred-d['tau_true'][dv])**2)))})
        subset=[r for r in grid if r['nuisance']==label]
        best=min(subset,key=lambda r:(r['dev_r_loss'],str(r['alpha'])))
        testpred=predict(models[best['alpha']],C[te])
        method=f'clinical_R_{label}_Rloss_selected'
        rows.append(score(method,testpred,d['tau_true'][te],seed,20260918,'ORIGINAL_PRIMARY',
                          'new TRAIN-fitted clinical effect with '+label+' nuisance'))
        rows[-1].update({'selected_alpha':best['alpha'],'dev_selection_criterion':'observed_R_loss',
                         'selected_dev_r_loss':best['dev_r_loss']})
        if label=='full_fitted':
            oracle_choice=min(subset,key=lambda r:(r['dev_true_cate_root_pehe_diagnostic'],str(r['alpha'])))
            opred=predict(models[oracle_choice['alpha']],C[te])
            rows.append(score('clinical_R_full_fitted_trueDEV_selected_DIAGNOSTIC',opred,
                              d['tau_true'][te],seed,20260918,'ORIGINAL_PRIMARY',
                              'nondeployable true-DEV-CATE selection diagnostic'))
            rows[-1].update({'selected_alpha':oracle_choice['alpha'],
                             'dev_selection_criterion':'true_DEV_CATE_non_deployable',
                             'selected_dev_r_loss':oracle_choice['dev_r_loss']})
    old=pd.read_csv(REV/'results/baselines/per_seed_results.csv')
    mapping={'clinical_only_X_learner':'archived_clinical_X',
             'clinical_only_T_learner':'archived_clinical_T',
             'X_learner':'archived_full_X','T_learner_regularized':'archived_full_T',
             'zero_effect':'archived_zero_effect'}
    for original,new in mapping.items():
        rec=old[(old.dgp_seed==seed)&(old.method==original)]
        if len(rec)!=1:raise RuntimeError(f'Missing archived comparator {seed} {original}')
        r=rec.iloc[0]
        rows.append({'dgp_seed':seed,'split_seed':20260918,'effect_regime':'ORIGINAL_PRIMARY',
                     'method':new,'root_pehe':float(r.root_pehe),
                     'policy_regret':float(r.policy_regret),'clipped_root_pehe':float(r.clipped_root_pehe),
                     'fraction_outside_minus1_plus1':float(r.fraction_outside_minus1_plus1),
                     'n_test':148,'source':'archived TEST predictions; no refit',
                     'selected_alpha':'','dev_selection_criterion':'none_archived',
                     'selected_dev_r_loss':np.nan})
    pd.DataFrame(grid).to_csv(folder/'dev_grid.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(folder/'nuisance_diagnostics.csv',index=False)
    pd.DataFrame(rows).to_csv(folder/'selected_results.csv',index=False)
    write(marker,{'status':'COMPLETE','dgp_seed':seed,'finished_utc':now(),
          'code_sha256':code_hash,'data_sha256':data_hash,
          'protocol_lock_sha256':sha(COR/'protocol_lock.json'),
          'n_train':len(tr),'n_dev':len(dv),'n_test':len(te),
          'model':'Weighted R-learner with clinical-only effect stage; full/clinical/oracle nuisance variants',
          'selection':'Observed DEV R-loss for operational variants; true DEV CATE only in labeled diagnostic',
          'test_access':'Final evaluation only'})
    return pd.DataFrame(rows)
def aggregate():
    parts=[pd.read_csv(p) for p in sorted(OUT.glob('DGP*/selected_results.csv'))]
    if parts:
        per=pd.concat(parts,ignore_index=True);per.to_csv(OUT/'per_seed_results.csv',index=False)
        sm=per.groupby('method',as_index=False).agg(mean_root_pehe=('root_pehe','mean'),
             sd_dgp=('root_pehe','std'),mean_policy_regret=('policy_regret','mean'),
             n_dgp=('dgp_seed','nunique'))
        sm['mcse_dgp']=sm.sd_dgp/np.sqrt(sm.n_dgp)
        sm.to_csv(OUT/'summary_results.csv',index=False)
    n=len(list(OUT.glob('DGP*/COMPLETE.json')))
    write(OUT/'RUN_STATUS.json',{'status':'COMPLETE' if n==5 else 'PARTIAL',
          'completed_dgp':n,'expected_dgp':5,'updated_utc':now(),
          'independence':'Five original DGP seeds; three archived network seeds are not independent'})
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();raw=load_raw()
    seeds=LOCK['clinical_adjustment']['original_primary_dgp_seeds'][:1] if a.smoke else LOCK['clinical_adjustment']['original_primary_dgp_seeds']
    for seed in seeds:
        try:run_one(raw,seed);print('DGP',seed,'COMPLETE',flush=True)
        except Exception:
            folder=OUT/f'DGP{seed}';folder.mkdir(exist_ok=True)
            (folder/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print('DGP',seed,'ERROR',flush=True)
        aggregate()
if __name__=='__main__':main()
