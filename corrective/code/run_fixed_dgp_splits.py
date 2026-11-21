from __future__ import annotations
import argparse,json,traceback
import numpy as np,pandas as pd
from scipy.stats import t as student_t

from corrective_common import COR,DATA,LOCK,SRC,fixed_dgp,fixed_split,load_raw,now,preprocess_model,rloss,score,sha,write
from run_clinical_adjustment import fit_candidates,predict
from cm_osr.baselines import TLearner,XLearner
from cm_osr.nuisance import crossfit_dr,FullNuisance

OUT=COR/'results/fixed_dgp_splits';RUNS=OUT/'runs';RUNS.mkdir(parents=True,exist_ok=True)
SEEDS=LOCK['fixed_dgp']['dgp_seeds'];REGIMES=LOCK['fixed_dgp']['effect_regimes']
SPLITS=LOCK['fixed_dgp']['model_train_dev_resplit_seeds']
def one(raw,d,dgp_info,seed,regime,split_seed):
    tag=f'DGP{seed}_{regime}_SPLIT{split_seed}'
    folder=RUNS/tag;folder.mkdir(exist_ok=True)
    marker=folder/'COMPLETE.json';code_hash=sha(__file__);dep_hash=sha(COR/'code/run_clinical_adjustment.py')
    if marker.exists():
        old=json.loads(marker.read_text(encoding='utf-8'))
        if (old.get('code_sha256'),old.get('dependency_sha256'),old.get('dgp_npz_sha256'))==(
                code_hash,dep_hash,dgp_info['npz_sha256']):return 'SKIPPED'
    sp=fixed_split(len(raw['clinical']),split_seed);tr,dv,te=sp['train'],sp['dev'],sp['test']
    mods,X,C=preprocess_model(raw,tr)
    truth=d['tau_true'][te];ttr=d['t'][tr];ytr=d['y'][tr]
    rows=[score('zero_effect',np.zeros(len(te)),truth,seed,split_seed,regime,'fixed DGP')]
    rows.append(score('oracle_TEST_mean_constant_DIAGNOSTIC',np.full(len(te),np.mean(truth)),
                      truth,seed,split_seed,regime,'nondeployable TEST truth reference'))
    for feature_name,features in [('full',X),('clinical',C)]:
        for learner_name,cls in [('X_learner',XLearner),('T_learner',TLearner)]:
            model=cls().fit(features[tr],ttr,ytr)
            rows.append(score(feature_name+'_'+learner_name,model.predict(features[te]),
                              truth,seed,split_seed,regime,'TRAIN-only original fixed hyperparameters'))
    trainmods={k:v[tr] for k,v in mods.items()}
    oof=crossfit_dr(trainmods,ttr,ytr,n_splits=5,seed=17,clip=(.05,.95))
    nuisance=FullNuisance(seed=17,clip=(.05,.95)).fit(trainmods,ttr,ytr)
    dev=nuisance.predict({k:v[dv] for k,v in mods.items()})
    models,rdiag=fit_candidates(C[tr],ttr,ytr,oof)
    grid=[]
    for alpha,model in models.items():
        pdev=predict(model,C[dv])
        grid.append({'alpha':alpha,'dev_r_loss':rloss(pdev,d['t'][dv],d['y'][dv],
            dev['e_hat'],dev['mu0_hat'],dev['mu1_hat']),
            'dev_true_cate_root_pehe_DIAGNOSTIC':float(np.sqrt(np.mean((pdev-d['tau_true'][dv])**2)))})
    best=min(grid,key=lambda q:(q['dev_r_loss'],str(q['alpha'])))
    ptest=predict(models[best['alpha']],C[te])
    rows.append(score('clinical_R_full_nuisance_Rloss_selected',ptest,truth,seed,split_seed,
                      regime,'TRAIN crossfit full nuisance; clinical-only effect; DEV observed R-loss'))
    for row in rows:
        row['dgp_npz_sha256']=dgp_info['npz_sha256']
        row['cate_test_sha256']=sha_bytes(truth)
        row['test_indices_sha256']=sha_bytes(te)
        row['selected_ridge_alpha']=best['alpha'] if row['method']=='clinical_R_full_nuisance_Rloss_selected' else ''
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(grid).to_csv(folder/'dev_ridge_grid.csv',index=False)
    write(marker,{'status':'COMPLETE','dgp_seed':seed,'effect_regime':regime,
       'split_seed':split_seed,'finished_utc':now(),
       'code_sha256':code_hash,'dependency_sha256':dep_hash,
       'data_sha256':sha(DATA),'protocol_lock_sha256':sha(COR/'protocol_lock.json'),
       'dgp_npz_sha256':dgp_info['npz_sha256'],
       'cate_full_sha256':dgp_info['cate_sha256'],
       'cate_test_sha256':sha_bytes(truth),'test_indices_sha256':sha_bytes(te),
       'n_train':len(tr),'n_dev':len(dv),'n_test':len(te),
       'selected_ridge_alpha':best['alpha'],'selected_dev_r_loss':best['dev_r_loss'],
       'nuisance_diagnostic':rdiag,
       'selection':'DEV observed R-loss only; TEST evaluated once'})
    return 'COMPLETE'
def sha_bytes(x):
    import hashlib
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def aggregate():
    markers=list(RUNS.glob('*/COMPLETE.json'))
    if markers:
        per=pd.concat([pd.read_csv(p.parent/'metrics.csv') for p in markers],ignore_index=True)
        per.to_csv(OUT/'per_split_results.csv',index=False)
        clusters=per.groupby(['dgp_seed','effect_regime','method'],as_index=False).agg(
            root_pehe=('root_pehe','mean'),policy_regret=('policy_regret','mean'),
            split_sd=('root_pehe','std'),n_splits=('split_seed','nunique'))
        clusters.to_csv(OUT/'dgp_cluster_results.csv',index=False)
        summary=clusters.groupby(['effect_regime','method'],as_index=False).agg(
            mean_root_pehe=('root_pehe','mean'),sd_dgp=('root_pehe','std'),
            mean_policy_regret=('policy_regret','mean'),mean_within_dgp_split_sd=('split_sd','mean'),
            n_dgp=('dgp_seed','nunique'))
        summary['mcse_dgp']=summary.sd_dgp/np.sqrt(summary.n_dgp)
        summary['ci95_lower']=summary.mean_root_pehe-summary.mcse_dgp*summary.n_dgp.map(
            lambda n:student_t.ppf(.975,n-1) if n>1 else np.nan)
        summary['ci95_upper']=summary.mean_root_pehe+summary.mcse_dgp*summary.n_dgp.map(
            lambda n:student_t.ppf(.975,n-1) if n>1 else np.nan)
        summary['inference_ready']=(summary.n_dgp==len(SEEDS))&(len(markers)==len(SEEDS)*len(REGIMES)*len(SPLITS))
        summary.to_csv(OUT/'summary_results.csv',index=False)
    write(OUT/'RUN_STATUS.json',{'status':'COMPLETE' if len(markers)==len(SEEDS)*len(REGIMES)*len(SPLITS) else 'PARTIAL',
       'completed_units':len(markers),'expected_units':len(SEEDS)*len(REGIMES)*len(SPLITS),
       'updated_utc':now(),'independence':'DGP seed cluster n=12; three resplits share fixed TEST and DGP truth'})
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--seed-start',type=int,default=0);ap.add_argument('--seed-count',type=int,default=12)
    a=ap.parse_args();raw=load_raw();seeds=SEEDS[a.seed_start:a.seed_start+a.seed_count]
    for seed in seeds:
        for regime in (REGIMES[:1] if a.smoke else REGIMES):
            d,info=fixed_dgp(raw,seed,regime)
            for split_seed in (SPLITS[:1] if a.smoke else SPLITS):
                try:
                    print(seed,regime,split_seed,one(raw,d,info,seed,regime,split_seed),flush=True)
                except Exception:
                    folder=RUNS/f'DGP{seed}_{regime}_SPLIT{split_seed}';folder.mkdir(exist_ok=True)
                    (folder/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
                    print(seed,regime,split_seed,'ERROR',flush=True)
                aggregate()
        if a.smoke:break
if __name__=='__main__':main()
