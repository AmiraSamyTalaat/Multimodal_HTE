from __future__ import annotations
import argparse,json,traceback
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from corrective_common import COR,DATA,LOCK,load_raw,now,preprocess_model,rloss,score,sha,write
from cm_osr.dgp import generate
from cm_osr.nuisance import FullNuisance
from cm_osr.splits import locked_split

OUT=COR/'results/nonoracle_tuning';OUT.mkdir(parents=True,exist_ok=True)
LEAVES=LOCK['nonoracle_tuning']['classical_min_samples_leaf_grid']
def forest(leaf,seed,X,y):
    return RandomForestRegressor(n_estimators=300,min_samples_leaf=int(leaf),
        random_state=seed,n_jobs=-1).fit(X,y)
class TModel:
    def __init__(self,leaf):self.leaf=leaf
    def fit(self,X,t,y):
        self.m=[forest(self.leaf,17+a,X[t==a],y[t==a]) for a in [0,1]]
        return self
    def predict(self,X):return self.m[1].predict(X)-self.m[0].predict(X)
class XModel:
    def __init__(self,leaf):self.leaf=leaf
    def fit(self,X,t,y):
        self.mu=[forest(self.leaf,31+a,X[t==a],y[t==a]) for a in [0,1]]
        d1=y[t==1]-self.mu[0].predict(X[t==1]);d0=self.mu[1].predict(X[t==0])-y[t==0]
        self.t1=forest(self.leaf,33,X[t==1],d1)
        self.t0=forest(self.leaf,35,X[t==0],d0)
        self.e=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(X,t)
        return self
    def predict(self,X):
        e=self.e.predict_proba(X)[:,1]
        return e*self.t0.predict(X)+(1-e)*self.t1.predict(X)
def one(raw,seed):
    folder=OUT/f'DGP{seed}';folder.mkdir(exist_ok=True)
    marker=folder/'COMPLETE.json';code_hash=sha(__file__)
    if marker.exists():
        info=json.loads(marker.read_text(encoding='utf-8'))
        if info.get('code_sha256')==code_hash and info.get('data_sha256')==sha(DATA):return 'SKIPPED'
    sp=locked_split(len(raw['clinical']),seed=20260918);tr,dv,te=sp['train'],sp['dev'],sp['test']
    mods,X,C=preprocess_model(raw,tr)
    d=generate(mods,int(seed),'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
    t=d['t'][tr];y=d['y'][tr]
    nuisance=FullNuisance(seed=17,clip=(.05,.95)).fit({k:v[tr] for k,v in mods.items()},t,y)
    ndv=nuisance.predict({k:v[dv] for k,v in mods.items()})
    grid=[];selected=[]
    for feature_name,features in [('full',X),('clinical',C)]:
        for learner_name,cls in [('X_learner',XModel),('T_learner',TModel)]:
            models={}
            for leaf in LEAVES:
                model=cls(leaf).fit(features[tr],t,y);models[leaf]=model
                pdev=model.predict(features[dv])
                grid.append({'dgp_seed':seed,'feature_set':feature_name,
                    'learner':learner_name,'min_samples_leaf':leaf,
                    'dev_observed_r_loss':rloss(pdev,d['t'][dv],d['y'][dv],
                        ndv['e_hat'],ndv['mu0_hat'],ndv['mu1_hat']),
                    'dev_true_cate_root_pehe_DIAGNOSTIC':float(np.sqrt(np.mean((pdev-d['tau_true'][dv])**2)))})
            choices=[r for r in grid if r['feature_set']==feature_name and r['learner']==learner_name]
            for criterion,key,tag in [('observed_R_loss','dev_observed_r_loss','NONORACLE'),
                                      ('true_DEV_CATE','dev_true_cate_root_pehe_DIAGNOSTIC','ORACLE_DIAGNOSTIC')]:
                best=min(choices,key=lambda r:(r[key],r['min_samples_leaf']))
                pred=models[best['min_samples_leaf']].predict(features[te])
                row=score(f'{feature_name}_{learner_name}_{tag}',pred,d['tau_true'][te],
                          seed,20260918,'ORIGINAL_PRIMARY',
                          'TRAIN fit; '+criterion+' DEV selection')
                row.update({'selected_min_samples_leaf':best['min_samples_leaf'],
                            'selection_criterion':criterion,'selected_dev_score':best[key],
                            'n_candidates':len(LEAVES)})
                selected.append(row)
    pd.DataFrame(grid).to_csv(folder/'dev_grid.csv',index=False)
    pd.DataFrame(selected).to_csv(folder/'selected_test.csv',index=False)
    write(marker,{'status':'COMPLETE','dgp_seed':seed,'finished_utc':now(),
       'code_sha256':code_hash,'data_sha256':sha(DATA),
       'protocol_lock_sha256':sha(COR/'protocol_lock.json'),
       'grid':{'min_samples_leaf':LEAVES,'n_estimators':[300]},
       'selection':'Observed DEV R-loss for NONORACLE; true DEV CATE only for labeled diagnostic',
       'test_access':'TEST evaluated only after both DEV selections'})
    return 'COMPLETE'
def aggregate():
    files=sorted(OUT.glob('DGP*/selected_test.csv'))
    if files:
        per=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
        per.to_csv(OUT/'per_seed_results.csv',index=False)
        sm=per.groupby('method',as_index=False).agg(mean_root_pehe=('root_pehe','mean'),
            sd_dgp=('root_pehe','std'),mean_policy_regret=('policy_regret','mean'),
            n_dgp=('dgp_seed','nunique'))
        sm['mcse_dgp']=sm.sd_dgp/np.sqrt(sm.n_dgp)
        sm.to_csv(OUT/'summary_results.csv',index=False)
    n=len(list(OUT.glob('DGP*/COMPLETE.json')))
    write(OUT/'RUN_STATUS.json',{'status':'COMPLETE' if n==5 else 'PARTIAL',
        'completed_dgp':n,'expected_dgp':5,'updated_utc':now(),
        'n_candidate_fits_per_dgp':4*len(LEAVES),
        'independence':'Five original DGP seeds; do not count network repeats'})
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();raw=load_raw()
    seeds=LOCK['nonoracle_tuning']['primary_dgp_seeds'][:1] if a.smoke else LOCK['nonoracle_tuning']['primary_dgp_seeds']
    for seed in seeds:
        try:print(seed,one(raw,seed),flush=True)
        except Exception:
            folder=OUT/f'DGP{seed}';folder.mkdir(exist_ok=True)
            (folder/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print(seed,'ERROR',flush=True)
        aggregate()
if __name__=='__main__':main()
