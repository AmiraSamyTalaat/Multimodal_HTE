from __future__ import annotations
import hashlib,json,platform,sys,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,scipy,sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path('/kaggle/working');OUT=ROOT/'classical_sensitivity';OUT.mkdir(exist_ok=True)
SOURCE=Path('/kaggle/input/b01-causaltx-mm-core-runtime/B01_CORE_BUNDLE')
if not SOURCE.exists():
    found=list(Path('/kaggle/input').rglob('ispy2_core_multimodal.npz'))
    if len(found)!=1:raise RuntimeError(f'Expected one attached original B01 dataset, got {found}')
    SOURCE=found[0].parents[2]
sys.path.insert(0,str(SOURCE/'01_CODE'))
from cm_osr.dgp import generate
from cm_osr.metrics import root_pehe,policy_regret
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
from cm_osr.utils import concat_modalities

SEEDS=[101,211,307,401,503];LEAVES=[4,8,16]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False),encoding='utf-8')
def forest(leaf,seed,X,y):
    return RandomForestRegressor(n_estimators=300,min_samples_leaf=leaf,
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
        d1=y[t==1]-self.mu[0].predict(X[t==1])
        d0=self.mu[1].predict(X[t==0])-y[t==0]
        self.t1=forest(self.leaf,33,X[t==1],d1)
        self.t0=forest(self.leaf,35,X[t==0],d0)
        self.e=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(X,t)
        return self
    def predict(self,X):
        e=self.e.predict_proba(X)[:,1]
        return e*self.t0.predict(X)+(1-e)*self.t1.predict(X)
def main():
    data=SOURCE/'02_DATA/processed/ispy2_core_multimodal.npz'
    with np.load(data,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    sp=locked_split(len(raw['clinical']));tr,dv,te=sp['train'],sp['dev'],sp['test']
    mods=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}).transform(raw)
    X=concat_modalities(mods);C=mods['clinical'];datahash=sha(data)
    finished=[];errors=[]
    for seed in SEEDS:
        folder=OUT/f'DGP{seed}';folder.mkdir(exist_ok=True)
        if (folder/'COMPLETE.json').exists():finished.append(seed);continue
        try:
            d=generate(mods,seed,'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
            t=d['t'][tr];y=d['y'][tr]
            grid=[];selected=[]
            for feature_name,features in [('full',X),('clinical',C)]:
                for learner_name,constructor in [('T_learner',TModel),('X_learner',XModel)]:
                    models={}
                    for leaf in LEAVES:
                        model=constructor(leaf).fit(features[tr],t,y)
                        score=root_pehe(model.predict(features[dv]),d['tau_true'][dv])
                        grid.append({'dgp_seed':seed,'feature_set':feature_name,
                          'learner':learner_name,'min_samples_leaf':leaf,'n_estimators':300,
                          'dev_root_pehe':score})
                        models[leaf]=model
                    subset=[r for r in grid if r['feature_set']==feature_name and r['learner']==learner_name]
                    best=min(subset,key=lambda r:(r['dev_root_pehe'],r['min_samples_leaf']))
                    pred=models[best['min_samples_leaf']].predict(features[te])
                    selected.append({'dgp_seed':seed,'feature_set':feature_name,'learner':learner_name,
                      'selected_min_samples_leaf':best['min_samples_leaf'],
                      'selected_dev_root_pehe':best['dev_root_pehe'],
                      'test_root_pehe':root_pehe(pred,d['tau_true'][te]),
                      'test_policy_regret':policy_regret(pred,d['tau_true'][te]),
                      'n_test':len(te)})
            pd.DataFrame(grid).to_csv(folder/'dev_grid.csv',index=False)
            pd.DataFrame(selected).to_csv(folder/'selected_test.csv',index=False)
            write(folder/'COMPLETE.json',{'status':'COMPLETE','dgp_seed':seed,
                'finished_utc':datetime.now(timezone.utc).isoformat(),
                'data_sha256':datahash,'n_train':len(tr),'n_dev':len(dv),'n_test':len(te),
                'selection':'DEV true CATE selects min_samples_leaf; TEST only evaluates selected model',
                'grid':{'min_samples_leaf':LEAVES,'n_estimators':[300]},
                'software':{'python':platform.python_version(),'numpy':np.__version__,
                            'scipy':scipy.__version__,'sklearn':sklearn.__version__}})
            finished.append(seed)
        except Exception as exc:
            (folder/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
            errors.append({'seed':seed,'error':str(exc)})
        print('DGP',seed,'COMPLETE' if seed in finished else 'ERROR',flush=True)
    frames=[pd.read_csv(OUT/f'DGP{s}'/'selected_test.csv') for s in finished]
    if frames:
        pd.concat(frames,ignore_index=True).to_csv(OUT/'per_seed_results.csv',index=False)
        pd.concat([pd.read_csv(OUT/f'DGP{s}'/'dev_grid.csv') for s in finished],ignore_index=True).to_csv(OUT/'dev_grid.csv',index=False)
    status={'status':'COMPLETE' if len(finished)==len(SEEDS) else 'PARTIAL',
      'completed_seeds':finished,'errors':errors,'finished_utc':datetime.now(timezone.utc).isoformat(),
      'data_sha256':datahash}
    write(OUT/'RUN_STATUS.json',status)
    with zipfile.ZipFile(ROOT/'B01_P1_CLASSICAL_SENSITIVITY_OUTPUT.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
    print(json.dumps(status),flush=True)
if __name__=='__main__':main()
