from __future__ import annotations
import argparse, hashlib, json, platform, sys, traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,scipy,sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REV=Path(__file__).resolve().parents[1];SRC=REV/'source_snapshot'
sys.path.insert(0,str(SRC/'01_CODE'))
from cm_osr.baselines import TLearner,XLearner
from cm_osr.metrics import root_pehe,policy_regret
from cm_osr.nuisance import crossfit_dr
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
from cm_osr.stage7_dgp import generate_stage7
from cm_osr.utils import concat_modalities

DGP_SEEDS=[3001+67*i for i in range(30)]
SPLIT_SEEDS=[20260918,20260929,20261013]
REGIMES=['NO_INCREMENTAL_MOLECULAR_SIGNAL','TRANSCRIPTOME_INCREMENTAL','JOINT_MOLECULAR_SYNERGY']
ALPHAS=[1.,10.,100.,1000.,10000.]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def score(method,pred,true,seed,split_seed,regime):
    return {'dgp_seed':seed,'patient_split_seed':split_seed,'effect_regime':regime,'method':method,
            'n_test':len(true),'root_pehe':root_pehe(pred,true),'policy_regret':policy_regret(pred,true),
            'clipped_root_pehe':root_pehe(np.clip(pred,-1,1),true),
            'fraction_outside_minus1_plus1':float(np.mean(np.abs(pred)>1))}
def run_unit(raw,seed,split_seed,regime,out,data_hash,code_hash):
    tag=f'DGP{seed}_SPLIT{split_seed}_{regime}';folder=out/'runs'/tag;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'COMPLETE.json'
    if marker.exists():
        old=json.loads(marker.read_text(encoding='utf-8'))
        if old.get('code_sha256')==code_hash and old.get('data_sha256')==data_hash:
            return 'SKIPPED'
    started=datetime.now(timezone.utc).isoformat()
    sp=locked_split(len(raw['clinical']),seed=split_seed)
    tr,dv,te=sp['train'],sp['dev'],sp['test']
    assert (len(tr),len(dv),len(te))==(441,147,148)
    mods=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}).transform(raw)
    d=generate_stage7(mods,seed,regime,reference_idx=tr)
    true=d['tau_true'][te];X=concat_modalities(mods);C=mods['clinical']
    t=d['t'][tr];y=d['y'][tr]
    rows=[]
    rows.append(score('zero_effect',np.zeros(len(te)),true,seed,split_seed,regime))
    rows.append(score('oracle_test_mean_constant',np.full(len(te),np.mean(true)),true,seed,split_seed,regime))
    rows.append(score('full_X_learner',XLearner().fit(X[tr],t,y).predict(X[te]),true,seed,split_seed,regime))
    rows.append(score('full_T_learner',TLearner().fit(X[tr],t,y).predict(X[te]),true,seed,split_seed,regime))
    rows.append(score('clinical_X_learner',XLearner().fit(C[tr],t,y).predict(C[te]),true,seed,split_seed,regime))
    rows.append(score('clinical_T_learner',TLearner().fit(C[tr],t,y).predict(C[te]),true,seed,split_seed,regime))
    oof=crossfit_dr({k:v[tr] for k,v in mods.items()},t,y,n_splits=5,seed=17,clip=(.05,.95))
    ridge_grid=[];preds={}
    for alpha in ALPHAS:
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(X[tr],oof['phi'])
        pdev=model.predict(X[dv]);preds[alpha]=model.predict(X[te])
        ridge_grid.append({'alpha':alpha,'dev_root_pehe':root_pehe(pdev,d['tau_true'][dv])})
    best=min(ridge_grid,key=lambda x:(x['dev_root_pehe'],x['alpha']))
    rows.append(score('DR_Ridge_alpha10',preds[10.],true,seed,split_seed,regime))
    rows.append(score('DR_Ridge_DEV_selected',preds[best['alpha']],true,seed,split_seed,regime))
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(ridge_grid).to_csv(folder/'ridge_dev_grid.csv',index=False)
    write(marker,{'status':'COMPLETE','tag':tag,'started_utc':started,
                  'finished_utc':datetime.now(timezone.utc).isoformat(),
                  'dgp_seed':seed,'patient_split_seed':split_seed,'effect_regime':regime,
                  'data_sha256':data_hash,'code_sha256':code_hash,
                  'n_train':len(tr),'n_dev':len(dv),'n_test':len(te),
                  'effect_projections':'stage7_dgp.generate_stage7 draws incremental projection using DGP seed',
                  'ridge_alpha_selected_on_DEV':best['alpha'],'ridge_dev_grid':ridge_grid,
                  'test_access':'Only final evaluation; oracle constant is a labeled nondeployable reference',
                  'software':{'python':platform.python_version(),'numpy':np.__version__,
                              'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__}})
    return 'COMPLETE'
def aggregate(out,expected):
    frames=[];units=[]
    for p in sorted((out/'runs').glob('*/COMPLETE.json')):
        info=json.loads(p.read_text(encoding='utf-8'))
        frames.append(pd.read_csv(p.parent/'metrics.csv'))
        units.append(info)
    if frames:
        per=pd.concat(frames,ignore_index=True)
        per.to_csv(out/'per_seed_results.csv',index=False)
        summary=(per.groupby(['effect_regime','method'],as_index=False)
                 .agg(mean_root_pehe=('root_pehe','mean'),sd_all_units=('root_pehe','std'),
                      mean_policy_regret=('policy_regret','mean'),n_dgp=('dgp_seed','nunique'),
                      n_splits=('patient_split_seed','nunique'),n_units=('root_pehe','count')))
        cluster=(per.groupby(['effect_regime','method','dgp_seed'],as_index=False)
                 .agg(root_pehe=('root_pehe','mean'),policy_regret=('policy_regret','mean')))
        mcse=(cluster.groupby(['effect_regime','method'])['root_pehe'].agg(['std','count']).reset_index())
        mcse['mcse_dgp_cluster']=mcse['std']/np.sqrt(mcse['count'])
        summary=summary.merge(mcse[['effect_regime','method','mcse_dgp_cluster']],on=['effect_regime','method'])
        summary.to_csv(out/'summary_results.csv',index=False)
        cluster.to_csv(out/'dgp_cluster_means.csv',index=False)
        splitmeans=(per.groupby(['effect_regime','method','patient_split_seed'],as_index=False)
                    .agg(mean_root_pehe=('root_pehe','mean')))
        splitmeans.to_csv(out/'split_sensitivity.csv',index=False)
    status={'stage':'REPLICATION_POSTREVIEW','status':'COMPLETE' if len(units)==expected else 'PARTIAL',
            'expected_units':expected,'completed_units':len(units),
            'independence':'30 DGP seeds; 3 reused patient-level split seeds; 3 effect regimes. DGP-cluster MCSE averages split means within each seed.',
            'last_updated_utc':datetime.now(timezone.utc).isoformat()}
    write(out/'RUN_STATUS.json',status)
    return status
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--seed-start',type=int,default=0);ap.add_argument('--seed-count',type=int,default=30)
    a=ap.parse_args();out=REV/'results'/'replication';out.mkdir(parents=True,exist_ok=True);(out/'runs').mkdir(exist_ok=True)
    data=SRC/'02_DATA/processed/ispy2_core_multimodal.npz';data_hash=sha(data);code_hash=sha(__file__)
    with np.load(data,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    seeds=DGP_SEEDS[a.seed_start:a.seed_start+a.seed_count]
    if a.smoke:seeds=seeds[:1];splits=SPLIT_SEEDS[:1];regimes=REGIMES[:1]
    else:splits=SPLIT_SEEDS;regimes=REGIMES
    errors=[]
    for seed in seeds:
        for split_seed in splits:
            for regime in regimes:
                tag=f'DGP{seed}_SPLIT{split_seed}_{regime}'
                try:print(tag,run_unit(raw,seed,split_seed,regime,out,data_hash,code_hash),flush=True)
                except Exception as exc:
                    folder=out/'runs'/tag;folder.mkdir(exist_ok=True)
                    (folder/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
                    errors.append({'tag':tag,'error':str(exc)})
                aggregate(out,len(DGP_SEEDS)*len(SPLIT_SEEDS)*len(REGIMES))
    write(out/'ERRORS.json',errors)
    print('batch done',len(seeds),'seeds','errors',len(errors),flush=True)
if __name__=='__main__':main()
