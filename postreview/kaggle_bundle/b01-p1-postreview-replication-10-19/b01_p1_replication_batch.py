from __future__ import annotations
import argparse, hashlib, json, platform, sys, traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,scipy,sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REV=Path('/kaggle/working')
SRC=Path('/kaggle/input/b01-causaltx-mm-core-runtime/B01_CORE_BUNDLE')
if not SRC.exists():
    matches=list(Path('/kaggle/input').rglob('ispy2_core_multimodal.npz'))
    if len(matches)!=1:raise RuntimeError(f'Expected one original B01 private dataset; got {matches}')
    SRC=matches[0].parents[2]

sys.path.insert(0,str(SRC/'01_CODE'))
import types
_stage7=types.ModuleType('cm_osr.stage7_dgp')
_stage7.__package__='cm_osr'
sys.modules['cm_osr.stage7_dgp']=_stage7
exec('from __future__ import annotations\nimport numpy as np\nfrom scipy.special import expit\nfrom sklearn.decomposition import PCA\nfrom sklearn.linear_model import Ridge\n\nfrom .dgp import _proj\n\nREGIMES = [\n    "NO_INCREMENTAL_MOLECULAR_SIGNAL",\n    "TRANSCRIPTOME_INCREMENTAL",\n    "RPPA_INCREMENTAL",\n    "JOINT_MOLECULAR_SYNERGY",\n    "PATIENT_SUBGROUP_SWITCHING",\n]\n\n\ndef _std_train(z, tr):\n    z=np.asarray(z,dtype=float)\n    mu=float(np.mean(z[tr])); sd=float(np.std(z[tr]))\n    return ((z-mu)/(sd+1e-8)).astype(np.float32)\n\n\ndef _residualized_pc_score(X, C, tr, seed, n_components=5):\n    X=np.asarray(X,dtype=float); C=np.asarray(C,dtype=float); tr=np.asarray(tr,dtype=int)\n    k=max(1,min(int(n_components),X.shape[1],len(tr)-1))\n    pca=PCA(n_components=k,svd_solver="full").fit(X[tr])\n    pc=pca.transform(X)\n    rng=np.random.default_rng(int(seed))\n    w=rng.normal(size=k); w=w/(np.linalg.norm(w)+1e-12)\n    raw=pc@w\n    reg=Ridge(alpha=1.0).fit(C[tr],raw[tr])\n    resid=raw-reg.predict(C)\n    return _std_train(resid,tr)\n\n\ndef _joint_score(g,p,C,tr):\n    raw=np.asarray(g)*np.asarray(p)\n    Z=np.column_stack([C,np.asarray(g),np.asarray(p)])\n    reg=Ridge(alpha=1.0).fit(Z[tr],raw[tr])\n    resid=raw-reg.predict(Z)\n    return _std_train(resid,tr)\n\n\ndef generate_stage7(modalities, seed, regime, reference_idx, incremental_ratio=0.60, propensity_clip=(0.05,0.95)):\n    if regime not in REGIMES:\n        raise ValueError(f"Unknown Stage7 regime: {regime}")\n    tr=np.asarray(reference_idx,dtype=int)\n    C=np.asarray(modalities["clinical"],dtype=np.float32)\n    T=np.asarray(modalities["transcriptomic"],dtype=np.float32)\n    R=np.asarray(modalities["rppa"],dtype=np.float32)\n\n    c=_proj(C,1101,tr)\n    g_prog=_proj(T,2203,tr); p_prog=_proj(R,3307,tr)\n    g2=_proj(T,4409,tr); p2=_proj(R,5519,tr)\n\n    # Prognostic/assignment mechanisms remain multimodal, so molecular variables\n    # can be confounder/prognostic information even in the no-incremental-effect regime.\n    logit_e=0.25*c+0.45*g_prog+0.45*p_prog+0.65*g_prog*p_prog\n    e=np.clip(expit(logit_e),propensity_clip[0],propensity_clip[1])\n    mu0=expit(-0.25+0.45*c+0.30*g_prog-0.20*p_prog+0.25*g2*p2)\n\n    g_inc=_residualized_pc_score(T,C,tr,int(seed)+7101,5)\n    p_inc=_residualized_pc_score(R,C,tr,int(seed)+8209,5)\n    j_inc=_joint_score(g_inc,p_inc,C,tr)\n\n    base=0.10+0.12*c\n    base_sd=float(np.std(base[tr]))\n    alpha=float(incremental_ratio)*base_sd\n    source=np.zeros(len(c),dtype=np.int8)\n    inc=np.zeros(len(c),dtype=np.float32)\n\n    if regime=="TRANSCRIPTOME_INCREMENTAL":\n        inc=alpha*g_inc; source[:]=1\n    elif regime=="RPPA_INCREMENTAL":\n        inc=alpha*p_inc; source[:]=2\n    elif regime=="JOINT_MOLECULAR_SYNERGY":\n        inc=alpha*j_inc; source[:]=3\n    elif regime=="PATIENT_SUBGROUP_SWITCHING":\n        cut=float(np.median(c[tr])); upper=c>=cut\n        inc=alpha*np.where(upper,g_inc,p_inc).astype(np.float32)\n        source=np.where(upper,1,2).astype(np.int8)\n\n    tau_raw=np.clip(base+inc,-0.35,0.45)\n    mu1=np.clip(mu0+tau_raw,0.02,0.98)\n    tau=(mu1-mu0).astype(np.float32)\n\n    rng=np.random.default_rng(int(seed))\n    t=rng.binomial(1,e).astype(np.int64)\n    py=np.where(t==1,mu1,mu0)\n    y=rng.binomial(1,py).astype(np.int64)\n    return {\n        "t":t,"y":y,"e_true":e.astype(np.float32),"mu0_true":mu0.astype(np.float32),\n        "mu1_true":mu1.astype(np.float32),"tau_true":tau,\n        "incremental_source":source,"incremental_signal":inc.astype(np.float32),\n        "g_incremental":g_inc,"p_incremental":p_inc,"joint_incremental":j_inc,\n        "incremental_alpha":np.float32(alpha),\n    }\n',_stage7.__dict__)

from cm_osr.baselines import TLearner,XLearner
from cm_osr.metrics import root_pehe,policy_regret
from cm_osr.nuisance import crossfit_dr
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
from cm_osr.stage7_dgp import generate_stage7
from cm_osr.utils import concat_modalities

DGP_SEEDS=[3671, 3738, 3805, 3872, 3939, 4006, 4073, 4140, 4207, 4274]
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
                  'data_sha256':data_hash,'code_sha256':code_hash,'kaggle_execution_code_sha256':sha(__file__),
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
    data=SRC/'02_DATA/processed/ispy2_core_multimodal.npz';data_hash=sha(data);code_hash='928e7bb4c95cd6a77640ea9ad9c43f2d691438bcdcd6f8590e43312e1d775dbc'
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

import zipfile
with zipfile.ZipFile(REV/'B01_P1_REPLICATION_BATCH_OUTPUT.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in (REV/'results/replication').rglob('*'):
        if path.is_file():archive.write(path,path.relative_to(REV).as_posix())
