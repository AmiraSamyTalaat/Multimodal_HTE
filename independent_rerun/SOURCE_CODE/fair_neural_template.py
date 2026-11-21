from __future__ import annotations
import copy, hashlib, json, os, platform, sys, time, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, scipy, sklearn, torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset

CONFIG={'seeds': [9001, 9102], 'regimes': ['NO_INCREMENTAL_MOLECULAR_SIGNAL', 'JOINT_MOLECULAR_SYNERGY'], 'split_seed': 60601, 'net_seed': 17, 'max_epochs': 250, 'patience': 30, 'calibration_grid': [0, 0.25, 0.5, 1], 'source_hashes': {'01_CODE\\cm_osr\\stage7_dgp.py': '83494ac53d5594a0ed0bb23bd6be9ddc4cfbd38cec35887b2fae89ccfd47a84f', '01_CODE\\cm_osr\\model.py': '17a883c9daa96c95dc574a8d40577bb6f7555bffcd994cf77516da99fc604742', '01_CODE\\cm_osr\\experts.py': 'f51357c98b84b7522bc007811dbd108b722b9f4f6783cc7fc3e1fc75d278b06d', '01_CODE\\cm_osr\\nuisance.py': 'd579f16581d1d5eb400a6154d1b8ae90840318dff4139f2f9a0ee85db82d5f4c', '01_CODE\\cm_osr\\router.py': 'ceadb09d941bdfe474b0c64168624ee12afbeb6323462e167371a1b35b605469', '01_CODE\\cm_osr\\deep_baselines.py': '5372fdbb681a290c2481e3be7b37f5c748156231796d621cb4d8eefa816b56a5', '01_CODE\\cm_osr\\preprocess.py': '4432b72c63ca4a1237125b201263f3242b2d04c60c733d055ed6027319e75568'}, 'protocol_lock_sha256': '8226206fd60d134d7b86a5ed2fe0a4d32783099a925a85abbac8d90eccadb5e8', 'template_sha256': '9eb0d011f5603b5c2ca37e19d065a64b330e6ef9642daac9387d3ed6520b9213'}
CONFIG['seeds']=[int(x) for x in os.environ.get('B01_FAIR_SEEDS','9001,9102').split(',')]
ROOT=Path('/kaggle/working'); OUT=ROOT/'fair_neural_results';OUT.mkdir(exist_ok=True)
SOURCE=Path('/kaggle/input/b01-causaltx-mm-core-runtime/B01_CORE_BUNDLE')
if not SOURCE.exists():
    matches=list(Path('/kaggle/input').rglob('ispy2_core_multimodal.npz'))
    if len(matches)!=1:raise RuntimeError(f'Expected one private B01 dataset: {len(matches)} found')
    SOURCE=matches[0].parents[2]
sys.path.insert(0,str(SOURCE/'01_CODE'))
import types
stage7=types.ModuleType('cm_osr.stage7_dgp');stage7.__package__='cm_osr'
sys.modules['cm_osr.stage7_dgp']=stage7
exec('from __future__ import annotations\nimport numpy as np\nfrom scipy.special import expit\nfrom sklearn.decomposition import PCA\nfrom sklearn.linear_model import Ridge\n\nfrom .dgp import _proj\n\nREGIMES = [\n    "NO_INCREMENTAL_MOLECULAR_SIGNAL",\n    "TRANSCRIPTOME_INCREMENTAL",\n    "RPPA_INCREMENTAL",\n    "JOINT_MOLECULAR_SYNERGY",\n    "PATIENT_SUBGROUP_SWITCHING",\n]\n\n\ndef _std_train(z, tr):\n    z=np.asarray(z,dtype=float)\n    mu=float(np.mean(z[tr])); sd=float(np.std(z[tr]))\n    return ((z-mu)/(sd+1e-8)).astype(np.float32)\n\n\ndef _residualized_pc_score(X, C, tr, seed, n_components=5):\n    X=np.asarray(X,dtype=float); C=np.asarray(C,dtype=float); tr=np.asarray(tr,dtype=int)\n    k=max(1,min(int(n_components),X.shape[1],len(tr)-1))\n    pca=PCA(n_components=k,svd_solver="full").fit(X[tr])\n    pc=pca.transform(X)\n    rng=np.random.default_rng(int(seed))\n    w=rng.normal(size=k); w=w/(np.linalg.norm(w)+1e-12)\n    raw=pc@w\n    reg=Ridge(alpha=1.0).fit(C[tr],raw[tr])\n    resid=raw-reg.predict(C)\n    return _std_train(resid,tr)\n\n\ndef _joint_score(g,p,C,tr):\n    raw=np.asarray(g)*np.asarray(p)\n    Z=np.column_stack([C,np.asarray(g),np.asarray(p)])\n    reg=Ridge(alpha=1.0).fit(Z[tr],raw[tr])\n    resid=raw-reg.predict(Z)\n    return _std_train(resid,tr)\n\n\ndef generate_stage7(modalities, seed, regime, reference_idx, incremental_ratio=0.60, propensity_clip=(0.05,0.95)):\n    if regime not in REGIMES:\n        raise ValueError(f"Unknown Stage7 regime: {regime}")\n    tr=np.asarray(reference_idx,dtype=int)\n    C=np.asarray(modalities["clinical"],dtype=np.float32)\n    T=np.asarray(modalities["transcriptomic"],dtype=np.float32)\n    R=np.asarray(modalities["rppa"],dtype=np.float32)\n\n    c=_proj(C,1101,tr)\n    g_prog=_proj(T,2203,tr); p_prog=_proj(R,3307,tr)\n    g2=_proj(T,4409,tr); p2=_proj(R,5519,tr)\n\n    # Prognostic/assignment mechanisms remain multimodal, so molecular variables\n    # can be confounder/prognostic information even in the no-incremental-effect regime.\n    logit_e=0.25*c+0.45*g_prog+0.45*p_prog+0.65*g_prog*p_prog\n    e=np.clip(expit(logit_e),propensity_clip[0],propensity_clip[1])\n    mu0=expit(-0.25+0.45*c+0.30*g_prog-0.20*p_prog+0.25*g2*p2)\n\n    g_inc=_residualized_pc_score(T,C,tr,int(seed)+7101,5)\n    p_inc=_residualized_pc_score(R,C,tr,int(seed)+8209,5)\n    j_inc=_joint_score(g_inc,p_inc,C,tr)\n\n    base=0.10+0.12*c\n    base_sd=float(np.std(base[tr]))\n    alpha=float(incremental_ratio)*base_sd\n    source=np.zeros(len(c),dtype=np.int8)\n    inc=np.zeros(len(c),dtype=np.float32)\n\n    if regime=="TRANSCRIPTOME_INCREMENTAL":\n        inc=alpha*g_inc; source[:]=1\n    elif regime=="RPPA_INCREMENTAL":\n        inc=alpha*p_inc; source[:]=2\n    elif regime=="JOINT_MOLECULAR_SYNERGY":\n        inc=alpha*j_inc; source[:]=3\n    elif regime=="PATIENT_SUBGROUP_SWITCHING":\n        cut=float(np.median(c[tr])); upper=c>=cut\n        inc=alpha*np.where(upper,g_inc,p_inc).astype(np.float32)\n        source=np.where(upper,1,2).astype(np.int8)\n\n    tau_raw=np.clip(base+inc,-0.35,0.45)\n    mu1=np.clip(mu0+tau_raw,0.02,0.98)\n    tau=(mu1-mu0).astype(np.float32)\n\n    rng=np.random.default_rng(int(seed))\n    t=rng.binomial(1,e).astype(np.int64)\n    py=np.where(t==1,mu1,mu0)\n    y=rng.binomial(1,py).astype(np.int64)\n    return {\n        "t":t,"y":y,"e_true":e.astype(np.float32),"mu0_true":mu0.astype(np.float32),\n        "mu1_true":mu1.astype(np.float32),"tau_true":tau,\n        "incremental_source":source,"incremental_signal":inc.astype(np.float32),\n        "g_incremental":g_inc,"p_incremental":p_inc,"joint_incremental":j_inc,\n        "incremental_alpha":np.float32(alpha),\n    }\n',stage7.__dict__)
from cm_osr.stage7_dgp import generate_stage7
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
from cm_osr.utils import concat_modalities
from cm_osr.nuisance import crossfit_dr,FullNuisance
from cm_osr.model import CMOSR
from cm_osr.deep_baselines import EarlyFusionDRNet,count_params
from cm_osr.metrics import root_pehe,policy_regret
import cm_osr.experts as experts

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def sha_array(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def rloss(pred,t,y,n):
    e=n['e_hat'];m=e*n['mu1_hat']+(1-e)*n['mu0_hat']
    return float(np.mean(((y-m)-(t-e)*pred)**2))
def metric(method,pred,truth,seed,regime,mode,calibration):
    return dict(dgp_seed=seed,effect_regime=regime,split_seed=CONFIG['split_seed'],
      network_seed=CONFIG['net_seed'],method=method,calibration=calibration,mode=mode,
      root_pehe=root_pehe(pred,truth),policy_regret=policy_regret(pred,truth),
      clipped_root_pehe=root_pehe(np.clip(pred,-1,1),truth),
      fraction_outside_minus1_plus1=float(np.mean(np.abs(pred)>1)),n_test=len(truth))

# This replaces only the internal expert fit to log the actual stopping epoch;
# architecture, optimization, shuffling and stopping criterion match the original.
expert_fits=[]
def logged_expert_fit(X,y,seed=17,hidden=64,epochs=250,patience=30,lr=1e-3,batch_size=64):
    tic=time.monotonic();torch.manual_seed(seed);device=experts._device()
    scaler=StandardScaler().fit(X); Z=scaler.transform(X).astype('float32')
    model=experts.MLPRegressor(Z.shape[1],hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(torch.from_numpy(Z),torch.from_numpy(y.astype('float32'))),
        batch_size=min(batch_size,len(y)),shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=(float('inf'),None);bad=0;zx=torch.from_numpy(Z).to(device);zy=torch.from_numpy(y.astype('float32')).to(device)
    for epoch in range(1,epochs+1):
        model.train()
        for xb,yb in loader:
            xb=xb.to(device);yb=yb.to(device);opt.zero_grad();loss=torch.mean((model(xb)-yb)**2);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():cur=float(torch.mean((model(zx)-zy)**2).detach().cpu())
        if cur<best[0]-1e-6:best=(cur,copy.deepcopy(model.state_dict()));bad=0
        else:
            bad+=1
            if bad>=patience:break
    model.load_state_dict(best[1]);expert_fits.append(dict(seed=seed,n_train=len(y),params=experts.count_params(model),
      epochs_run=epoch,optimizer_steps=epoch*len(loader),best_train_mse=best[0],wall_seconds=time.monotonic()-tic))
    return model,scaler
experts._fit_one=logged_expert_fit

def bag_fit(X,y,hidden,seed,epoch_cap=None):
    tic=time.monotonic();torch.manual_seed(seed);dev=experts._device()
    scaler=StandardScaler().fit(X);Z=scaler.transform(X).astype('float32')
    model=EarlyFusionDRNet(Z.shape[1],hidden).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loader=DataLoader(TensorDataset(torch.from_numpy(Z),torch.from_numpy(y.astype('float32'))),
      batch_size=min(64,len(y)),shuffle=True,generator=torch.Generator().manual_seed(seed))
    zx=torch.from_numpy(Z).to(dev);zy=torch.from_numpy(y.astype('float32')).to(dev)
    best=(float('inf'),None);bad=0
    for epoch in range(1,(epoch_cap or CONFIG['max_epochs'])+1):
        model.train()
        for xb,yb in loader:
            xb=xb.to(dev);yb=yb.to(dev);opt.zero_grad();loss=((model(xb)-yb)**2).mean();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():cur=float(((model(zx)-zy)**2).mean().detach().cpu())
        if cur<best[0]-1e-6:best=(cur,copy.deepcopy(model.state_dict()));bad=0
        else:
            bad+=1
            if bad>=CONFIG['patience']:break
    model.load_state_dict(best[1])
    info=dict(seed=seed,n_train=len(y),params=count_params(model),epochs_run=epoch,
      optimizer_steps=epoch*len(loader),best_train_mse=best[0],wall_seconds=time.monotonic()-tic)
    return model,scaler,info
def bag_predict(model,scaler,X):
    dev=next(model.parameters()).device
    with torch.no_grad():return model(torch.from_numpy(scaler.transform(X).astype('float32')).to(dev)).detach().cpu().numpy()
def hidden_for_budget(d,target,n_models=5):
    return min(range(8,256),key=lambda h:abs(n_models*count_params(EarlyFusionDRNet(d,h))-target))
def one(raw,data_file,seed,regime):
    tag=f'DGP{seed}_{regime}';folder=OUT/tag;folder.mkdir(exist_ok=True)
    marker=folder/'COMPLETE.json'
    if marker.exists():return 'SKIPPED'
    anchor=locked_split(len(raw['clinical']),seed=20260918)
    ref=np.sort(np.r_[anchor['train'],anchor['dev']])
    tr,dv=train_test_split(ref,train_size=len(anchor['train']),random_state=CONFIG['split_seed'],shuffle=True)
    tr=np.sort(tr);dv=np.sort(dv);te=anchor['test']
    dgp_prep=ModalityPreprocessor().fit({k:v[ref] for k,v in raw.items()})
    dgp_mods=dgp_prep.transform(raw)
    d=generate_stage7(dgp_mods,int(seed),regime,reference_idx=ref)
    prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()})
    mods=prep.transform(raw);X=concat_modalities(mods)
    trainmods={k:v[tr] for k,v in mods.items()};t=d['t'][tr];y=d['y'][tr]
    tic=time.monotonic();oof=crossfit_dr(trainmods,t,y,n_splits=5,seed=17,clip=(.05,.95))
    nuisance=FullNuisance(seed=17,clip=(.05,.95)).fit(trainmods,t,y)
    ndv=nuisance.predict({k:v[dv] for k,v in mods.items()})
    ntest=nuisance.predict({k:v[te] for k,v in mods.items()})
    nuisance_seconds=time.monotonic()-tic
    expert_fits.clear();tic=time.monotonic()
    cm=CMOSR(hidden=64,folds=5,seed=CONFIG['net_seed'],expert_epochs=CONFIG['max_epochs'],
      expert_patience=CONFIG['patience']).fit(trainmods,oof['phi'],oof['e_hat'])
    cm_seconds=time.monotonic()-tic
    cmdev=cm.predict({k:v[dv] for k,v in mods.items()},ndv['e_hat'])
    cmtest=cm.predict({k:v[te] for k,v in mods.items()},ntest['e_hat'])
    deployed=sum(x['param_count'] for x in cm.experts)+sum(p.numel() for p in cm.router.parameters())
    hidden=hidden_for_budget(X.shape[1],deployed)
    kf=KFold(n_splits=5,shuffle=True,random_state=CONFIG['net_seed']+44)
    bpdev=[];bptest=[];bfits=[];tic=time.monotonic()
    for fold,(fitidx,_) in enumerate(kf.split(X[tr])):
        model,scaler,fit=bag_fit(X[tr][fitidx],oof['phi'][fitidx],hidden,CONFIG['net_seed']+fold)
        bpdev.append(bag_predict(model,scaler,X[dv]));bptest.append(bag_predict(model,scaler,X[te]));bfits.append(fit)
    bag_seconds=time.monotonic()-tic
    # Four independent repetitions of five-fold fits: 20 early fusion fits.
    # Each model sees a four-fifths TRAIN subset. No DEV CATE or TEST is used for fit selection.
    hidden20=hidden_for_budget(X.shape[1],deployed,20)
    bp20dev=[];bp20test=[];bfits20=[];tic=time.monotonic()
    for repeat in range(4):
        repeated=KFold(n_splits=5,shuffle=True,random_state=CONFIG['net_seed']+440+repeat)
        for fold,(fitidx,_) in enumerate(repeated.split(X[tr])):
            fitseed=CONFIG['net_seed']+100+repeat*5+fold
            model,scaler,fit=bag_fit(X[tr][fitidx],oof['phi'][fitidx],hidden20,fitseed,epoch_cap=150)
            bp20dev.append(bag_predict(model,scaler,X[dv]))
            bp20test.append(bag_predict(model,scaler,X[te]))
            bfits20.append(fit)
    bag20_seconds=time.monotonic()-tic
    pairs={
      'CM_OSR':(cmdev['tau_hat'],cmtest['tau_hat']),
      'uniform_same_experts':(cmdev['expert_pred'].mean(1),cmtest['expert_pred'].mean(1)),
      'bagged_early_fusion_5fold_capacity_matched':(np.mean(bpdev,axis=0),np.mean(bptest,axis=0)),
      'bagged_early_fusion_20fit_budget_targeted':(np.mean(bp20dev,axis=0),np.mean(bp20test,axis=0))}
    rows=[];grid=[]
    for name,(pdev,ptest) in pairs.items():
        rows.append(metric(name,ptest,d['tau_true'][te],seed,regime,'neural','raw'))
        for scale in CONFIG['calibration_grid']:
            grid.append(dict(dgp_seed=seed,effect_regime=regime,method=name,scale=scale,
              dev_observed_r_loss=rloss(scale*pdev,d['t'][dv],d['y'][dv],ndv)))
        best=min((q for q in grid if q['method']==name),key=lambda q:(q['dev_observed_r_loss'],q['scale']))
        row=metric(name,best['scale']*ptest,d['tau_true'][te],seed,regime,'neural','DEV_Rloss_selected')
        row['selected_scale']=best['scale'];row['dev_observed_r_loss']=best['dev_observed_r_loss'];rows.append(row)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(grid).to_csv(folder/'dev_calibration_grid.csv',index=False)
    write(marker,dict(status='COMPLETE',finished_utc=now(),dgp_seed=seed,effect_regime=regime,
      split_seed=CONFIG['split_seed'],network_seed=CONFIG['net_seed'],data_sha256=sha(data_file),
      source_code_sha256=CONFIG['source_hashes'],dgp_cate_sha256=sha_array(d['tau_true']),
      test_cate_sha256=sha_array(d['tau_true'][te]),test_indices_sha256=sha_array(te),
      test_cate_mean=float(np.mean(d['tau_true'][te])),test_cate_sd=float(np.std(d['tau_true'][te])),
      zero_effect_root_pehe=root_pehe(np.zeros(len(te)),d['tau_true'][te]),
      n_train=len(tr),n_dev=len(dv),n_test=len(te),nuisance_folds=5,
      nuisance_seconds=nuisance_seconds,cm_seconds=cm_seconds,bag_seconds=bag_seconds,bag20_seconds=bag20_seconds,
      cm_deployed_params=deployed,bag_deployed_params=5*count_params(EarlyFusionDRNet(X.shape[1],hidden)),
      bag20_deployed_params=20*count_params(EarlyFusionDRNet(X.shape[1],hidden20)),
      cm_expert_fit_details=expert_fits.copy(),bag_fit_details=bfits,bag20_fit_details=bfits20,
      cm_expert_fit_count=len(expert_fits),bag_fit_count=len(bfits),bag20_fit_count=len(bfits20),
      cm_expert_optimizer_steps=sum(x['optimizer_steps'] for x in expert_fits),
      bag20_optimizer_steps=sum(x['optimizer_steps'] for x in bfits20),
      cm_router_epochs=400,cm_router_optimizer_steps=400,
      calibration='Observed DEV R-loss; no true CATE or TEST used for selection',
      caveat='Same nuisance and early stopping rule; 20 bag fits capped at 150 epochs target CM optimizer and parameter budgets. Actual budgets must be checked. Uniform uses identical experts to CM; routing effect limited to that contrast.'))
    return 'COMPLETE'
def main():
    data_file=SOURCE/'02_DATA/processed/ispy2_core_multimodal.npz'
    with np.load(data_file,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    failures=[]
    for seed in CONFIG['seeds']:
        for regime in CONFIG['regimes']:
            tag=f'DGP{seed}_{regime}'
            try:print(tag,one(raw,data_file,seed,regime),flush=True)
            except Exception:
                (OUT/tag).mkdir(exist_ok=True)
                (OUT/tag/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
                failures.append(tag);print(tag,'ERROR',flush=True)
            files=list(OUT.glob('DGP*/COMPLETE.json'))
            write(OUT/'RUN_STATUS.json',dict(status='COMPLETE' if len(files)==len(CONFIG['seeds'])*len(CONFIG['regimes']) else 'PARTIAL',
              completed=len(files),expected=len(CONFIG['seeds'])*len(CONFIG['regimes']),failures=failures,updated_utc=now(),
              independence='DGP seed cluster; one network seed and one patient split'))
    allmetrics=[pd.read_csv(p) for p in OUT.glob('DGP*/metrics.csv')]
    if allmetrics:pd.concat(allmetrics,ignore_index=True).to_csv(OUT/'per_run_results.csv',index=False)
    with zipfile.ZipFile(ROOT/'B01_P1_FAIR_NEURAL_BATCH.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
    print('BATCH_ARCHIVED',len(allmetrics),len(failures),flush=True)
if __name__=='__main__':main()
