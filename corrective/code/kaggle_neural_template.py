from __future__ import annotations
import copy, hashlib, json, platform, sys, time, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, scipy, sklearn, torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset

CONFIG=__BATCH_CONFIG__
ROOT=Path('/kaggle/working'); OUT=ROOT/'corrective_neural_results';OUT.mkdir(exist_ok=True)
SOURCE=Path('/kaggle/input/b01-causaltx-mm-core-runtime/B01_CORE_BUNDLE')
if not SOURCE.exists():
    matches=list(Path('/kaggle/input').rglob('ispy2_core_multimodal.npz'))
    if len(matches)!=1:raise RuntimeError(f'Expected one private B01 dataset: {len(matches)} found')
    SOURCE=matches[0].parents[2]
sys.path.insert(0,str(SOURCE/'01_CODE'))
import types
stage7=types.ModuleType('cm_osr.stage7_dgp');stage7.__package__='cm_osr'
sys.modules['cm_osr.stage7_dgp']=stage7
exec(__STAGE7_CODE__,stage7.__dict__)
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

def bag_fit(X,y,hidden,seed):
    tic=time.monotonic();torch.manual_seed(seed);dev=experts._device()
    scaler=StandardScaler().fit(X);Z=scaler.transform(X).astype('float32')
    model=EarlyFusionDRNet(Z.shape[1],hidden).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loader=DataLoader(TensorDataset(torch.from_numpy(Z),torch.from_numpy(y.astype('float32'))),
      batch_size=min(64,len(y)),shuffle=True,generator=torch.Generator().manual_seed(seed))
    zx=torch.from_numpy(Z).to(dev);zy=torch.from_numpy(y.astype('float32')).to(dev)
    best=(float('inf'),None);bad=0
    for epoch in range(1,CONFIG['max_epochs']+1):
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
def hidden_for_budget(d,target):
    return min(range(8,256),key=lambda h:abs(5*count_params(EarlyFusionDRNet(d,h))-target))
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
    pairs={
      'CM_OSR':(cmdev['tau_hat'],cmtest['tau_hat']),
      'uniform_same_experts':(cmdev['expert_pred'].mean(1),cmtest['expert_pred'].mean(1)),
      'bagged_early_fusion_5fold_capacity_matched':(np.mean(bpdev,axis=0),np.mean(bptest,axis=0))}
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
      nuisance_seconds=nuisance_seconds,cm_seconds=cm_seconds,bag_seconds=bag_seconds,
      cm_deployed_params=deployed,bag_deployed_params=5*count_params(EarlyFusionDRNet(X.shape[1],hidden)),
      cm_expert_fit_details=expert_fits.copy(),bag_fit_details=bfits,
      cm_expert_fit_count=len(expert_fits),bag_fit_count=len(bfits),
      cm_router_epochs=400,cm_router_optimizer_steps=400,
      calibration='Observed DEV R-loss; no true CATE or TEST used for selection',
      caveat='Same nuisance and early stopping rule; unequal model fit budgets and training objectives; routing not causally isolated'))
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
    with zipfile.ZipFile(ROOT/'B01_P1_CORRECTIVE_NEURAL_BATCH.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
    print('BATCH_ARCHIVED',len(allmetrics),len(failures),flush=True)
if __name__=='__main__':main()
