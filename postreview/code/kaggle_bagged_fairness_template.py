from __future__ import annotations
import copy, hashlib, json, platform, sys, time, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy, sklearn, torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CONFIG = __BATCH_CONFIG__
ROOT=Path('/kaggle/working')
OUT=ROOT/'postreview_results';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/kaggle/input/b01-causaltx-mm-core-runtime/B01_CORE_BUNDLE')
if not SOURCE.exists():
    matches=list(Path('/kaggle/input').rglob('ispy2_core_multimodal.npz'))
    if len(matches)!=1:raise RuntimeError(f'Expected one original processed dataset; found {matches}')
    SOURCE=matches[0].parents[2]
sys.path.insert(0,str(SOURCE/'01_CODE'))
from cm_osr.deep_baselines import EarlyFusionDRNet, count_params
from cm_osr.dgp import generate
from cm_osr.metrics import root_pehe,policy_regret
from cm_osr.nuisance import crossfit_dr
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
from cm_osr.utils import concat_modalities

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def train_model(X,y,hidden,seed,max_epochs,patience,run_dir,tag):
    torch.manual_seed(int(seed))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler=StandardScaler().fit(X)
    Z=scaler.transform(X).astype('float32')
    model=EarlyFusionDRNet(Z.shape[1],hidden=hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loader=DataLoader(TensorDataset(torch.from_numpy(Z),torch.from_numpy(np.asarray(y,dtype='float32'))),
                      batch_size=min(64,len(y)),shuffle=True,generator=torch.Generator().manual_seed(seed))
    zx=torch.from_numpy(Z).to(device);zy=torch.from_numpy(np.asarray(y,dtype='float32')).to(device)
    best=float('inf');best_state=None;bad=0;history=[]
    for epoch in range(1,max_epochs+1):
        model.train()
        for xb,yb in loader:
            xb=xb.to(device);yb=yb.to(device)
            opt.zero_grad();loss=((model(xb)-yb)**2).mean();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():train_loss=float(((model(zx)-zy)**2).mean().detach().cpu())
        history.append({'tag':tag,'epoch':epoch,'train_mse':train_loss})
        if train_loss<best-1e-6:
            best=train_loss;best_state=copy.deepcopy(model.state_dict());bad=0
        else:
            bad+=1
            if bad>=patience:break
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(run_dir/f'{tag}_training_history.csv',index=False)
    return model,scaler,{'tag':tag,'epochs_run':len(history),'best_train_mse':best,'params':count_params(model),'n_fit':len(y)}
def predict(model,scaler,X):
    dev=next(model.parameters()).device
    with torch.no_grad():
        x=torch.from_numpy(scaler.transform(X).astype('float32')).to(dev)
        return model(x).detach().cpu().numpy()
def pick_hidden(d,target_total,n_models):
    candidates=[(abs(count_params(EarlyFusionDRNet(d,h))*n_models-target_total),h,count_params(EarlyFusionDRNet(d,h))*n_models) for h in range(8,256)]
    return min(candidates)
def main():
    started=datetime.now(timezone.utc).isoformat()
    data_file=SOURCE/'02_DATA/processed/ispy2_core_multimodal.npz'
    with np.load(data_file,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    split=locked_split(len(raw['clinical']));tr,te=split['train'],split['test']
    prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()})
    mods=prep.transform(raw);X=concat_modalities(mods)
    records=[];failures=[]
    for dgp_seed in CONFIG['dgp_seeds']:
        d=generate(mods,int(dgp_seed),'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
        for net_seed in CONFIG['net_seeds']:
            tag=f'DGP{dgp_seed}_NET{net_seed}';rd=OUT/tag;rd.mkdir(exist_ok=True)
            if (rd/'COMPLETE.json').exists():
                records.extend(pd.read_csv(rd/'metrics.csv').to_dict('records'));continue
            try:
                oof=crossfit_dr({k:v[tr] for k,v in mods.items()},d['t'][tr],d['y'][tr],n_splits=5,seed=int(net_seed),clip=(.05,.95))
                phi=oof['phi'];target_total=126440
                _,hidden,total=pick_hidden(X.shape[1],target_total,5)
                kf=KFold(n_splits=5,shuffle=True,random_state=int(net_seed)+44)
                fold_preds=[];fits=[]
                for fold,(fit_idx,hold_idx) in enumerate(kf.split(X[tr])):
                    model,scaler,fit=train_model(X[tr][fit_idx],phi[fit_idx],hidden,int(net_seed)+fold,
                            CONFIG['max_epochs'],CONFIG['patience'],rd,f'bag_fold{fold}')
                    fold_preds.append(predict(model,scaler,X[te]));fits.append(fit)
                bag=np.mean(np.stack(fold_preds),axis=0)
                _,single_hidden,_=pick_hidden(X.shape[1],target_total,1)
                single,scaler,single_fit=train_model(X[tr],phi,single_hidden,int(net_seed)+999,
                    CONFIG['max_epochs'],CONFIG['patience'],rd,'single_early_stopped')
                single_pred=predict(single,scaler,X[te])
                truth=d['tau_true'][te]
                np.savez_compressed(rd/'test_predictions.npz',tau_true=truth,bagged_early_fusion=bag,
                                    single_early_stopped_early_fusion=single_pred)
                metrics=[]
                for method,pred in [('bagged_early_fusion',bag),('single_early_stopped_early_fusion',single_pred)]:
                    metrics.append({'dgp_seed':dgp_seed,'net_seed':net_seed,'method':method,'root_pehe':root_pehe(pred,truth),
                            'policy_regret':policy_regret(pred,truth),'clipped_root_pehe':root_pehe(np.clip(pred,-1,1),truth),
                            'fraction_outside_minus1_plus1':float(np.mean(np.abs(pred)>1)),
                            'mode':CONFIG['mode'],'max_epochs':CONFIG['max_epochs'],'patience':CONFIG['patience']})
                pd.DataFrame(metrics).to_csv(rd/'metrics.csv',index=False)
                write(rd/'COMPLETE.json',{'status':'COMPLETE','mode':CONFIG['mode'],'dgp_seed':dgp_seed,
                    'net_seed':net_seed,'finished_utc':datetime.now(timezone.utc).isoformat(),
                    'data_sha256':sha(data_file),'dgp_code_sha256':sha(SOURCE/'01_CODE/cm_osr/dgp.py'),
                    'n_train':len(tr),'n_test':len(te),'nuisance_folds':5,'bagging_folds':5,
                    'bagged_params_total':total,'single_params':single_fit['params'],
                    'cm_osr_archived_params':target_total,'bagged_fit_details':fits,'single_fit_details':single_fit,
                    'early_stopping':'training MSE improvement >1e-6; same patience criterion as original CM-OSR experts',
                    'test_access':'metrics only after fitting and fixed design; no tuning'})
                records.extend(metrics)
            except Exception as exc:
                (rd/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
                failures.append({'pair':tag,'error':str(exc)})
            print(tag,'complete' if (rd/'COMPLETE.json').exists() else 'failed',flush=True)
    if records:pd.DataFrame(records).to_csv(OUT/'per_run_results.csv',index=False)
    status={'status':'COMPLETE' if not failures and len(records)==2*len(CONFIG['dgp_seeds'])*len(CONFIG['net_seeds']) else 'PARTIAL',
            'config':CONFIG,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
            'completed_pairs':len(records)//2,'failed_pairs':failures,'data_sha256':sha(data_file),
            'software':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                        'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__}}
    write(OUT/'RUN_STATUS.json',status)
    with zipfile.ZipFile(ROOT/'B01_P1_BAGGED_FAIRNESS_OUTPUT.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
    print(json.dumps(status),flush=True)
if __name__=='__main__':main()
