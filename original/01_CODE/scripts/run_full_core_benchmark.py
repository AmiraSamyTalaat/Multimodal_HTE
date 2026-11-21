from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml, torch
CODE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(CODE))
from cm_osr.splits import locked_split
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.dgp import generate
from cm_osr.nuisance import crossfit_dr, FullNuisance
from cm_osr.model import CMOSR
from cm_osr.baselines import TLearner, XLearner, DRLearner, RLearner
from cm_osr.deep_baselines import (fit_outcome_net,predict_outcome_tau,fit_transformer_outcome,predict_transformer_tau,
    fit_phi_mlp,predict_phi_mlp,fit_multimodal_phi,predict_multimodal_phi,count_params,find_hidden_for_target,
    find_attention_dim_for_target,find_moe_hidden_for_target)
from cm_osr.metrics import root_pehe,policy_regret,cate_calibration_error,selective_curve
from cm_osr.utils import concat_modalities,seed_everything,write_json
from cm_osr.runio import init_run,finish_run


def evalrow(name,p,true): return {'method':name,'root_pehe':root_pehe(p,true),'policy_regret':policy_regret(p,true),'cate_calibration_error':cate_calibration_error(p,true)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--data',required=True); ap.add_argument('--dgp-seed',type=int,required=True); ap.add_argument('--net-seed',type=int,required=True); ap.add_argument('--epochs',type=int,default=None); a=ap.parse_args()
    root=Path(a.root); cfg=yaml.safe_load((root/'00_CONFIG'/'project.yaml').read_text(encoding='utf-8')); epochs=int(a.epochs or cfg['training']['max_epochs']); patience=int(cfg['training']['patience']); tol=float(cfg['capacity_tolerance_percent'])/100
    z=np.load(a.data,allow_pickle=False); raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}; sp=locked_split(len(z['clinical'])); tr,te=sp['train'],sp['test']
    prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}); mods=prep.transform(raw); d=generate(mods,a.dgp_seed,cfg['primary_regime'],reference_idx=tr); mtr={k:v[tr] for k,v in mods.items()}; mte={k:v[te] for k,v in mods.items()}; seed_everything(a.net_seed)
    rd,man=init_run(root,f"CORE_DGP{a.dgp_seed}_NET{a.net_seed}_{int(time.time())}",vars(a)|{'epochs_effective':epochs,'device':('cuda' if torch.cuda.is_available() else 'cpu')})
    oof=crossfit_dr(mtr,d['t'][tr],d['y'][tr],cfg['nuisance_folds'],a.net_seed); nuis=FullNuisance(a.net_seed).fit(mtr,d['t'][tr],d['y'][tr]); nte=nuis.predict(mte); Xtr=concat_modalities(mtr); Xte=concat_modalities(mte); true=d['tau_true'][te]; preds={}; param={}
    cm=CMOSR(hidden=64,folds=cfg['effect_folds'],seed=a.net_seed,expert_epochs=epochs,expert_patience=patience).fit(mtr,oof['phi'],oof['e_hat']); cpred=cm.predict(mte,nte['e_hat']); preds['CM_OSR']=cpred['tau_hat']; param['CM_OSR']=sum(e['param_count'] for e in cm.experts)+sum(p.numel() for p in cm.router.parameters())
    preds['T_learner_regularized']=TLearner().fit(Xtr,d['t'][tr],d['y'][tr]).predict(Xte); preds['X_learner']=XLearner().fit(Xtr,d['t'][tr],d['y'][tr]).predict(Xte); preds['DR_learner']=DRLearner().fit(Xtr,oof['phi']).predict(Xte); preds['R_learner']=RLearner().fit(Xtr,d['t'][tr],d['y'][tr]).predict(Xte)
    for kind,label in [('tarnet','TARNet'),('cfrnet','CFRNet'),('dragonnet','DragonNet')]:
        m,sc=fit_outcome_net(kind,Xtr,d['t'][tr],d['y'][tr],hidden=128,seed=a.net_seed,epochs=epochs); preds[label]=predict_outcome_tau(m,sc,Xte,kind); param[label]=count_params(m)
    tm=fit_transformer_outcome(mtr,d['t'][tr],d['y'][tr],a.net_seed,epochs); preds['Transformer_HTE']=predict_transformer_tau(tm,mte); param['Transformer_HTE']=count_params(tm)
    h,npar=find_hidden_for_target(Xtr.shape[1],param['CM_OSR'],tol); em,esc=fit_phi_mlp(Xtr,oof['phi'],h,a.net_seed,epochs); preds['early_fusion_DR_capacity_matched']=predict_phi_mlp(em,esc,Xte); param['early_fusion_DR_capacity_matched']=count_params(em)
    dims={k:v.shape[1] for k,v in mtr.items()}; adim,anpar,aerr=find_attention_dim_for_target(dims,param['CM_OSR'],tol); mdim,mnpar,merr=find_moe_hidden_for_target(dims,param['CM_OSR'],tol)
    am=fit_multimodal_phi('attention',mtr,oof['phi'],a.net_seed,epochs,hidden_or_dim=adim); preds['generic_multimodal_attention_DR_capacity_matched']=predict_multimodal_phi(am,mte); param['generic_multimodal_attention_DR_capacity_matched']=count_params(am)
    gm=fit_multimodal_phi('moe',mtr,oof['phi'],a.net_seed,epochs,hidden_or_dim=mdim); preds['generic_MoE_DR_capacity_matched']=predict_multimodal_phi(gm,mte); param['generic_MoE_DR_capacity_matched']=count_params(gm)
    cap={k:{'params':int(v),'relative_error_vs_CM_OSR':float(abs(v-param['CM_OSR'])/max(param['CM_OSR'],1))} for k,v in param.items() if 'capacity_matched' in k}; cap['attention_d_model']=int(adim); cap['moe_hidden']=int(mdim); cap['tolerance']=tol
    if any(v['relative_error_vs_CM_OSR']>tol+1e-12 for k,v in cap.items() if isinstance(v,dict)): raise RuntimeError(f'Capacity matching failed: {cap}')
    preds['joint_only_CM_OSR_ablation']=cpred['expert_pred'][:,-1]; preds['uniform_all_experts_ablation']=cpred['expert_pred'].mean(1); preds['modality_only_uniform_fusion_ablation']=cpred['expert_pred'][:,:3].mean(1)
    rows=[evalrow(k,v,true) for k,v in preds.items()]; pd.DataFrame(rows).to_csv(rd/'metrics'/'summary.csv',index=False); write_json(rd/'metrics'/'parameter_counts.json',param); write_json(rd/'metrics'/'capacity_fairness.json',cap)
    np.savez_compressed(rd/'predictions'/'test_predictions.npz',tau_true=true,support=cpred['support'],**preds); write_json(rd/'metrics'/'CM_OSR_selective.json',selective_curve(preds['CM_OSR'],true,cpred['support'])); finish_run(rd,man)
    print('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'); print(pd.DataFrame(rows).sort_values('root_pehe').to_string(index=False)); print('PARAMETERS',param); print('CAPACITY',cap)
if __name__=='__main__': main()
