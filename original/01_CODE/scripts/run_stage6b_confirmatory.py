from __future__ import annotations
import argparse, math, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, yaml, torch
from scipy.special import expit

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))

from cm_osr.splits import locked_split
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.dgp import generate, _proj
from cm_osr.nuisance import crossfit_dr, FullNuisance
from cm_osr.model import CMOSR
from cm_osr.baselines import TLearner, XLearner, DRLearner, RLearner
from cm_osr.deep_baselines import (
    fit_phi_mlp,predict_phi_mlp,fit_multimodal_phi,predict_multimodal_phi,
    count_params,find_hidden_for_target,find_attention_dim_for_target,find_moe_hidden_for_target
)
from cm_osr.metrics import root_pehe, policy_regret, cate_calibration_error, selective_curve
from cm_osr.utils import concat_modalities, seed_everything, write_json
from cm_osr.q1_candidates import AnchoredOrthogonalSupportRouter

METHODS=[
    'AOSR_X_gamma2','CM_OSR_v1_frozen_reference','CMOSR_support_shrink_gamma2_predeclared_control',
    'X_learner_frozen_reference','T_learner_frozen_reference','DR_learner_frozen_reference',
    'R_learner_frozen_reference','early_fusion_DR_capacity_matched',
    'generic_multimodal_attention_DR_capacity_matched','generic_MoE_DR_capacity_matched'
]

def evalrow(condition,name,p,true,dgp_seed,net_seed,extra=None):
    r={'dgp_seed':int(dgp_seed),'net_seed':int(net_seed),'condition':condition,'method':name,
       'root_pehe':root_pehe(p,true),'policy_regret':policy_regret(p,true),
       'cate_calibration_error':cate_calibration_error(p,true),'mean_tau_hat':float(np.mean(p)),
       'mean_abs_tau_hat':float(np.mean(np.abs(p)))}
    if extra:r.update(extra)
    return r

def conformal90(dev_pred,dev_true,test_pred,test_true):
    err=np.abs(np.asarray(dev_pred)-np.asarray(dev_true)); n=len(err)
    k=min(max(int(math.ceil((n+1)*0.90)),1),n)
    q=float(np.partition(err,k-1)[k-1]); lo=np.asarray(test_pred)-q; hi=np.asarray(test_pred)+q
    return {'nominal':0.90,'q_abs_error':q,'coverage':float(np.mean((test_true>=lo)&(test_true<=hi))),
            'mean_width':float(2*q),'n_dev':int(n),'n_test':int(len(test_true))}

def fit_pack(mtr,ttr,ytr,net_seed,cfg):
    seed_everything(net_seed)
    folds=int(cfg['nuisance_folds']); efolds=int(cfg['effect_folds'])
    epochs=int(cfg['training']['max_epochs']); patience=int(cfg['training']['patience'])
    tol=float(cfg['capacity_tolerance_percent'])/100
    oof=crossfit_dr(mtr,ttr,ytr,folds,net_seed)
    nuis=FullNuisance(net_seed).fit(mtr,ttr,ytr); Xtr=concat_modalities(mtr)
    cm=CMOSR(hidden=64,folds=efolds,seed=net_seed,expert_epochs=epochs,expert_patience=patience).fit(mtr,oof['phi'],oof['e_hat'])
    cm_params=sum(e['param_count'] for e in cm.experts)+sum(p.numel() for p in cm.router.parameters())
    aosr=AnchoredOrthogonalSupportRouter(anchor_kind='X',gamma=2.0,hidden=64,folds=efolds,seed=net_seed,
                                         expert_epochs=epochs,expert_patience=patience).fit(mtr,ttr,ytr,oof['phi'],oof['e_hat'])
    x=XLearner().fit(Xtr,ttr,ytr); t=TLearner().fit(Xtr,ttr,ytr)
    dr=DRLearner().fit(Xtr,oof['phi']); r=RLearner().fit(Xtr,ttr,ytr)
    h,_=find_hidden_for_target(Xtr.shape[1],cm_params,tol); early,early_sc=fit_phi_mlp(Xtr,oof['phi'],h,net_seed,epochs)
    dims={k:v.shape[1] for k,v in mtr.items()}
    adim,_,_=find_attention_dim_for_target(dims,cm_params,tol); mhidden,_,_=find_moe_hidden_for_target(dims,cm_params,tol)
    att=fit_multimodal_phi('attention',mtr,oof['phi'],net_seed,epochs,hidden_or_dim=adim)
    moe=fit_multimodal_phi('moe',mtr,oof['phi'],net_seed,epochs,hidden_or_dim=mhidden)
    cap={'CM_OSR_v1_frozen_reference':int(cm_params),'early_fusion_DR_capacity_matched':int(count_params(early)),
         'generic_multimodal_attention_DR_capacity_matched':int(count_params(att)),
         'generic_MoE_DR_capacity_matched':int(count_params(moe)),'capacity_tolerance':float(tol)}
    for k in ['early_fusion_DR_capacity_matched','generic_multimodal_attention_DR_capacity_matched','generic_MoE_DR_capacity_matched']:
        if abs(cap[k]-cm_params)/max(cm_params,1)>tol+1e-12:raise RuntimeError(f'Capacity fairness failed: {cap}')
    return {'nuis':nuis,'cm':cm,'aosr':aosr,'x':x,'t':t,'dr':dr,'r':r,'early':early,'early_sc':early_sc,'att':att,'moe':moe,'capacity':cap}

def predict_pack(pack,mods,availability=None):
    X=concat_modalities(mods); e=pack['nuis'].predict(mods)['e_hat']
    cm=pack['cm'].predict(mods,e,availability=availability); ao=pack['aosr'].predict(mods,e,availability=availability)
    s=np.clip(np.asarray(cm['support'],dtype=np.float32),0,1)
    p={'AOSR_X_gamma2':np.asarray(ao['tau_hat'],dtype=np.float32),
       'CM_OSR_v1_frozen_reference':np.asarray(cm['tau_hat'],dtype=np.float32),
       'CMOSR_support_shrink_gamma2_predeclared_control':(s**2)*np.asarray(cm['tau_hat'],dtype=np.float32),
       'X_learner_frozen_reference':np.asarray(pack['x'].predict(X),dtype=np.float32),
       'T_learner_frozen_reference':np.asarray(pack['t'].predict(X),dtype=np.float32),
       'DR_learner_frozen_reference':np.asarray(pack['dr'].predict(X),dtype=np.float32),
       'R_learner_frozen_reference':np.asarray(pack['r'].predict(X),dtype=np.float32),
       'early_fusion_DR_capacity_matched':np.asarray(predict_phi_mlp(pack['early'],pack['early_sc'],X),dtype=np.float32),
       'generic_multimodal_attention_DR_capacity_matched':np.asarray(predict_multimodal_phi(pack['att'],mods),dtype=np.float32),
       'generic_MoE_DR_capacity_matched':np.asarray(predict_multimodal_phi(pack['moe'],mods),dtype=np.float32)}
    return p,{'aosr_support':np.asarray(ao['support'],dtype=np.float32),
              'aosr_routed_correction':np.asarray(ao['routed_correction'],dtype=np.float32)}

def make_missing(mods,idx,c_full,dgp,cond,split):
    sub={k:np.array(v[idx],copy=True) for k,v in mods.items()}; n=len(idx); av=np.ones((n,4),dtype=np.float32)
    code={'MCAR_40':40,'MAR_CLINICAL':77}[cond]; sc=1 if split=='dev' else 2
    rng=np.random.default_rng(860000+int(dgp)*100+code*10+sc)
    probs=np.full(n,0.40) if cond=='MCAR_40' else np.clip(0.10+0.25*expit(np.asarray(c_full)[idx]),0,0.95)
    for j,name in [(1,'transcriptomic'),(2,'rppa')]:
        miss=rng.random(n)<probs; av[:,j]=(~miss).astype(np.float32); sub[name][miss,:]=0.0
    av[:,0]=1.0; av[:,3]=(av[:,0]*av[:,1]*av[:,2]).astype(np.float32)
    return sub,av,{'dgp_seed':int(dgp),'condition':cond,'split':split,
                   'transcriptomic_missing_fraction':float(1-av[:,1].mean()),
                   'rppa_missing_fraction':float(1-av[:,2].mean()),
                   'both_molecular_missing_fraction':float(np.mean((av[:,1]==0)&(av[:,2]==0))),
                   'mean_target_probability':float(np.mean(probs))}

def append_condition(rows,selective,coverage,cond,pdev,pte,true_dev,true_te,diag,dgp,net):
    for m,p in pte.items():
        extra={}
        if m=='AOSR_X_gamma2':
            extra={'mean_abs_routed_correction':float(np.mean(np.abs(diag['aosr_routed_correction']))),
                   'mean_support':float(np.mean(diag['aosr_support']))}
        rows.append(evalrow(cond,m,p,true_te,dgp,net,extra))
    coverage.append({'dgp_seed':dgp,'net_seed':net,'condition':cond,'method':'AOSR_X_gamma2',
                     **conformal90(pdev['AOSR_X_gamma2'],true_dev,pte['AOSR_X_gamma2'],true_te)})
    for rr in selective_curve(pte['AOSR_X_gamma2'],true_te,diag['aosr_support']):
        selective.append({'dgp_seed':dgp,'net_seed':net,'condition':cond,'method':'AOSR_X_gamma2',**rr})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--data',required=True)
    ap.add_argument('--dgp-seed',type=int,required=True); ap.add_argument('--net-seed',type=int,required=True); a=ap.parse_args()
    root=Path(a.root); lock=yaml.safe_load((root/'00_CONFIG'/'stage6b_confirmatory_lock.yaml').read_text())
    cfg=yaml.safe_load((root/'00_CONFIG'/'project.yaml').read_text()); sel=json.loads((root/'06_RESULTS'/'STAGE6A'/'STAGE6A_SELECTION.json').read_text())
    if lock['status']!='LOCKED_BEFORE_STAGE6B_RESULTS' or sel.get('selected_candidate')!='AOSR_X_gamma2' or sel.get('status')!='CANDIDATE_FROZEN_FOR_STAGE6B':raise SystemExit('Frozen selection/lock mismatch.')
    if a.dgp_seed not in lock['confirmatory_dgp_seeds'] or a.net_seed not in lock['confirmatory_network_seeds']:raise SystemExit('Seed outside Stage6B lock.')
    z=np.load(a.data,allow_pickle=False); raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}; sp=locked_split(len(z['clinical']))
    tr,dev,te=sp['train'],sp['dev'],sp['test']; prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}); mods=prep.transform(raw)
    mtr={k:v[tr] for k,v in mods.items()}; mdev={k:v[dev] for k,v in mods.items()}; mte={k:v[te] for k,v in mods.items()}
    rd=root/'08_CONFIRMATORY_RUNS'/f"CONF6B_DGP{a.dgp_seed}_NET{a.net_seed}_{int(time.time())}"; (rd/'metrics').mkdir(parents=True,exist_ok=True)
    manifest={'stage':'STAGE6B_CONFIRMATORY','dgp_seed':a.dgp_seed,'net_seed':a.net_seed,'selected_candidate':'AOSR_X_gamma2',
              'training_split':'TRAIN_ONLY','evaluation_split':'TEST_ONCE','candidate_retuned':False,'test_used_for_selection':False,
              'stage3_primary_rerun':False,'stage5a_rerun':False,'stage6a_rerun':False,'device':'cuda' if torch.cuda.is_available() else 'cpu','status':'RUNNING','started':time.time()}
    write_json(rd/'RUN_MANIFEST.json',manifest); rows=[]; selective=[]; coverage=[]; missing=[]; caps={}
    d=generate(mods,a.dgp_seed,'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr); pack=fit_pack(mtr,d['t'][tr],d['y'][tr],a.net_seed,cfg); caps['primary_and_missing']=pack['capacity']
    pdv,_=predict_pack(pack,mdev); pte,diag=predict_pack(pack,mte); append_condition(rows,selective,coverage,'PRIMARY_CROSS_MODAL',pdv,pte,d['tau_true'][dev],d['tau_true'][te],diag,a.dgp_seed,a.net_seed)
    c_full=_proj(mods['clinical'],1101,reference_idx=tr)
    for cond in ['MCAR_40','MAR_CLINICAL']:
        md,avd,rdv=make_missing(mods,dev,c_full,a.dgp_seed,cond,'dev'); mt,avt,rte=make_missing(mods,te,c_full,a.dgp_seed,cond,'test'); missing += [dict(rdv,net_seed=a.net_seed),dict(rte,net_seed=a.net_seed)]
        pdv,_=predict_pack(pack,md,avd); pte,diag=predict_pack(pack,mt,avt); append_condition(rows,selective,coverage,cond,pdv,pte,d['tau_true'][dev],d['tau_true'][te],diag,a.dgp_seed,a.net_seed)
    ds=generate(mods,a.dgp_seed,'severe_weak_overlap_stress_only',reference_idx=tr); ps=fit_pack(mtr,ds['t'][tr],ds['y'][tr],a.net_seed,cfg); caps['severe']=ps['capacity']
    pdv,_=predict_pack(ps,mdev); pte,diag=predict_pack(ps,mte); append_condition(rows,selective,coverage,'SEVERE_WEAK_OVERLAP',pdv,pte,ds['tau_true'][dev],ds['tau_true'][te],diag,a.dgp_seed,a.net_seed)
    dn=generate(mods,a.dgp_seed,'null_treatment_effect_negative_control',reference_idx=tr); pn=fit_pack(mtr,dn['t'][tr],dn['y'][tr],a.net_seed,cfg); caps['null']=pn['capacity']
    pdv,_=predict_pack(pn,mdev); pte,diag=predict_pack(pn,mte); append_condition(rows,selective,coverage,'NULL_EFFECT_NEGATIVE_CONTROL',pdv,pte,dn['tau_true'][dev],dn['tau_true'][te],diag,a.dgp_seed,a.net_seed)
    df=pd.DataFrame(rows)
    if set(df.method)!=set(METHODS):raise RuntimeError('Method set mismatch.')
    if df[['root_pehe','mean_abs_tau_hat']].isna().any().any():raise RuntimeError('NaN in confirmatory metrics.')
    df.to_csv(rd/'metrics'/'summary_test.csv',index=False); pd.DataFrame(selective).to_csv(rd/'metrics'/'selective_test.csv',index=False); pd.DataFrame(coverage).to_csv(rd/'metrics'/'coverage90_test.csv',index=False); pd.DataFrame(missing).to_csv(rd/'metrics'/'missingness_realized.csv',index=False); write_json(rd/'metrics'/'capacity_fairness.json',caps)
    manifest['status']='COMPLETE'; manifest['finished']=time.time(); write_json(rd/'RUN_MANIFEST.json',manifest); write_json(rd/'COMPLETE.json',{'stage':'STAGE6B_CONFIRMATORY','dgp_seed':a.dgp_seed,'net_seed':a.net_seed,'status':'COMPLETE','selected_candidate':'AOSR_X_gamma2','test_used_once':True,'candidate_retuned':False})
    print('STAGE6B CONFIRMATORY PAIR COMPLETE',a.dgp_seed,a.net_seed,'rows',len(df),'device',manifest['device'])

if __name__=='__main__':main()
