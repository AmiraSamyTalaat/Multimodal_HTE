
from __future__ import annotations
import argparse, json, math, re, sys, time
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

STAGE5A_CODE_FIX_ID='MASKED_TRUE_TE_V1'

METHODS_BASE=[
    'CM_OSR','T_learner_regularized','X_learner','DR_learner','R_learner',
    'early_fusion_DR_capacity_matched',
    'generic_multimodal_attention_DR_capacity_matched',
    'generic_MoE_DR_capacity_matched'
]

def device_of(m):
    return next(m.parameters()).device

def evalrow(family,condition,name,p,true,extra=None):
    r={
        'family':family,'condition':condition,'method':name,
        'root_pehe':root_pehe(p,true),
        'policy_regret':policy_regret(p,true),
        'cate_calibration_error':cate_calibration_error(p,true),
        'mean_tau_hat':float(np.mean(p)),
        'mean_abs_tau_hat':float(np.mean(np.abs(p)))
    }
    if extra: r.update(extra)
    return r

def fit_pack(mtr,ttr,ytr,net_seed,epochs,patience,nuisance_folds,effect_folds,tol):
    seed_everything(net_seed)
    oof=crossfit_dr(mtr,ttr,ytr,nuisance_folds,net_seed)
    nuis=FullNuisance(net_seed).fit(mtr,ttr,ytr)
    Xtr=concat_modalities(mtr)

    cm=CMOSR(hidden=64,folds=effect_folds,seed=net_seed,expert_epochs=epochs,expert_patience=patience).fit(mtr,oof['phi'],oof['e_hat'])
    cm_params=sum(e['param_count'] for e in cm.experts)+sum(p.numel() for p in cm.router.parameters())

    tmdl=TLearner().fit(Xtr,ttr,ytr)
    xmdl=XLearner().fit(Xtr,ttr,ytr)
    drmdl=DRLearner().fit(Xtr,oof['phi'])
    rmdl=RLearner().fit(Xtr,ttr,ytr)

    h,_=find_hidden_for_target(Xtr.shape[1],cm_params,tol)
    early,early_sc=fit_phi_mlp(Xtr,oof['phi'],h,net_seed,epochs)
    early_params=count_params(early)

    dims={k:v.shape[1] for k,v in mtr.items()}
    adim,_,_=find_attention_dim_for_target(dims,cm_params,tol)
    mhidden,_,_=find_moe_hidden_for_target(dims,cm_params,tol)
    att=fit_multimodal_phi('attention',mtr,oof['phi'],net_seed,epochs,hidden_or_dim=adim)
    moe=fit_multimodal_phi('moe',mtr,oof['phi'],net_seed,epochs,hidden_or_dim=mhidden)
    att_params=count_params(att); moe_params=count_params(moe)

    cap={
        'CM_OSR':int(cm_params),
        'early_fusion_DR_capacity_matched':int(early_params),
        'generic_multimodal_attention_DR_capacity_matched':int(att_params),
        'generic_MoE_DR_capacity_matched':int(moe_params),
        'attention_d_model':int(adim),'moe_hidden':int(mhidden),'tolerance':float(tol)
    }
    for k in ['early_fusion_DR_capacity_matched','generic_multimodal_attention_DR_capacity_matched','generic_MoE_DR_capacity_matched']:
        if abs(cap[k]-cap['CM_OSR'])/max(cap['CM_OSR'],1)>tol+1e-12:
            raise RuntimeError(f'Capacity fairness failed in Stage5A: {cap}')

    return {
        'cm':cm,'nuis':nuis,'t':tmdl,'x':xmdl,'dr':drmdl,'r':rmdl,
        'early':early,'early_sc':early_sc,'att':att,'moe':moe,'capacity':cap
    }

def _mods_t(mods,dev):
    return {k:torch.from_numpy(np.asarray(v,dtype=np.float32)).to(dev) for k,v in mods.items()}

def predict_attention_masked(m,mods,av3):
    dev=device_of(m)
    mt=_mods_t(mods,dev)
    with torch.no_grad():
        e=torch.stack([m.enc[k](mt[k]) for k in m.names],1)
        logits=m.score(e).squeeze(-1)
        av=torch.from_numpy(av3.astype('float32')).to(dev)
        logits=logits.masked_fill(av<=0,-1e9)
        w=torch.softmax(logits,1)
        z=(w.unsqueeze(-1)*e).sum(1)
        return m.head(z).squeeze(-1).detach().cpu().numpy()

def predict_moe_masked(m,mods,av3):
    dev=device_of(m)
    mt=_mods_t(mods,dev)
    with torch.no_grad():
        pred=torch.stack([m.experts[k](mt[k]).squeeze(-1) for k in m.names],1)
        joint=torch.cat([mt[k] for k in m.names],1)
        logits=m.router(joint)
        av=torch.from_numpy(av3.astype('float32')).to(dev)
        logits=logits.masked_fill(av<=0,-1e9)
        w=torch.softmax(logits,1)
        return (w*pred).sum(1).detach().cpu().numpy()

def predict_pack(pack,mods,availability4=None,include_masked_controls=False):
    X=concat_modalities(mods)
    nte=pack['nuis'].predict(mods)
    cm=pack['cm'].predict(mods,nte['e_hat'],availability=availability4)
    p={
        'CM_OSR':cm['tau_hat'],
        'T_learner_regularized':pack['t'].predict(X),
        'X_learner':pack['x'].predict(X),
        'DR_learner':pack['dr'].predict(X),
        'R_learner':pack['r'].predict(X),
        'early_fusion_DR_capacity_matched':predict_phi_mlp(pack['early'],pack['early_sc'],X),
        'generic_multimodal_attention_DR_capacity_matched':predict_multimodal_phi(pack['att'],mods),
        'generic_MoE_DR_capacity_matched':predict_multimodal_phi(pack['moe'],mods)
    }
    if include_masked_controls and availability4 is not None:
        av3=availability4[:,:3]
        p['generic_attention_same_model_availability_masked_inference']=predict_attention_masked(pack['att'],mods,av3)
        p['generic_MoE_same_model_availability_masked_inference']=predict_moe_masked(pack['moe'],mods,av3)
    return p,cm

def conformal90(dev_pred,dev_true,test_pred,test_true):
    err=np.abs(np.asarray(dev_pred)-np.asarray(dev_true))
    n=len(err); k=int(math.ceil((n+1)*0.90))
    k=min(max(k,1),n)
    q=float(np.partition(err,k-1)[k-1])
    lo=np.asarray(test_pred)-q; hi=np.asarray(test_pred)+q
    cov=float(np.mean((test_true>=lo)&(test_true<=hi)))
    return {'nominal':0.90,'q_abs_error':q,'coverage':cov,'mean_width':2*q,'n_dev':int(n),'n_test':int(len(test_true))}

def overlap_rows(family,condition,preds,true,e_true_train,e_true_test,dgp_seed,net_seed):
    s_tr=4*np.asarray(e_true_train)*(1-np.asarray(e_true_train))
    s_te=4*np.asarray(e_true_test)*(1-np.asarray(e_true_test))
    q1,q2=np.quantile(s_tr,[1/3,2/3])
    groups=[('low',s_te<=q1),('mid',(s_te>q1)&(s_te<=q2)),('high',s_te>q2)]
    out=[]
    for label,mask in groups:
        for m,p in preds.items():
            if mask.sum()<2: continue
            out.append({'dgp_seed':dgp_seed,'net_seed':net_seed,'family':family,'condition':condition,
                        'stratum':label,'n':int(mask.sum()),'train_q1':float(q1),'train_q2':float(q2),
                        'method':m,'root_pehe':root_pehe(np.asarray(p)[mask],np.asarray(true)[mask])})
    return out

def make_missing(mods_full,indices,c_full,dgp_seed,condition,rate=None,base=.10,slope=.25,split_tag='test'):
    sub={k:np.array(v[indices],copy=True) for k,v in mods_full.items()}
    n=len(indices)
    av=np.ones((n,4),dtype='float32')
    # Fixed across network seeds and methods; differs by DGP, condition, and split only.
    cond_code={'MCAR_10':10,'MCAR_25':25,'MCAR_40':40,'MAR_CLINICAL':77}[condition]
    split_code=1 if split_tag=='dev' else 2
    rng=np.random.default_rng(710000 + int(dgp_seed)*100 + cond_code*10 + split_code)
    if condition.startswith('MCAR'):
        probs=np.full(n,float(rate),dtype=float)
    else:
        probs=np.clip(float(base)+float(slope)*expit(np.asarray(c_full)[indices]),0,0.95)
    for j,name in [(1,'transcriptomic'),(2,'rppa')]:
        miss=rng.random(n)<probs
        av[:,j]=(~miss).astype('float32')
        sub[name][miss,:]=0.0
    av[:,0]=1.0
    av[:,3]=(av[:,0]*av[:,1]*av[:,2]).astype('float32')
    realized={
        'condition':condition,'split':split_tag,
        'transcriptomic_missing_fraction':float(1-av[:,1].mean()),
        'rppa_missing_fraction':float(1-av[:,2].mean()),
        'both_molecular_missing_fraction':float(np.mean((av[:,1]==0)&(av[:,2]==0))),
        'mean_target_probability':float(np.mean(probs))
    }
    return sub,av,realized

def run_family(root,raw,mods,sp,dgp_seed,net_seed,family,regime,cfg,scfg,rows,selective,coverage,overlap,missing_info,cap_records):
    tr,dev,te=sp['train'],sp['dev'],sp['test']
    d=generate(mods,dgp_seed,regime,reference_idx=tr)
    mtr={k:v[tr] for k,v in mods.items()}
    pack=fit_pack(mtr,d['t'][tr],d['y'][tr],net_seed,int(cfg['training']['max_epochs']),int(cfg['training']['patience']),
                  int(cfg['nuisance_folds']),int(cfg['effect_folds']),float(cfg['capacity_tolerance_percent'])/100)
    cap_records[family]=pack['capacity']

    mdev={k:v[dev] for k,v in mods.items()}
    mte={k:v[te] for k,v in mods.items()}
    pdev,cmdev=predict_pack(pack,mdev)
    pte,cmte=predict_pack(pack,mte)
    true_dev=d['tau_true'][dev]; true_te=d['tau_true'][te]

    condition='CLEAN_REFERENCE' if family=='missing_modality' else family.upper()
    for m,p in pte.items():
        rows.append(evalrow(family,condition,m,p,true_te,{'dgp_seed':dgp_seed,'net_seed':net_seed}))
    coverage.append({'dgp_seed':dgp_seed,'net_seed':net_seed,'family':family,'condition':condition,'method':'CM_OSR',**conformal90(pdev['CM_OSR'],true_dev,pte['CM_OSR'],true_te)})
    for rr in selective_curve(pte['CM_OSR'],true_te,cmte['support']):
        selective.append({'dgp_seed':dgp_seed,'net_seed':net_seed,'family':family,'condition':condition,'method':'CM_OSR',**rr})
    thr=float(scfg['selective_analysis']['locked_support_threshold'])
    eligible=cmte['support']>=thr
    selective.append({'dgp_seed':dgp_seed,'net_seed':net_seed,'family':family,'condition':condition,'method':'CM_OSR_support_ge_0.15',
                      'retained':float(np.mean(eligible)),'root_pehe':root_pehe(pte['CM_OSR'][eligible],true_te[eligible]) if eligible.sum()>=2 else float('nan')})
    overlap.extend(overlap_rows(family,condition,pte,true_te,d['e_true'][tr],d['e_true'][te],dgp_seed,net_seed))

    if family!='missing_modality':
        return

    c_full=_proj(mods['clinical'],1101,reference_idx=tr)
    miss_cfg=scfg['stress_families']['missing_modality']
    conditions=[('MCAR_10',.10),('MCAR_25',.25),('MCAR_40',.40),('MAR_CLINICAL',None)]
    for cond,rate in conditions:
        devmods,avdev,rdev=make_missing(mods,dev,c_full,dgp_seed,cond,rate,miss_cfg['mar_base_rate'],miss_cfg['mar_clinical_slope'],'dev')
        temods,avte,rte=make_missing(mods,te,c_full,dgp_seed,cond,rate,miss_cfg['mar_base_rate'],miss_cfg['mar_clinical_slope'],'test')
        missing_info += [dict({'dgp_seed':dgp_seed,'net_seed':net_seed},**rdev),dict({'dgp_seed':dgp_seed,'net_seed':net_seed},**rte)]
        pdv,cmdv=predict_pack(pack,devmods,avdev,include_masked_controls=True)
        pts,cmt=predict_pack(pack,temods,avte,include_masked_controls=True)
        for m,p in pts.items():
            rows.append(evalrow(family,cond,m,p,true_te,{'dgp_seed':dgp_seed,'net_seed':net_seed}))
        coverage.append({'dgp_seed':dgp_seed,'net_seed':net_seed,'family':family,'condition':cond,'method':'CM_OSR',**conformal90(pdv['CM_OSR'],true_dev,pts['CM_OSR'],true_te)})
        for rr in selective_curve(pts['CM_OSR'],true_te,cmt['support']):
            selective.append({'dgp_seed':dgp_seed,'net_seed':net_seed,'family':family,'condition':cond,'method':'CM_OSR',**rr})
        eligible=cmt['support']>=thr
        selective.append({'dgp_seed':dgp_seed,'net_seed':net_seed,'family':family,'condition':cond,'method':'CM_OSR_support_ge_0.15',
                          'retained':float(np.mean(eligible)),'root_pehe':root_pehe(pts['CM_OSR'][eligible],true_te[eligible]) if eligible.sum()>=2 else float('nan')})
        overlap.extend(overlap_rows(family,cond,pts,true_te,d['e_true'][tr],d['e_true'][te],dgp_seed,net_seed))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True); ap.add_argument('--data',required=True)
    ap.add_argument('--dgp-seed',type=int,required=True); ap.add_argument('--net-seed',type=int,required=True)
    a=ap.parse_args()
    root=Path(a.root)
    cfg=yaml.safe_load((root/'00_CONFIG'/'project.yaml').read_text(encoding='utf-8'))
    scfg=yaml.safe_load((root/'00_CONFIG'/'stage5a_stress_lock.yaml').read_text(encoding='utf-8'))
    if a.dgp_seed not in scfg['algorithmic_seeds'] or a.net_seed not in scfg['network_seeds']:
        raise SystemExit('Requested seed is outside Stage5A lock.')

    z=np.load(a.data,allow_pickle=False)
    raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    sp=locked_split(len(z['clinical']))
    tr=sp['train']
    prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()})
    mods=prep.transform(raw)

    run_id=f"STRESS_DGP{a.dgp_seed}_NET{a.net_seed}_{int(time.time())}"
    rd=root/'05_STRESS_RUNS'/run_id
    for sub in ['metrics','predictions']:
        (rd/sub).mkdir(parents=True,exist_ok=True)

    manifest={'run_id':run_id,'dgp_seed':a.dgp_seed,'net_seed':a.net_seed,'stage':'STAGE5A',
              'scientific_settings_changed':False,'primary_grid_rerun':False,'device':'cuda' if torch.cuda.is_available() else 'cpu',
              'stage5a_code_fix_id':STAGE5A_CODE_FIX_ID,
              'started':time.time(),'status':'RUNNING'}
    write_json(rd/'RUN_MANIFEST.json',manifest)

    rows=[]; selective=[]; coverage=[]; overlap=[]; missing_info=[]; cap_records={}
    run_family(root,raw,mods,sp,a.dgp_seed,a.net_seed,'missing_modality',
               'cross_modal_interaction_plus_moderate_weak_overlap',cfg,scfg,rows,selective,coverage,overlap,missing_info,cap_records)
    run_family(root,raw,mods,sp,a.dgp_seed,a.net_seed,'severe_weak_overlap',
               'severe_weak_overlap_stress_only',cfg,scfg,rows,selective,coverage,overlap,missing_info,cap_records)
    run_family(root,raw,mods,sp,a.dgp_seed,a.net_seed,'null_effect_negative_control',
               'null_treatment_effect_negative_control',cfg,scfg,rows,selective,coverage,overlap,missing_info,cap_records)

    pd.DataFrame(rows).to_csv(rd/'metrics'/'summary.csv',index=False)
    pd.DataFrame(selective).to_csv(rd/'metrics'/'selective.csv',index=False)
    pd.DataFrame(coverage).to_csv(rd/'metrics'/'coverage90.csv',index=False)
    pd.DataFrame(overlap).to_csv(rd/'metrics'/'overlap_strata.csv',index=False)
    pd.DataFrame(missing_info).to_csv(rd/'metrics'/'missingness_realized.csv',index=False)
    write_json(rd/'metrics'/'capacity_fairness_by_family.json',cap_records)
    manifest['status']='COMPLETE'; manifest['finished']=time.time()
    write_json(rd/'RUN_MANIFEST.json',manifest)
    write_json(rd/'COMPLETE.json',{'dgp_seed':a.dgp_seed,'net_seed':a.net_seed,'status':'COMPLETE'})
    print('STAGE5A PAIR COMPLETE',a.dgp_seed,a.net_seed,'rows',len(rows),'device',manifest['device'])

if __name__=='__main__': main()
