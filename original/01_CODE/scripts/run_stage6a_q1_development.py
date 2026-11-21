
from __future__ import annotations
import argparse, time, sys
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
from cm_osr.baselines import TLearner, XLearner, RLearner
from cm_osr.metrics import root_pehe, policy_regret, cate_calibration_error
from cm_osr.utils import concat_modalities, seed_everything, write_json
from cm_osr.q1_candidates import AnchoredOrthogonalSupportRouter

CANDIDATES=[
    'CM_OSR_v1_frozen_reference',
    'X_learner_frozen_reference',
    'T_learner_frozen_reference',
    'R_learner_frozen_reference',
    'AOSR_X_gamma1',
    'AOSR_X_gamma2',
    'AOSR_R_gamma1',
    'AOSR_R_gamma2',
    'CMOSR_support_shrink_gamma1',
    'CMOSR_support_shrink_gamma2',
]

NON_NULL_CONDITIONS=[
    'PRIMARY_CROSS_MODAL',
    'SEVERE_WEAK_OVERLAP',
    'MCAR_40',
    'MAR_CLINICAL',
]

def evalrow(condition,name,p,true,dgp_seed,net_seed,extra=None):
    r={
        'dgp_seed':int(dgp_seed),'net_seed':int(net_seed),
        'condition':condition,'method':name,
        'root_pehe':root_pehe(p,true),
        'policy_regret':policy_regret(p,true),
        'cate_calibration_error':cate_calibration_error(p,true),
        'mean_tau_hat':float(np.mean(p)),
        'mean_abs_tau_hat':float(np.mean(np.abs(p))),
    }
    if extra: r.update(extra)
    return r

def _variants_from_cm(cmout):
    s=np.clip(np.asarray(cmout['support'],dtype=np.float32),0.0,1.0)
    tau=np.asarray(cmout['tau_hat'],dtype=np.float32)
    return {
        'CM_OSR_v1_frozen_reference':tau,
        'CMOSR_support_shrink_gamma1':s*tau,
        'CMOSR_support_shrink_gamma2':(s**2)*tau,
    }

def _variants_from_aosr(out,prefix):
    s=np.clip(np.asarray(out['support'],dtype=np.float32),0.0,1.0)
    a=np.asarray(out['anchor_tau'],dtype=np.float32)
    r=np.asarray(out['residual_tau'],dtype=np.float32)
    return {
        f'{prefix}_gamma1':a+s*r,
        f'{prefix}_gamma2':a+(s**2)*r,
    }

def fit_family(mtr,ttr,ytr,net_seed,cfg):
    seed_everything(net_seed)
    oof=crossfit_dr(mtr,ttr,ytr,int(cfg['nuisance_folds']),net_seed)
    nuis=FullNuisance(net_seed).fit(mtr,ttr,ytr)
    Xtr=concat_modalities(mtr)

    cm=CMOSR(hidden=64,folds=int(cfg['effect_folds']),seed=net_seed,
             expert_epochs=int(cfg['training']['max_epochs']),
             expert_patience=int(cfg['training']['patience'])).fit(
                 mtr,oof['phi'],oof['e_hat'])

    x=XLearner().fit(Xtr,ttr,ytr)
    t=TLearner().fit(Xtr,ttr,ytr)
    r=RLearner().fit(Xtr,ttr,ytr)

    aosrx=AnchoredOrthogonalSupportRouter(
        anchor_kind='X',gamma=1.0,hidden=64,folds=int(cfg['effect_folds']),
        seed=net_seed,expert_epochs=int(cfg['training']['max_epochs']),
        expert_patience=int(cfg['training']['patience'])
    ).fit(mtr,ttr,ytr,oof['phi'],oof['e_hat'])

    aosrr=AnchoredOrthogonalSupportRouter(
        anchor_kind='R',gamma=1.0,hidden=64,folds=int(cfg['effect_folds']),
        seed=net_seed,expert_epochs=int(cfg['training']['max_epochs']),
        expert_patience=int(cfg['training']['patience'])
    ).fit(mtr,ttr,ytr,oof['phi'],oof['e_hat'])

    return {'nuis':nuis,'cm':cm,'x':x,'t':t,'r':r,'aosrx':aosrx,'aosrr':aosrr}

def predict_family(pack,mods,availability=None):
    X=concat_modalities(mods)
    e=pack['nuis'].predict(mods)['e_hat']
    cm=pack['cm'].predict(mods,e,availability=availability)
    ax=pack['aosrx'].predict(mods,e,availability=availability)
    ar=pack['aosrr'].predict(mods,e,availability=availability)

    p={
        'X_learner_frozen_reference':np.asarray(pack['x'].predict(X),dtype=np.float32),
        'T_learner_frozen_reference':np.asarray(pack['t'].predict(X),dtype=np.float32),
        'R_learner_frozen_reference':np.asarray(pack['r'].predict(X),dtype=np.float32),
        **_variants_from_cm(cm),
        **_variants_from_aosr(ax,'AOSR_X'),
        **_variants_from_aosr(ar,'AOSR_R'),
    }
    extra={
        'AOSR_X':{
            'mean_abs_routed_correction_gamma1':float(np.mean(np.abs(np.asarray(ax['support'])*np.asarray(ax['residual_tau'])))),
            'mean_abs_routed_correction_gamma2':float(np.mean(np.abs((np.asarray(ax['support'])**2)*np.asarray(ax['residual_tau'])))),
            'mean_support':float(np.mean(ax['support'])),
        },
        'AOSR_R':{
            'mean_abs_routed_correction_gamma1':float(np.mean(np.abs(np.asarray(ar['support'])*np.asarray(ar['residual_tau'])))),
            'mean_abs_routed_correction_gamma2':float(np.mean(np.abs((np.asarray(ar['support'])**2)*np.asarray(ar['residual_tau'])))),
            'mean_support':float(np.mean(ar['support'])),
        },
        'CM_OSR':{'mean_support':float(np.mean(cm['support']))},
    }
    return p,extra

def make_missing_dev(mods_full,dev,c_full,dgp_seed,condition):
    sub={k:np.array(v[dev],copy=True) for k,v in mods_full.items()}
    n=len(dev); av=np.ones((n,4),dtype=np.float32)
    code={'MCAR_40':40,'MAR_CLINICAL':77}[condition]
    rng=np.random.default_rng(860000 + int(dgp_seed)*100 + code*10 + 1)
    if condition=='MCAR_40':
        probs=np.full(n,0.40,dtype=float)
    else:
        probs=np.clip(0.10+0.25*expit(np.asarray(c_full)[dev]),0,0.95)
    for j,name in [(1,'transcriptomic'),(2,'rppa')]:
        miss=rng.random(n)<probs
        av[:,j]=(~miss).astype(np.float32)
        sub[name][miss,:]=0.0
    av[:,0]=1.0
    av[:,3]=(av[:,0]*av[:,1]*av[:,2]).astype(np.float32)
    realized={
        'dgp_seed':int(dgp_seed),'condition':condition,'split':'DEV',
        'transcriptomic_missing_fraction':float(1-av[:,1].mean()),
        'rppa_missing_fraction':float(1-av[:,2].mean()),
        'both_molecular_missing_fraction':float(np.mean((av[:,1]==0)&(av[:,2]==0))),
        'mean_target_probability':float(np.mean(probs)),
    }
    return sub,av,realized

def append_predictions(rows,condition,preds,true,dgp_seed,net_seed,diag):
    for name,p in preds.items():
        extra={}
        if name=='AOSR_X_gamma1':
            extra['mean_abs_routed_correction']=diag['AOSR_X']['mean_abs_routed_correction_gamma1']
            extra['mean_support']=diag['AOSR_X']['mean_support']
        elif name=='AOSR_X_gamma2':
            extra['mean_abs_routed_correction']=diag['AOSR_X']['mean_abs_routed_correction_gamma2']
            extra['mean_support']=diag['AOSR_X']['mean_support']
        elif name=='AOSR_R_gamma1':
            extra['mean_abs_routed_correction']=diag['AOSR_R']['mean_abs_routed_correction_gamma1']
            extra['mean_support']=diag['AOSR_R']['mean_support']
        elif name=='AOSR_R_gamma2':
            extra['mean_abs_routed_correction']=diag['AOSR_R']['mean_abs_routed_correction_gamma2']
            extra['mean_support']=diag['AOSR_R']['mean_support']
        elif name.startswith('CM_OSR') or name.startswith('CMOSR_'):
            extra['mean_support']=diag['CM_OSR']['mean_support']
        rows.append(evalrow(condition,name,p,true,dgp_seed,net_seed,extra))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--data',required=True)
    ap.add_argument('--dgp-seed',type=int,required=True)
    ap.add_argument('--net-seed',type=int,required=True)
    a=ap.parse_args()

    root=Path(a.root)
    lock=yaml.safe_load((root/'00_CONFIG'/'stage6a_q1_method_development_lock.yaml').read_text(encoding='utf-8'))
    cfg=yaml.safe_load((root/'00_CONFIG'/'project.yaml').read_text(encoding='utf-8'))

    if lock['status']!='LOCKED_BEFORE_STAGE6A_RESULTS':
        raise SystemExit('Stage6A lock status is invalid.')
    if lock.get('test_split_access_in_stage6a') is not False:
        raise SystemExit('Stage6A TEST access is not forbidden by the lock.')
    if a.dgp_seed not in lock['development_dgp_seeds']:
        raise SystemExit('DGP seed outside Stage6A development lock.')
    if a.net_seed not in lock['development_network_seeds']:
        raise SystemExit('Network seed outside Stage6A development lock.')

    z=np.load(a.data,allow_pickle=False)
    raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    sp=locked_split(len(z['clinical']))
    tr,dev=sp['train'],sp['dev']
    prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()})
    mods=prep.transform(raw)
    mtr={k:v[tr] for k,v in mods.items()}

    run_id=f"DEV6A_DGP{a.dgp_seed}_NET{a.net_seed}_{int(time.time())}"
    rd=root/'07_METHOD_DEV_RUNS'/run_id
    (rd/'metrics').mkdir(parents=True,exist_ok=True)

    manifest={
        'run_id':run_id,'stage':'STAGE6A_Q1_METHOD_DEVELOPMENT',
        'dgp_seed':int(a.dgp_seed),'net_seed':int(a.net_seed),
        'split_used_for_candidate_evaluation':'DEV_ONLY',
        'test_accessed':False,'primary_grid_rerun':False,
        'stage5a_rerun':False,'device':'cuda' if torch.cuda.is_available() else 'cpu',
        'started':time.time(),'status':'RUNNING'
    }
    write_json(rd/'RUN_MANIFEST.json',manifest)

    rows=[]; missing=[]

    d=generate(mods,a.dgp_seed,'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
    pack=fit_family(mtr,d['t'][tr],d['y'][tr],a.net_seed,cfg)
    mdev={k:v[dev] for k,v in mods.items()}
    p,diag=predict_family(pack,mdev)
    append_predictions(rows,'PRIMARY_CROSS_MODAL',p,d['tau_true'][dev],a.dgp_seed,a.net_seed,diag)

    c_full=_proj(mods['clinical'],1101,reference_idx=tr)
    for cond in ['MCAR_40','MAR_CLINICAL']:
        mm,av,real=make_missing_dev(mods,dev,c_full,a.dgp_seed,cond)
        missing.append(dict(real,net_seed=int(a.net_seed)))
        p2,diag2=predict_family(pack,mm,availability=av)
        append_predictions(rows,cond,p2,d['tau_true'][dev],a.dgp_seed,a.net_seed,diag2)

    ds=generate(mods,a.dgp_seed,'severe_weak_overlap_stress_only',reference_idx=tr)
    packs=fit_family(mtr,ds['t'][tr],ds['y'][tr],a.net_seed,cfg)
    ps,diags=predict_family(packs,mdev)
    append_predictions(rows,'SEVERE_WEAK_OVERLAP',ps,ds['tau_true'][dev],a.dgp_seed,a.net_seed,diags)

    dn=generate(mods,a.dgp_seed,'null_treatment_effect_negative_control',reference_idx=tr)
    packn=fit_family(mtr,dn['t'][tr],dn['y'][tr],a.net_seed,cfg)
    pn,diagn=predict_family(packn,mdev)
    append_predictions(rows,'NULL_EFFECT_NEGATIVE_CONTROL',pn,dn['tau_true'][dev],a.dgp_seed,a.net_seed,diagn)

    df=pd.DataFrame(rows)
    if set(df['method']) != set(CANDIDATES):
        raise RuntimeError(f'Candidate method mismatch: got={sorted(df.method.unique())}')
    if set(df['condition']) != set(NON_NULL_CONDITIONS+['NULL_EFFECT_NEGATIVE_CONTROL']):
        raise RuntimeError(f'Condition mismatch: got={sorted(df.condition.unique())}')
    if df[['root_pehe','mean_abs_tau_hat']].isna().any().any():
        raise RuntimeError('NaN detected in Stage6A development metrics.')

    df.to_csv(rd/'metrics'/'summary_dev.csv',index=False)
    pd.DataFrame(missing).to_csv(rd/'metrics'/'missingness_realized_dev.csv',index=False)
    manifest['status']='COMPLETE'; manifest['finished']=time.time()
    write_json(rd/'RUN_MANIFEST.json',manifest)
    write_json(rd/'COMPLETE.json',{
        'stage':'STAGE6A_Q1_METHOD_DEVELOPMENT',
        'dgp_seed':int(a.dgp_seed),'net_seed':int(a.net_seed),
        'status':'COMPLETE','test_accessed':False
    })
    print('STAGE6A DEV PAIR COMPLETE',a.dgp_seed,a.net_seed,'rows',len(df),'device',manifest['device'])

if __name__=='__main__':
    main()
