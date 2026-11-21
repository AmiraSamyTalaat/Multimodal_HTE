from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT_CODE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT_CODE))
from cm_osr.splits import locked_split
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.dgp import generate
from cm_osr.nuisance import crossfit_dr, FullNuisance
from cm_osr.model import CMOSR
from cm_osr.baselines import TLearner, DRLearner, RLearner
from cm_osr.metrics import root_pehe, policy_regret, cate_calibration_error, selective_curve
from cm_osr.utils import concat_modalities, write_json, seed_everything
from cm_osr.runio import init_run, finish_run

def load_npz(path):
    z=np.load(path,allow_pickle=False)
    return {k:z[k] for k in ['clinical','transcriptomic','rppa']}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--data',required=True); ap.add_argument('--dgp-seed',type=int,default=101); ap.add_argument('--net-seed',type=int,default=17); a=ap.parse_args()
    root=Path(a.root); cfg=yaml.safe_load((root/'00_CONFIG'/'project.yaml').read_text(encoding='utf-8'))
    seed_everything(a.net_seed); raw=load_npz(a.data); sp=locked_split(len(next(iter(raw.values()))))
    pp=ModalityPreprocessor().fit({k:v[sp['train']] for k,v in raw.items()}); mods=pp.transform(raw)
    d=generate(mods,a.dgp_seed,cfg['primary_regime'],reference_idx=sp['train'])
    run_id=f"SEMISYN_DGP{a.dgp_seed}_NET{a.net_seed}_{int(time.time())}"; rd,man=init_run(root,run_id,{'dgp_seed':a.dgp_seed,'net_seed':a.net_seed,'primary_regime':cfg['primary_regime']})
    tr,te=sp['train'],sp['test']; mtr={k:v[tr] for k,v in mods.items()}; mte={k:v[te] for k,v in mods.items()}
    oof=crossfit_dr(mtr,d['t'][tr],d['y'][tr],n_splits=cfg['nuisance_folds'],seed=a.net_seed)
    nuis=FullNuisance(a.net_seed).fit(mtr,d['t'][tr],d['y'][tr]); nte=nuis.predict(mte)
    proposed=CMOSR(hidden=64,folds=cfg['effect_folds'],seed=a.net_seed).fit(mtr,oof['phi'],oof['e_hat']); ppred=proposed.predict(mte,nte['e_hat'])
    Xtr=concat_modalities(mtr); Xte=concat_modalities(mte)
    preds={'CM_OSR':ppred['tau_hat']}
    preds['T_learner']=TLearner().fit(Xtr,d['t'][tr],d['y'][tr]).predict(Xte)
    preds['DR_learner']=DRLearner().fit(Xtr,oof['phi']).predict(Xte)
    preds['R_learner']=RLearner().fit(Xtr,d['t'][tr],d['y'][tr]).predict(Xte)
    rows=[]
    for name,p in preds.items(): rows.append({'method':name,'root_pehe':root_pehe(p,d['tau_true'][te]),'policy_regret':policy_regret(p,d['tau_true'][te]),'cate_calibration_error':cate_calibration_error(p,d['tau_true'][te])})
    pd.DataFrame(rows).to_csv(rd/'metrics'/'summary.csv',index=False)
    np.savez_compressed(rd/'predictions'/'test_predictions.npz',tau_true=d['tau_true'][te],support=ppred['support'],**preds)
    write_json(rd/'metrics'/'CM_OSR_selective.json',selective_curve(ppred['tau_hat'],d['tau_true'][te],ppred['support']))
    finish_run(rd,man); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
