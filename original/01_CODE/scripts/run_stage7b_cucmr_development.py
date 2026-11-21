from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from cm_osr.splits import locked_split
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.stage7_dgp import generate_stage7, REGIMES
from cm_osr.cucmr import CUCMRCore, CANDIDATES
from cm_osr.metrics import root_pehe
from cm_osr.utils import write_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--data',required=True)
    ap.add_argument('--dgp-seed',type=int,required=True); ap.add_argument('--learner-seed',type=int,required=True); a=ap.parse_args()
    root=Path(a.root)
    lock=yaml.safe_load((root/'00_CONFIG'/'stage7a_cucmr_execution_lock_v4.yaml').read_text(encoding='utf-8'))
    if lock['status']!='LOCKED_BEFORE_STAGE7_RESULTS': raise SystemExit('Stage7 V3 lock invalid.')
    if a.dgp_seed not in lock['development']['dgp_seeds'] or a.learner_seed not in lock['development']['learner_seeds']:
        raise SystemExit('Seed outside Stage7B development lock.')
    z=np.load(a.data,allow_pickle=False)
    raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    sp=locked_split(len(z['clinical'])); tr,dev=sp['train'],sp['dev']
    # Explicit hard guard: Stage7B uses only TRAIN and DEV.
    prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}); mods=prep.transform(raw)
    mtr={k:v[tr] for k,v in mods.items()}; mdev={k:v[dev] for k,v in mods.items()}
    rd=root/'08_STAGE7_DEV_RUNS'/f'DEV7_DGP{a.dgp_seed}_L{a.learner_seed}_{int(time.time())}'
    (rd/'metrics').mkdir(parents=True,exist_ok=True)
    manifest={'stage':'STAGE7B_CUCMR_DEVELOPMENT','dgp_seed':a.dgp_seed,'learner_seed':a.learner_seed,
              'fit_split':'TRAIN_ONLY','evaluation_split':'DEV_ONLY','test_accessed':False,
              'confirmatory_seed_accessed':False,'status':'RUNNING','started':time.time(),'regimes':REGIMES}
    write_json(rd/'RUN_MANIFEST.json',manifest)
    pred_rows=[]; metric_rows=[]; diag=[]
    for regime in REGIMES:
        d=generate_stage7(mods,a.dgp_seed,regime,reference_idx=tr,
                          incremental_ratio=float(lock['dgp']['incremental_sd_ratio_to_clinical_effect_component']),
                          propensity_clip=tuple(lock['dgp']['propensity_clip']))
        core=CUCMRCore(seed=a.learner_seed, nuisance_folds=int(lock['crossfit']['nuisance_folds']), anchor_folds=int(lock['crossfit']['anchor_folds']), candidate_folds=int(lock['crossfit']['candidate_folds']), utility_folds=int(lock['crossfit']['utility_score_folds'])).fit(mtr,d['t'][tr],d['y'][tr])
        p=core.predict(mdev); phi=core.phi_eval(p['nuisance_pred'],d['t'][dev],d['y'][dev])
        methods={'FIXED_CLINICAL_X_ANCHOR':p['anchor_tau'],'FULL_MULTIMODAL_X_LEARNER':p['full_x_tau'],
                 'DR_LOSS_GLOBAL_STACKING':p['stack_tau'],
                 'TRAIN_OOF_GLOBAL_BEST_SINGLE_UPDATE':p['global_best_tau']}
        for m in CANDIDATES: methods[f'SINGLE_UPDATE_{m}']=p[f'tau_{m}']
        for name,tau in methods.items():
            metric_rows.append({'dgp_seed':a.dgp_seed,'learner_seed':a.learner_seed,'regime':regime,'method':name,
                                'root_pehe':float(root_pehe(tau,d['tau_true'][dev]))})
        diag.append({'dgp_seed':a.dgp_seed,'learner_seed':a.learner_seed,'regime':regime,
                     'stack_weights':json.dumps([float(x) for x in core.stack_weights]),
                     'global_best_candidate_train_oof':core.global_best_candidate,
                     'train_mean_D':json.dumps(core.train_mean_D,sort_keys=True),
                     'incremental_alpha':float(d['incremental_alpha'])})
        for local_i,global_i in enumerate(dev):
            r={'dgp_seed':a.dgp_seed,'learner_seed':a.learner_seed,'regime':regime,
               'sample_index':int(global_i),'true_tau':float(d['tau_true'][global_i]),
               'true_source':int(d['incremental_source'][global_i]),'phi_dev':float(phi[local_i]),
               'anchor_tau':float(p['anchor_tau'][local_i]),'full_x_tau':float(p['full_x_tau'][local_i]),
               'stack_tau':float(p['stack_tau'][local_i]),
               'global_best_tau':float(p['global_best_tau'][local_i])}
            for m in CANDIDATES:
                tau=float(p[f'tau_{m}'][local_i]); delta=float(p[f'delta_{m}'][local_i]); u=float(p[f'utility_{m}'][local_i]); up=float(p[f'utility_pct_{m}'][local_i])
                D=(float(phi[local_i])-r['anchor_tau'])**2-(float(phi[local_i])-tau)**2
                r[f'tau_{m}']=tau; r[f'delta_{m}']=delta; r[f'utility_{m}']=u; r[f'utility_pct_{m}']=up; r[f'D_{m}']=float(D)
            pred_rows.append(r)
    pd.DataFrame(pred_rows).to_csv(rd/'metrics'/'patient_dev_predictions.csv',index=False)
    pd.DataFrame(metric_rows).to_csv(rd/'metrics'/'baseline_dev_metrics.csv',index=False)
    pd.DataFrame(diag).to_csv(rd/'metrics'/'fit_diagnostics.csv',index=False)
    manifest['status']='COMPLETE'; manifest['finished']=time.time(); write_json(rd/'RUN_MANIFEST.json',manifest)
    write_json(rd/'COMPLETE.json',{'stage':'STAGE7B_CUCMR_DEVELOPMENT','dgp_seed':a.dgp_seed,'learner_seed':a.learner_seed,
                                   'status':'COMPLETE','test_accessed':False})
    print('STAGE7B DEVELOPMENT PAIR COMPLETE',a.dgp_seed,a.learner_seed,'patient_rows',len(pred_rows))

if __name__=='__main__': main()
