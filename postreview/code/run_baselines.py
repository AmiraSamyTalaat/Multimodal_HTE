from __future__ import annotations

import argparse, csv, hashlib, json, platform, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

REVISION = Path(__file__).resolve().parents[1]
SOURCE = REVISION / 'source_snapshot'
sys.path.insert(0, str(SOURCE / '01_CODE'))
from cm_osr.baselines import TLearner, XLearner
from cm_osr.dgp import generate
from cm_osr.metrics import policy_regret, root_pehe
from cm_osr.nuisance import crossfit_dr
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split

DGP_SEEDS = [101, 211, 307, 401, 503]
NET_SEEDS = [17, 29, 43]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def metrics(method, prediction, truth, dgp_seed, net_seed, provenance, run_name):
    pred=np.asarray(prediction,dtype=float); truth=np.asarray(truth,dtype=float)
    return {'dgp_seed':dgp_seed,'net_seed':net_seed,'run':run_name,'method':method,
            'root_pehe':root_pehe(pred,truth),'policy_regret':policy_regret(pred,truth),
            'clipped_root_pehe':root_pehe(np.clip(pred,-1,1),truth),
            'fraction_outside_minus1_plus1':float(np.mean((pred < -1)|(pred > 1))),
            'prediction_min':float(pred.min()),'prediction_q01':float(np.quantile(pred,.01)),
            'prediction_q05':float(np.quantile(pred,.05)),'prediction_median':float(np.median(pred)),
            'prediction_q95':float(np.quantile(pred,.95)),'prediction_q99':float(np.quantile(pred,.99)),
            'prediction_max':float(pred.max()),'prediction_mean':float(pred.mean()),
            'prediction_sd':float(pred.std(ddof=0)),
            'true_cate_mean':float(truth.mean()),'true_cate_sd':float(truth.std(ddof=0)),
            'true_cate_min':float(truth.min()),'true_cate_max':float(truth.max()),
            'prediction_source':provenance}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(REVISION/'results'/'baselines'))
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data_file=SOURCE/'02_DATA'/'processed'/'ispy2_core_multimodal.npz'
    with np.load(data_file,allow_pickle=False) as data:
        raw={k:data[k] for k in ['clinical','transcriptomic','rppa']}
        patient_ids=data['patient_id'].copy()
    split=locked_split(len(patient_ids))
    train,dev,test=split['train'],split['dev'],split['test']
    assert (len(train),len(dev),len(test))==(441,147,148)
    assert len(set(train)&set(dev))==len(set(train)&set(test))==len(set(dev)&set(test))==0
    prep=ModalityPreprocessor().fit({k:v[train] for k,v in raw.items()})
    modalities=prep.transform(raw)
    clinical_train=modalities['clinical'][train]
    clinical_test=modalities['clinical'][test]
    modalities_train={k:v[train] for k,v in modalities.items()}
    all_rows=[]; checks=[]; manifests=[]
    start=datetime.now(timezone.utc).isoformat()
    for dgp_seed in DGP_SEEDS:
        dgp=generate(modalities,dgp_seed,'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=train)
        truth=np.asarray(dgp['tau_true'][test],dtype=float)
        treatment=np.asarray(dgp['t'][train]); outcome=np.asarray(dgp['y'][train])
        treated=outcome[treatment==1];control=outcome[treatment==0]
        if not len(treated) or not len(control):raise RuntimeError(f'Empty treatment group for DGP {dgp_seed}')
        crude_difference=float(treated.mean()-control.mean())
        oof=crossfit_dr(modalities_train,treatment,outcome,n_splits=5,seed=17,clip=(0.05,0.95))
        ate=float(np.mean(oof['phi']))
        tl=TLearner().fit(clinical_train,treatment,outcome)
        xl=XLearner().fit(clinical_train,treatment,outcome)
        new_preds={
            'zero_effect':np.zeros(len(test)),
            'oracle_constant_test_cate':np.full(len(test),truth.mean()),
            'train_estimated_ATE_constant':np.full(len(test),ate),
            'crude_train_arm_difference_constant':np.full(len(test),crude_difference),
            'clinical_only_T_learner':tl.predict(clinical_test),
            'clinical_only_X_learner':xl.predict(clinical_test),
        }
        run_manifest={'dgp_seed':dgp_seed,'status':'COMPLETE','started_utc':start,
                      'finished_utc':datetime.now(timezone.utc).isoformat(),
                      'train_size':len(train),'dev_size':len(dev),'test_size':len(test),
                      'split_seed':20260918,'source_data_sha256':sha256(data_file),
                      'train_treated':int(len(treated)),'train_control':int(len(control)),
                      'train_estimated_ate':ate,
                      'train_estimated_ate_method':'mean 5-fold cross-fitted DR pseudo-outcome on TRAIN; nuisance clip 0.05-0.95',
                      'crude_train_arm_difference':crude_difference,
                      'oracle_constant_definition':'mean true CATE on TEST; reference lower bound only; never fit a deployable model',
                      'clinical_models':'original cm_osr.baselines.TLearner and XLearner with clinical features only; fit TRAIN; fixed hyperparameters; DEV unused',
                      'training_access':'TRAIN only; oracle constant uses TEST truth only for a nondeployable reference',
                      'software':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__},
                      'source_code_sha256':{str(p.relative_to(SOURCE)):sha256(p) for p in [SOURCE/'01_CODE/cm_osr/baselines.py',SOURCE/'01_CODE/cm_osr/dgp.py',SOURCE/'01_CODE/cm_osr/splits.py',SOURCE/'01_CODE/cm_osr/nuisance.py']}}
        (out/f'DGP{dgp_seed}_manifest.json').write_text(json.dumps(run_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        manifests.append(run_manifest)
        for net_seed in NET_SEEDS:
            matches=list((SOURCE/'04_RUNS').glob(f'CORE_DGP{dgp_seed}_NET{net_seed}_*'))
            if len(matches)!=1:raise RuntimeError(f'Expected one archived run for DGP {dgp_seed}, net {net_seed}: {matches}')
            run=matches[0];predfile=run/'predictions'/'test_predictions.npz'
            with np.load(predfile,allow_pickle=False) as z:
                if not np.allclose(z['tau_true'],truth,rtol=0,atol=1e-6):
                    raise RuntimeError(f'Archived tau_true differs from regenerated DGP: {run.name}')
                original={k:z[k].copy() for k in z.files if k not in ['tau_true','support']}
            archived_metrics=pd.read_csv(run/'metrics'/'summary.csv')
            for method,pred in original.items():
                row=metrics(method,pred,truth,dgp_seed,net_seed,'archived_test_predictions',run.name)
                archived=float(archived_metrics.loc[archived_metrics['method']==method,'root_pehe'].iloc[0])
                if abs(row['root_pehe']-archived)>2e-6:raise RuntimeError(f'Archived metric mismatch: {run.name} {method}')
                all_rows.append(row)
            for method,pred in new_preds.items():
                all_rows.append(metrics(method,pred,truth,dgp_seed,net_seed,'postreview_train_fit_or_oracle_reference',run.name))
            opt=float(np.maximum(truth,0).mean())
            checks.append({'dgp_seed':dgp_seed,'net_seed':net_seed,'run':run.name,
                           'n_test':len(test),'true_cate_sd':float(truth.std(ddof=0)),
                           'treat_all_policy_regret':float(opt-truth.mean()),
                           'treat_none_policy_regret':opt,'train_estimated_ate':ate,
                           'source_prediction_sha256':sha256(predfile)})
    per_run=pd.DataFrame(all_rows)
    per_run.to_csv(out/'per_run_results.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'reference_policies.csv',index=False)
    # Each DGP is one independent cluster; network repeats are averaged within it.
    per_seed=(per_run.groupby(['dgp_seed','method'],as_index=False)
              .agg(root_pehe=('root_pehe','mean'),policy_regret=('policy_regret','mean'),
                   clipped_root_pehe=('clipped_root_pehe','mean'),
                   fraction_outside_minus1_plus1=('fraction_outside_minus1_plus1','mean'),
                   true_cate_sd=('true_cate_sd','mean'),prediction_min=('prediction_min','min'),
                   prediction_max=('prediction_max','max'),network_repeats=('net_seed','nunique')))
    per_seed.to_csv(out/'per_seed_results.csv',index=False)
    summary=(per_seed.groupby('method',as_index=False)
             .agg(mean_root_pehe=('root_pehe','mean'),sd_across_dgp=('root_pehe','std'),
                  mean_policy_regret=('policy_regret','mean'),
                  mean_clipped_root_pehe=('clipped_root_pehe','mean'),
                  mean_fraction_outside_minus1_plus1=('fraction_outside_minus1_plus1','mean'),
                  n_independent_dgp=('dgp_seed','nunique')))
    summary['mcse_across_5_dgp']=summary['sd_across_dgp']/np.sqrt(summary['n_independent_dgp'])
    summary.sort_values('mean_root_pehe').to_csv(out/'summary_results.csv',index=False)
    pd.DataFrame(checks).groupby('dgp_seed',as_index=False).first().to_csv(out/'per_seed_reference_policies.csv',index=False)
    status={'stage':'BASELINE_POSTREVIEW','status':'COMPLETE','started_utc':start,
            'finished_utc':datetime.now(timezone.utc).isoformat(),
            'dgp_seeds':DGP_SEEDS,'network_seeds':NET_SEEDS,'n_dgp_independent':5,
            'n_original_runs':15,'n_test_per_run':148,'source_data_sha256':sha256(data_file),
            'code_sha256':sha256(Path(__file__)),'errors':[],
            'note':'Post-review analysis; oracle test-CATE constant is nondeployable; no original result was modified.'}
    (out/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.sort_values('mean_root_pehe')[['method','mean_root_pehe','mean_clipped_root_pehe','mean_fraction_outside_minus1_plus1','n_independent_dgp']].to_string(index=False))

if __name__=='__main__':main()
