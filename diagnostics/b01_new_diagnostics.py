#!/usr/bin/env python3
"""Post-review validation from archived per-patient predictions.
Reads only previously preserved predictions; does not train or modify models.
Usage: python b01_new_diagnostics.py --archive path/to/B01_POST_REVIEW_2026.zip --out outputs
"""
from __future__ import annotations
import argparse, json, io, re, hashlib
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

def metric_pehe(h, t):return float(np.sqrt(np.mean((h-t)**2)))
def metric_regret(h,t):return float(np.maximum(t,0).mean()-np.where(h>0,t,0).mean())
def metric_calibration(h,t,bins=10):
    q=np.unique(np.quantile(h,np.linspace(0,1,bins+1)))
    if len(q)<3:return np.nan
    idx=np.digitize(h,q[1:-1],right=True)
    return float(np.mean([abs(h[idx==b].mean()-t[idx==b].mean()) for b in np.unique(idx)]))
def deciles(h,t):
    # Identical quantile-bin convention to the original script.
    q=np.unique(np.quantile(h,np.linspace(0,1,11)))
    idx=np.digitize(h,q[1:-1],right=True)
    if len(np.unique(idx))!=10:
        return None
    return np.asarray([(np.mean(h[idx==j]),np.mean(t[idx==j]),(idx==j).sum()) for j in range(10)])
def draw_bootstrap(runs,method,B=2000,seed=20261121):
    # Two-level resample: 5 DGP clusters (with the 3 net seeds kept together)
    # and common TEST patient IDs resampled jointly across all selected runs.
    rng=np.random.default_rng(seed)
    groups={k:[r for r in runs if r['seed']==k] for k in sorted({r['seed'] for r in runs})}
    keys=list(groups)
    original=np.stack([deciles(r[method],r['tau_true']) for r in runs]).mean(axis=0)
    arr=np.full((B,10,2),np.nan)
    for b in range(B):
        pick=rng.choice(keys,len(keys),replace=True)
        # Shared patient-index bootstrap preserves reuse of the same TEST set among DGP seeds.
        patients=rng.integers(0,148,148)
        d=[]
        for key in pick:
            for r in groups[key]:
                v=deciles(r[method][patients],r['tau_true'][patients])
                if v is not None: d.append(v[:,:2])
        if d: arr[b]=np.mean(d,axis=0)
    res=[]
    for k in range(10):
        p=arr[:,k,0];t=arr[:,k,1];diff=p-t
        res.append({'method':method,'bin':k+1,'mean_estimated':original[k,0],
          'mean_true':original[k,1], 'signed_bias':original[k,0]-original[k,1],
          'mean_patients_in_bin':original[k,2],
          'estimated_ci_low':np.nanquantile(p,.025),'estimated_ci_high':np.nanquantile(p,.975),
          'true_ci_low':np.nanquantile(t,.025),'true_ci_high':np.nanquantile(t,.975),
          'signed_bias_ci_low':np.nanquantile(diff,.025),'signed_bias_ci_high':np.nanquantile(diff,.975),
          'valid_resamples':int(np.isfinite(diff).sum()), 'bootstrap_B':B,'dgp_clusters':5,
          'test_patients':148})
    return res

def main():
    a=argparse.ArgumentParser();a.add_argument('--archive',required=True);a.add_argument('--out',required=True)
    a.add_argument('--bootstrap',type=int,default=2000);args=a.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.archive) as z:
        runs=[]
        for f in z.namelist():
            if '/source_snapshot/04_RUNS/CORE_DGP' not in f or not f.endswith('/predictions/test_predictions.npz'):continue
            p=re.search(r'CORE_DGP(\d+)_NET(\d+)',f)
            arr=np.load(io.BytesIO(z.read(f)),allow_pickle=False)
            rr={k:np.array(arr[k],dtype=float) for k in arr.files}
            rr['seed']=int(p[1]);rr['net']=int(p[2]);rr['run']=p.group(0);runs.append(rr)
        assert len(runs)==15, 'Missing archived original TEST predictions'
        data=pd.read_csv(io.BytesIO(z.read(next(x for x in z.namelist() if x.endswith('/06_RESULTS/ALL_CORE_METRICS.csv')))))
        check=[]
        for r in runs:
            for method in [k for k in r if isinstance(r[k],np.ndarray) and k not in ('tau_true','support')]:
                v=data[(data.dgp_seed==r['seed'])&(data.net_seed==r['net'])&(data.method==method)]
                assert len(v)==1,(r['run'],method)
                for metric,fn in [('root_pehe',metric_pehe),('policy_regret',metric_regret),('cate_calibration_error',metric_calibration)]:
                    calc=fn(r[method],r['tau_true']);arch=float(v.iloc[0][metric]);tol=1e-6+1e-3*abs(arch)
                    check.append({'dgp_seed':r['seed'],'net_seed':r['net'],'method':method,'metric':metric,
                                  'archived':arch,'recalculated':calc,'absolute_difference':abs(calc-arch),
                                  'within_rtol_1e3_atol_1e6':bool(abs(calc-arch)<=tol)})
        checks=pd.DataFrame(check);checks.to_csv(out/'archived_prediction_metric_reconstruction.csv',index=False)
        print('reconstructed',len(checks),'metric checks, pass',checks.within_rtol_1e3_atol_1e6.sum(), 'max_diff',checks.absolute_difference.max())
        if not bool(checks.within_rtol_1e3_atol_1e6.all()):
            print(checks[~checks.within_rtol_1e3_atol_1e6].head().to_string());raise RuntimeError('Metric reconstruction does not pass')
        # TPR/FPR from preserved Stage 7B individualized DEV predictions and certification.
        file=next(x for x in z.namelist() if x.endswith('/06_RESULTS/STAGE7B/ALL_STAGE7B_PATIENT_DEV_PREDICTIONS.csv'))
        stage=pd.read_csv(io.BytesIO(z.read(file)))
        file=next(x for x in z.namelist() if x.endswith('/results/additional/utility_gate_rates_by_seed.csv'))
        archived_gate=pd.read_csv(io.BytesIO(z.read(file)))
        records=[];rows=[]
        for key,g in stage.groupby(['dgp_seed','learner_seed','regime'],sort=True):
            y=g.true_tau.to_numpy(float);anchor=g.anchor_tau.to_numpy(float)
            candidates=['TRANSCRIPTOMIC','RPPA','JOINT_MOLECULAR']
            errors=np.stack([(g['tau_'+m].to_numpy(float)-y)**2 for m in candidates])
            anc=(anchor-y)**2; benefit=(errors<anc[None,:]-1e-12).any(axis=0)
            # The archived certified gate, not the separate unconstrained stacking baseline,
            # determines the decision. Only candidates named in selected_certification can be accepted.
            decision=json.loads(z.read(next(x for x in z.namelist() if x.endswith('/06_RESULTS/STAGE7B/STAGE7B_DEVELOPMENT_SELECTION.json'))))
            certified=[m for m in candidates if decision['selected_certification'][m] is not None]
            actual_accept=np.zeros(len(g),dtype=bool)
            for m in certified:
                threshold=float(decision['selected_certification'][m]['percentile_threshold'])
                actual_accept |= g['utility_pct_'+m].to_numpy(float)>=threshold
            ss=g.true_source.to_numpy(int)!=0
            for name,truth in [('oracle_any_candidate_improves_CATE_squared_error',benefit),('known_nonclinical_effect_source',ss)]:
                pos=int(truth.sum());neg=int((~truth).sum())
                TP=int((actual_accept&truth).sum());FP=int((actual_accept&~truth).sum())
                FN=pos-TP;TN=neg-FP
                records.append({'dgp_seed':key[0],'learner_seed':key[1],'regime':key[2],
                  'truth_definition':name,'n_dev':len(g),'n_oracle_positive':pos,'n_oracle_negative':neg,
                  'n_accepted':int(actual_accept.sum()),'TP':TP,'FP':FP,'TN':TN,'FN':FN,
                  'TPR':TP/pos if pos else np.nan,'FPR':FP/neg if neg else np.nan})
            rows.append({'dgp_seed':key[0],'learner_seed':key[1],'regime':key[2],
                 'oracle_positive_fraction_recalc':float(np.mean(benefit)),
                 'gate_accepted_fraction_recalc':float(np.mean(actual_accept))})
        gate=pd.DataFrame(records);gate.to_csv(out/'stage7b_oracle_gate_confusion_per_run.csv',index=False)
        sums=gate.groupby(['regime','truth_definition'],as_index=False).agg(n_evaluations=('n_dev','sum'),
                  n_oracle_positive=('n_oracle_positive','sum'),n_oracle_negative=('n_oracle_negative','sum'),
                  TP=('TP','sum'),FP=('FP','sum'),TN=('TN','sum'),FN=('FN','sum'))
        sums['TPR']=sums.TP/sums.n_oracle_positive.replace(0,np.nan)
        sums['FPR']=sums.FP/sums.n_oracle_negative.replace(0,np.nan)
        sums['oracle_positive_pct']=100*sums.n_oracle_positive/sums.n_evaluations
        sums.to_csv(out/'stage7b_oracle_gate_summary.csv',index=False)
        compare=pd.DataFrame(rows).merge(archived_gate,on=['dgp_seed','learner_seed','regime'])
        print('Gate 60 groups; max oracle-positive-rate difference',
             np.max(abs(compare.oracle_positive_fraction_recalc-compare.oracle_beneficial_candidate_fraction)))
        print('Max gate-acceptance fraction difference',np.max(abs(compare.gate_accepted_fraction_recalc-compare.gate_acceptance_rate)))
        if (compare.gate_accepted_fraction_recalc-compare.gate_acceptance_rate).abs().max()>1e-8:
            raise RuntimeError('Gate acceptance differs from archived analysis')
        if (compare.oracle_positive_fraction_recalc-compare.oracle_beneficial_candidate_fraction).abs().max()>1e-6:
            raise RuntimeError('Oracle labels differ from archived derived gate rates')
        # Calibration deciles, multiple models and two-dimensional bootstrap uncertainty
        cal=[]
        for method in ['X_learner','T_learner_regularized','CM_OSR']:
            print('Bootstrapping',method,flush=True)
            cal+=draw_bootstrap(runs,method,B=args.bootstrap,seed=20261121+['X_learner','T_learner_regularized','CM_OSR'].index(method)*10)
        pd.DataFrame(cal).to_csv(out/'ten_bin_cate_calibration_intervals.csv',index=False)
        # original decision metric verification (not independent retraining)
        contrast=checks[checks.metric=='root_pehe'].pivot(index=['dgp_seed','net_seed'],columns='method',values='recalculated')
        raw_diff=contrast['CM_OSR']-contrast['early_fusion_DR_capacity_matched']
        old_gate=json.loads(z.read(next(x for x in z.namelist() if x.endswith('06_RESULTS/PRIMARY_GATE.json'))))
        assert abs(raw_diff.mean()-float(old_gate['mean']))<1e-4
        ci={'source_archives_sha256':hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
          'original_test_files':15,'source_rows':len(stage),'recomputed_stage4_values':len(checks),
          'recomputed_stage4_all_within_tolerance':bool(checks.within_rtol_1e3_atol_1e6.all()),
          'atol':1e-6,'rtol':1e-3,'archived_primary_delta':old_gate['mean'],'recomputed_primary_delta':float(raw_diff.mean()),
          'stages_independently_retrained':0,'archived_results_recomputed_from_saved_predictions':True,
          'interpretation':'Metric reconstruction and archival integrity check only; NOT independent retraining of the original pipeline.'}
        (out/'reconstruction_validation.json').write_text(json.dumps(ci,indent=2),encoding='utf-8')
        print('Completed',out)
if __name__=='__main__': main()
