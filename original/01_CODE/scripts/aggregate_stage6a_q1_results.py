
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd, yaml

def pair_from_name(name):
    m=re.match(r'^DEV6A_DGP(\d+)_NET(\d+)_',name)
    return (int(m.group(1)),int(m.group(2))) if m else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args()
    root=Path(a.root)
    lock=yaml.safe_load((root/'00_CONFIG'/'stage6a_q1_method_development_lock.yaml').read_text(encoding='utf-8'))
    expected={(int(d),int(n)) for d in lock['development_dgp_seeds'] for n in lock['development_network_seeds']}

    complete={}
    p=root/'07_METHOD_DEV_RUNS'
    if not p.exists(): raise SystemExit('Stage6A run folder is missing.')
    for rd in p.glob('DEV6A_DGP*_NET*_*'):
        if not (rd/'COMPLETE.json').exists(): continue
        pair=pair_from_name(rd.name)
        if not pair: continue
        if pair not in complete or complete[pair].stat().st_mtime < rd.stat().st_mtime:
            complete[pair]=rd

    missing_pairs=sorted(expected-set(complete))
    if missing_pairs:
        raise SystemExit(f'Stage6A development grid incomplete: {len(set(complete)&expected)}/{len(expected)}; missing={missing_pairs}')

    frames=[]; mframes=[]
    for pair,rd in sorted(complete.items()):
        if pair not in expected: continue
        x=pd.read_csv(rd/'metrics'/'summary_dev.csv'); x['run_dir']=rd.name; frames.append(x)
        q=rd/'metrics'/'missingness_realized_dev.csv'
        if q.exists():
            m=pd.read_csv(q); m['run_dir']=rd.name; mframes.append(m)

    df=pd.concat(frames,ignore_index=True)
    out=root/'06_RESULTS'/'STAGE6A'; out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'ALL_STAGE6A_DEV_METRICS.csv',index=False)
    if mframes:
        pd.concat(mframes,ignore_index=True).to_csv(out/'ALL_STAGE6A_MISSINGNESS_REALIZED.csv',index=False)

    summary=(df.groupby(['condition','method'])
             .agg(n=('root_pehe','size'),root_pehe_mean=('root_pehe','mean'),
                  root_pehe_sd=('root_pehe','std'),
                  mean_abs_tau_hat_mean=('mean_abs_tau_hat','mean'),
                  mean_abs_routed_correction_mean=('mean_abs_routed_correction','mean'),
                  mean_support_mean=('mean_support','mean'))
             .reset_index())
    summary.to_csv(out/'STAGE6A_METHOD_CONDITION_SUMMARY.csv',index=False)

    null=df[df.condition=='NULL_EFFECT_NEGATIVE_CONTROL']
    nonnull=df[df.condition!='NULL_EFFECT_NEGATIVE_CONTROL']
    null_mean=null.groupby('method').mean_abs_tau_hat.mean()
    nonnull_mean=nonnull.groupby('method').root_pehe.mean()
    corr_mean=nonnull.groupby('method').mean_abs_routed_correction.mean()

    null_guard_threshold=2.0*min(
        float(null_mean['X_learner_frozen_reference']),
        float(null_mean['T_learner_frozen_reference'])
    )
    x_nonnull=float(nonnull_mean['X_learner_frozen_reference'])
    cm_nonnull=float(nonnull_mean['CM_OSR_v1_frozen_reference'])

    eligible=['AOSR_X_gamma1','AOSR_X_gamma2','AOSR_R_gamma1','AOSR_R_gamma2']
    rows=[]
    for m in eligible:
        mn=float(nonnull_mean[m]); nu=float(null_mean[m])
        corr=float(corr_mean[m]) if m in corr_mean.index and np.isfinite(corr_mean[m]) else 0.0
        rows.append({
            'method':m,
            'null_mean_abs_tau':nu,
            'null_guard_threshold':null_guard_threshold,
            'gate1_null_guard':bool(nu <= null_guard_threshold),
            'non_null_mean_root_pehe':mn,
            'x_learner_non_null_mean_root_pehe':x_nonnull,
            'gate2_within_10pct_X':bool(mn <= 1.10*x_nonnull),
            'cm_osr_v1_non_null_mean_root_pehe':cm_nonnull,
            'gate3_better_than_CM_OSR_v1':bool(mn < cm_nonnull),
            'mean_abs_routed_correction':corr,
            'gate4_nonzero_routed_residual':bool(corr > 1e-4),
        })
    gates=pd.DataFrame(rows)
    gates['all_gates_pass']=gates[['gate1_null_guard','gate2_within_10pct_X','gate3_better_than_CM_OSR_v1','gate4_nonzero_routed_residual']].all(axis=1)
    gates.to_csv(out/'STAGE6A_LOCKED_SELECTION_GATES.csv',index=False)

    passing=gates[gates.all_gates_pass].copy()
    selected=None
    if len(passing):
        best=float(passing.non_null_mean_root_pehe.min())
        near=passing[passing.non_null_mean_root_pehe <= best+0.01].copy()
        near['gamma_pref']=near.method.map(lambda x: 0 if x.endswith('gamma1') else 1)
        near['anchor_pref']=near.method.map(lambda x: 0 if x.startswith('AOSR_X') else 1)
        near=near.sort_values(['gamma_pref','anchor_pref','non_null_mean_root_pehe'])
        selected=str(near.iloc[0].method)

    decision={
        'stage':'STAGE6A_Q1_METHOD_DEVELOPMENT',
        'grid_complete':True,'observed_pairs':len(expected),'required_pairs':len(expected),
        'candidate_selection_split':'DEV_ONLY',
        'test_accessed_for_selection':False,
        'stage3_primary_preserved':True,'stage5a_preserved':True,
        'null_guard_threshold':null_guard_threshold,
        'selected_candidate':selected,
        'status':'CANDIDATE_FROZEN_FOR_STAGE6B' if selected else 'NO_CANDIDATE_PASSED_LOCKED_STAGE6A_GATES',
        'next':'STAGE6B_CONFIRMATORY_FRESH_SEEDS_TEST_ONCE' if selected else 'NO_FORCED_METHOD_WINNER_PROCEED_WITH_BOUNDED_CM_OSR_CLAIMS_AND_REAL_TRIAL_VALIDATION',
    }
    (out/'STAGE6A_SELECTION.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')

    freeze={
        'stage':'STAGE6B_CONFIRMATORY',
        'status':'FROZEN_AFTER_STAGE6A_SELECTION',
        'selected_candidate':selected,
        'dgp_seeds':lock['confirmatory_stage6b_dgp_seeds_locked_now_do_not_run_in_stage6a'],
        'network_seeds':lock['confirmatory_stage6b_network_seeds_locked_now'],
        'evaluation_split':'TEST_ONCE','no_retuning':True,'no_seed_search':True,'no_TRIO_selection':True,
    }
    (out/'STAGE6B_FREEZE.yaml').write_text(yaml.safe_dump(freeze,sort_keys=False),encoding='utf-8')
    print(json.dumps(decision,indent=2))
    print('Stage6A aggregation complete:',out)

if __name__=='__main__':
    main()
