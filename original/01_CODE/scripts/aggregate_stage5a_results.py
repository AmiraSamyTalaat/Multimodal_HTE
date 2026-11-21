
from pathlib import Path
import argparse, json, re
import numpy as np, pandas as pd

EXPECTED={(d,n) for d in [101,211,307,401,503] for n in [17,29,43]}
B=10000
BOOT_SEED=20260920

def pair_from_name(name):
    m=re.match(r'^STRESS_DGP(\d+)_NET(\d+)_',name)
    return (int(m.group(1)),int(m.group(2))) if m else None

def collect(root,filename):
    frames=[]
    for rd in (root/'05_STRESS_RUNS').glob('STRESS_DGP*_NET*_*'):
        if not (rd/'COMPLETE.json').exists(): continue
        p=rd/'metrics'/filename
        if p.exists():
            x=pd.read_csv(p); x['run_dir']=rd.name; frames.append(x)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def cluster_boot(diff_df,value_col):
    cl=diff_df.groupby('dgp_seed')[value_col].mean().to_numpy()
    rng=np.random.default_rng(BOOT_SEED)
    boots=np.array([rng.choice(cl,size=len(cl),replace=True).mean() for _ in range(B)])
    return float(np.mean(cl)),float(np.quantile(boots,.025)),float(np.quantile(boots,.975)),len(cl)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args()
    root=Path(a.root); out=root/'06_RESULTS'/'STAGE5A'; out.mkdir(parents=True,exist_ok=True)

    complete={}
    for rd in (root/'05_STRESS_RUNS').glob('STRESS_DGP*_NET*_*'):
        if (rd/'COMPLETE.json').exists():
            pair=pair_from_name(rd.name)
            if pair: complete[pair]=rd
    missing=sorted(EXPECTED-set(complete))
    if missing: raise SystemExit(f'Stage5A grid incomplete: {len(complete)}/15; missing={missing}')

    metrics=collect(root,'summary.csv')
    sel=collect(root,'selective.csv')
    cov=collect(root,'coverage90.csv')
    ovs=collect(root,'overlap_strata.csv')
    miss=collect(root,'missingness_realized.csv')
    metrics.to_csv(out/'ALL_STAGE5A_METRICS.csv',index=False)
    sel.to_csv(out/'ALL_STAGE5A_SELECTIVE.csv',index=False)
    cov.to_csv(out/'ALL_STAGE5A_COVERAGE90.csv',index=False)
    ovs.to_csv(out/'ALL_STAGE5A_OVERLAP_STRATA.csv',index=False)
    miss.to_csv(out/'ALL_STAGE5A_MISSINGNESS_REALIZED.csv',index=False)

    summary=(metrics.groupby(['family','condition','method'])
             .agg(n=('root_pehe','size'),root_pehe_mean=('root_pehe','mean'),root_pehe_sd=('root_pehe','std'),
                  policy_regret_mean=('policy_regret','mean'),calibration_error_mean=('cate_calibration_error','mean'),
                  mean_abs_tau_hat_mean=('mean_abs_tau_hat','mean')).reset_index())
    summary.to_csv(out/'METHOD_STRESS_SUMMARY.csv',index=False)

    # Pairwise direct performance under stress, clustered by DGP seed.
    targets=[
        'early_fusion_DR_capacity_matched',
        'generic_multimodal_attention_DR_capacity_matched',
        'generic_MoE_DR_capacity_matched',
        'generic_attention_same_model_availability_masked_inference',
        'generic_MoE_same_model_availability_masked_inference',
        'X_learner','T_learner_regularized','R_learner'
    ]
    pair_rows=[]
    for (fam,cond),g in metrics.groupby(['family','condition']):
        piv=g.pivot_table(index=['dgp_seed','net_seed'],columns='method',values='root_pehe',aggfunc='mean')
        if 'CM_OSR' not in piv: continue
        for comp in targets:
            if comp not in piv: continue
            d=(piv['CM_OSR']-piv[comp]).dropna().rename('diff').reset_index()
            if len(d)!=15: continue
            mean,lo,hi,ncl=cluster_boot(d,'diff')
            pair_rows.append({'family':fam,'condition':cond,'comparator':comp,'mean_CM_OSR_minus_comparator':mean,
                              'ci95_low':lo,'ci95_high':hi,'n_pairs':len(d),'n_dgp_clusters':ncl,
                              'negative_favors_CM_OSR':True})
    pairwise=pd.DataFrame(pair_rows)
    pairwise.to_csv(out/'H2_PAIRWISE_CLUSTER_BOOTSTRAP.csv',index=False)

    # Graceful-degradation analysis for missing-modality family against its own clean reference.
    mg=metrics[metrics.family=='missing_modality']
    clean=mg[mg.condition=='CLEAN_REFERENCE'][['dgp_seed','net_seed','method','root_pehe']].rename(columns={'root_pehe':'clean_root_pehe'})
    stress=mg[mg.condition!='CLEAN_REFERENCE'].merge(clean,on=['dgp_seed','net_seed','method'],how='left')
    stress['degradation']=stress['root_pehe']-stress['clean_root_pehe']
    stress.to_csv(out/'MISSINGNESS_DEGRADATION_PER_PAIR.csv',index=False)
    deg_rows=[]
    for cond in sorted(stress.condition.unique()):
        sg=stress[stress.condition==cond]
        piv=sg.pivot_table(index=['dgp_seed','net_seed'],columns='method',values='degradation',aggfunc='mean')
        if 'CM_OSR' not in piv: continue
        for comp in targets:
            if comp not in piv: continue
            d=(piv['CM_OSR']-piv[comp]).dropna().rename('diff').reset_index()
            if len(d)!=15: continue
            mean,lo,hi,ncl=cluster_boot(d,'diff')
            deg_rows.append({'condition':cond,'comparator':comp,'mean_delta_degradation_CM_OSR_minus_comparator':mean,
                             'ci95_low':lo,'ci95_high':hi,'n_pairs':len(d),'n_dgp_clusters':ncl,
                             'negative_means_CM_OSR_degrades_less':True})
    pd.DataFrame(deg_rows).to_csv(out/'H2_MISSINGNESS_DEGRADATION_CLUSTER_BOOTSTRAP.csv',index=False)

    sel_summary=(sel.groupby(['family','condition','method','retained']).agg(n=('root_pehe','size'),
                 root_pehe_mean=('root_pehe','mean'),root_pehe_sd=('root_pehe','std')).reset_index())
    sel_summary.to_csv(out/'SELECTIVE_SUMMARY.csv',index=False)

    cov_summary=(cov.groupby(['family','condition','method']).agg(n=('coverage','size'),
                 coverage_mean=('coverage','mean'),coverage_sd=('coverage','std'),
                 mean_width=('mean_width','mean')).reset_index())
    cov_summary.to_csv(out/'COVERAGE_SUMMARY.csv',index=False)

    ovs_summary=(ovs.groupby(['family','condition','stratum','method']).agg(n_runs=('root_pehe','size'),
                 root_pehe_mean=('root_pehe','mean'),root_pehe_sd=('root_pehe','std')).reset_index())
    ovs_summary.to_csv(out/'OVERLAP_STRATA_SUMMARY.csv',index=False)

    audit={
        'stage':'STAGE5A',
        'grid_complete':True,'observed_pairs':15,'required_pairs':15,
        'primary_grid_changed':False,
        'status':'EVIDENCE_READY_FOR_RESULTS_AUDITOR',
        'automatic_H2_pass_not_assigned':True,
        'reason':'H2 is a prespecified secondary robustness hypothesis without a single locked scalar pass threshold; report paired cluster intervals and selective/coverage evidence without promoting a new primary endpoint.',
        'next':'RESULTS_AUDITOR_STATISTICIAN_STAGE5A_THEN_STAGE5B_OR_REAL_TRIAL'
    }
    (out/'STAGE5A_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
    print('Stage5A aggregation complete:',out)

if __name__=='__main__': main()
