from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np,pandas as pd,yaml
B=10000

def pair_from_name(name):
    m=re.match(r'^CONF6B_DGP(\d+)_NET(\d+)_',name)
    return (int(m.group(1)),int(m.group(2))) if m else None

def cluster_boot(d,seed):
    cl=d.groupby('dgp_seed')['diff'].mean().to_numpy(); rng=np.random.default_rng(seed)
    boot=np.array([rng.choice(cl,size=len(cl),replace=True).mean() for _ in range(B)])
    return float(cl.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975)),len(cl)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args(); root=Path(a.root)
    lock=yaml.safe_load((root/'00_CONFIG'/'stage6b_confirmatory_lock.yaml').read_text())
    expected={(int(d),int(n)) for d in lock['confirmatory_dgp_seeds'] for n in lock['confirmatory_network_seeds']}
    complete={}; p=root/'08_CONFIRMATORY_RUNS'
    if not p.exists():raise SystemExit('Stage6B run folder missing.')
    for rd in p.glob('CONF6B_DGP*_NET*_*'):
        if not (rd/'COMPLETE.json').exists():continue
        pair=pair_from_name(rd.name)
        if pair and (pair not in complete or complete[pair].stat().st_mtime<rd.stat().st_mtime):complete[pair]=rd
    missing=sorted(expected-set(complete))
    if missing:raise SystemExit(f'Stage6B incomplete: {len(set(complete)&expected)}/{len(expected)}; missing={missing}')
    frames=[];sframes=[];cframes=[];mframes=[]
    for pair,rd in sorted(complete.items()):
        if pair not in expected:continue
        x=pd.read_csv(rd/'metrics'/'summary_test.csv');x['run_dir']=rd.name;frames.append(x)
        s=pd.read_csv(rd/'metrics'/'selective_test.csv');s['run_dir']=rd.name;sframes.append(s)
        c=pd.read_csv(rd/'metrics'/'coverage90_test.csv');c['run_dir']=rd.name;cframes.append(c)
        q=rd/'metrics'/'missingness_realized.csv'
        if q.exists():m=pd.read_csv(q);m['run_dir']=rd.name;mframes.append(m)
    df=pd.concat(frames,ignore_index=True);sel=pd.concat(sframes,ignore_index=True);cov=pd.concat(cframes,ignore_index=True)
    out=root/'06_RESULTS'/'STAGE6B';out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'ALL_STAGE6B_TEST_METRICS.csv',index=False);sel.to_csv(out/'ALL_STAGE6B_SELECTIVE_TEST.csv',index=False);cov.to_csv(out/'ALL_STAGE6B_COVERAGE90_TEST.csv',index=False)
    if mframes:pd.concat(mframes,ignore_index=True).to_csv(out/'ALL_STAGE6B_MISSINGNESS_REALIZED.csv',index=False)
    summ=df.groupby(['condition','method']).agg(n=('root_pehe','size'),root_pehe_mean=('root_pehe','mean'),root_pehe_sd=('root_pehe','std'),mean_abs_tau_hat_mean=('mean_abs_tau_hat','mean'),mean_abs_routed_correction_mean=('mean_abs_routed_correction','mean')).reset_index();summ.to_csv(out/'STAGE6B_METHOD_CONDITION_SUMMARY.csv',index=False)
    cand='AOSR_X_gamma2';x='X_learner_frozen_reference';t='T_learner_frozen_reference';cm='CM_OSR_v1_frozen_reference'
    non=df[df.condition!='NULL_EFFECT_NEGATIVE_CONTROL'];nul=df[df.condition=='NULL_EFFECT_NEGATIVE_CONTROL'];mn=non.groupby('method').root_pehe.mean();mu=nul.groupby('method').mean_abs_tau_hat.mean();corr=float(non[non.method==cand].mean_abs_routed_correction.mean());thr=2*min(float(mu[x]),float(mu[t]))
    comps=[cm,'early_fusion_DR_capacity_matched','generic_multimodal_attention_DR_capacity_matched','generic_MoE_DR_capacity_matched','CMOSR_support_shrink_gamma2_predeclared_control',x,t,'R_learner_frozen_reference']
    piv=non.pivot_table(index=['dgp_seed','net_seed','condition'],columns='method',values='root_pehe');rows=[]
    for comp in comps:
        d=(piv[cand]-piv[comp]).rename('diff').reset_index().groupby(['dgp_seed','net_seed'],as_index=False)['diff'].mean();mean,lo,hi,n=cluster_boot(d,int(lock['bootstrap']['seed']));rows.append({'comparator':comp,'mean_AOSR_minus_comparator':mean,'ci95_low':lo,'ci95_high':hi,'n_dgp_clusters':n})
    pair=pd.DataFrame(rows);pair.to_csv(out/'STAGE6B_PAIRED_CLUSTER_BOOTSTRAP.csv',index=False)
    ss=sel[(sel.method==cand)&(sel.condition!='NULL_EFFECT_NEGATIVE_CONTROL')&(sel.retained.isin([0.5,1.0]))];sp=ss.pivot_table(index=['dgp_seed','net_seed','condition'],columns='retained',values='root_pehe');sd=(sp[0.5]-sp[1.0]).rename('diff').reset_index().groupby(['dgp_seed','net_seed'],as_index=False)['diff'].mean();sm,slo,shi,sn=cluster_boot(sd,int(lock['bootstrap']['seed'])+1);pd.DataFrame([{'mean_retained50_minus_100':sm,'ci95_low':slo,'ci95_high':shi,'n_dgp_clusters':sn}]).to_csv(out/'STAGE6B_SELECTIVE_GATE.csv',index=False)
    cov.groupby(['condition','method']).agg(n=('coverage','size'),coverage_mean=('coverage','mean'),coverage_sd=('coverage','std'),mean_width=('mean_width','mean')).reset_index().to_csv(out/'STAGE6B_COVERAGE_SUMMARY.csv',index=False)
    def upper(c):return float(pair.loc[pair.comparator==c,'ci95_high'].iloc[0])
    gates={'gate1_null_guard':bool(float(mu[cand])<=thr),'gate2_within_10pct_X':bool(float(mn[cand])<=1.10*float(mn[x])),'gate3_better_than_CM_OSR_ci':bool(upper(cm)<0),'gate4_better_than_early_fusion_ci':bool(upper('early_fusion_DR_capacity_matched')<0),'gate4_better_than_generic_attention_ci':bool(upper('generic_multimodal_attention_DR_capacity_matched')<0),'gate4_better_than_generic_MoE_ci':bool(upper('generic_MoE_DR_capacity_matched')<0),'gate5_support_selectivity_ci':bool(shi<0),'gate6_nonzero_routed_correction':bool(corr>1e-4)};gates['all_confirmatory_gates_pass']=bool(all(gates.values()))
    dec={'stage':'STAGE6B_CONFIRMATORY','grid_complete':True,'observed_pairs':len(expected),'required_pairs':len(expected),'selected_candidate':cand,'evaluation_split':'TEST_ONCE','candidate_retuned':False,'null_guard_threshold':thr,'candidate_null_mean_abs_tau':float(mu[cand]),'candidate_non_null_mean_root_pehe':float(mn[cand]),'x_learner_non_null_mean_root_pehe':float(mn[x]),'mean_abs_routed_correction_non_null':corr,'gates':gates,'status':'CONFIRMATORY_PASS' if gates['all_confirmatory_gates_pass'] else 'CONFIRMATORY_MIXED_OR_FAIL','next':'MECHANISM_ABLATIONS_FOR_FROZEN_AOSR_THEN_CHER_LOB_THEN_LOCKED_TRIO' if gates['all_confirmatory_gates_pass'] else 'STOP_STRONG_METHOD_CLAIM_AND_REVIEW_FAILED_GATES_BEFORE_REAL_TRIAL'}
    (out/'STAGE6B_CONFIRMATORY_DECISION.json').write_text(json.dumps(dec,indent=2));print(json.dumps(dec,indent=2));print('Stage6B aggregation complete:',out)
if __name__=='__main__':main()

