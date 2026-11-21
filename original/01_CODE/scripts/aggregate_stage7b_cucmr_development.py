from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd, yaml
from cmath import nan

CANDS=['TRANSCRIPTOMIC','RPPA','JOINT_MOLECULAR']
INC=['TRANSCRIPTOME_INCREMENTAL','RPPA_INCREMENTAL','JOINT_MOLECULAR_SYNERGY','PATIENT_SUBGROUP_SWITCHING']

def boot_lcb(vals,B,seed,alpha=.05):
    vals=np.asarray(vals,dtype=float); rng=np.random.default_rng(seed)
    b=np.asarray([rng.choice(vals,size=len(vals),replace=True).mean() for _ in range(B)])
    return float(vals.mean()),float(np.quantile(b,alpha))

def boot_ci(vals,B,seed):
    vals=np.asarray(vals,dtype=float); rng=np.random.default_rng(seed)
    b=np.asarray([rng.choice(vals,size=len(vals),replace=True).mean() for _ in range(B)])
    return float(vals.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args(); root=Path(a.root)
    lock=yaml.safe_load((root/'00_CONFIG'/'stage7a_cucmr_execution_lock_v4.yaml').read_text(encoding='utf-8'))
    expected={(int(d),int(s)) for d in lock['development']['dgp_seeds'] for s in lock['development']['learner_seeds']}
    pat=re.compile(r'^DEV7_DGP(\d+)_L(\d+)_'); complete={}
    base=root/'08_STAGE7_DEV_RUNS'
    if not base.exists(): raise SystemExit('No Stage7B run folder.')
    for rd in base.iterdir():
        m=pat.match(rd.name)
        if not m or not (rd/'COMPLETE.json').exists(): continue
        pair=(int(m.group(1)),int(m.group(2)))
        if pair in expected and (pair not in complete or complete[pair].stat().st_mtime<rd.stat().st_mtime): complete[pair]=rd
    miss=sorted(expected-set(complete))
    if miss: raise SystemExit(f'Stage7B incomplete {len(complete)}/{len(expected)} missing={miss}')
    frames=[]; mframes=[]; dframes=[]
    for pair,rd in sorted(complete.items()):
        frames.append(pd.read_csv(rd/'metrics'/'patient_dev_predictions.csv'))
        mframes.append(pd.read_csv(rd/'metrics'/'baseline_dev_metrics.csv'))
        dframes.append(pd.read_csv(rd/'metrics'/'fit_diagnostics.csv'))
    df=pd.concat(frames,ignore_index=True); bm=pd.concat(mframes,ignore_index=True); dg=pd.concat(dframes,ignore_index=True)
    out=root/'06_RESULTS'/'STAGE7B'; out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'ALL_STAGE7B_PATIENT_DEV_PREDICTIONS.csv',index=False); bm.to_csv(out/'ALL_STAGE7B_BASELINE_DEV_METRICS.csv',index=False); dg.to_csv(out/'ALL_STAGE7B_FIT_DIAGNOSTICS.csv',index=False)
    B=int(lock['development_certification']['bootstrap_B']); seed=int(lock['development_certification']['bootstrap_seed'])
    coverages=[float(x) for x in lock['development_certification']['candidate_acceptance_coverages']]
    selected={}; cert_rows=[]
    for ci,c in enumerate(CANDS):
        picked=None
        for q in coverages: # already largest -> smallest
            thr=1.0-q
            x=df[df[f'utility_pct_{c}']>=thr]
            per=(x.groupby(['dgp_seed','learner_seed'],as_index=False)[f'D_{c}'].mean()
                   .groupby('dgp_seed',as_index=False)[f'D_{c}'].mean())
            if len(per)==len(lock['development']['dgp_seeds']):
                mean,lcb=boot_lcb(per[f'D_{c}'].to_numpy(),B,seed+ci*10+int(q*100))
            else:
                mean,lcb=float('nan'),float('nan')
            cert_rows.append({'candidate':c,'coverage':q,'percentile_threshold':thr,'mean_D':mean,'one_sided_95_lcb':lcb,'n_dgp_clusters':len(per),'pass':bool(np.isfinite(lcb) and lcb>0)})
            if picked is None and np.isfinite(lcb) and lcb>0:
                picked={'coverage':q,'percentile_threshold':thr,'mean_D':mean,'lcb':lcb}
        selected[c]=picked
    pd.DataFrame(cert_rows).to_csv(out/'STAGE7B_UTILITY_CERTIFICATION.csv',index=False)

    # Reconstruct frozen-development CUCMR rule.
    preds=[]
    for _,r in df.iterrows():
        eligible=[]
        for c in CANDS:
            sel=selected[c]
            if sel is not None and float(r[f'utility_pct_{c}'])>=float(sel['percentile_threshold']):
                eligible.append(c)
        if eligible:
            c=max(eligible,key=lambda x: float(r[f'utility_{x}']))
            tau=float(r[f'tau_{c}']); src={'TRANSCRIPTOMIC':1,'RPPA':2,'JOINT_MOLECULAR':3}[c]
        else:
            c='ANCHOR'; tau=float(r['anchor_tau']); src=0
        z=dict(r); z['cucmr_tau']=tau; z['selected_candidate']=c; z['selected_source']=src; z['accepted_update']=int(src!=0); preds.append(z)
    qdf=pd.DataFrame(preds); qdf.to_csv(out/'STAGE7B_CUCMR_DEV_PREDICTIONS_FROZEN_RULE.csv',index=False)

    rows=[]
    methods=['CUCMR','FIXED_CLINICAL_X_ANCHOR','FULL_MULTIMODAL_X_LEARNER','DR_LOSS_GLOBAL_STACKING','TRAIN_OOF_GLOBAL_BEST_SINGLE_UPDATE']
    col={'CUCMR':'cucmr_tau','FIXED_CLINICAL_X_ANCHOR':'anchor_tau','FULL_MULTIMODAL_X_LEARNER':'full_x_tau','DR_LOSS_GLOBAL_STACKING':'stack_tau','TRAIN_OOF_GLOBAL_BEST_SINGLE_UPDATE':'global_best_tau'}
    for (d,s,reg),g in qdf.groupby(['dgp_seed','learner_seed','regime']):
        true=g['true_tau'].to_numpy()
        for m in methods:
            p=g[col[m]].to_numpy(); rows.append({'dgp_seed':d,'learner_seed':s,'regime':reg,'method':m,'root_pehe':float(np.sqrt(np.mean((p-true)**2)))})
    met=pd.DataFrame(rows); met.to_csv(out/'STAGE7B_METHOD_REGIME_METRICS.csv',index=False)

    def mean_metric(method,regs):
        x=met[(met.method==method)&met.regime.isin(regs)]; per=x.groupby('dgp_seed',as_index=False).root_pehe.mean(); return float(per.root_pehe.mean())
    inc_c=mean_metric('CUCMR',INC); inc_a=mean_metric('FIXED_CLINICAL_X_ANCHOR',INC)
    sw_c=mean_metric('CUCMR',['PATIENT_SUBGROUP_SWITCHING']); sw_x=mean_metric('FULL_MULTIMODAL_X_LEARNER',['PATIENT_SUBGROUP_SWITCHING'])
    neg=qdf[qdf.regime=='NO_INCREMENTAL_MOLECULAR_SIGNAL']; neg_acc=float(neg.accepted_update.mean())
    neg_c=mean_metric('CUCMR',['NO_INCREMENTAL_MOLECULAR_SIGNAL']); neg_a=mean_metric('FIXED_CLINICAL_X_ANCHOR',['NO_INCREMENTAL_MOLECULAR_SIGNAL'])
    sw=qdf[(qdf.regime=='PATIENT_SUBGROUP_SWITCHING')&(qdf.accepted_update==1)]
    correct=float((sw.selected_source==sw.true_source).mean()) if len(sw) else 0.0
    incdf=qdf[qdf.regime.isin(INC)]; inc_acc=float(incdf.accepted_update.mean())
    certified=any(v is not None for v in selected.values())
    gates={
        'D1_incremental_vs_anchor': bool(inc_c<=inc_a),
        'D2_switching_vs_fullX': bool(sw_c<=sw_x),
        'D3_negative_control': bool(neg_acc<=0.15 and neg_c<=1.05*neg_a),
        'D4_switching_localization': bool(correct>=0.55),
        'D5_nontrivial_use': bool(inc_acc>=0.15),
        'D6_at_least_one_certified_candidate': bool(certified),
    }
    status=lock['development_freeze_rule']['pass_status'] if all(gates.values()) else lock['development_freeze_rule']['fail_status']
    selection={'stage':'STAGE7B_CUCMR_DEVELOPMENT','grid_complete':True,'observed_pairs':len(expected),'required_pairs':len(expected),
               'fit_split':'TRAIN_ONLY','evaluation_split':'DEV_ONLY','test_accessed':False,
               'selected_certification':selected,'metrics':{'incremental_cucmr_root_pehe':inc_c,'incremental_anchor_root_pehe':inc_a,
               'switching_cucmr_root_pehe':sw_c,'switching_fullX_root_pehe':sw_x,'negative_control_acceptance_rate':neg_acc,
               'negative_control_cucmr_root_pehe':neg_c,'negative_control_anchor_root_pehe':neg_a,
               'switching_correct_source_rate_among_accepted':correct,'incremental_acceptance_rate':inc_acc},
               'gates':gates,'status':status,
               'next':'STAGE7C_CONFIRMATORY_WITH_RESERVED_SEEDS' if all(gates.values()) else 'STOP_BEFORE_CONFIRMATORY_OR_BEGIN_A_NEW_PRELOCKED_DEVELOPMENT_CYCLE'}
    (out/'STAGE7B_DEVELOPMENT_SELECTION.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(json.dumps(selection,indent=2)); print('Stage7B aggregation complete:',out)

if __name__=='__main__': main()
