from pathlib import Path
import argparse, json, numpy as np, pandas as pd


def bci_cluster(x,seed=20260919,B=10000):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); boot=np.empty(B)
    for i in range(B): boot[i]=rng.choice(x,len(x),replace=True).mean()
    return {'mean':float(x.mean()),'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975)),'n_clusters':int(len(x)),'bootstrap_B':B}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args(); root=Path(a.root); rows=[]
    for mf in (root/'04_RUNS').glob('CORE_DGP*_NET*/RUN_MANIFEST.json'):
        m=json.loads(mf.read_text(encoding='utf-8'))
        if m.get('status')!='COMPLETE': continue
        sf=mf.parent/'metrics'/'summary.csv'
        if not sf.exists(): continue
        d=pd.read_csv(sf); cfg=m.get('config',{}); d['run']=m.get('run_id',mf.parent.name); d['dgp_seed']=int(cfg['dgp_seed']); d['net_seed']=int(cfg['net_seed']); rows.append(d)
    if not rows: raise SystemExit('No COMPLETE CORE benchmark runs found.')
    allm=pd.concat(rows,ignore_index=True); allm.to_csv(root/'06_RESULTS'/'ALL_CORE_METRICS.csv',index=False)
    w=allm.pivot_table(index=['dgp_seed','net_seed'],columns='method',values='root_pehe',aggfunc='first').reset_index()
    req=['CM_OSR','early_fusion_DR_capacity_matched']
    if any(c not in w.columns for c in req): raise SystemExit(f'Missing primary methods: {req}')
    w['delta_CM_minus_early']=w['CM_OSR']-w['early_fusion_DR_capacity_matched']; w.to_csv(root/'06_RESULTS'/'PRIMARY_PAIRED_REPLICATES.csv',index=False)
    cl=w.groupby('dgp_seed',as_index=False)['delta_CM_minus_early'].mean(); cl.to_csv(root/'06_RESULTS'/'PRIMARY_DGP_CLUSTER_DELTAS.csv',index=False)
    ci=bci_cluster(cl['delta_CM_minus_early'].to_numpy()); ci['comparison']='CM_OSR - early_fusion_DR_capacity_matched root_PEHE'; ci['direction']='negative_favors_CM_OSR'; ci['required_seed_pairs']=15; ci['observed_seed_pairs']=int(len(w)); ci['required_dgp_clusters']=5; ci['observed_dgp_clusters']=int(len(cl))
    complete=(len(w)==15 and len(cl)==5); ci['grid_complete']=bool(complete); ci['primary_gate']='PASS' if complete and ci['mean']<0 and ci['ci95_high']<0 else ('UNRESOLVED' if complete else 'INCOMPLETE')
    (root/'06_RESULTS'/'PRIMARY_GATE.json').write_text(json.dumps(ci,indent=2),encoding='utf-8'); print(json.dumps(ci,indent=2))
if __name__=='__main__': main()
