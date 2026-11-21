from pathlib import Path
import argparse, json, numpy as np, pandas as pd

def bootstrap_ci(x,seed=20260918,B=10000):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); b=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(B)])
    return float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args(); root=Path(a.root)
    rows=[]
    for f in (root/'04_RUNS').glob('SEMISYN_*/metrics/summary.csv'):
        d=pd.read_csv(f); d['run']=f.parents[1].name; rows.append(d)
    if not rows: raise SystemExit('No benchmark summaries found.')
    all=pd.concat(rows,ignore_index=True); wide=all.pivot(index='run',columns='method',values='root_pehe').dropna()
    if 'CM_OSR' in wide and 'DR_learner' in wide:
        delta=wide['CM_OSR']-wide['DR_learner']; print('CM_OSR - DR_learner root_PEHE:',bootstrap_ci(delta))
    all.to_csv(root/'06_RESULTS'/'ALL_SEED_METRICS.csv',index=False)
if __name__=='__main__': main()
