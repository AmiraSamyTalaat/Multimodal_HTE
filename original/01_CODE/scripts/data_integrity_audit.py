from pathlib import Path
import argparse, json, numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); a=p.parse_args(); z=np.load(a.data,allow_pickle=False)
    ks=['clinical','transcriptomic','rppa']; n=[z[k].shape[0] for k in ks]; checks={'modalities_present':all(k in z for k in ks),'same_n':len(set(n))==1,'n':n[0],'finite_fraction':{k:float(np.isfinite(z[k]).mean()) for k in ks}}
    print(json.dumps(checks,indent=2));
    if not checks['modalities_present'] or not checks['same_n']: raise SystemExit(2)
if __name__=='__main__': main()
