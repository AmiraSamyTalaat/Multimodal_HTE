
from pathlib import Path
import json, re, sys, yaml

ROOT=Path(__file__).resolve().parents[2]
EXPECTED={(d,n) for d in [101,211,307,401,503] for n in [17,29,43]}

def core_pairs():
    out=set()
    p=ROOT/'04_RUNS'
    if not p.exists(): return out
    pat=re.compile(r'^CORE_DGP(\d+)_NET(\d+)_')
    for d in p.iterdir():
        if not d.is_dir(): continue
        m=pat.match(d.name)
        if not m: continue
        if (d/'metrics'/'summary.csv').exists() and (d/'predictions'/'test_predictions.npz').exists() and (d/'RUN_MANIFEST.json').exists():
            out.add((int(m.group(1)),int(m.group(2))))
    return out

def main():
    failures=[]
    gate=ROOT/'06_RESULTS'/'PRIMARY_GATE.json'
    metrics=ROOT/'06_RESULTS'/'ALL_CORE_METRICS.csv'
    cfg=ROOT/'00_CONFIG'/'stage5a_stress_lock.yaml'
    if not gate.exists(): failures.append('PRIMARY_GATE.json missing')
    else:
        g=json.loads(gate.read_text(encoding='utf-8'))
        if g.get('primary_gate')!='PASS' or not g.get('grid_complete'): failures.append('primary gate is not PASS+complete')
    if not metrics.exists(): failures.append('ALL_CORE_METRICS.csv missing')
    pairs=core_pairs()
    if not EXPECTED.issubset(pairs): failures.append(f'primary 15/15 grid incomplete locally; missing={sorted(EXPECTED-pairs)}')
    if not cfg.exists(): failures.append('stage5a_stress_lock.yaml missing')
    else:
        c=yaml.safe_load(cfg.read_text(encoding='utf-8'))
        if c.get('status')!='LOCKED_BEFORE_STAGE5A_RESULTS': failures.append('Stage5A stress lock status invalid')
        if not c.get('primary_grid_rerun_forbidden'): failures.append('primary rerun prohibition missing')
    req=[
        ROOT/'01_CODE'/'scripts'/'run_stage5a_secondary_stress.py',
        ROOT/'01_CODE'/'scripts'/'aggregate_stage5a_results.py',
        ROOT/'01_CODE'/'scripts'/'kaggle_stage5a_autopilot.py',
        ROOT/'02_DATA'/'processed'/'ispy2_core_multimodal.npz',
        ROOT/'03_STATE'/'EXPERIMENT_CONTRACT.yaml'
    ]
    for p in req:
        if not p.exists(): failures.append(f'missing required file: {p.relative_to(ROOT)}')
    report={
        'status':'PASS' if not failures else 'FAIL',
        'failures':failures,
        'primary_pairs_found':len(EXPECTED & pairs),
        'primary_grid_rerun_forbidden':True,
        'scientific_training_started_by_preflight':False,
        'next':'RUN_STAGE5A_KAGGLE_AUTOPILOT_B01.cmd' if not failures else 'STOP_AND_FIX'
    }
    out=ROOT/'03_STATE'/'STAGE5A_PREFLIGHT.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if failures: sys.exit(2)
    print('STAGE 5A PREFLIGHT PASS. Primary grid remains frozen; secondary stress training is permitted.')

if __name__=='__main__': main()
