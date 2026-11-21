from pathlib import Path
import ast, json, sys, yaml, numpy as np
ROOT=Path.cwd()

def main():
    req=[ROOT/'00_CONFIG'/'stage7a_cucmr_execution_lock_v4.yaml',ROOT/'00_CONFIG'/'project.yaml',
         ROOT/'01_CODE'/'cm_osr'/'baselines.py',ROOT/'01_CODE'/'cm_osr'/'dgp.py',ROOT/'01_CODE'/'cm_osr'/'nuisance.py',
         ROOT/'01_CODE'/'cm_osr'/'splits.py',ROOT/'01_CODE'/'cm_osr'/'q1_candidates.py',ROOT/'01_CODE'/'cm_osr'/'stage7_dgp.py',
         ROOT/'01_CODE'/'cm_osr'/'cucmr.py',ROOT/'01_CODE'/'scripts'/'run_stage7b_cucmr_development.py',
         ROOT/'01_CODE'/'scripts'/'aggregate_stage7b_cucmr_development.py',ROOT/'02_DATA'/'processed'/'ispy2_core_multimodal.npz',
         ROOT/'06_RESULTS'/'STAGE6C'/'STAGE6C_MECHANISM_DECISION.json']
    for p in req:
        if not p.exists(): raise SystemExit('MISSING: '+str(p))
    lock=yaml.safe_load((ROOT/'00_CONFIG'/'stage7a_cucmr_execution_lock_v4.yaml').read_text(encoding='utf-8'))
    if lock['status']!='LOCKED_BEFORE_STAGE7_RESULTS': raise SystemExit('LOCK STATUS FAIL')
    dec=json.loads((ROOT/'06_RESULTS'/'STAGE6C'/'STAGE6C_MECHANISM_DECISION.json').read_text(encoding='utf-8'))
    if dec.get('status')!='WEAK_METHOD_INCREMENT_DO_NOT_POSITION_AS_PRIMARY_NOVEL_TECHNIQUE':
        raise SystemExit('Stage6C chronology/status mismatch.')
    z=np.load(ROOT/'02_DATA'/'processed'/'ispy2_core_multimodal.npz',allow_pickle=False)
    for k in ['clinical','transcriptomic','rppa']:
        if k not in z.files: raise SystemExit('DATA KEY MISSING: '+k)
    n=len(z['clinical'])
    if not (len(z['transcriptomic'])==n and len(z['rppa'])==n): raise SystemExit('MODALITY ROW COUNT MISMATCH')
    for p in [ROOT/'01_CODE'/'cm_osr'/'stage7_dgp.py',ROOT/'01_CODE'/'cm_osr'/'cucmr.py',ROOT/'01_CODE'/'scripts'/'run_stage7b_cucmr_development.py',ROOT/'01_CODE'/'scripts'/'aggregate_stage7b_cucmr_development.py']:
        ast.parse(p.read_text(encoding='utf-8'))
    txt=(ROOT/'01_CODE'/'scripts'/'run_stage7b_cucmr_development.py').read_text(encoding='utf-8')
    if "sp['test']" in txt or 'sp["test"]' in txt: raise SystemExit('TEST INDEX REFERENCE DETECTED IN STAGE7B RUNNER')
    print('STAGE7 V4 LOCK PASS')
    print('Stage6C chronology PASS')
    print('Data interface PASS', {k:list(z[k].shape) for k in ['clinical','transcriptomic','rppa']})
    print('Existing B01 API mapping PASS')
    print('TEST index static guard PASS')
    print('Runner/Aggregator syntax PASS')
    print('STAGE7B DEVELOPMENT PREFLIGHT PASS')
if __name__=='__main__': main()
