from pathlib import Path
import json,yaml,ast,hashlib
ROOT = Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
    lockp=ROOT/'00_CONFIG'/'stage6b_confirmatory_lock.yaml';selp=ROOT/'06_RESULTS'/'STAGE6A'/'STAGE6A_SELECTION.json';runp=ROOT/'01_CODE'/'scripts'/'run_stage6b_confirmatory.py';aggp=ROOT/'01_CODE'/'scripts'/'aggregate_stage6b_confirmatory.py'
    for p in [lockp,selp,runp,aggp]:
        if not p.exists():raise SystemExit('MISSING: '+str(p))
    lock=yaml.safe_load(lockp.read_text(encoding='utf-8'));sel=json.loads(selp.read_text(encoding='utf-8'))
    assert lock['status']=='LOCKED_BEFORE_STAGE6B_RESULTS';assert lock['selected_candidate']=='AOSR_X_gamma2';assert sel['selected_candidate']=='AOSR_X_gamma2';assert sel['status']=='CANDIDATE_FROZEN_FOR_STAGE6B';assert sel['candidate_selection_split']=='DEV_ONLY';assert sel['test_accessed_for_selection'] is False;assert not set(lock['development_dgp_seeds_not_reused']) & set(lock['confirmatory_dgp_seeds'])
    ast.parse(runp.read_text(encoding='utf-8'));ast.parse(aggp.read_text(encoding='utf-8'))
    print('STAGE6B LOCK PASS');print('Frozen candidate PASS: AOSR_X_gamma2');print('Fresh confirmatory DGP seeds PASS');print('TEST-once policy PASS');print('No candidate retuning PASS');print('Runner/Aggregator syntax PASS');print('LOCK SHA256:',sha(lockp));print('STAGE6B CONFIRMATORY PREFLIGHT PASS')
if __name__=='__main__':main()
