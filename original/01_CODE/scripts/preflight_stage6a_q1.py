from pathlib import Path
import ast, hashlib, sys, yaml

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "00_CONFIG" / "stage6a_q1_method_development_lock.yaml"
CODE = ROOT / "01_CODE" / "cm_osr" / "q1_candidates.py"

def sha256(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main():
    print("B01 root:", ROOT)
    print("Stage6A Q1 method-development PRE-FLIGHT ONLY")
    print("No training. No Kaggle. No Stage3/Stage5A rerun.")
    if not LOCK.exists(): raise SystemExit("MISSING: "+str(LOCK))
    if not CODE.exists(): raise SystemExit("MISSING: "+str(CODE))
    cfg=yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    assert cfg["status"]=="LOCKED_BEFORE_STAGE6A_RESULTS"
    assert cfg["test_split_access_in_stage6a"] is False
    assert cfg["primary_stage3_stage5a_results"]=="FROZEN_AND_PRESERVED"
    assert len(cfg["development_dgp_seeds"])==len(set(cfg["development_dgp_seeds"]))
    assert not set(cfg["development_dgp_seeds"]) & set(cfg["confirmatory_stage6b_dgp_seeds_locked_now_do_not_run_in_stage6a"])
    ast.parse(CODE.read_text(encoding="utf-8"))
    print("LOCK PASS")
    print("DEV-only policy PASS")
    print("Stage6B confirmatory seeds isolated PASS")
    print("Candidate code syntax PASS")
    print("LOCK SHA256:",sha256(LOCK))
    print("CODE SHA256:",sha256(CODE))
    print("STAGE6A Q1 PRE-FLIGHT PASS")

if __name__=="__main__":
    main()
