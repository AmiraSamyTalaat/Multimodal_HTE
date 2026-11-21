
from pathlib import Path
import ast, json, yaml

ROOT=Path.cwd()

def main():
    req=[
        ROOT/"00_CONFIG"/"stage6c_mechanism_lock.yaml",
        ROOT/"00_CONFIG"/"project.yaml",
        ROOT/"03_STATE"/"EXPERIMENT_CONTRACT.yaml",
        ROOT/"01_CODE"/"scripts"/"run_stage6c_mechanism.py",
        ROOT/"01_CODE"/"scripts"/"aggregate_stage6c_mechanism.py",
        ROOT/"01_CODE"/"cm_osr"/"q1_candidates.py",
        ROOT/"02_DATA"/"processed"/"ispy2_core_multimodal.npz",
        ROOT/"06_RESULTS"/"STAGE6B"/"STAGE6B_CONFIRMATORY_DECISION.json",
    ]
    for p in req:
        if not p.exists(): raise SystemExit("MISSING: "+str(p))

    lock=yaml.safe_load((ROOT/"00_CONFIG"/"stage6c_mechanism_lock.yaml").read_text(encoding="utf-8"))
    exp=yaml.safe_load((ROOT/"03_STATE"/"EXPERIMENT_CONTRACT.yaml").read_text(encoding="utf-8"))
    dec=json.loads((ROOT/"06_RESULTS"/"STAGE6B"/"STAGE6B_CONFIRMATORY_DECISION.json").read_text(encoding="utf-8"))
    assert lock["status"]=="LOCKED_BEFORE_STAGE6C_RESULTS"
    assert lock["test_policy"]=="TEST_FORBIDDEN_STAGE6C"
    assert dec["selected_candidate"]=="AOSR_X_gamma2"
    assert dec["candidate_retuned"] is False
    assert set(lock["regimes"]).issubset(set(exp["DGP_regimes_prespecified"]))
    for p in [ROOT/"01_CODE"/"scripts"/"run_stage6c_mechanism.py",
              ROOT/"01_CODE"/"scripts"/"aggregate_stage6c_mechanism.py"]:
        ast.parse(p.read_text(encoding="utf-8"))
    print("STAGE6C LOCK PASS")
    print("Frozen AOSR candidate PASS")
    print("Originally prespecified unused DGP regimes PASS")
    print("TEST forbidden PASS")
    print("No candidate reselection PASS")
    print("Runner/Aggregator syntax PASS")
    print("STAGE6C MECHANISM PREFLIGHT PASS")

if __name__=="__main__":
    main()
