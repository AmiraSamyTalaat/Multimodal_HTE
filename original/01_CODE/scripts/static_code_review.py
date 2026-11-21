from pathlib import Path
import json, re
ROOT=Path(__file__).resolve().parents[2]
checks={}
contract=(ROOT/'03_STATE'/'EXPERIMENT_CONTRACT.yaml').read_text(encoding='utf-8')
checks['contract_locked']='status: LOCKED_PRE_IMPLEMENTATION' in contract
checks['test_tuning_forbidden']='changing_primary_endpoint' in contract and 'threshold_tuning_on_test_or_external' in contract
checks['dgp_lock_exists']=(ROOT/'00_CONFIG'/'dgp_lock.yaml').exists()
checks['raw_predictions_directory']=(ROOT/'06_RESULTS').exists()
checks['checkpoint_directory']=(ROOT/'05_CHECKPOINTS').exists()
checks['state_exists']=(ROOT/'03_STATE'/'STATE.json').exists()
print(json.dumps(checks,indent=2))
if not all(checks.values()): raise SystemExit(2)
