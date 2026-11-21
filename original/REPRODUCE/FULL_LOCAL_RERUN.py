from pathlib import Path
import shutil, subprocess, sys, json
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'REPRO_WORK'/'B01'
if not WORK.exists():
    for rel in ['00_CONFIG','01_CODE','02_DATA','03_STATE']:
        src=ROOT/rel
        if src.exists(): shutil.copytree(src,WORK/rel,dirs_exist_ok=True)
    shutil.copy2(ROOT/'requirements.txt',WORK/'requirements.txt')
DATA=WORK/'02_DATA/processed/ispy2_core_multimodal.npz'; PY=sys.executable

def run(args):
    print('>',*map(str,args),flush=True); subprocess.run([str(x) for x in args],cwd=WORK,check=True)

def stage(script,seeds,seedflag):
    for d,s in seeds:
        run([PY,WORK/'01_CODE/scripts'/script,'--root',WORK,'--data',DATA,'--dgp-seed',d,seedflag,s])

stage('run_full_core_benchmark.py',[(d,s) for d in [101,211,307,401,503] for s in [17,29,43]],'--net-seed')
run([PY,WORK/'01_CODE/scripts/aggregate_core_results.py','--root',WORK])
stage('run_stage5a_secondary_stress.py',[(d,s) for d in [101,211,307,401,503] for s in [17,29,43]],'--net-seed')
run([PY,WORK/'01_CODE/scripts/aggregate_stage5a_results.py','--root',WORK])
stage('run_stage6a_q1_development.py',[(d,s) for d in [607,719,823,941,1061] for s in [19,37]],'--net-seed')
run([PY,WORK/'01_CODE/scripts/aggregate_stage6a_q1_results.py','--root',WORK])
stage('run_stage6b_confirmatory.py',[(d,s) for d in [1201,1327,1451,1579,1699] for s in [17,29,43]],'--net-seed')
run([PY,WORK/'01_CODE/scripts/aggregate_stage6b_confirmatory.py','--root',WORK])
stage('run_stage6c_mechanism.py',[(d,s) for d in [1811,1931,2053,2179,2297] for s in [23,41]],'--net-seed')
run([PY,WORK/'01_CODE/scripts/aggregate_stage6c_mechanism.py','--root',WORK])
stage('run_stage7b_cucmr_development.py',[(d,s) for d in [2411,2539,2671,2797,2909,2963] for s in [31,47]],'--learner-seed')
run([PY,WORK/'01_CODE/scripts/aggregate_stage7b_cucmr_development.py','--root',WORK])
print('Full local rerun complete:',WORK)
