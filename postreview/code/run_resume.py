from __future__ import annotations
import argparse,json,shutil,subprocess,sys,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path

REV=Path(__file__).resolve().parents[1]
CODE=REV/'code'
KAGGLE_SEEDS=[101,211,307,401,503]
LOG=REV/'logs'/'resume_orchestrator.jsonl'
def now():return datetime.now(timezone.utc).isoformat()
def event(kind,**data):
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open('a',encoding='utf-8') as f:f.write(json.dumps({'time_utc':now(),'event':kind,**data},ensure_ascii=False)+'\n')
    print(kind,data,flush=True)
def call(args,timeout=None):
    p=subprocess.run([str(x) for x in args],capture_output=True,text=True,timeout=timeout)
    return p.returncode,(p.stdout+'\n'+p.stderr).strip()
def stage_complete(relative):
    p=REV/relative
    if not p.exists():return False
    try:return json.loads(p.read_text(encoding='utf-8')).get('status')=='COMPLETE'
    except:return False
def run_local(script,status_path):
    if stage_complete(status_path):event('local_stage_already_complete',script=script);return
    event('local_stage_started',script=script)
    code,output=call([sys.executable,CODE/script])
    (REV/'logs'/f'{Path(script).stem}_{int(time.time())}.log').write_text(output,encoding='utf-8')
    if code or not stage_complete(status_path):raise RuntimeError(f'{script} failed; see logs')
    event('local_stage_complete',script=script)
def kernel_name(seed):return f'b01-p1-postreview-fairness-dgp{seed}'
def kernel_status(seed):
    code,output=call(['kaggle','kernels','status',f'prinprin/{kernel_name(seed)}'],timeout=90)
    if code:
        return ('NOT_FOUND' if '404' in output or 'not found' in output.lower() else 'QUERY_ERROR'),output
    for value in ['COMPLETE','RUNNING','PENDING','ERROR','CANCELLED']:
        if f'KernelWorkerStatus.{value}' in output:return value,output
    return 'UNKNOWN',output
def local_kernel_complete(seed):
    folder=REV/'runs'/f'kaggle_fairness_dgp{seed}'
    status=folder/'postreview_results/RUN_STATUS.json'
    archive=folder/'B01_P1_BAGGED_FAIRNESS_OUTPUT.zip'
    if not status.exists() or not archive.exists():return False
    try:
        data=json.loads(status.read_text(encoding='utf-8'))
        if data.get('status')!='COMPLETE' or data.get('completed_pairs')!=3 or data.get('config',{}).get('mode')!='full':return False
        with zipfile.ZipFile(archive) as z:return z.testzip() is None
    except:return False
def prepare(seed):
    name=kernel_name(seed)
    code,output=call([sys.executable,CODE/'prepare_kaggle_fairness.py','--name',name,
                      '--dgp-seeds',seed,'--net-seeds',17,29,43,'--mode','full'])
    if code:raise RuntimeError(output)
    return REV/'kaggle_bundle'/name
def harvest(seed):
    folder=REV/'runs'/f'kaggle_fairness_dgp{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    code,output=call(['kaggle','kernels','output',f'prinprin/{kernel_name(seed)}','-p',folder,'-o'],timeout=600)
    (REV/'logs'/f'harvest_dgp{seed}_{int(time.time())}.log').write_text(output,encoding='utf-8')
    if code or not local_kernel_complete(seed):raise RuntimeError(f'Harvest/verification failed DGP {seed}: {output[-1000:]}')
    event('kaggle_batch_harvested',seed=seed,folder=str(folder))
def run_kaggle(max_retries=2):
    attempts={seed:0 for seed in KAGGLE_SEEDS}
    while True:
        pending=[s for s in KAGGLE_SEEDS if not local_kernel_complete(s)]
        if not pending:return
        running=0;progress=False
        for seed in pending:
            state,output=kernel_status(seed)
            if state=='QUERY_ERROR':event('kaggle_status_query_failed',seed=seed,error=output[-500:]);continue
            if state=='COMPLETE':harvest(seed);progress=True;continue
            if state in ('RUNNING','PENDING'):running+=1;continue
            if state in ('NOT_FOUND','ERROR','CANCELLED') and attempts[seed]<max_retries and running<2:
                if state in ('ERROR','CANCELLED'):
                    failure_dir=REV/'runs'/f'kaggle_fairness_dgp{seed}'/'failed_attempts'/str(int(time.time()))
                    failure_dir.mkdir(parents=True,exist_ok=True)
                    call(['kaggle','kernels','output',f'prinprin/{kernel_name(seed)}','-p',failure_dir,'-o'],timeout=300)
                folder=prepare(seed)
                code,response=call(['kaggle','kernels','push','-p',folder],timeout=120)
                if code==0:
                    attempts[seed]+=1;running+=1;progress=True
                    event('kaggle_batch_pushed',seed=seed,attempt=attempts[seed])
                elif 'Maximum batch GPU session count' in response:
                    running=2
                else:
                    attempts[seed]+=1
                    event('kaggle_push_failed',seed=seed,attempt=attempts[seed],error=response[-1000:])
            elif state not in ('RUNNING','PENDING'):
                raise RuntimeError(f'Kaggle batch DGP {seed} requires review: {state}: {output[-1000:]}')
        if not progress:time.sleep(30)
def run_single_kernel(kind,slug,script_name,output_dir,archive_name,max_retries=2):
    name=f'prinprin/{slug}'
    target=REV/'results'/kind
    if stage_complete(f'results/{kind}/RUN_STATUS.json'):return
    attempts=0
    while True:
        code,output=call(['kaggle','kernels','status',name],timeout=90)
        state=('NOT_FOUND' if '404' in output or 'not found' in output.lower() else 'QUERY_ERROR') if code else next((v for v in ['COMPLETE','RUNNING','PENDING','ERROR','CANCELLED']
                                if f'KernelWorkerStatus.{v}' in output),'UNKNOWN')
        if state=='QUERY_ERROR':event(f'kaggle_{kind}_status_query_failed',error=output[-500:]);time.sleep(30);continue
        if state=='COMPLETE':
            folder=REV/'runs'/f'kaggle_{kind}';folder.mkdir(parents=True,exist_ok=True)
            code,response=call(['kaggle','kernels','output',name,'-p',folder,'-o'],timeout=600)
            (REV/'logs'/f'harvest_{kind}_{int(time.time())}.log').write_text(response,encoding='utf-8')
            source=folder/output_dir
            archive=folder/archive_name
            if code or not (source/'RUN_STATUS.json').exists() or not archive.exists():
                raise RuntimeError(f'{kind} Kaggle output harvest failed: '+response[-1000:])
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None:raise RuntimeError(f'{kind} ZIP integrity failed')
            shutil.copytree(source,target,dirs_exist_ok=True)
            if not stage_complete(f'results/{kind}/RUN_STATUS.json'):
                raise RuntimeError(f'{kind} output is partial; inspect per-seed ERROR.txt')
            event(f'kaggle_{kind}_harvested',folder=str(folder));return
        if state in ('RUNNING','PENDING'):
            time.sleep(30);continue
        if state in ('NOT_FOUND','ERROR','CANCELLED') and attempts<max_retries:
            if state in ('ERROR','CANCELLED'):
                failure_dir=REV/'runs'/f'kaggle_{kind}'/'failed_attempts'/str(int(time.time()))
                failure_dir.mkdir(parents=True,exist_ok=True)
                call(['kaggle','kernels','output',name,'-p',failure_dir,'-o'],timeout=300)
            bundle=REV/'kaggle_bundle'/slug
            shutil.copy2(CODE/script_name,bundle/('b01_p1_'+kind+'.py'))
            code,response=call(['kaggle','kernels','push','-p',bundle],timeout=120)
            attempts+=1
            if code:raise RuntimeError(f'{kind} kernel push failed: '+response[-1000:])
            event(f'kaggle_{kind}_pushed',attempt=attempts)
            time.sleep(20);continue
        raise RuntimeError(f'{kind} Kaggle batch requires review: {state}: {output[-1000:]}')
def batch_merged(start):
    marker=REV/'runs'/f'kaggle_replication_{start}_{start+9}'/'MERGED.json'
    if not marker.exists():return False
    try:return json.loads(marker.read_text(encoding='utf-8')).get('status')=='COMPLETE'
    except:return False
def harvest_replication(start):
    slug=f'b01-p1-postreview-replication-{start:02d}-{start+9:02d}'
    folder=REV/'runs'/f'kaggle_replication_{start}_{start+9}'
    folder.mkdir(parents=True,exist_ok=True)
    code,output=call(['kaggle','kernels','output',f'prinprin/{slug}','-p',folder,'-o'],timeout=900)
    (REV/'logs'/f'harvest_replication_{start}_{int(time.time())}.log').write_text(output,encoding='utf-8')
    source=folder/'results'/'replication'
    archive=folder/'B01_P1_REPLICATION_BATCH_OUTPUT.zip'
    if code or not (source/'RUN_STATUS.json').exists() or not archive.exists():
        raise RuntimeError(f'Replication batch {start} harvest failed: '+output[-1000:])
    status=json.loads((source/'RUN_STATUS.json').read_text(encoding='utf-8'))
    if status.get('status')!='COMPLETE' or status.get('completed_units')!=90:
        raise RuntimeError(f'Replication batch {start} incomplete: {status}')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError(f'Replication batch {start} ZIP invalid')
    from run_replication import aggregate,sha
    local_code=sha(CODE/'run_replication.py')
    local_data=sha(REV/'source_snapshot/02_DATA/processed/ispy2_core_multimodal.npz')
    merged=0;already=0
    for unit in (source/'runs').glob('*/COMPLETE.json'):
        info=json.loads(unit.read_text(encoding='utf-8'))
        if info.get('code_sha256')!=local_code or info.get('data_sha256')!=local_data:
            raise RuntimeError(f'Replication batch provenance mismatch: {unit}')
        dest=REV/'results/replication/runs'/unit.parent.name
        if (dest/'COMPLETE.json').exists():already+=1;continue
        shutil.copytree(unit.parent,dest,dirs_exist_ok=True);merged+=1
    if merged+already!=90:raise RuntimeError(f'Replication batch {start} has {merged+already} of 90 units')
    aggregate(REV/'results/replication',270)
    (folder/'MERGED.json').write_text(json.dumps({'status':'COMPLETE','merged_units':merged,
        'already_local_units':already,'archive_sha256':sha(archive),'kernel':f'prinprin/{slug}',
        'merged_utc':now()},indent=2),encoding='utf-8')
    event('kaggle_replication_batch_harvested',start=start,merged=merged,already_local=already)
def run_replication_kaggle(max_retries=2):
    attempts={10:0,20:0}
    while True:
        pending=[s for s in [10,20] if not batch_merged(s)]
        if not pending:return
        progress=False;running=0
        for start in pending:
            slug=f'b01-p1-postreview-replication-{start:02d}-{start+9:02d}'
            code,output=call(['kaggle','kernels','status',f'prinprin/{slug}'],timeout=90)
            state=('NOT_FOUND' if '404' in output or 'not found' in output.lower() else 'QUERY_ERROR') if code else next((v for v in ['COMPLETE','RUNNING','PENDING','ERROR','CANCELLED']
                                      if f'KernelWorkerStatus.{v}' in output),'UNKNOWN')
            if state=='QUERY_ERROR':event('kaggle_replication_status_query_failed',start=start,error=output[-500:]);continue
            if state=='COMPLETE':
                try:
                    harvest_replication(start);progress=True;continue
                except Exception as exc:
                    attempts[start]+=1
                    event('kaggle_replication_harvest_failed',start=start,attempt=attempts[start],error=str(exc))
                    if attempts[start]>=max_retries:raise
                    code,response=call([sys.executable,CODE/'prepare_kaggle_replication.py',
                        '--start',start,'--count',10])
                    if code:raise RuntimeError(response)
                    code,response=call(['kaggle','kernels','push','-p',REV/'kaggle_bundle'/slug],timeout=120)
                    if code:raise RuntimeError(f'Replication batch {start} retry failed: '+response[-1000:])
                    event('kaggle_replication_batch_retried',start=start,attempt=attempts[start])
                    progress=True;running+=1;continue
            if state in ('RUNNING','PENDING'):running+=1;continue
            if state in ('NOT_FOUND','ERROR','CANCELLED') and attempts[start]<max_retries and running<2:
                if state in ('ERROR','CANCELLED'):
                    failure_dir=REV/'runs'/f'kaggle_replication_{start}_{start+9}'/'failed_attempts'/str(int(time.time()))
                    failure_dir.mkdir(parents=True,exist_ok=True)
                    call(['kaggle','kernels','output',f'prinprin/{slug}','-p',failure_dir,'-o'],timeout=300)
                code,response=call([sys.executable,CODE/'prepare_kaggle_replication.py',
                    '--start',start,'--count',10])
                if code:raise RuntimeError(response)
                bundle=REV/'kaggle_bundle'/slug
                code,response=call(['kaggle','kernels','push','-p',bundle],timeout=120)
                if code:raise RuntimeError(f'Replication batch {start} push failed: '+response[-1000:])
                attempts[start]+=1;running+=1;progress=True
                event('kaggle_replication_batch_pushed',start=start,attempt=attempts[start])
            elif state not in ('RUNNING','PENDING'):
                raise RuntimeError(f'Replication batch {start} requires review: {state}: {output[-1000:]}')
        if not progress:time.sleep(30)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--skip-kaggle',action='store_true')
    ap.add_argument('--skip-replication',action='store_true');a=ap.parse_args()
    try:
        run_local('run_baselines.py','results/baselines/RUN_STATUS.json')
        run_local('run_oracle_fairness.py','results/oracle_fairness/RUN_STATUS.json')
        run_local('run_additional_diagnostics.py','results/additional/RUN_STATUS.json')
        run_local('assess_clean_overlap.py','results/additional/CLEAN_OVERLAP_STATUS.json')
        run_local('run_imputation_diagnostics.py','results/imputation/RUN_STATUS.json')
        run_local('assess_specified_propensity.py','results/oracle_fairness/SPECIFIED_PROPENSITY_STATUS.json')
        if not a.skip_kaggle:
            run_kaggle()
            run_single_kernel('missingness','b01-p1-postreview-missingness',
                'kaggle_missingness.py','missingness_postreview','B01_P1_MISSINGNESS_OUTPUT.zip')
            run_single_kernel('classical_sensitivity','b01-p1-postreview-classical-sensitivity',
                'kaggle_classical_sensitivity.py','classical_sensitivity',
                'B01_P1_CLASSICAL_SENSITIVITY_OUTPUT.zip')
            if not a.skip_replication:run_replication_kaggle()
            run_single_kernel('ridge_extension','b01-p1-postreview-ridge-extension',
                'kaggle_ridge_extension.py','ridge_extension',
                'B01_P1_RIDGE_EXTENSION_OUTPUT.zip')
        run_local('aggregate_fairness.py','results/fairness/RUN_STATUS.json')
        if not a.skip_replication:run_local('run_replication.py','results/replication/RUN_STATUS.json')
        code,output=call([sys.executable,CODE/'finalize_revision.py'])
        if code:raise RuntimeError('Final packaging failed: '+output[-2000:])
        event('resume_cycle_complete')
    except Exception as exc:
        event('resume_cycle_error',error=str(exc),traceback=traceback.format_exc())
        raise
if __name__=='__main__':main()
