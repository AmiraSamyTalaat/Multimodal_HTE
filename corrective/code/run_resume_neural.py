from __future__ import annotations
import argparse,json,re,subprocess,sys,time,traceback,zipfile
from pathlib import Path
from corrective_common import COR,LOCK,now,sha,write
from prepare_kaggle_neural import prepare

OUT=COR/'kaggle_neural_downloads';OUT.mkdir(exist_ok=True)
STATUS=OUT/'RUN_STATUS.json';EXPECTED=LOCK['fair_neural']['dgp_seeds']
def cli(*args,timeout=120):
    result=subprocess.run([sys.executable,'-m','kaggle',*args],capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError('kaggle '+ ' '.join(args)+'\n'+result.stdout+'\n'+result.stderr)
    return result.stdout+'\n'+result.stderr
def batch_state(start,count):
    folder=OUT/f'batch_{start:02d}_{start+count-1:02d}'
    zipfile_path=folder/'B01_P1_CORRECTIVE_NEURAL_BATCH.zip'
    if not zipfile_path.exists():return None
    try:
        with zipfile.ZipFile(zipfile_path) as z:
            markers=[n for n in z.namelist() if n.endswith('/COMPLETE.json')]
            statuses=[n for n in z.namelist() if n.endswith('/RUN_STATUS.json')]
            if len(markers)!=2*count or len(statuses)!=1:return None
            st=json.loads(z.read(statuses[0]))
            if st['status']!='COMPLETE' or st['completed']!=2*count:return None
        return dict(status='COMPLETE',archive_sha256=sha(zipfile_path),n_units=len(markers))
    except Exception:return None
def run_batch(start,count,max_retries,poll_seconds):
    folder=prepare(start,count)
    meta=json.loads((folder/'kernel-metadata.json').read_text(encoding='utf-8'))
    kernel=meta['id'];dest=OUT/f'batch_{start:02d}_{start+count-1:02d}';dest.mkdir(exist_ok=True)
    old=batch_state(start,count)
    if old:return dict(kernel=kernel,**old,action='resumed_existing_download')
    attempts=[]
    for attempt in range(1,max_retries+1):
        try:
            try:status=cli('kernels','status',kernel)
            except RuntimeError:status='NOT_FOUND'
            if not any(x in status for x in ['RUNNING','QUEUED','COMPLETE']):
                push=cli('kernels','push','-p',str(folder),timeout=180)
                attempts.append(dict(attempt=attempt,utc=now(),event='push',message=push.strip()))
            while True:
                status=cli('kernels','status',kernel)
                if 'COMPLETE' in status:break
                if 'ERROR' in status or 'CANCELLED' in status:raise RuntimeError(status)
                time.sleep(poll_seconds)
            output=cli('kernels','output',kernel,'-p',str(dest),timeout=180)
            attempts.append(dict(attempt=attempt,utc=now(),event='download',message=output.strip()))
            state=batch_state(start,count)
            if state:
                write(dest/'manifest.json',dict(kernel=kernel,attempts=attempts,**state))
                return dict(kernel=kernel,**state,attempts=attempt)
            raise RuntimeError('Downloaded archive incomplete or missing; inspect Kaggle log and batch zip')
        except Exception as exc:
            attempts.append(dict(attempt=attempt,utc=now(),event='error',message=str(exc),traceback=traceback.format_exc()))
            write(dest/'attempts.json',attempts)
            if attempt<max_retries:
                # A failed complete kernel needs a new version. Resume from any verified local output first.
                try:cli('kernels','push','-p',str(folder),timeout=180)
                except Exception as push_exc:attempts.append(dict(attempt=attempt,utc=now(),event='retry_push_error',message=str(push_exc)))
    return dict(kernel=kernel,status='FAILED',attempts=attempts)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--max-retries',type=int,default=2)
    ap.add_argument('--poll-seconds',type=int,default=30)
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--count',type=int,default=len(EXPECTED))
    a=ap.parse_args();results=[]
    for start in range(a.start,a.start+a.count,a.batch_size):
        count=min(a.batch_size,a.start+a.count-start)
        result=run_batch(start,count,a.max_retries,a.poll_seconds)
        results.append(result)
        write(STATUS,dict(updated_utc=now(),completed_batches=sum(x.get('status')=='COMPLETE' for x in results),
            expected_batches=(a.count+a.batch_size-1)//a.batch_size,batches=results,
            persistence='Private Kaggle saved kernel outputs plus verified local ZIP after each batch'))
        print(start,count,result['status'],flush=True)
    if any(x['status']!='COMPLETE' for x in results):sys.exit(1)
if __name__=='__main__':main()
