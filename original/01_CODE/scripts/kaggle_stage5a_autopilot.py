from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time, zipfile, hashlib, site, re
from pathlib import Path

OWNER='prinprin'
DATASET_SLUG='b01-causaltx-mm-stage5a-runtime'
DATASET_REF=f'{OWNER}/{DATASET_SLUG}'
KERNEL_SLUG='b01-causaltx-mm-stage5a'
KERNEL_REF=f'{OWNER}/{KERNEL_SLUG}'
STAGE5A_CODE_FIX_ID='MASKED_TRUE_TE_V1'
RUNTIME_FIX_ID='MASKED_TRUE_TE_RUNTIME_V3'

def run(cmd,check=True,cwd=None,capture=True):
    print('>', ' '.join(str(x) for x in cmd),flush=True)
    r=subprocess.run([str(x) for x in cmd],cwd=cwd,text=True,capture_output=capture)
    if r.stdout: print(r.stdout.strip(),flush=True)
    if r.stderr: print(r.stderr.strip(),flush=True)
    if check and r.returncode!=0: raise RuntimeError(f'Command failed ({r.returncode}): {cmd}')
    return r

def find_kaggle():
    p=shutil.which('kaggle')
    if p: return p
    exe=Path(sys.executable)
    cands=[exe.parent/'Scripts'/'kaggle.exe',exe.parent/'kaggle.exe']
    try: cands.append(Path(site.USER_BASE)/'Scripts'/'kaggle.exe')
    except Exception: pass
    for c in cands:
        if c.exists(): return str(c)
    return None

def ensure_kaggle():
    k=find_kaggle()
    if k: return k
    run([sys.executable,'-m','pip','install','--upgrade','kaggle'],capture=False)
    k=find_kaggle()
    if not k: raise RuntimeError('Kaggle CLI installed but executable not found.')
    return k

def auth_check(k):
    return run([k,'kernels','list','-m','--page-size','1'],check=False).returncode==0

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def prepare_dataset(root):
    prep=root/'01_CODE'/'scripts'/'prepare_stage5a_bundle.py'
    run([sys.executable,str(prep),'--root',str(root)],capture=False)
    ddir=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE5A_UPLOAD_DATASET'
    z=ddir/'B01_STAGE5A_BUNDLE.zip'
    if not z.exists(): raise RuntimeError('Stage5A bundle was not created.')
    return ddir,z

def _txt(r): return ((r.stdout or '')+' '+(r.stderr or '')).lower()

def ensure_dataset(k,ddir):
    versioned=False
    for attempt in range(1,5):
        r=run([k,'datasets','version','-p',str(ddir),'-m','Locked B01 Stage5A runtime refresh','-t'],check=False)
        t=_txt(r)
        if r.returncode==0 and not any(x in t for x in ['error:','creation error']):
            versioned=True; break
        if any(x in t for x in ['not found','does not exist','404']): break
        if any(x in t for x in ['timed out','timeout','max retries exceeded','connection']):
            print(f'Stage5A dataset API transient failure {attempt}/4; retrying same action.',flush=True)
            time.sleep(20); continue
        if 'already in use by a dataset' in t or 'already exists' in t:
            time.sleep(20); continue
        break
    if not versioned:
        r=run([k,'datasets','create','-p',str(ddir),'-t'],check=False)
        t=_txt(r)
        if r.returncode!=0:
            if 'already in use by a dataset' in t or 'already exists' in t:
                r2=run([k,'datasets','version','-p',str(ddir),'-m','Locked B01 Stage5A runtime refresh after create race','-t'],check=False)
                if r2.returncode!=0: raise RuntimeError('Could not version existing Stage5A dataset.')
            else:
                raise RuntimeError('Could not create/version Stage5A runtime dataset.')
    for _ in range(60):
        r=run([k,'datasets','status',DATASET_REF],check=False)
        t=_txt(r)
        if r.returncode==0 and any(x in t for x in ['ready','complete','active']): return
        time.sleep(15)
    raise RuntimeError('Stage5A private dataset did not reach READY within 15 minutes.')

def complete_pairs(root):
    out=set()
    p=root/'05_STRESS_RUNS'
    if not p.exists(): return out
    pat=re.compile(r'^STRESS_DGP(\d+)_NET(\d+)_')
    for d in p.iterdir():
        if not d.is_dir() or not (d/'COMPLETE.json').exists(): continue
        m=pat.match(d.name)
        if m: out.add((int(m.group(1)),int(m.group(2))))
    return out

def kernel_script(tasks):
    template = r'''from pathlib import Path
import json, shutil, subprocess, sys, time, traceback, zipfile
TASKS=__TASKS__
WORK=Path('/kaggle/working/B01')
EXPORT=Path('/kaggle/working/STAGE5A_EXPORT')
EXPORT.mkdir(parents=True,exist_ok=True)
(EXPORT/'logs').mkdir(parents=True,exist_ok=True)
status={'tasks':[list(x) for x in TASKS],'started':time.time(),'status':'STARTING','results':[],'stage':'STAGE5A','scientific_settings_changed':False,'stage5a_code_fix_id':'__CODE_FIX_ID__'}

def save():
    status['finished']=time.time()
    (EXPORT/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2,default=str))
    if WORK.exists():
        src=WORK/'05_STRESS_RUNS'
        dst=EXPORT/'05_STRESS_RUNS'
        if src.exists(): shutil.copytree(src,dst,dirs_exist_ok=True)
    z=Path('/kaggle/working/B01_STAGE5A_SESSION.zip')
    if z.exists(): z.unlink()
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
        for p in EXPORT.rglob('*'):
            if p.is_file(): zz.write(p,p.relative_to(EXPORT))

try:
    input_root=Path('/kaggle/input')
    zips=[p for p in input_root.rglob('B01_STAGE5A_BUNDLE.zip') if p.is_file()]
    dirs=[p for p in input_root.rglob('B01_STAGE5A_BUNDLE') if p.is_dir()]
    shutil.rmtree(WORK,ignore_errors=True); WORK.mkdir(parents=True,exist_ok=True)
    if zips:
        with zipfile.ZipFile(zips[0]) as zz: zz.extractall(WORK)
    elif dirs:
        shutil.copytree(dirs[0],WORK,dirs_exist_ok=True)
    else:
        found=None
        for req in input_root.rglob('stage5a_stress_lock.yaml'):
            par=req.parent.parent
            if (par/'01_CODE'/'scripts'/'run_stage5a_secondary_stress.py').exists():
                found=par; break
        if found is None: raise RuntimeError('Stage5A runtime bundle not found in Kaggle inputs.')
        shutil.copytree(found,WORK,dirs_exist_ok=True)

    data=WORK/'02_DATA'/'processed'/'ispy2_core_multimodal.npz'
    script=WORK/'01_CODE'/'scripts'/'run_stage5a_secondary_stress.py'
    for req in [data,script,WORK/'00_CONFIG'/'stage5a_stress_lock.yaml']:
        if not req.exists(): raise RuntimeError('Required Stage5A runtime file missing: '+str(req))

    probe=subprocess.run([sys.executable,str(script),'--help'],text=True,capture_output=True)
    (EXPORT/'logs'/'IMPORT_PROBE.stdout.txt').write_text(probe.stdout or '')
    (EXPORT/'logs'/'IMPORT_PROBE.stderr.txt').write_text(probe.stderr or '')
    if probe.returncode!=0: raise RuntimeError('Stage5A scientific runtime import probe failed.')

    status['status']='RUNNING'
    for dgp,net in TASKS:
        cmd=[sys.executable,str(script),'--root',str(WORK),'--data',str(data),'--dgp-seed',str(dgp),'--net-seed',str(net)]
        print('RUN',cmd,flush=True); t0=time.time()
        r=subprocess.run(cmd,text=True,capture_output=True)
        base=EXPORT/'logs'/f'STAGE5A_DGP{dgp}_NET{net}'
        Path(str(base)+'.stdout.txt').write_text(r.stdout or '')
        Path(str(base)+'.stderr.txt').write_text(r.stderr or '')
        print(r.stdout,flush=True); print(r.stderr,flush=True)
        status['results'].append({'dgp_seed':dgp,'net_seed':net,'returncode':r.returncode,'seconds':time.time()-t0})
        if r.returncode!=0:
            status['status']='FAIL'; status['failed_task']=[dgp,net]; status['failure_kind']='scientific_subprocess_error'; break
    else:
        status['status']='PASS'
except BaseException as e:
    status['status']='FAIL'; status['failure_kind']='wrapper_or_runtime_error'; status['exception']=repr(e); status['traceback']=traceback.format_exc()
    (EXPORT/'KERNEL_EXCEPTION.txt').write_text(status['traceback'])
finally:
    save()
    print(json.dumps(status,indent=2,default=str),flush=True)
'''
    return template.replace('__TASKS__',repr(tasks)).replace('__CODE_FIX_ID__',STAGE5A_CODE_FIX_ID)

def prepare_kernel(root,tasks):
    kdir=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE5A_KERNEL'
    shutil.rmtree(kdir,ignore_errors=True); kdir.mkdir(parents=True)
    (kdir/'b01_stage5a_session.py').write_text(kernel_script(tasks),encoding='utf-8')
    meta={'id':KERNEL_REF,'title':'B01 CausalTx MM Stage5A','code_file':'b01_stage5a_session.py',
          'language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':False,
          'machine_shape':'NvidiaTeslaT4','dataset_sources':[DATASET_REF],'competition_sources':[],
          'kernel_sources':[],'model_sources':[]}
    (kdir/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return kdir

def parse_status(stdout='',stderr=''):
    raw=((stdout or '')+' '+(stderr or '')).strip()
    m=re.search(r'KernelWorkerStatus\.([A-Z_]+)',raw,re.I)
    if m:
        st=m.group(1).upper()
        if st in {'COMPLETE','COMPLETED'}: return 'COMPLETE',raw
        if st in {'ERROR','FAILED','CANCELLED','CANCELED'}: return 'FAILED',raw
        if st=='RUNNING': return 'RUNNING',raw
        if st in {'QUEUED','PENDING'}: return 'QUEUED',raw
    low=raw.lower()
    if any(x in low for x in ['not found','404','does not exist','no kernel']):
        return 'NOT_FOUND',raw
    if ('cannot access kernel' in low or
        "permission 'kernels.get' was denied" in low or
        'wrong kernel slug' in low):
        return 'ACCESS_DENIED_OR_WRONG_SLUG',raw
    if any(x in low for x in ['connecttimeouterror','readtimeouterror','connectionerror','max retries exceeded','timed out','timeout','connection reset','name resolution']):
        return 'TRANSIENT',raw
    return 'UNKNOWN',raw

def _search_exact_stage5a_kernel(k):
    r=run([k,'kernels','list','-m','-s',KERNEL_SLUG,'--page-size','100','--sort-by','dateRun'],check=False)
    raw=((r.stdout or '')+' '+(r.stderr or ''))
    if r.returncode!=0:
        return None,raw
    refs=[]
    for line in raw.splitlines():
        s=line.strip()
        if s.lower().startswith(OWNER.lower()+'/'):
            refs.append(s.split()[0].strip())
    exact=any(x.lower()==KERNEL_REF.lower() for x in refs)
    return exact,raw

def query_status(k):
    r=run([k,'kernels','status',KERNEL_REF],check=False)
    st,raw=parse_status(r.stdout or '',r.stderr or '')
    if st!='ACCESS_DENIED_OR_WRONG_SLUG':
        return st,raw
    exact,list_raw=_search_exact_stage5a_kernel(k)
    if exact is False:
        return 'NOT_FOUND',raw+'\nExact Stage5A kernel is absent from owner search results.'
    if exact is True:
        return 'TRANSIENT',raw+'\nExact Stage5A kernel exists in owner search results; refusing blind duplicate push.'
    return 'TRANSIENT',raw+'\nOwner kernel search could not be resolved because the Kaggle API is temporarily unavailable.'

def resolve_current_stage5a(k,max_wait_sec=900):
    t=time.time(); last=''
    while time.time()-t<max_wait_sec:
        st,raw=query_status(k); last=raw
        if st in {'RUNNING','QUEUED','COMPLETE','FAILED','NOT_FOUND'}:
            return st,raw
        print('Stage5A current-kernel status unresolved because of temporary API/network state. Waiting rather than pushing a duplicate.',flush=True)
        time.sleep(30)
    return 'UNRESOLVED',last

def poll_kernel(k,timeout_sec=42000):
    t=time.time(); last=''
    while time.time()-t<timeout_sec:
        st,raw=query_status(k); last=raw
        if st in {'COMPLETE','FAILED'}: return st,raw
        if st in {'RUNNING','QUEUED'}:
            time.sleep(60); continue
        print(f'Stage5A Kaggle status API temporary state={st}. No duplicate kernel will be pushed.',flush=True)
        time.sleep(60)
    return 'TIMEOUT',last

def _transient(s):
    s=(s or '').lower()
    return any(x in s for x in ['connecttimeouterror','readtimeouterror','connectionerror','max retries exceeded','timed out','timeout','connection reset','name resolution'])

def retry_output(k,out,attempts=10):
    text=''
    for a in range(1,attempts+1):
        r=run([k,'kernels','output',KERNEL_REF,'-p',str(out),'-o'],check=False)
        text=(r.stdout or '')+' '+(r.stderr or '')
        if r.returncode==0: return True,text
        if _transient(text):
            wait=min(30*a,180)
            print(f'Stage5A harvest API transient failure {a}/{attempts}; retrying SAME completed kernel in {wait}s.',flush=True)
            time.sleep(wait); continue
        return False,text
    return False,text

def harvest(k,root,dgp):
    out=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE5A_HARVEST'/f'DGP{dgp}'
    out.mkdir(parents=True,exist_ok=True)
    ok,last=retry_output(k,out)
    if not ok: raise RuntimeError('Stage5A output harvest failed: '+last)
    z=out/'B01_STAGE5A_SESSION.zip'
    if not z.exists():
        hits=list(out.rglob('B01_STAGE5A_SESSION.zip'))
        if hits: z=hits[0]
    if not z.exists(): raise RuntimeError('B01_STAGE5A_SESSION.zip missing after COMPLETE kernel.')
    ex=out/'extracted'; shutil.rmtree(ex,ignore_errors=True); ex.mkdir(parents=True)
    with zipfile.ZipFile(z) as zz: zz.extractall(ex)
    sf=ex/'SESSION_STATUS.json'
    if not sf.exists(): raise RuntimeError('Stage5A SESSION_STATUS.json missing.')
    sj=json.loads(sf.read_text(encoding='utf-8'))
    src=ex/'05_STRESS_RUNS'; dst=root/'05_STRESS_RUNS'; dst.mkdir(parents=True,exist_ok=True)
    if src.exists():
        for p in src.iterdir():
            if p.is_dir(): shutil.copytree(p,dst/p.name,dirs_exist_ok=True)
    return sj,out

def write_progress(root,obj):
    p=root/'03_STATE'/'KAGGLE_STAGE5A_PROGRESS.json'
    p.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def pre_fix_scientific_failure_can_retry(sj,tasks):
    same=[tuple(x) for x in sj.get('tasks',[])]==tasks
    if not same or sj.get('status')=='PASS':
        return False
    if sj.get('failure_kind')!='scientific_subprocess_error':
        return False
    # The failed DGP101/NET17 session was produced before the masked-ground-truth
    # indexing correction and therefore has no stage5a_code_fix_id.
    return not sj.get('stage5a_code_fix_id')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args(); root=Path(a.root)
    pf=root/'03_STATE'/'STAGE5A_PREFLIGHT.json'
    if not pf.exists() or json.loads(pf.read_text(encoding='utf-8')).get('status')!='PASS':
        raise SystemExit('STAGE5A_PREFLIGHT PASS required. Run RUN_STAGE5A_PREFLIGHT_B01.cmd first.')

    k=ensure_kaggle()
    if not auth_check(k):
        r=run([k,'auth','login','--force'],check=False,capture=False)
        if r.returncode!=0 or not auth_check(k): raise SystemExit('Kaggle OAuth not completed as prinprin.')

    dgp_seeds=[101,211,307,401,503]; nets=[17,29,43]
    prog={'status':'RUNNING','dataset':DATASET_REF,'kernel':KERNEL_REF,'sessions':[],
          'locked_account':'prinprin','stage':'STAGE5A','runtime_fix':RUNTIME_FIX_ID,'scientific_code_fix':STAGE5A_CODE_FIX_ID,'primary_grid_changed':False}
    write_progress(root,prog)
    dataset_ready=False

    def ensure_runtime():
        nonlocal dataset_ready
        if dataset_ready: return
        ddir,z=prepare_dataset(root)
        print('STAGE5A BUNDLE',z,'SHA256',sha256(z),flush=True)
        ensure_dataset(k,ddir); dataset_ready=True

    first_incomplete=True
    for dgp in dgp_seeds:
        done=complete_pairs(root)
        tasks=[(dgp,n) for n in nets if (dgp,n) not in done]
        if not tasks:
            print('SKIP STAGE5A DGP',dgp,'all 3 net seeds COMPLETE')
            continue

        if first_incomplete:
            first_incomplete=False
            st,raw=resolve_current_stage5a(k)
            if st=='UNRESOLVED':
                raise SystemExit('Stage5A Kaggle API remained unresolved for 15 minutes. No dataset refresh and no new kernel were started; rerun later.')
            if st in {'RUNNING','QUEUED'}:
                print('Existing Stage5A kernel is',st,'; attaching instead of pushing duplicate.',flush=True)
                st,raw=poll_kernel(k)
                if st=='COMPLETE':
                    try:
                        sj,out=harvest(k,root,dgp)
                    except Exception as e:
                        raise SystemExit('Existing Stage5A kernel completed but harvest is not currently available. No duplicate was pushed. Rerun later. '+repr(e))
                    if [tuple(x) for x in sj.get('tasks',[])]==tasks and sj.get('status')=='PASS':
                        print('Recovered Stage5A DGP',dgp,'from existing kernel.',flush=True)
                        continue
                    if [tuple(x) for x in sj.get('tasks',[])]==tasks and sj.get('status')!='PASS':
                        if pre_fix_scientific_failure_can_retry(sj,tasks):
                            print('Recovered a PRE-FIX Stage5A scientific subprocess failure. The failure was caused by a deterministic masked-ground-truth shape bug; an exact locked rerun with the code correction is allowed.',flush=True)
                        else:
                            raise SystemExit(f"Existing Stage5A DGP{dgp} diagnostic FAIL ({sj.get('failure_kind')}). This session already contains the current code-fix marker or is not the known pre-fix failure; stop for diagnostics.")
                elif st=='TIMEOUT':
                    raise SystemExit('Current Stage5A kernel unresolved. No duplicate pushed; rerun later.')
            elif st=='COMPLETE':
                try:
                    sj,out=harvest(k,root,dgp)
                except Exception as e:
                    raise SystemExit('Existing Stage5A COMPLETE kernel could not be harvested. No duplicate was pushed. Rerun later. '+repr(e))
                if [tuple(x) for x in sj.get('tasks',[])]==tasks and sj.get('status')=='PASS':
                    print('Recovered COMPLETE Stage5A DGP',dgp,'from Kaggle.',flush=True)
                    continue
                if [tuple(x) for x in sj.get('tasks',[])]==tasks and sj.get('status')!='PASS':
                    if pre_fix_scientific_failure_can_retry(sj,tasks):
                        print('Recovered a PRE-FIX Stage5A scientific subprocess failure. The failure was caused by a deterministic masked-ground-truth shape bug; an exact locked rerun with the code correction is allowed.',flush=True)
                    else:
                        raise SystemExit(f"Existing Stage5A DGP{dgp} diagnostic FAIL ({sj.get('failure_kind')}). This session already contains the current code-fix marker or is not the known pre-fix failure; stop for diagnostics.")
                print('Existing COMPLETE Stage5A kernel belongs to earlier tasks; a new locked session is required.',flush=True)
            elif st=='FAILED':
                print('Existing Stage5A worker explicitly failed; a fresh identical locked attempt is allowed.',flush=True)
            elif st=='NOT_FOUND':
                print('No existing Stage5A kernel found. First locked Stage5A session may be launched.',flush=True)

        ensure_runtime()
        kdir=prepare_kernel(root,tasks)
        print('Launching Stage5A tasks',tasks,flush=True)
        final='FAILED'; raw=''
        for attempt in (1,2):
            run([k,'kernels','push','-p',str(kdir),'--accelerator','NvidiaTeslaT4','-t','43200'])
            final,raw=poll_kernel(k)
            if final=='COMPLETE': break
            if final=='TIMEOUT': raise SystemExit('Stage5A polling timeout. No duplicate kernel pushed; rerun later.')
            if attempt==1:
                print('Explicit Stage5A worker failure. Retrying identical tasks once after 90s.',flush=True)
                time.sleep(90)
        if final!='COMPLETE': raise SystemExit('Stage5A worker explicitly failed twice.')
        sj,out=harvest(k,root,dgp)
        prog['sessions'].append({'dgp_seed':dgp,'status':sj.get('status'),'details':sj,'harvest':str(out)})
        write_progress(root,prog)
        if sj.get('status')!='PASS':
            raise SystemExit(f"Stage5A DGP{dgp} diagnostic FAIL ({sj.get('failure_kind')}). Do not change scientific settings.")

    expected={(d,n) for d in dgp_seeds for n in nets}; done=complete_pairs(root)
    prog['complete_pairs']=len(done & expected)
    prog['status']='STAGE5A_GRID_COMPLETE' if expected.issubset(done) else 'INCOMPLETE'
    write_progress(root,prog)
    if not expected.issubset(done): raise SystemExit(f'Stage5A incomplete {len(done & expected)}/15')

    ag=root/'01_CODE'/'scripts'/'aggregate_stage5a_results.py'
    run([sys.executable,str(ag),'--root',str(root)],capture=False)
    print('STAGE5A COMPLETE: 15/15 stress pairs harvested. See 06_RESULTS/STAGE5A/STAGE5A_AUDIT.json')

if __name__=='__main__': main()
