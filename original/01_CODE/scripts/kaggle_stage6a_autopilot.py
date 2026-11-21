
from __future__ import annotations
import json, re, shutil, site, subprocess, sys, time
from pathlib import Path

OWNER='prinprin'
DATASET_SLUG='b01-causaltx-mm-stage6a-runtime'
DATASET_REF=f'{OWNER}/{DATASET_SLUG}'
KERNEL_SLUG='b01-causaltx-mm-stage6a'
KERNEL_REF=f'{OWNER}/{KERNEL_SLUG}'
STAGE6A_RUNTIME_FIX_ID='DATASET_ATTACHMENT_RECOVERY_V2'
DGP_SEEDS=[607,719,823,941,1061]
NET_SEEDS=[19,37]

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

def _txt(r): return ((r.stdout or '')+' '+(r.stderr or '')).lower()

def prepare_dataset(root):
    run([sys.executable,str(root/'01_CODE'/'scripts'/'prepare_stage6a_bundle.py'),'--root',str(root)],capture=False)
    d=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE6A_UPLOAD_DATASET'
    z=d/'B01_STAGE6A_BUNDLE.zip'
    if not z.exists(): raise RuntimeError('Stage6A bundle was not created.')
    return d,z

def _auth_required_text(text):
    low=(text or '').lower()
    return 'authentication required' in low or 'kaggle auth login' in low

def _dataset_status(k):
    r=run([k,'datasets','status',DATASET_REF],check=False)
    return r, _txt(r)

def ensure_dataset(k,ddir):
    """
    Runtime-only recovery logic.
    If the private Stage6A dataset already exists and is READY, reuse it.
    Otherwise version/create it, then wait for Kaggle input attachment propagation
    before the kernel is pushed.
    """
    rs,ts=_dataset_status(k)
    if _auth_required_text(ts):
        print('Kaggle authentication expired before Stage6A dataset check. Refreshing OAuth.',flush=True)
        run([k,'auth','login','--force'],capture=False)
        rs,ts=_dataset_status(k)

    already_ready=(rs.returncode==0 and any(x in ts for x in ['ready','complete','active']))
    if already_ready:
        print('Existing Stage6A runtime dataset is READY; reusing it for the runtime-only fix.',flush=True)
    else:
        versioned=False
        for attempt in range(1,5):
            r=run([k,'datasets','version','-p',str(ddir),'-m','Locked B01 Stage6A DEV-only runtime refresh','-t'],check=False)
            t=_txt(r)
            if r.returncode==0 and not any(x in t for x in ['error:','creation error']):
                versioned=True
                break
            if _auth_required_text(t):
                print('Kaggle authentication expired during dataset versioning. Refreshing OAuth.',flush=True)
                run([k,'auth','login','--force'],capture=False)
                continue
            # A 403 can also occur when a version was requested before the dataset
            # existed. Check the actual dataset state before deciding to create.
            rs2,ts2=_dataset_status(k)
            if rs2.returncode==0 and any(x in ts2 for x in ['ready','complete','active']):
                versioned=True
                print('Stage6A dataset already exists and is READY after version API rejection; reusing it.',flush=True)
                break
            if any(x in t for x in ['not found','does not exist','404']) or any(x in ts2 for x in ['not found','does not exist','404']):
                break
            if any(x in t for x in ['timed out','timeout','max retries exceeded','connection']):
                print(f'Stage6A dataset API transient failure {attempt}/4; retrying same action.',flush=True)
                time.sleep(20)
                continue
            break

        if not versioned:
            r=run([k,'datasets','create','-p',str(ddir),'-t'],check=False)
            t=_txt(r)
            if r.returncode!=0:
                rs3,ts3=_dataset_status(k)
                if rs3.returncode==0 and any(x in ts3 for x in ['ready','complete','active']):
                    print('Stage6A dataset exists despite create/version response; reusing READY dataset.',flush=True)
                elif 'already in use by a dataset' in t or 'already exists' in t:
                    pass
                else:
                    raise RuntimeError('Could not create/version Stage6A runtime dataset.')

    # READY is necessary but a brand-new private dataset can take additional time
    # before it becomes attachable as /kaggle/input in a kernel.
    ready=False
    for _ in range(60):
        r,t=_dataset_status(k)
        if r.returncode==0 and any(x in t for x in ['ready','complete','active']):
            ready=True
            break
        if _auth_required_text(t):
            print('Kaggle authentication expired while waiting for dataset READY. Refreshing OAuth.',flush=True)
            run([k,'auth','login','--force'],capture=False)
        time.sleep(15)
    if not ready:
        raise RuntimeError('Stage6A private dataset did not reach READY within 15 minutes.')

    # Optional file-list verification. Do not fail merely because an older CLI
    # lacks the option; the kernel wrapper will independently verify the mount.
    rf=run([k,'datasets','files',DATASET_REF,'--page-size','100'],check=False)
    ftxt=((rf.stdout or '')+' '+(rf.stderr or ''))
    if rf.returncode==0 and 'B01_STAGE6A_BUNDLE.zip' in ftxt:
        print('Stage6A bundle is visible in Kaggle dataset files.',flush=True)
    else:
        print('Dataset file-list verification was inconclusive; applying propagation wait before kernel push.',flush=True)

    print('Waiting 90 seconds for private-dataset attachment propagation before Stage6A kernel push...',flush=True)
    time.sleep(90)


def complete_pairs(root):
    out=set(); p=root/'07_METHOD_DEV_RUNS'
    if not p.exists(): return out
    pat=re.compile(r'^DEV6A_DGP(\d+)_NET(\d+)_')
    for d in p.iterdir():
        if not d.is_dir() or not (d/'COMPLETE.json').exists(): continue
        m=pat.match(d.name)
        if m: out.add((int(m.group(1)),int(m.group(2))))
    return out

def kernel_script(tasks):
    template=r'''from pathlib import Path
import json, shutil, subprocess, sys, time, traceback, zipfile
TASKS=__TASKS__
WORK=Path('/kaggle/working/B01')
EXPORT=Path('/kaggle/working/STAGE6A_EXPORT')
EXPORT.mkdir(parents=True,exist_ok=True)
(EXPORT/'logs').mkdir(parents=True,exist_ok=True)
status={'tasks':[list(x) for x in TASKS],'stage':'STAGE6A_Q1_METHOD_DEVELOPMENT',
        'split':'DEV_ONLY','test_accessed':False,'primary_grid_rerun':False,
        'stage5a_rerun':False,'runtime_fix_id':'DATASET_ATTACHMENT_RECOVERY_V2',
        'started':time.time(),'status':'STARTING','results':[]}

def save():
    status['finished']=time.time()
    (EXPORT/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2,default=str))
    src=WORK/'07_METHOD_DEV_RUNS'; dst=EXPORT/'07_METHOD_DEV_RUNS'
    if src.exists(): shutil.copytree(src,dst,dirs_exist_ok=True)
    z=Path('/kaggle/working/B01_STAGE6A_SESSION.zip')
    if z.exists(): z.unlink()
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
        for p in EXPORT.rglob('*'):
            if p.is_file(): zz.write(p,p.relative_to(EXPORT))

try:
    inp=Path('/kaggle/input')
    inventory=[]
    if inp.exists():
        for p in inp.rglob('*'):
            try:
                inventory.append(str(p))
            except Exception:
                pass
    (EXPORT/'KAGGLE_INPUT_INVENTORY.txt').write_text('\n'.join(inventory))

    zips=[p for p in inp.rglob('B01_STAGE6A_BUNDLE.zip') if p.is_file()] if inp.exists() else []
    shutil.rmtree(WORK,ignore_errors=True); WORK.mkdir(parents=True,exist_ok=True)

    if zips:
        status['input_mode']='bundle_zip'
        status['input_bundle_path']=str(zips[0])
        with zipfile.ZipFile(zips[0]) as zz:
            zz.extractall(WORK)
    else:
        # Recovery for a backend that exposes the dataset contents already extracted.
        roots=[]
        if inp.exists():
            for lockp in inp.rglob('stage6a_q1_method_development_lock.yaml'):
                if lockp.parent.name=='00_CONFIG':
                    candidate=lockp.parent.parent
                    if (candidate/'01_CODE'/'scripts'/'run_stage6a_q1_development.py').exists() and (candidate/'02_DATA'/'processed'/'ispy2_core_multimodal.npz').exists():
                        roots.append(candidate)
        if roots:
            status['input_mode']='already_extracted_dataset'
            status['input_bundle_path']=str(roots[0])
            shutil.copytree(roots[0],WORK,dirs_exist_ok=True)
        else:
            raise RuntimeError(
                'Stage6A runtime dataset was not mounted in /kaggle/input. '
                'No B01_STAGE6A_BUNDLE.zip and no extracted B01 runtime tree were found. '
                'See STAGE6A_EXPORT/KAGGLE_INPUT_INVENTORY.txt.'
            )

    data=WORK/'02_DATA'/'processed'/'ispy2_core_multimodal.npz'
    script=WORK/'01_CODE'/'scripts'/'run_stage6a_q1_development.py'
    lock=WORK/'00_CONFIG'/'stage6a_q1_method_development_lock.yaml'
    for req in [data,script,lock]:
        if not req.exists(): raise RuntimeError('Required Stage6A runtime file missing: '+str(req))

    probe=subprocess.run([sys.executable,str(script),'--help'],text=True,capture_output=True)
    (EXPORT/'logs'/'IMPORT_PROBE.stdout.txt').write_text(probe.stdout or '')
    (EXPORT/'logs'/'IMPORT_PROBE.stderr.txt').write_text(probe.stderr or '')
    if probe.returncode!=0: raise RuntimeError('Stage6A scientific runtime import probe failed.')

    status['status']='RUNNING'
    for dgp,net in TASKS:
        cmd=[sys.executable,str(script),'--root',str(WORK),'--data',str(data),
             '--dgp-seed',str(dgp),'--net-seed',str(net)]
        print('RUN',cmd,flush=True); t0=time.time()
        r=subprocess.run(cmd,text=True,capture_output=True)
        base=EXPORT/'logs'/f'STAGE6A_DGP{dgp}_NET{net}'
        Path(str(base)+'.stdout.txt').write_text(r.stdout or '')
        Path(str(base)+'.stderr.txt').write_text(r.stderr or '')
        print(r.stdout,flush=True); print(r.stderr,flush=True)
        status['results'].append({'dgp_seed':dgp,'net_seed':net,'returncode':r.returncode,'seconds':time.time()-t0})
        if r.returncode!=0:
            status['status']='FAIL'; status['failed_task']=[dgp,net]
            status['failure_kind']='scientific_subprocess_error'; break
    else:
        status['status']='PASS'
except BaseException as e:
    status['status']='FAIL'; status['failure_kind']='wrapper_or_runtime_error'
    status['exception']=repr(e); status['traceback']=traceback.format_exc()
    (EXPORT/'KERNEL_EXCEPTION.txt').write_text(status['traceback'])
finally:
    save()
    print(json.dumps(status,indent=2,default=str),flush=True)
'''
    return template.replace('__TASKS__',repr(tasks))

def prepare_kernel(root,tasks):
    kdir=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE6A_KERNEL'
    shutil.rmtree(kdir,ignore_errors=True); kdir.mkdir(parents=True)
    (kdir/'b01_stage6a_session.py').write_text(kernel_script(tasks),encoding='utf-8')
    meta={'id':KERNEL_REF,'title':'B01 CausalTx MM Stage6A','code_file':'b01_stage6a_session.py',
          'language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':False,
          'machine_shape':'NvidiaTeslaT4','dataset_sources':[DATASET_REF],
          'competition_sources':[],'kernel_sources':[],'model_sources':[]}
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
    if 'authentication required' in low or 'kaggle auth login' in low:
        return 'AUTH_REQUIRED',raw
    if any(x in low for x in ['not found','404','does not exist','no kernel']):
        return 'NOT_FOUND',raw
    if ('cannot access kernel' in low or "permission 'kernels.get' was denied" in low or 'wrong kernel slug' in low):
        return 'ACCESS_DENIED_OR_WRONG_SLUG',raw
    if any(x in low for x in ['connecttimeouterror','readtimeouterror','connectionerror','max retries exceeded','timed out','timeout','connection reset','name resolution']):
        return 'TRANSIENT',raw
    return 'UNKNOWN',raw

def exact_kernel_exists(k):
    r=run([k,'kernels','list','-m','-s',KERNEL_SLUG,'--page-size','100','--sort-by','dateRun'],check=False)
    if r.returncode!=0: return None
    refs=[]
    for line in ((r.stdout or '')+'\n'+(r.stderr or '')).splitlines():
        s=line.strip()
        if s.lower().startswith(OWNER.lower()+'/'): refs.append(s.split()[0])
    return any(x.lower()==KERNEL_REF.lower() for x in refs)

def query_status(k):
    r=run([k,'kernels','status',KERNEL_REF],check=False)
    st,raw=parse_status(r.stdout or '',r.stderr or '')
    if st!='ACCESS_DENIED_OR_WRONG_SLUG': return st,raw
    ex=exact_kernel_exists(k)
    if ex is False: return 'NOT_FOUND',raw+'\nExact Stage6A kernel absent from owner search.'
    return 'TRANSIENT',raw+'\nExact Stage6A kernel existence ambiguous/present; refusing blind duplicate.'

def poll_kernel(k,timeout_sec=42000):
    t=time.time(); last=''; auth_refreshes=0
    while time.time()-t<timeout_sec:
        st,raw=query_status(k); last=raw
        if st=='COMPLETE': return 'COMPLETE',raw
        if st=='FAILED': return 'FAILED',raw
        if st=='AUTH_REQUIRED':
            auth_refreshes += 1
            if auth_refreshes > 2:
                return 'AUTH_REQUIRED',raw
            print('Kaggle OAuth expired while the Stage6A kernel was running. Refreshing authentication; the Kaggle job itself is not being restarted.',flush=True)
            run([k,'auth','login','--force'],capture=False)
            time.sleep(5)
            continue
        time.sleep(30)
    return 'TIMEOUT',last

def session_status(out):
    p=out/'STAGE6A_EXPORT'/'SESSION_STATUS.json'
    if not p.exists(): raise RuntimeError('SESSION_STATUS.json missing after Stage6A harvest.')
    return json.loads(p.read_text(encoding='utf-8'))

def harvest(k,root,dgp):
    out=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE6A_HARVEST'/f'DGP{dgp}'
    shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    run([k,'kernels','output',KERNEL_REF,'-p',str(out),'-o'])
    sj=session_status(out)
    src=out/'STAGE6A_EXPORT'/'07_METHOD_DEV_RUNS'
    dst=root/'07_METHOD_DEV_RUNS'; dst.mkdir(parents=True,exist_ok=True)
    if src.exists():
        for rd in src.iterdir():
            if rd.is_dir(): shutil.copytree(rd,dst/rd.name,dirs_exist_ok=True)
    return sj,out

def main():
    root=Path.cwd()
    print('B01 root:',root)
    print('Kaggle lock: prinprin / Avast Browser')
    print('Stage6A = NEW post-Stage5A method development on DEV only.')
    print('Original Primary and Stage5A are frozen. TEST is forbidden in Stage6A.')

    lock=root/'00_CONFIG'/'stage6a_q1_method_development_lock.yaml'
    runner=root/'01_CODE'/'scripts'/'run_stage6a_q1_development.py'
    if not lock.exists() or not runner.exists():
        raise SystemExit('Stage6A files missing. Apply runtime package inside the true B01 root.')

    k=ensure_kaggle()
    if not auth_check(k):
        run([k,'auth','login','--force'],capture=False)
        if not auth_check(k): raise SystemExit('Kaggle authentication not completed.')

    expected={(d,n) for d in DGP_SEEDS for n in NET_SEEDS}
    if not expected.issubset(complete_pairs(root)):
        ddir,z=prepare_dataset(root)
        print('STAGE6A BUNDLE',z)
        ensure_dataset(k,ddir)

        for dgp in DGP_SEEDS:
            pending=[(dgp,n) for n in NET_SEEDS if (dgp,n) not in complete_pairs(root)]
            if not pending: continue

            ex=exact_kernel_exists(k)
            if ex:
                st,_=query_status(k)
                if st in {'RUNNING','QUEUED'}:
                    print('Existing Stage6A kernel is active; attaching instead of pushing duplicate.')
                    final,_=poll_kernel(k)
                    if final!='COMPLETE': raise SystemExit('Existing Stage6A kernel did not complete.')
                    sj,_=harvest(k,root,dgp)
                    if sj.get('status')!='PASS':
                        raise SystemExit(f"Stage6A DGP{dgp} diagnostic FAIL ({sj.get('failure_kind')}).")
                    pending=[(dgp,n) for n in NET_SEEDS if (dgp,n) not in complete_pairs(root)]
                    if not pending: continue

            kdir=prepare_kernel(root,pending)
            print('Launching Stage6A DEV-only tasks',pending)
            r=run([k,'kernels','push','-p',str(kdir),'--accelerator','NvidiaTeslaT4','-t','43200'],check=False)
            if r.returncode!=0: raise SystemExit('Stage6A kernel push failed. Do not alter scientific settings.')
            final,_=poll_kernel(k)
            if final!='COMPLETE': raise SystemExit(f'Stage6A kernel ended as {final}.')
            sj,_=harvest(k,root,dgp)
            if sj.get('status')!='PASS':
                raise SystemExit(f"Stage6A DGP{dgp} diagnostic FAIL ({sj.get('failure_kind')}). Stop for diagnostics.")

    done=complete_pairs(root)
    if not expected.issubset(done):
        raise SystemExit(f'Stage6A incomplete: {len(done & expected)}/{len(expected)}.')

    ag=root/'01_CODE'/'scripts'/'aggregate_stage6a_q1_results.py'
    run([sys.executable,str(ag),'--root',str(root)],capture=False)
    print('STAGE6A COMPLETE: 10/10 development pairs harvested and selection rule applied.')
    print('Read 06_RESULTS/STAGE6A/STAGE6A_SELECTION.json before Stage6B.')

if __name__=='__main__':
    main()
