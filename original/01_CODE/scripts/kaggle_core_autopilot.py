from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time, zipfile, hashlib, site, re
from pathlib import Path

OWNER='prinprin'
DATASET_SLUG='b01-causaltx-mm-core-runtime'
DATASET_REF=f'{OWNER}/{DATASET_SLUG}'
KERNEL_SLUG='b01-causaltx-mm-core'
KERNEL_REF=f'{OWNER}/{KERNEL_SLUG}'


def run(cmd, check=True, cwd=None, capture=True):
    print('>', ' '.join(str(x) for x in cmd), flush=True)
    r=subprocess.run([str(x) for x in cmd],cwd=cwd,text=True,capture_output=capture)
    if r.stdout: print(r.stdout.strip(), flush=True)
    if r.stderr: print(r.stderr.strip(), flush=True)
    if check and r.returncode!=0: raise RuntimeError(f'Command failed ({r.returncode}): {cmd}')
    return r


def find_kaggle():
    p=shutil.which('kaggle')
    if p: return p
    cands=[]; exe=Path(sys.executable)
    cands += [exe.parent/'Scripts'/'kaggle.exe', exe.parent/'kaggle.exe']
    try: cands += [Path(site.USER_BASE)/'Scripts'/'kaggle.exe']
    except Exception: pass
    for c in cands:
        if c.exists(): return str(c)
    return None


def ensure_kaggle():
    k=find_kaggle()
    if k: return k
    print('Kaggle CLI not found. Installing official kaggle package...', flush=True)
    run([sys.executable,'-m','pip','install','--upgrade','kaggle'],check=True,capture=False)
    k=find_kaggle()
    if not k: raise RuntimeError('Kaggle CLI installed but kaggle executable was not found. Reopen terminal and rerun.')
    return k


def auth_check(k):
    r=run([k,'kernels','list','-m','--page-size','1'],check=False)
    return r.returncode==0


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def zip_tree(src, out):
    if out.exists(): out.unlink()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in src.rglob('*'):
            if not p.is_file(): continue
            if '__pycache__' in p.parts or p.suffix in {'.pyc','.pyo'} or '.pytest_cache' in p.parts: continue
            z.write(p,p.relative_to(src))


def sync_core_bundle(root):
    bundle=root/'09_REPRODUCIBILITY'/'KAGGLE_CORE_BUNDLE'; bundle.mkdir(parents=True,exist_ok=True)
    for name in ['00_CONFIG','01_CODE','03_STATE']:
        shutil.copytree(root/name,bundle/name,dirs_exist_ok=True)
    shutil.copy2(root/'requirements.txt',bundle/'requirements.txt')
    for rel in [Path('02_DATA/processed/ispy2_core_multimodal.npz'),Path('02_DATA/manifests/CORE_DATA_MANIFEST.json'),Path('02_DATA/manifests/PRETRAIN_MANIFEST_REVIEW.json')]:
        s=root/rel
        if s.exists():
            d=bundle/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(s,d)
    req=bundle/'02_DATA'/'processed'/'ispy2_core_multimodal.npz'
    if not req.exists(): raise RuntimeError('KAGGLE_CORE_BUNDLE lacks ispy2_core_multimodal.npz. Run Stage 2 again.')
    return bundle


def prepare_dataset(root,bundle):
    ddir=root/'09_REPRODUCIBILITY'/'KAGGLE_UPLOAD_DATASET'; shutil.rmtree(ddir,ignore_errors=True); ddir.mkdir(parents=True)
    z=ddir/'B01_CORE_BUNDLE.zip'; zip_tree(bundle,z)
    meta={
      'title':'B01 CausalTx Core Runtime','id':DATASET_REF,'licenses':[{'name':'other'}],
      'subtitle':'Private runtime bundle for locked CM-OSR experiments',
      'description':'Private computational transfer artifact for B01. Contains derived public GEO covariates, locked code/config/state, and manifests. Rights in source data remain with original authors/depositories.'
    }
    (ddir/'dataset-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (ddir/'B01_CORE_BUNDLE.sha256.txt').write_text(sha256(z)+'  B01_CORE_BUNDLE.zip\n',encoding='utf-8')
    return ddir,z


def ensure_dataset(k,ddir):
    # Always refresh the existing private runtime dataset with the CURRENT local bundle.
    # Kaggle/API timeouts are retried and never converted into a scientific-setting change.
    def _text(r):
        return ((r.stdout or '')+' '+(r.stderr or '')).lower()

    # First try to version the known dataset directly. This avoids the false
    # "create" branch when `datasets status` temporarily times out.
    versioned=False
    for attempt in range(1,5):
        r=run([k,'datasets','version','-p',str(ddir),'-m','Locked B01 core runtime refresh V8','-t'],check=False)
        t=_text(r)
        if r.returncode==0 and not any(x in t for x in ['error:', 'creation error']):
            versioned=True
            break
        if any(x in t for x in ['not found','does not exist','404']):
            break
        if any(x in t for x in ['timed out','timeout','max retries exceeded','connection']):
            print(f'Dataset version API transient failure {attempt}/4; retrying without changing science.',flush=True)
            time.sleep(20)
            continue
        # Some Kaggle CLI builds print duplicate/title errors with unusual exit codes.
        if 'already in use by a dataset' in t or 'already exists' in t:
            # Dataset exists; retry version after propagation delay.
            time.sleep(20)
            continue
        break

    if not versioned:
        # Only create if versioning indicates the dataset is absent.
        created=False
        for attempt in range(1,4):
            r=run([k,'datasets','create','-p',str(ddir),'-t'],check=False)
            t=_text(r)
            if r.returncode==0 and 'creation error' not in t:
                created=True
                break
            if 'already in use by a dataset' in t or 'already exists' in t:
                # It exists after all: version it, do not create a second dataset.
                r2=run([k,'datasets','version','-p',str(ddir),'-m','Locked B01 core runtime refresh V8 after create race','-t'],check=False)
                if r2.returncode==0:
                    versioned=True
                    break
            if any(x in t for x in ['timed out','timeout','max retries exceeded','connection']):
                time.sleep(20)
                continue
            if not versioned:
                raise RuntimeError('Could not create/version the private Kaggle runtime dataset. No scientific run was launched.')
        if not (created or versioned):
            raise RuntimeError('Could not refresh the private Kaggle runtime dataset after retries. No scientific run was launched.')

    for _ in range(60):
        r=run([k,'datasets','status',DATASET_REF],check=False)
        t=_text(r)
        if r.returncode==0 and any(x in t for x in ['ready','complete','active']): return
        time.sleep(15)
    raise RuntimeError('Private Kaggle runtime dataset did not reach READY within 15 minutes. No scientific run was launched.')


def complete_pairs(root):
    done=set(); runs=root/'04_RUNS'
    if not runs.exists(): return done
    for mf in runs.glob('CORE_DGP*_NET*/RUN_MANIFEST.json'):
        try:
            j=json.loads(mf.read_text(encoding='utf-8'))
            if j.get('status')!='COMPLETE': continue
            c=j.get('config',{}); done.add((int(c['dgp_seed']),int(c['net_seed'])))
        except Exception: pass
    return done


def kernel_script(tasks):
    # IMPORTANT: all top-level exceptions are caught and exported. This turns Python/runtime
    # failures into harvestable diagnostics instead of opaque KernelWorkerStatus.ERROR.
    return f'''from pathlib import Path\nimport json, os, shutil, subprocess, sys, time, traceback, zipfile, platform\nTASKS={tasks!r}\nWORK=Path('/kaggle/working/B01')\nEXPORT=Path('/kaggle/working/EXPORT')\nshutil.rmtree(EXPORT,ignore_errors=True); EXPORT.mkdir(parents=True,exist_ok=True)\n(EXPORT/'logs').mkdir(parents=True,exist_ok=True)\nstatus={{'tasks':TASKS,'started':time.time(),'device':'unknown','results':[],'status':'BOOTING','scientific_settings_changed':False}}\nenv={{'python':sys.version,'platform':platform.platform(),'cwd':os.getcwd(),'input_root_exists':Path('/kaggle/input').exists()}}\n\ndef save_artifacts():\n    try:\n        status['finished']=time.time()\n        (EXPORT/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2,default=str))\n        (EXPORT/'RUNTIME_ENV.json').write_text(json.dumps(env,indent=2,default=str))\n        if (WORK/'04_RUNS').exists(): shutil.copytree(WORK/'04_RUNS',EXPORT/'04_RUNS',dirs_exist_ok=True)\n        shutil.make_archive('/kaggle/working/B01_CORE_SESSION','zip',root_dir=EXPORT)\n    except Exception:\n        Path('/kaggle/working/EXPORT_WRITE_FAILURE.txt').write_text(traceback.format_exc())\n\ntry:\n    # Environment diagnostic (NO TRAINING)\n    for mod in ['numpy','pandas','sklearn','scipy','yaml','torch']:\n        try:\n            m=__import__(mod); env[mod+'_version']=getattr(m,'__version__','unknown')\n        except Exception as e:\n            env[mod+'_import_error']=repr(e)\n    try:\n        import torch\n        env['torch_cuda_available']=bool(torch.cuda.is_available())\n        env['torch_cuda_version']=getattr(torch.version,'cuda',None)\n        env['cuda_device_count']=int(torch.cuda.device_count())\n        if torch.cuda.is_available(): env['cuda_device_name']=torch.cuda.get_device_name(0)\n        status['device']='cuda' if torch.cuda.is_available() else 'cpu'\n    except Exception as e:\n        status['device']='torch_error:'+repr(e)\n    input_root=Path('/kaggle/input')\n    input_files=[str(p) for p in input_root.rglob('*') if p.is_file()]\n    env['input_file_count']=len(input_files); env['input_files_first_200']=input_files[:200]\n    zip_inputs=[p for p in input_root.rglob('B01_CORE_BUNDLE.zip') if p.is_file()]\n    dir_inputs=[p for p in input_root.rglob('B01_CORE_BUNDLE') if p.is_dir()]\n    # Fallback for a future Kaggle mount layout: locate the bundle by its locked structure.\n    if not dir_inputs:\n        for req in input_root.rglob('requirements.txt'):\n            parent=req.parent\n            if (parent/'01_CODE').is_dir() and (parent/'00_CONFIG').is_dir() and (parent/'02_DATA'/'processed'/'ispy2_core_multimodal.npz').is_file():\n                dir_inputs.append(parent); break\n    shutil.rmtree(WORK,ignore_errors=True); WORK.mkdir(parents=True,exist_ok=True)\n    if zip_inputs:\n        env['bundle_mount_mode']='zip'\n        env['bundle_source']=str(zip_inputs[0])\n        with zipfile.ZipFile(zip_inputs[0]) as z: z.extractall(WORK)\n    elif dir_inputs:\n        env['bundle_mount_mode']='extracted_directory'\n        env['bundle_source']=str(dir_inputs[0])\n        shutil.copytree(dir_inputs[0],WORK,dirs_exist_ok=True)\n    else:\n        raise RuntimeError('B01 core runtime bundle not found in Kaggle inputs as ZIP or extracted directory.')\n    data=WORK/'02_DATA'/'processed'/'ispy2_core_multimodal.npz'\n    script=WORK/'01_CODE'/'scripts'/'run_full_core_benchmark.py'\n    cfg=WORK/'00_CONFIG'/'project.yaml'\n    for req in [data,script,cfg]:\n        if not req.exists(): raise RuntimeError('Required runtime file missing after extraction: '+str(req))\n    # Import/CLI probe: imports the exact scientific stack but argparse --help exits before any training.\n    probe=subprocess.run([sys.executable,str(script),'--help'],text=True,capture_output=True)\n    (EXPORT/'logs'/'IMPORT_PROBE.stdout.txt').write_text(probe.stdout or '')\n    (EXPORT/'logs'/'IMPORT_PROBE.stderr.txt').write_text(probe.stderr or '')\n    env['import_probe_returncode']=probe.returncode\n    if probe.returncode!=0: raise RuntimeError('Scientific runtime import probe failed; see IMPORT_PROBE.stderr.txt')\n    # Data-load probe: shapes/keys only, no fitting and no outcomes are inspected for tuning.\n    import numpy as np\n    zz=np.load(data,allow_pickle=False)\n    env['npz_keys']=list(zz.files)\n    env['npz_shapes']={{k:list(zz[k].shape) for k in zz.files if k in ['clinical','transcriptomic','rppa']}}\n    status['status']='RUNNING'\n    print('RUNTIME_PREFLIGHT_PASS',json.dumps(env,default=str),flush=True)\n    # Scientific tasks: locked seeds/methods/settings only.\n    for dgp,net in TASKS:\n        cmd=[sys.executable,str(script),'--root',str(WORK),'--data',str(data),'--dgp-seed',str(dgp),'--net-seed',str(net)]\n        print('RUN',cmd,flush=True); t0=time.time()\n        r=subprocess.run(cmd,text=True,capture_output=True)\n        logbase=EXPORT/'logs'/f'DGP{{dgp}}_NET{{net}}'\n        (Path(str(logbase)+'.stdout.txt')).write_text(r.stdout or '')\n        (Path(str(logbase)+'.stderr.txt')).write_text(r.stderr or '')\n        print(r.stdout,flush=True); print(r.stderr,flush=True)\n        status['results'].append({{'dgp_seed':dgp,'net_seed':net,'returncode':r.returncode,'seconds':time.time()-t0}})\n        if r.returncode!=0:\n            status['status']='FAIL'; status['failed_task']=[dgp,net]; status['failure_kind']='scientific_subprocess_error'; break\n    else:\n        status['status']='PASS'\nexcept BaseException as e:\n    status['status']='FAIL'; status['failure_kind']='wrapper_or_runtime_error'; status['exception']=repr(e); status['traceback']=traceback.format_exc()\n    (EXPORT/'KERNEL_EXCEPTION.txt').write_text(status['traceback'])\nfinally:\n    save_artifacts()\n    print(json.dumps(status,indent=2,default=str),flush=True)\n# Deliberately do not re-raise: internal SESSION_STATUS controls success/failure so diagnostics are harvestable.\n'''


def prepare_kernel(root,tasks):
    kdir=root/'09_REPRODUCIBILITY'/'KAGGLE_CORE_KERNEL'; shutil.rmtree(kdir,ignore_errors=True); kdir.mkdir(parents=True)
    (kdir/'b01_core_session.py').write_text(kernel_script(tasks),encoding='utf-8')
    meta={'id':KERNEL_REF,'title':'B01 CausalTx MM Core','code_file':'b01_core_session.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':[DATASET_REF],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (kdir/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return kdir


def parse_kernel_status_text(stdout='',stderr=''):
    """Parse only explicit Kaggle worker states.
    Transport/API errors such as ConnectTimeoutError are TRANSIENT, not worker failures.
    """
    raw=((stdout or '')+' '+(stderr or '')).strip()
    m=re.search(r'KernelWorkerStatus\.([A-Z_]+)',raw,re.I)
    if m:
        st=m.group(1).upper()
        if st in {'COMPLETE','COMPLETED'}: return 'COMPLETE',raw
        if st in {'ERROR','FAILED','CANCELLED','CANCELED'}: return 'FAILED',raw
        if st in {'RUNNING'}: return 'RUNNING',raw
        if st in {'QUEUED','PENDING'}: return 'QUEUED',raw
        return st,raw
    low=raw.lower()
    if any(x in low for x in ['connecttimeouterror','readtimeouterror','connectionerror',
                               'max retries exceeded','timed out','timeout',
                               'temporary failure','connection reset','name resolution']):
        return 'TRANSIENT',raw
    return 'UNKNOWN',raw


def query_kernel_status(k):
    r=run([k,'kernels','status',KERNEL_REF],check=False)
    return parse_kernel_status_text(r.stdout or '',r.stderr or '')


def poll_kernel(k,timeout_sec=42000):
    t=time.time(); last='UNKNOWN'; transient_count=0
    while time.time()-t<timeout_sec:
        st,raw=query_kernel_status(k); last=raw
        if st=='COMPLETE': return 'COMPLETE',last
        if st=='FAILED': return 'FAILED',last
        if st in {'RUNNING','QUEUED'}:
            transient_count=0
            time.sleep(60)
            continue
        if st in {'TRANSIENT','UNKNOWN'}:
            transient_count+=1
            print(f'Kaggle status API is temporarily unavailable ({st}, count={transient_count}). '
                  'The existing kernel is NOT being marked failed and no duplicate kernel will be pushed.',
                  flush=True)
            time.sleep(60)
            continue
        time.sleep(60)
    return 'TIMEOUT',last


def resolve_existing_current_session(k,max_wait_sec=900):
    """Resolve the latest Kaggle session before pushing a new version.
    This prevents duplicate scientific runs after a local/API polling interruption.
    """
    t=time.time(); last=''
    while time.time()-t<max_wait_sec:
        st,raw=query_kernel_status(k); last=raw
        if st in {'RUNNING','QUEUED','COMPLETE','FAILED'}:
            return st,raw
        print('Current Kaggle session status could not be resolved because of a transient API problem. '
              'Waiting rather than pushing a duplicate kernel.',flush=True)
        time.sleep(30)
    return 'UNRESOLVED',last


def download_any_outputs(k,root,label):
    out=root/'09_REPRODUCIBILITY'/'KAGGLE_FAILURE_DIAGNOSTICS'/label
    shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True,exist_ok=True)
    files=run([k,'kernels','files',KERNEL_REF,'--page-size','200'],check=False)
    (out/'KERNEL_FILES_LIST.txt').write_text((files.stdout or '')+'\n'+(files.stderr or ''),encoding='utf-8')
    dl=run([k,'kernels','output',KERNEL_REF,'-p',str(out),'-o'],check=False)
    (out/'KERNEL_OUTPUT_DOWNLOAD.txt').write_text((dl.stdout or '')+'\n'+(dl.stderr or ''),encoding='utf-8')
    return out


def _is_transient_api_text(s):
    s=(s or '').lower()
    return any(x in s for x in [
        'connecttimeouterror','readtimeouterror','connectionerror',
        'max retries exceeded','timed out','timeout',
        'temporary failure','connection reset','name resolution',
        'remote end closed connection'
    ])


def _retry_kaggle_output(k, out, attempts=10):
    last=''
    for attempt in range(1,attempts+1):
        r=run([k,'kernels','output',KERNEL_REF,'-p',str(out),'-o'],check=False)
        text=((r.stdout or '')+' '+(r.stderr or ''))
        last=text
        if r.returncode==0:
            return True,last
        if _is_transient_api_text(text):
            wait=min(30*attempt,180)
            print(f'Kaggle API transient failure during harvest ({attempt}/{attempts}); '
                  f'waiting {wait}s and retrying the SAME completed kernel. '
                  'No new kernel will be pushed.',flush=True)
            time.sleep(wait)
            continue
        return False,last
    return False,last


def harvest(k,root,dgp):
    out=root/'09_REPRODUCIBILITY'/'KAGGLE_HARVEST'/f'DGP{dgp}'
    # Preserve partial downloads from a previous API timeout.
    out.mkdir(parents=True,exist_ok=True)

    ok,last=_retry_kaggle_output(k,out,attempts=10)
    if not ok:
        if _is_transient_api_text(last):
            raise RuntimeError(
                f'Kaggle kernel is COMPLETE but the output API remained temporarily unavailable for DGP{dgp}. '
                'No duplicate kernel was launched. Rerun later to retry harvesting the same completed kernel.'
            )
        raise RuntimeError(f'Non-transient Kaggle output download failure for DGP{dgp}: {last}')

    z=out/'B01_CORE_SESSION.zip'
    if not z.exists():
        hits=list(out.rglob('B01_CORE_SESSION.zip'))
        if hits: z=hits[0]

    # If the compact ZIP is not visible yet, retry the SAME harvest a few more times.
    if not z.exists():
        for attempt in range(1,7):
            wait=min(30*attempt,120)
            print(f'Completed kernel output is only partially visible; retrying same harvest in {wait}s.',flush=True)
            time.sleep(wait)
            ok,last=_retry_kaggle_output(k,out,attempts=5)
            if not ok and not _is_transient_api_text(last):
                break
            z=out/'B01_CORE_SESSION.zip'
            if not z.exists():
                hits=list(out.rglob('B01_CORE_SESSION.zip'))
                if hits: z=hits[0]
            if z.exists():
                break

    if not z.exists():
        raise RuntimeError(
            f'Kaggle kernel is COMPLETE but B01_CORE_SESSION.zip is not yet harvestable for DGP{dgp}. '
            'No duplicate kernel was launched. Rerun later to retry harvest only.'
        )

    ex=out/'extracted'
    shutil.rmtree(ex,ignore_errors=True)
    ex.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(z) as zz:
        zz.extractall(ex)

    sf=ex/'SESSION_STATUS.json'
    if not sf.exists():
        hits=list(ex.rglob('SESSION_STATUS.json'))
        if hits: sf=hits[0]
    if not sf.exists():
        raise RuntimeError(
            f'SESSION_STATUS.json missing after harvesting COMPLETE DGP{dgp}. '
            'No duplicate kernel was launched.'
        )

    sj=json.loads(sf.read_text(encoding='utf-8'))
    src=ex/'04_RUNS'; dst=root/'04_RUNS'; dst.mkdir(parents=True,exist_ok=True)
    if src.exists():
        for p in src.iterdir():
            if p.is_dir():
                shutil.copytree(p,dst/p.name,dirs_exist_ok=True)
    return sj,out


def write_progress(root,obj):
    p=root/'03_STATE'/'KAGGLE_CORE_PROGRESS.json'; p.write_text(json.dumps(obj,indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args(); root=Path(a.root)
    pf=root/'03_STATE'/'CORE_PREFLIGHT_FAIRNESS.json'
    if not pf.exists() or json.loads(pf.read_text(encoding='utf-8')).get('status')!='PASS':
        raise SystemExit('CORE_PREFLIGHT_FAIRNESS PASS is required. Run RUN_STAGE3_PREFLIGHT_B01.cmd first.')

    k=ensure_kaggle()
    if not auth_check(k):
        print('\nKAGGLE CLI AUTH REQUIRED. The OAuth window/URL must be authorized as prinprin.\n',flush=True)
        r=run([k,'auth','login','--force'],check=False,capture=False)
        if r.returncode!=0 or not auth_check(k):
            raise SystemExit('Kaggle OAuth not completed. Rerun and authorize prinprin.')

    dgp_seeds=[101,211,307,401,503]
    net_seeds=[17,29,43]
    prog={
        'status':'RUNNING',
        'dataset':DATASET_REF,
        'kernel':KERNEL_REF,
        'sessions':[],
        'locked_account':'prinprin',
        'runtime_fix':'V8_RESUME_BEFORE_DATASET_REFRESH',
        'scientific_settings_changed':False
    }
    write_progress(root,prog)

    # Critical V8 rule:
    # Do NOT rebuild/version the runtime dataset merely to harvest or reattach to
    # an already-running/already-complete kernel. Refresh the private dataset only
    # immediately before a genuinely NEW kernel push.
    runtime_dataset_ready=False

    def ensure_runtime_dataset_for_new_push():
        nonlocal runtime_dataset_ready
        if runtime_dataset_ready:
            return
        print('A new Kaggle kernel is actually required. Preparing/versioning the locked runtime dataset now.',flush=True)
        bundle=sync_core_bundle(root)
        ddir,z=prepare_dataset(root,bundle)
        print('CORE BUNDLE',z,'SHA256',sha256(z),flush=True)
        ensure_dataset(k,ddir)
        runtime_dataset_ready=True

    resume_probe_done=False

    for dgp in dgp_seeds:
        done=complete_pairs(root)
        tasks=[(dgp,n) for n in net_seeds if (dgp,n) not in done]

        if not tasks:
            print('SKIP DGP',dgp,'all 3 network seeds already COMPLETE')
            prog['sessions'].append({'dgp_seed':dgp,'status':'ALREADY_COMPLETE'})
            write_progress(root,prog)
            continue

        print('Locked tasks pending:',tasks,flush=True)
        worker_status=None
        raw=''
        reused_existing=False

        # FIRST action for the first incomplete DGP: inspect/attach/harvest the
        # current Kaggle session BEFORE any dataset upload/version action.
        if not resume_probe_done:
            resume_probe_done=True
            cur,cur_raw=resolve_existing_current_session(k)

            if cur=='UNRESOLVED':
                prog['status']='WAITING_FOR_KAGGLE_API'
                prog['sessions'].append({
                    'dgp_seed':dgp,
                    'status':'UNRESOLVED_CURRENT_SESSION',
                    'raw_status':cur_raw
                })
                write_progress(root,prog)
                raise SystemExit(
                    'Kaggle API remained unreachable for 15 minutes. '
                    'No dataset refresh and no new kernel were started. '
                    'Rerun later; V8 will check the same current session first.'
                )

            if cur in {'RUNNING','QUEUED'}:
                print(
                    f'Existing Kaggle session is {cur}. Attaching to it BEFORE any dataset refresh. '
                    'No duplicate kernel will be pushed.',
                    flush=True
                )
                worker_status,raw=poll_kernel(k)
                reused_existing=True

            elif cur=='COMPLETE':
                print(
                    'Existing Kaggle session is COMPLETE. Harvesting that SAME run BEFORE any dataset refresh.',
                    flush=True
                )
                worker_status='COMPLETE'
                raw=cur_raw
                reused_existing=True

            elif cur=='FAILED':
                print(
                    'Latest Kaggle session has an explicit worker FAILED/ERROR state. '
                    'Only now may an identical new infrastructure attempt be considered.',
                    flush=True
                )

        # Resume/harvest existing session.
        if reused_existing and worker_status=='COMPLETE':
            sj,out=harvest(k,root,dgp)
            sj_tasks=[tuple(x) for x in sj.get('tasks',[])]

            if sj_tasks==tasks:
                prog['sessions'].append({
                    'dgp_seed':dgp,
                    'status':sj.get('status'),
                    'details':sj,
                    'harvest':str(out),
                    'resumed_existing_session':True
                })
                write_progress(root,prog)

                if sj.get('status')!='PASS':
                    prog['status']='STOPPED'
                    write_progress(root,prog)
                    raise SystemExit(
                        f"Kaggle session DGP{dgp} produced harvestable diagnostic FAIL "
                        f"({sj.get('failure_kind')}). Inspect {out/'extracted'}; "
                        "do not change scientific settings."
                    )

                print(f'DGP{dgp} PASS harvested from existing Kaggle session.',flush=True)
                continue

            print(
                'Current COMPLETE session does not match the currently pending locked task list. '
                'It will not be counted as this DGP.',
                flush=True
            )
            worker_status=None

        if reused_existing and worker_status=='FAILED':
            diag=download_any_outputs(k,root,f'DGP{dgp}_EXISTING_WORKER_FAILED')
            print('Explicit worker failure diagnostics saved to',diag,flush=True)
            worker_status=None

        if reused_existing and worker_status=='TIMEOUT':
            prog['status']='WAITING_FOR_CURRENT_KERNEL'
            prog['sessions'].append({
                'dgp_seed':dgp,
                'status':'POLL_TIMEOUT_NO_DUPLICATE_PUSH',
                'raw_status':raw
            })
            write_progress(root,prog)
            raise SystemExit(
                'Current Kaggle kernel could not be resolved before the local polling horizon. '
                'No dataset refresh and no duplicate kernel were started. '
                'Rerun later to reattach.'
            )

        # We reach here only if a NEW kernel is actually required.
        ensure_runtime_dataset_for_new_push()
        kdir=prepare_kernel(root,tasks)
        print('Launching NEW Kaggle session for',tasks,flush=True)

        worker_status='FAILED'
        raw=''

        for infra_attempt in (1,2):
            print(f'Kaggle infrastructure attempt {infra_attempt}/2 for locked tasks {tasks}',flush=True)
            run([k,'kernels','push','-p',str(kdir),'--accelerator','NvidiaTeslaT4','-t','43200'])
            worker_status,raw=poll_kernel(k)

            if worker_status=='COMPLETE':
                break

            if worker_status=='TIMEOUT':
                prog['status']='WAITING_FOR_CURRENT_KERNEL'
                prog['sessions'].append({
                    'dgp_seed':dgp,
                    'status':'POLL_TIMEOUT_NO_DUPLICATE_PUSH',
                    'raw_status':raw
                })
                write_progress(root,prog)
                raise SystemExit(
                    'Kaggle kernel polling reached the local horizon. '
                    'No duplicate kernel was pushed. Rerun later to reattach.'
                )

            diag=download_any_outputs(k,root,f'DGP{dgp}_WORKER_{worker_status}_ATTEMPT{infra_attempt}')
            print('Explicit worker-level failure diagnostics saved to',diag,flush=True)

            if infra_attempt==1:
                print(
                    'No scientific settings changed. Waiting 90 seconds before one identical infrastructure retry.',
                    flush=True
                )
                time.sleep(90)

        if worker_status!='COMPLETE':
            prog['status']='STOPPED'
            prog['sessions'].append({
                'dgp_seed':dgp,
                'status':worker_status,
                'raw_status':raw
            })
            write_progress(root,prog)
            raise SystemExit(
                'Kaggle worker explicitly failed twice. '
                'No seeds/DGP/thresholds/methods were changed.'
            )

        sj,out=harvest(k,root,dgp)
        prog['sessions'].append({
            'dgp_seed':dgp,
            'status':sj.get('status'),
            'details':sj,
            'harvest':str(out)
        })
        write_progress(root,prog)

        if sj.get('status')!='PASS':
            prog['status']='STOPPED'
            write_progress(root,prog)
            raise SystemExit(
                f"Kaggle session DGP{dgp} produced harvestable diagnostic FAIL "
                f"({sj.get('failure_kind')}). Inspect {out/'extracted'}; "
                "do not change scientific settings."
            )

    done=complete_pairs(root)
    expected={(d,n) for d in dgp_seeds for n in net_seeds}
    prog['complete_pairs']=len(done & expected)
    prog['status']='CORE_GRID_COMPLETE' if expected.issubset(done) else 'INCOMPLETE'
    write_progress(root,prog)

    if not expected.issubset(done):
        raise SystemExit(f'Core grid incomplete: {len(done & expected)}/15')

    ag=root/'01_CODE'/'scripts'/'aggregate_core_results.py'
    run([sys.executable,str(ag),'--root',str(root)],capture=False)
    print('KAGGLE CORE GRID COMPLETE: 15/15 runs harvested locally. See 06_RESULTS/PRIMARY_GATE.json')

if __name__=='__main__': main()
