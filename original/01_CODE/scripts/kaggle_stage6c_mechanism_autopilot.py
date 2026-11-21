from __future__ import annotations
import json, re, shutil, site, subprocess, sys, time, zipfile
from pathlib import Path

OWNER='prinprin'
DATASET_REF=f'{OWNER}/b01-causaltx-mm-stage6c-mechanism-runtime'
KERNEL_REF=f'{OWNER}/b01-causaltx-mm-stage6c-mechanism'
DGP=[1811,1931,2053,2179,2297]
NET=[23,41]

def run(cmd,check=True,capture=True):
    print('>', ' '.join(str(x) for x in cmd),flush=True)
    r=subprocess.run([str(x) for x in cmd],text=True,capture_output=capture)
    if r.stdout: print(r.stdout.strip(),flush=True)
    if r.stderr: print(r.stderr.strip(),flush=True)
    if check and r.returncode!=0: raise RuntimeError(f'Command failed: {cmd}')
    return r

def kaggle():
    p=shutil.which('kaggle')
    if p:return p
    exe=Path(sys.executable)
    cands=[exe.parent/'Scripts'/'kaggle.exe',exe.parent/'kaggle.exe']
    try:cands.append(Path(site.USER_BASE)/'Scripts'/'kaggle.exe')
    except Exception:pass
    for c in cands:
        if c.exists():return str(c)
    raise RuntimeError('Kaggle CLI not found.')

def auth(k):
    r=run([k,'kernels','list','-m','--page-size','1'],check=False)
    t=((r.stdout or '')+' '+(r.stderr or '')).lower()
    if r.returncode==0 and 'authentication required' not in t:return
    run([k,'auth','login','--force'],capture=False)

def complete_pairs(root):
    out=set(); p=root/'08_MECHANISM_RUNS'; pat=re.compile(r'^MECH6C_DGP(\d+)_NET(\d+)_')
    if not p.exists():return out
    for d in p.iterdir():
        if d.is_dir() and (d/'COMPLETE.json').exists():
            m=pat.match(d.name)
            if m:out.add((int(m.group(1)),int(m.group(2))))
    return out

def prepare_dataset(root,k):
    run([sys.executable,str(root/'01_CODE'/'scripts'/'prepare_stage6c_bundle.py'),'--root',str(root)],capture=False)
    d=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE6C_UPLOAD_DATASET'
    rs=run([k,'datasets','status',DATASET_REF],check=False)
    low=((rs.stdout or '')+' '+(rs.stderr or '')).lower()
    if rs.returncode==0 and 'ready' in low:
        print('Existing Stage6C runtime dataset READY; reusing.',flush=True)
    else:
        rv=run([k,'datasets','version','-p',str(d),'-m','Locked Stage6C mechanism runtime','-t'],check=False)
        if rv.returncode!=0:
            rc=run([k,'datasets','create','-p',str(d),'-t'],check=False)
            if rc.returncode!=0:
                rs2=run([k,'datasets','status',DATASET_REF],check=False)
                if rs2.returncode!=0:raise RuntimeError('Could not create/reuse Stage6C dataset.')
    for _ in range(60):
        rs=run([k,'datasets','status',DATASET_REF],check=False)
        low=((rs.stdout or '')+' '+(rs.stderr or '')).lower()
        if rs.returncode==0 and 'ready' in low:
            print('Stage6C dataset READY; waiting 90 sec for attachment propagation.',flush=True)
            time.sleep(90);return
        if 'authentication required' in low:auth(k)
        time.sleep(15)
    raise RuntimeError('Stage6C dataset not READY.')

def kernel_script(tasks):
    tmpl = r'''from pathlib import Path
import json,shutil,subprocess,sys,time,traceback,zipfile
TASKS=__TASKS__
WORK=Path('/kaggle/working/B01')
EXPORT=Path('/kaggle/working/STAGE6C_EXPORT'); EXPORT.mkdir(parents=True,exist_ok=True)
(EXPORT/'logs').mkdir(parents=True,exist_ok=True)
status={'stage':'STAGE6C_MECHANISM_GENERALIZATION','tasks':[list(x) for x in TASKS],
        'evaluation_split':'DEV_ONLY_SECONDARY_MECHANISM','test_accessed':False,
        'candidate_reselection':False,'started':time.time(),'status':'STARTING','results':[]}
def save():
    status['finished']=time.time()
    (EXPORT/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2,default=str))
    src=WORK/'08_MECHANISM_RUNS'; dst=EXPORT/'08_MECHANISM_RUNS'
    if src.exists():shutil.copytree(src,dst,dirs_exist_ok=True)
    z=Path('/kaggle/working/B01_STAGE6C_SESSION.zip')
    if z.exists():z.unlink()
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
        for p in EXPORT.rglob('*'):
            if p.is_file():zz.write(p,p.relative_to(EXPORT))
try:
    inp=Path('/kaggle/input')
    inv=[str(p) for p in inp.rglob('*')] if inp.exists() else []
    (EXPORT/'KAGGLE_INPUT_INVENTORY.txt').write_text('\n'.join(inv))
    zips=[p for p in inp.rglob('B01_STAGE6C_BUNDLE.zip') if p.is_file()] if inp.exists() else []
    shutil.rmtree(WORK,ignore_errors=True); WORK.mkdir(parents=True,exist_ok=True)
    if zips:
        status['input_mode']='bundle_zip'
        with zipfile.ZipFile(zips[0]) as zz:zz.extractall(WORK)
    else:
        roots=[]
        if inp.exists():
            for lockp in inp.rglob('stage6c_mechanism_lock.yaml'):
                if lockp.parent.name=='00_CONFIG':
                    r=lockp.parent.parent
                    if (r/'01_CODE'/'scripts'/'run_stage6c_mechanism.py').exists() and (r/'02_DATA'/'processed'/'ispy2_core_multimodal.npz').exists():
                        roots.append(r)
        if not roots:raise RuntimeError('Stage6C runtime dataset not mounted. See KAGGLE_INPUT_INVENTORY.txt')
        status['input_mode']='already_extracted_dataset'
        shutil.copytree(roots[0],WORK,dirs_exist_ok=True)
    data=WORK/'02_DATA'/'processed'/'ispy2_core_multimodal.npz'
    runner=WORK/'01_CODE'/'scripts'/'run_stage6c_mechanism.py'
    probe=subprocess.run([sys.executable,str(runner),'--help'],text=True,capture_output=True)
    (EXPORT/'logs'/'IMPORT_PROBE.stdout.txt').write_text(probe.stdout or '')
    (EXPORT/'logs'/'IMPORT_PROBE.stderr.txt').write_text(probe.stderr or '')
    if probe.returncode!=0:raise RuntimeError('Stage6C import probe failed.')
    status['status']='RUNNING'
    for d,n in TASKS:
        cmd=[sys.executable,str(runner),'--root',str(WORK),'--data',str(data),'--dgp-seed',str(d),'--net-seed',str(n)]
        r=subprocess.run(cmd,text=True,capture_output=True)
        base=EXPORT/'logs'/f'STAGE6C_DGP{d}_NET{n}'
        Path(str(base)+'.stdout.txt').write_text(r.stdout or '')
        Path(str(base)+'.stderr.txt').write_text(r.stderr or '')
        print(r.stdout,flush=True); print(r.stderr,flush=True)
        status['results'].append({'dgp_seed':d,'net_seed':n,'returncode':r.returncode})
        if r.returncode!=0:
            status['status']='FAIL'; status['failure_kind']='scientific_subprocess_error'; status['failed_task']=[d,n]; break
    else:status['status']='PASS'
except BaseException as e:
    status['status']='FAIL'; status['failure_kind']='wrapper_or_runtime_error'
    status['exception']=repr(e); status['traceback']=traceback.format_exc()
    (EXPORT/'KERNEL_EXCEPTION.txt').write_text(status['traceback'])
finally:
    save(); print(json.dumps(status,indent=2,default=str),flush=True)
'''
    return tmpl.replace('__TASKS__',repr(tasks))

def prepare_kernel(root,tasks):
    d=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE6C_KERNEL'
    shutil.rmtree(d,ignore_errors=True); d.mkdir(parents=True)
    (d/'b01_stage6c_session.py').write_text(kernel_script(tasks),encoding='utf-8')
    meta={'id':KERNEL_REF,'title':'B01 CausalTx MM Stage6C Mechanism',
          'code_file':'b01_stage6c_session.py','language':'python','kernel_type':'script',
          'is_private':True,'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
          'dataset_sources':[DATASET_REF],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (d/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return d

def status(k):
    r=run([k,'kernels','status',KERNEL_REF],check=False)
    raw=((r.stdout or '')+' '+(r.stderr or '')).lower()
    if 'authentication required' in raw:return 'AUTH'
    m=re.search(r'kernelworkerstatus\.([a-z_]+)',raw)
    if m:
        s=m.group(1).upper()
        if s in {'COMPLETE','COMPLETED'}:return 'COMPLETE'
        if s=='RUNNING':return 'RUNNING'
        if s in {'QUEUED','PENDING'}:return 'QUEUED'
        if s in {'FAILED','ERROR','CANCELLED','CANCELED'}:return 'FAILED'
    if 'cannot access kernel' in raw or 'kernels.get' in raw:return 'AUTH'
    if 'not found' in raw:return 'NOT_FOUND'
    return 'UNKNOWN'

def wait(k):
    refresh=0
    while True:
        s=status(k); print('Stage6C status:',s,flush=True)
        if s in {'COMPLETE','FAILED','NOT_FOUND'}:return s
        if s=='AUTH':
            refresh+=1
            if refresh>3:return 'AUTH'
            auth(k);time.sleep(5);continue
        time.sleep(30)

def harvest(k,root,dgp):
    out=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE6C_HARVEST'/f'DGP{dgp}'
    shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    run([k,'kernels','output',KERNEL_REF,'-p',str(out),'-o'])
    ss=out/'STAGE6C_EXPORT'/'SESSION_STATUS.json'
    if not ss.exists():raise RuntimeError('Stage6C SESSION_STATUS missing.')
    d=json.loads(ss.read_text(encoding='utf-8'))
    src=out/'STAGE6C_EXPORT'/'08_MECHANISM_RUNS'; dst=root/'08_MECHANISM_RUNS'; dst.mkdir(parents=True,exist_ok=True)
    if src.exists():
        for rd in src.iterdir():
            if rd.is_dir():shutil.copytree(rd,dst/rd.name,dirs_exist_ok=True)
    return d

def main():
    root=Path.cwd()
    print('B01 root:',root)
    print('Stage6C mechanism validation - DEV only, TEST forbidden, no candidate reselection.')
    pre=root/'01_CODE'/'scripts'/'preflight_stage6c_mechanism.py'
    run([sys.executable,str(pre)],capture=False)
    k=kaggle();auth(k)
    expected={(d,n) for d in DGP for n in NET}
    if not expected.issubset(complete_pairs(root)):
        prepare_dataset(root,k)
        for d in DGP:
            pending=[(d,n) for n in NET if (d,n) not in complete_pairs(root)]
            if not pending:continue
            print('Launching locked Stage6C tasks',pending,flush=True)
            kd=prepare_kernel(root,pending)
            r=run([k,'kernels','push','-p',str(kd),'--accelerator','NvidiaTeslaT4','-t','43200'],check=False)
            if r.returncode!=0:raise SystemExit('Stage6C kernel push failed.')
            s=wait(k)
            if s!='COMPLETE':raise SystemExit(f'Stage6C kernel ended as {s}.')
            sj=harvest(k,root,d)
            if sj.get('status')!='PASS':
                raise SystemExit(f"Stage6C DGP{d} diagnostic FAIL ({sj.get('failure_kind')}).")
    done=complete_pairs(root)
    if not expected.issubset(done):
        raise SystemExit(f'Stage6C incomplete {len(done & expected)}/{len(expected)}')
    ag=root/'01_CODE'/'scripts'/'aggregate_stage6c_mechanism.py'
    run([sys.executable,str(ag),'--root',str(root)],capture=False)
    print('STAGE6C COMPLETE: 10/10 mechanism pairs harvested and interpretation rule applied.')
    print('Read 06_RESULTS/STAGE6C/STAGE6C_MECHANISM_DECISION.json')

if __name__=='__main__':
    main()
