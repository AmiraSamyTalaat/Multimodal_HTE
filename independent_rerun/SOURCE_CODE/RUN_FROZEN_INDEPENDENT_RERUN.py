#!/usr/bin/env python3
"""Resume-safe independent training rerun of frozen original benchmark seed grids.

Requires a private, locally reconstructed I-SPY2 processed NPZ and the complete
reviewer-code ZIP. Never bundles or uploads participant-level records.

Example (SMOKE):
 python RUN_FROZEN_INDEPENDENT_RERUN.py --release B01_Paper1_Complete_Reviewer_Code_Updated.zip --private-data ispy2_core_multimodal.npz --work private_work --stages stage4 --max-jobs 1

Example (all original stages, resumable):
 python RUN_FROZEN_INDEPENDENT_RERUN.py --release B01_Paper1_Complete_Reviewer_Code_Updated.zip --private-data ispy2_core_multimodal.npz --work private_work --stages all

Original experiments have a locked 5-DGP grid; this script does NOT turn them into
30 new independent DGP mechanisms. Post-review 30-DGP experiments remain separate.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from datetime import datetime,timezone

RUN_FOLDERS={'stage4':'04_RUNS','stage5a':'05_STRESS_RUNS','stage6a':'07_METHOD_DEV_RUNS','stage6b':'08_CONFIRMATORY_RUNS','stage6c':'08_MECHANISM_RUNS','stage7b':'08_STAGE7_DEV_RUNS'}
STAGES={
 'stage4':('run_full_core_benchmark.py','aggregate_core_results.py',[101,211,307,401,503],[17,29,43],'--net-seed'),
 'stage5a':('run_stage5a_secondary_stress.py','aggregate_stage5a_results.py',[101,211,307,401,503],[17,29,43],'--net-seed'),
 'stage6a':('run_stage6a_q1_development.py','aggregate_stage6a_q1_results.py',[607,719,823,941,1061],[19,37],'--net-seed'),
 'stage6b':('run_stage6b_confirmatory.py','aggregate_stage6b_confirmatory.py',[1201,1327,1451,1579,1699],[17,29,43],'--net-seed'),
 'stage6c':('run_stage6c_mechanism.py','aggregate_stage6c_mechanism.py',[1811,1931,2053,2179,2297],[23,41],'--net-seed'),
 'stage7b':('run_stage7b_cucmr_development.py','aggregate_stage7b_cucmr_development.py',[2411,2539,2671,2797,2909,2963],[31,47],'--learner-seed'),
}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def now():return datetime.now(timezone.utc).isoformat(timespec='seconds')

def atomic_json(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(path)

def prepare(release,privdata,work):
 if not privdata.is_file():raise FileNotFoundError('The private reconstructed I-SPY2 matrix is missing: '+str(privdata))
 project=work/'B01';project.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(release) as z:
  for x in z.infolist():
   if x.is_dir() or not x.filename.startswith('original/'):continue
   rel=x.filename[len('original/'):]
   if rel.startswith(('00_CONFIG/','01_CODE/','03_STATE/')) or rel=='requirements.txt':
    target=project/rel
    if not target.is_file():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(x))
 dest=project/'02_DATA/processed/ispy2_core_multimodal.npz';dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.is_file():shutil.copy2(privdata,dest)
 if sha(dest)!=sha(privdata):raise ValueError('Data file in workdir differs from supplied private input')
 (project/'06_RESULTS').mkdir(exist_ok=True)
 return project

def has_complete_manifest(project, stage, dgp_seed, learner_seed):
    folder=project/RUN_FOLDERS[stage]
    if not folder.exists():return False
    key='learner_seed' if stage=='stage7b' else 'net_seed'
    for mf in folder.rglob('RUN_MANIFEST.json'):
        try:
            meta=json.loads(mf.read_text(encoding='utf-8'))
            conf=meta.get('config',meta)
            if meta.get('status')=='COMPLETE' and int(conf.get('dgp_seed',-1))==dgp_seed and int(conf.get(key,-1))==learner_seed:
                return True
        except Exception:continue
    return False

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--release',type=Path,required=True)
 ap.add_argument('--private-data',type=Path,required=True)
 ap.add_argument('--work',type=Path,required=True)
 ap.add_argument('--stages',default='stage4',help='all or comma-separated stage4,stage5a,stage6a,stage6b,stage6c,stage7b')
 ap.add_argument('--max-jobs',type=int,default=0,help='0 = no limit; use 1 for a smoke test')
 ap.add_argument('--python',default=sys.executable)
 ap.add_argument('--dry-run',action='store_true')
 a=ap.parse_args()
 selected=list(STAGES) if a.stages=='all' else [s.strip() for s in a.stages.split(',')]
 if not selected or any(s not in STAGES for s in selected):ap.error('Unknown stage name')
 if not a.release.is_file():ap.error('Release ZIP does not exist')
 if not a.dry_run and not a.private_data.is_file():ap.error('Private data not found')
 project=None if a.dry_run else prepare(a.release,a.private_data,a.work)
 if not a.dry_run and 'stage6b' in selected and 'stage6a' not in selected:
  # Stage 6B was confirmatory with AOSR_X_gamma2 frozen before its TEST runs.
  # Reuse that exact archived selection as an explicit dependency; never reselect on TEST.
  selection=project/'06_RESULTS/STAGE6A/STAGE6A_SELECTION.json'
  if not selection.is_file():
   with zipfile.ZipFile(a.release) as z:
    payload=z.read('original/06_RESULTS/STAGE6A/STAGE6A_SELECTION.json')
   selection.parent.mkdir(parents=True,exist_ok=True);selection.write_bytes(payload)
   atomic_json(a.work/'progress'/'stage6b_archived_selection_dependency.json',
               {'source':'original/06_RESULTS/STAGE6A/STAGE6A_SELECTION.json','sha256':sha(selection),
                'dependency_type':'frozen_archived_stage6a_selection','copied_utc':now(),
                'note':'Stage6A grid is not independently rerun in this Stage6B job; candidate remains frozen.'})
 data=(a.work/'B01/02_DATA/processed/ispy2_core_multimodal.npz')
 statuses=a.work/'progress';statuses.mkdir(parents=True,exist_ok=True) if not a.dry_run else None
 done=0
 for stage in selected:
  script,aggregate,dgp,seed,flag=STAGES[stage]; jobs=[]
  for d in dgp:
   for n in seed:
    jobs.append((d,n))
  for d,n in jobs:
   tag=f'{stage}_DGP{d}_S{n}';marker=statuses/(tag+'.json');log=statuses/(tag+'.log')
   input_id={'release_sha256':sha(a.release),'data_sha256':sha(a.private_data) if a.private_data.is_file() else '', 'stage':stage,'dgp_seed':d,'learner_seed':n}
   if marker.is_file():
    meta=json.loads(marker.read_text(encoding='utf-8'))
    if meta.get('status')=='COMPLETE' and all(meta.get(k)==v for k,v in input_id.items()) and has_complete_manifest(project,stage,d,n):
     print('SKIP',tag,flush=True);continue
   args=[a.python,str((project or a.work/'B01')/'01_CODE/scripts'/script),'--root',str(project or a.work/'B01'), '--data',str(data),'--dgp-seed',str(d),flag,str(n)]
   if a.dry_run:
    print('PLAN',tag, ' '.join(args),flush=True)
    continue
   print('RUN',tag,flush=True)
   with open(log,'a',encoding='utf-8') as f:
    f.write('\nSTART '+now()+'\n'+' '.join(args)+'\n');f.flush()
    proc=subprocess.run(args,cwd=project,stdout=f,stderr=subprocess.STDOUT,check=False)
    f.write('FINISH '+now()+' RETURN '+str(proc.returncode)+'\n')
   validated=(proc.returncode==0 and has_complete_manifest(project,stage,d,n))
   meta={**input_id,'status':'COMPLETE' if validated else 'FAILED','returncode':proc.returncode,'run_manifest_complete':validated,'completed_utc':now(),'log_sha256':sha(log)}
   atomic_json(marker,meta)
   if not validated:raise RuntimeError(f'{tag} failed or lacks a matching COMPLETE RUN_MANIFEST; inspect {log}')
   done+=1
   if a.max_jobs and done>=a.max_jobs:
    print('BATCH_LIMIT',done,'training jobs; run same command again to resume')
    return
  if a.dry_run:continue
  ready=sum(1 for d,n in jobs if (statuses/f'{stage}_DGP{d}_S{n}.json').is_file() and json.loads((statuses/f'{stage}_DGP{d}_S{n}.json').read_text()).get('status')=='COMPLETE' and has_complete_manifest(project,stage,d,n))
  if ready==len(jobs):
   cmd=[a.python,str(project/'01_CODE/scripts'/aggregate),'--root',str(project)]
   print('AGGREGATE',stage,flush=True)
   p=subprocess.run(cmd,cwd=project,text=True,capture_output=True)
   (statuses/(stage+'_aggregate.log')).write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
   if p.returncode:raise RuntimeError(f'{stage} aggregation failed: {p.stderr[-1000:]}')
   atomic_json(statuses/(stage+'_aggregation_COMPLETE.json'),{'completed_utc':now(),'stage':stage,'jobs':len(jobs),'status':'COMPLETE'})
 print('FINISHED requested stages. Compare only the selected, completed stages with COMPARE_INDEPENDENT_RERUN.py.')
if __name__=='__main__':main()
