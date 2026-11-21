#!/usr/bin/env python3
"""Compare independent re-execution output with archived original benchmark results.
This program never runs training. It requires result files produced separately
by a completed re-execution of the original methods and seed grids.
Usage: python COMPARE_INDEPENDENT_RERUN.py --original-code-archive code.zip --rerun-results /path/to/06_RESULTS --report comparison.json
"""
from __future__ import annotations
import argparse, csv, io, json, math
from pathlib import Path
from zipfile import ZipFile

DECISIONS={
 'PRIMARY_GATE.json':['primary_gate','grid_complete'],
 'STAGE6A/STAGE6A_SELECTION.json':['status','selected_candidate','grid_complete'],
 'STAGE6B/STAGE6B_CONFIRMATORY_DECISION.json':['gates','selected_candidate','grid_complete'],
 'STAGE6C/STAGE6C_MECHANISM_DECISION.json':['gates','frozen_method','grid_complete'],
 'STAGE7B/STAGE7B_DEVELOPMENT_SELECTION.json':['gates','selected_certification','grid_complete'],
}
CSVS=[
 'ALL_CORE_METRICS.csv','PRIMARY_PAIRED_REPLICATES.csv',
 'STAGE5A/ALL_STAGE5A_METRICS.csv',
 'STAGE6B/ALL_STAGE6B_TEST_METRICS.csv',
 'STAGE6C/ALL_STAGE6C_MECHANISM_DEV_METRICS.csv',
 'STAGE7B/STAGE7B_METHOD_REGIME_METRICS.csv',
]
STAGE_FILES={
 'stage4':(['PRIMARY_GATE.json'],['ALL_CORE_METRICS.csv','PRIMARY_PAIRED_REPLICATES.csv']),
 'stage5a':([],['STAGE5A/ALL_STAGE5A_METRICS.csv']),
 'stage6a':(['STAGE6A/STAGE6A_SELECTION.json'],['STAGE6A/ALL_STAGE6A_DEV_METRICS.csv']),
 'stage6b':(['STAGE6B/STAGE6B_CONFIRMATORY_DECISION.json'],['STAGE6B/ALL_STAGE6B_TEST_METRICS.csv']),
 'stage6c':(['STAGE6C/STAGE6C_MECHANISM_DECISION.json'],['STAGE6C/ALL_STAGE6C_MECHANISM_DEV_METRICS.csv']),
 'stage7b':(['STAGE7B/STAGE7B_DEVELOPMENT_SELECTION.json'],['STAGE7B/STAGE7B_METHOD_REGIME_METRICS.csv']),
}
def read_table(binary):
 rows=list(csv.DictReader(io.StringIO(binary.decode('utf-8-sig'))))
 if not rows:raise ValueError('Empty table')
 return rows

def read_arc(z,rel):return z.read('original/06_RESULTS/'+rel)
def isfloat(s):
 try: return math.isfinite(float(s))
 except: return False

def compare_rows(a,b,rtol,atol):
 if len(a)!=len(b):return False,{'rows_old':len(a),'rows_new':len(b)}
 if set(a[0])!=set(b[0]):return False,{'error':'column mismatch'}
 cols=list(a[0])
 # Align rows by frozen run/model identities, including numeric seed columns.
 # Sorting by method alone incorrectly pairs different DGP/network seeds.
 volatile={'run','run_id','run_dir','created_unix','finished_unix','started','finished'}
 identifiers={'dgp_seed','net_seed','learner_seed','split_seed','patient_split_seed',
              'condition','method','regime','family','stratum','candidate','gamma'}
 keys=[c for c in cols if c not in volatile and c in identifiers]
 if not keys:raise ValueError('No stable row identity columns')
 sortkey=lambda row:tuple(row.get(c,'') for c in keys)
 old_keys=[sortkey(row) for row in a];new_keys=[sortkey(row) for row in b]
 if len(set(old_keys))!=len(a) or len(set(new_keys))!=len(b):
  raise ValueError('Non-unique comparison row identity: '+','.join(keys))
 if set(old_keys)!=set(new_keys):
  return False,{'error':'row identity mismatch','identity_columns':keys,
                'missing_from_rerun':[list(k) for k in sorted(set(old_keys)-set(new_keys))],
                'new_in_rerun':[list(k) for k in sorted(set(new_keys)-set(old_keys))]}
 a=sorted(a,key=sortkey);b=sorted(b,key=sortkey)
 errors=[]; differences=[]
 for i,(x,y) in enumerate(zip(a,b)):
  for k in cols:
   if k in volatile:continue
   va=x[k].strip();vb=y[k].strip()
   if isfloat(va) and isfloat(vb):
    delta=float(vb)-float(va)
    ok=math.isclose(float(va),float(vb),rel_tol=rtol,abs_tol=atol)
    differences.append({'row':i+1,'column':k,'original':va,'rerun':vb,'signed_difference':delta,'absolute_difference':abs(delta),'pass':ok})
    if not ok:errors.append({'row':i+1,'column':k,'original':va,'rerun':vb,'signed_difference':delta})
   elif va!=vb:
    errors.append({'row':i+1,'column':k,'original':va,'rerun':vb})
 return len(errors)==0,{'rows':len(a),'columns':len(cols),'mismatches':errors,'numeric_differences':differences}

def json_numbers(old,new,rtol,atol,prefix=''):
 differences=[]
 if isinstance(old,dict) and isinstance(new,dict):
  for key in old.keys() & new.keys():differences.extend(json_numbers(old[key],new[key],rtol,atol,prefix+'.'+key))
 elif isinstance(old,list) and isinstance(new,list):
  for i,(x,y) in enumerate(zip(old,new)):differences.extend(json_numbers(x,y,rtol,atol,prefix+f'[{i}]'))
 elif not isinstance(old,bool) and not isinstance(new,bool) and isinstance(old,(int,float)) and isinstance(new,(int,float)):
  delta=float(new)-float(old)
  differences.append({'field':prefix.lstrip('.'),'original':old,'rerun':new,'signed_difference':delta,'absolute_difference':abs(delta),'pass':math.isclose(float(old),float(new),rel_tol=rtol,abs_tol=atol)})
 return differences

PER_RUN={
 'stage6a':('07_METHOD_DEV_RUNS','summary_dev.csv','STAGE6A/ALL_STAGE6A_DEV_METRICS.csv','net_seed'),
 'stage6b':('08_CONFIRMATORY_RUNS','summary_test.csv','STAGE6B/ALL_STAGE6B_TEST_METRICS.csv','net_seed'),
 'stage6c':('08_MECHANISM_RUNS','mechanism_dev_metrics.csv','STAGE6C/ALL_STAGE6C_MECHANISM_DEV_METRICS.csv','net_seed'),
 'stage7b':('08_STAGE7_DEV_RUNS','baseline_dev_metrics.csv','STAGE7B/ALL_STAGE7B_BASELINE_DEV_METRICS.csv','learner_seed'),
}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--original-code-archive',required=True)
 ap.add_argument('--rerun-results',required=True);ap.add_argument('--report',required=True)
 ap.add_argument('--stages',default='stage4',help='comma-separated completed stages or all')
 ap.add_argument('--rerun-work',help='workspace containing individual run metrics; enables per-run checks')
 ap.add_argument('--per-run-only',action='store_true',help='compare completed individual runs before full-stage aggregation')
 ap.add_argument('--rtol',type=float,default=1e-3);ap.add_argument('--atol',type=float,default=1e-6)
 a=ap.parse_args();root=Path(a.rerun_results)
 if not root.is_dir():raise SystemExit('Rerun output directory does not exist')
 stages=list(STAGE_FILES) if a.stages=='all' else [x.strip() for x in a.stages.split(',')]
 if not stages or any(s not in STAGE_FILES for s in stages):ap.error('Unknown stage')
 decisions=[] if a.per_run_only else [x for s in stages for x in STAGE_FILES[s][0]]
 tables=[] if a.per_run_only else [x for s in stages for x in STAGE_FILES[s][1]]
 checks=[]
 with ZipFile(a.original_code_archive) as z:
  for rel in decisions:
   keys=DECISIONS[rel]
   try:
    old=json.loads(read_arc(z,rel));new=json.loads((root/rel).read_text(encoding='utf-8'))
    mismatch=[key for key in keys if old.get(key)!=new.get(key)]
    differences=json_numbers(old,new,a.rtol,a.atol)
    checks.append({'file':rel,'type':'scientific_decision','pass':not mismatch and all(x['pass'] for x in differences),'mismatched_decisions':mismatch,'numeric_differences':differences})
   except Exception as e:checks.append({'file':rel,'type':'scientific_decision','pass':False,'error':str(e)})
  for rel in tables:
   try:
    old=read_table(read_arc(z,rel));new=read_table((root/rel).read_bytes())
    ok,info=compare_rows(old,new,a.rtol,a.atol)
    checks.append({'file':rel,'type':'numeric_comparison','pass':ok,**info})
   except Exception as e:checks.append({'file':rel,'type':'numeric_comparison','pass':False,'error':str(e)})
  if a.rerun_work and 'stage4' in stages:
   import csv
   work=Path(a.rerun_work)
   archive=read_table(read_arc(z,'ALL_CORE_METRICS.csv'))
   for manifest in sorted((work/'B01'/'04_RUNS').glob('CORE_DGP*_NET*/RUN_MANIFEST.json')):
    try:
     m=json.loads(manifest.read_text(encoding='utf-8'));cfg=m['config']
     old=[r for r in archive if int(r['dgp_seed'])==int(cfg['dgp_seed']) and int(r['net_seed'])==int(cfg['net_seed'])]
     new=read_table((manifest.parent/'metrics'/'summary.csv').read_bytes())
     old_by_method={r['method']:r for r in old};errors=[];differences=[]
     for row in new:
      expected=old_by_method.get(row['method'])
      if expected is None:errors.append({'method':row['method'],'error':'missing archived method'});continue
      for col in ['root_pehe','policy_regret','cate_calibration_error']:
       x=float(expected[col]);y=float(row[col]);delta=y-x;ok=math.isclose(x,y,rel_tol=a.rtol,abs_tol=a.atol)
       differences.append({'method':row['method'],'metric':col,'original':x,'rerun':y,'signed_difference':delta,'absolute_difference':abs(delta),'pass':ok})
       if not ok:errors.append(differences[-1])
     checks.append({'file':str(manifest.parent.name+'/metrics/summary.csv'),'type':'per_run','pass':not errors and len(new)==len(old),'numeric_differences':differences,'mismatches':errors,'rows_old':len(old),'rows_new':len(new)})
    except Exception as e:checks.append({'file':str(manifest),'type':'per_run','pass':False,'error':str(e)})
  if a.rerun_work:
   work=Path(a.rerun_work)
   for stage in stages:
    if stage not in PER_RUN:continue
    folder,metric_file,archive_file,seed_key=PER_RUN[stage]
    archived=read_table(read_arc(z,archive_file))
    for mf in sorted((work/'B01'/folder).glob('*/RUN_MANIFEST.json')):
     try:
      meta=json.loads(mf.read_text(encoding='utf-8'));cfg=meta.get('config',meta)
      if meta.get('status')!='COMPLETE':continue
      old=[r for r in archived if int(r['dgp_seed'])==int(cfg['dgp_seed']) and int(r[seed_key])==int(cfg[seed_key])]
      new=read_table((mf.parent/'metrics'/metric_file).read_bytes())
      if old and 'run_dir' in old[0] and 'run_dir' not in new[0]:
       for row in new:row['run_dir']=mf.parent.name
      ok,info=compare_rows(old,new,a.rtol,a.atol)
      checks.append({'file':str(mf.parent.name+'/metrics/'+metric_file),'type':'per_run','stage':stage,'pass':ok,**info})
     except Exception as e:checks.append({'file':str(mf),'type':'per_run','stage':stage,'pass':False,'error':str(e)})
 report={'rtol':a.rtol,'atol':a.atol,'source':'Completed externally regenerated original-stage result tree',
         'stages':stages,
         'all_comparisons_pass':all(c['pass'] for c in checks),'checks':checks}
 Path(a.report).parent.mkdir(parents=True,exist_ok=True)
 Path(a.report).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 print('PASS' if report['all_comparisons_pass'] else 'FAIL',sum(c['pass'] for c in checks),'/',len(checks))
 raise SystemExit(0 if report['all_comparisons_pass'] else 2)
if __name__=='__main__': main()
