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
 cols=list(a[0]);paired=sorted(zip(a,b),key=lambda t:tuple(t[0].get(c,'') for c in cols))
 # Sort both sources independently by all fields that are *not* numeric.
 keys=[c for c in cols if any(not isfloat(r[c]) for r in a)]
 sortkey=lambda row:tuple(row.get(c,'') for c in keys)
 a=sorted(a,key=sortkey);b=sorted(b,key=sortkey)
 errors=[]
 for i,(x,y) in enumerate(zip(a,b)):
  for k in cols:
   va=x[k].strip();vb=y[k].strip()
   if isfloat(va) and isfloat(vb):
    if not math.isclose(float(va),float(vb),rel_tol=rtol,abs_tol=atol):
     if len(errors)<20:errors.append({'row':i+1,'column':k,'original':va,'rerun':vb})
   elif va!=vb:
    if len(errors)<20:errors.append({'row':i+1,'column':k,'original':va,'rerun':vb})
 return len(errors)==0,{'rows':len(a),'columns':len(cols),'first_mismatches':errors}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--original-code-archive',required=True)
 ap.add_argument('--rerun-results',required=True);ap.add_argument('--report',required=True)
 ap.add_argument('--rtol',type=float,default=1e-3);ap.add_argument('--atol',type=float,default=1e-6)
 a=ap.parse_args();root=Path(a.rerun_results)
 if not root.is_dir():raise SystemExit('Rerun output directory does not exist')
 checks=[]
 with ZipFile(a.original_code_archive) as z:
  for rel,keys in DECISIONS.items():
   try:
    old=json.loads(read_arc(z,rel));new=json.loads((root/rel).read_text(encoding='utf-8'))
    mismatch=[key for key in keys if old.get(key)!=new.get(key)]
    checks.append({'file':rel,'type':'scientific_decision','pass':not mismatch,'mismatched_decisions':mismatch})
   except Exception as e:checks.append({'file':rel,'type':'scientific_decision','pass':False,'error':str(e)})
  for rel in CSVS:
   try:
    old=read_table(read_arc(z,rel));new=read_table((root/rel).read_bytes())
    ok,info=compare_rows(old,new,a.rtol,a.atol)
    checks.append({'file':rel,'type':'numeric_comparison','pass':ok,**info})
   except Exception as e:checks.append({'file':rel,'type':'numeric_comparison','pass':False,'error':str(e)})
 report={'rtol':a.rtol,'atol':a.atol,'source':'Completed externally regenerated original-stage result tree',
         'all_comparisons_pass':all(c['pass'] for c in checks),'checks':checks}
 Path(a.report).parent.mkdir(parents=True,exist_ok=True)
 Path(a.report).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 print('PASS' if report['all_comparisons_pass'] else 'FAIL',sum(c['pass'] for c in checks),'/',len(checks))
 raise SystemExit(0 if report['all_comparisons_pass'] else 2)
if __name__=='__main__': main()
