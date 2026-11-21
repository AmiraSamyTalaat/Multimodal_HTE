from __future__ import annotations
import csv,importlib.metadata,json,platform,shutil,zipfile
from pathlib import Path
from corrective_common import COR,REV,now,sha,write

ROOT=COR;REPORT=COR/'validation_report.json'
DENY_NAMES={'patient_id','patient_ids','patient_index','sample_id','sample_ids','tau_true','test_predictions'}
TABLE_DIRS=[COR/'results/clinical_adjustment',COR/'results/fixed_dgp_splits',
            COR/'results/nonoracle_tuning',COR/'results/fair_neural',COR/'reporting']
CODE_DIRS=[COR/'code',REV/'source_snapshot/01_CODE/cm_osr',
           REV/'source_snapshot/01_CODE/scripts',REV/'postreview_code']
TOP=[COR/'protocol_lock.json',COR/'protocol_amendment_01.json',COR/'DATA_RIGHTS_REVIEW.md',
     COR/'RUN_RESUME_CORRECTIVE_NEURAL.cmd',COR/'RUN_RESUME_CORRECTIVE.cmd',
     COR/'CORRECTIVE_REVIEW_REPORT.md',COR/'CORRECTIVE_REVIEW_REPORT_AR.md',
     COR/'missing_requirements_report.md',COR/'README_RESUME_AR.md',COR/'requirements_corrective.txt',
     COR/'reviewer_objection_matrix.csv',COR/'per_seed_results.csv',COR/'summary_results.csv',
     COR/'experiment_manifest.csv',COR/'RUN_STATUS.json',COR/'validation_report.json',
     COR/'scientific_validation.json',COR/'scientific_validation.csv',
     COR/'environment_kaggle_reference.json',
     REV/'source_snapshot/requirements.txt']
def allowed_table(p):
    if p.suffix.lower() not in ('.csv','.json','.md','.txt'):return False
    if any(x.lower() in p.name.lower() for x in ('prediction','patient_audit','private_dgp')):return False
    if p.suffix.lower()=='.csv':
        with p.open(encoding='utf-8-sig',newline='') as f:header=next(csv.reader(f),[])
        if any(x.lower() in DENY_NAMES for x in header):raise RuntimeError('Unsafe column '+str(p))
    return True
def source_files():
    files=[p for p in TOP if p.exists()]
    for folder in CODE_DIRS:files += [p for p in folder.rglob('*.py') if '__pycache__' not in p.parts]
    for folder in TABLE_DIRS:files += [p for p in folder.rglob('*') if p.is_file() and allowed_table(p)]
    files += [p for p in (COR/'figures').rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.pdf')]
    files += [p for p in (COR/'kaggle_bundle').rglob('*.json') if p.is_file()]
    files += [p for p in (COR/'kaggle_bundle').rglob('*.py') if p.is_file()]
    files += [p for p in (COR/'kaggle_neural_downloads').rglob('*') if p.is_file() and
              (p.suffix.lower()=='.log' or allowed_table(p))]
    return sorted(set(files))
def main():
    run_status=json.loads((COR/'RUN_STATUS.json').read_text(encoding='utf-8'))
    if run_status.get('status')!='COMPLETE':
        raise RuntimeError('Publication candidate requires COMPLETE corrective RUN_STATUS.json')
    scientific=json.loads((COR/'scientific_validation.json').read_text(encoding='utf-8'))
    if scientific.get('status')!='PASS':
        raise RuntimeError('Publication candidate requires passing scientific validation')
    env={'created_utc':now(),'python':platform.python_version(),'platform':platform.platform(),
         'packages':{name:importlib.metadata.version(name) for name in
            ['numpy','pandas','scipy','scikit-learn','kaggle']}}
    write(COR/'environment_local.json',env)
    write(REPORT,dict(status='PASS_PREFLIGHT',checked_utc=now(),patient_level_files=0,
      checks=['allowlisted source code and aggregate DGP-level CSV/JSON only','no private_dgp, 02_DATA, 04_RUNS, NPZ or per-patient predictions'],
      rights='Aggregate-only release candidate; rights clearance remains a human decision; no Zenodo upload'))
    files=source_files()+[COR/'environment_local.json']
    forbidden=[str(p) for p in files if p.suffix.lower() in ('.npz','.npy','.parquet','.pkl','.pt','.pth')
               or 'private_dgp' in p.parts or '02_DATA' in p.parts or '04_RUNS' in p.parts]
    if forbidden:raise RuntimeError('Blocked patient-level files: '+str(forbidden))
    inventory=[dict(path=str(p.relative_to(REV)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    write(COR/'publication_manifest.json',dict(created_utc=now(),files=inventory,
      patient_level_files=0,zip_policy='Code and DGP-level aggregates only; no Zenodo upload'))
    archive=REV/'B01_P1_PUBLICATION_SAFE_CODE_AGGREGATES.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[COR/'publication_manifest.json']:
            z.write(p,str(p.relative_to(REV)).replace('\\','/'))
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();bad=[n for n in names if n.lower().endswith(('.npz','.npy','.parquet','.pkl','.pt','.pth'))
                                  or '/private_dgp/' in n or '/02_DATA/' in n or '/04_RUNS/' in n]
        if bad:raise RuntimeError('Archive validation failed: '+str(bad))
        status=dict(status='PASS',created_utc=now(),archive=str(archive),archive_sha256=sha(archive),
                    files=len(names),patient_level_files=0,forbidden_files=bad,
                    review='GEO public access does not clear all contributor IP; aggregate-only release candidate; no publication performed')
    write(REPORT,status)
    shutil.copy2(archive,REV/'B01_P1_CORRECTIVE_REVIEW_RESULTS.zip')
    print(archive,status['files'])
if __name__=='__main__':main()
