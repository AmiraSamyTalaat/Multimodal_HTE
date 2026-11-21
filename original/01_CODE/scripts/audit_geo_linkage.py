from __future__ import annotations
import argparse, json, sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from cm_osr.geoio import matrix_candidates, download_first, parse_series_matrix, explode_characteristics
from cm_osr.utils import sha256_file, write_json

# IMPORTANT: TRIO-US B07 pretreatment features come from GSE130786 (Baseline).
# GSE130787 contains the on-treatment/run-in comparison and is not a valid
# pretreatment predictor source for the locked external test.
SETS={
 'GSE194040':['GPL20078','GPL30493'],
 'GSE196093':['GPL28470'],
 'GSE66305':['GPL570'],
 'GSE130786':['GPL6480'],
}
EXPECTED={
 'GSE194040': {
     'count_mode':'union', 'min_samples':988, 'min_unique_patients':987,
     'must_have_any':['arm','pcr','her2','hr']
 },
 'GSE196093': {'count_mode':'max', 'min_samples':736, 'must_have_any':['arm','pcr','her2','hr']},
 'GSE66305': {'count_mode':'max', 'min_samples':88, 'must_have_any':['cher-lob_arm','arm_description','pcr_(1=yes)']},
 'GSE130786': {'count_mode':'max', 'min_samples':110, 'must_have_any':['treatment','response','pcr','drug']},
}

def candidate_urls(acc,gpls):
    urls=[]
    for u in matrix_candidates(acc,gpls):
        if u not in urls: urls.append(u)
    return urls

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args()
    root=Path(a.root)
    raw=root/'02_DATA'/'raw'/'geo_matrix'; raw.mkdir(parents=True,exist_ok=True)
    (root/'02_DATA'/'manifests').mkdir(parents=True,exist_ok=True)
    report={}; failures=[]

    for acc,gpls in SETS.items():
        items=[]; attempts=[]; all_sample_ids=set(); all_patient_ids=set()
        for url in candidate_urls(acc,gpls):
            try:
                fp,_=download_first([url],raw)
                meta,_=parse_series_matrix(fp,load_values=False)
                chars=explode_characteristics(meta)
                cols=[str(c) for c in chars.columns]
                low=[c.lower() for c in cols]
                exp=EXPECTED[acc]
                present=[k for k in exp['must_have_any'] if any(k.lower() in c for c in low)]
                ids=set(chars['geo_accession'].dropna().astype(str)) if 'geo_accession' in chars else set()
                all_sample_ids |= ids
                if 'patient_id' in chars:
                    all_patient_ids |= set(v for v in chars['patient_id'].dropna().astype(str) if v and v.lower()!='nan')
                item={
                    'url':url,'file':fp.name,'sha256':sha256_file(fp),
                    'n_samples':int(len(meta)),'columns':cols,
                    'required_field_hits':present,
                    'examples':chars.head(3).to_dict('records')
                }
                items.append(item)
                attempts.append({'url':url,'status':'OK','n_samples':int(len(meta))})
            except Exception as e:
                attempts.append({'url':url,'status':'ERROR','error':repr(e)})

        if not items:
            failures.append(f'{acc}: no GEO series matrix retrieved')
        else:
            exp=EXPECTED[acc]
            if exp.get('count_mode')=='union':
                observed=len(all_sample_ids)
                if observed < exp['min_samples']:
                    failures.append(f'{acc}: combined unique sample count={observed} < {exp["min_samples"]}')
                min_pat=exp.get('min_unique_patients')
                if min_pat is not None and len(all_patient_ids) < min_pat:
                    failures.append(f'{acc}: combined unique patient_id count={len(all_patient_ids)} < {min_pat}')
            else:
                observed=max(i['n_samples'] for i in items)
                if observed < exp['min_samples']:
                    failures.append(f'{acc}: unexpected sample count max={observed} < {exp["min_samples"]}')

            if not any(i['required_field_hits'] for i in items):
                failures.append(f'{acc}: expected treatment/response characteristic fields not detected across GEO channels')

        report[acc]={
            'items':items,'attempts':attempts,
            'combined_unique_sample_ids':len(all_sample_ids),
            'combined_unique_patient_ids':len(all_patient_ids) if all_patient_ids else None,
            'expected':EXPECTED[acc],
        }

    report['_summary']={
        'status':'PASS' if not failures else 'FAIL',
        'failures':failures,
        'scientific_training_started':False,
        'notes':[
            'GSE194040 is intentionally split across two platform matrices (654 + 334 = 988 sample records); the audit evaluates their union.',
            'TRIO-US B07 external pretreatment source is GSE130786 (110 baseline samples). GSE130787 is on-treatment/run-in and is excluded from pretreatment predictors.',
            'Treatment/response fields are scanned across all GEO sample-characteristic channels, not ch1 only.'
        ]
    }
    out=root/'02_DATA'/'manifests'/'GEO_LINKAGE_AUDIT.json'
    write_json(out,report)
    print(json.dumps(report,indent=2,ensure_ascii=False)[:40000])
    if failures:
        raise SystemExit('DATA/LINKAGE AUDIT FAIL: ' + ' | '.join(failures))
    print('DATA/LINKAGE AUDIT PASS')

if __name__=='__main__': main()
