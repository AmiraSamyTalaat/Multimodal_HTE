from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import pandas as pd
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from cm_osr.geoio import parse_series_matrix, explode_characteristics
from cm_osr.utils import write_json, sha256_file

def nid(x):
    s=str(x).strip()
    s=re.sub(r'^ISPY2[_-]*','',s,flags=re.I)
    s=re.sub(r'\.0$','',s)
    return s

def norm_arm(x):
    return re.sub(r'[^a-z0-9]+','',str(x).lower())

def load_chars(path):
    meta,_=parse_series_matrix(path,load_values=False)
    return explode_characteristics(meta)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args()
    root=Path(a.root); raw=root/'02_DATA'/'raw'/'geo_matrix'; man=root/'02_DATA'/'manifests'; man.mkdir(parents=True,exist_ok=True)
    fail=[]; rep={'scientific_training_started':False}

    # I-SPY2 RNA metadata union
    rna_files=sorted(raw.glob('GSE194040-*_series_matrix.txt.gz'))
    if len(rna_files)<2: fail.append('I-SPY2 RNA: expected both platform matrices in geo_matrix')
    rna=[]
    for fp in rna_files:
        c=load_chars(fp); c['patient_id_norm']=c.get('patient_id','').map(nid); c['source_file']=fp.name; rna.append(c)
    if rna:
        R=pd.concat(rna,ignore_index=True)
        rna_unique=R.drop_duplicates('geo_accession')
        pat=R[R['patient_id_norm'].astype(str)!=''].copy()
        rep['ispy2_rna']={'sample_records':int(rna_unique['geo_accession'].nunique()),'unique_patients':int(pat['patient_id_norm'].nunique()),'files':[f.name for f in rna_files]}
        if rep['ispy2_rna']['sample_records']<988: fail.append(f'I-SPY2 RNA sample union {rep["ispy2_rna"]["sample_records"]} < 988')
        if rep['ispy2_rna']['unique_patients']<987: fail.append(f'I-SPY2 RNA patient union {rep["ispy2_rna"]["unique_patients"]} < 987')
    else: R=pd.DataFrame()

    # I-SPY2 RPPA and explicit cross-modal linkage
    rp=raw/'GSE196093_series_matrix.txt.gz'
    if not rp.exists(): fail.append('I-SPY2 RPPA series matrix missing')
    else:
        P=load_chars(rp)
        if 'sample_title' not in P: fail.append('I-SPY2 RPPA sample_title missing; cannot map patient IDs')
        else:
            P['patient_id_norm']=P['sample_title'].map(nid)
            rep['ispy2_rppa']={'samples':int(P['geo_accession'].nunique()),'unique_patient_ids':int(P['patient_id_norm'].nunique()),'sha256':sha256_file(rp)}
            if P['patient_id_norm'].nunique()!=736: fail.append(f'I-SPY2 RPPA unique patient IDs={P["patient_id_norm"].nunique()} != 736')
            if not R.empty:
                rset=set(R['patient_id_norm'].dropna().astype(str)); pset=set(P['patient_id_norm'].dropna().astype(str)); overlap=rset & pset
                rep['ispy2_crossmodal_linkage']={'rppa_patients':len(pset),'rppa_in_rna':len(overlap),'missing_rppa_in_rna':sorted(pset-rset)[:20]}
                if len(overlap)!=736: fail.append(f'I-SPY2 RNA/RPPA patient overlap={len(overlap)} != 736')
                # Clinical-label concordance on matched patients. Collapse RNA duplicate patient records only after checking consistency.
                fields=['hr','her2','pcr','arm']
                rcols=[c for c in ['patient_id_norm']+fields if c in R.columns]
                pcols=[c for c in ['patient_id_norm']+fields if c in P.columns]
                RR=R[rcols].copy(); PP=P[pcols].copy()
                # per-patient first non-empty; duplicates should not disagree
                dup_conf={}
                for f in fields:
                    if f in RR:
                        g=RR.groupby('patient_id_norm')[f].apply(lambda s: sorted(set(str(v).strip() for v in s if str(v).strip() not in ('','nan'))))
                        dup_conf[f]=int(sum(len(v)>1 for v in g))
                rep['ispy2_crossmodal_linkage']['rna_duplicate_patient_conflicts']=dup_conf
                if any(v>0 for v in dup_conf.values()): fail.append('I-SPY2 duplicate RNA patient records disagree on clinical labels')
                RR=RR.drop_duplicates('patient_id_norm'); PP=PP.drop_duplicates('patient_id_norm'); M=PP.merge(RR,on='patient_id_norm',suffixes=('_rppa','_rna'))
                mism={}
                for f in fields:
                    a1=f+'_rppa'; a2=f+'_rna'
                    if a1 in M and a2 in M:
                        if f=='arm': bad=(M[a1].map(norm_arm)!=M[a2].map(norm_arm))
                        else: bad=(M[a1].astype(str).str.strip()!=M[a2].astype(str).str.strip())
                        mism[f]=int(bad.sum())
                rep['ispy2_crossmodal_linkage']['clinical_label_mismatches']=mism
                for f in ['hr','her2','pcr']:
                    if mism.get(f,0)>0: fail.append(f'I-SPY2 RNA/RPPA {f} mismatch count={mism[f]}')

    # CHER-LOB expression cohort
    ch=raw/'GSE66305_series_matrix.txt.gz'
    if not ch.exists(): fail.append('CHER-LOB GSE66305 matrix missing')
    else:
        C=load_chars(ch); arm='cher-lob_arm'; y='pcr_(1=yes)'
        src='source_name' if 'source_name' in C else None
        pret=int(C[src].astype(str).str.lower().str.contains('pre-treated|pretreat|pre-treatment',regex=True).sum()) if src else 0
        rep['cherlob']={'n':len(C),'pretreatment_marked':pret,'arms':sorted(C[arm].dropna().astype(str).unique().tolist()) if arm in C else [],'pcr_values':sorted(C[y].dropna().astype(str).unique().tolist()) if y in C else []}
        if len(C)!=88: fail.append(f'CHER-LOB expression n={len(C)} != 88')
        if pret!=88: fail.append(f'CHER-LOB pretreatment-marked samples={pret} != 88')
        if arm not in C or not {'A','B','C'}.issubset(set(C[arm].astype(str))): fail.append('CHER-LOB arms A/B/C not all present')
        if y not in C or not set(C[y].astype(str)).issubset({'0','1'}): fail.append('CHER-LOB pCR is not clean binary 0/1')

    # TRIO-US B07 baseline cohort only
    tr=raw/'GSE130786_series_matrix.txt.gz'
    if not tr.exists(): fail.append('TRIO-US B07 baseline GSE130786 matrix missing')
    else:
        T=load_chars(tr); tg='treatment_group'; dr='drug_response'
        basemask=pd.Series(False,index=T.index)
        for c in [x for x in T.columns if 'sample_type' in x or 'source_name' in x]: basemask |= T[c].astype(str).str.lower().str.contains('baseline')
        rep['trio_b07']={'n':len(T),'baseline_marked':int(basemask.sum()),'treatment_groups':sorted(T[tg].dropna().astype(str).unique().tolist()) if tg in T else [],'responses':sorted(T[dr].dropna().astype(str).unique().tolist()) if dr in T else []}
        if len(T)!=110: fail.append(f'TRIO-US B07 baseline n={len(T)} != 110')
        if int(basemask.sum())!=110: fail.append(f'TRIO-US B07 baseline-marked samples={int(basemask.sum())} != 110')
        if tg not in T or len(set(T[tg].astype(str)))<3: fail.append('TRIO-US B07 expected three randomized treatment groups not found')
        if dr not in T or not set(T[dr].astype(str)).issubset({'PCR','RD'}): fail.append('TRIO-US B07 response values are not restricted to PCR/RD')
        txt=' '.join(T.astype(str).head(110).fillna('').values.ravel()).lower()
        if 'run-in' in txt or 'run in' in txt: fail.append('TRIO-US B07 baseline matrix unexpectedly contains run-in marker')

    rep['_summary']={'status':'PASS' if not fail else 'FAIL','failures':fail,'scientific_training_started':False,
                     'note':'This gate verifies exact cross-modal patient linkage and pretreatment/randomization metadata before any scientific model training.'}
    write_json(man/'PRETRAIN_MANIFEST_REVIEW.json',rep)
    print(json.dumps(rep,indent=2,ensure_ascii=False)[:50000])
    if fail: raise SystemExit('PRETRAIN MANIFEST REVIEW FAIL: '+' | '.join(fail))
    print('PRETRAIN MANIFEST REVIEW PASS')
if __name__=='__main__': main()
