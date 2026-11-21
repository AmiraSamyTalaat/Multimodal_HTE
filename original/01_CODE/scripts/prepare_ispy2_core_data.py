from __future__ import annotations
import argparse, gzip, hashlib, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, requests, yaml
CODE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(CODE))
from cm_osr.geoio import parse_series_matrix, explode_characteristics
from cm_osr.utils import write_json, sha256_file

def nid(x):
    s=str(x).strip(); s=re.sub(r'^ISPY2[_-]*','',s,flags=re.I); s=re.sub(r'\.0$','',s); return s

def dl(url,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>0: return path
    with requests.get(url,stream=True,timeout=300) as r:
        r.raise_for_status()
        with open(path,'wb') as f:
            for c in r.iter_content(1024*1024):
                if c: f.write(c)
    return path

def parse_gmt(path,min_genes,max_genes):
    sets={}
    with open(path,encoding='utf-8',errors='replace') as f:
        for line in f:
            p=line.rstrip('\n').split('\t')
            if len(p)<3: continue
            genes=[g.strip().upper().split(',')[0] for g in p[2:] if g.strip()]
            genes=list(dict.fromkeys(genes))
            if min_genes<=len(genes)<=max_genes: sets[p[0]]=set(genes)
    if not sets: raise RuntimeError('No gene sets parsed from fixed pathway library.')
    return sets

def read_expr(path, patient_ids):
    # GEO supplement is gene-level and can be either genes x samples or samples x genes.
    df=pd.read_csv(path,sep='\t',compression='gzip',index_col=0,low_memory=False)
    df.index=df.index.astype(str); df.columns=df.columns.astype(str)
    pset=set(patient_ids)
    idx_overlap=sum(nid(x) in pset for x in df.index)
    col_overlap=sum(nid(x) in pset for x in df.columns)
    if col_overlap>idx_overlap:
        X=df.T
    elif idx_overlap>col_overlap:
        X=df
    else:
        raise RuntimeError(f'Cannot infer I-SPY2 expression orientation: row overlap={idx_overlap}, column overlap={col_overlap}')
    X.index=[nid(x) for x in X.index]
    X=X.loc[[i for i in X.index if i in pset]].copy()
    X.columns=[str(c).strip().upper() for c in X.columns]
    X=X.apply(pd.to_numeric,errors='coerce')
    if X.columns.duplicated().any(): X=X.T.groupby(level=0).mean().T
    return X

def pathway_means(expr, sets, min_mapped=10, max_mapped=300):
    genes=set(expr.columns); names=[]; arrays=[]; counts={}
    for name in sorted(sets):
        gg=sorted(genes & sets[name])
        if min_mapped<=len(gg)<=max_mapped:
            names.append(name); counts[name]=len(gg)
            arrays.append(np.nanmean(expr[gg].to_numpy(dtype=float),axis=1))
    if len(names)<25: raise RuntimeError(f'Only {len(names)} fixed pathways map adequately; expected >=25. Inspect gene identifiers/library.')
    A=np.column_stack(arrays).astype('float32')
    return A,names,counts

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args(); root=Path(a.root)
    man=root/'02_DATA'/'manifests'; proc=root/'02_DATA'/'processed'; raw=root/'02_DATA'/'raw'; proc.mkdir(parents=True,exist_ok=True)
    rev=json.loads((man/'PRETRAIN_MANIFEST_REVIEW.json').read_text(encoding='utf-8'))
    if rev.get('_summary',{}).get('status')!='PASS': raise SystemExit('PRETRAIN_MANIFEST_REVIEW is not PASS; refusing core data preparation.')
    cfg=yaml.safe_load((root/'00_CONFIG'/'data_sources.yaml').read_text(encoding='utf-8'))

    # RNA metadata union
    rr=[]
    for fp in sorted((raw/'geo_matrix').glob('GSE194040-*_series_matrix.txt.gz')):
        m,_=parse_series_matrix(fp,load_values=False); c=explode_characteristics(m); c['patient_id_norm']=c['patient_id'].map(nid); rr.append(c)
    R=pd.concat(rr,ignore_index=True)
    # Verify duplicate patient metadata before collapse.
    fields=['hr','her2','mp','pcr','arm']
    for f in fields:
        if f in R:
            g=R.groupby('patient_id_norm')[f].apply(lambda s: set(str(v).strip() for v in s if str(v).strip() not in ('','nan')))
            if any(len(v)>1 for v in g): raise RuntimeError(f'RNA duplicate patient metadata conflict in {f}')
    R=R.drop_duplicates('patient_id_norm').set_index('patient_id_norm')

    # RPPA matrix and patient mapping
    rp=raw/'geo_matrix'/'GSE196093_series_matrix.txt.gz'; pm,pv=parse_series_matrix(rp,load_values=True); P=explode_characteristics(pm)
    P['patient_id_norm']=P['sample_title'].map(nid); pmap=dict(zip(P['geo_accession'].astype(str),P['patient_id_norm']))
    pv.columns=[str(c).strip('"') for c in pv.columns]
    unknown=[c for c in pv.columns if c not in pmap]
    if unknown: raise RuntimeError(f'RPPA value columns not found in metadata: {unknown[:5]}')
    RP=pv.T.copy(); RP.index=[pmap[c] for c in RP.index]; RP=RP.apply(pd.to_numeric,errors='coerce')
    if RP.index.duplicated().any(): raise RuntimeError('Duplicate RPPA patient IDs after mapping')
    if RP.shape[0]!=736: raise RuntimeError(f'RPPA rows={RP.shape[0]} != 736')

    # Fixed complete-case patient order = RPPA cohort; official resource states these patients have RNA.
    ids=[i for i in RP.index if i in R.index]
    if len(ids)!=736: raise RuntimeError(f'Complete RNA/RPPA overlap={len(ids)} != 736')

    # Download fixed processed expression matrix.
    exurl=cfg['ispy2']['expression_processed_url']; exfp=dl(exurl,raw/'GSE194040_genelevel_n988.txt.gz')
    EX=read_expr(exfp,set(ids))
    missing=[i for i in ids if i not in EX.index]
    if missing: raise RuntimeError(f'Expression supplement missing {len(missing)} RPPA patients; examples={missing[:10]}')
    EX=EX.loc[ids]

    # Download/freeze fixed public pathway library. The library is outcome-blind and fixed before model training.
    pcfg=cfg['pathways']; gmt=dl(pcfg['library_url'],raw/(pcfg['library_name']+'.gmt'))
    sets=parse_gmt(gmt,pcfg['min_mapped_genes'],pcfg['max_mapped_genes'])
    TX,pnames,pcounts=pathway_means(EX,sets,pcfg['min_mapped_genes'],pcfg['max_mapped_genes'])

    # Baseline clinical covariates only. Never include observed pCR or original treatment arm in X.
    clin=[]
    for i in ids:
        row=R.loc[i]
        clin.append([float(row['hr']),float(row['her2']),float(row['mp'])])
    CL=np.asarray(clin,dtype='float32')
    RP=RP.loc[ids]
    RPA=RP.to_numpy(dtype='float32')

    out=proc/'ispy2_core_multimodal.npz'
    np.savez_compressed(out,clinical=CL,transcriptomic=TX,rppa=RPA,
                        patient_id=np.asarray(ids,dtype='U32'),
                        clinical_feature_names=np.asarray(['HR','HER2','MP'],dtype='U16'),
                        transcriptomic_feature_names=np.asarray(pnames,dtype='U256'),
                        rppa_feature_names=np.asarray([str(x) for x in RP.columns],dtype='U128'))
    meta=R.loc[ids,[c for c in ['hr','her2','mp','pcr','arm'] if c in R.columns]].copy(); meta.insert(0,'patient_id',ids); meta.to_csv(proc/'ispy2_core_metadata_AUDIT_ONLY.csv',index=False)

    manifest={
      'status':'PASS','scientific_training_started':False,'n_patients':len(ids),
      'shapes':{'clinical':list(CL.shape),'transcriptomic_pathways':list(TX.shape),'rppa':list(RPA.shape)},
      'feature_policy':{'clinical_predictors':['HR','HER2','MP'],'forbidden_from_X':['pCR','original_trial_arm','post-treatment measurements'],
                        'transcriptomic_mapping':pcfg,'pathways_retained':len(pnames)},
      'input_sha256':{'expression':sha256_file(exfp),'rppa_series_matrix':sha256_file(rp),'pathway_library':sha256_file(gmt)},
      'output_sha256':sha256_file(out),
      'pathway_mapped_gene_counts':pcounts,
      'notes':['Pathway summaries use fixed public gene sets and within-sample gene expression only; no outcome/treatment information is used.',
               'Training-fold-only imputation/scaling remains deferred to the model pipeline.',
               'Observed I-SPY2 pCR and original treatment arm are preserved only in AUDIT_ONLY metadata and are not included in the semi-synthetic predictor NPZ.']
    }
    write_json(man/'CORE_DATA_MANIFEST.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='pathway_mapped_gene_counts'},indent=2,ensure_ascii=False))
    print('CORE DATA PREPARATION PASS')
if __name__=='__main__': main()
