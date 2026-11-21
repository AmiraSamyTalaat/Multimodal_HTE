from __future__ import annotations
import gzip, io, re
from pathlib import Path
import numpy as np, pandas as pd, requests

def geo_bucket(accession:str)->str:
    m=re.match(r'GSE(\d+)',accession); n=m.group(1); return f"GSE{n[:-3]}nnn"

def matrix_candidates(accession, platforms=None):
    base=f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_bucket(accession)}/{accession}/matrix"
    out=[f"{base}/{accession}_series_matrix.txt.gz"]
    for gpl in platforms or []: out.append(f"{base}/{accession}-{gpl}_series_matrix.txt.gz")
    return out

def download_first(urls,out_dir,timeout=120):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True); errors=[]
    for url in urls:
        try:
            r=requests.get(url,stream=True,timeout=timeout)
            if r.status_code!=200: errors.append((url,r.status_code)); continue
            p=out_dir/url.rsplit('/',1)[-1]
            with open(p,'wb') as f:
                for c in r.iter_content(1024*1024):
                    if c: f.write(c)
            return p,url
        except Exception as e: errors.append((url,repr(e)))
    raise RuntimeError(f"No GEO matrix candidate downloaded: {errors}")

def parse_series_matrix(path,load_values=True):
    opener=gzip.open if str(path).endswith('.gz') else open
    headers=[]; data_lines=[]; in_table=False
    with opener(path,'rt',encoding='utf-8',errors='replace') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'): in_table=True; continue
            if line.startswith('!series_matrix_table_end'): in_table=False; continue
            if in_table:
                if load_values: data_lines.append(line)
            elif line.startswith('!Sample_'): headers.append(line.rstrip('\n'))
    meta={}
    for line in headers:
        key,rest=line.split('\t',1); vals=[v.strip('"') for v in rest.split('\t')]; meta.setdefault(key,[]).append(vals)
    sample_ids=(meta.get('!Sample_geo_accession') or [[]])[0]
    rows=[]
    for i,sid in enumerate(sample_ids):
        d={'geo_accession':sid}
        for key,blocks in meta.items():
            for b,vals in enumerate(blocks):
                if i<len(vals): d[f"{key[1:]}__{b}"]=vals[i]
        rows.append(d)
    mdf=pd.DataFrame(rows)
    values=None
    if load_values and data_lines:
        values=pd.read_csv(io.StringIO(''.join(data_lines)),sep='\t',quotechar='"',index_col=0)
    return mdf,values

def explode_characteristics(meta:pd.DataFrame):
    """Flatten GEO sample characteristics across *all* channels.

    Two-colour GEO studies may store clinically important fields (e.g. treatment
    arm / response) in channel 2 rather than channel 1.  Restricting parsing to
    ch1 can therefore create a false linkage failure.  We preserve the first
    non-empty value under the canonical key and, only when a conflicting value
    exists in another channel, retain that value in a channel-qualified column.
    """
    out=meta[['geo_accession']].copy()
    cols=[c for c in meta if c.startswith('Sample_characteristics_ch')]

    def norm_key(k:str)->str:
        k=k.strip().lower().replace(' ','_')
        k=re.sub(r'[^a-z0-9_()=+.-]+','_',k)
        return k.strip('_')

    for c in cols:
        m=re.search(r'(ch\d+)',c)
        channel=m.group(1) if m else 'ch'
        vals=meta[c].fillna('').astype(str)
        for i,text in vals.items():
            if ':' not in text:
                continue
            k,v=text.split(':',1)
            k=norm_key(k); v=v.strip()
            if not k:
                continue
            if k not in out.columns:
                out[k]=''
            existing=str(out.at[i,k]) if i in out.index else ''
            if existing in ('','nan','None'):
                out.at[i,k]=v
            elif v and existing != v:
                alt=f'{k}__{channel}'
                if alt not in out.columns:
                    out[alt]=''
                out.at[i,alt]=v

    # Keep source/title/type fields because they are useful for pretreatment audits.
    for prefix,label in [
        ('Sample_title','sample_title'),
        ('Sample_source_name_ch','source_name'),
        ('Sample_type','sample_type'),
    ]:
        matches=[c for c in meta if c.startswith(prefix)]
        for j,c in enumerate(matches):
            name=label if j==0 else f'{label}_{j+1}'
            out[name]=meta[c]
    return out
