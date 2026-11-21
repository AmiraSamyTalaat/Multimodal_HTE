from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import requests, yaml

def download(url, out):
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    with requests.get(url,stream=True,timeout=120) as r:
        r.raise_for_status()
        with open(out,'wb') as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk); h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args(); root=Path(a.root)
    cfg=yaml.safe_load((root/'00_CONFIG'/'data_sources.yaml').read_text(encoding='utf-8'))
    manifest={}
    url=cfg['ispy2']['expression_processed_url']; name=url.rsplit('/',1)[1]
    path=root/'02_DATA'/'raw'/name; manifest[name]={'url':url,'sha256':download(url,path)}
    (root/'02_DATA'/'manifests'/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
