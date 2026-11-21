
from pathlib import Path
import argparse, hashlib, json, shutil, zipfile

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args()
    root=Path(a.root)
    out=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE5A_UPLOAD_DATASET'
    shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    zpath=out/'B01_STAGE5A_BUNDLE.zip'
    include_dirs=['00_CONFIG','01_CODE']
    include_files=[
        'requirements.txt',
        '02_DATA/processed/ispy2_core_multimodal.npz',
        '02_DATA/manifests/CORE_DATA_MANIFEST.json',
        '02_DATA/manifests/PRETRAIN_MANIFEST_REVIEW.json',
        '03_STATE/EXPERIMENT_CONTRACT.yaml',
        '03_STATE/ABLATION_PLAN.md',
        '03_STATE/STAGE4_DECISION.json',
        '03_STATE/STAGE5A_STRESS_LOCK.md',
        '03_STATE/STAGE5A_EXECUTION_PLAN.md',
        '03_STATE/STAGE5A_PREFLIGHT.json',
        '06_RESULTS/PRIMARY_GATE.json',
        '06_RESULTS/ALL_CORE_METRICS.csv'
    ]
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for d in include_dirs:
            base=root/d
            for p in base.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
                    z.write(p,p.relative_to(root))
        for rel in include_files:
            p=root/rel
            if p.exists(): z.write(p,p.relative_to(root))
    digest=sha256(zpath)
    (out/'B01_STAGE5A_BUNDLE.sha256.txt').write_text(digest+'  B01_STAGE5A_BUNDLE.zip\n',encoding='utf-8')
    meta={'title':'B01 CausalTx Stage5A Runtime','id':'prinprin/b01-causaltx-mm-stage5a-runtime','licenses':[{'name':'other'}]}
    (out/'dataset-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(zpath); print('SHA256',digest)

if __name__=='__main__': main()
