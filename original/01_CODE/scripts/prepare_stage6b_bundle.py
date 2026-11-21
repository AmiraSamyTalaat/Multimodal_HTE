from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root);out=root/'09_REPRODUCIBILITY'/'KAGGLE_STAGE6B_UPLOAD_DATASET';shutil.rmtree(out,ignore_errors=True);out.mkdir(parents=True);zpath=out/'B01_STAGE6B_BUNDLE.zip'
    files=['requirements.txt','02_DATA/processed/ispy2_core_multimodal.npz','02_DATA/manifests/CORE_DATA_MANIFEST.json','03_STATE/EXPERIMENT_CONTRACT.yaml','03_STATE/STAGE6B_CONFIRMATORY_LOCK.md','06_RESULTS/STAGE6A/STAGE6A_SELECTION.json','06_RESULTS/STAGE6A/STAGE6B_FREEZE.yaml']
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for d in ['00_CONFIG','01_CODE']:
            b=root/d
            if b.exists():
                for p in b.rglob('*'):
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':z.write(p,p.relative_to(root))
        for rel in files:
            p=root/rel
            if p.exists():z.write(p,p.relative_to(root))
    digest=sha256(zpath);(out/'B01_STAGE6B_BUNDLE.sha256.txt').write_text(digest+'  B01_STAGE6B_BUNDLE.zip\n');(out/'dataset-metadata.json').write_text(json.dumps({'title':'B01 CausalTx Stage6B Runtime','id':'prinprin/b01-causaltx-mm-stage6b-runtime','licenses':[{'name':'other'}]},indent=2));print(zpath);print('SHA256',digest)
if __name__=='__main__':main()
