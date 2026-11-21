from pathlib import Path
import argparse, shutil, json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args(); root=Path(a.root)
    data=root/'02_DATA'/'processed'/'ispy2_core_multimodal.npz'
    manifest=root/'02_DATA'/'manifests'/'CORE_DATA_MANIFEST.json'
    if not data.exists() or not manifest.exists(): raise SystemExit('Core processed data/manifest missing; run RUN_PREPARE_CORE_DATA_B01.cmd first.')
    m=json.loads(manifest.read_text(encoding='utf-8'))
    if m.get('status')!='PASS': raise SystemExit('CORE_DATA_MANIFEST is not PASS.')
    out=root/'09_REPRODUCIBILITY'/'KAGGLE_CORE_BUNDLE'; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    for name in ['00_CONFIG','01_CODE','03_STATE']:
        shutil.copytree(root/name,out/name,dirs_exist_ok=True)
    shutil.copy2(root/'requirements.txt',out/'requirements.txt')
    (out/'02_DATA'/'processed').mkdir(parents=True,exist_ok=True); (out/'02_DATA'/'manifests').mkdir(parents=True,exist_ok=True)
    shutil.copy2(data,out/'02_DATA'/'processed'/data.name); shutil.copy2(manifest,out/'02_DATA'/'manifests'/manifest.name)
    rev=root/'02_DATA'/'manifests'/'PRETRAIN_MANIFEST_REVIEW.json'
    if rev.exists(): shutil.copy2(rev,out/'02_DATA'/'manifests'/rev.name)
    print(out)
if __name__=='__main__': main()
