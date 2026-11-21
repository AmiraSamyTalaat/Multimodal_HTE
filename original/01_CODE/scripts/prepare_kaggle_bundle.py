from pathlib import Path
import argparse, shutil, json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args(); root=Path(a.root)
    out=root/'09_REPRODUCIBILITY'/'KAGGLE_BUNDLE'; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    for name in ['00_CONFIG','01_CODE','03_STATE']:
        shutil.copytree(root/name,out/name,dirs_exist_ok=True)
    shutil.copy2(root/'requirements.txt',out/'requirements.txt')
    print(out)
if __name__=='__main__': main()
