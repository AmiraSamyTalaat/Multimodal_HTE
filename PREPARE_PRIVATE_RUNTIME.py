from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
source=root/'original'
target=root/'postreview'/'source_snapshot'
for p in ['00_CONFIG','01_CODE','03_STATE','06_RESULTS']:
    src=source/p
    if src.exists():
        shutil.copytree(src,target/p,dirs_exist_ok=True)
(target/'02_DATA'/'raw').mkdir(parents=True,exist_ok=True)
(target/'02_DATA'/'processed').mkdir(parents=True,exist_ok=True)
(target/'02_DATA'/'manifests').mkdir(parents=True,exist_ok=True)
print('Prepared private source tree:',target)
print('Source dataset is not in this archive. Rebuild from GEO using original/01_CODE/scripts/prepare_ispy2_core_data.py and original/00_CONFIG/data_sources.yaml.')
