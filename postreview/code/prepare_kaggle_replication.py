from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

REV=Path(__file__).resolve().parents[1]
SRC=REV/'postreview_code/run_replication.py'
OLD="REV=Path(__file__).resolve().parents[1];SRC=REV/'source_snapshot'"
NEW="""REV=Path('/kaggle/working')
SRC=Path('/kaggle/input/b01-causaltx-mm-core-runtime/B01_CORE_BUNDLE')
if not SRC.exists():
    matches=list(Path('/kaggle/input').rglob('ispy2_core_multimodal.npz'))
    if len(matches)!=1:raise RuntimeError(f'Expected one original B01 private dataset; got {matches}')
    SRC=matches[0].parents[2]
"""
def prepare(start,count):
    name=f'b01-p1-postreview-replication-{start:02d}-{start+count-1:02d}'
    folder=REV/'kaggle_bundle'/name;folder.mkdir(parents=True,exist_ok=True)
    code=SRC.read_text(encoding='utf-8')
    if OLD not in code:raise RuntimeError('Original replication source signature changed')
    code=code.replace(OLD,NEW)
    stage7=(REV/'source_snapshot/01_CODE/cm_osr/stage7_dgp.py').read_text(encoding='utf-8')
    import_line="sys.path.insert(0,str(SRC/'01_CODE'))"
    injector=(import_line+"\nimport types\n_stage7=types.ModuleType('cm_osr.stage7_dgp')\n"
              "_stage7.__package__='cm_osr'\n"
              "sys.modules['cm_osr.stage7_dgp']=_stage7\n"
              f"exec({stage7!r},_stage7.__dict__)\n")
    if import_line not in code:raise RuntimeError('Original source path signature changed')
    code=code.replace(import_line,injector)
    line='DGP_SEEDS=[3001+67*i for i in range(30)]'
    if line not in code:raise RuntimeError('Original DGP seed signature changed')
    code=code.replace(line,f'DGP_SEEDS={[3001+67*i for i in range(start,start+count)]!r}')
    original_hash=hashlib.sha256(SRC.read_bytes()).hexdigest()
    code=code.replace('code_hash=sha(__file__)',f"code_hash={original_hash!r}")
    code=code.replace("'code_sha256':code_hash,",
                      "'code_sha256':code_hash,'kaggle_execution_code_sha256':sha(__file__),")
    code+='''\nimport zipfile\nwith zipfile.ZipFile(REV/'B01_P1_REPLICATION_BATCH_OUTPUT.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:\n    for path in (REV/'results/replication').rglob('*'):\n        if path.is_file():archive.write(path,path.relative_to(REV).as_posix())\n'''
    (folder/'b01_p1_replication_batch.py').write_text(code,encoding='utf-8')
    meta={'id':f'prinprin/{name}','title':name.replace('-',' '),
          'code_file':'b01_p1_replication_batch.py','language':'python','kernel_type':'script',
          'is_private':True,'enable_gpu':True,'enable_internet':False,
          'dataset_sources':['prinprin/b01-causaltx-mm-core-runtime'],
          'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return folder
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True)
    ap.add_argument('--count',type=int,required=True);a=ap.parse_args()
    print(prepare(a.start,a.count))
