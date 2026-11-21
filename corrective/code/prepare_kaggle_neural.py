from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from corrective_common import COR,LOCK,SRC,sha,write

TEMPLATE=COR/'code/kaggle_neural_template.py'
STAGE7=SRC/'01_CODE/cm_osr/stage7_dgp.py'
SOURCE_FILES=[STAGE7,SRC/'01_CODE/cm_osr/model.py',SRC/'01_CODE/cm_osr/experts.py',
    SRC/'01_CODE/cm_osr/nuisance.py',SRC/'01_CODE/cm_osr/router.py',
    SRC/'01_CODE/cm_osr/deep_baselines.py',SRC/'01_CODE/cm_osr/preprocess.py']
def prepare(start,count,smoke=False):
    seeds=LOCK['fair_neural']['dgp_seeds'][start:start+count]
    if len(seeds)!=count:raise ValueError('Requested batch exceeds locked seeds')
    name=f'b01-p1-corrective-neural-{start:02d}-{start+count-1:02d}'+('-smoke' if smoke else '')
    folder=COR/'kaggle_bundle'/name;folder.mkdir(parents=True,exist_ok=True)
    config=dict(seeds=seeds,regimes=LOCK['fair_neural']['effect_regimes'][:1] if smoke else LOCK['fair_neural']['effect_regimes'],
      split_seed=LOCK['fair_neural']['patient_split_seed'],net_seed=LOCK['fair_neural']['network_seed'],
      max_epochs=LOCK['fair_neural']['max_epochs'],patience=LOCK['fair_neural']['patience'],
      calibration_grid=LOCK['fair_neural']['shared_calibration_grid'],
      source_hashes={str(p.relative_to(SRC)):sha(p) for p in SOURCE_FILES},
      protocol_lock_sha256=sha(COR/'protocol_lock.json'),template_sha256=sha(TEMPLATE))
    code=TEMPLATE.read_text(encoding='utf-8').replace('__BATCH_CONFIG__',repr(config)).replace('__STAGE7_CODE__',repr(STAGE7.read_text(encoding='utf-8')))
    if '__BATCH_CONFIG__' in code or '__STAGE7_CODE__' in code:raise RuntimeError('Unexpanded template token')
    script=folder/'b01_p1_corrective_neural.py';script.write_text(code,encoding='utf-8')
    meta=dict(id=f'prinprin/{name}',title=name.replace('-',' '),code_file=script.name,
      language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
      dataset_sources=['prinprin/b01-causaltx-mm-core-runtime'],competition_sources=[],kernel_sources=[],model_sources=[])
    write(folder/'kernel-metadata.json',meta)
    write(folder/'batch_manifest.json',dict(config=config,script_sha256=sha(script),source='Existing private Kaggle B01 dataset; no new patient upload'))
    return folder
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True)
    ap.add_argument('--count',type=int,required=True);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();print(prepare(a.start,a.count,a.smoke))
