from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

COR=Path(__file__).resolve().parents[1]
REV=COR.parent
SRC=REV/'source_snapshot'
sys.path.insert(0,str(SRC/'01_CODE'))
from cm_osr.metrics import root_pehe,policy_regret
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
from cm_osr.stage7_dgp import generate_stage7
from cm_osr.utils import concat_modalities

LOCK=json.loads((COR/'protocol_lock.json').read_text(encoding='utf-8'))
DATA=SRC/'02_DATA/processed/ispy2_core_multimodal.npz'
PRIVATE=COR/'private_dgp';PRIVATE.mkdir(parents=True,exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def load_raw():
    with np.load(DATA,allow_pickle=False) as z:
        return {k:z[k] for k in ['clinical','transcriptomic','rppa']}
def fixed_split(n,seed):
    anchor=locked_split(n,seed=LOCK['fixed_dgp']['anchor_test_split_seed'])
    devpool=np.sort(np.concatenate([anchor['train'],anchor['dev']]))
    train,dev=train_test_split(devpool,train_size=len(anchor['train']),
                               random_state=int(seed),shuffle=True)
    return {'train':np.sort(train),'dev':np.sort(dev),'test':anchor['test'],
            'dgp_reference':devpool}
def dgp_path(seed,regime):return PRIVATE/f'DGP{seed}_{regime}.npz'
def fixed_dgp(raw,seed,regime):
    path=dgp_path(seed,regime);manifest=path.with_suffix('.json')
    lock_hash=sha(COR/'protocol_lock.json');data_hash=sha(DATA)
    dgp_code_hash=sha(SRC/'01_CODE/cm_osr/stage7_dgp.py')
    if path.exists() and manifest.exists():
        info=json.loads(manifest.read_text(encoding='utf-8'))
        if (info.get('lock_sha256'),info.get('data_sha256'),info.get('dgp_code_sha256')) != (
                lock_hash,data_hash,dgp_code_hash):
            raise RuntimeError('Fixed-DGP cache provenance mismatch: '+str(path))
        if info.get('npz_sha256')!=sha(path):raise RuntimeError('Fixed-DGP cache SHA mismatch')
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files},info
    ref=fixed_split(len(raw['clinical']),LOCK['fixed_dgp']['model_train_dev_resplit_seeds'][0])['dgp_reference']
    dgp_prep=ModalityPreprocessor().fit({k:v[ref] for k,v in raw.items()})
    dgp_mods=dgp_prep.transform(raw)
    d=generate_stage7(dgp_mods,int(seed),regime,reference_idx=ref)
    keys=['t','y','e_true','mu0_true','mu1_true','tau_true','incremental_source',
          'incremental_signal','g_incremental','p_incremental','joint_incremental']
    np.savez_compressed(path,**{k:d[k] for k in keys})
    info={'dgp_seed':seed,'effect_regime':regime,'created_utc':now(),
          'data_sha256':data_hash,'dgp_code_sha256':dgp_code_hash,
          'lock_sha256':lock_hash,'npz_sha256':sha(path),
          'reference_n':len(ref),'reference_indices_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),
          'n_patients':len(raw['clinical']),
          't_sha256':hashlib.sha256(d['t'].tobytes()).hexdigest(),
          'y_sha256':hashlib.sha256(d['y'].tobytes()).hexdigest(),
          'cate_sha256':hashlib.sha256(d['tau_true'].tobytes()).hexdigest(),
          'condition':'DGP fitted once using development covariates only; no model TEST labels or CATE used'}
    write(manifest,info)
    return {k:d[k] for k in keys},info
def preprocess_model(raw,tr):
    prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()})
    mods=prep.transform(raw)
    return mods,concat_modalities(mods),mods['clinical']
def rloss(pred,t,y,e,m0,m1):
    e=np.asarray(e);m=e*np.asarray(m1)+(1-e)*np.asarray(m0)
    return float(np.mean(((np.asarray(y)-m)-(np.asarray(t)-e)*np.asarray(pred))**2))
def score(method,pred,true,seed,split_seed,regime,source):
    pred=np.asarray(pred);true=np.asarray(true)
    return {'dgp_seed':seed,'split_seed':split_seed,'effect_regime':regime,
      'method':method,'root_pehe':root_pehe(pred,true),
      'policy_regret':policy_regret(pred,true),
      'clipped_root_pehe':root_pehe(np.clip(pred,-1,1),true),
      'fraction_outside_minus1_plus1':float(np.mean(np.abs(pred)>1)),
      'n_test':len(true),'source':source}
