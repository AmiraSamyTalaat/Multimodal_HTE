from __future__ import annotations
import hashlib,json,platform,sys,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,scipy,sklearn
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors

ROOT=Path('/kaggle/working');OUT=ROOT/'missingness_postreview';OUT.mkdir(exist_ok=True)
SOURCE=Path('/kaggle/input/b01-causaltx-mm-core-runtime/B01_CORE_BUNDLE')
if not SOURCE.exists():
    matches=list(Path('/kaggle/input').rglob('ispy2_core_multimodal.npz'))
    if not matches:raise RuntimeError('Original private B01 core dataset is not attached')
    SOURCE=matches[0].parents[2]
sys.path.insert(0,str(SOURCE/'01_CODE'))
from cm_osr.baselines import XLearner
from cm_osr.dgp import generate,_proj
from cm_osr.metrics import root_pehe,policy_regret
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
from cm_osr.utils import concat_modalities

SEEDS=[101,211,307,401,503]
CONDITIONS=['MCAR_40','MAR_CLINICAL']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def mask_test(mods,idx,c,seed,condition):
    code={'MCAR_40':40,'MAR_CLINICAL':77}[condition]
    rng=np.random.default_rng(710000+seed*100+code*10+2)
    prob=np.full(len(idx),.4) if condition=='MCAR_40' else np.clip(.1+.25*expit(c[idx]),0,.95)
    mt=rng.random(len(idx))<prob;mr=rng.random(len(idx))<prob
    return mt,mr
def with_mask(mods,idx,mt,mr,donors=None):
    sub={k:np.array(v[idx],copy=True) for k,v in mods.items()}
    for name,mask in [('transcriptomic',mt),('rppa',mr)]:
        if donors is None:sub[name][mask]=0
        else:sub[name][mask]=np.mean(mods[name][donors[mask]],axis=1)
    flags=np.column_stack([(~mt).astype('float32'),(~mr).astype('float32')])
    return concat_modalities(sub),flags
def main():
    data=SOURCE/'02_DATA/processed/ispy2_core_multimodal.npz'
    with np.load(data,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    sp=locked_split(len(raw['clinical']));tr,te=sp['train'],sp['test']
    mods=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}).transform(raw)
    X=concat_modalities(mods)
    knn=NearestNeighbors(n_neighbors=10).fit(mods['clinical'][tr])
    donors=tr[knn.kneighbors(mods['clinical'][te],return_distance=False)]
    all_rows=[];failures=[]
    for seed in SEEDS:
        folder=OUT/f'DGP{seed}';folder.mkdir(exist_ok=True)
        if (folder/'COMPLETE.json').exists():
            all_rows.extend(pd.read_csv(folder/'metrics.csv').to_dict('records'));continue
        try:
            d=generate(mods,seed,'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
            t=d['t'][tr];y=d['y'][tr];truth=d['tau_true'][te]
            c=_proj(mods['clinical'],1101,tr)
            clean=XLearner().fit(X[tr],t,y)
            # Clean and independently masked TRAIN replicas; availability flags are model inputs.
            rng=np.random.default_rng(880000+seed)
            mt_train=rng.random(len(tr))<.4;mr_train=rng.random(len(tr))<.4
            masked_train,flags_train=with_mask(mods,tr,mt_train,mr_train)
            clean_flags=np.ones((len(tr),2),dtype='float32')
            X_aug=np.vstack([np.column_stack([X[tr],clean_flags]),np.column_stack([masked_train,flags_train])])
            t_aug=np.concatenate([t,t]);y_aug=np.concatenate([y,y])
            maskaware=XLearner().fit(X_aug,t_aug,y_aug)
            rows=[{'dgp_seed':seed,'condition':'CLEAN','method':'clean_full_X',
                   'root_pehe':root_pehe(clean.predict(X[te]),truth),
                   'policy_regret':policy_regret(clean.predict(X[te]),truth),'n_test':len(te)}]
            for condition in CONDITIONS:
                mt,mr=mask_test(mods,te,c,seed,condition)
                zero,flags=with_mask(mods,te,mt,mr)
                clinical_knn,_=with_mask(mods,te,mt,mr,donors)
                predictions={'zero_imputation_clean_X':clean.predict(zero),
                             'clinical_10nn_imputation_clean_X':clean.predict(clinical_knn),
                             'mask_indicator_augmented_X':maskaware.predict(np.column_stack([zero,flags]))}
                for method,pred in predictions.items():
                    rows.append({'dgp_seed':seed,'condition':condition,'method':method,
                                 'root_pehe':root_pehe(pred,truth),'policy_regret':policy_regret(pred,truth),
                                 'n_test':len(te),'transcriptomic_missing_fraction':float(mt.mean()),
                                 'rppa_missing_fraction':float(mr.mean())})
            pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
            write(folder/'COMPLETE.json',{'status':'COMPLETE','dgp_seed':seed,'finished_utc':datetime.now(timezone.utc).isoformat(),
                'data_sha256':sha(data),'n_train':len(tr),'n_test':len(te),
                'training':'clean X on TRAIN only; mask-aware X on clean + independently masked TRAIN replica',
                'test_masks':'exact original Stage5A DEV/TEST mask generator for MCAR40 and MAR',
                'clinical_10nn':'donors from TRAIN only; no TEST label or CATE used',
                'software':{'python':platform.python_version(),'numpy':np.__version__,
                            'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__}})
            all_rows.extend(rows)
        except Exception as exc:
            (folder/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
            failures.append({'seed':seed,'error':str(exc)})
        print('DGP',seed,'complete' if (folder/'COMPLETE.json').exists() else 'failed',flush=True)
    if all_rows:pd.DataFrame(all_rows).to_csv(OUT/'per_seed_results.csv',index=False)
    status={'status':'COMPLETE' if len(all_rows)==len(SEEDS)*(1+len(CONDITIONS)*3) and not failures else 'PARTIAL',
            'completed_seeds':len(all_rows)//7,'failed':failures,'finished_utc':datetime.now(timezone.utc).isoformat(),
            'data_sha256':sha(data)}
    write(OUT/'RUN_STATUS.json',status)
    with zipfile.ZipFile(ROOT/'B01_P1_MISSINGNESS_OUTPUT.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
    print(json.dumps(status),flush=True)
if __name__=='__main__':main()
