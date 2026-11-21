from __future__ import annotations
import hashlib,json,platform,sys,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,scipy,sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ON_KAGGLE=Path('/kaggle/input').exists()
ROOT=Path('/kaggle/working') if ON_KAGGLE else Path(__file__).resolve().parents[1]
OUT=(ROOT/'ridge_extension') if ON_KAGGLE else (ROOT/'results/ridge_extension')
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/kaggle/input/b01-causaltx-mm-core-runtime/B01_CORE_BUNDLE') if ON_KAGGLE else ROOT/'source_snapshot'
if not SOURCE.exists():
    found=list(Path('/kaggle/input').rglob('ispy2_core_multimodal.npz'))
    if len(found)!=1:raise RuntimeError(f'Expected one original B01 dataset, got {found}')
    SOURCE=found[0].parents[2]
sys.path.insert(0,str(SOURCE/'01_CODE'))
from cm_osr.dgp import generate
from cm_osr.metrics import root_pehe,policy_regret
from cm_osr.nuisance import crossfit_dr
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
from cm_osr.utils import concat_modalities

DGP=[101,211,307,401,503];NET=[17,29,43]
ALPHAS=[1e4,1e5,1e6,1e7,1e8]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def pseudo(t,y,e,m0,m1):return (m1-m0)+t*(y-m1)/e-(1-t)*(y-m0)/(1-e)
def main():
    data=SOURCE/'02_DATA/processed/ispy2_core_multimodal.npz'
    with np.load(data,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    sp=locked_split(len(raw['clinical']));tr,dv,te=sp['train'],sp['dev'],sp['test']
    mods=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}).transform(raw)
    X=concat_modalities(mods);hashdata=sha(data);finished=[];errors=[]
    for seed in DGP:
        d=generate(mods,seed,'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
        for net in NET:
            folder=OUT/f'DGP{seed}_NET{net}';folder.mkdir(exist_ok=True)
            if (folder/'COMPLETE.json').exists():finished.append((seed,net));continue
            try:
                fitted=crossfit_dr({k:v[tr] for k,v in mods.items()},d['t'][tr],d['y'][tr],
                                   n_splits=5,seed=net,clip=(.05,.95))['phi']
                oracle=pseudo(d['t'][tr],d['y'][tr],d['e_true'][tr],
                              d['mu0_true'][tr],d['mu1_true'][tr])
                rows=[];chosen=[]
                for label,phi in [('fitted',fitted),('oracle',oracle)]:
                    candidates={}
                    for alpha in ALPHAS:
                        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(X[tr],phi)
                        candidates[str(alpha)]=(model.predict(X[dv]),model.predict(X[te]))
                    candidates['constant_limit']=(np.full(len(dv),np.mean(phi)),
                                                   np.full(len(te),np.mean(phi)))
                    for name,(dev,_) in candidates.items():
                        rows.append({'dgp_seed':seed,'net_seed':net,'nuisance':label,
                          'alpha':name,'dev_root_pehe':root_pehe(dev,d['tau_true'][dv])})
                    best=min((r for r in rows if r['nuisance']==label),
                             key=lambda r:(r['dev_root_pehe'],r['alpha']))
                    test=candidates[best['alpha']][1]
                    chosen.append({'dgp_seed':seed,'net_seed':net,'nuisance':label,
                      'selected_alpha':best['alpha'],'selected_dev_root_pehe':best['dev_root_pehe'],
                      'test_root_pehe':root_pehe(test,d['tau_true'][te]),
                      'test_policy_regret':policy_regret(test,d['tau_true'][te])})
                pd.DataFrame(rows).to_csv(folder/'dev_grid.csv',index=False)
                pd.DataFrame(chosen).to_csv(folder/'selected_test.csv',index=False)
                write(folder/'COMPLETE.json',{'status':'COMPLETE','dgp_seed':seed,'net_seed':net,
                  'finished_utc':datetime.now(timezone.utc).isoformat(),'data_sha256':hashdata,
                  'alpha_grid':ALPHAS+['constant_limit'],
                  'selection':'DEV true CATE only; TEST selected-model evaluation only',
                  'software':{'python':platform.python_version(),'numpy':np.__version__,
                              'scipy':scipy.__version__,'sklearn':sklearn.__version__}})
                finished.append((seed,net))
            except Exception as exc:
                (folder/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
                errors.append({'dgp_seed':seed,'net_seed':net,'error':str(exc)})
            print('PAIR',seed,net,'COMPLETE' if (seed,net) in finished else 'ERROR',flush=True)
    if finished:
        pd.concat([pd.read_csv(OUT/f'DGP{s}_NET{n}'/'selected_test.csv') for s,n in finished],ignore_index=True).to_csv(OUT/'per_run_results.csv',index=False)
        pd.concat([pd.read_csv(OUT/f'DGP{s}_NET{n}'/'dev_grid.csv') for s,n in finished],ignore_index=True).to_csv(OUT/'dev_grid.csv',index=False)
    status={'status':'COMPLETE' if len(finished)==15 else 'PARTIAL',
            'completed_pairs':len(finished),'errors':errors,'finished_utc':datetime.now(timezone.utc).isoformat(),
            'data_sha256':hashdata}
    write(OUT/'RUN_STATUS.json',status)
    with zipfile.ZipFile(ROOT/'B01_P1_RIDGE_EXTENSION_OUTPUT.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
    print(json.dumps(status),flush=True)
if __name__=='__main__':main()
