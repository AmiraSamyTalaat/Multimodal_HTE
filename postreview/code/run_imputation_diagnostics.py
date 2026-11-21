from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.neighbors import NearestNeighbors

REV=Path(__file__).resolve().parents[1];SRC=REV/'source_snapshot'
sys.path.insert(0,str(SRC/'01_CODE'))
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=REV/'results'/'imputation';out.mkdir(parents=True,exist_ok=True)
    data=SRC/'02_DATA/processed/ispy2_core_multimodal.npz'
    with np.load(data,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    sp=locked_split(len(raw['clinical']));tr=sp['train']
    mods=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}).transform(raw)
    avail=pd.read_csv(REV/'results/additional/availability_masks_postreview.csv')
    neighbors=NearestNeighbors(n_neighbors=10,metric='euclidean').fit(mods['clinical'][tr])
    rows=[]
    for (seed,condition,part),g in avail.groupby(['dgp_seed','condition','split']):
        idx=g.patient_index.to_numpy(dtype=int)
        donor=tr[neighbors.kneighbors(mods['clinical'][idx],return_distance=False)]
        for modality in ['transcriptomic','rppa']:
            missing=g[f'{modality}_missing_indicator'].to_numpy(dtype=bool)
            if not missing.any():continue
            target=mods[modality][idx[missing]]
            zero=np.zeros_like(target)
            knn=np.mean(mods[modality][donor[missing]],axis=1)
            rows.append({'dgp_seed':seed,'condition':condition,'split':part,'modality':modality,
                         'n_missing_patients':int(missing.sum()),'n_features':int(target.shape[1]),
                         'zero_after_train_standardization_rmse':float(np.sqrt(np.mean((zero-target)**2))),
                         'clinical_10nn_imputation_rmse':float(np.sqrt(np.mean((knn-target)**2))),
                         'clinical_10nn_minus_zero_rmse':float(np.sqrt(np.mean((knn-target)**2))-np.sqrt(np.mean((zero-target)**2)))})
    pd.DataFrame(rows).to_csv(out/'imputation_feature_reconstruction.csv',index=False)
    status={'stage':'IMPUTATION_DIAGNOSTICS_POSTREVIEW','status':'COMPLETE','finished_utc':datetime.now(timezone.utc).isoformat(),
            'code_sha256':sha(__file__),'data_sha256':sha(data),
            'methods':['zero after TRAIN-only standardization','10-nearest clinical donors from TRAIN only'],
            'scope':'Feature reconstruction diagnostic on originally generated DEV/TEST masks; no treatment-effect model retraining yet'}
    (out/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby('modality')[['zero_after_train_standardization_rmse','clinical_10nn_imputation_rmse']].mean().to_string())
if __name__=='__main__':main()
