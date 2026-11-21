from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit

REV=Path(__file__).resolve().parents[1];SRC=REV/'source_snapshot'
sys.path.insert(0,str(SRC/'01_CODE'))
from cm_osr.dgp import generate,_proj
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.splits import locked_split

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ess(w):return float(np.sum(w)**2/np.sum(w*w))
def main():
    file=SRC/'02_DATA/processed/ispy2_core_multimodal.npz'
    with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    sp=locked_split(len(raw['clinical']));tr=sp['train']
    mods=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}).transform(raw)
    c=_proj(mods['clinical'],1101,tr);g=_proj(mods['transcriptomic'],2203,tr)
    p=_proj(mods['rppa'],3307,tr)
    linear=.25*c+.45*g+.45*p+.65*g*p
    rows=[]
    for seed in [101,211,307,401,503]:
        original=generate(mods,seed,'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
        for label,scale in [('PRIMARY',1.35),('SEVERE_PROPENSITY_ONLY',2.25)]:
            prob=np.clip(expit(scale*linear),.02,.98)
            if label=='PRIMARY' and not np.allclose(prob,original['e_true'],atol=1e-7):
                raise RuntimeError('Primary propensity reconstruction mismatch')
            # Common random treatment uniforms isolate propensity perturbation.
            uniforms=np.random.default_rng(seed+909000).random(len(prob))
            t=(uniforms<prob).astype(int)
            for split,idx in sp.items():
                e=prob[idx];z=t[idx]
                tw=1/e[z==1];cw=1/(1-e[z==0])
                rows.append({'dgp_seed':seed,'condition':label,'split':split,
                    'n':len(idx),'treatment_prevalence':float(np.mean(z)),
                    'true_cate_mean':float(np.mean(original['tau_true'][idx])),
                    'true_cate_sd':float(np.std(original['tau_true'][idx])),
                    'true_cate_sha256':hashlib.sha256(original['tau_true'][idx].tobytes()).hexdigest(),
                    'e_mean':float(np.mean(e)),'e_q01':float(np.quantile(e,.01)),
                    'e_q05':float(np.quantile(e,.05)),'e_q95':float(np.quantile(e,.95)),
                    'e_q99':float(np.quantile(e,.99)),
                    'fraction_true_e_at_002_or_098':float(np.mean((e<=.0200001)|(e>=.9799999))),
                    'ess_treated':ess(tw),'ess_control':ess(cw)})
    out=REV/'results/additional';pd.DataFrame(rows).to_csv(out/'clean_overlap_only_perturbation.csv',index=False)
    (out/'CLEAN_OVERLAP_STATUS.json').write_text(json.dumps({'status':'COMPLETE',
        'finished_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),
        'data_sha256':sha(file),'scope':'Pure propensity-scale contrast with shared CATE/mu; diagnostics only',
        'n_dgp':5,'common_random_numbers':'treatment draws use shared uniforms per DGP'},indent=2),encoding='utf-8')
    print('clean overlap diagnostics complete')
if __name__=='__main__':main()
