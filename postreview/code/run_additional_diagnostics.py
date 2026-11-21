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

SEEDS=[101,211,307,401,503]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def desc(x):
    x=np.asarray(x,dtype=float)
    return {'mean':float(x.mean()),'sd':float(x.std()),'min':float(x.min()),
            'q01':float(np.quantile(x,.01)),'q05':float(np.quantile(x,.05)),
            'median':float(np.median(x)),'q95':float(np.quantile(x,.95)),
            'q99':float(np.quantile(x,.99)),'max':float(x.max())}
def ess(w):
    w=np.asarray(w,dtype=float)
    return float(w.sum()**2/(np.sum(w*w)+1e-12)) if w.size else 0.0
def missing_mask(mods,indices,c,seed,condition,split_tag):
    code={'MCAR_10':10,'MCAR_25':25,'MCAR_40':40,'MAR_CLINICAL':77}[condition]
    split_code=1 if split_tag=='dev' else 2
    rng=np.random.default_rng(710000+seed*100+code*10+split_code)
    n=len(indices)
    prob=np.full(n,float(condition.split('_')[1])/100) if condition.startswith('MCAR') else np.clip(.10+.25*expit(c[indices]),0,.95)
    mask_t=rng.random(n)<prob;mask_r=rng.random(n)<prob
    return prob,mask_t,mask_r
def run():
    out=REV/'results'/'additional';out.mkdir(parents=True,exist_ok=True)
    data=SRC/'02_DATA/processed/ispy2_core_multimodal.npz'
    with np.load(data,allow_pickle=False) as z:raw={k:z[k] for k in ['clinical','transcriptomic','rppa']}
    sp=locked_split(len(raw['clinical']));tr=sp['train']
    mods=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()}).transform(raw)
    rows=[];shift=[];maskrows=[];masksummary=[]
    for seed in SEEDS:
        primary=generate(mods,seed,'cross_modal_interaction_plus_moderate_weak_overlap',reference_idx=tr)
        severe=generate(mods,seed,'severe_weak_overlap_stress_only',reference_idx=tr)
        null=generate(mods,seed,'null_treatment_effect_negative_control',reference_idx=tr)
        c=_proj(mods['clinical'],1101,tr)
        for regime,d in [('PRIMARY',primary),('SEVERE_WEAK_OVERLAP',severe),('NULL_EFFECT',null)]:
            for part,ix in sp.items():
                e=np.asarray(d['e_true'][ix],dtype=float);t=np.asarray(d['t'][ix],dtype=int)
                tau=np.asarray(d['tau_true'][ix],dtype=float)
                wt=1/e[t==1];wc=1/(1-e[t==0])
                rows.append({'dgp_seed':seed,'regime':regime,'split':part,'n':len(ix),
                             'treatment_prevalence':float(t.mean()),'n_treated':int(t.sum()),
                             'ess_treated':ess(wt),'ess_control':ess(wc),
                             'propensity_clip_fraction':float(np.mean((e<=.0200001)|(e>=.9799999))),
                             **{'e_'+k:v for k,v in desc(e).items()},
                             **{'true_cate_'+k:v for k,v in desc(tau).items()}})
        for part,ix in sp.items():
            a=np.asarray(primary['tau_true'][ix]);b=np.asarray(severe['tau_true'][ix])
            shift.append({'dgp_seed':seed,'split':part,'n':len(ix),
                          'true_cate_rmse_primary_vs_severe':float(np.sqrt(np.mean((a-b)**2))),
                          'true_cate_mean_absolute_change':float(np.mean(np.abs(a-b))),
                          'true_cate_fraction_changed_gt_1e4':float(np.mean(np.abs(a-b)>1e-4)),
                          'primary_true_cate_sd':float(np.std(a)),
                          'severe_true_cate_sd':float(np.std(b))})
        for condition in ['MCAR_10','MCAR_25','MCAR_40','MAR_CLINICAL']:
            for part in ['dev','test']:
                ix=sp[part]
                prob,mt,mr=missing_mask(mods,ix,c,seed,condition,part)
                for j,patient_ix in enumerate(ix):
                    maskrows.append({'dgp_seed':seed,'condition':condition,'split':part,
                                     'patient_index':int(patient_ix),'missing_probability':float(prob[j]),
                                     'clinical_available':1,'transcriptomic_available':int(not mt[j]),
                                     'rppa_available':int(not mr[j]),'joint_available':int(not(mt[j] or mr[j])),
                                     'transcriptomic_missing_indicator':int(mt[j]),'rppa_missing_indicator':int(mr[j])})
                masksummary.append({'dgp_seed':seed,'condition':condition,'split':part,
                                    'transcriptomic_missing_fraction':float(mt.mean()),
                                    'rppa_missing_fraction':float(mr.mean()),
                                    'both_missing_fraction':float(np.mean(mt&mr)),
                                    'mean_target_probability':float(prob.mean())})
    pd.DataFrame(rows).to_csv(out/'overlap_true_cate_by_seed.csv',index=False)
    pd.DataFrame(shift).to_csv(out/'stage5a_effect_structure_shift.csv',index=False)
    pd.DataFrame(maskrows).to_csv(out/'availability_masks_postreview.csv',index=False)
    pd.DataFrame(masksummary).to_csv(out/'missingness_realized_postreview.csv',index=False)
    original=pd.read_csv(SRC/'06_RESULTS/STAGE5A/ALL_STAGE5A_MISSINGNESS_REALIZED.csv')
    our=pd.DataFrame(masksummary)
    errors=[]
    for _,r in original.iterrows():
        key=our[(our.dgp_seed==r.dgp_seed)&(our.condition==r.condition)&(our.split==r.split)]
        if len(key)!=1:errors.append(f'mask row count for {r.dgp_seed} {r.condition} {r.split}');continue
        v=key.iloc[0]
        for a,b in [('transcriptomic_missing_fraction','transcriptomic_missing_fraction'),('rppa_missing_fraction','rppa_missing_fraction'),('both_missing_fraction','both_molecular_missing_fraction')]:
            if abs(float(v[a])-float(r[b]))>2e-6:errors.append(f'mask mismatch {r.dgp_seed} {r.condition} {r.split} {a}')
    df=pd.read_csv(SRC/'06_RESULTS/STAGE7B/STAGE7B_CUCMR_DEV_PREDICTIONS_FROZEN_RULE.csv')
    truth=df.true_tau.to_numpy();anchor=df.anchor_tau.to_numpy()
    candidates=np.column_stack([df[f'tau_{name}'].to_numpy() for name in ['TRANSCRIPTOMIC','RPPA','JOINT_MOLECULAR']])
    se_anchor=(anchor-truth)**2
    se_cand=(candidates-truth[:,None])**2
    best=np.min(se_cand,axis=1)
    df['oracle_any_candidate_improves']=best<se_anchor
    df['oracle_best_candidate_sq_error_gain']=np.maximum(0,se_anchor-best)
    df['true_molecular_source_present']=df.true_source>0
    df['gate_accepts']=df.accepted_update.astype(bool)
    df['gate_true_positive_by_source']=df.gate_accepts&df.true_molecular_source_present
    df['gate_false_positive_by_source']=df.gate_accepts&(~df.true_molecular_source_present)
    group=[]
    for (seed,learner,regime),g in df.groupby(['dgp_seed','learner_seed','regime']):
        def rates(positive):
            positive=np.asarray(positive,dtype=bool);accepted=g.gate_accepts.to_numpy()
            return float(np.mean(accepted[positive])) if positive.any() else float('nan'),float(np.mean(accepted[~positive])) if (~positive).any() else float('nan')
        tpr_source,fpr_source=rates(g.true_molecular_source_present)
        tpr_gain,fpr_gain=rates(g.oracle_any_candidate_improves)
        group.append({'dgp_seed':seed,'learner_seed':learner,'regime':regime,'n_dev':len(g),
                      'gate_acceptance_rate':float(g.gate_accepts.mean()),
                      'tpr_by_true_source':tpr_source,'fpr_by_true_source':fpr_source,
                      'tpr_by_realized_sq_error_gain':tpr_gain,'fpr_by_realized_sq_error_gain':fpr_gain,
                      'oracle_beneficial_candidate_fraction':float(g.oracle_any_candidate_improves.mean()),
                      'oracle_best_candidate_mean_sq_error_gain':float(g.oracle_best_candidate_sq_error_gain.mean()),
                      'anchor_root_pehe':float(np.sqrt(np.mean((g.anchor_tau-g.true_tau)**2))),
                      'gate_root_pehe':float(np.sqrt(np.mean((g.cucmr_tau-g.true_tau)**2))),
                      'oracle_best_candidate_root_pehe':float(np.sqrt(np.mean(np.minimum((g.anchor_tau-g.true_tau)**2,np.min(np.column_stack([(g[f'tau_{n}']-g.true_tau)**2 for n in ['TRANSCRIPTOMIC','RPPA','JOINT_MOLECULAR']]),axis=1)))))} )
    pd.DataFrame(group).to_csv(out/'utility_gate_rates_by_seed.csv',index=False)
    df[['dgp_seed','learner_seed','regime','sample_index','true_tau','true_source','gate_accepts','oracle_any_candidate_improves','oracle_best_candidate_sq_error_gain']].to_csv(out/'utility_gate_patient_audit.csv',index=False)
    status={'stage':'ADDITIONAL_DIAGNOSTICS_POSTREVIEW','status':'COMPLETE' if not errors else 'ERROR',
            'finished_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),
            'data_sha256':sha(data),'original_stage5a_mask_comparison_errors':errors,
            'scope':'Read-only DGP reconstruction and Stage7B DEV prediction audit; no model training'}
    (out/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print('status',status['status'],'mask_errors',len(errors))
    print(pd.DataFrame(shift).query("split=='test'").to_string(index=False))
    print(pd.DataFrame(group).groupby('regime')[['gate_acceptance_rate','oracle_beneficial_candidate_fraction','oracle_best_candidate_mean_sq_error_gain']].mean().to_string())
if __name__=='__main__':run()
