
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from sklearn.model_selection import StratifiedKFold

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from cm_osr.splits import locked_split
from cm_osr.preprocess import ModalityPreprocessor
from cm_osr.dgp import generate
from cm_osr.nuisance import crossfit_dr, FullNuisance
from cm_osr.model import CMOSR
from cm_osr.baselines import TLearner
from cm_osr.metrics import root_pehe, policy_regret, cate_calibration_error
from cm_osr.utils import concat_modalities, seed_everything, write_json
from cm_osr.q1_candidates import AnchoredOrthogonalSupportRouter

METHODS = [
    "AOSR_FULL_FROZEN","ANCHOR_ONLY_X","AOSR_UNGATED_RESIDUAL","AOSR_UNIFORM_ROUTING",
    "AOSR_NO_RELIABILITY_GATE","AOSR_NO_DENSITY_GATE","AOSR_NO_OVERLAP_GATE",
    "AOSR_NO_JOINT_EXPERT","AOSR_NONORTHOGONAL_T_RESIDUAL","DIRECT_CM_OSR_DR",
    "CMOSR_SUPPORT_SHRINK_GAMMA2_PREDECLARED_CONTROL",
]
REGIMES = [
    "shared_effect_modifier",
    "transcriptome_local_effect_modifier",
    "RPPA_local_effect_modifier",
]

def strata(t,y):
    return np.asarray(t).astype(int)*2 + np.asarray(y).astype(int)

def crossfit_tlearner(X,t,y,folds,seed):
    oof=np.zeros(len(y),dtype=np.float32)
    skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    for tr,va in skf.split(X,strata(t,y)):
        m=TLearner().fit(X[tr],t[tr],y[tr])
        oof[va]=np.asarray(m.predict(X[va]),dtype=np.float32)
    return oof

def metrics_row(regime,method,pred,true_tau,dgp_seed,net_seed):
    return {
        "dgp_seed":int(dgp_seed),"net_seed":int(net_seed),"regime":regime,"method":method,
        "root_pehe":float(root_pehe(pred,true_tau)),
        "policy_regret":float(policy_regret(pred,true_tau)),
        "cate_calibration_error":float(cate_calibration_error(pred,true_tau)),
        "mean_tau_hat":float(np.mean(pred)),
        "mean_abs_tau_hat":float(np.mean(np.abs(pred))),
    }

def fit_regime(mtr,ttr,ytr,net_seed,cfg):
    seed_everything(net_seed)
    folds=int(cfg["nuisance_folds"]); efolds=int(cfg["effect_folds"])
    epochs=int(cfg["training"]["max_epochs"]); patience=int(cfg["training"]["patience"])
    oof=crossfit_dr(mtr,ttr,ytr,folds,net_seed)
    nuis=FullNuisance(net_seed).fit(mtr,ttr,ytr)

    aosr=AnchoredOrthogonalSupportRouter(
        anchor_kind="X",gamma=2.0,hidden=64,folds=efolds,seed=net_seed,
        expert_epochs=epochs,expert_patience=patience
    ).fit(mtr,ttr,ytr,oof["phi"],oof["e_hat"])

    direct_cm=CMOSR(
        hidden=64,folds=efolds,seed=net_seed,
        expert_epochs=epochs,expert_patience=patience
    ).fit(mtr,oof["phi"],oof["e_hat"])

    Xtr=concat_modalities(mtr)
    t_oof=crossfit_tlearner(Xtr,ttr,ytr,efolds,net_seed+5001)
    nonorth_residual=np.asarray(t_oof,dtype=np.float32)-np.asarray(aosr.anchor_oof_,dtype=np.float32)
    nonorth_router=CMOSR(
        hidden=64,folds=efolds,seed=net_seed+9001,
        expert_epochs=epochs,expert_patience=patience
    ).fit(mtr,nonorth_residual,oof["e_hat"])
    return {"nuis":nuis,"aosr":aosr,"direct_cm":direct_cm,"nonorth_router":nonorth_router}

def predict_ablation(pack,mods):
    e=np.asarray(pack["nuis"].predict(mods)["e_hat"],dtype=np.float32)
    ao=pack["aosr"].predict(mods,e)
    rr=pack["aosr"].residual_router.predict(mods,e)

    anchor=np.asarray(ao["anchor_tau"],dtype=np.float32)
    residual=np.asarray(rr["tau_hat"],dtype=np.float32)
    weights=np.asarray(rr["weights"],dtype=np.float32)
    rel=np.asarray(rr["reliability"],dtype=np.float32)
    dens=np.asarray(rr["density"],dtype=np.float32)
    overlap=np.clip(4.0*e*(1.0-e),0.0,1.0).astype(np.float32)

    full=np.asarray(ao["tau_hat"],dtype=np.float32)
    ungated=anchor+residual

    m=weights.shape[1]
    uw=np.full_like(weights,1.0/m,dtype=np.float32)
    uniform_residual=np.sum(uw*np.asarray(rr["expert_pred"],dtype=np.float32),axis=1)
    uniform_support=np.sum(uw*rel*dens,axis=1)*overlap
    uniform=anchor+(np.clip(uniform_support,0,1)**2)*uniform_residual

    no_rel_support=np.sum(weights*dens,axis=1)*overlap
    no_rel=anchor+(np.clip(no_rel_support,0,1)**2)*residual

    no_den_support=np.sum(weights*rel,axis=1)*overlap
    no_den=anchor+(np.clip(no_den_support,0,1)**2)*residual

    no_ov_support=np.sum(weights*rel*dens,axis=1)
    no_ov=anchor+(np.clip(no_ov_support,0,1)**2)*residual

    av=np.ones((len(e),m),dtype=np.float32); av[:,-1]=0.0
    no_joint=np.asarray(pack["aosr"].predict(mods,e,availability=av)["tau_hat"],dtype=np.float32)

    rn=pack["nonorth_router"].predict(mods,e)
    ns=np.clip(np.asarray(rn["support"],dtype=np.float32),0,1)
    nonorth=anchor+(ns**2)*np.asarray(rn["tau_hat"],dtype=np.float32)

    dc=pack["direct_cm"].predict(mods,e)
    direct=np.asarray(dc["tau_hat"],dtype=np.float32)
    dcs=np.clip(np.asarray(dc["support"],dtype=np.float32),0,1)
    shrink=(dcs**2)*direct

    preds={
        "AOSR_FULL_FROZEN":full,
        "ANCHOR_ONLY_X":anchor,
        "AOSR_UNGATED_RESIDUAL":ungated,
        "AOSR_UNIFORM_ROUTING":uniform,
        "AOSR_NO_RELIABILITY_GATE":no_rel,
        "AOSR_NO_DENSITY_GATE":no_den,
        "AOSR_NO_OVERLAP_GATE":no_ov,
        "AOSR_NO_JOINT_EXPERT":no_joint,
        "AOSR_NONORTHOGONAL_T_RESIDUAL":nonorth,
        "DIRECT_CM_OSR_DR":direct,
        "CMOSR_SUPPORT_SHRINK_GAMMA2_PREDECLARED_CONTROL":shrink,
    }
    diag={
        "mean_w_clinical":float(np.mean(weights[:,0])),
        "mean_w_transcriptomic":float(np.mean(weights[:,1])),
        "mean_w_rppa":float(np.mean(weights[:,2])),
        "mean_w_joint":float(np.mean(weights[:,3])),
        "mean_support":float(np.mean(np.asarray(rr["support"],dtype=np.float32))),
        "mean_abs_routed_correction":float(np.mean(np.abs(np.asarray(ao["routed_correction"],dtype=np.float32)))),
        "mean_abs_residual_raw":float(np.mean(np.abs(residual))),
    }
    return preds,diag

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True); ap.add_argument("--data",required=True)
    ap.add_argument("--dgp-seed",type=int,required=True); ap.add_argument("--net-seed",type=int,required=True)
    a=ap.parse_args()

    root=Path(a.root)
    lock=yaml.safe_load((root/"00_CONFIG"/"stage6c_mechanism_lock.yaml").read_text(encoding="utf-8"))
    cfg=yaml.safe_load((root/"00_CONFIG"/"project.yaml").read_text(encoding="utf-8"))
    exp=yaml.safe_load((root/"03_STATE"/"EXPERIMENT_CONTRACT.yaml").read_text(encoding="utf-8"))

    if lock["status"]!="LOCKED_BEFORE_STAGE6C_RESULTS" or lock["test_policy"]!="TEST_FORBIDDEN_STAGE6C":
        raise SystemExit("Invalid Stage6C lock.")
    if a.dgp_seed not in lock["mechanism_dgp_seeds"] or a.net_seed not in lock["mechanism_network_seeds"]:
        raise SystemExit("Seed outside Stage6C lock.")
    if not set(REGIMES).issubset(set(exp.get("DGP_regimes_prespecified",[]))):
        raise SystemExit("Stage6C regime not originally prespecified.")

    z=np.load(a.data,allow_pickle=False)
    raw={k:z[k] for k in ["clinical","transcriptomic","rppa"]}
    sp=locked_split(len(z["clinical"]))
    tr=sp["train"]; dev=sp["dev"]
    prep=ModalityPreprocessor().fit({k:v[tr] for k,v in raw.items()})
    mods=prep.transform(raw)
    mtr={k:v[tr] for k,v in mods.items()}; mdev={k:v[dev] for k,v in mods.items()}

    rd=root/"08_MECHANISM_RUNS"/f"MECH6C_DGP{a.dgp_seed}_NET{a.net_seed}_{int(time.time())}"
    (rd/"metrics").mkdir(parents=True,exist_ok=True)
    manifest={
        "stage":"STAGE6C_MECHANISM_GENERALIZATION","purpose":"secondary mechanism validation",
        "dgp_seed":int(a.dgp_seed),"net_seed":int(a.net_seed),
        "training_split":"TRAIN_ONLY","evaluation_split":"DEV_ONLY_SECONDARY_MECHANISM",
        "test_accessed":False,"candidate_reselection":False,"gamma_changed":False,"seed_search":False,
        "regimes":REGIMES,"device":"cuda" if torch.cuda.is_available() else "cpu",
        "status":"RUNNING","started":time.time()
    }
    write_json(rd/"RUN_MANIFEST.json",manifest)

    rows=[]; rrows=[]
    for regime in REGIMES:
        d=generate(mods,a.dgp_seed,regime,reference_idx=tr)
        pack=fit_regime(mtr,d["t"][tr],d["y"][tr],a.net_seed,cfg)
        preds,diag=predict_ablation(pack,mdev)
        true_tau=d["tau_true"][dev]
        if set(preds)!=set(METHODS):
            raise RuntimeError("Stage6C method set mismatch.")
        for method,pred in preds.items():
            rows.append(metrics_row(regime,method,pred,true_tau,a.dgp_seed,a.net_seed))
        rrows.append({"dgp_seed":int(a.dgp_seed),"net_seed":int(a.net_seed),"regime":regime,**diag})

    df=pd.DataFrame(rows); rr=pd.DataFrame(rrows)
    if df[["root_pehe","mean_abs_tau_hat"]].isna().any().any():
        raise RuntimeError("NaN in Stage6C metrics.")
    df.to_csv(rd/"metrics"/"mechanism_dev_metrics.csv",index=False)
    rr.to_csv(rd/"metrics"/"router_localization_dev.csv",index=False)

    manifest["status"]="COMPLETE"; manifest["finished"]=time.time()
    write_json(rd/"RUN_MANIFEST.json",manifest)
    write_json(rd/"COMPLETE.json",{
        "stage":"STAGE6C_MECHANISM_GENERALIZATION","dgp_seed":int(a.dgp_seed),"net_seed":int(a.net_seed),
        "status":"COMPLETE","test_accessed":False,"candidate_reselection":False
    })
    print("STAGE6C MECHANISM PAIR COMPLETE",a.dgp_seed,a.net_seed,"rows",len(df),"device",manifest["device"])

if __name__=="__main__":
    main()
