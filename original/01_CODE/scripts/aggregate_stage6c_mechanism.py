
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd, yaml

FULL="AOSR_FULL_FROZEN"; ANCHOR="ANCHOR_ONLY_X"
NONORTH="AOSR_NONORTHOGONAL_T_RESIDUAL"; DIRECT="DIRECT_CM_OSR_DR"; UNIFORM="AOSR_UNIFORM_ROUTING"

def pair_from_name(name):
    m=re.match(r"^MECH6C_DGP(\d+)_NET(\d+)_",name)
    return (int(m.group(1)),int(m.group(2))) if m else None

def cluster_boot(d,B,seed):
    cl=d.groupby("dgp_seed",as_index=False)["diff"].mean()["diff"].to_numpy()
    rng=np.random.default_rng(seed)
    boot=np.asarray([rng.choice(cl,size=len(cl),replace=True).mean() for _ in range(B)])
    return {"mean_diff":float(cl.mean()),"ci95_low":float(np.quantile(boot,.025)),
            "ci95_high":float(np.quantile(boot,.975)),"n_dgp_clusters":int(len(cl))}

def contrast(df,a,b,regimes,B,seed):
    x=df[df["regime"].isin(regimes)]
    piv=x.pivot_table(index=["dgp_seed","net_seed","regime"],columns="method",values="root_pehe")
    d=(piv[a]-piv[b]).rename("diff").reset_index()
    d=d.groupby(["dgp_seed","net_seed"],as_index=False)["diff"].mean()
    return cluster_boot(d,B,seed)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); a=ap.parse_args()
    root=Path(a.root)
    lock=yaml.safe_load((root/"00_CONFIG"/"stage6c_mechanism_lock.yaml").read_text(encoding="utf-8"))
    expected={(int(d),int(n)) for d in lock["mechanism_dgp_seeds"] for n in lock["mechanism_network_seeds"]}
    B=int(lock["bootstrap"]["B"]); bseed=int(lock["bootstrap"]["seed"])

    p=root/"08_MECHANISM_RUNS"; complete={}
    if not p.exists(): raise SystemExit("Stage6C run folder missing.")
    for rd in p.glob("MECH6C_DGP*_NET*_*"):
        if not (rd/"COMPLETE.json").exists(): continue
        pair=pair_from_name(rd.name)
        if pair not in expected: continue
        if pair not in complete or complete[pair].stat().st_mtime<rd.stat().st_mtime:
            complete[pair]=rd
    missing=sorted(expected-set(complete))
    if missing: raise SystemExit(f"Stage6C incomplete: {len(set(complete)&expected)}/{len(expected)}; missing={missing}")

    frames=[]; rframes=[]
    for pair,rd in sorted(complete.items()):
        x=pd.read_csv(rd/"metrics"/"mechanism_dev_metrics.csv"); x["run_dir"]=rd.name; frames.append(x)
        r=pd.read_csv(rd/"metrics"/"router_localization_dev.csv"); r["run_dir"]=rd.name; rframes.append(r)
    df=pd.concat(frames,ignore_index=True); router=pd.concat(rframes,ignore_index=True)

    out=root/"06_RESULTS"/"STAGE6C"; out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"ALL_STAGE6C_MECHANISM_DEV_METRICS.csv",index=False)
    router.to_csv(out/"ALL_STAGE6C_ROUTER_LOCALIZATION.csv",index=False)

    summary=(df.groupby(["regime","method"])
             .agg(n=("root_pehe","size"),root_pehe_mean=("root_pehe","mean"),
                  root_pehe_sd=("root_pehe","std"),policy_regret_mean=("policy_regret","mean"),
                  calibration_mean=("cate_calibration_error","mean"),
                  mean_abs_tau_hat=("mean_abs_tau_hat","mean")).reset_index())
    summary.to_csv(out/"STAGE6C_METHOD_REGIME_SUMMARY.csv",index=False)

    regimes=list(lock["regimes"])
    tests={}
    tests["H1_full_minus_nonorth"]=contrast(df,FULL,NONORTH,regimes,B,bseed+1)
    tests["H2_full_minus_direct"]=contrast(df,FULL,DIRECT,regimes,B,bseed+2)
    tests["H3_full_minus_uniform"]=contrast(df,FULL,UNIFORM,regimes,B,bseed+3)
    tests["full_minus_anchor_pooled"]=contrast(df,FULL,ANCHOR,regimes,B,bseed+4)
    tests["full_minus_anchor_transcriptome_local"]=contrast(
        df,FULL,ANCHOR,["transcriptome_local_effect_modifier"],B,bseed+5)
    tests["full_minus_anchor_rppa_local"]=contrast(
        df,FULL,ANCHOR,["RPPA_local_effect_modifier"],B,bseed+6)

    q=router.copy()
    q["tr_minus_rppa"]=q["mean_w_transcriptomic"]-q["mean_w_rppa"]
    piv=q.pivot_table(index=["dgp_seed","net_seed"],columns="regime",values="tr_minus_rppa")
    d=(piv["transcriptome_local_effect_modifier"]-piv["RPPA_local_effect_modifier"]).rename("diff").reset_index()
    tests["H6_router_localization"]=cluster_boot(d,B,bseed+7)

    pd.DataFrame([{"contrast":k,**v} for k,v in tests.items()]).to_csv(
        out/"STAGE6C_PAIRED_CLUSTER_BOOTSTRAP.csv",index=False)

    mean_by=df.groupby("method")["root_pehe"].mean()
    full_mean=float(mean_by[FULL]); anchor_mean=float(mean_by[ANCHOR])
    H1=tests["H1_full_minus_nonorth"]["ci95_high"]<0
    H2=tests["H2_full_minus_direct"]["ci95_high"]<0
    H3=tests["H3_full_minus_uniform"]["ci95_high"]<0
    H4=full_mean<=1.10*anchor_mean
    H5=(tests["full_minus_anchor_transcriptome_local"]["ci95_high"]<0 or
        tests["full_minus_anchor_rppa_local"]["ci95_high"]<0)
    H6=tests["H6_router_localization"]["ci95_low"]>0

    gates={
        "H1_orthogonal_residual_target":bool(H1),
        "H2_anchor_residual_vs_direct":bool(H2),
        "H3_learned_routing":bool(H3),
        "H4_anchor_noninferiority_10pct":bool(H4),
        "H5_incremental_value_in_at_least_one_localized_regime":bool(H5),
        "H6_router_localization":bool(H6),
    }

    if all(gates.values()):
        status="STRONG_MECHANISTIC_SUPPORT_FOR_NOVEL_TECHNIQUE"
        nxt="PROCEED_CHER_LOB_THEN_LOCKED_TRIO_WITH_FROZEN_AOSR"
    elif H1 and H2 and H3 and H4 and H6 and not H5:
        status="PARTIAL_MECHANISTIC_SUPPORT_REQUIRES_REAL_TRIAL_VALUE"
        nxt="PROCEED_REAL_TRIAL_VALIDATION_BUT_DO_NOT_CLAIM_INCREMENTAL_PEHE_OVER_ANCHOR"
    else:
        status="WEAK_METHOD_INCREMENT_DO_NOT_POSITION_AS_PRIMARY_NOVEL_TECHNIQUE"
        nxt="DO_NOT_FORCE_Q1_METHOD_CLAIM; REFRAME_OR_STOP_BEFORE_REAL_TRIAL_PRIMARY_METHOD_POSITIONING"

    decision={
        "stage":"STAGE6C_MECHANISM_GENERALIZATION","grid_complete":True,
        "observed_pairs":len(expected),"required_pairs":len(expected),
        "frozen_method":"AOSR_X_gamma2","training_split":"TRAIN_ONLY",
        "evaluation_split":"DEV_ONLY_SECONDARY_MECHANISM","test_accessed":False,
        "candidate_reselection":False,"regimes":regimes,
        "full_mean_root_pehe":full_mean,"anchor_only_mean_root_pehe":anchor_mean,
        "gates":gates,"status":status,"next":nxt,
        "note":"Secondary mechanism validation on originally prespecified unused DGP regimes; Stage6B remains unchanged."
    }
    (out/"STAGE6C_MECHANISM_DECISION.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
    print(json.dumps(decision,indent=2))
    print("Stage6C aggregation complete:",out)

if __name__=="__main__":
    main()
