from __future__ import annotations
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .dgp import _proj

REGIMES = [
    "NO_INCREMENTAL_MOLECULAR_SIGNAL",
    "TRANSCRIPTOME_INCREMENTAL",
    "RPPA_INCREMENTAL",
    "JOINT_MOLECULAR_SYNERGY",
    "PATIENT_SUBGROUP_SWITCHING",
]


def _std_train(z, tr):
    z=np.asarray(z,dtype=float)
    mu=float(np.mean(z[tr])); sd=float(np.std(z[tr]))
    return ((z-mu)/(sd+1e-8)).astype(np.float32)


def _residualized_pc_score(X, C, tr, seed, n_components=5):
    X=np.asarray(X,dtype=float); C=np.asarray(C,dtype=float); tr=np.asarray(tr,dtype=int)
    k=max(1,min(int(n_components),X.shape[1],len(tr)-1))
    pca=PCA(n_components=k,svd_solver="full").fit(X[tr])
    pc=pca.transform(X)
    rng=np.random.default_rng(int(seed))
    w=rng.normal(size=k); w=w/(np.linalg.norm(w)+1e-12)
    raw=pc@w
    reg=Ridge(alpha=1.0).fit(C[tr],raw[tr])
    resid=raw-reg.predict(C)
    return _std_train(resid,tr)


def _joint_score(g,p,C,tr):
    raw=np.asarray(g)*np.asarray(p)
    Z=np.column_stack([C,np.asarray(g),np.asarray(p)])
    reg=Ridge(alpha=1.0).fit(Z[tr],raw[tr])
    resid=raw-reg.predict(Z)
    return _std_train(resid,tr)


def generate_stage7(modalities, seed, regime, reference_idx, incremental_ratio=0.60, propensity_clip=(0.05,0.95)):
    if regime not in REGIMES:
        raise ValueError(f"Unknown Stage7 regime: {regime}")
    tr=np.asarray(reference_idx,dtype=int)
    C=np.asarray(modalities["clinical"],dtype=np.float32)
    T=np.asarray(modalities["transcriptomic"],dtype=np.float32)
    R=np.asarray(modalities["rppa"],dtype=np.float32)

    c=_proj(C,1101,tr)
    g_prog=_proj(T,2203,tr); p_prog=_proj(R,3307,tr)
    g2=_proj(T,4409,tr); p2=_proj(R,5519,tr)

    # Prognostic/assignment mechanisms remain multimodal, so molecular variables
    # can be confounder/prognostic information even in the no-incremental-effect regime.
    logit_e=0.25*c+0.45*g_prog+0.45*p_prog+0.65*g_prog*p_prog
    e=np.clip(expit(logit_e),propensity_clip[0],propensity_clip[1])
    mu0=expit(-0.25+0.45*c+0.30*g_prog-0.20*p_prog+0.25*g2*p2)

    g_inc=_residualized_pc_score(T,C,tr,int(seed)+7101,5)
    p_inc=_residualized_pc_score(R,C,tr,int(seed)+8209,5)
    j_inc=_joint_score(g_inc,p_inc,C,tr)

    base=0.10+0.12*c
    base_sd=float(np.std(base[tr]))
    alpha=float(incremental_ratio)*base_sd
    source=np.zeros(len(c),dtype=np.int8)
    inc=np.zeros(len(c),dtype=np.float32)

    if regime=="TRANSCRIPTOME_INCREMENTAL":
        inc=alpha*g_inc; source[:]=1
    elif regime=="RPPA_INCREMENTAL":
        inc=alpha*p_inc; source[:]=2
    elif regime=="JOINT_MOLECULAR_SYNERGY":
        inc=alpha*j_inc; source[:]=3
    elif regime=="PATIENT_SUBGROUP_SWITCHING":
        cut=float(np.median(c[tr])); upper=c>=cut
        inc=alpha*np.where(upper,g_inc,p_inc).astype(np.float32)
        source=np.where(upper,1,2).astype(np.int8)

    tau_raw=np.clip(base+inc,-0.35,0.45)
    mu1=np.clip(mu0+tau_raw,0.02,0.98)
    tau=(mu1-mu0).astype(np.float32)

    rng=np.random.default_rng(int(seed))
    t=rng.binomial(1,e).astype(np.int64)
    py=np.where(t==1,mu1,mu0)
    y=rng.binomial(1,py).astype(np.int64)
    return {
        "t":t,"y":y,"e_true":e.astype(np.float32),"mu0_true":mu0.astype(np.float32),
        "mu1_true":mu1.astype(np.float32),"tau_true":tau,
        "incremental_source":source,"incremental_signal":inc.astype(np.float32),
        "g_incremental":g_inc,"p_incremental":p_inc,"joint_incremental":j_inc,
        "incremental_alpha":np.float32(alpha),
    }
