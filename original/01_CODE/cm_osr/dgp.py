from __future__ import annotations
import numpy as np
from scipy.special import expit


def _proj(x: np.ndarray, seed: int, reference_idx=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=x.shape[1]); w /= np.linalg.norm(w) + 1e-12
    z = x @ w
    ref = z if reference_idx is None else z[np.asarray(reference_idx)]
    return (z - ref.mean()) / (ref.std() + 1e-8)


def generate(modalities: dict[str, np.ndarray], seed: int, regime: str,
             propensity_clip=(0.02, 0.98), reference_idx=None) -> dict[str, np.ndarray]:
    c = _proj(modalities["clinical"], 1101, reference_idx)
    g = _proj(modalities["transcriptomic"], 2203, reference_idx)
    p = _proj(modalities["rppa"], 3307, reference_idx)
    g2 = _proj(modalities["transcriptomic"], 4409, reference_idx)
    p2 = _proj(modalities["rppa"], 5519, reference_idx)

    logit_e = 0.25*c + 0.45*g + 0.45*p + 0.65*g*p
    if regime in {"cross_modal_interaction_plus_moderate_weak_overlap", "moderate_weak_overlap"}:
        logit_e *= 1.35
    elif regime == "severe_weak_overlap_stress_only":
        logit_e *= 2.25
    e = np.clip(expit(logit_e), *propensity_clip)

    mu0 = expit(-0.25 + 0.45*c + 0.30*g - 0.20*p + 0.25*g2*p2)
    if regime in {"cross_modal_interaction_plus_moderate_weak_overlap", "cross_modal_interaction_effect_modifier"}:
        tau_raw = 0.10 + 0.16*g*p + 0.08*c*g - 0.05*p
    elif regime == "shared_effect_modifier":
        tau_raw = 0.10 + 0.12*c
    elif regime == "transcriptome_local_effect_modifier":
        tau_raw = 0.10 + 0.18*g
    elif regime == "RPPA_local_effect_modifier":
        tau_raw = 0.10 + 0.18*p
    elif regime == "null_treatment_effect_negative_control":
        tau_raw = np.zeros_like(c)
    else:
        tau_raw = 0.10 + 0.10*g*p
    tau = np.clip(tau_raw, -0.35, 0.45)
    mu1 = np.clip(mu0 + tau, 0.02, 0.98)
    tau = mu1 - mu0

    rng = np.random.default_rng(seed)
    t = rng.binomial(1, e).astype(np.int64)
    p_y = np.where(t == 1, mu1, mu0)
    y = rng.binomial(1, p_y).astype(np.int64)
    return {"t": t, "y": y, "e_true": e.astype(np.float32), "mu0_true": mu0.astype(np.float32),
            "mu1_true": mu1.astype(np.float32), "tau_true": tau.astype(np.float32)}
