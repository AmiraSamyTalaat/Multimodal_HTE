from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .baselines import XLearner, RLearner
from .model import CMOSR
from .utils import concat_modalities


def _strata(t, y):
    return (np.asarray(t).astype(int) * 2 + np.asarray(y).astype(int))


def _new_anchor(kind: str):
    k = kind.upper()
    if k == "X":
        return XLearner()
    if k == "R":
        return RLearner()
    raise ValueError(f"Unknown anchor kind: {kind}")


def crossfit_anchor(X, t, y, kind="X", n_splits=5, seed=17):
    """Leakage-safe OOF anchor effect predictions plus a full-train anchor."""
    X = np.asarray(X)
    t = np.asarray(t)
    y = np.asarray(y)
    oof = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, _strata(t, y)):
        m = _new_anchor(kind).fit(X[tr], t[tr], y[tr])
        oof[va] = np.asarray(m.predict(X[va]), dtype=np.float32)
    final = _new_anchor(kind).fit(X, t, y)
    return oof, final


class AnchoredOrthogonalSupportRouter:
    """
    AOSR: strong conventional causal anchor + support-routed orthogonal residual.

    Training target for the routed multimodal correction:
        residual_phi = cross_fitted_DR_phi - cross_fitted_anchor_tau

    Inference:
        tau_hat = tau_anchor + support**gamma * delta_routed

    No true CATE is used by the model. gamma is selected only in Stage6A DEV and
    must be frozen before Stage6B confirmatory evaluation.
    """
    def __init__(self, anchor_kind="X", gamma=1.0, hidden=64, folds=5, seed=17,
                 expert_epochs=250, expert_patience=30):
        self.anchor_kind = anchor_kind.upper()
        self.gamma = float(gamma)
        self.hidden = int(hidden)
        self.folds = int(folds)
        self.seed = int(seed)
        self.expert_epochs = int(expert_epochs)
        self.expert_patience = int(expert_patience)

    def fit(self, modalities, t, y, phi, e_hat):
        X = concat_modalities(modalities)
        anchor_oof, self.anchor = crossfit_anchor(
            X, t, y, kind=self.anchor_kind, n_splits=self.folds, seed=self.seed + 5001
        )
        residual = np.asarray(phi, dtype=np.float32) - anchor_oof
        self.residual_router = CMOSR(
            hidden=self.hidden,
            folds=self.folds,
            seed=self.seed + 9001,
            expert_epochs=self.expert_epochs,
            expert_patience=self.expert_patience,
        ).fit(modalities, residual, e_hat)
        self.anchor_oof_ = anchor_oof
        self.residual_target_oof_ = residual
        return self

    def predict(self, modalities, e_hat, availability=None):
        X = concat_modalities(modalities)
        anchor_tau = np.asarray(self.anchor.predict(X), dtype=np.float32)
        rr = self.residual_router.predict(modalities, e_hat, availability=availability)
        support = np.clip(np.asarray(rr["support"], dtype=np.float32), 0.0, 1.0)
        gate = support ** self.gamma
        correction = gate * np.asarray(rr["tau_hat"], dtype=np.float32)
        tau = anchor_tau + correction
        out = dict(rr)
        out.update({
            "tau_hat": tau.astype(np.float32),
            "anchor_tau": anchor_tau.astype(np.float32),
            "residual_tau": np.asarray(rr["tau_hat"], dtype=np.float32),
            "support_gate": gate.astype(np.float32),
            "routed_correction": correction.astype(np.float32),
            "anchor_kind": self.anchor_kind,
            "gamma": self.gamma,
        })
        return out


class SupportShrunkCMOSR:
    """
    Conservative reference: shrink the frozen CM-OSR effect toward zero by its
    own support score. Included as a development control, not as the preferred
    new method.
    """
    def __init__(self, gamma=1.0, hidden=64, folds=5, seed=17,
                 expert_epochs=250, expert_patience=30):
        self.gamma = float(gamma)
        self.base = CMOSR(
            hidden=hidden, folds=folds, seed=seed,
            expert_epochs=expert_epochs, expert_patience=expert_patience
        )

    def fit(self, modalities, phi, e_hat):
        self.base.fit(modalities, phi, e_hat)
        return self

    def predict(self, modalities, e_hat, availability=None):
        rr = self.base.predict(modalities, e_hat, availability=availability)
        support = np.clip(np.asarray(rr["support"], dtype=np.float32), 0.0, 1.0)
        gate = support ** self.gamma
        tau = gate * np.asarray(rr["tau_hat"], dtype=np.float32)
        out = dict(rr)
        out.update({
            "tau_hat": tau.astype(np.float32),
            "support_gate": gate.astype(np.float32),
            "unshrunk_tau": np.asarray(rr["tau_hat"], dtype=np.float32),
            "gamma": self.gamma,
        })
        return out
