from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .utils import concat_modalities


def _propensity_model(seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed))


def _outcome_model(seed: int):
    return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)


def crossfit_dr(modalities: dict[str, np.ndarray], t: np.ndarray, y: np.ndarray,
                n_splits: int = 5, seed: int = 17, clip=(0.05, 0.95)) -> dict[str, np.ndarray]:
    X = concat_modalities(modalities)
    n = len(y)
    e = np.zeros(n); m0 = np.zeros(n); m1 = np.zeros(n)
    strata = (t.astype(int) * 2 + y.astype(int))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in splitter.split(X, strata):
        pm = _propensity_model(seed); pm.fit(X[tr], t[tr]); e[va] = pm.predict_proba(X[va])[:, 1]
        for arm, dest in [(0, m0), (1, m1)]:
            idx = tr[t[tr] == arm]
            om = _outcome_model(seed + arm + 1)
            om.fit(X[idx], y[idx]); dest[va] = om.predict_proba(X[va])[:, 1]
    e = np.clip(e, clip[0], clip[1])
    phi = (m1 - m0) + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    return {"e_hat": e.astype(np.float32), "mu0_hat": m0.astype(np.float32),
            "mu1_hat": m1.astype(np.float32), "phi": phi.astype(np.float32)}

class FullNuisance:
    """Frozen train-only nuisance models for DEV/TEST inference."""
    def __init__(self, seed=17, clip=(0.05,0.95)):
        self.seed=seed; self.clip=clip
    def fit(self, modalities, t, y):
        X=concat_modalities(modalities)
        self.propensity=_propensity_model(self.seed).fit(X,t)
        self.outcomes=[]
        for arm in [0,1]:
            idx=np.where(t==arm)[0]
            mdl=_outcome_model(self.seed+arm+1).fit(X[idx],y[idx]); self.outcomes.append(mdl)
        return self
    def predict(self, modalities):
        X=concat_modalities(modalities)
        e=np.clip(self.propensity.predict_proba(X)[:,1],self.clip[0],self.clip[1])
        m0=self.outcomes[0].predict_proba(X)[:,1]; m1=self.outcomes[1].predict_proba(X)[:,1]
        return {"e_hat":e.astype("float32"),"mu0_hat":m0.astype("float32"),"mu1_hat":m1.astype("float32")}
