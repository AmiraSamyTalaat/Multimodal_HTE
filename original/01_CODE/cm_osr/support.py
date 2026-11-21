from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class DensitySupport:
    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        k = min(15, max(3, len(X)//20))
        self.nn = NearestNeighbors(n_neighbors=k).fit(Z)
        d, _ = self.nn.kneighbors(Z)
        self.scale = np.median(d[:, -1]) + 1e-8
        return self
    def score(self, X):
        Z = self.scaler.transform(X); d, _ = self.nn.kneighbors(Z)
        return np.exp(-d[:, -1] / self.scale).astype(np.float32)

class ReliabilityModel:
    def __init__(self, seed=17):
        self.model = RandomForestRegressor(n_estimators=200, min_samples_leaf=8, random_state=seed, n_jobs=-1)
    def fit(self, features, abs_error): self.model.fit(features, abs_error); return self
    def predict_reliability(self, features):
        err = np.maximum(self.model.predict(features), 0.0)
        return (1.0 / (1.0 + err)).astype(np.float32)
