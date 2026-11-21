from __future__ import annotations
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class ModalityPreprocessor:
    def __init__(self): self.parts={}
    def fit(self, modalities):
        for k,X in modalities.items():
            imp=SimpleImputer(strategy='median').fit(X)
            Z=imp.transform(X)
            sc=StandardScaler().fit(Z)
            self.parts[k]=(imp,sc)
        return self
    def transform(self, modalities):
        return {k:self.parts[k][1].transform(self.parts[k][0].transform(X)).astype('float32') for k,X in modalities.items()}
