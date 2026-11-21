from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

def read_gmt(path):
    sets={}
    with open(path,encoding='utf-8') as f:
        for line in f:
            p=line.rstrip('\n').split('\t')
            if len(p)>=3: sets[p[0]]=set(g.upper() for g in p[2:])
    return sets

class PathwayMeanMapper:
    """Outcome-blind fixed gene-set mapping. Expression is samples x genes."""
    def __init__(self,gene_sets,min_genes=5): self.gene_sets=gene_sets; self.min_genes=min_genes
    def fit(self,X: pd.DataFrame):
        self.genes=[str(c).upper() for c in X.columns]
        arr=X.to_numpy(float); self.mean=np.nanmean(arr,axis=0); self.std=np.nanstd(arr,axis=0)+1e-8
        self.indices={n:[i for i,g in enumerate(self.genes) if g in s] for n,s in self.gene_sets.items()}
        self.indices={n:i for n,i in self.indices.items() if len(i)>=self.min_genes}
        if not self.indices: raise ValueError('No pathway has enough mapped genes; inspect gene identifiers/GMT.')
        self.names=sorted(self.indices); return self
    def transform(self,X: pd.DataFrame):
        A=X.to_numpy(float); Z=(np.where(np.isnan(A),self.mean,A)-self.mean)/self.std
        return np.column_stack([Z[:,self.indices[n]].mean(axis=1) for n in self.names]).astype('float32')
