from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split

def locked_split(n, seed=20260918, proportions=(.60,.20,.20)):
    idx=np.arange(n)
    tr, rem=train_test_split(idx,test_size=1-proportions[0],random_state=seed,shuffle=True)
    dev_rel=proportions[1]/(proportions[1]+proportions[2])
    dev,te=train_test_split(rem,train_size=dev_rel,random_state=seed+1,shuffle=True)
    return {"train":np.sort(tr),"dev":np.sort(dev),"test":np.sort(te)}
