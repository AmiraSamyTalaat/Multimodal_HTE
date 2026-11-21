from __future__ import annotations
import numpy as np

def root_pehe(tau_hat,tau_true): return float(np.sqrt(np.mean((np.asarray(tau_hat)-np.asarray(tau_true))**2)))
def cate_calibration_error(tau_hat,tau_true,bins=10):
    q=np.unique(np.quantile(tau_hat,np.linspace(0,1,bins+1)))
    if len(q)<3: return float("nan")
    idx=np.digitize(tau_hat,q[1:-1],right=True); err=[]
    for b in np.unique(idx):
        m=idx==b; err.append(abs(np.mean(tau_hat[m])-np.mean(tau_true[m])))
    return float(np.mean(err))
def policy_regret(tau_hat,tau_true):
    opt=np.maximum(tau_true,0).mean(); got=np.where(tau_hat>0,tau_true,0).mean(); return float(opt-got)
def selective_curve(tau_hat,tau_true,support,fractions=(1.0,.9,.8,.7,.6,.5)):
    out=[]
    order=np.argsort(-support)
    for f in fractions:
        k=max(2,int(len(order)*f)); ii=order[:k]; out.append({"retained":float(f),"root_pehe":root_pehe(tau_hat[ii],tau_true[ii])})
    return out
