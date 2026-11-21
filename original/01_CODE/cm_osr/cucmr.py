from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from .baselines import XLearner
from .nuisance import crossfit_dr, FullNuisance
from .utils import concat_modalities

CANDIDATES=("TRANSCRIPTOMIC","RPPA","JOINT_MOLECULAR")


def _strata(t,y): return np.asarray(t).astype(int)*2+np.asarray(y).astype(int)

def _phi_from_predictions(pred,t,y,clip=(0.05,0.95)):
    e=np.clip(np.asarray(pred["e_hat"],dtype=float),clip[0],clip[1]); m0=np.asarray(pred["mu0_hat"],dtype=float); m1=np.asarray(pred["mu1_hat"],dtype=float)
    t=np.asarray(t,dtype=float); y=np.asarray(y,dtype=float)
    phi=(m1-m0)+t*(y-m1)/e-(1-t)*(y-m0)/(1-e)
    return phi.astype(np.float32),e.astype(np.float32)


def _candidate_X(mods,name):
    C=np.asarray(mods["clinical"],dtype=np.float32)
    if name=="TRANSCRIPTOMIC": return np.concatenate([C,np.asarray(mods["transcriptomic"],dtype=np.float32)],axis=1)
    if name=="RPPA": return np.concatenate([C,np.asarray(mods["rppa"],dtype=np.float32)],axis=1)
    if name=="JOINT_MOLECULAR": return concat_modalities(mods)
    raise ValueError(name)


class ClinicalXLearner:
    def __init__(self,seed=31,n_estimators=100,min_samples_leaf=8):
        self.seed=int(seed); self.n_estimators=int(n_estimators); self.leaf=int(min_samples_leaf)
    def _rf(self,off):
        return RandomForestRegressor(n_estimators=self.n_estimators,min_samples_leaf=self.leaf,random_state=self.seed+off,n_jobs=-1)
    def fit(self,X,t,y):
        X=np.asarray(X); t=np.asarray(t); y=np.asarray(y); self.mu=[]
        for a in [0,1]: self.mu.append(self._rf(10+a).fit(X[t==a],y[t==a]))
        d1=y[t==1]-self.mu[0].predict(X[t==1]); d0=self.mu[1].predict(X[t==0])-y[t==0]
        self.t1=self._rf(21).fit(X[t==1],d1); self.t0=self._rf(22).fit(X[t==0],d0)
        self.e=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=self.seed+23)).fit(X,t); return self
    def predict(self,X):
        X=np.asarray(X); e=self.e.predict_proba(X)[:,1]
        return e*self.t0.predict(X)+(1-e)*self.t1.predict(X)


def _crossfit_anchor(C,t,y,folds,seed):
    C=np.asarray(C); t=np.asarray(t); y=np.asarray(y); oof=np.zeros(len(y),dtype=np.float32)
    skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    for f,(tr,va) in enumerate(skf.split(C,_strata(t,y))):
        m=ClinicalXLearner(seed+100*f).fit(C[tr],t[tr],y[tr]); oof[va]=m.predict(C[va]).astype(np.float32)
    return oof,ClinicalXLearner(seed+999).fit(C,t,y)


def _residual_model(seed):
    return HistGradientBoostingRegressor(learning_rate=0.05,max_iter=120,max_leaf_nodes=15,min_samples_leaf=15,l2_regularization=1.0,early_stopping=False,random_state=int(seed))

def _utility_model(seed):
    return HistGradientBoostingRegressor(learning_rate=0.05,max_iter=120,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1.0,early_stopping=False,random_state=int(seed))

def _crossfit_regressor(X,y,strata,folds,seed,builder):
    X=np.asarray(X); y=np.asarray(y); out=np.zeros(len(y),dtype=np.float32)
    skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    for f,(tr,va) in enumerate(skf.split(X,strata)):
        m=builder(seed+100*f).fit(X[tr],y[tr]); out[va]=m.predict(X[va]).astype(np.float32)
    return out,builder(seed+999).fit(X,y)

def _z(C,anchor,delta,e):
    C=np.asarray(C,dtype=np.float32); a=np.asarray(anchor,dtype=np.float32).reshape(-1,1); d=np.asarray(delta,dtype=np.float32).reshape(-1,1); e=np.asarray(e,dtype=np.float32).reshape(-1,1)
    return np.concatenate([C,a,d,np.abs(d),e,np.minimum(e,1-e)],axis=1).astype(np.float32)

def _fit_stack(P,target):
    P=np.asarray(P,dtype=float); y=np.asarray(target,dtype=float); k=P.shape[1]
    def obj(w): return float(np.mean((y-P@w)**2))
    x0=np.zeros(k); x0[0]=1.0
    res=minimize(obj,x0,method='SLSQP',bounds=[(0,1)]*k,constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1}],options={'maxiter':300,'ftol':1e-10})
    if not res.success or not np.all(np.isfinite(res.x)): return x0.astype(np.float32)
    w=np.clip(res.x,0,1); return (w/(w.sum()+1e-12)).astype(np.float32)


class CUCMRCore:
    """Multi-stage OOF CUCMR development core; thresholds are selected by the DEV aggregator."""
    def __init__(self,seed=31,nuisance_folds=3,anchor_folds=3,candidate_folds=3,utility_folds=3):
        self.seed=int(seed); self.nuisance_folds=int(nuisance_folds); self.anchor_folds=int(anchor_folds); self.candidate_folds=int(candidate_folds); self.utility_folds=int(utility_folds)
    def fit(self,mods,t,y):
        t=np.asarray(t); y=np.asarray(y); C=np.asarray(mods['clinical'],dtype=np.float32); strata=_strata(t,y)
        dr=crossfit_dr(mods,t,y,n_splits=self.nuisance_folds,seed=self.seed+1000); phi=np.asarray(dr['phi'],dtype=np.float32); e=np.asarray(dr['e_hat'],dtype=np.float32)
        anchor_oof,self.anchor=_crossfit_anchor(C,t,y,self.anchor_folds,self.seed+2000); residual=phi-anchor_oof
        self.candidate_models={}; self.utility_models={}; self.utility_score_reference={}; self.train_D={}; base_oof=[anchor_oof]
        for j,m in enumerate(CANDIDATES):
            X=_candidate_X(mods,m)
            delta_oof,final=_crossfit_regressor(X,residual,strata,self.candidate_folds,self.seed+3000+j*1000,_residual_model)
            self.candidate_models[m]=final; cand=anchor_oof+delta_oof; D=(phi-anchor_oof)**2-(phi-cand)**2; Z=_z(C,anchor_oof,delta_oof,e)
            uoof,ufinal=_crossfit_regressor(Z,D,strata,self.utility_folds,self.seed+6000+j*1000,_utility_model)
            self.utility_models[m]=ufinal; self.utility_score_reference[m]=np.sort(uoof.astype(np.float64)); self.train_D[m]=D.astype(np.float32); base_oof.append(cand)
        self.stack_weights=_fit_stack(np.column_stack(base_oof),phi)
        means={m:float(np.mean(self.train_D[m])) for m in CANDIDATES}; best=max(means,key=means.get); self.global_best_candidate=best if means[best]>0 else 'ANCHOR'; self.train_mean_D=means
        self.full_x=XLearner().fit(concat_modalities(mods),t,y)
        self.nuisance=FullNuisance(self.seed+9000).fit(mods,t,y)
        return self
    def _percentile(self,m,s):
        ref=self.utility_score_reference[m]; s=np.asarray(s,dtype=float); return (np.searchsorted(ref,s,side='right')/max(1,len(ref))).astype(np.float32)
    def predict(self,mods):
        C=np.asarray(mods['clinical'],dtype=np.float32); X=concat_modalities(mods); npred=self.nuisance.predict(mods); e=np.asarray(npred['e_hat'],dtype=np.float32)
        anchor=np.asarray(self.anchor.predict(C),dtype=np.float32); out={'anchor_tau':anchor,'full_x_tau':np.asarray(self.full_x.predict(X),dtype=np.float32),'e_hat':e,'nuisance_pred':npred}; bases=[anchor]
        for m in CANDIDATES:
            delta=np.asarray(self.candidate_models[m].predict(_candidate_X(mods,m)),dtype=np.float32); tau=anchor+delta; u=np.asarray(self.utility_models[m].predict(_z(C,anchor,delta,e)),dtype=np.float32)
            out[f'delta_{m}']=delta; out[f'tau_{m}']=tau; out[f'utility_{m}']=u; out[f'utility_pct_{m}']=self._percentile(m,u); bases.append(tau)
        out['stack_tau']=(np.column_stack(bases)@self.stack_weights).astype(np.float32)
        out['global_best_tau']=anchor.copy() if self.global_best_candidate=='ANCHOR' else out[f'tau_{self.global_best_candidate}'].copy()
        return out
    @staticmethod
    def phi_eval(pred,t,y): return _phi_from_predictions(pred,t,y)[0]
