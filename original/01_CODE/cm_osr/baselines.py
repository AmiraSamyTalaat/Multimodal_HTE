from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class TLearner:
    def fit(self,X,t,y):
        self.m=[]
        for a in [0,1]:
            mdl=RandomForestRegressor(n_estimators=300,min_samples_leaf=8,random_state=17+a,n_jobs=-1)
            mdl.fit(X[t==a],y[t==a]); self.m.append(mdl)
        return self
    def predict(self,X): return self.m[1].predict(X)-self.m[0].predict(X)


class XLearner:
    def fit(self,X,t,y):
        self.mu=[]
        for a in [0,1]:
            m=RandomForestRegressor(n_estimators=300,min_samples_leaf=8,random_state=31+a,n_jobs=-1).fit(X[t==a],y[t==a]); self.mu.append(m)
        d1=y[t==1]-self.mu[0].predict(X[t==1]); d0=self.mu[1].predict(X[t==0])-y[t==0]
        self.t1=RandomForestRegressor(n_estimators=300,min_samples_leaf=8,random_state=33,n_jobs=-1).fit(X[t==1],d1)
        self.t0=RandomForestRegressor(n_estimators=300,min_samples_leaf=8,random_state=35,n_jobs=-1).fit(X[t==0],d0)
        self.e=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(X,t); return self
    def predict(self,X):
        e=self.e.predict_proba(X)[:,1]; return e*self.t0.predict(X)+(1-e)*self.t1.predict(X)

class DRLearner:
    def fit(self,X,phi):
        self.m=make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(X,phi); return self
    def predict(self,X): return self.m.predict(X)

class RLearner:
    def fit(self,X,t,y):
        e=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(X,t).predict_proba(X)[:,1]
        m=RandomForestRegressor(n_estimators=300,min_samples_leaf=8,random_state=19,n_jobs=-1).fit(X,y).predict(X)
        rY=y-m; rT=t-e; denom=np.where(np.abs(rT)<0.05,np.where(rT>=0,0.05,-0.05),rT); z=rY/denom
        w=rT**2
        self.m=RandomForestRegressor(n_estimators=300,min_samples_leaf=8,random_state=23,n_jobs=-1).fit(X,z,sample_weight=w); return self
    def predict(self,X): return self.m.predict(X)
