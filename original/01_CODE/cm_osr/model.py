from __future__ import annotations
import numpy as np
from .experts import crossfit_expert, predict
from .support import DensitySupport, ReliabilityModel
from .router import make_router_features, train_router


class CMOSR:
    def __init__(self, hidden=64, folds=5, seed=17, expert_epochs=250, expert_patience=30):
        self.hidden=hidden; self.folds=folds; self.seed=seed; self.expert_epochs=expert_epochs; self.expert_patience=expert_patience

    def fit(self, modalities, phi, e_hat):
        self.names = list(modalities.keys()) + ["joint"]
        Xs = [modalities[k] for k in modalities]; Xs.append(np.concatenate([modalities[k] for k in modalities], axis=1))
        self.experts=[]; self.density_models=[]; self.reliability_models=[]; oof=[]; dens=[]; rel=[]
        overlap=(4*e_hat*(1-e_hat)).astype("float32"); availability=np.ones((len(phi), len(Xs)), dtype="float32")
        for j,X in enumerate(Xs):
            ex=crossfit_expert(X, phi, n_splits=self.folds, seed=self.seed+11*j, hidden=self.hidden,
                               epochs=self.expert_epochs, patience=self.expert_patience, lr=1e-3, batch_size=64)
            self.experts.append(ex); oof.append(ex["oof"]); ds=DensitySupport().fit(X); self.density_models.append(ds); dens.append(ds.score(X))
        oof=np.column_stack(oof); dens=np.column_stack(dens); disagreement=np.std(oof,axis=1)
        for j in range(oof.shape[1]):
            rf=np.column_stack([overlap,dens[:,j],disagreement,availability[:,j]])
            rm=ReliabilityModel(self.seed+j).fit(rf,np.abs(phi-oof[:,j])); self.reliability_models.append(rm); rel.append(rm.predict_reliability(rf))
        rel=np.column_stack(rel); feats=make_router_features(overlap,availability,dens,rel,oof)
        self.router=train_router(feats,availability,oof,phi,seed=self.seed+777,reliability=rel); self.train_oof=oof; self.train_rel=rel; return self

    def predict(self, modalities, e_hat, availability=None):
        Xs=[modalities[k] for k in modalities]; Xs.append(np.concatenate(Xs,axis=1)); pred=np.column_stack([predict(ex["model"],ex["scaler"],X) for ex,X in zip(self.experts,Xs)])
        n,m=pred.shape
        if availability is None: availability=np.ones((n,m),dtype="float32")
        overlap=(4*e_hat*(1-e_hat)).astype("float32"); dens=np.column_stack([dm.score(X) for dm,X in zip(self.density_models,Xs)]); disagreement=np.std(pred,axis=1); rel=[]
        for j,rm in enumerate(self.reliability_models):
            rf=np.column_stack([overlap,dens[:,j],disagreement,availability[:,j]]); rel.append(rm.predict_reliability(rf))
        rel=np.column_stack(rel); feats=make_router_features(overlap,availability,dens,rel,pred)
        import torch
        dev=next(self.router.parameters()).device
        with torch.no_grad(): w=self.router(torch.from_numpy(feats).to(dev),torch.from_numpy(availability.astype("float32")).to(dev)).detach().cpu().numpy()
        tau=(w*pred).sum(1); support=(w*rel*dens).sum(1)*overlap
        return {"tau_hat":tau.astype("float32"),"weights":w.astype("float32"),"support":support.astype("float32"),"expert_pred":pred.astype("float32"),"reliability":rel.astype("float32"),"density":dens.astype("float32")}
