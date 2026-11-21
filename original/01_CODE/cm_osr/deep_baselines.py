from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_device(m):
    try:
        return next(m.parameters()).device
    except StopIteration:
        return _device()


class EarlyFusionDRNet(nn.Module):
    def __init__(self,d,hidden=128):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,x): return self.net(x).squeeze(-1)


class TARNet(nn.Module):
    def __init__(self,d,hidden=128):
        super().__init__(); self.rep=nn.Sequential(nn.Linear(d,hidden),nn.ELU(),nn.Linear(hidden,hidden),nn.ELU())
        self.h0=nn.Linear(hidden,1); self.h1=nn.Linear(hidden,1)
    def forward(self,x):
        z=self.rep(x); return self.h0(z).squeeze(-1),self.h1(z).squeeze(-1),z


class DragonNet(TARNet):
    def __init__(self,d,hidden=128): super().__init__(d,hidden); self.prop=nn.Linear(hidden,1)
    def forward(self,x):
        y0,y1,z=super().forward(x); return y0,y1,self.prop(z).squeeze(-1),z


class TransformerHTE(nn.Module):
    def __init__(self,dims,d_model=48,nhead=4,layers=2):
        super().__init__(); self.names=list(dims); self.proj=nn.ModuleDict({k:nn.Linear(d,d_model) for k,d in dims.items()})
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=d_model*2,batch_first=True,dropout=0.1)
        self.enc=nn.TransformerEncoder(enc,num_layers=layers); self.h0=nn.Linear(d_model,1); self.h1=nn.Linear(d_model,1)
    def forward(self,mods):
        toks=torch.stack([self.proj[k](mods[k]) for k in self.names],1); z=self.enc(toks).mean(1)
        return self.h0(z).squeeze(-1),self.h1(z).squeeze(-1)


class GenericAttentionDR(nn.Module):
    def __init__(self,dims,d_model=48):
        super().__init__(); self.names=list(dims); self.enc=nn.ModuleDict({k:nn.Sequential(nn.Linear(d,d_model),nn.GELU()) for k,d in dims.items()})
        self.score=nn.Linear(d_model,1); self.head=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Linear(d_model,1))
    def forward(self,mods):
        e=torch.stack([self.enc[k](mods[k]) for k in self.names],1); w=torch.softmax(self.score(e).squeeze(-1),1); z=(w.unsqueeze(-1)*e).sum(1); return self.head(z).squeeze(-1)


class GenericMoEDR(nn.Module):
    def __init__(self,dims,hidden=48):
        super().__init__(); self.names=list(dims)
        self.experts=nn.ModuleDict({k:nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,1)) for k,d in dims.items()})
        self.router=nn.Sequential(nn.Linear(sum(dims.values()),hidden),nn.GELU(),nn.Linear(hidden,len(dims)))
    def forward(self,mods):
        pred=torch.stack([self.experts[k](mods[k]).squeeze(-1) for k in self.names],1)
        joint=torch.cat([mods[k] for k in self.names],1); w=torch.softmax(self.router(joint),1); return (w*pred).sum(1)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _best_size(builder,target,tolerance=.10,lo=8,hi=1024,step=4):
    best=None
    for h in range(lo,hi+1,step):
        n=count_params(builder(h)); err=abs(n-target)/max(target,1)
        if best is None or err<best[0]: best=(err,h,n)
    return best[1],best[2],best[0]


def find_hidden_for_target(d_in,target,tolerance=.10,lo=8,hi=1024):
    h,n,e=_best_size(lambda x: EarlyFusionDRNet(d_in,x),target,tolerance,lo,hi,8)
    return h,n


def find_attention_dim_for_target(dims,target,tolerance=.10,lo=8,hi=512):
    return _best_size(lambda x: GenericAttentionDR(dims,d_model=x),target,tolerance,lo,hi,4)


def find_moe_hidden_for_target(dims,target,tolerance=.10,lo=8,hi=512):
    return _best_size(lambda x: GenericMoEDR(dims,hidden=x),target,tolerance,lo,hi,4)


def _loader(*arrs,batch=64,seed=17):
    ts=[torch.from_numpy(np.asarray(a,dtype=np.float32)) for a in arrs]; ds=TensorDataset(*ts)
    return DataLoader(ds,batch_size=min(batch,len(ds)),shuffle=True,generator=torch.Generator().manual_seed(seed))


def fit_phi_mlp(X,phi,hidden=128,seed=17,epochs=250,lr=1e-3):
    sc=StandardScaler().fit(X); Z=sc.transform(X).astype('float32'); torch.manual_seed(seed); dev=_device(); m=EarlyFusionDRNet(Z.shape[1],hidden).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    for _ in range(epochs):
        for xb,yb in _loader(Z,phi,batch=64,seed=seed):
            xb=xb.to(dev); yb=yb.to(dev); opt.zero_grad(); loss=((m(xb)-yb)**2).mean(); loss.backward(); opt.step()
    return m,sc


def predict_phi_mlp(m,sc,X):
    dev=_model_device(m)
    with torch.no_grad(): return m(torch.from_numpy(sc.transform(X).astype('float32')).to(dev)).detach().cpu().numpy()


def fit_outcome_net(kind,X,t,y,hidden=128,seed=17,epochs=250,alpha_cfr=0.1):
    sc=StandardScaler().fit(X); Z=sc.transform(X).astype('float32'); torch.manual_seed(seed); dev=_device()
    m=(DragonNet(Z.shape[1],hidden) if kind=='dragonnet' else TARNet(Z.shape[1],hidden)).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); bce=nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for xb,tb,yb in _loader(Z,t,y,batch=64,seed=seed):
            xb=xb.to(dev); tb=tb.to(dev); yb=yb.to(dev); opt.zero_grad()
            if kind=='dragonnet': y0,y1,pl,z=m(xb)
            else: y0,y1,z=m(xb)
            yf=torch.where(tb>0.5,y1,y0); loss=bce(yf,yb)
            if kind=='cfrnet':
                z0=z[tb<.5]; z1=z[tb>=.5]
                if len(z0)>0 and len(z1)>0: loss=loss+alpha_cfr*((z0.mean(0)-z1.mean(0))**2).mean()
            if kind=='dragonnet': loss=loss+0.2*bce(pl,tb)
            loss.backward(); opt.step()
    return m,sc


def predict_outcome_tau(m,sc,X,kind):
    dev=_model_device(m); Z=torch.from_numpy(sc.transform(X).astype('float32')).to(dev)
    with torch.no_grad():
        if kind=='dragonnet': y0,y1,_,_=m(Z)
        else: y0,y1,_=m(Z)
        return (torch.sigmoid(y1)-torch.sigmoid(y0)).detach().cpu().numpy()


def _mods_t(mods,device=None):
    device=device or _device(); return {k:torch.from_numpy(np.asarray(v,dtype=np.float32)).to(device) for k,v in mods.items()}


def fit_multimodal_phi(kind,mods,phi,seed=17,epochs=250,hidden_or_dim=None):
    dims={k:v.shape[1] for k,v in mods.items()}; torch.manual_seed(seed); dev=_device()
    if kind=='attention': m=GenericAttentionDR(dims,d_model=int(hidden_or_dim or 48))
    else: m=GenericMoEDR(dims,hidden=int(hidden_or_dim or 48))
    m=m.to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); keys=list(mods); n=len(phi); rng=np.random.default_rng(seed)
    for _ in range(epochs):
        perm=rng.permutation(n)
        for start in range(0,n,64):
            ii=perm[start:start+64]; mb={k:torch.from_numpy(mods[k][ii].astype('float32')).to(dev) for k in keys}; yb=torch.from_numpy(phi[ii].astype('float32')).to(dev)
            opt.zero_grad(); loss=((m(mb)-yb)**2).mean(); loss.backward(); opt.step()
    return m


def predict_multimodal_phi(m,mods):
    dev=_model_device(m)
    with torch.no_grad(): return m(_mods_t(mods,dev)).detach().cpu().numpy()


def fit_transformer_outcome(mods,t,y,seed=17,epochs=250):
    dims={k:v.shape[1] for k,v in mods.items()}; torch.manual_seed(seed); dev=_device(); m=TransformerHTE(dims).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); bce=nn.BCEWithLogitsLoss(); n=len(y); rng=np.random.default_rng(seed); keys=list(mods)
    for _ in range(epochs):
        perm=rng.permutation(n)
        for start in range(0,n,64):
            ii=perm[start:start+64]; mb={k:torch.from_numpy(mods[k][ii].astype('float32')).to(dev) for k in keys}; tb=torch.from_numpy(t[ii].astype('float32')).to(dev); yb=torch.from_numpy(y[ii].astype('float32')).to(dev)
            opt.zero_grad(); y0,y1=m(mb); yf=torch.where(tb>.5,y1,y0); loss=bce(yf,yb); loss.backward(); opt.step()
    return m


def predict_transformer_tau(m,mods):
    dev=_model_device(m)
    with torch.no_grad():
        y0,y1=m(_mods_t(mods,dev)); return (torch.sigmoid(y1)-torch.sigmoid(y0)).detach().cpu().numpy()
