from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def _device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _model_device(m):
    try: return next(m.parameters()).device
    except StopIteration: return _device()


class MLPRegressor(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _fit_one(X, y, seed=17, hidden=64, epochs=250, patience=30, lr=1e-3, batch_size=64):
    torch.manual_seed(seed); dev=_device()
    scaler = StandardScaler().fit(X); Z = scaler.transform(X).astype("float32")
    model = MLPRegressor(Z.shape[1], hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(Z), torch.from_numpy(y.astype("float32")))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best = (float("inf"), None); bad = 0; Zall=torch.from_numpy(Z).to(dev); yall=torch.from_numpy(y.astype("float32")).to(dev)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb=xb.to(dev); yb=yb.to(dev); opt.zero_grad(); pred = model(xb); loss = torch.mean((pred-yb)**2); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): cur = float(torch.mean((model(Zall)-yall)**2).detach().cpu())
        if cur < best[0] - 1e-6:
            best = (cur, copy.deepcopy(model.state_dict())); bad = 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best[1]); return model, scaler


def predict(model, scaler, X):
    Z = scaler.transform(X).astype("float32"); model.eval(); dev=_model_device(model)
    with torch.no_grad(): return model(torch.from_numpy(Z).to(dev)).detach().cpu().numpy()


def crossfit_expert(X, target, n_splits=5, seed=17, hidden=64, **train_kw):
    oof = np.zeros(len(target), dtype=np.float32); kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(kf.split(X)):
        m, s = _fit_one(X[tr], target[tr], seed=seed+k, hidden=hidden, **train_kw); oof[va] = predict(m, s, X[va])
    final, scaler = _fit_one(X, target, seed=seed+999, hidden=hidden, **train_kw)
    return {"oof": oof, "model": final, "scaler": scaler, "param_count": count_params(final)}
