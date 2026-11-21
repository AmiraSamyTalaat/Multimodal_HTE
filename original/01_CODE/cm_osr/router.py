from __future__ import annotations
import numpy as np
import torch
from torch import nn


def _device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SupportRouter(nn.Module):
    def __init__(self, n_experts: int, features_per_expert: int = 4, hidden: int = 32, temperature: float = 0.75):
        super().__init__(); self.n_experts=n_experts; self.temperature=temperature
        self.net = nn.Sequential(nn.Linear(n_experts*features_per_expert + 2, hidden), nn.GELU(), nn.Linear(hidden, n_experts))
    def forward(self, f, availability=None):
        logits = self.net(f) / self.temperature
        if availability is not None: logits = logits.masked_fill(availability <= 0, -1e9)
        return torch.softmax(logits, dim=1)


def make_router_features(overlap, availability, density, reliability, expert_pred):
    n, m = expert_pred.shape; disagreement = np.mean(np.abs(expert_pred - expert_pred.mean(axis=1, keepdims=True)), axis=1); chunks=[]
    for j in range(m): chunks += [overlap[:,None], availability[:,j:j+1], density[:,j:j+1], reliability[:,j:j+1]]
    base=np.concatenate(chunks, axis=1)
    return np.concatenate([base, disagreement[:,None], expert_pred.std(axis=1, keepdims=True)], axis=1).astype("float32")


def train_router(features, availability, expert_oof, target, seed=17, epochs=400, lr=1e-3,
                 reliability=None, reliability_penalty=0.05, entropy_penalty=0.001):
    torch.manual_seed(seed); dev=_device()
    f=torch.from_numpy(features.astype("float32")).to(dev); av=torch.from_numpy(availability.astype("float32")).to(dev)
    p=torch.from_numpy(expert_oof.astype("float32")).to(dev); y=torch.from_numpy(target.astype("float32")).to(dev); m=expert_oof.shape[1]
    model=SupportRouter(m, features_per_expert=4).to(dev); opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rel=torch.from_numpy(reliability.astype("float32")).to(dev) if reliability is not None else None
    for _ in range(epochs):
        opt.zero_grad(); w=model(f,av); pred=(w*p).sum(1); loss=((pred-y)**2).mean()
        if rel is not None: loss = loss + reliability_penalty * (w*(1-rel)).mean()
        ent = -(w.clamp_min(1e-8)*w.clamp_min(1e-8).log()).sum(1).mean(); loss = loss - entropy_penalty*ent
        loss.backward(); opt.step()
    return model
