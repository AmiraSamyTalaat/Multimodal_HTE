import numpy as np
from cm_osr.dgp import generate
from cm_osr.metrics import root_pehe
from cm_osr.splits import locked_split

def test_split_disjoint():
    s=locked_split(100)
    assert len(set(s['train']) & set(s['dev']))==0
    assert len(set(s['train']) & set(s['test']))==0
    assert len(set(s['dev']) & set(s['test']))==0
    assert sum(map(len,s.values()))==100

def test_dgp_reproducible():
    rng=np.random.default_rng(1)
    m={'clinical':rng.normal(size=(50,3)),'transcriptomic':rng.normal(size=(50,4)),'rppa':rng.normal(size=(50,2))}
    a=generate(m,101,'cross_modal_interaction_plus_moderate_weak_overlap')
    b=generate(m,101,'cross_modal_interaction_plus_moderate_weak_overlap')
    assert np.allclose(a['tau_true'],b['tau_true']) and np.array_equal(a['t'],b['t']) and np.array_equal(a['y'],b['y'])

def test_metric_zero():
    x=np.array([-.1,.2,.3]); assert root_pehe(x,x)==0.0
