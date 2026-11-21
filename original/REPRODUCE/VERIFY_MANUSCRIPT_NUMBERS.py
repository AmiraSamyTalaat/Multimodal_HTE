from pathlib import Path
import json, math
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def close(a,b,tol=5e-4):
    return math.isclose(float(a),float(b),abs_tol=tol,rel_tol=0)

def check(label,actual,expected,tol=5e-4):
    ok=close(actual,expected,tol)
    print(('PASS' if ok else 'FAIL'),label,'actual=',actual,'expected=',expected)
    if not ok: raise SystemExit(2)

p=pd.read_csv(ROOT/'06_RESULTS/METHOD_SUMMARY_STAGE4.csv').set_index('method')
check('Primary X root-PEHE',p.loc['X_learner','root_pehe_mean'],0.1881)
check('Primary T root-PEHE',p.loc['T_learner_regularized','root_pehe_mean'],0.2024)
check('Primary CM-OSR root-PEHE',p.loc['CM_OSR','root_pehe_mean'],0.8671)
check('Primary early fusion root-PEHE',p.loc['early_fusion_DR_capacity_matched','root_pehe_mean'],2.8350)
check('Primary attention root-PEHE',p.loc['generic_multimodal_attention_DR_capacity_matched','root_pehe_mean'],3.2813)
check('Primary MoE root-PEHE',p.loc['generic_MoE_DR_capacity_matched','root_pehe_mean'],3.7670)

g=json.loads((ROOT/'06_RESULTS/PRIMARY_GATE.json').read_text())
# find common keys defensively
flat=json.dumps(g)
print('PRIMARY_GATE loaded; archived gate keys:',list(g)[:12])

s=pd.read_csv(ROOT/'06_RESULTS/STAGE5A/METHOD_STRESS_SUMMARY.csv')
def stress(cond,method):
    x=s[(s.condition==cond)&(s.method==method)]
    if len(x)!=1: raise SystemExit(f'Missing stress {cond} {method}')
    return x.iloc[0].root_pehe_mean
check('Severe X',stress('SEVERE_WEAK_OVERLAP','X_learner'),0.1364)
check('Severe CM-OSR',stress('SEVERE_WEAK_OVERLAP','CM_OSR'),0.7684)
check('MCAR40 X',stress('MCAR_40','X_learner'),0.2096)
check('MCAR40 CM-OSR',stress('MCAR_40','CM_OSR'),0.7427)

b=pd.read_csv(ROOT/'06_RESULTS/STAGE6B/STAGE6B_METHOD_CONDITION_SUMMARY.csv')
def bval(cond,method,col='root_pehe_mean'):
    x=b[(b.condition==cond)&(b.method==method)]
    if len(x)!=1: raise SystemExit(f'Missing 6B {cond} {method}')
    return x.iloc[0][col]
check('6B primary X',bval('PRIMARY_CROSS_MODAL','X_learner_frozen_reference'),0.1752)
check('6B primary AOSR',bval('PRIMARY_CROSS_MODAL','AOSR_X_gamma2'),0.1787)
check('6B severe X',bval('SEVERE_WEAK_OVERLAP','X_learner_frozen_reference'),0.1412)
check('6B severe AOSR',bval('SEVERE_WEAK_OVERLAP','AOSR_X_gamma2'),0.1485)
check('6B null AOSR mean abs CATE',bval('NULL_EFFECT_NEGATIVE_CONTROL','AOSR_X_gamma2','mean_abs_tau_hat_mean'),0.1028)
check('6B null CM-OSR mean abs CATE',bval('NULL_EFFECT_NEGATIVE_CONTROL','CM_OSR_v1_frozen_reference','mean_abs_tau_hat_mean'),0.6400)

c=pd.read_csv(ROOT/'06_RESULTS/STAGE6C/STAGE6C_METHOD_REGIME_SUMMARY.csv')
def cval(reg,method):
    x=c[(c.regime==reg)&(c.method==method)]
    if len(x)!=1: raise SystemExit(f'Missing 6C {reg} {method}')
    return x.iloc[0].root_pehe_mean
check('6C shared anchor',cval('shared_effect_modifier','ANCHOR_ONLY_X'),0.1617)
check('6C shared AOSR',cval('shared_effect_modifier','AOSR_FULL_FROZEN'),0.1681)
check('6C transcriptome anchor',cval('transcriptome_local_effect_modifier','ANCHOR_ONLY_X'),0.1632)
check('6C transcriptome AOSR',cval('transcriptome_local_effect_modifier','AOSR_FULL_FROZEN'),0.1676)
check('6C RPPA anchor',cval('RPPA_local_effect_modifier','ANCHOR_ONLY_X'),0.1886)
check('6C RPPA AOSR',cval('RPPA_local_effect_modifier','AOSR_FULL_FROZEN'),0.1926)

u=json.loads((ROOT/'06_RESULTS/STAGE7B/STAGE7B_DEVELOPMENT_SELECTION.json').read_text())
check('7B utility/anchor incremental root-PEHE',u['metrics']['incremental_cucmr_root_pehe'],0.10883,1e-5)
check('7B anchor incremental root-PEHE',u['metrics']['incremental_anchor_root_pehe'],0.10883,1e-5)
check('7B acceptance',u['metrics']['incremental_acceptance_rate'],0.0,1e-12)
print('FINAL: ALL MANUSCRIPT HEADLINE NUMBERS VERIFIED')
