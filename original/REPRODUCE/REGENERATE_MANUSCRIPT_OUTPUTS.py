from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'REPRODUCED_MANUSCRIPT_OUTPUTS'; OUT.mkdir(exist_ok=True)

# Table 2
p=pd.read_csv(ROOT/'06_RESULTS/METHOD_SUMMARY_STAGE4.csv')
p.to_csv(OUT/'Table2_primary_full.csv',index=False)
# Table 3 selected methods/conditions
s=pd.read_csv(ROOT/'06_RESULTS/STAGE5A/METHOD_STRESS_SUMMARY.csv')
sel_methods=['X_learner','T_learner_regularized','CM_OSR','early_fusion_DR_capacity_matched','generic_multimodal_attention_DR_capacity_matched','generic_MoE_DR_capacity_matched']
conds=['CLEAN_REFERENCE','MAR_CLINICAL','MCAR_10','MCAR_25','MCAR_40','SEVERE_WEAK_OVERLAP','NULL_EFFECT_NEGATIVE_CONTROL']
piv=(s[s.method.isin(sel_methods)&s.condition.isin(conds)].pivot_table(index='method',columns='condition',values='root_pehe_mean',aggfunc='first'))
piv.to_csv(OUT/'Table3_stress_selected.csv')
# Table 4 source contrasts
pd.read_csv(ROOT/'06_RESULTS/STAGE6B/STAGE6B_PAIRED_CLUSTER_BOOTSTRAP.csv').to_csv(OUT/'Table4_stage6b_contrasts.csv',index=False)
pd.read_csv(ROOT/'06_RESULTS/STAGE6C/STAGE6C_PAIRED_CLUSTER_BOOTSTRAP.csv').to_csv(OUT/'Table4_stage6c_contrasts.csv',index=False)

# Figure 2
order=p.sort_values('root_pehe_mean')
fig,ax=plt.subplots(figsize=(8,6)); ax.barh(order.method,order.root_pehe_mean); ax.invert_yaxis(); ax.set_xlabel('Root-PEHE (lower is better)'); ax.set_title('Primary held-out benchmark on real multimodal covariates'); fig.tight_layout(); fig.savefig(OUT/'Figure2_primary_regenerated.png',dpi=300); plt.close(fig)

# Figure 3
name={'X_learner':'X-learner','T_learner_regularized':'T-learner','CM_OSR':'CM-OSR','early_fusion_DR_capacity_matched':'Early-fusion DR','generic_multimodal_attention_DR_capacity_matched':'Generic attention DR','generic_MoE_DR_capacity_matched':'Generic MoE DR'}
fig,ax=plt.subplots(figsize=(9,5.5))
for m in sel_methods:
    vals=[]
    for cnd in conds:
        x=s[(s.method==m)&(s.condition==cnd)]
        vals.append(float(x.iloc[0].root_pehe_mean))
    ax.plot(range(len(conds)),vals,marker='o',label=name[m])
ax.set_yscale('log'); ax.set_xticks(range(len(conds)),['Clean','MAR','MCAR10','MCAR25','MCAR40','Severe overlap','Null effect'],rotation=0); ax.set_ylabel('Root-PEHE (log scale; lower is better)'); ax.legend(ncol=3,fontsize=8); fig.tight_layout(); fig.savefig(OUT/'Figure3_stress_regenerated.png',dpi=300); plt.close(fig)

# Figure 4 values in machine-readable form
b=pd.read_csv(ROOT/'06_RESULTS/STAGE6B/STAGE6B_METHOD_CONDITION_SUMMARY.csv')
c=pd.read_csv(ROOT/'06_RESULTS/STAGE6C/STAGE6C_METHOD_REGIME_SUMMARY.csv')
u=json.loads((ROOT/'06_RESULTS/STAGE7B/STAGE7B_DEVELOPMENT_SELECTION.json').read_text())
summary={'stage7b':u,'stage6b_primary':b[b.condition=='PRIMARY_CROSS_MODAL'].to_dict(orient='records'),'stage6b_null':b[b.condition=='NULL_EFFECT_NEGATIVE_CONTROL'].to_dict(orient='records'),'stage6c':c.to_dict(orient='records')}
(OUT/'Figure4_source_values.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Regenerated manuscript outputs in',OUT)
