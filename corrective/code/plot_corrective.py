from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from corrective_common import COR

FIG=COR/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':160,'savefig.dpi':220,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fixed_file=COR/'results/fixed_dgp_splits/dgp_cluster_results.csv'
if fixed_file.exists():
    d=pd.read_csv(fixed_file)
    methods=['zero_effect','clinical_X_learner','full_X_learner','clinical_R_full_nuisance_Rloss_selected']
    labels=['Zero','Clinical X','Full X','Clinical R + full nuisance']
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,(regime,g) in zip(axes,d.groupby('effect_regime')):
        vals=[g[g.method==m].root_pehe.to_numpy() for m in methods]
        ax.boxplot(vals,tick_labels=labels,showfliers=True,medianprops={'color':'#c64747'})
        ax.set_title(regime.replace('_',' ').title());ax.tick_params(axis='x',rotation=25)
        ax.set_ylabel('Root-PEHE (DGP mean across three resplits)')
        ax.grid(axis='y',alpha=.25)
    fig.suptitle('Fixed DGP: split sensitivity with unchanged TEST truth')
    fig.tight_layout();fig.savefig(FIG/'fixed_dgp_split_sensitivity.png',bbox_inches='tight');fig.savefig(FIG/'fixed_dgp_split_sensitivity.pdf',bbox_inches='tight');plt.close(fig)
fair_file=COR/'results/fair_neural/summary_results.csv'
if fair_file.exists():
    d=pd.read_csv(fair_file)
    methods=['CM_OSR','uniform_same_experts','bagged_early_fusion_5fold_capacity_matched',
             'clinical_X_learner','full_X_learner','zero_effect']
    labels=['CM-OSR','Uniform','Bagged fusion','Clinical X','Full X','Zero']
    regimes=sorted(d.effect_regime.unique())
    fig,axes=plt.subplots(len(regimes),2,figsize=(11,4.2*len(regimes)),squeeze=False)
    for row,regime in enumerate(regimes):
        for col,cal in enumerate(['raw','DEV_Rloss_selected']):
            ax=axes[row,col];vals=[];lo=[];hi=[];names=[]
            for m,label in zip(methods,labels):
                c=cal if m in methods[:3] else 'raw'
                if col==0 and m not in methods[:3]:continue
                z=d[(d.effect_regime==regime)&(d.method==m)&(d.calibration==c)]
                if len(z)!=1 or not bool(z.inference_ready.iloc[0]):continue
                x=z.iloc[0];names.append(label);vals.append(x['mean']);lo.append(x['mean']-x.ci95_lower);hi.append(x.ci95_upper-x['mean'])
            if vals:
                ax.bar(range(len(vals)),vals,color=['#34699a','#759ab8','#b2b9c1','#5b8e63','#b08744','#888888'][:len(vals)])
                ax.errorbar(range(len(vals)),vals,yerr=[lo,hi],fmt='none',ecolor='black',capsize=3,lw=1)
                ax.set_xticks(range(len(vals)),names,rotation=25,ha='right')
            ax.set_ylabel('Root-PEHE, DGP-cluster 95% CI')
            ax.set_title(regime.replace('_',' ').title()+' — '+('raw' if col==0 else 'DEV R-loss shrinkage'))
            ax.grid(axis='y',alpha=.25)
    fig.suptitle('Neural comparison: raw and operationally calibrated results')
    fig.tight_layout();fig.savefig(FIG/'fair_neural_comparison.png',bbox_inches='tight');fig.savefig(FIG/'fair_neural_comparison.pdf',bbox_inches='tight');plt.close(fig)
print('figures',len(list(FIG.iterdir())))
