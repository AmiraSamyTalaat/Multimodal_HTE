from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from corrective_common import COR,REV,LOCK,now

def rd(path):return pd.read_csv(path) if path.exists() else pd.DataFrame()
def table(df,cols,rename=None):
    if df.empty:return '_التشغيل جارٍ، ولم تكتمل وحدات الاستدلال بعد._'
    z=df[cols].copy()
    for c in z.select_dtypes('number'):
        if c.startswith('n_') or c=='completed_units':continue
        z[c]=z[c].map(lambda x:f'{x:.4f}' if pd.notna(x) else '')
    if rename:z=z.rename(columns=rename)
    return z.to_markdown(index=False)
clinical=rd(COR/'results/clinical_adjustment/summary_results.csv')
clinical_paired=rd(COR/'results/clinical_adjustment/paired_dgp_comparisons.csv')
tuning=rd(COR/'results/nonoracle_tuning/summary_results.csv')
fixed=rd(COR/'results/fixed_dgp_splits/summary_results.csv')
fair=rd(COR/'results/fair_neural/summary_results.csv')
fair_per=rd(COR/'results/fair_neural/per_seed_results.csv')
paired=rd(COR/'results/fair_neural/paired_dgp_comparisons.csv')
cross=rd(COR/'results/fair_neural/cross_platform_validation.csv')
old_drift=rd(COR/'reporting/old_replication_split_truth_drift.csv')
kaggle_code=rd(COR/'reporting/kaggle_dataset_code_verification.csv')
fixed_status=json.loads((COR/'results/fixed_dgp_splits/RUN_STATUS.json').read_text()) if (COR/'results/fixed_dgp_splits/RUN_STATUS.json').exists() else {}
fair_status=json.loads((COR/'results/fair_neural/RUN_STATUS.json').read_text()) if (COR/'results/fair_neural/RUN_STATUS.json').exists() else {}
report_status=json.loads((COR/'reporting/reporting_checks.json').read_text())
required=['archived_clinical_X','archived_full_X','archived_zero_effect','clinical_R_full_fitted_Rloss_selected']
c=clinical[clinical.method.isin(required)]
f=fixed[fixed.method.isin(['zero_effect','full_X_learner','clinical_X_learner','clinical_R_full_nuisance_Rloss_selected'])]
f=f[f.inference_ready==True] if not f.empty else f
n=tuning[tuning.method.isin(['clinical_X_learner_NONORACLE','clinical_X_learner_ORACLE_DIAGNOSTIC',
                              'full_X_learner_NONORACLE','full_X_learner_ORACLE_DIAGNOSTIC'])]
fair_display=fair[fair.method.isin(['CM_OSR','uniform_same_experts','bagged_early_fusion_5fold_capacity_matched',
                                   'full_X_learner','full_T_learner','zero_effect'])] if not fair.empty else fair
fair_display=fair_display[fair_display.inference_ready==True] if not fair_display.empty else fair_display
lines=[f'# B01 benchmark — corrective post-review scientific report\n\nGenerated {now()}. These analyses were designed after reviewer feedback and inspection of archived results. They are **post-review**, not prespecified primary experiments. Original benchmark results and negative findings remain unchanged. Paper 2 was not used.\n',
'''## Reviewer objection disposition

The accompanying `reviewer_objection_matrix.csv` maps all 50 items in the prior forensic audit to evidence, required manuscript wording, or a genuinely additional experiment. The six scientifically consequential objections are summarized here.

| Objection | Evidence produced | Disposition |
|---|---|---|
| Clinical-only X/T omit molecular confounders | Refit clinical effect stage with TRAIN cross-fitted all-modality propensity/outcome nuisance; compare clinical-fitted and oracle-true nuisance, archived Clinical X/T, Full X/T and zero | Experiment complete; manuscript must narrow the causal adjustment claim and disclose nuisance limitations |
| Split sensitivity changes CATE in the old replication | New fixed-DGP cache freezes t, Y(0), Y(1), CATE and anchor TEST before three TRAIN/DEV resplits; verifies TEST truth and IDs hashes across resplits | Experiment complete only when 72 units finish; explain old replication as combined DGP/preprocessing perturbation |
| True DEV CATE used to choose hyperparameters | Equal 3-candidate leaf grids for X/T full/clinical; observed DEV R-loss versus separately labeled true-DEV-CATE diagnostic; no TEST selection | Experiment complete; manuscript must distinguish operational from simulation-only tuning |
| Neural fairness and routing attribution | CM-OSR, uniform same experts and five-model bagged fusion on shared pseudo-outcomes, same stopping rule and matched deployed parameter count; X/T and zero on same DGP/test | Conditional experiment complete only when 24 neural units finish; unequal optimizer budgets and one learner seed mean routing alone cannot be credited |
| CI units and metric definitions | Recomputed intervals from DGP-cluster means; checked 15 prediction files and 225 policies; Stage5A CATE shift quantified | Checks complete; wording correction needed for Oracle constant and severe overlap |
| Reproducibility and rights | Code, DGP-level aggregates, environment, manifests and validation in release-candidate ZIP; no individual records | Release candidate only; rights clearance and Zenodo publication require separate human decision |
''',
'''## 1. Clinical-only causal adjustment

The original treatment propensity and baseline outcome simulation depend on transcriptomic and RPPA features. Clinical X/T fit on clinical features alone, so their contrast is not a test of fully adjusted causal estimation; under molecular effect mechanisms it also cannot represent the full individual-level CATE. The new R-learner uses all modalities to estimate propensity and outcome nuisance on TRAIN with five-fold cross-fitting, then fits a weighted clinical-only effect stage. Ridge alpha is selected by observed DEV R-loss. With variable propensity, this R-loss stage targets a propensity-variance-weighted projection onto clinical features, not necessarily the unweighted conditional mean CATE given clinical features. The oracle-true nuisance is simulation-only. The no-incremental-molecular-signal fixed-DGP regime helps distinguish the target-function restriction from the adjustment issue, but all findings remain conditional on this simulator.

Mean Root-PEHE across five independent original DGP seeds:
''',table(c,['method','mean_root_pehe','mcse_dgp','n_dgp']),
'''The full-nuisance clinical R estimate differs from zero-effect by only about -0.0013 Root-PEHE on these five DGP seeds (paired Monte Carlo SE about 0.0063); it does not establish superiority. Its mean 0.1611 is also worse than the clinical-fitted nuisance version 0.1598. The fitted full-nuisance result is a finite-sample diagnostic, not proof that high-dimensional nuisance models consistently remove confounding. The true-nuisance diagnostic and unfavorable contrasts remain in the CSV.

Paired DGP differences (clinical R with full fitted nuisance minus comparator; negative favors the clinical R predictor):
''',table(clinical_paired,['method_b','mean','ci95_lower','ci95_upper','n_independent_dgp']),
'''## 2. Fixed-DGP patient split sensitivity

The old replication called Stage7 generation after fitting each split's TRAIN preprocessing and supplied `reference_idx=tr`; its CATE therefore changes with the split. The new experiment fits a DGP reference preprocessing once on the 588-patient development pool, generates all potential outcomes and truth once per seed and effect regime, then freezes them across three resplits of that pool. Each learner still fits its own preprocessing on its TRAIN subset. The anchor TEST set remains the same 148 patients. DGP seed is the Monte Carlo cluster; three resplits are correlated repeated measurements, conditional on the fixed patient cohort.

Status: **'''+fixed_status.get('status','NOT_STARTED')+f"** ({fixed_status.get('completed_units',0)}/{fixed_status.get('expected_units',72)} units).\n\n"+table(f,['effect_regime','method','mean_root_pehe','mean_within_dgp_split_sd','mcse_dgp','n_dgp']),
'''## 3. Non-oracle tuning

For each of the five original DGP seeds, X/T on full or clinical features use 300 trees and the same `min_samples_leaf` candidates 4, 8 and 16. Nuisance models are fit on TRAIN; the operational criterion is observed DEV R-loss. True DEV CATE is used only in the separately labeled simulation diagnostic. TEST is evaluated after each criterion's DEV selection. This limited 12-fit-per-DGP budget is not a broad hyperparameter search.
''',table(n,['method','mean_root_pehe','mcse_dgp','n_dgp']),
'''## 4. Neural comparison and training budget

The neural comparison uses 12 newly locked DGP seeds in each of two effect regimes, a single fixed patient split and learner seed 17. CM-OSR and Uniform share the same trained experts; bagged early fusion has five cross-fit models, with deployed trainable parameter counts matched within 10%. The three neural methods use the same TRAIN-only five-fold nuisance pseudo-outcomes and train-MSE stopping criterion (max 250 epochs, patience 30). X/T instead fit observed arm outcomes with original fixed hyperparameters, as their definitions require. Actual fit counts, epochs, optimizer steps, parameters and wall time are in `training_budget.csv`. The router has 400 additional optimizer steps. Raw estimates and a shared shrinkage grid selected by observed DEV R-loss are separate rows. Selection can collapse to zero; that negative outcome is retained. The experiment does not equalize gradient budget, architecture, nuisance estimation error, or multiple learner seeds, so it does not isolate routing causally.

Status: **'''+fair_status.get('status','NOT_STARTED')+f"** ({fair_status.get('completed_units',0)}/{fair_status.get('expected_units',24)} neural units).\n\n"+table(fair_display,['effect_regime','method','calibration','mean','mcse_dgp','n_independent_dgp']),
'''Paired raw Root-PEHE differences use DGP seed as the unit; negative means CM-OSR has smaller error:
''',table(paired,['effect_regime','method_a','method_b','mean','ci95_lower','ci95_upper','n_independent_dgp']),
'''Raw neural prediction extremes and clipping diagnostics averaged across the two regimes and 12 DGP seeds each:
''',table(fair_per[(fair_per['mode']=='neural')&(fair_per.calibration=='raw')].groupby('method',as_index=False).agg(
  root_pehe=('root_pehe','mean'),clipped_root_pehe=('clipped_root_pehe','mean'),
  out_of_range_fraction=('fraction_outside_minus1_plus1','mean')) if not fair_per.empty else fair_per,
  ['method','root_pehe','clipped_root_pehe','out_of_range_fraction']),
'''## 5. Reporting corrections

All seven reporting checks passed. The earlier replication intervals use 30 DGP clusters per regime, after averaging three patient splits within each seed; 270 regime-by-split rows are not 270 independent observations. However, because the old code regenerated the DGP under each split, that old interval targets a combined split/preprocessing/DGP perturbation. The new fixed-DGP experiment separates this issue.

The TEST-CATE-mean constant has Root-PEHE equal to the SD of true TEST CATE. It is the best *constant* squared-error predictor for that TEST set and is nondeployable; it is not a universal lower bound for heterogeneous estimators. The estimated ATE is the mean five-fold cross-fitted DR pseudo-outcome from TRAIN, distinct from the crude TRAIN arm difference. Policy regret is `mean(max(tau,0)) - mean(tau * 1[pred>0])`, a simulation-only oracle comparison rather than observed clinical benefit. In the archived severe-overlap Stage5A contrast, mean TEST CATE RMSE versus the primary regime is 0.0872 and about 99.3% of TEST CATE entries change. Use the separate clean propensity perturbation for an overlap-only statement.

## 6. Release and remaining work

The release candidate contains no patient matrices, generated individual potential outcomes/CATE, identifiers, or individual predictions. NCBI GEO's public access policy does not itself clear every contributor's redistribution rights. The private Kaggle dataset remains private, and nothing was uploaded to Zenodo. Before public release, a human must confirm rights and inspect the aggregate-only archive. The manuscript needs wording changes, but the original manuscript has not been edited here.

Additional experiments genuinely needed for broader claims: independent patient cohorts or externally sourced outcomes; multiple learner seeds and a genuinely matched total training budget if attributing a difference to routing; and stronger nuisance validation if claiming causal adjustment rather than performance in a known simulation. These are limits on the claim, not grounds to discard negative results or select seeds after TEST inspection.

Sources for rights: https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html , https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194040 , https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196093 .\n''']
if not cross.empty:
    lines.append(f"\nCross-platform numerical check: {len(cross)} matching fixed-DGP units; TEST index hashes all match = {bool(cross.test_index_sha_match.all())}; maximum zero-effect Root-PEHE difference = {cross.zero_root_pehe_absolute_difference.max():.3g}. Exact CATE byte hashes may differ between the Windows and Kaggle numerical stacks; this is recorded, not suppressed.\n")
if not old_drift.empty:
    lines.append('\nArchived split-dependent truth drift, recomputed on all 736 same patients for 30 original DGP seeds × three mechanisms × three split pairs (simulator truth only; no learner rerun):\n\n'+
      table(old_drift.groupby('effect_regime',as_index=False).agg(mean_cate_rmse=('full_cohort_cate_rmse','mean'),
           mean_fraction_changed=('full_cohort_fraction_changed_gt_1e4','mean')),
           ['effect_regime','mean_cate_rmse','mean_fraction_changed']))
if not kaggle_code.empty:
    lines.append(f"\nPrivate Kaggle dataset code provenance: {int(kaggle_code['match'].sum())}/{len(kaggle_code)} downloaded core Python modules match the benchmark source snapshot SHA-256. The private dataset itself is not included in the release ZIP.\n")
(COR/'CORRECTIVE_REVIEW_REPORT.md').write_text('\n\n'.join(lines),encoding='utf-8')
print(COR/'CORRECTIVE_REVIEW_REPORT.md')
