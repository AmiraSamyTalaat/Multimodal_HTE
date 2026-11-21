from __future__ import annotations
import pandas as pd
from corrective_common import COR,REV,now

AUDIT=REV.parent/'B01_PAPER1_COWORK_AUDIT_20261118/B01_PAPER1_reviewer_requirement_matrix.csv'
if not AUDIT.exists():
    AUDIT=REV.parent.parent/'B01_PAPER1_COWORK_AUDIT_20261118/B01_PAPER1_reviewer_requirement_matrix.csv'
base=pd.read_csv(AUDIT)[['section','id','requirement']]
old='B01_POST_REVIEW_2026/results/'
new='B01_POST_REVIEW_2026/CORRECTIVE_REVIEW_2026/'
evidence={
 'A':old+'baselines/per_seed_results.csv',
 'B':old+'oracle_fairness/per_seed_results.csv',
 'C':old+'replication/uncertainty_intervals.csv',
 'D':old+'fairness/per_seed_results.csv',
 'E':old+'additional/overlap_true_cate_by_seed.csv',
 'F':old+'additional/utility_gate_rates_by_seed.csv'}
specific={
 'A6':old+'baselines/reference_policies.csv',
 'A7':old+'baselines/per_run_results.csv',
 'A8':old+'baselines/per_seed_results.csv',
 'A9':old+'baselines/per_seed_results.csv',
 'A4':new+'results/clinical_adjustment/per_seed_results.csv',
 'B1':old+'oracle_fairness/DGP101_dgp_diagnostics.json',
 'B2':old+'oracle_fairness/DGP101_dgp_diagnostics.json',
 'B3':old+'oracle_fairness/per_run_results.csv',
 'B4':old+'oracle_fairness/pseudooutcome_distributions.csv',
 'B5':old+'oracle_fairness/ridge_sensitivity.csv',
 'B6':old+'oracle_fairness/ridge_sensitivity.csv',
 'B7':old+'oracle_fairness/propensity_gof.csv',
 'B8':old+'oracle_fairness/propensity_gof.csv',
 'C1':old+'replication/per_seed_results.csv',
 'C2':old+'replication/per_seed_results.csv',
 'C3':new+'results/fixed_dgp_splits/fixed_truth_validation.csv',
 'C4':old+'replication/paired_dgp_comparisons.csv',
 'C5':new+'reporting/verified_dgp_cluster_intervals.csv',
 'C6':old+'replication/uncertainty_intervals.csv',
 'C7':new+'reporting/cluster_uncertainty_extension.csv',
 'C8':old+'replication/split_sensitivity.csv',
 'D1':new+'results/fair_neural/per_seed_results.csv',
 'D2':new+'results/fair_neural/training_budget.csv',
 'D3':new+'results/fair_neural/training_budget.csv',
 'D4':new+'results/nonoracle_tuning/per_seed_results.csv',
 'D5':old+'classical_sensitivity/dev_grid.csv',
 'D6':new+'results/fair_neural/training_budget.csv',
 'D7':new+'results/nonoracle_tuning/per_seed_results.csv',
 'E3':old+'imputation/imputation_feature_reconstruction.csv',
 'E1':old+'additional/missingness_realized_postreview.csv',
 'E2':old+'additional/availability_masks_postreview.csv',
 'E4':old+'additional/overlap_true_cate_by_seed.csv',
 'E5':old+'additional/overlap_true_cate_by_seed.csv',
 'E6':old+'additional/overlap_true_cate_by_seed.csv',
 'E7':old+'additional/overlap_true_cate_by_seed.csv',
 'E8':old+'oracle_fairness/propensity_gof.csv',
 'E9':old+'replication/per_seed_results.csv',
 'E10':new+'results/fair_neural/per_seed_results.csv',
 'F1':new+'results/fair_neural/paired_dgp_comparisons.csv',
 'F4':old+'additional/utility_gate_rates_by_seed.csv',
 'F6':new+'results/fair_neural/per_seed_results.csv',
 'F7':'B01_POST_REVIEW_2026/postreview_code/validate_revision.py',
 'F8':'B01_POST_REVIEW_2026/postreview_code/validate_revision.py'}
needs_experiment={'C8':'Hierarchical variance with repeated learner seeds and patient cohorts has not been estimated.',
 'D5':'Broader neural architecture and optimizer sensitivity would be needed for a general tuning claim.',
 'D6':'A genuinely equal total optimizer-step budget remains needed to isolate architecture/training cost.',
 'F5':'Direct oracle incremental-value analysis across the new fixed DGP mechanisms remains to be run for this specific claim.',
 'F7':'Numerical rerun tolerance was specified in old code, but an independent tolerance-checked rerun is not supplied.',
 'F8':'A frozen-decision rerun audit is still needed if claiming exact decision reproducibility.'}
limitations={'A4':'Explain clinical-only X/T molecular confounding and full-nuisance R diagnostic; no robust superiority over zero.',
 'A2':'Call the TEST-CATE mean the best constant squared-error reference, not a universal lower bound.',
 'C1':'Thirty Monte Carlo DGP seeds are conditional on one patient covariate cohort, not thirty patient cohorts.',
 'C2':'Original multiple-split replication changed CATE; use new fixed-DGP experiment for split sensitivity.',
 'C3':'Distinguish fixed-DGP split variation from the old combined mechanism/preprocessing variation.',
 'C5':'DGP seed is the cluster; three resplits and 270 total rows are correlated.',
 'C7':'Policy-regret and calibration intervals are descriptive; calibration has five DGP seeds.',
 'D2':'Same train-MSE stopping rule was used; no DEV-early-stopped neural refit was performed.',
 'D4':'The documented grids are limited; cross-family search budgets differ.',
 'D6':'Matched deployed parameters do not imply matched optimizer steps; report 24 CM expert fits plus router versus five bag fits.',
 'E10':'Full neural comparison covers two effect regimes, twelve DGP seeds, one split and one learner seed, not all old replication regimes.',
 'F1':'Only CM versus uniform on identical experts isolates learned routing within that architecture; bagging/regularization remain alternative explanations across architectures.',
 'F3':'Preserve failed selectivity and gate cases.',
 'F6':'Preserve negative results including zero-shrinkage DEV selections.'}
completed={'A','B','E'}
rows=[]
for r in base.itertuples():
    status='evidence_complete_writing_only'
    if r.id in needs_experiment:status='additional_experiment_needed'
    if r.id in ['C3','D1','D2','D3','E10','F1']:
        path=REV.parent/specific.get(r.id,'')
        if not path.exists():status='experiment_in_progress'
        elif r.id in ['C3']:
            z=COR/'results/fixed_dgp_splits/RUN_STATUS.json'
            if z.exists() and '"PARTIAL"' in z.read_text():status='experiment_in_progress'
        elif r.id in ['D1','D2','D3','E10','F1']:
            z=COR/'results/fair_neural/RUN_STATUS.json'
            if z.exists() and '"PARTIAL"' in z.read_text():status='experiment_in_progress'
    if r.id in ['D4']:
        status='limited_experiment_complete_writing_needed'
    rows.append(dict(section=r.section,id=r.id,requirement=r.requirement,
       corrective_status=status,evidence=specific.get(r.id,evidence[r.section]),
       manuscript_action=limitations.get(r.id,'Report as post-review evidence with scope and negative findings.'),
       actual_additional_experiment=needs_experiment.get(r.id,''),reviewed_utc=now()))
pd.DataFrame(rows).to_csv(COR/'reviewer_objection_matrix.csv',index=False)
print(len(rows),'reviewer requirements mapped')
