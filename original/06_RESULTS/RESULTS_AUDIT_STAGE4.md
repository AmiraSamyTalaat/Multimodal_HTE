# B01 Stage 4 — Results Auditor + Statistician

## Scope
This audit uses the completed locked primary semi-synthetic grid:
- 5 prespecified DGP seeds
- 3 network seeds per DGP seed
- 15/15 completed locked seed pairs
- no seed search and no post-result change to the primary endpoint or method.

## Confirmatory primary gate
**PASS.**

Locked comparison:
`CM_OSR - early_fusion_DR_capacity_matched` on root-PEHE.

- Mean paired difference: **-1.967963**
- 95% paired DGP-cluster bootstrap CI: **[-2.199146, -1.691623]**
- DGP clusters: **5/5**
- Seed pairs: **15/15**
- Bootstrap replicates: **10000**
- Decision: **PASS**

CM-OSR mean root-PEHE = **0.8671** versus capacity-matched early-fusion DR = **2.8350**,
a descriptive relative reduction of **69.4%**.

All 15 paired runs favor CM-OSR over the capacity-matched early-fusion DR baseline.

## Mechanism / ablation evidence
CM-OSR also has lower root-PEHE than each locked architecture control in all 15 paired runs:
- joint-only CM-OSR ablation
- uniform-all-experts ablation
- modality-only uniform-fusion ablation
- capacity-matched generic multimodal attention DR
- capacity-matched generic MoE DR

Exploratory DGP-cluster bootstrap intervals for these non-primary comparisons are stored in
`CM_OSR_PAIRWISE_STAGE4.csv`. They are **not promoted to confirmatory claims** here.

## Important benchmark limitation
The primary gate passed, but CM-OSR is **not the best method in the full benchmark**.

Across the 15 runs:
- X-learner mean root-PEHE = **0.1881**
- regularized T-learner = **0.2024**
- CM-OSR = **0.8671**
- R-learner = **0.8677**

CM-OSR loses to X-learner and T-learner in all 15 paired runs. It is essentially tied with
R-learner on mean root-PEHE, and several deep causal representation baselines have lower mean
root-PEHE than CM-OSR in this primary regime.

Therefore the paper must **not** claim:
- overall state-of-the-art performance;
- universal superiority over causal learners;
- that CM-OSR is the best estimator on the primary benchmark.

The defensible primary claim is narrower:
**the locked post-orthogonal cross-modal routing design materially outperforms the fair
capacity-matched early-fusion and generic multimodal fusion controls, with ablations supporting
the contribution of routing/fusion design.**

## Full descriptive benchmark

| Method | Mean root-PEHE | Mean policy regret | Mean CATE calibration error |
|---|---:|---:|---:|
| X_learner | 0.1881 | 0.0307 | 0.1109 |
| T_learner_regularized | 0.2024 | 0.0312 | 0.1249 |
| DragonNet | 0.5906 | 0.0486 | 0.4402 |
| TARNet | 0.5928 | 0.0493 | 0.4428 |
| Transformer_HTE | 0.5964 | 0.0483 | 0.4476 |
| CFRNet | 0.6068 | 0.0494 | 0.4541 |
| CM_OSR | 0.8671 | 0.0527 | 0.6438 |
| R_learner | 0.8677 | 0.0402 | 0.6931 |
| modality_only_uniform_fusion_ablation | 1.5659 | 0.0551 | 1.1950 |
| uniform_all_experts_ablation | 1.6632 | 0.0547 | 1.2665 |
| joint_only_CM_OSR_ablation | 2.7607 | 0.0576 | 2.0335 |
| early_fusion_DR_capacity_matched | 2.8350 | 0.0589 | 2.0709 |
| generic_multimodal_attention_DR_capacity_matched | 3.2813 | 0.0605 | 2.3013 |
| generic_MoE_DR_capacity_matched | 3.7670 | 0.0614 | 2.6831 |
| DR_learner | 5.5528 | 0.0592 | 4.2622 |

## Statistician interpretation
The confirmatory primary hypothesis H1 is supported under its prespecified comparator and
decision rule.

However, the full benchmark creates a major interpretation flag because simple T/X learners
perform substantially better on the primary DGP. This is not a reason to alter the primary DGP,
seeds, method, or endpoint after seeing results. It is a reason to:
1. preserve and report the result transparently;
2. avoid broad superiority claims;
3. execute the already-prespecified secondary stress tests designed around H2;
4. execute the already-prespecified randomized real-trial validation designed around H3.

## Stage decision
**PROCEED TO PRESPECIFIED SECONDARY + REAL-TRIAL VALIDATION, WITH CLAIM LIMITATION.**

Do not rewrite CM-OSR or search for favorable seeds.

Before manuscript drafting, the following locked items remain:
- severe weak-overlap stress;
- missing-modality MCAR and MAR stress;
- selective root-PEHE versus retained fraction;
- overlap-stratified root-PEHE;
- known-tau uncertainty/coverage where implemented;
- null-treatment-effect negative control;
- CHER-LOB development using nested cross-fitting;
- frozen transfer to TRIO-US B07 with no external tuning;
- continuous treatment-by-benefit-score interaction and randomization-weighted policy value;
- Reviewer-from-Hell red-team audit;
- final Q1 gate.

## Q1 risk status
The core result is scientifically usable and the primary gate passed, but the paper is **not yet
Q1-ready** because H2/H3 and the late generalization/external validation gates remain open, and
because the strong T/X-learner results require explicit explanation and claim discipline.
