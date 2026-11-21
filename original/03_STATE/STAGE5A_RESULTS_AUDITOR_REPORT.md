# Stage 5A Results Auditor + Statistician Report

## Integrity
- Locked Stage 5A grid: **15/15 unique DGP×network-seed pairs**.
- All 15 run manifests are COMPLETE: **True**.
- Scientific settings changed: **False**.
- Primary grid rerun: **False**.
- All completed runs carry `MASKED_TRUE_TE_V1`: **True**.
- Missingness masks are identical across network seeds within each DGP/condition/split: **True**.

## Main statistical interpretation
H1 remains PASS from the frozen primary gate. Stage 5A does not alter that primary decision.

H2 is **MIXED / PARTIAL SUPPORT — NOT YET PASSED AS A WHOLE**.

### Evidence supporting the causal-support routing story
- Under **severe weak overlap**, CM-OSR minus capacity-matched early fusion root-PEHE = **-1.835** (95% cluster bootstrap CI **-2.050 to -1.633**).
- Under severe weak overlap, CM-OSR minus generic multimodal attention = **-2.162** (95% CI **-2.332 to -1.987**).
- Under severe weak overlap, CM-OSR minus generic MoE = **-2.490** (95% CI **-2.686 to -2.294**).
- The same direction is present across clean reference, MAR, MCAR10/25/40, and null-control direct comparisons versus the capacity-matched/generic multimodal architecture controls.
- Availability-masked generic attention/MoE controls remain much worse than CM-OSR in the missing-modality conditions.

### Evidence that limits H2
The prespecified phrase **“degrade more gracefully than generic attention/MoE under missing modality” is not supported by the clean-to-missingness degradation analysis**.
For every MCAR/MAR condition, CM-OSR-minus-generic degradation is positive with the 95% CI above zero. In this experiment, missingness improved the generic deep architecture controls even more than it improved CM-OSR. Therefore the manuscript must distinguish:
1. **better absolute stressed performance** — supported versus generic multimodal controls; from
2. **smaller clean-to-stress degradation** — not supported versus generic attention/MoE.

### Strong conventional controls
X-learner and regularized T-learner remain substantially better point estimators than CM-OSR:
- Severe overlap root-PEHE: CM-OSR **0.768**, X-learner **0.136**, T-learner **0.157**.
- MCAR40 root-PEHE: CM-OSR **0.743**, X-learner **0.210**, T-learner **0.262**.
- Clean reference root-PEHE: CM-OSR **0.867**, X-learner **0.188**, T-learner **0.202**.

This prohibits any overall-best/SOTA claim.

## Selective behavior
At retained fraction 50% versus 100%, CM-OSR root-PEHE improved in every evaluated family at the DGP-cluster level:
- Clean: mean difference **-0.241** (95% CI **-0.403 to -0.126**).
- MAR: **-0.195** (95% CI **-0.369 to -0.066**).
- MCAR10: **-0.251** (95% CI **-0.424 to -0.122**).
- MCAR25: **-0.185** (95% CI **-0.313 to -0.090**).
- MCAR40: **-0.159** (95% CI **-0.301 to -0.052**).
- Severe overlap: **-0.231** (95% CI **-0.286 to -0.177**).
- Null effect: **-0.309** (95% CI **-0.455 to -0.197**).

This supports the support-aware abstention/selective-risk component, but does not repair the null-effect point-estimation concern.

## 90% benchmark-only coverage
Mean coverage ranges from approximately **0.904 to 0.913** across all Stage 5A conditions, close to the nominal 0.90 target. This is valid only for the semi-synthetic benchmark where true tau is known and must not be described as a real-patient ITE confidence interval.

## Null-effect negative control
This is the main Stage 5A concern.
- CM-OSR root-PEHE: **1.077**; mean |tau_hat|: **0.738**.
- X-learner root-PEHE: **0.125**; mean |tau_hat|: **0.101**.
- T-learner root-PEHE: **0.149**; mean |tau_hat|: **0.119**.
- R-learner root-PEHE: **0.694**; mean |tau_hat|: **0.556**.

CM-OSR still beats the generic deep architecture controls in the null DGP, but it does **not** pass a strong “no spurious heterogeneity” interpretation relative to X/T learners.

## Decision
**PROCEED_TO_STAGE5B_PRESPECIFIED_MECHANISM_ABLATIONS_THEN_CHER_LOB_AND_LOCKED_TRIO_VALIDATION**

Strong-Q1 readiness is **not reached yet**. The next scientific work should complete the remaining prespecified mechanism tests (especially A4 remove orthogonal target, A5 remove overlap, A6 remove reliability, A8 modality-only/no-joint, random-router negative control; A9 is already informed by the Stage 5A selective/full-retention analysis), then proceed to CHER-LOB nested cross-fitting and frozen TRIO-US B07 external validation.

No seed search, primary rerun, stress-rate change, post-hoc threshold tuning, or overall-SOTA claim is justified.
