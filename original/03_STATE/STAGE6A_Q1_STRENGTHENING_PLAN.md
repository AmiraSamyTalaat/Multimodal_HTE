# STAGE 6A — Q1 STRENGTHENING METHOD DEVELOPMENT

## Why this stage exists

The goal is not to hide or overwrite the Stage 5A limitation. The goal is to test,
in a scientifically defensible new development phase, whether a stronger architecture
can retain CM-OSR's multimodal/support advantages while inheriting the strong null and
point-estimation behavior of conventional causal learners.

## New method family: AOSR

**Anchored Orthogonal Support Routing (AOSR)**

1. Fit the full multimodal nuisance path and compute cross-fitted DR pseudo-outcome phi.
2. Fit a strong leakage-safe cross-fitted causal anchor (X-learner or R-learner).
3. Define the orthogonal residual target:
   residual_phi = phi - anchor_tau_OOF
4. Train modality-resolved residual experts on residual_phi.
5. Route the residual experts using overlap, availability, density, OOF reliability,
   and disagreement.
6. Apply conservative support shrinkage:
   correction = support^gamma * routed_residual
7. Final effect:
   tau_hat = anchor_tau + correction

Interpretation: the strong conventional learner handles the stable effect surface;
multimodal routing is only allowed to add patient-specific residual evidence when
support is credible.

## Why this directly attacks the Stage 5A problem

If the true effect is null, a good anchor should stay near zero and the routed residual
is damped by support. This should reduce spurious heterogeneity.

If there is genuine cross-modal effect modification, the residual router can add value
beyond the anchor rather than forcing the whole treatment-effect surface to be learned
by a high-variance multimodal router.

## Candidate set

- Frozen CM-OSR v1
- Frozen X, T, R learners
- AOSR-X gamma=1
- AOSR-X gamma=2
- AOSR-R gamma=1
- AOSR-R gamma=2
- Support-shrunk CM-OSR gamma=1
- Support-shrunk CM-OSR gamma=2

No candidate may be added after Stage 6A results are viewed.

## Selection is DEV-only and lexicographic

A candidate must first satisfy the null guard, then point-estimation competitiveness,
then improve frozen CM-OSR v1, and must use a nonzero routed residual.

No weighted score is used because a weighted score could trade away a serious null
failure in exchange for gains elsewhere.

## Stage 6B is already isolated now

Fresh confirmatory DGP seeds are locked now and cannot be run during Stage 6A.
After one candidate is frozen, Stage 6B evaluates it once on TEST with the original
strong baselines and the full stress suite.

## Strong-Q1 gate after Stage 6B

Proceed to the real trials only with a strong method claim if:
- the selected candidate passes the Stage 6A null guard;
- confirmatory Stage 6B does not show catastrophic null failure;
- it remains competitive with X/T on point estimation;
- it preserves a resolved advantage over fair multimodal fusion controls;
- support-based selection remains beneficial.

Then:
CHER-LOB nested cross-fitting -> freeze score -> TRIO-US B07 locked external validation.

No publication tier can be guaranteed, but this design removes the largest current
reviewer vulnerabilities without post-hoc manipulation of the original test.
