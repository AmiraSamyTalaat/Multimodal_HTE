# DATA / LEAKAGE / REALISM AUDIT — PRE-CODE

Decision: **PASS WITH IMPLEMENTATION CHECKS**

## A. Clinical time zero and input realism
PASS.
The confirmatory real-trial inputs are restricted to pretreatment tumor / baseline clinical information. pCR is post-treatment and is never used as an input.

## B. Treatment assignment and causal identification
PASS for the real trial validation design, with trial-specific randomization details to be extracted from the protocol before analysis.
- CHER-LOB is randomized.
- TRIO-US B07 is randomized.
- Primary clinical contrast is the same drug strategy in both trials: dual trastuzumab+lapatinib vs trastuzumab.

This is materially stronger than the original generic observational-cohort design.

## C. True individual effect observability
PASS because claims are constrained.
No real patient has both potential outcomes observed. Therefore:
- PEHE and true-CATE coverage are restricted to the semi-synthetic benchmark;
- real trials are evaluated with randomized-treatment heterogeneity diagnostics, interaction, policy value and selective behavior;
- no manuscript phrase may imply that individual treatment effects are directly measured in the real trials.

## D. Cross-modal leakage
Implementation requirements:
1. all supervised gene filtering must occur inside training folds;
2. pathway definitions may be fixed external biological knowledge;
3. no pCR/treatment-derived feature selection before the split;
4. copy-number/expression preprocessing parameters learned from a fold must not use validation/test subjects;
5. no external trial outcome may influence normalization, architecture, pathway list, threshold, or hyperparameters.

## E. Cross-trial leakage
PASS by design if the external lock is respected.
TRIO-US B07 is invisible to model selection. Only an outcome-blind mapping to the already frozen common feature schema is allowed.

## F. Platform harmonization
Risk: CHER-LOB and TRIO use different expression platforms.
Required fix already prespecified:
- convert expression to fixed pathway-level / rank-robust representations within each study;
- freeze pathway definitions before viewing external outcomes;
- use only the common clinical feature intersection;
- no joint outcome-informed ComBat across discovery and external data.

## G. Missing modality realism
The router may receive a modality-availability flag at inference. It may not receive the external cohort identity as a shortcut for treatment-effect prediction.
If copy-number is unavailable externally, the model must route around it using the same prespecified missing-modality mechanism used during development.

## H. Trial-arm positivity / overlap
Because the real validation cohorts are randomized, positivity is primarily a design property. However, finite-sample support in molecular subgroups can still be weak.
Required reporting:
- arm counts overall and by locked baseline strata;
- support-density diagnostics;
- treatment-propensity diagnostics if the protocol uses unequal/stratified allocation;
- support-eligible retained fraction under abstention.

## I. Semi-synthetic benchmark integrity
The DGP must be generated before model-specific optimization and versioned. The primary DGP is fixed in EXPERIMENT_CONTRACT.yaml. Null and stress regimes cannot replace the primary regime after results are seen.

## J. Small-n deep-learning risk
CHER-LOB and TRIO are too small to justify a large unconstrained backbone. Therefore:
- the novel contribution is the causal information flow / routing algorithm;
- encoders are deliberately compact;
- transcriptomics are represented at pathway level;
- capacity-matched shallow/compact baselines are mandatory;
- the real-trial results are external mechanistic validation, not a claim of training a huge clinical foundation model.

## K. Data-source accessibility
Primary locked plan requires only public resources.
CALGB 40601 is optional until public subject-level treatment/outcome linkage is programmatically verified. No controlled-access data are required to complete the primary contract.

## Final pre-code verdict
**PASS WITH IMPLEMENTATION CHECKS.**
There is no known fatal leakage or deployment-realism defect in the locked design. The main implementation risks are cross-study preprocessing leakage, small-n overcapacity, and misuse of uncertainty language; all three have explicit preventive rules.
