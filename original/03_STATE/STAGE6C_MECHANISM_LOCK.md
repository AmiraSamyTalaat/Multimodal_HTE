# Stage6C Mechanism Generalization Lock

Stage6C is a secondary mechanistic validation of the frozen AOSR_X_gamma2.
It is not a model-selection round and cannot overwrite Stage6B.

The three regimes used here were already prespecified in the original EXPERIMENT_CONTRACT
before held-out results:
- shared_effect_modifier
- transcriptome_local_effect_modifier
- RPPA_local_effect_modifier

Stage6C trains on TRAIN and evaluates DEV only. TEST is forbidden.

The novelty being tested is not a generic reliability-aware MoE. The core mechanism is:
strong causal anchor -> orthogonal residual treatment-effect target ->
modality-resolved residual experts -> routed residual correction.

The Stage6B selective-retention claim remains failed and is not resurrected.
