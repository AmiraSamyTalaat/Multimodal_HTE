# Stage7A CUCMR Execution Lock V4 — final pre-run execution amendment

Created before any Stage7 scientific result. This supersedes the unrun V3 draft.

The core method is unchanged: a fixed clinical anchor, modality-specific residual-effect candidates, OOF incremental causal-utility targets, conditional utility prediction, development-frozen selective acceptance, and exact anchor fallback.

The implementation is made computationally tractable by using **multi-stage OOF cross-fitting** rather than repeatedly nesting full X-learners inside outer folds. Direct OOF predictions are still required at the nuisance, clinical-anchor, candidate-update, and utility-score stages.

The clinical anchor uses a fixed 100-tree clinical X-learner. Molecular residual updates use fixed HistGradientBoosting regressors. The existing 300-tree full-multimodal X-learner remains the strong comparator.

Stage7B is TRAIN-fit / DEV-development only. TEST, confirmatory seeds, CHER-LOB, and TRIO are forbidden.
