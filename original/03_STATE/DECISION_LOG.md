# DECISION LOG

| Date | Stage | Decision | Why | Before/After held-out? | Human approval required? |
|---|---|---|---|---|---|
| 2026-09-18 | INTAKE | Treat uploaded B01 package as the active research idea | It contains a specific project card and detailed research concept | Before | No |
| 2026-09-18 | NOVELTY_AUDIT | Reject the original generic architecture as sufficient novelty | 2026 literature has direct collisions in multimodal ITE, multimodal DML, weak-overlap orthogonal HTE and uncertainty | Before | No; required by novelty gate |
| 2026-09-18 | METHOD_DESIGN | Redesign around CM-OSR: joint nuisance correction -> orthogonal target -> modality effect experts -> support routing | Creates a new causal information flow instead of stacking standard modules | Before | No material alternative selected yet |
| 2026-09-18 | METHOD_DESIGN | Keep standard encoders/DR/conformal tools explicitly non-novel | Prevents overclaiming | Before | No |
| 2026-09-18 | EXECUTION POLICY | Do not select Kaggle account yet | Account selection is required only before first Kaggle launch | Before | Later: yes |
| 2026-09-18 | LOCAL WORKSPACE POLICY | Do not request/reuse a Windows path yet | Required immediately before local implementation; paths from unrelated projects must not be reused | Before | Later: yes |

## 2026-09-18 — Dataset/estimand lock
- Replaced the generic observational-cohort-first plan with a stronger randomized-trial validation path.
- Locked I-SPY2-990 real-covariate semi-synthetic benchmark for known-CATE algorithm testing.
- Locked CHER-LOB as real randomized discovery/clinical validation and TRIO-US B07 as external cross-trial replication for dual trastuzumab+lapatinib vs trastuzumab, outcome pCR.
- CALGB 40601 remains optional pending reproducible public subject-level label linkage.

## 2026-09-18 — Novelty recheck after Sep-2026 collision
- Added the 11 Sep 2026 causal multimodal chemosensitivity preprint as a nearest competitor.
- Prohibited unsupported category-level novelty claims about multimodal causal models for personalized breast-cancer treatment benefit.
- Locked novelty to orthogonal-score-level support/reliability routing and abstention under weak support plus randomized cross-trial validation.

## 2026-09-18 — Uncertainty language lock
- Prohibited real-data claims of directly observed patient-level ITE confidence intervals.
- Allowed known-CATE interval coverage only on semi-synthetic benchmark; real trials use score stability, bootstrap uncertainty and selective policy behavior.


## 2026-09-19 — Data/linkage audit correction before training
- The first linkage audit correctly stopped all scientific training.
- GSE194040 was falsely flagged by the audit because its 988 samples are split across two platform matrices (654 + 334); the corrected gate evaluates the combined union.
- TRIO-US B07 source was corrected from GSE130787 to **GSE130786 (Baseline)**. GSE130787 is the two-week on-treatment/run-in comparison and is prohibited as a pretreatment predictor input.
- The GEO parser now reads sample characteristics from all channels (ch1/ch2/...) so two-colour-array treatment/response metadata are not silently missed.
- This correction occurred before any scientific model training or held-out scoring and does not change the locked estimand, endpoint, method, or success criterion.
