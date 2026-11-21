# STAGE 5A EXECUTION PLAN

Prerequisites:
- Stage 4 primary gate PASS.
- 15/15 primary runs preserved.
- `06_RESULTS/PRIMARY_GATE.json` and `06_RESULTS/ALL_CORE_METRICS.csv` present.
- Kaggle account remains `prinprin` in Avast.

Execution:
- five Kaggle sessions, one per DGP seed (101, 211, 307, 401, 503);
- each session runs network seeds 17, 29, 43;
- each seed-pair evaluates the locked severe-overlap, missing-modality, and null-effect families;
- all Stage 5A outputs are written under `05_STRESS_RUNS`;
- the primary `04_RUNS` directory is read-only from the Stage 5A point of view.

After 15/15 Stage 5A pairs are harvested, the local aggregator writes:
`06_RESULTS/STAGE5A/ALL_STAGE5A_METRICS.csv`
`06_RESULTS/STAGE5A/H2_PAIRWISE_CLUSTER_BOOTSTRAP.csv`
`06_RESULTS/STAGE5A/SELECTIVE_SUMMARY.csv`
`06_RESULTS/STAGE5A/COVERAGE_SUMMARY.csv`
`06_RESULTS/STAGE5A/OVERLAP_STRATA_SUMMARY.csv`
`06_RESULTS/STAGE5A/STAGE5A_AUDIT.json`

No manuscript claim is finalized automatically. Results Auditor + Statistician must inspect
the full secondary evidence, especially because X/T learners were stronger than CM-OSR on the
primary regime.
