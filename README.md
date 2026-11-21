# Multimodal HTE Benchmarking – reproducibility code and aggregate results

This repository reproduces the computational workflows and aggregate results reported in the manuscript on semi-synthetic outcomes generated from public I-SPY2 pretreatment covariates. It contains original locked-stage code, post-review analyses, corrective analyses, Kaggle independent rerun source, stage configurations, dated experiment contract, manifests, aggregate results and numerical comparison reports.

## Layout
- `original/`: archived original benchmark code, stage configurations, experiment contract and aggregate stage results.
- `postreview/`: baseline, nuisance, calibration, imputation and 30-DGP classical replication analyses.
- `corrective/`: fixed-truth resplits and 12-DGP neural/control investigations.
- `independent_rerun/SOURCE_CODE/`: Kaggle retraining and decision/magnitude comparison scripts.
- `independent_rerun/STAGE_AGGREGATES/`, `STAGE4_AGGREGATES/`, `STAGE_COMPARISONS/`: regenerated stage results, categorical decisions and numeric differences.
- `independent_rerun/FAIR_NEURAL_SUMMARY.csv`, `FAIR_NEURAL_TRAINING_BUDGET.csv`: budget-targeted neural results and their run-level resource measurements.

## Source data and execution
The source material is GEO GSE194040 (transcriptomic data) and GSE196093 (RPPA), within GSE196096. The processed linked patient matrix and individual predictions are not redistributed. Download the public source records and run original preparation steps to create the patient-disjoint analysis matrix before model retraining. Individual reruns also require the same dependency versions and execution environment noted in the environment and configuration files. Reproducing some neural numbers on a different CPU/GPU/software platform may not meet strict numerical tolerance.

The original locked primary results and updated experiments are two separate evidence tiers. The primary grid uses five DGP seeds and three network seeds; additional classical replication uses 30 DGP seeds per regime, and the budget-targeted neural experiment has 12 DGP seeds per regime and a fixed TEST group. Do not treat multiple network seeds or repeated resplits as independently sampled patient cohorts.

## Independent retraining result
Frozen grids completed: Stage 4 15/15, Stage 6A 10/10, Stage 6B 15/15, Stage 6C 10/10, Stage 7B 12/12. Every **categorical stage decision** agreed with the archived original decision. Numeric tolerance-level agreement did **not** hold for all regenerated statistics (RTOL=0.001, ATOL=0.000001); inspect `independent_rerun/STAGE_DECISION_SUMMARY.csv` and `independent_rerun/INDEPENDENT_RERUN_COMPARISON.json`. An independent reconstruction of 675 archived prediction-derived values (without retraining) was numerically concordant, which is a distinct result.

## Integrity
`MANIFEST_SHA256.csv` contains hashes of every accompanying file except itself. Aggregate tables and stage results are provided for source-verifiable inspection; the archive does not claim to contain patient-level raw data or saved trained checkpoints.

## Integrity checks
From the extracted repository root:

```bash
python VERIFY_MANIFEST.py
python VERIFY_RELEASE.py
python original/REPRODUCE/VERIFY_PACKAGE.py
```

These commands check archived file integrity, Python source syntax and presence of original outputs; they are not substitutes for full model retraining. The release includes recorded independent retraining comparisons, which preserve the distinction between matching categorical stage decisions and numeric tolerance differences.

## Public repository and archive
Extract these files into a dedicated repository directory for GitHub Desktop. Upload this same release ZIP as the source-code archive to a new Zenodo record (or connect the GitHub repository to Zenodo for a versioned GitHub release). Raw linked patient data and patient-level predictions must not be uploaded.
