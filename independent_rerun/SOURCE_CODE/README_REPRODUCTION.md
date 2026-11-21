# Kaggle execution and resumption

This workspace is for **the Scientific Reports submission**, submission `417743dc-6d78-4ff8-9321-a80ca41e855e`. It does not modify the original project, Paper 2, manuscript, or response letter. New analyses are post-review, not prespecified.

## Private inputs and environment

- Kaggle account: `prinprin` (existing authenticated Kaggle CLI; do not place credentials in scripts or archives).
- Private source dataset: `prinprin/b01-causaltx-mm-core-runtime`. Its processed I-SPY2 NPZ has SHA-256 `e94166a00268aab5cb99cb74866ed81771a96ba88c9dc4368107bd0795081e1a`.
- Private code dataset: `prinprin/b01-p1-independent-rerun-code`. The underlying original code archive is `B01_REVIEW_CODE_ARCHIVE.bin`; despite the extension it is a ZIP of original benchmark code/configuration, contracts and archived aggregate results. The patched runner/comparator are separate files.
- Python requirements for original code are inside `B01_Paper1_Complete_Reviewer_Code_Updated.zip/original/requirements.txt`. Each new Kaggle batch manifest records Python/platform and, in later code versions, core library versions. Code and input SHA-256 values are recorded per batch.

All new kernels and datasets are private. The original patient matrix remains in the private runtime dataset. Private Stage 7B checkpoints must temporarily retain `patient_dev_predictions.csv` because the original utility-gate aggregator reads it. This file must be excluded from any distributable ZIP. No Zenodo publication is authorized.

The Kaggle CLI uses the account's existing local OAuth configuration. After a long break, the CLI may need write access to its existing `~/.kaggle/credentials.json` to refresh access; do not paste credentials into a chat, script or ZIP. Once authentication succeeds, the runner discovers completed private kernels and resumes from their saved outputs.

## Restart Stage 4

From this directory on Windows, run `RUN_RESUME_PAPER1.cmd --backend cpu` to continue the CPU branch from completed Kaggle outputs. Use `RUN_RESUME_PAPER1.cmd --backend gpu` for the separate GPU branch when Kaggle GPU capacity is available. The program reads current Kaggle kernel status, downloads and checks completed checkpoint hashes, skips them, retries failed batch kernels up to two times with saved logs, and launches only missing batches. It executes the frozen 15 Stage 4 pairs in chunks of 1, 2, 4, 4 and 4 jobs, followed by aggregation. It does not use `--stages all`.

The first CPU smoke had a scientific numeric comparison FAIL; its metrics-only checkpoint was copied into the private Dataset `prinprin/b01-p1-stage4-smoke-checkpoint`. A new CPU session restored that checkpoint, logged `SKIP stage4_DGP101_S17`, and trained the next two pairs. The analogous GPU smoke checkpoint is in private Dataset `prinprin/b01-p1-stage4-gpu-smoke-checkpoint`. Subsequent completed kernels serve as durable inputs to the next kernel. A scientific comparison FAIL does not erase the checkpoint or masquerade as a technical training error.

## Restart selected additional stages

Use `RUN_RESUME_PAPER1.cmd --stage stage6a --backend cpu`, similarly for `stage6b`, `stage6c`, and `stage7b`. Each starts from its completed one-pair smoke, waits for Kaggle quota when needed, and runs the original frozen grid in three-job chunks before aggregation. Stage 6B can use the archived Stage 6A selection only as a frozen pre-TEST dependency, with a provenance marker; this is explicitly distinct from independently regenerating the Stage 6A grid. The stage chains are separate, so a failure or changed selection is reported rather than silently repaired.

## Fair neural comparison

`collect_fair_neural.py` downloads and validates seven completed private Kaggle kernels covering the locked 12 DGP seeds and two effect mechanisms. It produces `FAIR_NEURAL_COMPARISON.csv`, `FAIR_NEURAL_SUMMARY.csv`, `FAIR_NEURAL_PAIRED_DGP.csv`, `FAIR_NEURAL_TRAINING_BUDGET.csv`, and `FAIR_NEURAL_VALIDATION.json`. Paired uncertainty uses the DGP seed as the independent unit within each mechanism. The one-seed 25-fit resource pilot is preserved separately and was not used as the final training-budget comparison.

## Final artifact

After a full Stage 4 aggregate is downloaded and validated, run `py -3.14 package_actual_results.py --backend cpu` or `--backend gpu`. It creates a ZIP with the specified comparison JSON, fair neural CSV, manifests, experiment CSV, logs, validation report, findings and code. The packager rejects missing Stage 4 seed pairs and patient-level NPZ/prediction files. The Stage 4 decision must be read from the new aggregate, not inferred from an archived reaggregation or one-pair smoke.

For each completed aggregate, run `py -3.14 revalidate_saved_aggregate.py --stage stage4 --backend cpu` (replace the stage as needed) before packaging. This uses the corrected seed-key comparator on the saved checkpoint without model retraining and writes comparator and checkpoint SHA-256 provenance. The earlier Kaggle comparator reports are retained beside the corrected reports so its original row-alignment error remains auditable.
