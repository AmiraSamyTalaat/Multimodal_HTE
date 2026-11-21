# Stage 3 — Core scientific benchmark execution

Precondition: DATA/LINKAGE AUDIT PASS + PRETRAIN MANIFEST REVIEW PASS + CORE DATA PREPARATION PASS.

## Mandatory fairness correction before first scientific training
The original implementation had capacity-matched early fusion, but generic attention and generic MoE were materially smaller than CM-OSR. Stage 3 fixes this **before any scientific model result exists** by choosing their width from parameter counts only. No outcome, TEST, or external label is used for this correction.

With the locked 3 / 702 / 139 dimensions, the expected CM-OSR neural parameter count is ~126k. The exact matching widths are computed dynamically and must be within the locked ±10% tolerance.

Neural methods use the same locked maximum epoch budget. CM-OSR cross-fitted experts retain cross-fitting because it is part of the method; the additional computation is disclosed rather than hidden.

## Kaggle execution
Account: `prinprin`; browser: Avast.

The 15 locked core replicates are split into five resumable Kaggle sessions, one DGP seed per session and three network seeds per session:
- DGP 101 × {17,29,43}
- DGP 211 × {17,29,43}
- DGP 307 × {17,29,43}
- DGP 401 × {17,29,43}
- DGP 503 × {17,29,43}

Each successful session is harvested back to local `04_RUNS`. Re-running the launcher skips complete seed pairs. No seed search is allowed.

The private Kaggle runtime dataset is only a transfer bundle. The core input is the already-audited processed I-SPY2 covariate matrix; no external labels are used here.

## Primary gate
After 15/15 runs, `aggregate_core_results.py` computes paired CM-OSR minus capacity-matched early-fusion root-PEHE. The three network seeds are averaged within each DGP seed, then a paired cluster bootstrap is run over the five DGP seeds. PASS requires a negative mean difference and a 95% CI entirely below zero, exactly as locked before results.
