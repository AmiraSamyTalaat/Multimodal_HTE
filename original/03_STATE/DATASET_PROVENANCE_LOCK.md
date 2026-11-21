# DATASET + ESTIMAND PROVENANCE LOCK — B01_CausalTx_Multimodal

Status: **LOCKED FOR PRE-CODE DESIGN; DATA BYTES NOT YET DOWNLOADED**
Date: 2026-09-18

## 1. Why the original generic observational-cohort plan was changed
The project card allowed public clinical/omics/imaging cohorts. The fresh audit found that a stronger design is available using randomized neoadjuvant breast-cancer trials. This reduces the central causal-identification weakness that would otherwise dominate review.

The paper is therefore framed as a **methodological causal-ML paper**, not as an observational clinical recommendation paper.

## 2. Dataset roles

### A. Core algorithmic benchmark — I-SPY2-990 real-covariate semi-synthetic benchmark
Source: GEO GSE194040 (mRNA + clinical) and GSE196093 (RPPA), I-SPY2.
Publicly described content:
- 987 patients with pretreatment transcriptome and clinical labels across 10 treatment arms;
- 736 with pretreatment RPPA as a second molecular modality;
- pCR and treatment-arm metadata are public.

Use in this paper:
- **Real pretreatment multimodal covariates only** are used as the substrate.
- Treatment and outcomes for the primary algorithmic benchmark are generated from a prespecified semi-synthetic DGP, so the true conditional treatment effect is known.
- This benchmark is the only place where PEHE / true-CATE coverage may be claimed.
- Actual I-SPY2 clinical treatment effects are not the confirmatory clinical claim of the paper.

Core modalities:
1. Clinical / biomarker covariates.
2. Transcriptomic pathway representation.
3. RPPA/proteomic representation (complete-case benchmark plus prespecified missing-modality stress tests).

### B. Real randomized discovery / clinical validation — CHER-LOB
Source: GEO GSE66399 / GSE66305 and related public sample metadata.
Trial design:
- randomized phase II HER2-positive neoadjuvant trial;
- treatment arms: chemotherapy + trastuzumab (A), chemotherapy + lapatinib (B), chemotherapy + trastuzumab+lapatinib (C);
- pCR is recorded in public GEO sample characteristics;
- baseline expression data and copy-number data are deposited in GEO.

Primary real-data contrast:
- **Arm C (dual HER2 blockade) vs Arm A (trastuzumab)**.

Secondary real-data contrast:
- Arm B (lapatinib) vs Arm A.

Outcome:
- pathologic complete response (pCR), binary.

### C. External cross-trial replication — TRIO-US B07
Source: GEO GSE130788 / **GSE130786 (Baseline)** and the published trial.

**Pre-training correction from the linkage audit:** GSE130787 is the two-week on-treatment/run-in series. It must **not** be used as the pretreatment predictor source for the locked external test. GSE130786 contains the 110 baseline pretreatment profiles and is therefore the allowed external feature source. GSE130787 is retained only as a provenance reference and is prohibited from the primary pretreatment model input.
Trial design:
- multicenter randomized phase II HER2-positive neoadjuvant trial;
- treatment arms: trastuzumab, lapatinib, or combination;
- pretreatment expression, treatment arm and pCR are public;
- same clinically meaningful treatment contrasts as CHER-LOB.

Role:
- **locked external trial**;
- no hyperparameter tuning, feature selection, threshold tuning or architecture changes on this cohort;
- primary external transfer is Arm combination vs trastuzumab.

Common external-transfer modalities:
- pretreatment transcriptomic pathway representation;
- baseline clinical variables restricted to the prespecified common intersection available in both trials.

Any CHER-LOB-only copy-number modality is treated as an optional discovery modality and a missing-modality stress test at external transfer; it must not be required for the external claim.

### D. Optional third randomized replication — CALGB 40601
Source: GEO GSE116335 + published supplementary molecular files.
This trial has the same central trastuzumab vs dual trastuzumab+lapatinib comparison and pCR endpoint, but the public patient-level linkage between all required labels and omics must be programmatically verified before use.

Rule:
- CALGB 40601 is **not required for the primary paper to pass**.
- It may be added only if public, reproducible subject-level treatment/pCR linkage is verified without controlled-access clinical data.
- No weakening of the locked CHER-LOB -> TRIO external design if CALGB is unavailable.

## 3. Locked causal estimand
For the primary real-data contrast, among eligible HER2-positive patients represented by the trial data:

τ(x) = E[Y(dual) - Y(trastuzumab) | X=x]

where Y is pCR.

This is a conditional risk-difference estimand for randomized treatment strategies in the corresponding trial populations.

The method estimates a CATE-like treatment-benefit score. We will **not** claim that an individual patient's unobservable ITE is directly observed or that a real-data interval is a literal confidence interval for Y(1)-Y(0).

## 4. Locked uncertainty semantics
- Semi-synthetic benchmark: calibrated interval/coverage for known τ(x) is allowed.
- Real randomized trials: report uncertainty for estimated conditional benefit / score stability and selective-risk behavior, not a literal observed individual treatment-effect interval.
- Abstention/support is an algorithmic safety output, not a treatment recommendation.

## 5. Data access and reproducibility rules
- Public data only for the confirmatory path.
- Pretreatment inputs only.
- No controlled-access data are required for the primary locked plan.
- All source IDs, sample mappings, checksums, and processed matrices must be frozen before training.
- If a required patient-level label cannot be reproduced from public sources, that dataset is dropped rather than manually reconstructed from figures or secondary reports.
