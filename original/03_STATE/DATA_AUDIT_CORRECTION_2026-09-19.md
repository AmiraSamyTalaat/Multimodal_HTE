# DATA AUDIT CORRECTION — 2026-09-19

Status: **PRE-TRAINING CORRECTION; NO SCIENTIFIC MODEL WAS TRAINED**

## What the first audit found
1. I-SPY2 GSE194040 appeared as 654 and 334 samples in two platform-specific GEO matrices. This is expected; together they represent 988 sample records (987 unique patients). The old audit incorrectly required one individual matrix to exceed 900.
2. TRIO-US B07 GSE130787 returned 89 records. That is the on-treatment/run-in series, not the allowed pretreatment feature source. The correct baseline series is **GSE130786**, with 110 baseline samples.
3. The old parser inspected only `Sample_characteristics_ch1`. Two-colour GEO studies can place treatment/response metadata in another channel. The parser now scans all sample-characteristic channels.

## Locked correction
- External trial remains TRIO-US B07.
- Pretreatment expression source: **GSE130786**.
- GSE130787: prohibited as a primary predictor source because it contains on-treatment information.
- Outcome remains pCR.
- Primary contrast remains dual trastuzumab+lapatinib vs trastuzumab.
- No architecture, estimand, endpoint, seeds, or pass/fail threshold changed.

## Why this strengthens the paper
The correction removes a potential post-treatment information-leakage/deployment-realism problem before any training, while preserving the prespecified external-trial scientific question.
