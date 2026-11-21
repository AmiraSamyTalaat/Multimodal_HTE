# STAGE 5A STRESS LOCK — B01 CausalTx Multimodal

Locked before any Stage 5A stress result is viewed.

The completed 15/15 primary grid and PRIMARY_GATE.json are frozen and must not be rerun or
changed. Stage 5A evaluates the already-prespecified H2 stress families from the experiment
contract:

1. severe weak overlap;
2. missing-modality MCAR at 10%, 25%, and 40%;
3. missing-modality MAR driven only by baseline clinical covariates;
4. null-treatment-effect negative control;
5. support-aware selective root-PEHE;
6. overlap-stratified root-PEHE;
7. 90% semi-synthetic CATE interval coverage calibrated on locked DEV only.

Missingness is patient-by-modality. Clinical features remain available. Transcriptomic and RPPA
modalities are independently missing under the locked mechanism. Missing molecular blocks are
set to zero *after train-only standardization* (the training mean). CM-OSR receives its designed
availability mask; its joint expert is unavailable whenever a core molecular modality is
missing. To make the missingness comparison stricter, the same already-trained generic
attention/MoE baselines are also evaluated with availability-masked routing at inference.

No Stage 5A result may be used to alter the primary DGP, seeds, endpoint, primary comparator,
model architecture, or primary PASS decision. X/T/R/DR learners remain in the stress suite as
transparent strong conventional controls. Secondary comparisons are reported with paired
DGP-cluster bootstrap intervals and are not promoted into a new confirmatory primary endpoint.

The 90% interval is benchmark-only: interval width is calibrated from known semi-synthetic
true CATE on the locked DEV split, then evaluated once on TEST. It must not be described as an
observed patient-level ITE confidence interval in real trials.
