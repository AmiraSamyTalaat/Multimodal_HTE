# Execution-environment comparison and attribution limits

This report concerns the frozen original benchmark reruns. It does not alter any experiment or explain a difference by assumption.

| Evidence | Archived original | New Kaggle CPU rerun |
|---|---|---|
| Stage 4 run manifest, DGP 101 / network 17 | Python 3.12.13, Linux 6.12.90 / glibc 2.35, `device=cuda` | Python 3.13.18, Linux 6.18.55 / glibc 2.39, `device=cpu` |
| Stage 6A run manifest | `device=cuda`; no Python/package versions recorded | `device=cpu`; kernel manifest records Python 3.13.18 |
| Stage 6B run manifest | `device=cuda`; no Python/package versions recorded | `device=cpu`; kernel manifest records Python 3.13.18 |
| Stage 6C run manifest | `device=cuda`; no Python/package versions recorded | `device=cpu`; kernel manifest records Python 3.13.18 |
| New kernel package versions | Exact original package freeze unavailable | NumPy 2.1.3, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.6.1, Torch 2.11.0+cpu |

The archived `requirements.txt` specifies minimum versions, not a lockfile. The original Stage 4 Python and OS/device differences are **established** by run manifests. Original Stage 6A/6B/6C device differences are likewise established. Exact original NumPy/Torch/scikit-learn versions for those stages are **not recorded**, so package-version drift in them cannot be verified. A later corrective Kaggle reference run used Python 3.13.16 and Torch 2.11.0+cu130, but it is not evidence of the original-stage environment.

CPU/GPU arithmetic, library version drift, stochastic optimization and nondeterministic kernels are plausible contributors to numeric differences. None has been isolated by a controlled one-factor rerun, so **no causal attribution is established**. The one-pair Stage 4 GPU rerun had fewer strict metric failures than its CPU counterpart (5/45 versus 22/45), but it also changed multiple runtime factors and cannot isolate device effects. Full-stage scientific decisions and exact numeric reproduction are reported separately.

SHA-256 of representative original, read-only run manifests:

| Stage | Relative source path under `Working folder/B01` | SHA-256 |
|---|---|---|
| Stage 4 | `04_RUNS/CORE_DGP101_NET17_1789843810/RUN_MANIFEST.json` | `fd7a5bbb1446fe3d8bd9a63b21a992539e872a959072b0f2889c21d23a2e798b` |
| Stage 6A | `07_METHOD_DEV_RUNS/DEV6A_DGP1061_NET19_1790140829/RUN_MANIFEST.json` | `e28ff74314e262ac966b8bc69db1241ef63daba941d31d8d3903883c958f06c4` |
| Stage 6B | `08_CONFIRMATORY_RUNS/CONF6B_DGP1201_NET17_1791280173/RUN_MANIFEST.json` | `b2188158ecb09c30921770f21471f498387207b04dbdcd3decdcf8451e779c77` |
| Stage 6C | `08_MECHANISM_RUNS/MECH6C_DGP1811_NET23_1791445334/RUN_MANIFEST.json` | `85e927022298446c4c0ec3be6094300edb51e70735391eb3c757f71d653051ec` |
