# STAGE 6B — Confirmatory Lock

Stage 6A completed 10/10 DEV-only development pairs and froze `AOSR_X_gamma2`.

Stage 6B is the one-shot confirmatory semi-synthetic test.

- Fresh DGP seeds: 1201, 1327, 1451, 1579, 1699
- Network seeds: 17, 29, 43
- Training: TRAIN only
- Evaluation: TEST once
- No candidate switch, retuning, seed search, or threshold tuning
- The original Stage 3 primary result and Stage 5A stress evidence remain preserved.

Confirmatory gates test null behavior, competitiveness with X-learner, improvement over
frozen CM-OSR v1, advantage over capacity-matched multimodal controls, support-selective
risk behavior, and a non-degenerate routed residual correction.

The support-shrunk CM-OSR gamma=2 development control is retained transparently as a
comparator but cannot replace the frozen AOSR candidate after Stage 6B results are seen.

If Stage 6B passes, the next work is mechanism ablation of the frozen AOSR, then
CHER-LOB nested cross-fitting, score freeze, and locked TRIO-US B07 external validation.
