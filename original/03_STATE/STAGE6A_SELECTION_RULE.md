# STAGE 6A SELECTION RULE — LOCKED BEFORE NEW RESULTS

The following order is binding:

1. Null guard:
   mean |tau_hat| on null <= 2.0 x min(X,T) on the same Stage6A DEV runs.
2. Point-estimation competitiveness:
   mean root-PEHE across non-null conditions <= 1.10 x X-learner.
3. Improvement over original CM-OSR:
   lower mean root-PEHE across non-null conditions.
4. Mechanistic non-degeneracy:
   routed correction must be nonzero; the method cannot reduce identically to X/R.
5. Tie-break:
   gamma=1 before gamma=2; otherwise smallest mean non-null root-PEHE.

If no candidate passes all gates, report that result and do not manufacture a new
'winning' architecture.
