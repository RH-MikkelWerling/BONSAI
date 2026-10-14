# Evaluate adapter epsilon and the source-bank claims

The lower-epsilon experiment changes optimization of both source banks, not the dataset's information content. In the default comparison, both fits retain anchored home updates, the same full source inventory, rank, residual scale, shrinkage, curriculum, encoder, seed, and head/encoder epsilon. The candidate gives only source down/up matrices epsilon 1e-8 instead of 1e-6.

## After the fits finish

```bash
bash scripts/check_router_adapter_epsilon_results.sh preflight
bash scripts/check_router_adapter_epsilon_results.sh run
```

The check requires completion markers and equal last-checkpoint epochs. It validates that only adapter epsilon differs, inspects the actual optimizer-group epsilon/moments, audits the same sampled training patients, and exports tuning predictions at every configured horizon. Paired outcome-level AUC/Brier contrasts are produced at 365 and 730 days, with under-supported cells skipped and recorded. It uses one GPU for inference/diagnostics and launches no new training. Held-out is not evaluated. Defaults use the anchored full reference and the full anchored epsilon-1e-8 candidate; override `OPERA_ADAPTER_EPS_REFERENCE_RUN` and `OPERA_EPS_ROUTER_RUN` together if another reference was selected. Choose a fresh `OPERA_EPS_EVALUATION_DIR` if rerunning.

The learning question is whether actual source updates and parameter magnitudes increase, particularly for outcome sources, and whether their FP32 projected corrections survive configured precision. The predictive question is whether paired tuning discrimination/calibration improve, including supported sparse endpoints and related grades. Bigger corrections or more varied gates do not establish benefit. The audit now uses the adapter group's epsilon for its matrix-gradient threshold rather than applying the global head epsilon to every matrix.

## Claims from the final source comparisons

Hold the selected optimization rule, curriculum, inputs, loss, and selection rule fixed. With N = neither source bank, D = disease sources, and DO = both banks:

- D versus N measures the incremental value of disease-indexed adaptation/borrowing above the joint multi-outcome baseline.
- DO versus D measures the incremental value of the outcome source bank conditional on disease sources.
- DO versus N measures their combined incremental value.

These contrasts do not mean disease or outcome information was absent from N. The shared encoder and outcome heads already learn from the outcome panel, and clinical histories contain disease information. Query context is retained in active arms. The effects include added conditional capacity, sharing and gate reallocation, rather than a uniquely identified biological relationship mechanism.

Three nested conditions are sufficient for the two incremental claims. An outcome-source-only O condition is additionally needed to show DO beats *either* bank or estimate the full factorial interaction. A same-capacity fixed/home-only routing comparison addresses the stronger claim that learned selective cross-source relationships drive the gain.

If the full epsilon-1e-8 model improves, it does not by itself show outcomes caused that improvement: disease matrices also changed epsilon. A matched D condition at epsilon 1e-8 is needed for that attribution. Existing N can be reused if all other settings, initialization and prediction behavior match, since its source matrices have no predictive effect and head/encoder epsilon remain unchanged. A changed curriculum or backbone would require a matched N again.

Nonzero outcome gradients and corrections already establish that outcome routing is mechanically possible at epsilon 1e-6. The experiment tests effective optimization and incremental predictive usefulness in this architecture/dataset. It does not establish a universal improvement for routers generally. If stronger outcome learning still yields no incremental gain, finalize and report that finding rather than continue tuning until the combined model wins.
