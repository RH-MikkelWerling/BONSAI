# Performance pipeline reconciliation — 27 October 2026

## Finding

There are real, material differences between quantities called tuning
performance and those drawn in paper figures. For the same patients and CIFs,
the underlying CIF IPCW AUC/Brier implementations agree numerically. The model
forward path also agrees between validation and standalone evaluation in a
local stable-router test. Server checkpoint/model identity remains unverified
until the artifact audit is returned.

## Pipeline comparison

| Stage | Population and censoring | Summary / selection |
| --- | --- | --- |
| Training validation `val/cr_cif_auc_macro` | Patients pooled across diseases within each outcome; administrative-censoring KM fitted on that pooled population; competing events are controls | Mean of finite outcome/horizon AUCs |
| Training `val/cr_cif_auc_endpoint_group_macro` | Same pooled patients; minimum case/control support at pooled outcome/horizon level | Endpoint-group aggregation, distinct from per-fine-disease performance |
| Checkpoint selection in paper configs | Full tuning population, known-horizon labels | Endpoint-group horizon BCE, lower is better; best-AUC epoch need not be selected |
| Standalone native `direct_cr_metrics.csv` | Cohort field of native export (usually grouped training disease), within outcome/horizon | AUC/Brier per native cohort/outcome/horizon |
| Standalone native `direct_cr_pooled_metrics.csv` | All diseases pooled within outcome/horizon | Same type of pooled AUC as training logs, for the evaluated checkpoint |
| Router tuning decision | Metadata maps to fine disease; IPCW recomputed in each fine disease/outcome/horizon | At least 5 cases/5 controls; each disease equal, outcomes equal within disease, horizons equal within outcome |
| Paper `effect_summaries.csv` | Same fine-cell metric engine as router decision if estimand is `cif_ipcw` | Finite cell gains averaged equally; no equivalent explicit 5/5 filter here; strict bootstrap inventory |
| Routing figure A and many heterogeneity forests | Fine-cell gains from plot tables | Median/IQR across cells, not weighted panel mean or a confidence interval |
| Figure bank panel / generated headline numbers | Effect-summary tables | Equal-cell mean/CI, not necessarily router-decision weighting |

Fine cells with only one case/control can have finite AUC and enter the current
figure summaries, although they do not enter tuning model selection. The paper
and decision bootstrap policies differ too: decision panels renormalize weights
over finite cells within each replicate; full paper effect summaries require
the fixed selected inventory to remain finite. Patient-subset analyses use a
third explicit policy, retaining cells finite in every replicate. This affects
intervals and coverage, not just display wording.

Pooled AUC includes cross-disease case/control comparisons; within-disease AUC
excludes them. It is mathematically possible to improve only between-disease
risk separation and improve pooled AUC while leaving every within-disease AUC
unchanged. A regression test demonstrates this. Therefore, a higher AUC in the
training log is not automatically evidence that the paper cell-level gain is
being calculated incorrectly.

The native/analysis metric parity test covers all six horizons, risk ties,
competing deaths, early administrative censoring and exact-horizon follow-up.
The stable-router forward test includes the global adapter and warm-up settings,
compares `forward_from_pooled` hazards with standalone evaluation hazards, then
compares the actual validation CIF accumulator with the standalone CIF formula.

## Model identity and provenance

`router_decision` uses canonical bundle names: its historical R_L refers to the
original final-source model. `paper_analysis_inventory` can remap R_AB_L to
analysis R_L, moving the original to R_ORIG_L. The same name in two different
reports therefore need not mean the same fit. OPERA is another alias specified
by `opera_model`; it is not intrinsically the router or automatically updated
when better models arrive.

Both training and evaluation use the contextual prediction-token state, source
adapter readout, shared risk-head weights and the same CIF conversion. The saved
model-init config and strict checkpoint load govern reconstruction. Figure
renderers consume tables and never run the neural forward pass themselves.

The +0.0138 AUC result recorded in the lock amendments is R_AB versus RBH_AB,
not R_AB versus J_L. It must not be substituted for routed-versus-joint evidence.
R2–R7 are also the original adapter-bank factorial; matched borrowing for the
fixed global-adapter design is `router_borrow_AB`.

## Server check

Use the exact environment/configuration that produced the figures:

```bash
python -m opera.run.audit_tuning_router --env-file .env.rehearsal \
  --route-diagnostics --verify-artifacts --recompute-metrics \
  > logs/rehearsal/router_audit_reconciled.json
```

This runs CPU checks on existing artifacts. It does not train, regenerate
predictions, change a winner or overwrite analysis results. Hash verification
and metric recomputation can take time; use an allocated CPU environment.

New `metric_reconciliation` records compare native and analysis AUC/Brier on
identical fine cells and recompute those estimates directly from the declared
prediction artifacts against saved `estimates.csv`. Nonzero discrepancies or
finite-status mismatches identify stale/inconsistent metric artifacts. It also
reports the native disease-pooled AUC mean on matching horizons, when that CSV
exists. Under the primary CIF IPCW estimand, the native patient-pooled summary is
also recomputed from the same CIFs and checked against its exported CSV. `point_summary_comparison_raw_auc` juxtaposes the current cell
median, equal-cell mean and tuning-decision weighted mean from saved contrasts.
These point summaries do not imply identical bootstrap intervals.

The artifact audit identifies the fit behind each alias, checkpoint/config
hashes, selection record and whether evaluation used the selected checkpoint.
It identifies route/source-removal provenance and reports route health.

## Required resolution before interpreting the draft

1. Establish actual J_L, R_AB, original-router and OPERA artifact identities.
2. Confirm the same selected checkpoints, tuning split, horizons and CIF estimand.
3. Verify numerical agreement between recomputed and saved fine-cell metrics.
4. Compare the SAME R_AB-minus-J contrast under all three aggregations and the
   native patient-pooled summary. Use the documented decision aggregation for
   reproducing the original decision; label heterogeneity medians separately.
5. Reconcile the final manuscript weighting/support rule explicitly. Existing
   docs contain competing historical policies; silently changing the scientific
   estimand to recover an expected gain would be inappropriate. Keep the raw
   tables and regenerate a fresh, coherent analysis once the rule is settled.

No evidence from the server has yet established which of identity, weighting,
support, checkpoint epoch or population pooling explains the observed gap.
