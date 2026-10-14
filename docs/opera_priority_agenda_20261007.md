# OPERA priority agenda — 7 October 2026

## Gate 0: identify the server router before new training

Run the read-only command in `router_query_audit_20261007.md` on the server.
Review `current_router_code` (loaded file, hash, Git HEAD, dirty status), both
resolved Hydra configs, and both saved checkpoint modes/rosters. The checkpoint
proves its saved query mode and fine-disease roster, but not an unsaved
training-time source file hash. Confirm that the current server code contains
the interacting branch and that both training and evaluation pass
`cohort_fine_index`, derived from `cohort_fine`. Do not launch capacity or
source-factorial runs until this check is reviewed.

## Parallel tracks after Gate 0

1. **GPU: router mechanism, then capacity.** Finish paired tuning routes and
   predictions for additive vs interacting. Quantify whether the interaction
   changed routes/CIFs. Start the disease-only/outcome-only/both source
   factorial if its candidate contracts pass. For capacity, use the controlled
   0.1/0.2/0.3 residual-scale plan in `router_capacity_ablation_20261007.md`;
   rerun the 0.1 arm under current code if historical code provenance remains
   unresolved. Prespecify common-epoch and fixed-checkpoint comparisons; add
   replicated seeds for a confirmatory claim. Held-out remains closed.
2. **CPU/figures: neural predictability atlas.** On the secure server, run
   `plot_neural_predictability_prototype --run` with the approved 69-outcome
   registry and primary-v2 display groups, as shown in
   `supervisor_validation_cohort_table1_20261007.md`. It produces the neural
   tuning overview, a 69-by-6 outcome atlas, and an aggregate CSV. Review its
   actual render and mark unavailable cells/support. Do not substitute the
   older local XGBoost binary-AUROC prototype: its metric and input contract
   are different.
3. **CPU/figures: borrowing redesign.** Keep the raw route-budget chart, but
   use *source-centered* specificity for a readable target-by-source view.
   Show outcome-specific (not only family-averaged) rows for prespecified
   clinical targets; retain the 69-by-69 matrix as supplement. For disease
   borrowing, stratify or facet by patients' **home fine disease** before
   interpreting target-outcome patterns: the existing outcome-to-disease
   matrix averages over changing home-disease mixtures. Mask unsupported
   cells, display row counts, distinguish own from borrowed sources, and label
   that route mass is preference—not contribution to predictive performance.
   Check any proposed figure against actual server exports.
4. **CPU/figures: finish drafts.** Review the 69-outcome forest spacing/legend,
   five outcome panels with denser PCA arrows and exploratory UMAP companions,
   and the current cohort-flow/Table 1 denominators. Keep patient-level files
   on the secure server and move only disclosure-reviewed aggregates.

## User's parallel track

The tabular feature-matrix repair is in a different repository and owned by
the user. It should proceed while GPU runs execute. Before a neural-vs-XGBoost
claim, verify the same approved outcomes, index, eligibility, temporal split,
prediction horizon, and evaluation metric; document quality-registry
missingness by era. No need to block router jobs on this repair.

## After the above

- **Index-time diagnosis sensitivity for routed models.** The 14-day Pass 1
   and 90-day Pass 2 post-treatment diagnosis windows accommodate coding lag
   after cancer treatment begins; do not assume a later registry date means
   the disease was clinically unknown at treatment. Nevertheless, `cohort_fine`
   directly enters the router, so the previous aggregate-only interpretation
   is insufficient. On the secure server, count assignments based on diagnoses
   dated after first-line treatment (by pass, lag, disease, split, and outcome
   horizon), and check whether Pass 2 confirmation uses records still later
   than its 90-day window. Compare the frozen routed model's predictions with
   index-available fine labels only, using its unknown/shared fallback where
   fine identity cannot be established at index; report performance overall
   and in the affected subgroup. Keep retrospective coding-lag ascertainment
   distinct from an at-treatment deployment claim, and disclose the rule and
   sensitivity in the manuscript. Do not alter the approved cohort or open
   held-out data as part of this exploratory audit.

Freeze the best tuning architecture/scale, then specify the global-to-disease
specialization experiment against that fixed baseline. Quality-registry
vocabulary expansion follows a coverage/harmonization audit. Held-out testing
is last, after model/figure choices and decision rules are frozen.
