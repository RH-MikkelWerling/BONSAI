# Analysis backbone for the no-additional-training questions

The analyses in section B of `contrast_plan_20261014.md` are now required
secondary analyses. They do not require new predictive-model training. Some
require metadata, source-removal inference or benchmark exports; absent inputs
are recorded in `analysis_status.csv`, not replaced with invented estimates.

## Run

Adapt `opera/configs/locked_prediction_analysis.example.yaml` on the secure
server, then run:

```bash
python -m opera.run.analyse_locked_predictions \
  --config /secure/analysis/locked_prediction_analysis.yaml \
  --output-dir /secure/analysis/locked_predictions_seed42_365d_v1
```

The output directory must be new. Analysis validates IDs, labels, probabilities,
contracts, follow-up where supplied, metadata and contrast alignment before
writing results. It reads saved predictions only and launches no model training.
Keep patient-level inputs, loss contributions and bootstrap artifacts on the
secure server. Aggregate tables are internal until disclosure review; names
without `_INTERNAL` do not imply clearance.

## Estimands

Set `estimand` in the recipe (or `OPERA_PAPER_ANALYSIS_ESTIMAND` for the paper
workflow) and use a separate output root for each:

- `binary_known_status_unweighted`: patients whose horizon status is known;
  matches the checkpoint-selection BCE population.
- `cif_ipcw`: the full risk set for cumulative incidence at each horizon.
  Competing events before the horizon are controls; patients censored before
  the horizon have weight zero; others are weighted by the inverse Kaplan-Meier
  probability of remaining uncensored (estimated per disease/outcome cell).
  AUC and Brier reproduce `compute_competing_risk_metrics_at_horizon` exactly,
  so these are the population-risk numbers, now with the same paired patient
  bootstrap and contrasts. Weights are held fixed across bootstrap draws
  (censoring-model uncertainty is not propagated). Tabular rows are joined to
  the full risk set; patients censored before the horizon have no tabular
  prediction and carry a flagged zero-weight placeholder.

## Input contract

Use explicit model aliases from `final_contrasts_20261014.yaml`, never infer
scientific identity from folder names. Combine specialist shards under `S_L`,
disease-fixed multi-outcome fits under `O_L`, and outcome-fixed multi-disease
fits under `D_L`. There must be exactly one row per model/seed/split/fine
disease/outcome/horizon/subject. Concatenating different checkpoints for one
cell under one alias fails. Preserve subject IDs as strings, including leading
zeros. Tuning and held-out patients must be disjoint.

Canonical CSV/parquet fields:

| Field | Meaning |
| --- | --- |
| model | Fixed scientific alias, e.g. J_L, R_L, S_L, J_E, J_EA, X |
| seed | Downstream training seed; reuse a deterministic tabular fit across seeds only with explicit provenance |
| split | tuning or held_out |
| cohort | Fine disease, not grouped parent |
| outcome / horizon_days | Exact endpoint identifier and prediction horizon |
| subject_id | Stable patient ID |
| label / probability | Binary horizon status and predicted probability |
| known | Optional boolean; absence asserts all rows have known horizon status |
| data_contract | Common release/split/eligibility/endpoint contract hash, not the model hash |
| time_days / event | Optional paired follow-up fields; validated against label/known if supplied |

Artifact declarations may supply model/seed/task constants absent from a file;
conflicts with existing values fail. Native `direct_cr_predictions.parquet`
wide `cif_*d` exports are supported with `format: direct_cif`. They derive
known status from primary event, competing death and observed follow-up.
Use `cohort_from_metadata: true` for pooled grouped exports, mapping subject IDs
through `patient_metadata.cohort_fine`. Do not relabel a whole grouped export
as one fine disease. Standard tabular CSVs must first be joined to the locked
labels/eligibility and keyed by cohort/outcome/horizon/split; the runner does
not guess these from names. Legacy NPZs need an explicit canonical table export.

Matching hashes are assertions from the supplied inventory. The runner hashes
actual input bytes and detects conflicting contracts, but cannot prove that
feature generation or label provenance was correct from prediction files alone.
Scientific run/checkpoint/config provenance must accompany the input inventory.

Patient metadata has one row per subject and explicit pre-index covariates.
Define continuous bins in the recipe before held-out analysis; categorical
fields have a cardinality limit. Provenance declarations are mandatory. Missing
metadata rows fail instead of silently shrinking the evaluation population.
An absent covariate value is retained as `missing`. Examples include age, sex,
pre-index EHR density, prediction year, train-defined token overlap and
baseline-defined atypicality. Never use post-index utilization/clinical outcomes
to define these subgroups. For domain mismatch, set `domain_mismatch_covariates`
to the recipe's declared pre-index/train-only feature names.

Cell metadata has unique `(cohort,outcome,horizon_days)` rows and must cover
all evaluated cells. Use target-cell training support (`n_events_train`,
`n_train`), clinical family and endpoint group. It describes target scarcity,
not the larger D/J pooled training denominator. Add fixed train-only support
bins with `cell_groups`; choose reporting columns with `cell_summary_strata`.

The planned coverage roster has unique model/seed/cohort/outcome/horizon rows
and `status` (complete, failed, unsupported, not_run, etc.). Include absent
specialists, rather than building the roster from successful prediction files.
Predictions can be two-class, single-class or absent; availability alone does
not establish statistical or clinical validity.

Route inputs use long model/seed/split/cohort/outcome/subject/source/bank/weight
rows. Include the shared route and all source rows: weights must sum to one.
Native JSON disease/outcome weight exports are also supported: supply
`route_model`, `route_seed` and patient fine-disease metadata. Source-removal
exports are prediction artifacts with separate
aliases and the same task contract. Declare their gain expressions in
`extra_contrasts` and names in `source_removal_contrasts`. This is extra inference,
not retraining, and retains the existing removal-plus-renormalization semantics.
For native exports use `probability_prefix: removed_source_cif_` on the removed
artifact entry to select the corresponding wide probabilities.
Compute inventories may record model, seed, GPU hours, steps, parameters,
inference throughput and storage; the runner saves them without guessing values.

## Outputs and inference rules

- `estimates.csv` and `contrasts.csv`: all available planned contrasts plus
  interactions, with AUC, Brier, log loss, fixed-bin ECE and signed calibration
  bias. Positive gain favors the first model for skill/loss metrics; calibration
  bias differences remain signed errors, not a superiority score.
- `contrast_coverage.csv`: requested comparisons missing a model stay visible.
- `effect_summaries.csv`: support/family/group/disease-stratified descriptive
  effects. Default means weight task cells equally, explicitly **not** patient
  population or endpoint groups. Main-paper hierarchy/weights remain an analysis
  lock decision; per-cell estimates feed the existing hierarchical-rarity tools.
  `rarity_inputs.csv` provides gain/difference, bootstrap standard error,
  minority-class count and target training support for the existing spline
  hierarchy. Filter to one contrast, split and horizon before fitting; do not
  pool all rows as independent observations. No Bayesian model is fit by this runner.
- `negative_transfer.csv`: cells defined by tuning-only J-S harm, joined to
  held-out J-S and router-J effects. Support thresholds and a descriptive
  tolerance are explicit. A negative point estimate is not proof of harm;
  inspect uncertainty and the direct router-S contrast too.
  `negative_transfer_summary.csv` summarizes those fixed tuning-defined strata
  with the same correlated bootstrap draws.
- `calibration.csv`: fixed-bin reliability counts/observed/predicted fractions.
  No held-out calibrator is fitted. ECE is diagnostic; Brier/log loss are the
  proper probability-score comparisons.
- `patient_gains_INTERNAL.csv` / `patient_subgroups.csv`: realized paired
  Brier/log-loss contributions and subgroup means with patient-bootstrap
  intervals. These are predictive gains, not individual treatment effects.
- `domain_mismatch.csv` / `calendar_heterogeneity.csv`: explicitly declared
  subgroup slices; missing features/origins show a missing-input status.
- `model_coverage.csv`: prediction support and failed/unavailable model cells.
- `strategy_selection.csv`: best S/D/O per disease/outcome/seed selected by
  tuning log loss averaged across supplied horizons, deterministic ties. Saved
  held-out comparisons against J/R never choose a winner using held-out labels.
  Missing candidates/tuning or selected held-out predictions are recorded.
- `route_weights.csv` / `source_removal_sensitivity.csv`: descriptive gate
  distributions and paired removal effects, without causal borrowing claims.
- `paired_bootstrap_index.csv` and compressed
  `paired_bootstrap_draws_INTERNAL.npz`: row-aligned draw vectors. One patient
  draw is shared across models, endpoints, horizons and training seeds within
  cohort/split.
  Aggregation preserves that dependence and a fixed finite cell inventory.
- `seed_averaged_effects.csv`: cellwise mean effects across the supplied fitted
  seed roster, using the same patient draws. Seed range/SD are reported
  separately; intervals do not estimate all training randomness.
- `analysis_status.csv`, `input_audit.csv`, `analysis_manifest.json`: input
  hashes, exclusions, recipe and explicit computation/missing-input statuses.

The backend uses **unweighted binary known-status evaluation**. Early censored
unknown horizons are excluded, competing deaths are known negatives, and
single-class cells retain probability losses but have undefined AUC. These
tables do not replace the existing full-risk-set IPCW CIF AUC/Brier evaluator.
Report the estimand and known-status counts, and use the existing survival-aware
comparisons for primary population-risk claims. Do not mix the two metric
definitions in one forest or silently call this BCE censoring-adjusted.

Intervals are conditional on selected fitted models, per downstream seed;
tuning intervals do not correct model/checkpoint selection. Fewer than 20 or
80% finite bootstrap draws suppress an interval and record the reason. Do not
claim equivalence from an inconclusive interval or pool patient-bootstrap draws
as independent training-seed replicates. Formal multiplicity/weighting decisions
remain separate from these descriptive tables.

Run main365 first and shard larger work by seed/horizon if needed. Bootstrap
ROC computation is vectorized in bounded chunks and draw storage is compressed;
these analyses avoid new GPU fits but are not literally cost-free. A missing
prediction is recorded; a present prediction with a different patient set,
label, or contract fails. No automatic patient intersection is allowed.

## Router fairness audit

The query consumes pre-index encoder states, fine-disease identity and target
outcome identity. It does not consume the observed target label. Target outcome
identity is analogous to selecting the relevant prediction head. Adapter
support/shrinkage is computed from training labels, not held-out labels. Both
anchored and unanchored variants are legitimate supervised learning rules.
They differ in gradient ownership, not access to test outcomes.

There is a concrete unresolved input-provenance issue:
`identification_first_line_treatment_v2.R` permits +14-day diagnosis evidence
in pass1 and +90-day evidence in the forward-rescue pass. The multicohort data
module maps the supplied fine-disease metadata directly into router inputs.
`docs/ascertainment_review.md` says retrospective rescue was intended for cohort
analysis only. Therefore a retrospective subtype must not automatically be
treated as an available-at-index model feature. This affects the claim for
disease/target-conditioned queries even in outcome-source-only mode.

`router_identity_audit.csv` flags evidence dates at/after index and missing
provenance. Supply `index_date`, `cohort_identity_evidence_date`, and optionally
`cohort_source`. The evidence date must be the date **all information needed
to assign the actual supplied subtype was available**, not merely its earliest
diagnosis code; future confirmation counts as future evidence. A separate
`baseline_disease_clinically_confirmed` boolean plus an independent
`baseline_confirmation_reference` can document clinician knowledge before
registry coding. Later coding alone does not prove future knowledge was used.
Missing provenance is not clearance. The audit never changes cohort labels.

For a prospective claim, audit the actual sealed metadata with clinical review.
If identity is retrospective-only, separate retrospective evaluation strata from
prediction-time identity, or use an explicitly defined shared fallback/coarser
baseline-known identity in new matched fits. Do not silently change inputs in
active/resumed runs or redefine existing trained sources. Specialists already
condition on disease through model selection; that selection must also be
available in the intended deployment. Cross-disease tabular models should get
the same legitimate baseline identity; cell-specific XGBoost already knows it
through its training/model choice. The shared-J comparison measures the whole
added conditional-routing architecture, including explicit identity and capacity;
home-only/fixed-routing controls are needed to attribute gains specifically to
learned borrowing. More parameters or more training labels are not leakage.

Study-team clarification on 14 October: the rare late-coded cases received
disease-specific cancer treatment before formal diagnosis registration, so the
clinical interpretation is coding delay rather than uncertain cancer diagnosis.
The metadata contains the diagnosis date used for classification and first-line
treatment date. Record this rationale; do not call later registration proof of
cheating or automatically exclude these patients. The remaining review is whether
the exact *fine subtype* supplied to the router was clinically known, since some
regimens overlap subtypes. The audit quantifies date lags by cohort/source and
can support a no-new-training sensitivity report excluding late-coded cases.
It is a provenance review, not a request to rebuild the signed-off cohort.
When both date fields are provided, the default `identity_date_sensitivity`
evaluates the SAME fitted models on patients with a nonmissing diagnosis date
no later than index. `identity_sensitivity_counts.csv` preserves exclusions and
empty cells; `late_diagnosis_exclusion_sensitivity.csv` reports paired contrasts.
This retrospective robustness check uses no retraining and does not redefine
the cohort or prove every remaining fine label was known at prediction time.
It also cannot undo any training-input provenance issue by itself.
Use `patient_metadata_column_map` to map actual diagnosis/first-line column
names to the canonical audit fields. A diagnosis date is a proxy until its
relationship to complete subtype evidence is confirmed.

## Question-aligned Bayesian handoff

Use `opera.run.bayesian_contrasts` with this runner's `contrasts.csv`, paired
bootstrap index/NPZ and manifest, rather than stacking seed rows into the legacy
independent-error rarity spline. Its manifest must include
`shared_bootstrap_axes` containing `training_seeds`. Include approved
`endpoint_group` in cell metadata for equal-disease/endpoint-group summaries.
The [Bayesian runbook](bayesian_question_analysis_20261014.md) describes exact
factorial identities, covariance, prior-only checks and sensitivity models.
