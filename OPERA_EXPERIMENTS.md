# OPERA Experiment Workflow

This note documents the reproducibility hooks added for the OPERA paper setup.

## Prospective Splits

Outcome creation can optionally assign splits from `index_date` instead of
trusting input directory names. The OPERA paper split is defined once in
`opera/configs/manifests/temporal_split.yaml`:

```yaml
prospective_split:
  contract: opera/configs/manifests/temporal_split.yaml
```

`bonsai.run.create_outcome` writes a split summary CSV next to each outcome
parquet. The summary records split counts, index-date ranges, and observed
events.

The prospective outcome split is independent of ehr2meds' random physical
train/tuning partition. OPERA pools physical subject-data files before applying
outcome membership, so patients are not dropped because their SSL partition
name differs from their downstream temporal split.

Validate a generated outcome file before training:

```bash
python -m bonsai.run.validate_splits \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --contract opera/configs/manifests/temporal_split.yaml \
  --fail_on_error
```

The validator reports subject overlap across splits and date-boundary
violations for prospective setups.

Before launching any OPERA stage, validate labels, split subject-data files,
DAPT inputs, and optional DAPT embedding stores against the same contract:

```bash
python -m opera.run.validate_split_contract \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --subject_data ssl_train=/data/dlbcl/subject_data_train.pt \
  --subject_data ssl_validation=/data/dlbcl/subject_data_tuning.pt \
  --dapt_subject_data ssl_train=/data/dlbcl/subject_data_train.pt \
  --dapt_subject_data ssl_validation=/data/dlbcl/subject_data_tuning.pt \
  --embedding_store /results/dapt_embeddings.pt \
  --fail_on_error
```

Finetuning entry points write `label_split_summary.csv` to the run directory.
This file records raw subjects, retained labelled subjects, label prevalence,
and insufficient-follow-up exclusions by split.

## Follow-Up Eligibility

Bounded-window binary labels support explicit minimum follow-up requirements by
split:

```yaml
labels:
  require_min_followup_train: true
  require_min_followup_val: true
  require_min_followup_test: true
```

Ordinary BCE defaults to full follow-up in every split. Survival and
survival-contrastive runners explicitly use ascertainment eligibility and keep
right-censored observations instead.

## Rare-Outcome Batch Construction

Event-aware batching uses outcome quotas only when an endpoint has enough
distinct evidence. Focused batches require at least one unique event and
(production configs) eight unique eligible patients before an outcome
receives focused batches — `min_unique_events_for_focus` was lowered from 2
to 1 across every maintained `event_aware` config (`contrastive.yaml`,
`generated/joint_opera_full_panel.yaml`, `joint_finetune.yaml`): a single observed event still anchors a real KM
cumulative-mass location, and the quota-drawing code
(`_draw_events`/`_draw_from_pool` in `stratified_sampling.py`) already caps
every quota at whatever's actually available, so the old threshold excluded
outcomes the mechanics already handled safely. `min_unique_valid_for_focus`
is left alone — that one guards the minimum for a non-degenerate pairwise
weight matrix. Quotas are capped by the unique pool: a patient is never
cloned inside a batch. Patients already used during the epoch are
progressively deprioritized so a patient labelled for many outcomes does not
become the default choice for every task. Outcomes below the threshold can
still contribute opportunistically when two or more genuinely distinct
eligible patients co-occur in a batch.

The contrastive loss also excludes equal-subject pairs from both its target
weights and softmax denominator. Joint validation accumulates predictions over
the full epoch, allowing rare cases and controls from different batches to form
one AUROC. Event enrichment and positive-class weighting are mutually
exclusive in joint training to avoid amplifying rare events twice.

## Rare-Cohort Batch Composition (diagnostic, not oversampling)

`MultiCohortContrastiveDataModule` pools patients from multiple disease
cohorts into one `ConcatDataset` for `EventAwareSurvivalBatchSampler`, which
was previously blind to which cohort each patient came from. Investigating
this surfaced that per-patient sampling probability (survival-time bucket
weighting + usage-count decay) is already independent of cohort population
size — a rare cohort isn't penalized per patient, it simply has fewer
patients, so its *aggregate* epoch contribution is proportionally smaller.
That's normal, not a bug.

Whether artificially inflating a rare cohort's per-patient exposure (e.g. via
effective-number-of-samples reweighting, the same technique
`cross_outcome_weighters.py` already uses for outcome class imbalance) would
actually improve held-out rare-cohort performance — versus simply adding
overfitting risk on a handful of patients — is an empirical question the
sampler design can't resolve on paper. Rather than guess, `stratified_sampling.py`
gained an optional `cohort_labels` parameter on `EventAwareSurvivalBatchSampler`
that feeds a **diagnostic-only** per-cohort epoch-coverage block into
`summary()` (expected draws per epoch per cohort, flagged with
`[WARNING: <1× per epoch on average]` the same way outcome coverage already
is). It never affects `__iter__`/the draw probabilities. `MultiCohortContrastiveDataModule`
computes and passes this automatically. Decide whether cohort-aware
oversampling is worth building only after looking at real coverage numbers
from this diagnostic on an actual training run.

## DAPT-Prior Mechanisms — Activation and Ablation

`MultiOutcomeSurvivalLoss` has always supported two mechanisms gated on a
`dapt_embedding_store` dict (`{subject_id: pre-projection DAPT embedding}`):

- **Pairwise similarity modulation** (`dapt_lambda_floor`): multiplies the
  KM-based pair weight by `floor + (1-floor) * cosine_similarity01`, damping
  (never zeroing, due to the floor) pairs whose frozen DAPT-stage clinical
  presentations look dissimilar — a check against purely-coincidental-timing
  pairs dominating the loss.
- **Anchor loss** (`dapt_anchor_weight`): pulls the pooled (pre-projection)
  encoder state toward its frozen DAPT position, guarding against
  representation drift during contrastive fine-tuning.

Both were dead code in practice: `dapt_embedding_store` defaults to `null` in
every config, and no run script ever populated it (`build_dapt_embedding_store`
in `opera/functional/extract.py` was fully implemented but called from
nowhere). `opera/run/build_dapt_embedding_store.py` now wires it up — loads a
DAPT checkpoint's encoder, runs it over the exact same pooled cohort
population contrastive training uses (via `MultiCohortContrastiveDataModule`),
and saves the pre-projection embeddings.

Starting values (`dapt_lambda_floor: 0.55`, `dapt_anchor_weight: 0.2` in
`contrastive.yaml` and `generated/joint_opera_full_panel.yaml`,
up from the previously-inert `0.3`/`0.0`) are informed, **not validated**.
The floor was raised specifically because DAPT-embedding outliers are
disproportionately likely to *be* the rare/unusual patients the rare-outcome
and rare-cohort work above cares about — aggressive clinical-similarity
gating could quietly re-suppress exactly those patients' pair weight.
Before trusting either number on a real run, log `dapt/weight_mean|std|min|max`
(already instrumented in `opera_nets.py`) and the raw anchor-loss magnitude
against the main contrastive loss on a few batches, and adjust if the ratio
looks off.

**Planned ablation** (2×2 minimum): `dapt_lambda_floor ∈ {1.0 (neutral/off), 0.55}`
× `dapt_anchor_weight ∈ {0.0 (off), 0.2}`, using `floor=1.0` rather than
dropping `dapt_embedding_store` entirely for the "off" arm so the control run
is identical in every other respect and only the multiplier is neutralized.
For each arm, re-run the existing hierarchical rarity aggregation on that
arm's predictions, not just the aggregate AUROC delta table — the question
that matters is whether anchoring to DAPT clinical similarity changes the
slope/intercept of the OPERA-vs-baseline benefit curve at low training-event
counts, which is genuine supporting evidence for (or an informative
qualification of) the paper's central small-cohort-benefit claim, distinct
from and complementary to the main comparison table.

## Long-Sequence Pretraining

Pretraining and DAPT can avoid wasting long patient histories with mixed-window
truncation:

```yaml
training:
  max_len: 8192
  truncation_strategy: mixed_window
  validation_truncation_strategy: tail
  tail_window_probability: 0.5
```

The input view is always `background tokens + clinical window`. Tail windows
keep recent history; random windows expose earlier or middle history without
expanding one patient into many samples per epoch. Autoregressive pretraining
masks the artificial background-to-window boundary target.

## Checkpoint Lineage

Lightning modules save:

- `model_config` and/or `encoder_config`
- `checkpoint_metadata`
- source checkpoint and training stage when provided by run scripts

Training entry points also write `checkpoint_metadata.json` next to the saved
Lightning checkpoints. This sidecar mirrors the config and lineage metadata so
operators can audit a run without loading a `.ckpt` file.

Standalone evaluation reconstructs models from the saved full config and loads
weights strictly by default. Old checkpoints without full config fail with an
explicit metadata error.

Checkpoints created by the former ModernBERT-backed BONSAI encoder are also
reported explicitly as legacy artifacts. The native RoPE/FlashAttention
architecture has different parameter names and semantics, so those weights are
not silently partially loaded. Evaluate them in their original environment or
retrain the pretraining/DAPT/OPERA chain for native-backbone comparisons.

## Evaluation Outputs

Outcome task configs may reuse one event-time parquet for multiple horizons:

```yaml
outcomes:
  mortality_1y:
    outcome_file: mortality.parquet
    n_hours_end_include: 8760
  mortality_2y:
    outcome_file: mortality.parquet
    n_hours_end_include: 17520
```

Use one event-time parquet per endpoint where possible. `censor_date` is the
end of observed follow-up and can be present for both event and non-event
patients; bounded binary windows derive full-follow-up eligibility from it.
The hour bounds are inclusive. Event-free follow-up is capped at the horizon,
and early administratively censored controls are excluded from BCE training and
binary evaluation. A configured death table produces competing `event=2` only
when death occurs inside the task window.

Each evaluation run writes:

- `metrics.json`
- `predictions.npz`
- `threshold_sweep.csv`
- `decision_curve.csv`
- `bootstrap_ci.csv`
- `result.csv`
- `result.jsonl`
- optional `subgroup_metrics.csv`

Subgroup metrics are enabled with:

```yaml
paths:
  subgroups: /path/to/subgroups.csv

subgroups:
  columns: [age_group, sex, calendar_period, cohort]
```

The subgroup table must contain `subject_id` plus the requested columns.
Prediction-file baselines support `--subgroups` and `--subgroup_columns`.
Aggregation writes `subgroup_results.csv` and, when OPERA and the named
baseline are both present, `subgroup_delta.csv`.

Tabular or external baselines can provide prediction files and use the same
metric suite:

```bash
python -m opera.run.evaluate_predictions \
  --predictions /results/tabular/tabular_ehr_dlbcl_mortality_1y_predictions.csv \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --cohort dlbcl \
  --outcome_name mortality_1y \
  --model_family tabular_ehr \
  --n_hours_end_include 8760 \
  --output_dir ./results/dlbcl/mortality_1y/tabular_ehr
```

Locked feature matrices can be converted into those prediction files with:

```bash
python -m opera.run.train_tabular_baselines \
  --features /data/dlbcl/features/tabular_ehr.parquet \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --cohort dlbcl \
  --outcome_name mortality_1y \
  --model_prefix tabular_ehr \
  --models xgboost \
  --n_hours_end_include 8760 \
  --output_dir /results/tabular
```

The feature matrix contract is one row per `subject_id`; all non-reserved
columns are treated as candidate features unless excluded with
`--exclude_columns`. Numeric features are median-imputed with missingness
indicators, categorical features keep missingness as a category, and each run
writes a feature-missingness report.

Binary tabular models fit only the temporal `train` rows. Tuning is enabled by
default on `tuning` (2022-2023), and each run saves separate tuning predictions
before writing the untouched `held_out` (2024+) predictions. Use `--no-tune`
only for an explicitly prespecified, non-selected ablation.

For low-n/high-p cells, run both `logistic` and `xgboost`. The logistic grid
includes ridge, elastic-net, and sparse fits; the XGBoost grid uses shallow
trees, column subsampling, minimum child support, and L1/L2 regularization. An
illustrative simulation with 100-500 training rows and 500-2,000 features found
that sparse logistic regression consistently improved on ridge, while the
regularized XGBoost profile retained essentially the same AUROC as the former
profile with better held-out log loss. Reproduce and extend it with:

```bash
python -m opera.run.simulate_tabular_low_n \
  --output_dir /results/simulations/tabular_low_n \
  --repeats 20
```

TabPFN can be run as an optional baseline in an environment with the optional
extra installed:

```bash
python -m pip install -e ".[tabpfn]"
python -m opera.run.train_tabular_baselines \
  --features /data/dlbcl/features/tabular_ehr.parquet \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --cohort dlbcl \
  --outcome_name mortality_1y \
  --model_prefix tabular_ehr_tabpfn \
  --models tabpfn \
  --tabpfn_max_features 500 \
  --tabpfn_max_train_rows 10000 \
  --n_hours_end_include 8760 \
  --output_dir /results/tabular
```

## Aggregating Results

Aggregate any directory tree of evaluation outputs:

```bash
python -m opera.run.aggregate_results \
  --results_dir ./results \
  --output_dir ./results/aggregated \
  --baseline per_cohort \
  --comparator joint \
  --rarity_plots \
  --diagnostic_plots
```

Before launching a long sweep, run:

```bash
python -m opera.run.check_readiness \
  --config opera/configs/sweep_example.yaml \
  --fail_on_issue
```

For confirmatory runs, validate source paths and build the outcome-specific
cohort-flow table:

```bash
python -m opera.run.check_readiness \
  --config opera/configs/sweep_example.yaml \
  --require_existing_paths \
  --fail_on_issue

python -m opera.run.summarize_cohort_flow \
  --config opera/configs/sweep_example.yaml \
  --output ./results/cohort_flow.csv
```

Eligibility sidecars are outcome-specific. They distinguish source coverage,
baseline adequacy, post-index observability, follow-up, and final eligibility;
unascertainable outcomes must not be encoded as negative labels. The sidecars
are enforced before label construction in training and evaluation, which gives
multi-outcome models a real patient-by-outcome missingness mask.

This writes:

- `all_results.csv`
- `results_wide.csv`
- `model_summary.csv`
- `task_size_summary.csv` when `n_total` is present
- `pretraining_scale_summary.csv` when `pretraining_scale` is present
- `joint_minus_per_cohort.csv` when baseline/comparator are provided
- rarity-specific delta tables when `--baseline` is provided
- rarity plots when `--rarity_plots` is set
- aggregate seed-stability and subgroup-delta figures when
  `--diagnostic_plots` is set and the corresponding tables are available

Use `--strict_aggregation` for paper outputs. Duplicate
cohort/outcome/split/seed/model keys are errors because selecting the first row
would make paired denominators ambiguous.

For task-size binned summaries, the default bins are `<100`, `100-499`,
`500-999`, `1k-4,999`, and `>=5k` labelled evaluation subjects.

## Joint vs Per-Cohort

Build the direct comparison table for the shared-learning claim:

```bash
python -m opera.run.joint_vs_per_cohort \
  --results_dir ./results \
  --output_dir ./results/joint_vs_per_cohort \
  --joint_model joint \
  --per_cohort_model per_cohort
```

This writes:

- `joint_vs_per_cohort.csv`
- `joint_vs_per_cohort_summary.csv`

An optional `--cohort_sizes cohort_sizes.csv` file can add size metadata to the
comparison table. It should share `cohort`, `outcome`, and optionally
`outcome_window_hours` with the result schema.

## Label Efficiency

These runs are synthetic rarity / label-scarcity experiments: the disease cell
is otherwise common, but training labels are artificially reduced while
validation and test sets are preserved.

The label-efficiency runner supports either a single task:

```bash
python -m opera.run.label_efficiency \
  --sweep_config opera/configs/sweep_example.yaml \
  --cohort dlbcl \
  --outcome mortality_1y
```

or many tasks:

```bash
python -m opera.run.label_efficiency \
  --sweep_config opera/configs/sweep_example.yaml \
  --tasks dlbcl:mortality_1y,mm:infection_90d \
  --fractions 0.01,0.02,0.05,0.1,0.25,0.5,1.0 \
  --seeds 42,43,44
```

Outputs include per-task results, a pooled median summary, baseline-delta
tables, a pooled label-efficiency plot, and secondary per-task plots for the
first tasks or the tasks named with `--plot_tasks`.

## Real Rare Cohorts

Real rare disease evaluation is configured separately from synthetic label
scarcity:

```bash
opera/configs/rare_cohort_benchmark.yaml
```

Use `rarity.mode=real` and fill `rarity.size_metadata` during evaluation, or
provide equivalent fields in result rows before aggregation. Result rows support:

- `rarity_mode`: `none`, `synthetic`, or `real`
- `rarity_tier`: optional tier such as `real_rare_disease` or `small_cohort_outcome_cell`
- `n_train`, `n_val`, `n_test`
- `n_events_train`, `n_events_val`, `n_events_test`
- `prevalence_train`, `prevalence_val`, `prevalence_test`

Aggregate real rarity without mixing it with synthetic rarity:

```bash
python -m opera.run.aggregate_results \
  --results_dir ./results \
  --output_dir ./results/aggregated \
  --baseline tabular_ehr \
  --rarity_plots \
  --min_train_events 5 \
  --min_test_events 5
```

The rarity tables emphasize `delta_auroc_vs_baseline` rather than raw AUROC:

- `synthetic_rarity_task_level.csv`
- `synthetic_rarity_pooled.csv`
- `real_rarity_task_level.csv`
- `real_rarity_pooled.csv`
- `real_rarity_pooled_main.csv` when event thresholds are used
- `rarity_delta_combined.png`

The minimum-event flags mark unstable real rare-cohort rows as
`supplement_only`; they do not drop raw result rows.
Plot helpers save companion `.pdf` files for paper figures.

## Natural Rarity Analysis

The legacy `rarity_experiments` runner is deliberately disabled until its
synthetic label-scarcity workflow is rebuilt from the shared-data registry with
a matched comparator. The current production natural-rarity workflow uses the
fine-only generated sweeps and `opera/configs/hierarchical_rarity.yaml`; it
fails closed if grouped or global result artifacts are present.

## Pretraining-Scale Ablations

Compare multiple upstream checkpoints on the same downstream tasks:

```bash
python -m opera.run.pretraining_scale_ablation \
  --sweep_config opera/configs/sweep_example.yaml \
  --tasks dlbcl:mortality_1y,myeloma:aki_30d \
  --checkpoints small="${BONSAI_CHECKPOINT_ROOT}/pretrain_small.ckpt",large="${BONSAI_CHECKPOINT_ROOT}/pretrain_large.ckpt" \
  --encoder_source pretrain \
  --output_dir "${BONSAI_RESULTS_ROOT}/pretraining_scale"
```

The runner tags evaluation rows with `model_family` and `pretraining_scale`,
so `opera.run.aggregate_results` can produce `pretraining_scale_summary.csv`.

## Cross-Outcome Weighting

Contrastive training computes one survival-aware loss for each outcome with
eligible patients and then applies the `cross_outcome` configuration:

```yaml
cross_outcome:
  weighter: uniform
  aggregation: macro
  class_balanced: false
  class_balanced_cap: 50.0
  class_balanced_beta: 0.9999
  class_counts: {}
```

Configured training currently supports `uniform` and `kendall`. Uniform with
macro aggregation is the explicit production setting in the checked-in
contrastive configs. The FAMO task-weighting core is present for isolated
algorithm work, but configured FAMO training fails closed because the current
Lightning path does not recompute same-batch task losses after the shared model
optimizer step. That lifecycle is required by the
[FAMO method](https://arxiv.org/abs/2306.03792) and must be implemented before
the confound panel is run. To reproduce the historical objective exactly, use:

```yaml
cross_outcome:
  weighter: kendall
  aggregation: pooled
  class_balanced: false
```

`aggregation: macro` divides the weighted objective by the number of active
outcomes in the batch. Outcomes with no eligible patients or no informative
pairs contribute zero and do not update weighter state.

Class-balanced normalization uses global positive and negative counts, not
batch prevalence. When enabled, supply `class_counts.<outcome>.positive` and
`class_counts.<outcome>.negative` for every configured outcome. Apply the same
normalization to tabular and single-task baselines used in rarity-delta
comparisons, otherwise the weighting choice becomes a model-specific
confounder.

## Gradient Conflict Diagnostic

Measure whether outcome gradients conflict on shared patient support without
running full-model backward passes:

```bash
python -m opera.diagnostics.representation_gradient_conflict \
  --config-name generated/joint_opera_full_panel \
  --checkpoints /checkpoints/epoch_01.ckpt /checkpoints/best.ckpt \
  --output-dir /results/gradient_conflict \
  --batches 16 \
  --min-overlap 8
```

Use repeated `--override key=value` arguments for Hydra overrides. The
diagnostic also writes `gradient_pair_batches.csv`, with one row per batch and
off-diagonal outcome pair. Each row contains the cosine when computable, joint
support, and whether the diagnostic overlap threshold was met.

Turn that batch-level artifact into the gradient-surgery verdict with:

```bash
python -m opera.diagnostics.conflict_verdict \
  --pair-batches "${BONSAI_RESULTS_ROOT}/gradient_conflict/00_best/gradient_pair_batches.csv" \
  --output-dir "${BONSAI_RESULTS_ROOT}/gradient_conflict/00_best/verdict" \
  --min-mean-support 8 \
  --n-bootstrap 2000
```

The verdict classifies adequately supported pairs from bootstrap intervals over
batches. Low-support pairs remain visible in the output table but are
indeterminate and do not justify surgery. Conflict clustering uses connected
components and reports a group as coherent only when the largest component is
dense and captures at least half of the significant conflict edges.

## Rarity Invariance Across Outcome Weighters

The analysis harness consumes standard evaluated result rows, not raw
contrastive encoder checkpoints. A contrastive encoder has no task prediction
head, so each weighter first needs matched downstream evaluation artifacts from
the existing evaluation pipeline.

Run the panel after Kendall, uniform, and a valid FAMO comparator have produced
held-out result rows:

```bash
python -m opera.analysis.rarity_invariance_panel \
  --results "${BONSAI_RESULTS_ROOT}/weighter_comparison" \
  --event-rates "${BONSAI_RESULTS_ROOT}/weighter_comparison/event_rates.csv" \
  --output-dir "${BONSAI_RESULTS_ROOT}/weighter_comparison/rarity_invariance" \
  --weighter-model kendall=opera_kendall \
  --weighter-model uniform=opera_uniform \
  --weighter-model famo=opera_famo \
  --baseline-model tabular_ehr \
  --evaluation-subset full \
  --n-bootstrap 2000
```

The command requires the baseline and evaluation subset explicitly. It rejects
IPI on the full subset, task-set differences, denominator mismatches, baseline
drift, and asymmetric class balancing. It computes deltas through
`build_delta_vs_baseline_table` and draws the panel through
`rarity_plots.py`.

For an IPI sensitivity analysis, use `--baseline-model ipi` together with
`--evaluation-subset ipi_complete`. This is a different patient population and
must not replace the current full-cohort tabular-EHR primary endpoint without a
locked analysis-plan change.

The current training path intentionally rejects `weighter: famo`. Final
three-weighter numbers remain blocked until the published same-batch,
post-optimizer FAMO update has been implemented and validated. The complete
input status is recorded in
`opera/configs/manifests/analysis_artifact_readiness.yaml`.
The diagnostic writes cosine and support matrices, event rates, a heatmap, and
ranked JSON summaries in one directory per checkpoint. It runs the encoder
once per batch and computes gradients only with respect to a detached
representation leaf.

## Disease-Treatment Embedding Atlas

The treatment atlas measures whether first-line regimen selection is accessible
from frozen embeddings without allowing the probe to score points merely by
recognizing the disease. It fits one regimen classifier per disease, then draws
all diseases and disease-specific regimens on one shared patient projection.

Provide one embedding artifact per representation stage. Tabular artifacts must
contain `subject_id` and numeric `embedding_*` columns. Canonical
`predictions.npz` files are also accepted when they contain non-empty
`subject_ids` and `embeddings` arrays. The metadata table must contain one row
per patient with disease, raw first-line regimen, and preferably the prospective
split. Use embeddings extracted at the shared first-line prediction origin;
post-index dose changes, completion, response, and toxicity must not enter.

```bash
python -m opera.run.treatment_embedding_atlas \
  --embeddings pretrain=/results/embeddings/pretrain.parquet \
  --embeddings dapt=/results/embeddings/dapt.parquet \
  --embeddings opera=/results/embeddings/opera.parquet \
  --metadata /data/hematology_first_line_metadata.parquet \
  --regimen_map opera/configs/treatment_regimen_groups.example.yaml \
  --disease_col disease \
  --treatment_col first_line_regimen \
  --atlas_model opera \
  --evaluation_mode held_out \
  --output_dir /results/treatment_embedding_atlas
```

Regimen normalization is disease-specific: the YAML maps raw DLBCL treatment
values independently from raw myeloma treatment values. Replace the example
strings with the locked source values before analysis. Unmapped values are
excluded unless `--keep_unmapped` is supplied.

Outputs include:

- `treatment_probe_results.csv`: held-out accessibility metrics per disease and
  embedding stage;
- `treatment_probe_predictions.csv`: patient-level probe predictions;
- `disease_treatment_atlas_coordinates.csv`: the reusable shared projection;
- `disease_treatment_atlas_centroids.csv`: disease, regimen, and joint centroids;
- `disease_treatment_geometry.csv`: ranked joint-cell cosine similarities in
  the original high-dimensional embedding space;
- `disease_treatment_atlas.png/.pdf`: disease geography, regimen geography, and
  the joint disease-regimen map;
- `treatment_probe_performance.png/.pdf`: pretrain/DAPT/OPERA probe comparison.

`evaluation_mode=auto` falls back to stratified cross-validation when an
explicit train/held-out split is unavailable, and labels those rows
`stratified_cv_exploratory`. Paper-facing treatment-accessibility claims should
use `evaluation_mode=held_out`. Probe performance describes historical
treatment-selection information in the embeddings; it is not a treatment
recommendation or a causal effect estimate.

## Patient and Vocabulary Embedding Visualizations

Use patient embeddings and vocabulary embeddings for different claims.

Patient embedding plots show the geometry of people at the prediction origin.
For outcome-facing claims, draw the projection on the held-out/test patients
used for evaluation, or make the caption explicit if the plot is descriptive
and includes train/tuning patients. For the treatment atlas, probe performance
should be held-out whenever possible, while the atlas itself may use all
eligible first-line patients as a descriptive map of disease and regimen
structure.

Vocabulary embedding plots show the geometry of model code tokens, not
patients. They are useful for asking whether the model has learned coherent
clinical code neighborhoods and which token families move during DAPT/OPERA.
They should be shown as a separate figure from patient maps unless the visual
explicitly links a patient cluster to exemplar nearest tokens.

```bash
python -m opera.run.vocabulary_embedding_atlas \
  --checkpoint pretrain=/results/checkpoints/pretrain.ckpt \
  --checkpoint dapt=/results/checkpoints/dapt.ckpt \
  --checkpoint opera=/results/checkpoints/opera.ckpt \
  --vocabulary /data/hematology/vocabulary.pt \
  --atlas_stage opera \
  --reference_stage pretrain \
  --projection umap \
  --highlight_tokens LPR3//DC833,RKKP//ann_arbor_III \
  --neighbor_tokens LPR3//DC833,RKKP//ann_arbor_III \
  --output_dir /results/vocabulary_embedding_atlas
```

Outputs include:

- `vocabulary_embeddings_<stage>.csv`: token-level embedding table per stage;
- `vocabulary_atlas_coordinates.csv`: reusable token projection coordinates;
- `vocabulary_atlas.png/.pdf`: code-token atlas colored by token family/source;
- `token_movement.csv`: token-level pretrain-to-DAPT/OPERA movement metrics;
- `token_movement.png/.pdf`: the most shifted tokens per comparator stage;
- `vocabulary_neighbors.csv`: optional nearest-neighbor table for highlighted
  query tokens.

## Hierarchical Natural-Rarity Analysis

The primary rarity analysis uses natural variation across all evaluable
cohort-outcome cells rather than treating synthetic downstream subsampling as
a complete emulation of disease rarity. It models paired OPERA-minus-tabular
performance deltas against the observed number of training events.

Prediction inputs are admitted only when they have standard `result.csv` and
`predictions.npz` artifacts. Neural artifacts may contain the wider survival
cohort, but their saved `binary_mask` is applied before fixed-horizon metrics.
Precomputed fixed-horizon predictions already have to match the canonical
eligible cohort exactly. The analysis then independently requires identical
patient IDs and labels between OPERA and its comparator.

Assemble and audit the inputs without installing PyMC:

```bash
python -m opera.run.hierarchical_rarity \
  --config opera/configs/hierarchical_rarity.yaml \
  --mode assemble
```

This writes the artifact inventory, canonical task-size table, paired deltas,
patient-bootstrap draws, and deidentified patient-overlap diagnostics. To fit
the robust Student-t hierarchy and generate the figure:

```bash
python -m pip install -e ".[bayesian]"
python -m opera.run.hierarchical_rarity --mode fit
python -m opera.run.hierarchical_rarity --mode plot
```

The model uses a penalized cubic spline on log2 training-event count, crossed
cohort/outcome/outcome-family effects, a cell effect, known per-run bootstrap
uncertainty, and an additional training-seed variance. It does not force a
monotonic rarity relationship. Variance components are omitted when only one
level is observed (for example, a core run containing one outcome family),
because a between-family variance is not identified by a single family. The
curve is an adjusted association across naturally occurring tasks, not a
causal effect of adding training events; the synthetic label-efficiency
analysis remains the intervention-style sensitivity analysis. Convergence is a hard gate by default: any
divergence or maximum R-hat above 1.01 stops publication output.

Primary outputs are:

- `paired_deltas.csv/.parquet` and `paired_bootstrap_draws.csv/.parquet`;
- `patient_overlap_summary.json` and `patient_overlap_pairs.csv/.parquet`;
- `model/posterior.nc`, exact model inputs, design metadata, and diagnostics;
- `model/posterior_curve.csv` and `model/posterior_cells.csv`;
- `figures/hierarchical_rarity_<metric>.png/.pdf/.svg` and its scatter data.

The figure distinguishes uncertainty about the population mean curve from the
predictive dispersion of a new cohort-outcome cell. Low-event test cells are
shown as hollow partial-pooling observations rather than silently removed.
Point color encodes the prespecified outcome family
(`opera/configs/generated/outcome_families.yaml`) and point shape
encodes the training cohort group, so cohort- and outcome-driven patterns
stay visually separable in one panel. Point size is not used to encode
held-out event counts: under the fixed train/val/test split, held-out events
scale near-linearly with the training-event rarity already on the x-axis, so
sizing by it would only redraw the x-position as area. Point size is therefore
fixed; filled versus hollow markers distinguish primary from partial-pooled
cells.

Cohort shapes are grouped in the legend under prespecified disease-course
headers (`opera/configs/hierarchical_rarity.yaml:figure.cohort_course_groups`,
e.g. "Aggressive course group" vs. "Indolent / chronic course group") rather
than one flat row of markers; cohort groups absent from that mapping still
render, filed under "Other cohorts". A curated set of cells is annotated with
leader lines, each tagged with why it was picked. Exact, prespecified
cohort-outcome anchors (for example DLBCL × treatment failure) take priority.
Labels are deliberately
*not* chosen by statistical extremeness (e.g. "biggest residual from the
curve") — that surfaces whichever cell is noisiest, not whichever a clinical
reader cares about. Priority order is: (1) exact clinical anchors from
`figure.highlight_cells`, followed by outcome-only fallbacks spread across
the rarity range; (2) the rarest and most data-rich evaluable cells for scale
context; (3) any remaining label budget filled with cells close to the curve
(typical cells at that information level). The x-axis uses plain training-event counts on a
log scale rather than log2-exponent tick labels. Risk-score-only survival
artifacts do not enter binary AUROC/AUPRC analyses; they require a separate
C-index analysis.

## Many-Outcome Training Guardrails

OPERA can train with sparse outcome availability: patients do not need labels
for every configured endpoint, and missing labels are represented by sentinel
values. For large endpoint panels, use the same task-balancing philosophy in
contrastive and joint fine-tuning:

- primary contrastive and joint runs should use uniform macro aggregation, so
  common endpoints do not dominate by sheer support;
- joint fine-tuning uses the shared cross-outcome weighter and can auto-fill
  train-set class counts from the configured cohort/outcome files;
- capped positive-class weighting is enabled for joint BCE by default, so rare
  positives are not washed out by many negative rows;
- one-class minibatches are skipped for joint BCE by default, preventing rare
  endpoints from contributing endless all-negative updates;
- real-rare cells should still be flagged by train/test event support before
  paper aggregation.

The production joint config mirrors the contrastive setting:

```yaml
cross_outcome:
  weighter: uniform
  aggregation: macro
  class_balanced: false
  positive_class_weighted: true
  positive_class_weight_cap: 50.0
  require_both_classes_per_batch: true
```

Use `class_balanced: true` only as a sensitivity run unless the analysis plan
explicitly wants an additional outcome-level rare-task boost on top of BCE
positive weighting.

## Manifests

Experiment manifests live under `opera/configs/manifests/`:

- `paper_core.yaml`
- `supplement.yaml`

They document the intended paper stages and expected artifacts. They are
descriptive manifests, not yet a job scheduler.
# Production outcome/cohort sweep registry

The locked production inventory lives in
`opera/configs/experiment_registry.yaml`. It is the single source of truth for
the 10 grouped cohorts, 24 fine cohorts, 87 outcomes, five IPCW horizons,
outcome families, seeds, checkpoint variants, and structural availability
rules. In particular, the three second-line-dependent outcomes are excluded
for `BL_LBL` and `HCL` (and therefore their fine cohorts); low sample size does
not otherwise remove a cohort/outcome cell.

Regenerate the executable configs after changing the registry:

```bash
python -m opera.run.generate_sweep_configs
```

This writes grouped and fine Cox configs plus grouped and fine IPCW-BCE configs
for 30, 90, 180, 365, and 730 days under `opera/configs/generated/`. Cox is the
primary survival analysis; IPCW-BCE provides the horizon-specific binary
analyses. Every non-death endpoint uses `overall_survival.parquet` as the
competing event, while overall survival itself does not.

The generated paths use three server environment variables:

```bash
export BONSAI_PROCESSED_DATA=/path/to/processed/subject_data
export BONSAI_COHORT_MEMBERSHIP=/path/to/cohort_membership.parquet
export BONSAI_OUTCOMES_DIR=/path/to/outcomes
export BONSAI_CHECKPOINT_ROOT=/path/to/checkpoints
export BONSAI_RESULTS_ROOT=/path/to/results
```

`opera_per_grouped` is intentionally not part of the primary model ladder. A
separate contrastive encoder per grouped disease family is a useful future
supplementary ablation (testing broad versus parent-group adaptation), but it
requires ten additional adaptation runs and should be added only after the
main full-panel experiment is stable.

For example:

```bash
python -m opera.run.sweep --config opera/configs/generated/grouped_cox.yaml
python -m opera.run.sweep --config opera/configs/generated/grouped_ipcw_365d.yaml
```
