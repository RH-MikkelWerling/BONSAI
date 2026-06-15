# OPERA Experiment Workflow

This note documents the reproducibility hooks added for the OPERA paper setup.

## Prospective Splits

Outcome creation can optionally assign splits from `index_date` instead of
trusting input directory names. Add a `prospective_split` block to an outcome
creation config:

```yaml
prospective_split:
  date_col: index_date
  train_end: "2023-12-31"
  val_start: "2023-07-01"
  val_end: "2023-12-31"
  test_start: "2024-01-01"
  test_end: null
  train_key: train
  val_key: tuning
  test_key: held_out
```

`bonsai.run.create_outcome` writes a split summary CSV next to each outcome
parquet. The summary records split counts, index-date ranges, and observed
events.

Validate a generated outcome file before training:

```bash
python -m bonsai.run.validate_splits \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --train_end 2023-12-31 \
  --val_start 2023-07-01 \
  --val_end 2023-12-31 \
  --test_start 2024-01-01 \
  --fail_on_error
```

The validator reports subject overlap across splits and date-boundary
violations for prospective setups.

Finetuning entry points write `label_split_summary.csv` to the run directory.
This file records raw subjects, retained labelled subjects, label prevalence,
and insufficient-follow-up exclusions by split.

## Follow-Up Eligibility

Bounded-window binary labels support explicit minimum follow-up requirements by
split:

```yaml
labels:
  require_min_followup_train: false
  require_min_followup_val: true
  require_min_followup_test: true
```

Validation and test default to full follow-up for bounded windows. Training
defaults to permissive to preserve censor-aware training setups.

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
  --rarity_plots
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
  --config-name leukemia_contrastive \
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

## Manifests

Experiment manifests live under `opera/configs/manifests/`:

- `paper_core.yaml`
- `supplement.yaml`

They document the intended paper stages and expected artifacts. They are
descriptive manifests, not yet a job scheduler.
