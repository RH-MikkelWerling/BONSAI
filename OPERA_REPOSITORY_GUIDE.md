# OPERA Repository Guide

This guide is the practical map for running and auditing the BONSAI/OPERA
codebase. It focuses on what each part does, how experiments connect, and what
should exist before starting a full paper run.

## Mental Model

The repository has two layers:

- `bonsai`: shared EHR sequence processing, outcome creation, pretraining, and
  ordinary finetuning.
- `opera`: hematology-specific adaptation, contrastive learning, joint
  finetuning, evaluation, plotting, and paper aggregation.

Most workflows follow this shape:

1. Create tokenized subject data and outcome parquet files.
2. Train or load an upstream encoder checkpoint.
3. Adapt the encoder if needed.
4. Finetune per task or jointly.
5. Evaluate every cohort-outcome-model cell.
6. Aggregate `result.jsonl` artifacts into paper tables and plots.

## Fresh Checkout Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests
```

The package config installs both `bonsai` and `opera`.

## Data And Outcome Creation

Use `bonsai.run.create_data` for tokenized subject data and
`bonsai.run.create_outcome` for outcome parquet files. Prefer one event-time
outcome parquet per clinical endpoint, then define look-forward windows in the
sweep config:

```yaml
outcomes:
  mortality_1y:
    outcome_file: mortality.parquet
    n_hours_end_include: 8760
  mortality_2y:
    outcome_file: mortality.parquet
    n_hours_end_include: 17520
```

The task name (`mortality_1y`) is used for output directories and result rows;
`outcome_file` is the shared source parquet.

Outcome rows should contain an event time and a censoring time. `censor_date`
is the end of observed follow-up and may be present for both event and non-event
patients. For horizon classification, patients without an event inside the
window are binary-eligible only when their censoring time reaches the window
end. For survival metrics, the same event-time file is used directly, so there
is no need to create separate outcome parquet files for every look-forward
window.

Prospective paper splits are configured with a `prospective_split` block. The
intended paper shape is:

- train: pre-2024 data
- validation: explicit pre-2024 validation period
- test: post-2023 prospective held-out period

Validation is the tuning/model-selection split. It should sit before the
prospective test period, often as a late pre-2024 calendar slice. A config can
therefore have `train_end: 2023-12-31` and `val_end: 2023-12-31` when the
training construction excludes the validation period via `val_start`/`val_end`;
the split validator is the source of truth for checking that no subject/index
date is assigned to more than one split.

Validate generated splits:

```bash
python -m bonsai.run.validate_splits \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --train_end 2023-12-31 \
  --val_start 2023-07-01 \
  --val_end 2023-12-31 \
  --test_start 2024-01-01 \
  --fail_on_error
```

Finetuning entry points write `label_split_summary.csv` with subject retention,
event counts, prevalence, and insufficient-follow-up exclusions.

## Training Stages

The canonical stage names are:

- `general_denmark_pretraining`
- `hematology_domain_adaptation`
- `non_contrastive_hematology_adaptation`
- `opera_contrastive_adaptation`
- `per_task_finetuning`
- `joint_finetuning`

Entry points:

```bash
python -m bonsai.run.pretrain
python -m opera.run.dapt
python -m opera.run.mol
python -m opera.run.contrastive
python -m opera.run.contrastive_multicohort
python -m opera.run.finetune
python -m opera.run.joint_finetune
```

Each training run should save checkpoint metadata and a sidecar
`checkpoint_metadata.json` so the lineage of a final checkpoint can be traced.
Checkpoint paths in manifests are placeholders until the corresponding upstream
training stage has actually been run. The normal handoff is to train a stage,
take its emitted `best.ckpt`, and place that concrete path into the next config.

OPERA imports BONSAI architecture and data helpers through
`opera.compat.bonsai` where practical. If a collaborator updates BONSAI and
moves a shared symbol, prefer repairing that compatibility adapter first rather
than patching every OPERA script separately.

In OPERA contrastive training, the frozen DAPT/BONSAI embedding store is a
similarity prior, not the whole learning signal. Per-outcome survival times and
censoring define the primary pair weights; DAPT cosine similarity only
multiplies those weights when `dapt_embedding_store` is provided. Pairs with
missing DAPT embeddings are neutral, and training logs `dapt/weight_*` plus
`dapt/coverage` so you can verify whether the prior is active or overly strong.
Set `model.dapt_lambda_floor=1.0` to ablate the prior while keeping the
survival contrastive objective unchanged.

## Long Sequences

Pretraining and DAPT support mixed-window truncation for patients whose records
exceed `training.max_len`:

```yaml
training:
  max_len: 8192
  truncation_strategy: mixed_window
  validation_truncation_strategy: tail
  tail_window_probability: 0.5
```

All strategies preserve the leading background tokens such as sex and date of
birth-derived age features. `tail` keeps the most recent clinical history.
`random_window` samples a contiguous chronological clinical window. `mixed_window`
uses the tail window with `tail_window_probability` and otherwise samples a
random window. Validation remains deterministic with `tail`.

For autoregressive pretraining, the code masks the artificial target at the
background-to-window boundary whenever the selected clinical window starts in
the middle of a patient history. This keeps real transitions inside the window
without teaching a fake jump from demographics to a later clinical event.

## Evaluation

Single-task model evaluation:

```bash
python -m opera.run.evaluate \
  ckpt_path=/ckpts/opera/best.ckpt \
  dataset=dlbcl \
  outcome=mortality_1y \
  output_dir=./results/dlbcl/mortality_1y/opera
```

Joint model evaluation for one cell:

```bash
python -m opera.run.evaluate_joint \
  ckpt_path=/ckpts/joint_finetune/best.ckpt \
  dataset=dlbcl \
  outcome=mortality_1y \
  outcome_name=mortality_1y \
  output_dir=./results/dlbcl/mortality_1y/opera_joint
```

Evaluation artifacts include:

- `metrics.json`
- `result.csv`
- `result.jsonl`
- `predictions.npz`
- `threshold_sweep.csv`
- `decision_curve.csv`
- `bootstrap_ci.csv`
- `high_risk_enrichment.csv`
- optional `subgroup_metrics.csv`
- plots under `plots/`

The `result.jsonl` files are the canonical aggregation input.

## Main Paper Comparisons

The core comparison should include:

- base pretrained encoder
- hematology DAPT
- non-contrastive hematology adaptation, via MOL
- OPERA contrastive adaptation
- OPERA joint finetuning
- tabular baselines, ideally including the strongest available tabular model

The example sweep config is `opera/configs/sweep_example.yaml`.

The sweep runner crosses every configured cohort with every configured outcome
task and model variant. To run beyond DLBCL/treatment failure, add the cohort
directories and all endpoint tasks to the same config; the scripts will produce
one output cell per cohort-outcome-model combination.

The foundation-model variants include both full outcome finetuning and frozen
linear probes. Linear-probe variants set `model.freeze_encoder=true` and
`model.head_type=linear_probe`, then train only a single linear classifier over
masked mean-pooled frozen encoder embeddings. This is the representation-quality
test after national pretraining, hematology domain adaptation, MOL, and OPERA
contrastive adaptation. Full-finetune variants remain the
deployable-performance test.

Tabular baselines are most reproducible as prediction files evaluated through
the same metric suite:

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

Prediction files need `subject_id` and `probability`. They may also include
`label`, `time_days`, and `event`; if `label` is absent, labels are derived from
the outcome parquet and window. Supplying predictions rather than precomputed
JSON keeps tabular baselines in the same calibration, decision-curve,
bootstrap-CI, and high-risk-enrichment evaluation pipeline as OPERA.

### Tabular Feature Matrix Baselines

For paper baselines, prefer a locked feature matrix per cohort/model family and
train tabular models with `opera.run.train_tabular_baselines`. This makes the
baseline contract explicit:

```text
feature matrix + outcome parquet -> prediction CSV -> evaluate_predictions
```

Feature matrices may be CSV or parquet and must contain one row per patient:

```text
subject_id                  required stable patient identifier
<feature columns>           numeric, boolean, categorical, or missing
```

Reserved columns are excluded automatically when present: `subject_id`, `split`,
`label`, `target`, `probability`, `time_days`, and `event`. Use
`--exclude_columns` for deliberately excluded fields such as leakage-prone
post-index variables or IPI columns when training the EHR-only baseline.

Each run writes `{cohort}_{outcome}_feature_contract.json` before fitting. The
contract fails on duplicate `subject_id` rows, reserved columns selected as
features, missing requested columns, empty feature sets, all-missing predictors
unless explicitly allowed, and optional `--max_missing_fraction` violations. Use
`--validate_only` on the offline server to check a locked matrix before
launching expensive baseline runs.

Train XGBoost from a locked EHR-only matrix:

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

For the horizon-specific survival-aware analogue to OPERA's `ipcw_bce` mode,
use IPCW-weighted tabular training:

```bash
python -m opera.run.train_tabular_baselines \
  --features /data/dlbcl/features/tabular_ehr.parquet \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --cohort dlbcl \
  --outcome_name mortality_1y \
  --model_prefix tabular_ehr \
  --models xgboost_ipcw_bce \
  --n_hours_end_include 8760 \
  --output_dir /results/tabular
```

Supported training models are `logistic`, `xgboost`, `logistic_ipcw_bce`,
`xgboost_ipcw_bce`, and optional `tabpfn`. The IPCW variants train on all
usable training patients with censoring weights and still write ordinary
`subject_id`, `probability` prediction files, so downstream sweep/evaluation
comparisons remain identical. Tabular Cox is intentionally not part of this
contract because it emits relative risk scores rather than calibrated horizon
probabilities.

For the RKKP-enriched tabular baseline, point `--features` at the corresponding
feature matrix and set `--model_prefix tabular_rkkp`. The runner writes
standard prediction CSVs, feature-importance files where available, and metadata
with feature counts, split sizes, event counts, and seed.

Missingness is handled explicitly. Numeric features are median-imputed with
missingness indicators; categorical features treat missingness as its own
category. Each run writes `{cohort}_{outcome}_feature_missingness.csv` so
feature availability can be reviewed by cohort, split, and endpoint.

TabPFN is available as an optional baseline when the environment supports it:

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

TabPFN is not included in `--models all` because it has tighter environment,
memory, and feature-count constraints than XGBoost. The runner limits TabPFN to
a stable subset of features by observedness and simple signal score; tune the
feature and row caps in the locked paper environment.

Fine-Gray competing-risk models are not implemented in this Python pipeline.
The current survival evaluation is cause-specific IPCW. That limitation should
be stated in the manuscript; the practical mitigation is to report competing
event counts and avoid over-claiming calibrated cumulative incidence for
non-fatal endpoints.

### IPI-Complete Credibility Subset

IPI-family scores are precomputed clinical registry scores joined into
`population_full.csv`. They are ordinal scores, not EHR-sequence model outputs,
and the rank-normalized IPI values used by `evaluate_predictions` should not be
interpreted as calibrated event probabilities.

The sweep therefore uses two evaluation subsets:

- `evaluation_subset: full` for the primary OPERA, DAPT/MOL, and tabular-EHR
  comparisons across all evaluable patients.
- `evaluation_subset: ipi_complete` for the standalone clinical-standard
  credibility check.

For cohorts with `ipi_score_col`, the sweep computes IPI coverage on held-out
patients. Coverage below 50% skips IPI rows. Coverage at or above 50% evaluates
IPI only on patients with non-null IPI scores and tags rows with
`ipi_coverage`; OPERA and tabular predictions used in the IPI figure should be
restricted to the same subject IDs. Aggregate rarity and dot-matrix figures
filter to `evaluation_subset == "full"` so they never mix patient populations.

Run a sweep:

```bash
python -m opera.run.sweep \
  --config opera/configs/sweep_example.yaml
```

Run a preflight check before launching a long sweep:

```bash
python -m opera.run.check_readiness \
  --config opera/configs/sweep_example.yaml \
  --fail_on_issue
```

The sweep runner now writes `result.jsonl` for precomputed tabular/IPI baselines
as well as model evaluations, so downstream delta aggregation can see the
baseline rows.

Subprocess calls from the sweep write full command, stdout, stderr, and status
files under each cell's `logs/` directory. The sweep also writes
`sweep_cell_status.csv` and `sweep_cell_status.jsonl` with one row per cell
stage. By default the sweep records failures and continues so a long server run
can finish other cells; pass `--fail-fast` when debugging a new config.

Aggregation preserves raw rows in `all_results.csv`, then writes
`paper_results.csv` after applying the paper-safe default filter:
`evaluation_subset == "full"` and no IPI rows. It also writes
`aggregation_validation.csv` when raw results contain mixed evaluation subsets,
missing subset tags, IPI rows, or duplicate aggregate keys. Use
`--evaluation_subset ipi_complete --include_ipi` only for the standalone IPI
credibility table.

## Rarity Analyses

The code explicitly separates two scientific regimes:

- `synthetic`: common disease tasks with artificially reduced training labels
- `real`: genuinely small disease cohorts or small cohort-outcome cells

Primary rarity comparison metric:

```text
delta_auroc_vs_baseline = auroc(model) - auroc(baseline)
```

Raw AUROC is still stored, but rarity plots and rarity summaries emphasize
delta ROC-AUC versus a named baseline.

Synthetic rarity / label scarcity:

```bash
python -m opera.run.label_efficiency \
  --sweep_config opera/configs/sweep_example.yaml \
  --tasks dlbcl:mortality_1y,myeloma:aki_30d \
  --fractions 0.01,0.02,0.05,0.1,0.25,0.5,1.0 \
  --seeds 42,43,44 \
  --baseline_model tabular_ehr
```

Real rarity benchmark config:

```text
opera/configs/rare_cohort_benchmark.yaml
```

Fill that config with the locked rare disease cohorts or rare
cohort-outcome cells before the full paper run.

Aggregate rarity outputs:

```bash
python -m opera.run.aggregate_results \
  --results_dir ./results \
  --output_dir ./results/aggregated \
  --baseline tabular_ehr \
  --rarity_plots \
  --min_train_events 5 \
  --min_test_events 5
```

Expected rarity exports:

- `synthetic_rarity_task_level.csv`
- `synthetic_rarity_pooled.csv`
- `real_rarity_task_level.csv`
- `real_rarity_pooled.csv`
- `real_rarity_pooled_main.csv` when event thresholds are used
- `synthetic_rarity_delta.png`
- `real_rarity_delta.png`
- `rarity_delta_combined.png`

The minimum-event flags mark unstable real rare-cohort rows as
`supplement_only`; they do not remove raw results.
Figure helpers save both raster `.png` files and companion vector `.pdf` files.

## Other Paper Ablations

Joint vs per-cohort:

```bash
python -m opera.run.joint_vs_per_cohort \
  --results_dir ./results \
  --output_dir ./results/joint_vs_per_cohort \
  --joint_model joint \
  --per_cohort_model per_cohort
```

Pretraining scale:

```bash
python -m opera.run.pretraining_scale_ablation \
  --sweep_config opera/configs/sweep_example.yaml \
  --tasks dlbcl:mortality_1y,myeloma:aki_30d \
  --checkpoints small=/ckpts/pretrain_small.ckpt,large=/ckpts/pretrain_large.ckpt \
  --encoder_source pretrain \
  --output_dir ./results/pretraining_scale
```

Subgroup robustness:

```yaml
paths:
  subgroups: /path/to/subgroups.csv
subgroups:
  columns: [age_group, sex, calendar_period, cohort]
```

Prediction-file baselines support the same subgroup path and columns through
`--subgroups` and `--subgroup_columns`.

Aggregating result directories also collects `subgroup_metrics.csv` files and,
when both models are present for the same subgroup cell, writes
`subgroup_delta.csv` with comparator-minus-baseline columns such as
`delta_auroc_vs_tabular_ehr`. By default this compares OPERA against
`tabular_ehr`; override with `--subgroup_baseline` and
`--subgroup_comparator`.

Calibration and clinical utility are emitted during ordinary evaluation through
`metrics.json`, calibration plots, `threshold_sweep.csv`,
`decision_curve.csv`, `bootstrap_ci.csv`, and `high_risk_enrichment.csv`.

## Press-Go Checklist

Before launching the full experiment sweep:

- Outcome parquet files exist for every cohort-outcome cell.
- Prospective split summaries have been generated and validated.
- Validation/test follow-up rules match the paper protocol.
- The model variants in `sweep_example.yaml` point to real checkpoints.
- `opera.run.check_readiness --fail_on_issue` passes for the sweep config.
- The rare-cohort benchmark config has real locked rare cells, not placeholders.
- The named rarity baseline exists as `result.jsonl` rows or precomputed files.
- Each final checkpoint has `checkpoint_metadata.json`.
- `python -m pytest tests` passes in a fresh editable install.
- `python -m compileall bonsai opera tests` passes.

After a sweep:

- Run `opera.run.aggregate_results`.
- Check `all_results.csv` for missing model families or missing `rarity_mode`.
- Check `label_split_summary.csv` for exclusion counts and prevalence.
- Check rarity tables to confirm `synthetic` and `real` were not mixed.
- Check calibration, subgroup, and threshold artifacts for key tasks.

## Where To Look

- Dataset immutability: `bonsai/functional/subject_data.py` and dataset classes
- Follow-up and prospective splits: `bonsai/functional/outcomes.py`
- Checkpoint reconstruction and sidecars: `bonsai/functional/checkpointing.py`
- Evaluation metrics: `opera/evaluation/metrics.py`
- Result schema: `opera/evaluation/results_schema.py`
- Aggregation: `opera/evaluation/aggregation.py`
- Rarity plots: `opera/visualization/rarity_plots.py`
- Experiment notes: `OPERA_EXPERIMENTS.md`
