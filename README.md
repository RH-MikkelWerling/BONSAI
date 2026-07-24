# BONSAI / OPERA

This branch keeps upstream BONSAI as the foundation-model and generic EHR
training layer. OPERA is an additive hematology research layer: it owns
outcome-aware adaptation, survival/IPCW training, prospective evaluation,
multi-cohort experiments, and paper analyses. Generic behavior stays in
`bonsai/` only when OPERA genuinely requires a reusable BONSAI capability.

## Repository Layout

- `bonsai/`: shared data processing, datasets, model modules, and training entry points
- `opera/`: OPERA adaptation, joint finetuning, evaluation, plotting, and aggregation
- `configs/`: upstream-compatible BONSAI data creation, training, and finetuning configs
- `opera/configs/`: OPERA finetuning, evaluation, sweep, and manifest configs
- `tests/`: unit tests for dataset immutability, checkpoint metadata, split logic, metrics, aggregation, and rarity helpers
- `OPERA_EXPERIMENTS.md`: paper workflow notes and experiment commands
- `OPERA_REPOSITORY_GUIDE.md`: practical guide to repository functionality and the press-go checklist

## Setup

Create a virtual environment and install the repo in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy `.env.example` to a local `.env` and adapt the artifact paths. `.env` is
intentionally ignored and must not contain secrets.

Optional research features are installed separately:

```bash
python -m pip install -e ".[tabular]"       # XGBoost
python -m pip install -e ".[visualization]" # PaCMAP, UMAP, LOWESS
python -m pip install -e ".[survival]"      # lifelines
python -m pip install -e ".[retrieval]"     # FAISS
python -m pip install -e ".[tabpfn]"        # TabPFN
python -m pip install -e ".[flash_attn]"    # FlashAttention 2 (Linux/CUDA)
```

FlashAttention is the default backend for BONSAI and OPERA training. The
encoder packs valid tokens before every transformer stack, so dynamically
padded batches use the variable-length FlashAttention kernel without allowing
padding to influence representations. CPU development and unsupported GPUs can
select the fully tested fallback with `model.attn_type=sdpa`.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The package configuration installs both `bonsai` and `opera`, so fresh
checkouts can import and run both namespaces without path hacks.

## Tests

Run the full test suite with:

```bash
python -m pytest tests
```

Convenience targets are also available:

```bash
make test
make smoke
make readiness
make rarity-demo
```

A quick syntax check that does not require all optional runtime dependencies:

```bash
python -m compileall bonsai opera tests
```

The complete local quality gate is:

```bash
ruff format --check bonsai opera tests
ruff check bonsai opera tests
coverage run -m pytest tests -q
coverage report
```

## Architecture

```text
ehr2meds MEDS cohort (physical 90/10 SSL partitions)
    -> BONSAI feature creation and tokenization
    -> subject_data_{train,tuning}.pt
    -> general pretraining / hematology DAPT / OPERA contrastive adaptation
    -> pool physical subject files
    -> select outcome train/tuning/held_out by the temporal manifest
    -> per-task, survival, hybrid, or joint finetuning at index_date
    -> prediction artifacts and validated result.jsonl rows
    -> aggregation, significance analysis, and paper figures
```

The physical ehr2meds split and the downstream outcome split are deliberately
different contracts. The physical 90/10 split controls self-supervised fitting
and numeric metadata; OPERA pools those files and uses outcome parquets as the
sole authority for prospective train/tuning/held-out membership. All supervised
model inputs end at `index_date`. `censor_date` is reserved for follow-up
eligibility and time-to-event calculations.

## First Production Run Contract

For the complete server setup, artifact placement, smoke tests, and staged
launch procedure, follow [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md).

Before training on registry data, require all of the following:

- ehr2meds uses a deterministic 90/10 subject split for SSL train/validation.
- Numeric normalization and bins are fitted only on the 90% SSL-training
  subjects and events with `time < 2022-01-01`.
- BONSAI data creation sets
  `vocabulary_cutoff_date={year:2022,month:1,day:1}`; codes first observed at
  or after the boundary map to `[UNK]`.
- General pretraining sets the same exclusive `training.cutoff_date` and uses
  the 10% SSL validation split for checkpoint selection and early stopping.
- Outcome parquets pass `opera/configs/manifests/temporal_split.yaml`.
- Tabular features are generated at the same patient-specific `index_date` as
  neural inputs and contain no later measurements.

The ehr2meds aggregate-numeric cutoff is an upstream prerequisite; this
repository cannot infer it safely from annotated event shards without its
metadata sidecar. Do not start a production run from numeric metadata that
lacks fitting-cutoff and split provenance.

## Core Commands

BONSAI pretraining and finetuning entry points:

```bash
python -m bonsai.run.pretrain
python -m bonsai.run.train
python -m bonsai.run.finetune
```

OPERA adaptation and finetuning entry points:

```bash
python -m opera.run.dapt
python -m opera.run.mol
python -m opera.run.contrastive
python -m opera.run.finetune
python -m opera.run.joint_finetune
```

Pretraining/DAPT configs support long histories with mixed-window truncation:

```yaml
training:
  max_len: 8192
  truncation_strategy: mixed_window
  validation_truncation_strategy: tail
  tail_window_probability: 0.5
```

Background tokens are always preserved. Training alternates between recent
history and random contiguous clinical windows; validation stays deterministic.

Standalone evaluation:

```bash
python -m opera.run.evaluate \
  ckpt_path="${BONSAI_CHECKPOINT_ROOT}/contrastive/best.ckpt" \
  dataset=dlbcl \
  outcome=mortality_1y \
  output_dir="${BONSAI_RESULTS_ROOT}/dlbcl/mortality_1y/opera"
```

Joint-model evaluation for one cohort-outcome cell:

```bash
python -m opera.run.evaluate_joint \
  ckpt_path="${BONSAI_CHECKPOINT_ROOT}/joint_finetune/best.ckpt" \
  dataset=dlbcl \
  outcome=mortality_1y \
  outcome_name=mortality_1y \
  output_dir="${BONSAI_RESULTS_ROOT}/dlbcl/mortality_1y/opera_joint"
```

## Prospective Splits And Follow-Up

Outcome creation supports prospective split definitions through
`prospective_split` config blocks. The OPERA paper split lives in
`opera/configs/manifests/temporal_split.yaml` and can be referenced from outcome
configs with `prospective_split.contract`. Generated outcome files can be
checked with:

```bash
python -m bonsai.run.validate_splits \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --contract opera/configs/manifests/temporal_split.yaml \
  --fail_on_error
```

Before launching a stage, validate split labels, coverage across the pooled
physical subject-data files, DAPT inputs, and optional embedding stores:

```bash
python -m opera.run.validate_split_contract \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --subject_data ssl_train=/data/dlbcl/subject_data_train.pt \
  --subject_data ssl_validation=/data/dlbcl/subject_data_tuning.pt \
  --fail_on_error
```

Bounded-window validation and test labels require sufficient follow-up by
default. Finetuning runs write `label_split_summary.csv` with retained subjects,
events, prevalence, and insufficient-follow-up exclusions by split.
The locked paper contract is train through 2021, tuning/model selection during
2022-2023, and one-time held-out evaluation from 2024 onward. General pretraining is
a separate random 90/10 subject split and must use only events strictly before
2022. Numeric metadata and vocabulary must be fitted on the 90% SSL-training
view before being frozen and applied elsewhere.

## Rarity Analyses

The code distinguishes two regimes:

- `synthetic`: common disease tasks with artificially reduced training labels
- `real`: genuinely small disease cohorts or small cohort-outcome cells

Synthetic rarity / label scarcity:

```bash
python -m opera.run.label_efficiency \
  --sweep_config opera/configs/sweep_example.yaml \
  --tasks dlbcl:mortality_1y,myeloma:aki_30d \
  --fractions 0.01,0.02,0.05,0.1,0.25,0.5,1.0 \
  --baseline_model tabular_ehr
```

Real rare-cohort benchmarks are configured separately in:

```text
opera/configs/rare_cohort_benchmark.yaml
```

Evaluation result rows include `rarity_mode`, `rarity_tier`, split sizes,
event counts, and split prevalence fields. Aggregate rarity outputs with:

```bash
python -m opera.run.aggregate_results \
  --results_dir ./results \
  --output_dir ./results/aggregated \
  --baseline tabular_ehr \
  --rarity_plots
```

This writes separate synthetic and real rarity delta tables so the two regimes
are not pooled unless an analysis explicitly does so.

Outcome configs can reuse one event-time parquet for multiple windows:

```yaml
mortality_1y:
  outcome_file: mortality.parquet
  n_hours_end_include: 8760
mortality_2y:
  outcome_file: mortality.parquet
  n_hours_end_include: 17520
```

`censor_date` is the end of observed follow-up, not only a fallback field for
non-events. A single event-time parquet can therefore support survival metrics
and multiple horizon-classification tasks.

Optional outcome-specific `eligibility_file` sidecars are applied before labels are
constructed. This lets a patient contribute to mortality while being masked
for an unascertainable laboratory endpoint. Fixed-horizon binary training
requires complete follow-up for event-free controls; configured death tables
encode death inside the risk window as competing `event=2`.

Sidecars may additionally provide `ascertainment_eligible`. Fixed-horizon
models use final `eligible` rows, while Cox, IPCW, and survival-contrastive
models retain ascertainable early-censored rows and use their observed risk
time. Events after `censor_date` are never treated as observed outcomes, and an
earlier competing event takes precedence over a later primary event.

Event-aware multi-outcome batches contain unique patients. Sparse endpoint
quotas are capped at the available unique evidence, outcome-focused sampling
requires configurable minimum unique event/patient counts, and repeat use over
an epoch is diversity-penalized. Same-subject contrastive pairs are masked as a
defensive backstop.

External/tabular baselines can be evaluated from prediction files with
`python -m opera.run.evaluate_predictions`.
Locked tabular feature matrices can be converted into those prediction files
with `python -m opera.run.train_tabular_baselines`. TabPFN is supported as an
optional baseline via `python -m pip install -e ".[tabpfn]"`.

Tabular estimators are fitted only on `train`. Hyperparameters are selected on
`tuning` by default, tuning predictions are saved as a separate OOT artifact,
and `held_out` is never folded back into fitting or selection. In low-n/high-p
cells the candidates use elastic-net logistic regression and shallow,
column-subsampled, regularized XGBoost. Reproduce the supporting simulation
with `python -m opera.run.simulate_tabular_low_n --output_dir ...`.

Natural rarity is analysed from the patient-level prediction artifacts with a
measurement-error-aware Bayesian spline hierarchy. The workflow requires exact
model/comparator patient and label parity, uses the fixed-horizon eligibility
mask already saved by neural evaluation, and partially pools cohort-outcome
deltas across cohorts and outcomes. Assemble inputs without optional Bayesian
packages, then fit and plot with:

```bash
python -m opera.run.hierarchical_rarity --mode assemble
python -m pip install -e ".[bayesian]"
python -m opera.run.hierarchical_rarity --mode fit
python -m opera.run.hierarchical_rarity --mode plot
```

The default configuration is `opera/configs/hierarchical_rarity.yaml`.

## Experiment Manifests

Paper-oriented manifests live in `opera/configs/manifests/`:

- `paper_core.yaml`
- `supplement.yaml`

They document intended experiment bundles and expected artifacts. Executable
sweep YAMLs are generated from `opera/configs/experiment_registry.yaml`:

```bash
python -m opera.run.generate_sweep_configs
python -m opera.run.generate_outcome_transfer_configs
```

Run `python -m opera.run.check_readiness --help` before production submission;
the generated configs are execution artifacts, while the registry and manifests
are the maintained source of truth.

## Upstream BONSAI Policy

`upstream/main` is the reference for the `bonsai/` package. Before a production
cycle, fetch upstream and review `git diff upstream/main...HEAD -- bonsai configs`.
OPERA-specific behavior belongs under `opera/`; changes under `bonsai/` should
be small, tested, and suitable for upstreaming. Generated results, local tool
settings, test caches, checkpoints, and one-off audit reports are not repository
source and must remain untracked.

## Citation

If you use BONSAI in your research, cite:

```bibtex
@article{Montgomery2025,
  author = {Montgomery, A. and others},
  title = {BONSAI: A framework for processing and analysing {E}lectronic {H}ealth {R}ecords ({EHR}) data using transformer-based models},
  journal = {Journal of Open Source Software},
  volume = {10},
  number = {114},
  pages = {8869},
  year = {2025},
  doi = {10.21105/joss.08869}
}
```
