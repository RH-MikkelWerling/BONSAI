# BONSAI / OPERA

BONSAI contains the core EHR data, outcome, pretraining, and finetuning
pipeline. OPERA adds hematology-specific adaptation, joint finetuning,
evaluation, aggregation, and paper experiment tooling.

## Repository Layout

- `bonsai/`: shared data processing, datasets, model modules, and training entry points
- `opera/`: OPERA adaptation, joint finetuning, evaluation, plotting, and aggregation
- `bonsai/configs/`: BONSAI data creation, training, and finetuning configs
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
  ckpt_path=/ckpts/opera/best.ckpt \
  dataset=dlbcl \
  outcome=mortality_1y \
  output_dir=./results/dlbcl/mortality_1y/opera
```

Joint-model evaluation for one cohort-outcome cell:

```bash
python -m opera.run.evaluate_joint \
  ckpt_path=/ckpts/joint_finetune/best.ckpt \
  dataset=dlbcl \
  outcome=mortality_1y \
  outcome_name=mortality_1y \
  output_dir=./results/dlbcl/mortality_1y/opera_joint
```

## Prospective Splits And Follow-Up

Outcome creation supports prospective split definitions through
`prospective_split` config blocks. Generated outcome files can be checked with:

```bash
python -m bonsai.run.validate_splits \
  --outcome /data/dlbcl/outcomes/mortality.parquet \
  --train_end 2023-12-31 \
  --val_start 2023-07-01 \
  --val_end 2023-12-31 \
  --test_start 2024-01-01 \
  --fail_on_error
```

Bounded-window validation and test labels require sufficient follow-up by
default. Finetuning runs write `label_split_summary.csv` with retained subjects,
events, prevalence, and insufficient-follow-up exclusions by split.
Validation is used for tuning/model selection and should be an explicit
pre-prospective period; the prospective held-out test period remains separate.

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

External/tabular baselines can be evaluated from prediction files with
`python -m opera.run.evaluate_predictions`.
Locked tabular feature matrices can be converted into those prediction files
with `python -m opera.run.train_tabular_baselines`. TabPFN is supported as an
optional baseline via `python -m pip install -e ".[tabpfn]"`.

## Experiment Manifests

Paper-oriented manifests live in `opera/configs/manifests/`:

- `paper_core.yaml`
- `supplement.yaml`

They document intended experiment bundles and expected artifacts. They are not a
job scheduler.

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
