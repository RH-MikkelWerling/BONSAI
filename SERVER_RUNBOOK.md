# OPERA Server Runbook

This is the operational checklist for the first production run. Run commands
from the repository root. Stop at the first failed gate; do not submit the full
sweep and hope that later cells recover.

## 0. Pin both repositories and lock the ehr2meds pipeline

The numeric-value work previously reviewed on ehr2meds'
`feature/numeric_values` branch is now present on `main`; the old feature ref
may no longer exist. Record exact commits rather than relying on a moving
branch:

```bash
git -C /project/ehr2meds checkout main
git -C /project/ehr2meds pull --ff-only
git -C /project/ehr2meds rev-parse HEAD
git -C /project/BONSAI rev-parse HEAD
```

Do not run ehr2meds' unmodified default split. Its example configuration uses
an 80/10/10 physical split. The production DALY-CARE pipeline must instead
resolve to:

```yaml
split_and_shard_subjects:
  split_fracs:
    train: 0.9
    tuning: 0.1
```

There is deliberately no physical `held_out` EHR partition: the two random
physical partitions together contain every patient. Prospective
train/tuning/held-out membership is assigned later from each patient's
`index_date`.

Before launching ehr2meds, inspect its fully resolved pipeline and require:

- `aggregate_numeric_metadata` fits only physical `train`;
- its exclusive event-time cutoff is `2022-01-01`;
- `annotate_numeric_values` reuses that fitted metadata for every partition;
- `join_numeric_bins` runs after annotation and produces `CODE//bin_k`;
- finalized shards remain ordered by subject, time, and `row_idx`;
- numeric metadata, resolved config, split manifest, and commit SHA are
  archived beside the MEDS output.

## 1. Required inputs

ehr2meds must provide one ready-to-ingest cohort root:

```text
$EHR2MEDS_OUTPUT/
├── data/
│   ├── train/*.parquet
│   └── tuning/*.parquet
└── metadata/
```

The physical train/tuning partition is the deterministic 90/10 SSL split.
Numeric normalization and joined-bin metadata must have been fitted on physical
train subjects using only events before `2022-01-01`.

Before copying anything, confirm there is no `data/held_out/` directory from an
accidental default ehr2meds run. If it exists, stop and correct the ehr2meds
split configuration; otherwise some prospective held-out patients will be
missing from OPERA's pooled subject data.

The external outcome pipeline must provide:

```text
population_full.csv
outcomes/
├── overall_survival.parquet
├── anemia_g2plus.parquet
└── ... one file for every configured endpoint
```

`population_full.csv` contains one row per patient and at minimum:

```text
subject_id,cohort_grouped,cohort_fine
```

Each outcome parquet contains one row per eligible patient and:

```text
subject_id,split,index_date,outcome_date,censor_date
```

The prospective values are:

- `train`: index date through 2021-12-31;
- `tuning`: index date from 2022-01-01 through 2023-12-31;
- `held_out`: index date from 2024-01-01 onward.

An absent `outcome_date` must mean an eligible censored/non-event observation.
Optional eligibility sidecars are needed only when an outcome parquet is not
already the final eligible risk set.

## 2. Install

Use Python 3.12 on a Linux GPU node:

```bash
git clone <repository-url> BONSAI
cd BONSAI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,tabular,survival]"
```

Install the CUDA-compatible PyTorch build required by the cluster before
installing FlashAttention. Then:

```bash
python -m pip install -e ".[flash_attn]"
```

If FlashAttention cannot be built immediately, continue the functional
preflight with `model.attn_type=sdpa`. Do not silently change the production
backend after results have started.

Verify the environment:

```bash
python - <<'PY'
import torch
import bonsai
import opera

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python -m compileall bonsai opera
python -m pytest -q
```

## 3. Configure paths

Create `.env` from `.env.example` and use absolute server paths:

```bash
cp .env.example .env
```

Example:

```dotenv
BONSAI_CONFIG_PATH=/project/BONSAI/configs
BONSAI_MODELS=/project/artifacts/models
BONSAI_PROCESSED_DATA=/project/artifacts/processed_data
BONSAI_PREDICTIONS=/project/artifacts/predictions
BONSAI_CHECKPOINT_ROOT=/project/artifacts/checkpoints
BONSAI_RESULTS_ROOT=/project/artifacts/results

EHR2MEDS_OUTPUT=/project/ehr2meds/daly_care

BONSAI_COHORT_MEMBERSHIP=/project/artifacts/processed_data/daly_care/population_full.csv
BONSAI_OUTCOMES_DIR=/project/artifacts/processed_data/daly_care/outcomes
```

Load the variables in shell jobs as well as Python:

```bash
set -a
source .env
set +a
mkdir -p \
  "$BONSAI_MODELS" \
  "$BONSAI_PROCESSED_DATA" \
  "$BONSAI_PREDICTIONS" \
  "$BONSAI_CHECKPOINT_ROOT" \
  "$BONSAI_RESULTS_ROOT"
```

Never place patient data, checkpoints, or results inside the Git worktree.

## 4. Create model-ready subject data

Inspect the locked configuration:

```bash
python -m bonsai.run.create_data \
  --config-name daly_care_data \
  --cfg job
```

Confirm that it shows:

```yaml
splits: [train, tuning]
numeric_value_mode: legacy
vocabulary_cutoff_date: {year: 2022, month: 1, day: 1}
```

Then run:

```bash
python -m bonsai.run.create_data --config-name daly_care_data
```

This writes:

```text
$BONSAI_PROCESSED_DATA/daly_care/
├── subject_data_train.pt
├── subject_data_tuning.pt
├── vocabulary.pt
└── population_full.csv
```

The generated population file is only a minimal list of tokenized subject IDs.
After preserving it for provenance, replace it with the vetted, enriched
population table and copy in the outcome directory:

```bash
cp \
  "$BONSAI_PROCESSED_DATA/daly_care/population_full.csv" \
  "$BONSAI_PROCESSED_DATA/daly_care/tokenized_subjects.csv"

cp /source/validated/population_full.csv "$BONSAI_COHORT_MEMBERSHIP"
mkdir -p "$BONSAI_OUTCOMES_DIR"
cp /source/validated/outcomes/*.parquet "$BONSAI_OUTCOMES_DIR/"
```

Do not run the generic `bonsai.run.create_outcome` over the production endpoints.
The vetted external outcome pipeline owns eligibility, index dates, event dates,
and censoring.

## 5. Generate and validate experiment configs

Generated YAML files are derived artifacts. Regenerate them from the registry:

```bash
python -m opera.run.generate_sweep_configs
```

The canonical fine config is intentionally large: 24 cohorts x 87 outcomes x
6 model variants x 3 seeds, plus clinical baselines. Do not use a full-config
dry run as the first launch check, and do not launch variants whose checkpoints
do not exist.

Create a one-cell launch config without editing the generated source:

```bash
python - <<'PY'
from pathlib import Path
import yaml

source = Path("opera/configs/generated/fine_ipcw_cif_30d.yaml")
target = Path("server_configs/launch_gate_fine_cif_30d.yaml")
cfg = yaml.safe_load(source.read_text())
cfg["cohorts"] = {"DLBCL": cfg["cohorts"]["DLBCL"]}
cfg["outcomes"] = {"sepsis": cfg["outcomes"]["sepsis"]}
cfg["model_variants"] = {
    "daly_care_pretrain": cfg["model_variants"]["daly_care_pretrain"]
}
cfg["seeds"] = [42]
cfg["output_dir"] = "${BONSAI_RESULTS_ROOT}/launch_gate/fine_cif_30d"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(target)
PY

SWEEP_CONFIG=server_configs/launch_gate_fine_cif_30d.yaml

python -m opera.run.check_readiness \
  --config "$SWEEP_CONFIG" \
  --fail_on_issue

python -m opera.run.sweep \
  --config "$SWEEP_CONFIG" \
  --dry-run
```

After this cell completes end to end, repeat the snippet with source
`opera/configs/generated/fine_ipcw_30d.yaml`, target
`server_configs/launch_gate_fine_net_30d.yaml`, and a distinct output directory
to exercise the corresponding net-risk path.

This initial structural readiness pass occurs before checkpoint production.
Repeat it with `--require_existing_paths` in section 7. That strict pass must
have no unresolved variables, missing patients, physical partition overlap,
split-contract errors, missing outcomes, or missing checkpoints. A full
canonical config will fail until all six configured model variants have been
produced or registered; this is expected and must not be bypassed.

The dry run must show commands using:

- the shared `daly_care` subject-data directory;
- the enriched population file;
- `train` for supervised fitting;
- `tuning` for model selection;
- `held_out` only for final evaluation.

## 6. Functional smoke test

First test data loading, model construction, forward/backward passes, checkpoint
writing, and validation without touching supervised held-out predictions:

```bash
python -m opera.run.daly_care_pretrain \
  model.attn_type=sdpa \
  model.max_seqlen=512 \
  training.max_len=512 \
  training.batch_size=2 \
  training.accumulate_grad_batches=1 \
  training.epochs=1 \
  training.limit_train_batches=2 \
  training.limit_val_batches=2 \
  hardware.num_workers=0 \
  hydra.run.dir="$BONSAI_RESULTS_ROOT/smoke/daly_care_pretrain"
```

Require:

- finite training and validation losses;
- a `best.ckpt` and checkpoint metadata sidecar;
- no missing-token, subject-ID, CUDA, or dataloader errors.

Repeat with the intended production backend:

```bash
python -m opera.run.daly_care_pretrain \
  model.attn_type=flash \
  model.max_seqlen=512 \
  training.max_len=512 \
  training.batch_size=2 \
  training.accumulate_grad_batches=1 \
  training.epochs=1 \
  training.limit_train_batches=2 \
  training.limit_val_batches=2 \
  hardware.num_workers=0 \
  hydra.run.dir="$BONSAI_RESULTS_ROOT/smoke/daly_care_pretrain_flash"
```

Do not use the production held-out split for iterative supervised smoke tests.
If an end-to-end supervised infrastructure test is needed, create a dedicated
non-paper smoke outcome in which a subset of historical training patients is
labelled as the temporary test split.

## 7. Produce or register checkpoints

The generated sweeps reference:

```text
$BONSAI_CHECKPOINT_ROOT/pretrain/best.ckpt
$BONSAI_CHECKPOINT_ROOT/daly_care_pretrain/best.ckpt
$BONSAI_CHECKPOINT_ROOT/dapt/best.ckpt
$BONSAI_CHECKPOINT_ROOT/multi_outcome/best.ckpt
$BONSAI_CHECKPOINT_ROOT/leukemia_contrastive/best.ckpt
```

Only include a model variant in the first sweep when its checkpoint and metadata
are present. A run from scratch is:

```bash
python -m opera.run.daly_care_pretrain
```

This config enforces the exclusive 2022 cutoff. Promote the selected checkpoint
to the corresponding stable path under `$BONSAI_CHECKPOINT_ROOT`; retain its
`checkpoint_metadata.json` beside it.

Checkpoint-producing OPERA commands are resumable when they use the same Hydra
output directory. A successful run writes:

```text
best.ckpt
checkpoint_metadata.json
training_complete.json
```

On a later invocation with the same resolved training configuration, these
artifacts cause training to be skipped. A changed configuration in the same
directory is rejected rather than silently reusing or replacing a checkpoint.
To intentionally rerun it, pass `overwrite=true`; otherwise choose a new output
directory. Partial directories without all completion artifacts are not treated
as finished.

DAPT requires a base checkpoint with an exactly compatible vocabulary, or an
explicit vocabulary-expansion plan:

```bash
python -m opera.run.dapt \
  pretrain_ckpt="$BONSAI_CHECKPOINT_ROOT/pretrain/best.ckpt" \
  dataset=daly_care
```

Do not bypass a vocabulary mismatch. Joined lab-bin tokens must either already
exist in the base vocabulary or be introduced through configured expansion.

The multi-outcome and OPERA adaptation configs are:

```bash
python -m opera.run.build_dapt_embedding_store \
  dapt_ckpt="$BONSAI_CHECKPOINT_ROOT/dapt/best.ckpt" \
  output_path="$BONSAI_CHECKPOINT_ROOT/dapt/dapt_embeddings.pt"

python -m opera.run.mol \
  --config-name generated/multi_outcome_full_panel

python -m opera.run.contrastive_multicohort \
  --config-name generated/joint_opera_full_panel
```

The generated OPERA config requires that embedding store. Training fails below
99% coverage rather than silently disabling the configured similarity floor or
anchor. Re-running the store command reuses the existing file unless
`overwrite=true` is supplied.

Inspect each composed config with `--cfg job` before submitting it. Supply or
promote the prerequisite checkpoint paths expected by that stage.

Once the selected variants have their checkpoints, run the final strict gate:

```bash
python -m opera.run.check_readiness \
  --config "$SWEEP_CONFIG" \
  --require_existing_paths \
  --fail_on_issue
```

## 8. Launch policy

Run the one-cell config created in section 5 through finetuning and held-out
evaluation first. Then expand in stages: one outcome family, one horizon, and
only the checkpoint variants currently available. Run the complete canonical
matrix only when its scope and compute budget have been reviewed. Do not modify
the canonical generated file in place. For a production config:

```bash
python -m opera.run.sweep \
  --config "$SWEEP_CONFIG" \
  --fail-fast
```

Use `--fail-fast` for the first server run. Once the workflow is proven,
omitting it allows independent cells to continue after a failure. Do not use
`--overwrite` unless rerunning completed cells is intentional.

The sweep applies the same rule at cell level. A cell with completed metrics is
loaded from disk; a cell with a checkpoint but no metrics resumes at evaluation;
and a cell without a checkpoint starts finetuning. `--overwrite` is propagated
to the underlying training command and deliberately reruns the complete cell.

A minimal Slurm wrapper is:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=opera-fine-cif30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/project/logs/%x-%j.out

set -euo pipefail
cd /project/BONSAI
source .venv/bin/activate
set -a
source .env
set +a

python -m opera.run.check_readiness \
  --config opera/configs/generated/fine_ipcw_cif_30d.yaml \
  --require_existing_paths \
  --fail_on_issue

python -m opera.run.sweep \
  --config opera/configs/generated/fine_ipcw_cif_30d.yaml \
  --fail-fast
```

Adjust memory, time, partition, account, and GPU directives for the cluster.

## 9. Monitor and recover

Every sweep cell writes command, stdout, stderr, and status artifacts below its
result directory. The sweep-level files are:

```text
sweep_cell_status.csv
sweep_cell_status.jsonl
result.jsonl
```

During the first run, check:

- GPU utilization and memory;
- dataloader throughput;
- train and tuning loss for NaN/divergence;
- tuning event counts and early-stopping behavior;
- `ipcw_weight_summary.csv` for observed cases/controls, extreme weights, and
  effective sample size in fixed-horizon runs;
- `survival_support_summary.csv` for primary-event and comparable-pair support
  in Cox runs;
- `competing_risk_interval_support.csv` for target/death support across the
  OPERA representation-training time grid;
- disk usage from checkpoints and prediction artifacts;
- failed cells in `sweep_cell_status.csv`.

Treat a low IPCW effective-sample-size fraction or extreme normalized maximum
weight warning as an analysis warning, not routine log noise. The runner fails
cells with no usable fixed-horizon case/control support and Cox cells with no
genuinely comparable primary event.

The runner skips completed cells on restart. Fix the cause, rerun readiness, and
submit the same command. Use `--overwrite` only for cells whose existing results
must be replaced.

## 10. Final release gate

Before the held-out results are treated as paper results:

- record the Git commit and ehr2meds commit;
- archive the resolved `.env` paths without credentials;
- archive numeric-metadata provenance and the vocabulary;
- archive the generated configs;
- verify one shared index date and split per patient across outcomes;
- verify checkpoint metadata and vocabulary identity;
- freeze tuning-selected hyperparameters;
- run held-out evaluation once;
- preserve raw predictions and `result.jsonl` before aggregation.
- record the planned cell count and estimated GPU-hours before submission;
- verify the selected fixed-horizon primary estimand (`ipcw_cif_bce`) and the
  prespecified net-risk sensitivity scope (`ipcw_bce`).

The first recommended sequence is therefore:

```text
install
→ create_data
→ replace minimal population with validated membership
→ copy validated outcomes
→ regenerate configs
→ strict readiness
→ dry run
→ SDPA smoke
→ FlashAttention smoke
→ produce/promote checkpoints
→ strict readiness again
→ one small production submission
→ full sweep
```
