# Focused OPERA outcome-transfer runbook

This is the narrow transfer experiment only. It leaves the production sweep,
rarity analyses, and general outcome registry unchanged.

The seven OPERA conditions and their three seeds are resolved from
`opera/configs/manifests/outcome_transfer.yaml`. The canonical full condition
is `opera/configs/generated/joint_opera_full_panel.yaml`; every ablation keeps
that configuration and changes only the contrastive outcome panel.

## Server setup

Run commands from the repository root. `BONSAI_CONFIG_PATH` must be the
top-level `configs` directory, not `opera/configs`.

```bash
cd /path/to/BONSAI

export BONSAI_CONFIG_PATH="$PWD/configs"
export BONSAI_PROCESSED_DATA=/data/processed/subject_data
export BONSAI_COHORT_MEMBERSHIP=/data/cohorts/cohort_membership.parquet
export BONSAI_OUTCOMES_DIR=/data/outcomes
export BONSAI_CHECKPOINT_ROOT=/data/checkpoints
export BONSAI_RESULTS_ROOT=/data/results

ROOT="$BONSAI_RESULTS_ROOT/outcome_transfer"
```

`$BONSAI_PROCESSED_DATA` must contain `vocabulary.pt` and the three shared
subject stores `subject_data_train.pt`, `subject_data_tuning.pt`, and
`subject_data_held_out.pt`. `$BONSAI_OUTCOMES_DIR` must contain every registry
outcome file, including `overall_survival.parquet` for competing-event labels.

## Resolve, generate, and audit labels

```bash
python -m opera.run.generate_outcome_transfer_configs --validate-only
python -m opera.run.generate_outcome_transfer_configs

python -m opera.run.outcome_transfer_preflight \
  --output-dir "$ROOT/preflight"
```

Review `outcome_transfer_support.csv` before launching. It reports every
target for all hematology patients and every grouped cohort; low-support cells
remain visible rather than being removed. In fixed-horizon labels, a competing
death is retained as `event=2`, has binary label `0`, is counted among
non-events, and is reported separately.

The resolver records SHA-256 hashes for the canonical full OPERA config and
the temporal split contract. Generated configs, new checkpoints, preflight,
and frozen embedding sidecars must match those hashes; regenerate/re-audit if
either checked-in input changes.

## Contrastive checkpoints

First inspect the exact 21 slots without starting a process:

```bash
python -m opera.run.outcome_transfer_train \
  --output-dir "$ROOT" \
  --full-checkpoint-template "$BONSAI_CHECKPOINT_ROOT/opera_full/seed_{seed}/best.ckpt" \
  --dry-run
```

Then launch the six ablations. `opera_full` is reused and DAPT is never
retrained. A legacy full checkpoint path must include `{seed}`; an unseeded
checkpoint is not accepted as seed-specific provenance.

```bash
python -m opera.run.outcome_transfer_train \
  --output-dir "$ROOT" \
  --full-checkpoint-template "$BONSAI_CHECKPOINT_ROOT/opera_full/seed_{seed}/best.ckpt" \
  --preflight-report "$ROOT/preflight/outcome_transfer_support.json" \
  --execute
```

The launcher writes `transfer_training_status.csv`. Use its `checkpoint_path`
column rather than assuming a logger version directory. It checks the outcome
panel, seed provenance, contrastive stage, and DAPT origin before reusing a
slot.

## Extract frozen representations

Extract one `cls_last` representation for each OPERA checkpoint/seed. The
extractor verifies that every transfer target has the same prediction origin
and censors each patient sequence at that origin before encoding.

```bash
python -m opera.run.extract_outcome_transfer_embeddings \
  --representation opera_no_g3 --seed 42 \
  --checkpoint "$ROOT/runs/opera_no_g3/seed_42/contrastive_multicohort_runs/version_0/best.ckpt" \
  --vocabulary "$BONSAI_PROCESSED_DATA/vocabulary.pt" \
  --subject-data-dir "$BONSAI_PROCESSED_DATA" \
  --output "$ROOT/embeddings/opera_no_g3_seed_42.npz"
```

Repeat for `opera_full` and each ablation seed listed in the training status.
If an existing `opera_full` checkpoint predates seed metadata, add
`--legacy-full-checkpoint-template "$BONSAI_CHECKPOINT_ROOT/opera_full/seed_{seed}/best.ckpt"`
to its extraction command; it must render to the supplied checkpoint.
For the existing DAPT baseline, extract its frozen embedding once; the same
DAPT artifact may be supplied for `dapt:42`, `dapt:43`, and `dapt:44` because
DAPT is not retrained for this experiment. Each extraction writes a required
adjacent `.metadata.json` provenance sidecar.

## Frozen probes and paired transfer analysis

The evaluator fits a separate standardized logistic-regression probe for each
target and representation using all eligible pan-hematology training patients,
selects its regularization on the pan-hematology tuning patients, and scores
the locked held-out patients once. Grouped-cohort rows are re-stratifications
of those saved held-out predictions; no cohort-specific probe is fitted.

Build the 24 required embedding arguments (DAPT, full OPERA, and six
ablations for each of the three seeds), then run:

```bash
embedding_args=()
for seed in 42 43 44; do
  embedding_args+=(--embedding "dapt:${seed}=$ROOT/embeddings/dapt_seed_42.npz")
  for condition in opera_full opera_no_g3 opera_no_transfusion_signal \
                   opera_no_hospitalisation_signal opera_no_infection_family \
                   opera_no_renal_family opera_no_cardiovascular_family; do
    embedding_args+=(--embedding "${condition}:${seed}=$ROOT/embeddings/${condition}_seed_${seed}.npz")
  done
done

python -m opera.run.outcome_transfer_evaluate \
  "${embedding_args[@]}" \
  --output-dir "$ROOT/probes"

python -m opera.run.aggregate_outcome_transfer \
  --predictions "$ROOT/probes/transfer_predictions.parquet" \
  --results "$ROOT/probes/transfer_results.csv" \
  --output-dir "$ROOT/aggregate" \
  --n-bootstrap 2000
```

The evaluator fails closed if representations have different eligible patients,
labels, event times, or competing-event indicators. Aggregation performs
patient-level paired bootstraps on those identical held-out denominators and
writes the three-panel PNG/PDF figure. It does not manufacture results for
unsupported targets.

## Outputs

The run produces the resolved plan/configs, support CSV/JSON, training status,
frozen-probe status/results/predictions/failures, paired deltas, grouped-cohort
results and macro summary, family summary, and `outcome_transfer.png/.pdf`.
No transfer claim should be made until the real held-out outputs have been
reviewed.
