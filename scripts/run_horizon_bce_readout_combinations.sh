#!/usr/bin/env bash
set -euo pipefail

# Run the three horizon-BCE/readout combinations sequentially on one GPU.
# Existing incomplete directories resume through training.auto_resume=true;
# completed directories are skipped.
#
# Usage:
#   bash scripts/run_horizon_bce_readout_combinations.sh [OUTPUT_ROOT]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${1:-$REPO_ROOT/data/models/horizon_bce_readout_combinations_v4_verified}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEQUENCE_DATA_DIR="${SEQUENCE_DATA_DIR:-${BONSAI_PROCESSED_DATA:?Set BONSAI_PROCESSED_DATA}/daly_care_t100_joined_bins_harmonized_v4}"
DAPT_CKPT="${DAPT_CKPT:-${BONSAI_MODELS:?Set BONSAI_MODELS}/daly_care_t100_joined_bins_harmonized_v4/pretrain_harmonized_fourier_len8192_allwindows_20ep/version_0/best.ckpt}"

RUNS=(
  direct_cr_objective_horizon_bce_outcome_mlp
  direct_cr_objective_horizon_bce_outcome_attention
  direct_cr_objective_horizon_bce_shared_attention_prototypes
  direct_cr_objective_horizon_bce_factorized_cohort_outcome_prototypes
)

mkdir -p "$OUTPUT_ROOT"

for REQUIRED_PATH in \
  "$SEQUENCE_DATA_DIR/subject_data_train.pt" \
  "$SEQUENCE_DATA_DIR/subject_data_tuning.pt" \
  "$SEQUENCE_DATA_DIR/vocabulary.pt" \
  "$DAPT_CKPT"; do
  if [[ ! -s "$REQUIRED_PATH" ]]; then
    echo "Missing or empty required input: $REQUIRED_PATH" >&2
    exit 1
  fi
done

echo "Sequence data: $SEQUENCE_DATA_DIR"
echo "DAPT checkpoint: $DAPT_CKPT"
echo "Output root: $OUTPUT_ROOT"

echo "[$(date -Is)] Regenerating experiment configs from the checked-in generator"
"$PYTHON_BIN" -m opera.run.generate_sweep_configs

echo "[$(date -Is)] Preflighting ${#RUNS[@]} horizon-BCE combination runs"
for RUN_NAME in "${RUNS[@]}"; do
  "$PYTHON_BIN" -u -m opera.run.preflight_direct_competing_risk \
    --config-name "generated/$RUN_NAME" \
    --require-existing-paths \
    --override paths.sequence_data_dir="$SEQUENCE_DATA_DIR" \
    --override dapt_ckpt="$DAPT_CKPT"
done

echo "[$(date -Is)] All preflights passed; beginning sequential GPU queue"
for RUN_NAME in "${RUNS[@]}"; do
  RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
  mkdir -p "$RUN_DIR"

  if [[ -f "$RUN_DIR/training_complete.json" ]]; then
    echo "[$(date -Is)] SKIP completed: $RUN_NAME"
    continue
  fi

  echo "[$(date -Is)] START: $RUN_NAME"
  echo "Command output: $RUN_DIR/launch.log"
  "$PYTHON_BIN" -u -m opera.run.contrastive_multicohort \
    --config-name "generated/$RUN_NAME" \
    paths.sequence_data_dir="$SEQUENCE_DATA_DIR" \
    dapt_ckpt="$DAPT_CKPT" \
    "hydra.run.dir=$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/launch.log"

  if [[ ! -f "$RUN_DIR/training_complete.json" ]]; then
    echo "[$(date -Is)] ERROR: process exited without training_complete.json: $RUN_NAME" >&2
    exit 1
  fi
  echo "[$(date -Is)] COMPLETE: $RUN_NAME"
done

echo "[$(date -Is)] All horizon-BCE combination runs completed"
