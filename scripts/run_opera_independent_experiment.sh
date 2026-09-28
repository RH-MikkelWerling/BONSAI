#!/usr/bin/env bash
set -euo pipefail

# One invocation, one experiment, one GPU. Does not regenerate shared configs.
# Run after source setup_v100.sh. See docs/independent_experiments.md.
TASK="${1:?Usage: bash scripts/run_opera_independent_experiment.sh TASK [OUTPUT_ROOT]}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
OUTPUT_ROOT="${2:-$REPO_ROOT/data/models/opera_independent_v4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${OPERA_SEED:-42}"
SEQUENCE_DATA_DIR="${SEQUENCE_DATA_DIR:-${BONSAI_PROCESSED_DATA:?Load setup_v100.sh}/daly_care_t100_joined_bins_harmonized_v4}"
DAPT_CKPT="${DAPT_CKPT:-${BONSAI_MODELS:?Load setup_v100.sh}/daly_care_t100_joined_bins_harmonized_v4/pretrain_harmonized_fourier_len8192_allwindows_20ep/version_0/best.ckpt}"
OVERRIDES=()
case "$TASK" in
  weightinggroups)
    CONFIG=direct_cr_bce_lowrank_endpoint_groups_v1 ;;
  weightingreference)
    CONFIG=direct_cr_bce_lowrank_weighting_reference_v1 ;;
  lowrank)
    CONFIG=direct_cr_objective_horizon_bce_factorized_cohort_outcome_prototypes ;;
  anchored)
    CONFIG=direct_cr_objective_horizon_bce_source_anchored_adapters ;;
  unanchored)
    CONFIG=direct_cr_objective_horizon_bce_unanchored_adapters ;;
  encoder005|encoder020|encoder100)
    CONFIG="${LR_READOUT_CONFIG:?Set LR_READOUT_CONFIG to the tuning-selected generated config basename}"
    case "$TASK" in
      encoder005) MULTIPLIER=0.05 ;;
      encoder020) MULTIPLIER=0.20 ;;
      encoder100) MULTIPLIER=1.00 ;;
    esac
    OVERRIDES+=("model.freeze_encoder=false" "training.encoder_lr_multiplier=$MULTIPLIER") ;;
  *) echo "Unknown task: $TASK (weightinggroups, weightingreference, lowrank, anchored, unanchored, encoder005, encoder020, encoder100)" >&2; exit 2 ;;
esac
[[ "$CONFIG" =~ ^[a-zA-Z0-9_]+$ && "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid config name or seed" >&2; exit 2; }
[[ -s "opera/configs/generated/$CONFIG.yaml" ]] || { echo "Sync generated config: $CONFIG" >&2; exit 1; }
if [[ "$TASK" == anchored || "$TASK" == unanchored ]]; then
  : "${OPERA_SHARED_BCE_CKPT:?Set an existing completed shared-BCE checkpoint}"
  [[ -s "$OPERA_SHARED_BCE_CKPT" ]] || { echo "Missing shared-BCE checkpoint" >&2; exit 1; }
  OVERRIDES+=("source_opera_ckpt=$OPERA_SHARED_BCE_CKPT")
fi
for INPUT in "$SEQUENCE_DATA_DIR/subject_data_train.pt" "$SEQUENCE_DATA_DIR/subject_data_tuning.pt" "$SEQUENCE_DATA_DIR/vocabulary.pt" "$DAPT_CKPT"; do
  [[ -s "$INPUT" ]] || { echo "Missing or empty input: $INPUT" >&2; exit 1; }
done
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
RUN_DIR="$OUTPUT_ROOT/${TASK}__${CONFIG}__seed${SEED}"
mkdir -p "$RUN_DIR"
# Atomic directory lock prevents two workers resuming the same experiment.
mkdir "$RUN_DIR/.launch_lock" 2>/dev/null || { echo "Run already locked: $RUN_DIR" >&2; exit 1; }
trap 'rmdir "$RUN_DIR/.launch_lock"' EXIT
if [[ -f "$RUN_DIR/training_complete.json" ]]; then
  echo "SKIP completed: $RUN_DIR"
  exit 0
fi
OVERRIDES+=("paths.sequence_data_dir=$SEQUENCE_DATA_DIR" "dapt_ckpt=$DAPT_CKPT" "seed=$SEED" "training.auto_resume=true" "hydra.run.dir=$RUN_DIR")
PREFLIGHT_ARGS=()
for OVERRIDE in "${OVERRIDES[@]}"; do PREFLIGHT_ARGS+=(--override "$OVERRIDE"); done
exec > >(tee -a "$RUN_DIR/launch.log") 2>&1
echo "[$(date -Is)] Task=$TASK config=$CONFIG seed=$SEED GPU=${CUDA_VISIBLE_DEVICES:-inherited}"
echo "Sequence data: $SEQUENCE_DATA_DIR"
echo "DAPT checkpoint: $DAPT_CKPT"
echo "Run directory: $RUN_DIR"
"$PYTHON_BIN" -u -m opera.run.preflight_direct_competing_risk \
  --config-name "generated/$CONFIG" --require-existing-paths "${PREFLIGHT_ARGS[@]}"
"$PYTHON_BIN" -u -m opera.run.contrastive_multicohort \
  --config-name "generated/$CONFIG" "${OVERRIDES[@]}"
[[ -f "$RUN_DIR/training_complete.json" ]] || { echo "No training_complete.json: run not complete" >&2; exit 1; }
echo "[$(date -Is)] COMPLETE: $TASK"
