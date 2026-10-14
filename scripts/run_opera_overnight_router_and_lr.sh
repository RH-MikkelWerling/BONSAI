#!/usr/bin/env bash
set -euo pipefail

# Sequential, validation-selected pilot on ONE allocated GPU. No held-out reads.
# After `source setup_v100.sh`, run with nohup inside an active GPU allocation.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${BONSAI_MODELS:?Source setup_v100.sh first}"
: "${BONSAI_PROCESSED_DATA:?Source setup_v100.sh first}"
export OPERA_SHARED_BCE_CKPT="${OPERA_SHARED_BCE_CKPT:-$REPO_ROOT/data/models/outcome_readout_pilot_harmonized_v4/direct_cr_objective_horizon_bce/best.ckpt}"
export LR_READOUT_CONFIG=direct_cr_bce_lowrank_endpoint_groups_v1
export OPERA_SEED="${OPERA_SEED:-42}"

# Explicit pilot data; the final MEDS release gets a new, versioned suite.
export SEQUENCE_DATA_DIR="${SEQUENCE_DATA_DIR:-$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_harmonized_v4}"
export DAPT_CKPT="${DAPT_CKPT:-$BONSAI_MODELS/daly_care_t100_joined_bins_harmonized_v4/pretrain_harmonized_fourier_len8192_allwindows_20ep/version_0/best.ckpt}"
RUN_ROOT="${1:-$REPO_ROOT/data/models/opera_overnight_router_lr_v4}"

for input in "$OPERA_SHARED_BCE_CKPT" "$DAPT_CKPT" \
  "$SEQUENCE_DATA_DIR/subject_data_train.pt" \
  "$SEQUENCE_DATA_DIR/subject_data_tuning.pt" \
  "$SEQUENCE_DATA_DIR/vocabulary.pt"; do
  [[ -s "$input" ]] || { echo "Missing or empty input: $input" >&2; exit 1; }
done
for config in \
  opera/configs/generated/direct_cr_bce_router_interacting_endpoint_groups_v1.yaml \
  opera/configs/generated/direct_cr_bce_lowrank_endpoint_groups_v1.yaml; do
  [[ -s "$config" ]] || { echo "Missing or empty generated config: $config" >&2; exit 1; }
done

mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
echo "[$(date -Is)] Overnight pilot root: $RUN_ROOT"
echo "Source shared BCE: $OPERA_SHARED_BCE_CKPT"
echo "DAPT: $DAPT_CKPT"
echo "Sequence data: $SEQUENCE_DATA_DIR"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-inherited allocation}"

# First: additive-versus-interacting query comparison from the same shared BCE
# checkpoint, both with a frozen encoder. The existing additive result is seed 42.
# Then: encoder-rate ablation on the leading low-rank endpoint-group BCE readout.
# Its already completed seed-42 weightinggroups run is the 0.05 baseline.
# Keep the default queue bounded to two 30-epoch jobs. The 1.00 multiplier is
# optional once runtime and the 0.20 result are known.
TASKS=(routerinteraction encoder020)
if [[ "${OPERA_OVERNIGHT_INCLUDE_ENCODER100:-0}" == 1 ]]; then
  TASKS+=(encoder100)
fi
for task in "${TASKS[@]}"; do
  echo "[$(date -Is)] START $task"
  bash scripts/run_opera_independent_experiment.sh "$task" "$RUN_ROOT"
  echo "[$(date -Is)] FINISHED $task"
done
echo "[$(date -Is)] OVERNIGHT PILOT COMPLETE"
