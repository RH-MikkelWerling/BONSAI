#!/usr/bin/env bash
# Run inside a GPU allocation; never submits a job.
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in preflight|launch|resume|resume-preflight) ;; *) echo "Usage: bash scripts/run_pretrain_hidden256.sh [preflight|launch|resume|resume-preflight]" >&2; exit 2 ;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [[ -f "$REPO/setup_v100.sh" ]]; then
  source "$REPO/setup_v100.sh"
fi
PYTHON_BIN="${OPERA_PYTHON:-python}"
RUN_DIR="${BONSAI_256_RUN_DIR:-$REPO/data/models/pretrain_local_scaled_time2vec_hidden256_20261013_v1}"
[[ "$RUN_DIR" == /* ]] || RUN_DIR="$REPO/$RUN_DIR"
if [[ "$MODE" == resume || "$MODE" == resume-preflight ]]; then
  CKPT="$RUN_DIR/version_0/last.ckpt"
  CONFIG_DIR="$RUN_DIR/hydra"
  [[ -f "$CONFIG_DIR/config.yaml" ]] || CONFIG_DIR="$RUN_DIR/.hydra"
  [[ -f "$CKPT" ]] || { echo "Cannot resume: missing $CKPT" >&2; exit 1; }
  [[ -f "$CONFIG_DIR/config.yaml" ]] || {
    echo "Cannot resume: no saved hydra/config.yaml or .hydra/config.yaml in $RUN_DIR" >&2
    exit 1
  }
  # Reuse the run's resolved architecture, data and optimizer settings.
  # No need to repeat architecture-confirmation environment variables.
  ARGS=(--config-path "$CONFIG_DIR" --config-name config
        "hydra.run.dir=$RUN_DIR" "paths.ckpt_path=$CKPT"
        training.epochs=20 training.early_stopping_patience=null)
  [[ "$MODE" == resume-preflight ]] && ARGS+=(--cfg job --resolve)
  echo "Resuming existing 256 run to 20 total epochs: $RUN_DIR"
  exec "$PYTHON_BIN" -m bonsai.run.pretrain "${ARGS[@]}"
fi
: "${BONSAI_256_HEADS:?Set the confirmed number of attention heads, not dimensions per head}"
: "${BONSAI_256_LAYERS:?Set the confirmed number of transformer layers}"
BATCH="${BONSAI_256_BATCH_SIZE:-1}"
"$PYTHON_BIN" - "$BONSAI_256_HEADS" "$BONSAI_256_LAYERS" "$BATCH" <<'PY'
import sys
heads, layers, batch = map(int, sys.argv[1:])
if heads <= 0 or 256 % heads or layers <= 0 or batch <= 0 or 128 % batch:
    raise SystemExit('Require positive layers, heads dividing 256, and physical batch dividing 128')
print(f'Width 256; {heads} heads; head dimension {256 // heads}; {layers} layers; effective single-GPU batch 128')
PY
ARGS=(--config-name pretrain_harmonized_scaled_time2vec_hidden256_20ep
      "model.num_attention_heads=$BONSAI_256_HEADS"
      "model.num_layers=$BONSAI_256_LAYERS"
      "training.batch_size=$BATCH"
      "training.accumulate_grad_batches=$((128 / BATCH))"
      "hydra.run.dir=$RUN_DIR")
if [[ "$MODE" == preflight ]]; then
  exec "$PYTHON_BIN" -m bonsai.run.pretrain "${ARGS[@]}" --cfg job --resolve
fi
[[ "${BONSAI_256_ARCHITECTURE_CONFIRMED:-0}" == 1 ]] || {
  echo 'Compare the complete resolved config with the collaborators; then set BONSAI_256_ARCHITECTURE_CONFIRMED=1.' >&2
  exit 1
}
exec "$PYTHON_BIN" -m bonsai.run.pretrain "${ARGS[@]}"
