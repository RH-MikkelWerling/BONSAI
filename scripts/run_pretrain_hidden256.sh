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
# BONSAI_WIDTH=512 runs the width probe (configs/..._hidden512_20ep.yaml).
# Every encoder is pretrained for 10 epochs (decided 2026-10-19, as the national
# encoder); BONSAI_EPOCHS overrides it.
WIDTH="${BONSAI_WIDTH:-256}"
EPOCHS="${BONSAI_EPOCHS:-10}"
case "$WIDTH" in 256|512) ;; *) echo "BONSAI_WIDTH must be 256 or 512" >&2; exit 2 ;; esac
# BONSAI_DATASET=<processed release> starts fresh runs on that release: both
# widths from their YAML (identical apart from width and heads), in their own
# run directories under data/models/<release>/. Without it, the historical
# run directories and the 256 run's saved config are used, as before.
DATASET="${BONSAI_DATASET:-}"
if [[ -n "$DATASET" ]]; then
  [[ -f "${BONSAI_PROCESSED_DATA:?}/$DATASET/vocabulary.pt" ]] || {
    echo "No processed release at $BONSAI_PROCESSED_DATA/$DATASET (run create_data first)" >&2
    exit 1
  }
  DEFAULT_DIR="$REPO/data/models/$DATASET/pretrain_scaled_time2vec_hidden${WIDTH}"
  if [[ "$WIDTH" == 256 ]]; then
    RUN_DIR="${BONSAI_256_RUN_DIR:-$DEFAULT_DIR}"
  else
    RUN_DIR="${BONSAI_512_RUN_DIR:-$DEFAULT_DIR}"
  fi
elif [[ "$WIDTH" == 256 ]]; then
  RUN_DIR="${BONSAI_256_RUN_DIR:-$REPO/data/models/pretrain_local_scaled_time2vec_hidden256_20261013_v1}"
else
  RUN_DIR="${BONSAI_512_RUN_DIR:-$REPO/data/models/pretrain_local_scaled_time2vec_hidden512_20261016_v1}"
fi
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
        "training.epochs=$EPOCHS" training.early_stopping_patience=null)
  [[ "$MODE" == resume-preflight ]] && ARGS+=(--cfg job --resolve)
  echo "Resuming existing $WIDTH run to $EPOCHS total epochs: $RUN_DIR"
  exec "$PYTHON_BIN" -m bonsai.run.pretrain "${ARGS[@]}"
fi
: "${BONSAI_256_HEADS:?Set the confirmed number of attention heads, not dimensions per head}"
: "${BONSAI_256_LAYERS:?Set the confirmed number of transformer layers}"
BATCH="${BONSAI_256_BATCH_SIZE:-1}"
HEADS="$BONSAI_256_HEADS"
if [[ "$WIDTH" == 512 ]]; then
  # Keep the head dimension of the 256 encoder unless set explicitly.
  HEADS="${BONSAI_512_HEADS:-$((BONSAI_256_HEADS * 2))}"
fi
"$PYTHON_BIN" - "$HEADS" "$BONSAI_256_LAYERS" "$BATCH" "$WIDTH" <<'PY'
import sys
heads, layers, batch, width = map(int, sys.argv[1:])
if heads <= 0 or width % heads or layers <= 0 or batch <= 0 or 128 % batch:
    raise SystemExit(f'Require positive layers, heads dividing {width}, and physical batch dividing 128')
print(f'Width {width}; {heads} heads; head dimension {width // heads}; {layers} layers; effective single-GPU batch 128')
PY
ARGS=(--config-name "pretrain_harmonized_scaled_time2vec_hidden${WIDTH}_20ep"
      "model.num_attention_heads=$HEADS"
      "model.num_layers=$BONSAI_256_LAYERS"
      "training.batch_size=$BATCH"
      "training.accumulate_grad_batches=$((128 / BATCH))"
      "hydra.run.dir=$RUN_DIR")
if [[ -n "$DATASET" ]]; then
  ARGS+=("dataset=$DATASET" "paths.dir=$BONSAI_PROCESSED_DATA/$DATASET")
elif [[ "$WIDTH" != 256 ]]; then
  # Width probe: start from the 256 run's own saved config, so data, vocabulary,
  # time encoding and optimiser are identical; change only width, heads, batch
  # and epochs (and the run directory).
  REF_DIR="${BONSAI_256_RUN_DIR:-$REPO/data/models/pretrain_local_scaled_time2vec_hidden256_20261013_v1}"
  [[ "$REF_DIR" == /* ]] || REF_DIR="$REPO/$REF_DIR"
  REF_CONFIG="$REF_DIR/hydra"
  [[ -f "$REF_CONFIG/config.yaml" ]] || REF_CONFIG="$REF_DIR/.hydra"
  [[ -f "$REF_CONFIG/config.yaml" ]] || {
    echo "No saved config of the 256 run under $REF_DIR (set BONSAI_256_RUN_DIR)" >&2
    exit 1
  }
  echo "Width probe based on the 256 run's config: $REF_CONFIG/config.yaml"
  ARGS=(--config-path "$REF_CONFIG" --config-name config
        "model.hidden_size=$WIDTH"
        "model.num_attention_heads=$HEADS"
        "model.num_layers=$BONSAI_256_LAYERS"
        "training.batch_size=$BATCH"
        "training.accumulate_grad_batches=$((128 / BATCH))"
        "paths.ckpt_path=null"
        "hydra.run.dir=$RUN_DIR")
fi
ARGS+=("training.epochs=$EPOCHS")
# The inherited epoch-10 milestone checkpoint only applies to longer runs.
(( EPOCHS <= 10 )) && ARGS+=("training.checkpoint_milestone_epochs=[]")
if [[ "$MODE" == preflight ]]; then
  exec "$PYTHON_BIN" -m bonsai.run.pretrain "${ARGS[@]}" --cfg job --resolve
fi
if [[ -f "$RUN_DIR/version_0/last.ckpt" ]]; then
  # auto_resume would silently continue that run (and its vocabulary).
  echo "$RUN_DIR already holds a run; use 'resume' or choose another run directory" >&2
  exit 1
fi
[[ "${BONSAI_256_ARCHITECTURE_CONFIRMED:-0}" == 1 ]] || {
  echo 'Compare the complete resolved config with the collaborators; then set BONSAI_256_ARCHITECTURE_CONFIRMED=1.' >&2
  exit 1
}
exec "$PYTHON_BIN" -m bonsai.run.pretrain "${ARGS[@]}"
