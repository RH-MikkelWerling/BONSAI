#!/usr/bin/env bash
# Run inside a GPU allocation; never submits a job.
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in preflight|launch) ;; *) echo "Usage: bash scripts/run_pretrain_hidden256.sh [preflight|launch]" >&2; exit 2 ;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"
: "${BONSAI_256_HEADS:?Set the confirmed number of attention heads, not dimensions per head}"
: "${BONSAI_256_LAYERS:?Set the confirmed number of transformer layers}"
BATCH="${BONSAI_256_BATCH_SIZE:-1}"
python - "$BONSAI_256_HEADS" "$BONSAI_256_LAYERS" "$BATCH" <<'PY'
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
      "hydra.run.dir=${BONSAI_256_RUN_DIR:-$REPO/data/models/pretrain_local_scaled_time2vec_hidden256_20261013_v1}")
if [[ "$MODE" == preflight ]]; then
  exec python -m bonsai.run.pretrain "${ARGS[@]}" --cfg job --resolve
fi
[[ "${BONSAI_256_ARCHITECTURE_CONFIRMED:-0}" == 1 ]] || {
  echo 'Compare the complete resolved config with the collaborators; then set BONSAI_256_ARCHITECTURE_CONFIRMED=1.' >&2
  exit 1
}
exec python -m bonsai.run.pretrain "${ARGS[@]}"
