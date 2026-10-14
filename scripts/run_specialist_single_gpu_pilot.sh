#!/usr/bin/env bash
# One allocation, one GPU: smoke/timing batch from the planned specialist grid.
# Does not submit a job or alter the existing manifest.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export OPERA_SPECIALIST_ROOT="${OPERA_SPECIALIST_ROOT:-$REPO/data/models/independent_specialist_grid_20261012_v1}"
export OPERA_SPECIALIST_SHARD_COUNT=4
export OPERA_SPECIALIST_SHARD_INDEX=0
export OPERA_SPECIALIST_MAX_NEW="${OPERA_SPECIALIST_MAX_NEW:-2}"
export OPERA_LOCAL_GPU="${OPERA_LOCAL_GPU:-0}"
export OPERA_DAPT_CKPT="${OPERA_DAPT_CKPT:-$REPO/data/models/daly_care_t100_joined_bins_with_sor_final_v1/pretrain_harmonized_scaled_time2vec_len8192_20ep/final_b4_20261003/version_0/best.ckpt}"

[[ -s "$OPERA_SPECIALIST_ROOT/manifest_report.json" ]] || {
  echo "Missing specialist grid plan; run 'bash scripts/run_independent_specialist_grid.sh plan' first" >&2
  exit 1
}
echo "Single-GPU specialist pilot: shard 0/4, at most $OPERA_SPECIALIST_MAX_NEW new fits, local GPU $OPERA_LOCAL_GPU"
exec bash "$REPO/scripts/run_independent_specialist_grid.sh" run
