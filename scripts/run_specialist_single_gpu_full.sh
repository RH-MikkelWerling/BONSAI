#!/usr/bin/env bash
# Work through every planned specialist shard on one GPU, restart-safely.
# Run inside a GPU allocation; no scheduler submission is performed here.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export OPERA_SPECIALIST_ROOT="${OPERA_SPECIALIST_ROOT:-$REPO/data/models/independent_specialist_grid_20261012_v1}"
export OPERA_SPECIALIST_SHARD_COUNT=4
export OPERA_SPECIALIST_MAX_NEW=1000000
export OPERA_RETRY_EMPTY_BATCH_FAILURES=1
export OPERA_LOCAL_GPU="${OPERA_LOCAL_GPU:-0}"
export OPERA_DAPT_CKPT="${OPERA_DAPT_CKPT:-$REPO/data/models/daly_care_t100_joined_bins_with_sor_final_v1/pretrain_harmonized_scaled_time2vec_len8192_20ep/final_b4_20261003/version_0/best.ckpt}"

[[ -s "$OPERA_SPECIALIST_ROOT/manifest_report.json" ]] || {
  echo "Missing specialist grid plan; run the plan command first" >&2
  exit 1
}
echo "One-GPU full specialist grid: four planned shards, local GPU $OPERA_LOCAL_GPU"
for OPERA_SPECIALIST_SHARD_INDEX in 0 1 2 3; do
  export OPERA_SPECIALIST_SHARD_INDEX
  echo "Starting/resuming specialist shard $OPERA_SPECIALIST_SHARD_INDEX of 4"
  bash "$REPO/scripts/run_independent_specialist_grid.sh" run
done
echo "One-GPU specialist grid pass finished; inspect status.tsv and failures in each shard."
