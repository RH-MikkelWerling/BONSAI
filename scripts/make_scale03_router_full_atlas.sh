#!/usr/bin/env bash
# Full tuning-only fine-disease × outcome router map, streaming aggregates.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

RUN="${OPERA_ROUTER_RUN:-$REPO/data/models/approved_neural_router_clean_20261007_v1/time2vec_router_interacting_scale03_e2e}"
OUT="${OPERA_ROUTER_ATLAS_DIR:-$REPO/data/results/router_scale03_full_borrowing_atlas_20261010_INTERNAL}"
export BONSAI_COHORT_MEMBERSHIP="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
export BONSAI_OUTCOMES_DIR="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}/outcomes"
export CUDA_VISIBLE_DEVICES="${OPERA_LOCAL_GPU:-0}"

for input in "$RUN/best.ckpt" "$RUN/hydra/config.yaml" \
             "$BONSAI_COHORT_MEMBERSHIP" "$BONSAI_OUTCOMES_DIR/overall_survival.parquet"; do
  [[ -s "$input" ]] || { echo "Missing or empty input: $input" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || {
  echo "Atlas output already exists; inspect it before choosing a new path: $OUT" >&2
  exit 1
}

python -u -m opera.run.export_router_borrowing_atlas \
  --checkpoint "$RUN/best.ckpt" \
  --config-dir "$RUN/hydra" --config-name config \
  --device cuda --batch-size "${OPERA_ATLAS_BATCH_SIZE:-16}" \
  --num-workers "${OPERA_ATLAS_NUM_WORKERS:-4}" \
  --min-patients "${OPERA_ATLAS_MIN_PATIENTS:-20}" \
  --output-dir "$OUT"

echo "Full atlas: $OUT/router_full_task_borrowing_atlas.pdf"
echo "Directed source flow: $OUT/router_full_source_flow.pdf"
echo "Task × source cube: $OUT/router_route_cube_INTERNAL.npz"
