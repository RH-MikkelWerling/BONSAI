#!/usr/bin/env bash
# Tuning-only pre/post-adapter embedding panels for prespecified clinical cells.
# Requires OPERA_ROUTER_RUN so the selected rank-8/rank-16 checkpoint is explicit.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

: "${OPERA_ROUTER_RUN:?Set OPERA_ROUTER_RUN to the selected model run directory}"
RUN="$OPERA_ROUTER_RUN"
OUT="${OPERA_EMBEDDING_PANEL_DIR:-$REPO/data/results/router_outcome_embedding_panels_INTERNAL}"
export BONSAI_COHORT_MEMBERSHIP="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
export BONSAI_OUTCOMES_DIR="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}/outcomes"
export CUDA_VISIBLE_DEVICES="${OPERA_LOCAL_GPU:-0}"

for input in "$RUN/best.ckpt" "$RUN/hydra/config.yaml" \
             "$BONSAI_COHORT_MEMBERSHIP" "$BONSAI_OUTCOMES_DIR/overall_survival.parquet"; do
  [[ -s "$input" ]] || { echo "Missing input: $input" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || {
  echo "Output already exists; choose a new OPERA_EMBEDDING_PANEL_DIR: $OUT" >&2; exit 1;
}
mkdir -p "$OUT"

# Fixed before viewing this checkpoint; override only to answer a stated question.
read -r -a CELLS <<<"${OPERA_EMBED_CELLS:-CLL:overall_survival CLL:sepsis DLBCL:overall_survival MM:aki_first_documented}"
read -r -a PROJECTIONS <<<"${OPERA_EMBED_PROJECTIONS:-pca}"
[[ ${#CELLS[@]} -gt 0 && ${#PROJECTIONS[@]} -gt 0 ]] || {
  echo "At least one cell and projection are required" >&2; exit 2;
}
for projection in "${PROJECTIONS[@]}"; do
  [[ "$projection" == pca || "$projection" == umap ]] || {
    echo "Projection must be pca or umap: $projection" >&2; exit 2;
  }
done

for cell in "${CELLS[@]}"; do
  [[ "$cell" =~ ^[A-Za-z0-9_]+:[A-Za-z0-9_]+$ ]] || {
    echo "Invalid cell: $cell" >&2; exit 2;
  }
  disease="${cell%%:*}"
  outcome="${cell#*:}"
  cell_dir="$OUT/${disease}__${outcome}"
  python -u -m opera.run.evaluate_direct_competing_risk \
    --checkpoint "$RUN/best.ckpt" --config-dir "$RUN/hydra" --config-name config \
    --split tuning --horizons 365 --device cuda \
    --cohort-fine "$disease" --outcome "$outcome" \
    --export-routed-outcome "$outcome" --output-dir "$cell_dir"
  for projection in "${PROJECTIONS[@]}"; do
    python -m opera.run.plot_censoring_aware_embeddings \
      --input "$cell_dir/direct_cr_predictions.parquet" \
      --router-vectors "$cell_dir/routed_patient_vectors_INTERNAL.npz" \
      --target-outcome "$outcome" --router-projection "$projection" \
      --router-panel movement --time-col time_days --event-col event \
      --predicted-cif-col cif_365d --horizon-days 365 \
      --title "${disease} · ${outcome//_/ }" \
      --output "$cell_dir/${projection}_router_movement.pdf"
  done
  echo "Completed $cell"
done
echo "Embedding panels: $OUT"
