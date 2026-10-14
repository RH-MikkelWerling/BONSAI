#!/usr/bin/env bash
# Export prespecified tuning cells from the clean scale-0.3 router and plot them.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

RUN="${OPERA_ROUTER_RUN:-$REPO/data/models/approved_neural_router_clean_20261007_v1/time2vec_router_interacting_scale03_e2e}"
EXPORT_ROOT="${OPERA_ROUTER_EXPORT_ROOT:-$REPO/data/results/router_scale03_cells_tuning_20261010_INTERNAL}"
FIGURE_DIR="${OPERA_ROUTER_FIGURE_DIR:-$REPO/data/results/router_scale03_cell_borrowing_readable_20261010_INTERNAL}"
export BONSAI_COHORT_MEMBERSHIP="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
export BONSAI_OUTCOMES_DIR="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}/outcomes"
export CUDA_VISIBLE_DEVICES="${OPERA_LOCAL_GPU:-0}"

# Prespecified for clinical coverage, not chosen after inspecting router weights.
# Override with a space-separated list of FINE_DISEASE:outcome cells if needed.
CELLS_TEXT="${OPERA_ROUTER_CELLS:-CLL:overall_survival CLL:sepsis DLBCL:overall_survival MM:aki_first_documented}"
read -r -a CELLS <<<"$CELLS_TEXT"
[[ ${#CELLS[@]} -gt 0 ]] || { echo "No router cells requested" >&2; exit 2; }

for input in "$RUN/best.ckpt" "$RUN/hydra/config.yaml" \
             "$BONSAI_COHORT_MEMBERSHIP" "$BONSAI_OUTCOMES_DIR/overall_survival.parquet"; do
  [[ -s "$input" ]] || { echo "Missing or empty input: $input" >&2; exit 1; }
done
[[ ! -e "$FIGURE_DIR" ]] || {
  echo "Figure directory already exists; inspect it before choosing a new output path: $FIGURE_DIR" >&2
  exit 1
}
mkdir -p "$EXPORT_ROOT"
PLOT_ARGS=()
for cell in "${CELLS[@]}"; do
  [[ "$cell" =~ ^[A-Za-z0-9_]+:[A-Za-z0-9_]+$ ]] || {
    echo "Invalid cell syntax: $cell" >&2; exit 2;
  }
  fine="${cell%%:*}"
  outcome="${cell#*:}"
  cell_dir="$EXPORT_ROOT/${fine}__${outcome}"
  routes="$cell_dir/source_route_weights.parquet"
  summary="$cell_dir/direct_cr_summary.json"
  if [[ -s "$routes" && -s "$summary" ]]; then
    echo "REUSE complete tuning export: $cell"
  elif [[ -e "$cell_dir" ]]; then
    echo "Partial export directory exists; inspect before retrying: $cell_dir" >&2
    exit 1
  else
    echo "EXPORT tuning routes: $cell"
    python -u -m opera.run.evaluate_direct_competing_risk \
      --checkpoint "$RUN/best.ckpt" \
      --config-dir "$RUN/hydra" --config-name config \
      --split tuning --horizons 365 --device cuda \
      --cohort-fine "$fine" --outcome "$outcome" \
      --export-routes --output-dir "$cell_dir"
    [[ -s "$routes" && -s "$summary" ]] || {
      echo "Route export did not produce the expected artifacts: $cell_dir" >&2
      exit 1
    }
  fi
  PLOT_ARGS+=(--cell "$cell" --routes "$routes")
done

python -m opera.run.plot_router_cell_borrowing \
  --config "$RUN/hydra/config.yaml" --checkpoint "$RUN/best.ckpt" \
  "${PLOT_ARGS[@]}" --output-dir "$FIGURE_DIR"
echo "Completed figure: $FIGURE_DIR/router_cell_borrowing.png"
