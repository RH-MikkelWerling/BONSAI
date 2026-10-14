#!/usr/bin/env bash
# Plot every tuning patient's source weights from existing route exports. No GPU work.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

RUN="${OPERA_ROUTER_RUN:-$REPO/data/models/approved_neural_router_loose_20261010_v1/time2vec_router_interacting_loose_scale07_e2e}"
EXPORT_ROOT="${OPERA_ROUTER_EXPORT_ROOT:-$REPO/data/results/router_loose07_cells_tuning_20261011_INTERNAL}"
OUTPUT="${OPERA_ROUTER_DISTRIBUTION_DIR:-$REPO/data/results/router_loose07_patient_weight_distributions_20261011_INTERNAL}"
CELLS_TEXT="${OPERA_ROUTER_CELLS:-CLL:overall_survival CLL:sepsis DLBCL:overall_survival MM:aki_first_documented}"
read -r -a CELLS <<<"$CELLS_TEXT"
[[ ${#CELLS[@]} -gt 0 ]] || { echo "No cells supplied" >&2; exit 2; }
[[ -s "$RUN/best.ckpt" && -s "$RUN/hydra/config.yaml" ]] || {
  echo "Missing completed router checkpoint or saved config: $RUN" >&2; exit 1;
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Output already exists; choose a new OPERA_ROUTER_DISTRIBUTION_DIR: $OUTPUT" >&2; exit 1;
}

ARGS=()
for cell in "${CELLS[@]}"; do
  [[ "$cell" =~ ^[A-Za-z0-9_]+:[A-Za-z0-9_]+$ ]] || {
    echo "Invalid cell: $cell" >&2; exit 2;
  }
  fine="${cell%%:*}"
  outcome="${cell#*:}"
  routes="$EXPORT_ROOT/${fine}__${outcome}/source_route_weights.parquet"
  [[ -s "$routes" ]] || { echo "Missing existing tuning route export: $routes" >&2; exit 1; }
  ARGS+=(--cell "$cell" --routes "$routes")
done

exec python -m opera.run.plot_router_weight_distributions \
  --config "$RUN/hydra/config.yaml" --checkpoint "$RUN/best.ckpt" \
  "${ARGS[@]}" --output-dir "$OUTPUT"
