#!/usr/bin/env bash
# Tuning-only matched outcome selection, reusing completed prediction/route exports.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

BASELINE="${OPERA_BASELINE_TUNING_PREDICTIONS:?Set OPERA_BASELINE_TUNING_PREDICTIONS to the matched plain Time2Vec prediction Parquet}"
ROUTER="${OPERA_SCALE03_TUNING_PREDICTIONS:?Set OPERA_SCALE03_TUNING_PREDICTIONS to the scale-0.3 tuning prediction Parquet}"
RUN="${OPERA_ROUTER_RUN:-$REPO/data/models/approved_neural_router_clean_20261007_v1/time2vec_router_interacting_scale03_e2e}"
OUT="${OPERA_GAIN_AUDIT_DIR:-$REPO/data/results/router_scale03_gain_mechanism_20261010_INTERNAL}"

for input in "$BASELINE" "$ROUTER" "$RUN/hydra/config.yaml"; do
  [[ -s "$input" ]] || { echo "Missing or empty input: $input" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "Audit output exists: $OUT" >&2; exit 1; }

ARGS=(--baseline-predictions "$BASELINE" --router-predictions "$ROUTER"
      --family-map "$RUN/hydra/config.yaml" --horizon "${OPERA_AUDIT_HORIZON:-365}"
      --bootstrap "${OPERA_AUDIT_BOOTSTRAP:-400}" --output-dir "$OUT")

# Optional, space-separated route exports from the same scale-0.3 checkpoint.
if [[ -n "${OPERA_SCALE03_ROUTE_EXPORTS:-}" ]]; then
  read -r -a ROUTE_FILES <<<"$OPERA_SCALE03_ROUTE_EXPORTS"
  for route in "${ROUTE_FILES[@]}"; do
    [[ -s "$route" ]] || { echo "Missing route export: $route" >&2; exit 1; }
    ARGS+=(--routes "$route")
  done
fi

python -m opera.run.audit_router_gain_mechanism "${ARGS[@]}"
echo "Review: $OUT/selected_mechanism_outcomes.csv"
echo "Review: $OUT/selected_route_patient_variation.csv"
