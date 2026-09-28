#!/usr/bin/env bash
set -euo pipefail

# Reuse one fixed projection and one completed prediction export. CPU only.
# Usage: bash scripts/plot_pooled_bce_censoring_maps.sh COORDINATES PREDICTIONS OUTPUT_ROOT [HORIZON]
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
COORDINATES="${1:?Provide patient_umap_coordinates.csv}"
PREDICTIONS="${2:?Provide tuning/direct_cr_predictions.parquet}"
OUTPUT_ROOT="${3:?Provide a dedicated map output directory}"
HORIZON="${4:-365}"
BANDWIDTH_SCALE="${BANDWIDTH_SCALE:-1.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
[[ "$HORIZON" =~ ^[0-9]+$ ]] || { echo "Horizon must be an integer in days" >&2; exit 2; }
for INPUT in "$COORDINATES" "$PREDICTIONS"; do
  [[ -s "$INPUT" ]] || { echo "Missing or empty input: $INPUT" >&2; exit 1; }
done
OUTCOMES=(
  creatinine_increased_g2plus
  anemia_g3plus
  overall_survival
  lymphopenia_g2plus
  lymphopenia_g3plus
  infections_all_definitions
)
for OUTCOME in "${OUTCOMES[@]}"; do
  echo "[$(date -Is)] Rendering all-cohort tuning map: $OUTCOME at ${HORIZON}d"
  "$PYTHON_BIN" -u -m opera.run.plot_censoring_aware_map \
    --coordinates "$COORDINATES" \
    --predictions "$PREDICTIONS" \
    --outcome "$OUTCOME" \
    --cohort-level all --split tuning --horizon "$HORIZON" \
    --bandwidth-scale "$BANDWIDTH_SCALE" --residual-probability-limit 0.2 \
    --output-dir "$OUTPUT_ROOT/${OUTCOME}_${HORIZON}d"
done
echo "[$(date -Is)] All six pooled maps written to $OUTPUT_ROOT"
