#!/usr/bin/env bash
# Matched, tuning-only predictions for both selected router checkpoints.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

CLEAN="${OPERA_CLEAN_ROUTER_RUN:-$REPO/data/models/approved_neural_router_clean_20261007_v1/time2vec_router_interacting_scale03_e2e}"
LOOSE="${OPERA_LOOSE_ROUTER_RUN:-$REPO/data/models/approved_neural_router_loose_20261010_v1/time2vec_router_interacting_loose_scale07_e2e}"
RELEASE="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}"
ROOT="${OPERA_ROUTER_PAIR_DIR:-$REPO/data/results/router_clean_loose_tuning_20261011_INTERNAL}"
export BONSAI_COHORT_MEMBERSHIP="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"
export CUDA_VISIBLE_DEVICES="${OPERA_LOCAL_GPU:-0}"

for input in "$CLEAN/best.ckpt" "$CLEAN/hydra/config.yaml" \
             "$LOOSE/best.ckpt" "$LOOSE/hydra/config.yaml" \
             "$RELEASE/experiment_registry_primary_v2.yaml" "$BONSAI_COHORT_MEMBERSHIP"; do
  [[ -s "$input" ]] || { echo "Missing or empty input: $input" >&2; exit 1; }
done
mkdir -p "$ROOT"
for name in clean loose; do
  if [[ "$name" == clean ]]; then run="$CLEAN"; else run="$LOOSE"; fi
  output="$ROOT/$name"
  if [[ -s "$output/direct_cr_predictions.parquet" && -s "$output/direct_cr_summary.json" ]]; then
    python - "$output/direct_cr_summary.json" "$run/best.ckpt" <<'PY'
import json,sys
from pathlib import Path
summary=json.loads(Path(sys.argv[1]).read_text())
if summary.get('split') != 'tuning' or Path(summary.get('checkpoint','')).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit('Existing prediction export has different split/checkpoint')
PY
    echo "REUSE $name tuning prediction export"
  elif [[ -e "$output" ]]; then
    echo "Partial export exists; inspect before retrying: $output" >&2; exit 1
  else
    python -u -m opera.run.evaluate_direct_competing_risk \
      --checkpoint "$run/best.ckpt" --config-dir "$run/hydra" --config-name config \
      --split tuning --horizons 30,60,90,180,365,730 --device cuda \
      --output-dir "$output"
  fi
done

COMPARE="$ROOT/paired_comparison"
[[ ! -e "$COMPARE" ]] || { echo "Comparison output exists: $COMPARE" >&2; exit 1; }
python -m opera.run.compare_approved_neural_tuning_predictions \
  --run "clean=$ROOT/clean/direct_cr_predictions.parquet" \
  --run "loose=$ROOT/loose/direct_cr_predictions.parquet" \
  --baseline clean --registry "$RELEASE/experiment_registry_primary_v2.yaml" \
  --family-map "$LOOSE/hydra/config.yaml" --horizon 365 \
  --skip-under-supported --output-dir "$COMPARE"
echo "Review: $COMPARE/tuning_forest_contrasts.csv"
