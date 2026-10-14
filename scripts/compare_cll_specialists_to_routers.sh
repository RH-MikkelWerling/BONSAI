#!/usr/bin/env bash
# Preliminary tuning-only CLL specialist versus pooled-router comparison.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

CLEAN="${OPERA_CLEAN_ROUTER_RUN:-$REPO/data/models/approved_neural_router_clean_20261007_v1/time2vec_router_interacting_scale03_e2e}"
LOOSE="${OPERA_LOOSE_ROUTER_RUN:-$REPO/data/models/approved_neural_router_loose_20261010_v1/time2vec_router_interacting_loose_scale07_e2e}"
SPECIALISTS="${OPERA_SPECIALIST_ROOT:-$REPO/data/models/cll_independent_specialists_20261010_v2}"
ROOT="${OPERA_CLL_COMPARISON_DIR:-$REPO/data/results/cll_specialist_router_comparison_20261011_INTERNAL}"
RELEASE="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}"
export BONSAI_COHORT_MEMBERSHIP="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"
export CUDA_VISIBLE_DEVICES="${OPERA_LOCAL_GPU:-0}"
for input in "$CLEAN/best.ckpt" "$LOOSE/best.ckpt" \
             "$CLEAN/hydra/config.yaml" "$LOOSE/hydra/config.yaml" \
             "$BONSAI_COHORT_MEMBERSHIP"; do
  [[ -s "$input" ]] || { echo "Missing input: $input" >&2; exit 1; }
done

read -r -a OUTCOMES <<<"${OPERA_CLL_OUTCOMES:-overall_survival sepsis aki_first_documented}"
[[ ${#OUTCOMES[@]} -gt 0 ]] || { echo "No outcomes selected" >&2; exit 2; }
mkdir -p "$ROOT"
for outcome in "${OUTCOMES[@]}"; do
  [[ "$outcome" =~ ^[A-Za-z0-9_]+$ ]] || { echo "Unsafe outcome: $outcome" >&2; exit 2; }
  specialist="$SPECIALISTS/runs/cll_$outcome"
  if [[ ! -s "$specialist/training_complete.json" || ! -s "$specialist/best.ckpt" ]]; then
    echo "SKIP no completed specialist: CLL:$outcome"
    continue
  fi
  [[ -s "$specialist/hydra/config.yaml" ]] || {
    echo "Missing saved specialist config: $specialist" >&2; exit 1;
  }
  cell="$ROOT/$outcome"
  mkdir -p "$cell"
  for name in specialist clean loose; do
    case "$name" in
      specialist) run="$specialist" ;;
      clean) run="$CLEAN" ;;
      loose) run="$LOOSE" ;;
    esac
    output="$cell/$name"
    if [[ -s "$output/direct_cr_predictions.parquet" && -s "$output/direct_cr_summary.json" ]]; then
      python - "$output/direct_cr_summary.json" "$run/best.ckpt" <<'PY'
import json,sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text())
if s.get('split')!='tuning' or Path(s.get('checkpoint','')).resolve()!=Path(sys.argv[2]).resolve():
    raise SystemExit('Existing CLL export has different split/checkpoint')
PY
      echo "REUSE $name CLL:$outcome predictions"
    elif [[ -e "$output" ]]; then
      echo "Partial output: $output" >&2; exit 1
    else
      python -u -m opera.run.evaluate_direct_competing_risk \
        --checkpoint "$run/best.ckpt" --config-dir "$run/hydra" --config-name config \
        --split tuning --horizons 30,60,90,180,365,730 --device cuda \
        --parent-cohort CLL_SLL --cohort-fine CLL --outcome "$outcome" \
        --output-dir "$output"
    fi
  done
  comparison="$cell/comparison"
  if [[ -e "$comparison" ]]; then
    echo "Comparison exists; inspect rather than overwrite: $comparison" >&2; exit 1
  fi
  python -m opera.run.compare_cll_specialist_tuning \
    --outcome "$outcome" \
    --specialist "$cell/specialist/direct_cr_predictions.parquet" \
    --clean "$cell/clean/direct_cr_predictions.parquet" \
    --loose "$cell/loose/direct_cr_predictions.parquet" \
    --output-dir "$comparison"
done
echo "CLL preliminary comparison: $ROOT"
