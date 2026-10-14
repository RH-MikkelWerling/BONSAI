#!/usr/bin/env bash
# Tuning-only comparison: loose rank 8, guarded rank 8, guarded rank 16.
# Does not submit a job or access held-out labels.
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in
  preflight|run) ;;
  *) echo "Usage: bash scripts/compare_router_scale07_finalists.sh [preflight|run]" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

LOOSE="${OPERA_LOOSE_ROUTER_RUN:-$REPO/data/models/approved_neural_router_loose_20261010_v1/time2vec_router_interacting_loose_scale07_e2e}"
GUARDED8="${OPERA_GUARDED8_ROUTER_RUN:-$REPO/data/models/approved_neural_router_guarded_20261011_v1/time2vec_router_interacting_guarded_scale07_e2e}"
GUARDED16="${OPERA_GUARDED16_ROUTER_RUN:-$REPO/data/models/approved_neural_router_guarded_rank16_20261011_v1/time2vec_router_interacting_guarded_scale07_rank16_e2e}"
ROOT="${OPERA_ROUTER_FINALIST_COMPARE_DIR:-$REPO/data/results/router_scale07_finalists_tuning_20261012_INTERNAL}"
RELEASE="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}"
export BONSAI_COHORT_MEMBERSHIP="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"
export CUDA_VISIBLE_DEVICES="${OPERA_LOCAL_GPU:-0}"

for run in "$LOOSE" "$GUARDED8" "$GUARDED16"; do
  for file in training_complete.json best.ckpt hydra/config.yaml; do
    [[ -s "$run/$file" ]] || {
      echo "Incomplete or missing run artifact: $run/$file" >&2; exit 1;
    }
  done
done
[[ -s "$RELEASE/experiment_registry_primary_v2.yaml" ]] || {
  echo "Missing approved outcome registry" >&2; exit 1;
}
[[ -s "$BONSAI_COHORT_MEMBERSHIP" ]] || {
  echo "Missing sealed cohort metadata" >&2; exit 1;
}

COMPARE_ARGS=(
  --run "loose_rank8=$LOOSE"
  --run "guarded_rank8=$GUARDED8"
  --run "guarded_rank16=$GUARDED16"
)
if [[ "$MODE" == preflight ]]; then
  python -m opera.run.compare_approved_neural_validation_v1 "${COMPARE_ARGS[@]}"
  echo "Preflight passed; run mode will export matched tuning predictions."
  exit 0
fi

mkdir -p "$ROOT"
if [[ -s "$ROOT/selection_comparison.json" ]]; then
  for run in "$LOOSE" "$GUARDED8" "$GUARDED16"; do
    [[ "$ROOT/selection_comparison.json" -nt "$run/best.ckpt" ]] || {
      echo "Cached selection comparison predates a checkpoint; use a new output directory" >&2; exit 1;
    }
  done
  echo "REUSE selection comparison"
elif [[ -e "$ROOT/selection_comparison.json" ]]; then
  echo "Empty selection comparison exists; inspect it before retrying" >&2; exit 1
else
  python -m opera.run.compare_approved_neural_validation_v1 \
    "${COMPARE_ARGS[@]}" --output "$ROOT/selection_comparison.json"
fi

for name in loose_rank8 guarded_rank8 guarded_rank16; do
  case "$name" in
    loose_rank8) run="$LOOSE" ;;
    guarded_rank8) run="$GUARDED8" ;;
    guarded_rank16) run="$GUARDED16" ;;
  esac
  output="$ROOT/$name"
  if [[ -s "$output/direct_cr_predictions.parquet" && -s "$output/direct_cr_summary.json" ]]; then
    [[ "$output/direct_cr_summary.json" -nt "$run/best.ckpt" ]] || {
      echo "Cached $name predictions predate its checkpoint; use a new output directory" >&2; exit 1;
    }
    python - "$output/direct_cr_summary.json" "$run/best.ckpt" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("split") != "tuning" or Path(summary.get("checkpoint", "")).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("Cached predictions belong to a different split or checkpoint")
PY
    echo "REUSE $name tuning predictions"
  elif [[ -e "$output" ]]; then
    echo "Partial export exists; inspect before retrying: $output" >&2; exit 1
  else
    python -u -m opera.run.evaluate_direct_competing_risk \
      --checkpoint "$run/best.ckpt" --config-dir "$run/hydra" --config-name config \
      --split tuning --horizons 30,60,90,180,365,730 --device cuda \
      --output-dir "$output"
  fi
done

FOREST="$ROOT/paired_outcome_comparison"
if [[ ! -e "$FOREST" ]]; then
  python -m opera.run.compare_approved_neural_tuning_predictions \
    --run "loose_rank8=$ROOT/loose_rank8/direct_cr_predictions.parquet" \
    --run "guarded_rank8=$ROOT/guarded_rank8/direct_cr_predictions.parquet" \
    --run "guarded_rank16=$ROOT/guarded_rank16/direct_cr_predictions.parquet" \
    --baseline loose_rank8 --registry "$RELEASE/experiment_registry_primary_v2.yaml" \
    --family-map "$LOOSE/hydra/config.yaml" --horizon 365 \
    --skip-under-supported --output-dir "$FOREST"
elif [[ -s "$FOREST/tuning_forest_estimates.csv" && -s "$FOREST/tuning_forest_contrasts.csv" ]]; then
  echo "REUSE paired outcome comparison"
else
  echo "Partial paired comparison exists; inspect before retrying: $FOREST" >&2; exit 1
fi

RANK_PAIR="$ROOT/paired_rank16_vs_rank8"
if [[ ! -e "$RANK_PAIR" ]]; then
  python -m opera.run.compare_approved_neural_tuning_predictions \
    --run "guarded_rank8=$ROOT/guarded_rank8/direct_cr_predictions.parquet" \
    --run "guarded_rank16=$ROOT/guarded_rank16/direct_cr_predictions.parquet" \
    --baseline guarded_rank8 --registry "$RELEASE/experiment_registry_primary_v2.yaml" \
    --family-map "$GUARDED8/hydra/config.yaml" --horizon 365 \
    --skip-under-supported --output-dir "$RANK_PAIR"
elif [[ -s "$RANK_PAIR/tuning_forest_estimates.csv" && -s "$RANK_PAIR/tuning_forest_contrasts.csv" ]]; then
  echo "REUSE direct rank comparison"
else
  echo "Partial rank comparison exists; inspect before retrying: $RANK_PAIR" >&2; exit 1
fi

echo "Primary tuning selection: $ROOT/selection_comparison.json"
echo "Paired outcome estimates: $FOREST/tuning_forest_estimates.csv"
echo "Paired outcome contrasts: $FOREST/tuning_forest_contrasts.csv"
echo "Direct rank effect: $RANK_PAIR/tuning_forest_contrasts.csv"
