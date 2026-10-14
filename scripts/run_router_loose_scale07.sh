#!/usr/bin/env bash
# Bounded exploratory Time2Vec router stress pilot. Does not submit a cluster job.
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in
  preflight|launch|resume) ;;
  *) echo "Usage: bash scripts/run_router_loose_scale07.sh [preflight|launch|resume]" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

export OPERA_FAMILY_DIR="${OPERA_FAMILY_DIR:-$REPO/data/results/approved_training_families_v1_20261006T052120Z}"
export OPERA_CANDIDATE_DIR="${OPERA_CANDIDATE_DIR:-$REPO/data/results/approved_neural_candidates_router_clean_20261007_v1}"
export OPERA_NEURAL_RUN_ROOT="${OPERA_NEURAL_RUN_ROOT:-$REPO/data/models/approved_neural_router_loose_20261010_v1}"
export OPERA_ROUTER_STRESS=loose_scale07
export OPERA_SAMPLING_ABLATION=none

[[ -s "$OPERA_CANDIDATE_DIR/time2vec_router_interacting_scale03_e2e.yaml" ]] || {
  echo "Missing clean scale-0.3 candidate config: $OPERA_CANDIDATE_DIR" >&2; exit 1;
}
[[ -s "$OPERA_FAMILY_DIR/outcome_training_families_primary_v1.yaml" ]] || {
  echo "Missing approved training-family directory: $OPERA_FAMILY_DIR" >&2; exit 1;
}

case "$MODE" in
  preflight) export OPERA_LAUNCH=0 OPERA_RESUME=0 ;;
  launch) export OPERA_LAUNCH=1 OPERA_RESUME=0 ;;
  resume) export OPERA_LAUNCH=1 OPERA_RESUME=1 ;;
esac

echo "Mode: $MODE"
echo "Run: $OPERA_NEURAL_RUN_ROOT/time2vec_router_interacting_loose_scale07_e2e"
exec bash "$REPO/scripts/run_approved_neural_candidate_v1.sh" time2vec_router_interacting_scale03_e2e
