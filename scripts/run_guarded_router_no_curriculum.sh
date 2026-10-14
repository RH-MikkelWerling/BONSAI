#!/usr/bin/env bash
# Matched full borrowed-update router with all outcomes active from epoch zero.
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in
  preflight|launch|resume) ;;
  *) echo "Usage: bash scripts/run_guarded_router_no_curriculum.sh [preflight|launch|resume]" >&2; exit 2 ;;
esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for script in "$REPO/scripts/run_guarded_router_no_curriculum.sh" \
              "$REPO/scripts/run_router_guarded_scale07.sh" \
              "$REPO/scripts/run_approved_neural_candidate_v1.sh"; do
  bash -n "$script"
done
REFERENCE="${OPERA_BORROWED_ROUTER_RUN:-$REPO/data/models/approved_neural_router_borrowed_updates_20261012_v1/time2vec_router_interacting_guarded_scale07_e2e_borrowed_updates}"
[[ -s "$REFERENCE/hydra/config.yaml" ]] || { echo "Missing curriculum-on reference config: $REFERENCE/hydra/config.yaml" >&2; exit 1; }
unset OPERA_ROUTER_MATCH_REFERENCE
export OPERA_ROUTER_CURRICULUM_REFERENCE="$REFERENCE/hydra/config.yaml"
export OPERA_ROUTER_BORROWED_UPDATES=1
export OPERA_ROUTER_CURRICULUM=off
export OPERA_NEURAL_RUN_ROOT="${OPERA_NEURAL_RUN_ROOT:-$REPO/data/models/approved_neural_router_no_curriculum_20261012_v1}"
echo "Curriculum-off comparison against $REFERENCE; reference training may still be running."
exec bash "$REPO/scripts/run_router_guarded_scale07.sh" "$MODE"
