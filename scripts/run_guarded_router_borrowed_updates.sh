#!/usr/bin/env bash
# Full interacting router: both borrowed banks receive parameter gradients.
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in
  preflight|launch|resume) ;;
  *) echo "Usage: bash scripts/run_guarded_router_borrowed_updates.sh [preflight|launch|resume]" >&2; exit 2 ;;
esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for script in "$REPO/scripts/run_guarded_router_borrowed_updates.sh" \
              "$REPO/scripts/run_router_guarded_scale07.sh" \
              "$REPO/scripts/run_approved_neural_candidate_v1.sh"; do
  bash -n "$script"
done
REFERENCE="${OPERA_LOCKED_ROUTER_RUN:-$REPO/data/models/approved_neural_router_guarded_20261011_v1/time2vec_router_interacting_guarded_scale07_e2e}"
for input in "$REFERENCE/hydra/config.yaml" "$REFERENCE/training_complete.json" "$REFERENCE/best.ckpt"; do
  [[ -s "$input" ]] || { echo "Missing complete guarded reference: $input" >&2; exit 1; }
done
export OPERA_ROUTER_BORROWED_UPDATES=1
export OPERA_ROUTER_CURRICULUM=inherit
unset OPERA_ROUTER_CURRICULUM_REFERENCE
export OPERA_ROUTER_MATCH_REFERENCE="$REFERENCE/hydra/config.yaml"
export OPERA_NEURAL_RUN_ROOT="${OPERA_NEURAL_RUN_ROOT:-$REPO/data/models/approved_neural_router_borrowed_updates_20261012_v1}"
echo "Full interacting borrowed-update comparison; preflight verifies all scientific settings against $REFERENCE."
exec bash "$REPO/scripts/run_router_guarded_scale07.sh" "$MODE"
