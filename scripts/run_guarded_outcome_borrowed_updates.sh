#!/usr/bin/env bash
# One matched experiment: outcome source parameters learn from borrowing targets.
# All other settings, including the existing curriculum, match outcome-only.
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in
  preflight|launch|resume) ;;
  *) echo "Usage: bash scripts/run_guarded_outcome_borrowed_updates.sh [preflight|launch|resume]" >&2; exit 2 ;;
esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPERA_ROUTER_BORROWED_UPDATES=1
echo "Outcome-only borrowed-update comparison; curriculum matches the existing arm."
exec bash "$REPO/scripts/run_guarded_router_source_arm.sh" outcome_only "$MODE"
