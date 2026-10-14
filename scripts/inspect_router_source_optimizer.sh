#!/usr/bin/env bash
# CPU-only checkpoint inspection; no patient data or training steps.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"
python -m opera.run.inspect_router_source_optimizer \
  --run-root "${OPERA_NEURAL_RUN_ROOT:-$REPO/data/models/approved_neural_router_guarded_source_controls_20261012_v1}" \
  --checkpoint "${OPERA_AUDIT_CHECKPOINT:-best}" \
  --output-dir "${OPERA_ROUTER_OPTIMIZER_AUDIT_DIR:-$REPO/data/results/router_source_optimizer_audit_20261012_INTERNAL}"
