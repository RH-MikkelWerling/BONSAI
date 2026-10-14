#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"
RUN="${OPERA_BORROWED_ROUTER_RUN:-$REPO/data/models/approved_neural_router_borrowed_updates_20261012_v1/time2vec_router_interacting_guarded_scale07_e2e_borrowed_updates}"
python -m opera.run.verify_borrowed_update_checkpoint --run "$RUN" \
  --checkpoint "${OPERA_AUDIT_CHECKPOINT:-best}"
