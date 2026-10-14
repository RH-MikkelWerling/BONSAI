#!/usr/bin/env bash
# Full disease-by-outcome tuning route atlas for the loose checkpoint.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPERA_ROUTER_RUN="${OPERA_ROUTER_RUN:-$REPO/data/models/approved_neural_router_loose_20261010_v1/time2vec_router_interacting_loose_scale07_e2e}"
export OPERA_ROUTER_ATLAS_DIR="${OPERA_ROUTER_ATLAS_DIR:-$REPO/data/results/router_loose07_full_borrowing_atlas_20261011_INTERNAL}"
exec bash "$REPO/scripts/make_scale03_router_full_atlas.sh"
