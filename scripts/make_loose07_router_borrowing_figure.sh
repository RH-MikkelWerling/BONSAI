#!/usr/bin/env bash
# Reuse the same prespecified cells and plot code as the clean 0.3 figure.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPERA_ROUTER_RUN="${OPERA_ROUTER_RUN:-$REPO/data/models/approved_neural_router_loose_20261010_v1/time2vec_router_interacting_loose_scale07_e2e}"
export OPERA_ROUTER_EXPORT_ROOT="${OPERA_ROUTER_EXPORT_ROOT:-$REPO/data/results/router_loose07_cells_tuning_20261011_INTERNAL}"
export OPERA_ROUTER_FIGURE_DIR="${OPERA_ROUTER_FIGURE_DIR:-$REPO/data/results/router_loose07_cell_borrowing_20261011_INTERNAL}"
exec bash "$REPO/scripts/make_scale03_router_borrowing_figure.sh"
