#!/usr/bin/env bash
# Render the publication atlas from an existing tuning-only aggregate cube.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

ATLAS="${OPERA_ROUTER_ATLAS_DIR:-$REPO/data/results/router_loose07_full_borrowing_atlas_20261011_INTERNAL}"
RUN="${OPERA_ROUTER_RUN:-$REPO/data/models/approved_neural_router_loose_20261010_v1/time2vec_router_interacting_loose_scale07_e2e}"
OUT="${OPERA_ROUTER_PUBLICATION_DIR:-$REPO/data/results/router_loose07_publication_atlas_20261011_INTERNAL}"

[[ -s "$ATLAS/router_route_cube_INTERNAL.npz" ]] || {
  echo "Missing aggregate cube: $ATLAS/router_route_cube_INTERNAL.npz" >&2; exit 1;
}
[[ -s "$RUN/hydra/config.yaml" ]] || {
  echo "Missing matching run config: $RUN/hydra/config.yaml" >&2; exit 1;
}
[[ ! -e "$OUT" ]] || {
  echo "Output already exists; set OPERA_ROUTER_PUBLICATION_DIR to a new path: $OUT" >&2; exit 1;
}

python -m opera.run.plot_router_publication_atlas \
  --cube "$ATLAS/router_route_cube_INTERNAL.npz" \
  --config "$RUN/hydra/config.yaml" \
  --min-patients "${OPERA_ATLAS_MIN_PATIENTS:-20}" \
  --output-dir "$OUT"

echo "Publication atlas: $OUT/router_borrowing_publication_atlas.pdf"
