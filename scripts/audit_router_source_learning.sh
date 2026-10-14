#!/usr/bin/env bash
# Aggregate training-data diagnostics; no parameter updates.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"
export CUDA_VISIBLE_DEVICES="${OPERA_LOCAL_GPU:-0}"
export BONSAI_COHORT_MEMBERSHIP="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
export BONSAI_OUTCOMES_DIR="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}/outcomes"
ROOT="${OPERA_NEURAL_RUN_ROOT:-$REPO/data/models/approved_neural_router_guarded_source_controls_20261012_v1}"
OUT="${OPERA_ROUTER_LEARNING_AUDIT_DIR:-$REPO/data/results/router_source_learning_audit_20261012_INTERNAL}"
ARM="${1:-all}"
case "$ARM" in
  all) ARMS=(disease_only outcome_only none) ;;
  disease_only|outcome_only|none) ARMS=("$ARM") ;;
  *) echo "Usage: bash scripts/audit_router_source_learning.sh [all|disease_only|outcome_only|none]" >&2; exit 2 ;;
esac
for arm in "${ARMS[@]}"; do
  variant="time2vec_router_${arm}_e2e_guarded_scale07_e2e"
  if [[ "$arm" == none ]]; then variant=time2vec_router_no_sources_e2e_guarded_scale07_e2e; fi
  python -m opera.run.audit_router_learning_dynamics \
    --run "$ROOT/$variant" --checkpoint "${OPERA_AUDIT_CHECKPOINT:-best}" \
    --patients "${OPERA_AUDIT_PATIENTS:-256}" --device "${OPERA_AUDIT_DEVICE:-cuda}" \
    --output-dir "$OUT/$arm"
done
