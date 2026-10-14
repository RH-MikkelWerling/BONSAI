#!/usr/bin/env bash
# One matched epsilon experiment; only source down/up matrices use lower epsilon.
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in
  preflight|launch|resume) ;;
  *) echo "Usage: bash scripts/run_guarded_router_adapter_epsilon.sh [preflight|launch|resume]" >&2; exit 2 ;;
esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"
for script in "$REPO/scripts/run_guarded_router_adapter_epsilon.sh" \
              "$REPO/scripts/run_approved_neural_candidate_v1.sh"; do
  bash -n "$script"
done
REFERENCE="${OPERA_ADAPTER_EPS_REFERENCE_RUN:-$REPO/data/models/approved_neural_router_guarded_20261011_v1/time2vec_router_interacting_guarded_scale07_e2e}"
[[ -s "$REFERENCE/hydra/config.yaml" ]] || { echo "Missing epsilon reference config: $REFERENCE/hydra/config.yaml" >&2; exit 1; }
REFERENCE_SETTINGS="$(python - "$REFERENCE/hydra/config.yaml" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
risk = config['competing_risk']
mode = risk.get('readout_mode')
sources = risk.get('adapter_source_mode', 'both')
if mode not in {'source_anchored_adapters', 'unanchored_adapters'} or sources not in {'both', 'disease_only', 'outcome_only'}:
    raise SystemExit('Epsilon reference must be a source router with a nonempty bank')
enabled = risk.get('curriculum', {}).get('enabled')
if enabled not in (True, False):
    raise SystemExit('Reference must state whether curriculum is enabled')
print('|'.join([sources, '1' if mode == 'unanchored_adapters' else '0', 'inherit' if enabled else 'off']))
PY
)"
IFS='|' read -r SOURCES OPERA_ROUTER_BORROWED_UPDATES OPERA_ROUTER_CURRICULUM <<<"$REFERENCE_SETTINGS"
export OPERA_ROUTER_BORROWED_UPDATES OPERA_ROUTER_CURRICULUM
case "$SOURCES" in
  both) VARIANT=time2vec_router_interacting_scale03_e2e ;;
  disease_only) VARIANT=time2vec_router_disease_only_e2e ;;
  outcome_only) VARIANT=time2vec_router_outcome_only_e2e ;;
esac
unset OPERA_ROUTER_MATCH_REFERENCE OPERA_ROUTER_CURRICULUM_REFERENCE
export OPERA_ROUTER_ADAPTER_EPSILON_REFERENCE="$REFERENCE/hydra/config.yaml"
export OPERA_ROUTER_ADAPTER_EPSILON="${OPERA_ROUTER_ADAPTER_EPSILON:-1e-8}"
export OPERA_ROUTER_STRESS=guarded_scale07
export OPERA_SAMPLING_ABLATION=none
export OPERA_FAMILY_DIR="${OPERA_FAMILY_DIR:-$REPO/data/results/approved_training_families_v1_20261006T052120Z}"
export OPERA_CANDIDATE_DIR="${OPERA_CANDIDATE_DIR:-$REPO/data/results/approved_neural_source_controls_20261012_v1}"
export OPERA_NEURAL_RUN_ROOT="${OPERA_NEURAL_RUN_ROOT:-$REPO/data/models/approved_neural_router_adapter_epsilon_20261013_v1}"
case "$MODE" in
  preflight) export OPERA_LAUNCH=0 OPERA_RESUME=0 ;;
  launch) export OPERA_LAUNCH=1 OPERA_RESUME=0 ;;
  resume) export OPERA_LAUNCH=1 OPERA_RESUME=1 ;;
esac
echo "Adapter-matrix epsilon $OPERA_ROUTER_ADAPTER_EPSILON; reference $REFERENCE; sources $SOURCES; borrowed updates $OPERA_ROUTER_BORROWED_UPDATES; curriculum $OPERA_ROUTER_CURRICULUM."
exec bash "$REPO/scripts/run_approved_neural_candidate_v1.sh" "$VARIANT"
