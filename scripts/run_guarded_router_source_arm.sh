#!/usr/bin/env bash
# One-GPU matched source-bank control for the locked guarded rank-8 router.
# First run: bash scripts/run_guarded_router_source_arm.sh prepare
# Then, on separate GPU allocations: ... disease_only|outcome_only|none launch
set -euo pipefail

ARM="${1:?Usage: bash scripts/run_guarded_router_source_arm.sh [prepare|disease_only|outcome_only|none] [preflight|launch|resume]}"
MODE="${2:-preflight}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"

export OPERA_FAMILY_DIR="${OPERA_FAMILY_DIR:-$REPO/data/results/approved_training_families_v1_20261006T052120Z}"
export OPERA_CANDIDATE_DIR="${OPERA_CANDIDATE_DIR:-$REPO/data/results/approved_neural_source_controls_20261012_v1}"
export OPERA_NEURAL_RUN_ROOT="${OPERA_NEURAL_RUN_ROOT:-$REPO/data/models/approved_neural_router_guarded_source_controls_20261012_v1}"
REFERENCE="${OPERA_LOCKED_ROUTER_RUN:-$REPO/data/models/approved_neural_router_guarded_20261011_v1/time2vec_router_interacting_guarded_scale07_e2e}"
RELEASE="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}"

[[ -s "$REFERENCE/training_complete.json" && -s "$REFERENCE/best.ckpt" && -s "$REFERENCE/hydra/config.yaml" ]] || {
  echo "Locked guarded rank-8 router is not complete: $REFERENCE" >&2; exit 1;
}

if [[ "$ARM" == prepare ]]; then
  [[ $# -eq 1 ]] || { echo "prepare takes no second argument" >&2; exit 2; }
  [[ ! -e "$OPERA_CANDIDATE_DIR" ]] || {
    echo "Candidate directory exists; inspect it rather than overwriting: $OPERA_CANDIDATE_DIR" >&2; exit 1;
  }
  python -m opera.run.prepare_approved_neural_runs_v1 \
    --release-dir "$RELEASE" --family-dir "$OPERA_FAMILY_DIR" \
    --output-dir "$OPERA_CANDIDATE_DIR"
  echo "Candidate source-bank configs: $OPERA_CANDIDATE_DIR"
  exit 0
fi

case "$ARM" in
  disease_only) VARIANT=time2vec_router_disease_only_e2e ;;
  outcome_only) VARIANT=time2vec_router_outcome_only_e2e ;;
  none) VARIANT=time2vec_router_no_sources_e2e ;;
  *) echo "Unknown source-bank arm: $ARM" >&2; exit 2 ;;
esac
case "$MODE" in
  preflight|launch|resume) ;;
  *) echo "Mode must be preflight, launch, or resume" >&2; exit 2 ;;
esac
[[ -s "$OPERA_CANDIDATE_DIR/$VARIANT.yaml" ]] || {
  echo "Missing candidate config. Run the prepare command first: $VARIANT" >&2; exit 1;
}

# Fail if the chosen reference is not the exact configuration being ablated.
python - "$REFERENCE/hydra/config.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
risk = c["competing_risk"]
expected = {"adapter_residual_scale": .7, "adapter_home_floor": .05,
            "adapter_event_tau": 10., "adapter_bottleneck": 8,
            "adapter_shared_logit_init": 0.}
for key, value in expected.items():
    if float(risk.get(key, 3. if key == "adapter_shared_logit_init" else -1)) != value:
        raise SystemExit(f"Reference is not the locked guarded rank-8 router: {key}")
if risk.get("adapter_source_mode", "both") != "both" or risk.get("adapter_query_mode") != "interacting":
    raise SystemExit("Reference must use both source banks and the interacting query")
PY

export OPERA_ROUTER_STRESS=guarded_scale07
export OPERA_SAMPLING_ABLATION=none
case "$MODE" in
  preflight) export OPERA_LAUNCH=0 OPERA_RESUME=0 ;;
  launch) export OPERA_LAUNCH=1 OPERA_RESUME=0 ;;
  resume) export OPERA_LAUNCH=1 OPERA_RESUME=1 ;;
esac

echo "Source-bank arm: $ARM"
echo "Locked reference: $REFERENCE"
SUFFIX=""
if [[ "${OPERA_ROUTER_BORROWED_UPDATES:-0}" == 1 ]]; then SUFFIX=_borrowed_updates; fi
echo "Run: $OPERA_NEURAL_RUN_ROOT/${VARIANT}_guarded_scale07_e2e${SUFFIX}"
exec bash "$REPO/scripts/run_approved_neural_candidate_v1.sh" "$VARIANT"
