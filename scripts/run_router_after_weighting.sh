#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
OUTPUT_ROOT="${1:?Usage: bash scripts/run_router_after_weighting.sh OUTPUT_ROOT [CURRENT_RUN_DIR]}"
CURRENT_RUN="${2:-$OUTPUT_ROOT/weightingreference__direct_cr_bce_lowrank_weighting_reference_v1__seed${OPERA_SEED:-42}}"
: "${OPERA_SHARED_BCE_CKPT:?Set the completed shared horizon-BCE best.ckpt (not a low-rank checkpoint)}"
[[ -s "$OPERA_SHARED_BCE_CKPT" && -d "$CURRENT_RUN" ]] || { echo "Missing source checkpoint or current run directory" >&2; exit 1; }
"${PYTHON_BIN:-python}" - "$OPERA_SHARED_BCE_CKPT" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
risk = checkpoint.get("hyper_parameters", {}).get("model_init_config", {}).get("competing_risk_config", {})
if risk.get("objective") != "horizon_bce" or risk.get("readout_mode", "shared") != "shared":
    raise SystemExit("Router requires a shared horizon-BCE source, not a low-rank checkpoint")
print("Shared horizon-BCE source type verified", flush=True)
PY
echo "Waiting for successful completion and launch-lock release: $CURRENT_RUN"
DEADLINE=$((SECONDS + ${OPERA_QUEUE_TIMEOUT_SECONDS:-43200}))
while [[ ! -f "$CURRENT_RUN/training_complete.json" || -d "$CURRENT_RUN/.launch_lock" ]]; do
  (( SECONDS < DEADLINE )) || { echo "Timed out; router NOT launched" >&2; exit 1; }
  sleep 30
done
echo "Current run completed; launching endpoint-group router"
bash scripts/run_opera_independent_experiment.sh routergroups "$OUTPUT_ROOT"
