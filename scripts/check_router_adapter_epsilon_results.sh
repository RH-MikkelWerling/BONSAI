#!/usr/bin/env bash
# After both fits complete: matched learning diagnostics and paired tuning predictions.
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in
  preflight|run) ;;
  *) echo "Usage: bash scripts/check_router_adapter_epsilon_results.sh [preflight|run]" >&2; exit 2 ;;
esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"
REFERENCE="${OPERA_ADAPTER_EPS_REFERENCE_RUN:-$REPO/data/models/approved_neural_router_guarded_20261011_v1/time2vec_router_interacting_guarded_scale07_e2e}"
CANDIDATE="${OPERA_EPS_ROUTER_RUN:-$REPO/data/models/approved_neural_router_adapter_epsilon_20261013_v1/time2vec_router_interacting_guarded_scale07_e2e_adaptereps_1e-8}"
ROOT="${OPERA_EPS_EVALUATION_DIR:-$REPO/data/results/router_adapter_epsilon_evaluation_20261013_INTERNAL}"
RELEASE="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}"
export BONSAI_COHORT_MEMBERSHIP="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"
export CUDA_VISIBLE_DEVICES="${OPERA_LOCAL_GPU:-0}"
for run in "$REFERENCE" "$CANDIDATE"; do
  for file in training_complete.json last.ckpt hydra/config.yaml; do
    [[ -s "$run/$file" ]] || { echo "Missing completed-run artifact: $run/$file" >&2; exit 1; }
  done
done
python -m opera.run.preflight_direct_competing_risk \
  --config-dir "$CANDIDATE/hydra" --config-name config --require-existing-paths \
  --adapter-epsilon-reference "$REFERENCE/hydra/config.yaml"
python - "$REFERENCE" "$CANDIDATE" <<'PY'
import sys, torch
from pathlib import Path
epochs = [torch.load(Path(run) / 'last.ckpt', map_location='cpu', weights_only=False)['epoch']
          for run in sys.argv[1:]]
if epochs[0] != epochs[1]:
    raise SystemExit(f'Last-checkpoint epochs differ: {epochs}; compare equal training budgets')
print('Matched last-checkpoint epoch:', epochs[0])
PY
if [[ "$MODE" == preflight ]]; then exit 0; fi
[[ ! -e "$ROOT" ]] || { echo "Output exists; choose a fresh OPERA_EPS_EVALUATION_DIR: $ROOT" >&2; exit 1; }
mkdir -p "$ROOT"
python -m opera.run.compare_approved_neural_validation_v1 \
  --run "baseline=$REFERENCE" --run "lower_adapter_epsilon=$CANDIDATE" \
  --output "$ROOT/training_curve_comparison.json"
python - "$ROOT/training_curve_comparison.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
report = json.loads(path.read_text())
delta = report['runs']['lower_adapter_epsilon']['common_epoch_auc'] - report['runs']['baseline']['common_epoch_auc']
report['common_epoch_auc_delta_lower_epsilon_minus_baseline'] = delta
path.write_text(json.dumps(report, indent=2) + '\n')
print('Common-epoch tuning AUC difference:', delta)
PY
python - "$REFERENCE" "$CANDIDATE" "$ROOT" <<'PY'
import json, sys
from pathlib import Path
from opera.run.inspect_router_source_optimizer import inspect_run
report = {'baseline': inspect_run(Path(sys.argv[1]), 'last'),
          'lower_adapter_epsilon': inspect_run(Path(sys.argv[2]), 'last')}
Path(sys.argv[3], 'optimizer_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
PY
for label in baseline lower_adapter_epsilon; do
  if [[ "$label" == baseline ]]; then run="$REFERENCE"; else run="$CANDIDATE"; fi
  python -m opera.run.audit_router_learning_dynamics \
    --run "$run" --checkpoint last --patients "${OPERA_AUDIT_PATIENTS:-256}" \
    --device cuda --output-dir "$ROOT/$label/learning"
  python -m opera.run.evaluate_direct_competing_risk \
    --checkpoint "$run/last.ckpt" --config-dir "$run/hydra" --config-name config \
    --split tuning --horizons 30,60,90,180,365,730 --device cuda \
    --output-dir "$ROOT/$label/predictions"
done
for horizon in 365 730; do
  python -m opera.run.compare_approved_neural_tuning_predictions \
    --run "baseline=$ROOT/baseline/predictions/direct_cr_predictions.parquet" \
    --run "lower_adapter_epsilon=$ROOT/lower_adapter_epsilon/predictions/direct_cr_predictions.parquet" \
    --baseline baseline --registry "$RELEASE/experiment_registry_primary_v2.yaml" \
    --family-map "$REFERENCE/hydra/config.yaml" --horizon "$horizon" \
    --skip-under-supported --output-dir "$ROOT/paired_${horizon}d"
done
echo "Optimizer comparison: $ROOT/optimizer_comparison.json"
echo "Outcome learning diagnostics: $ROOT/lower_adapter_epsilon/learning/source_diagnostics.csv"
echo "Paired tuning contrasts: $ROOT/paired_365d/tuning_forest_contrasts.csv"
