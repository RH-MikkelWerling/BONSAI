#!/usr/bin/env bash
# Dedicated free GPU allocation: replay first, then CPU diagnostics. No fitting.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
root=data/diagnostics/representation_absorption_20261112
plan="$root/replay_plan.json"
mkdir -p "$root" logs
exec 9> "$root/.overnight.lock"
flock -n 9 || { echo 'A representation diagnostic chain is already running'; exit 1; }
"$python_bin" scripts/check_results_resources.py --minimum-free-gib 24 --require-no-coordinator --require-free-gpu
if [[ ! -f "$plan" ]]; then
  "$python_bin" -m opera.run.prepare_representation_replay_plan \
    --national-root data/paper_256_v3_national_20261111 --output "$plan"
fi
modes=(full all)
if [[ "${1:-}" == --all-terms ]]; then
  modes=(full all own_disease own_outcome global)
fi
echo "Replay modes this pass: ${modes[*]}; completed verified passes are reused."
"$python_bin" -u -m opera.diagnostics.correction_replay \
  --plan "$plan" --output-dir "$root/replays" --split tuning --modes "${modes[@]}"
# Fresh output per completed pass set: never overwrite previous diagnostics.
label=core
if [[ "${1:-}" == --all-terms ]]; then label=all_terms; fi
if [[ -f "$root/tuning_$label/pipeline_complete.json" ]]; then
  echo "Diagnostics already present: $root/tuning_$label; preserve and inspect them."
  exit 0
fi
"$python_bin" scripts/check_results_resources.py --minimum-free-gib 48 --require-no-coordinator
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$python_bin" -u -m opera.analysis.representation_absorption \
  --analysis-config data/paper_256_v3_national_20261111/extended_tuning_comparison/analysis.yaml \
  --output-dir "$root/tuning_$label" \
  --replay-plan "$plan" --replay-root "$root/replays"
"$python_bin" - "$root/tuning_$label/pipeline_complete.json" <<'PYDONE'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'complete':True,'scope':'exploratory tuning diagnostics; no training launched'}))
PYDONE
echo "Diagnostic chain complete: $root/tuning_$label"
