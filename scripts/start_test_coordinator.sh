#!/usr/bin/env bash
# One dedicated high-RAM allocation. Requires successful checkpoint migration.
set -euo pipefail
cd "$(dirname "$0")/.."
repo_dir="$PWD"
python_bin="${OPERA_PYTHON:-$repo_dir/.venv-v100/bin/python}"
tag=local_test_20261109_four_allocations
root="$repo_dir/data/paper_256_v3/analysis/$tag"
mkdir -p logs
if [[ "${1:-}" != --run ]]; then
  "$python_bin" - "$root/_cpu_checkpoints_INTERNAL" <<'PY'
import json,sys
from pathlib import Path
from opera.evaluation.distributed_analysis import signature
import yaml
base=Path(sys.argv[1]); root=base.parent
config=yaml.safe_load((root/'analysis_inventory/analysis.yaml').read_text())
active=json.loads((base/'active.json').read_text())['generation']
if active != signature(config):
    raise SystemExit('Checkpoint generation differs from synced code/config. Complete migration before launching.')
if not (base/active/'prepared/prepared.json').is_file():
    raise SystemExit('Prepared checkpoint snapshot is missing')
PY
  log="logs/test_coordinator_$(date +%Y%m%d_%H%M%S).log"
  nohup bash "$0" --run > "$log" 2>&1 &
  echo "Coordinator wrapper PID $!; log: $log"
  exit
fi
exec 9> "$root/.coordinator_launcher.lock"
flock -n 9 || { echo 'A coordinator launcher is already active'; exit 1; }
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OPERA_ANALYSIS_SCRATCH="${OPERA_ANALYSIS_SCRATCH:-$root/.analysis_scratch}"
mkdir -p "$OPERA_ANALYSIS_SCRATCH"
exec "$python_bin" -u -m opera.run.test_cpu_pipeline \
  --tag "$tag" --role numerical --shards 4 \
  --analysis-workers 1 --summary-workers 1 --resume-analysis --distributed
