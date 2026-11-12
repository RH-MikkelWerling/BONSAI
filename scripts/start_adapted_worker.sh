#!/usr/bin/env bash
# One per single-GPU allocation: CPU analysis worker + adapted GPU training.
# GPU jobs claim fits dynamically; no shard numbers are needed.
set -euo pipefail
cd "$(dirname "$0")/.."
repo_dir="$PWD"
python_bin="${OPERA_PYTHON:-$repo_dir/.venv-v100/bin/python}"
national="$repo_dir/data/paper_256_v3_national_20261111"
checkpoint="$repo_dir/data/paper_256_v3/analysis/local_test_20261109_four_allocations/_cpu_checkpoints_INTERNAL"
host_name="$(hostname -s)"
mkdir -p logs
if [[ ! -f "$national/adapted_fits/bundle.json" ]]; then
  echo 'First audit/release the adapted_fits bundle with prepare_adapted_fit_release' >&2
  exit 1
fi
if [[ "${1:-}" != --run ]]; then
  "$python_bin" - "$checkpoint" <<'PYCODE'
import json,sys
from pathlib import Path
import yaml
from opera.evaluation.distributed_analysis import signature
base=Path(sys.argv[1])
config=yaml.safe_load((base.parent/'analysis_inventory/analysis.yaml').read_text())
active=json.loads((base/'active.json').read_text())['generation']
if active != signature(config):
    raise SystemExit('Complete checkpoint migration for the synced analysis code before launching workers.')
PYCODE
  log="logs/adapted_worker_${host_name}_$(date +%Y%m%d_%H%M%S).log"
  nohup bash "$0" --run > "$log" 2>&1 &
  echo "Worker wrapper PID $!; log: $log"
  exit
fi
exec 9> "$national/.worker_launcher_${host_name}.lock"
flock -n 9 || { echo 'A worker launcher is already active on this host'; exit 1; }
if pgrep -u "$(id -u)" -f '[p]ython.*-m (bonsai.run.pretrain|opera.run.contrastive_multicohort|opera.run.run_paper_queue)' >/dev/null; then
  echo 'GPU training already active on this host; inspect before starting another runner' >&2
  exit 1
fi
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
if pgrep -u "$(id -u)" -f '[p]ython.*-m opera.run.analysis_cpu_worker.*local_test_20261109_four_allocations' >/dev/null; then
  echo 'Existing CPU analysis worker detected; reusing it (must have been restarted after code sync)'
else
  cpu_log="$repo_dir/logs/test_cpu_worker_${host_name}_$(date +%Y%m%d_%H%M%S).log"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup "$python_bin" -u -m opera.run.analysis_cpu_worker \
      --checkpoint-dir "$checkpoint" --wait-hours 48 > "$cpu_log" 2>&1 9>&- &
  echo "CPU worker PID $!; log: $cpu_log"
fi
# Full-data jobs first; a free worker can take a fraction job while the last
# full-data fits train on other allocations. They use separate run claims.
for phase in external data_efficiency; do
  echo "Starting GPU queue phase: $phase"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$python_bin" -u -m opera.run.run_paper_queue \
      --bundle-dir "$national/adapted_fits" \
      --runs-dir "$national/neural_runs" \
      --phase "$phase" --external-hidden-size 256 --execute
done
echo 'GPU queue finished or remaining fits claimed elsewhere. CPU worker continues independently.'
