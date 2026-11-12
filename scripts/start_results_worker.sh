#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
mkdir -p logs
if pgrep -u "$(id -u)" -f '[p]ython.*-m opera.run.analysis_cpu_worker' >/dev/null; then
  echo 'An original CPU worker is active on this allocation. Stop that worker before starting the multi-queue worker.' >&2
  exit 1
fi
log="logs/results_worker_$(hostname -s)_$(date +%Y%m%d_%H%M%S).log"
nohup "$python_bin" -u -m opera.run.results_worker \
  --checkpoint-dir data/paper_256_v3_national_20261111/extended_test_comparison/checkpoints \
  --checkpoint-dir data/paper_256_v3/analysis/local_test_20261109_four_allocations/_cpu_checkpoints_INTERNAL \
  --wait-hours 48 > "$log" 2>&1 &
echo "Multi-queue worker PID $!; log: $log"
