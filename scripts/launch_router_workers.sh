#!/usr/bin/env bash
# Start one paper worker per GPU on this node. Each worker claims the next
# unclaimed fit (highest priority first; stage 8 in its queue order), trains it,
# and claims the next one when it finishes, until nothing runnable is left.
# Workers on other nodes coordinate through the same claim files.
#
#   bash scripts/launch_router_workers.sh 0 1 2 3      # GPU indices on this node
#   MAX_RUNS=5 bash scripts/launch_router_workers.sh 0 1   # optional cap per worker
#
# Stop every worker on this node:  pkill -f "worker --worker-name pf_$(hostname -s)_"
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [ "$#" -eq 0 ]; then
  echo "Usage: bash scripts/launch_router_workers.sh <gpu> [<gpu> ...]" >&2
  exit 1
fi
if [ ! -f data/paper_256_v3/bundles/stage8/bundle.json ]; then
  echo "Prepare stage 8 first: bash scripts/run_paper_experiments.sh prepare --stage 8 --execute" >&2
  exit 1
fi
export OPERA_PYTHON="${OPERA_PYTHON:-$repo_dir/.venv-v100/bin/python}"
mkdir -p logs
host="$(hostname -s)"
for gpu in "$@"; do
  name="pf_${host}_gpu${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup bash scripts/run_paper_experiments.sh worker \
    --worker-name "$name" --execute ${MAX_RUNS:+--max-runs "$MAX_RUNS"} \
    > "logs/worker_${name}.log" 2>&1 &
  echo "started $name (pid $!) -> logs/worker_${name}.log"
done
