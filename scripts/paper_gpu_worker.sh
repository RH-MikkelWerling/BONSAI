#!/usr/bin/env bash
# One GPU worker for the shared paper to-do list. Call this from each qsub job
# (one per GPU/terminal); every worker claims the next unclaimed fit in priority
# order, trains it, and continues until nothing runnable is left.
#
#   bash scripts/paper_gpu_worker.sh            # run until the list is empty
#   WAIT=1 bash scripts/paper_gpu_worker.sh     # keep polling for unblocked work
#   bash scripts/run_paper_experiments.sh queue-status   # overview, any terminal
#
# The scheduler's CUDA_VISIBLE_DEVICES is respected. Killed jobs leave a claim
# whose heartbeat is reported stale after 30 minutes. Confirm the allocation
# has stopped before clearing the claim; restarting resumes its own last.ckpt. Activate the server Python environment first or
# set OPERA_PYTHON.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${WORKER_NAME:-$(hostname)-gpu${CUDA_VISIBLE_DEVICES:-x}-$$}"
ARGS=(worker --execute --worker-name "$NAME")
[[ "${WAIT:-0}" == 1 ]] && ARGS+=(--wait)
exec bash scripts/run_paper_experiments.sh "${ARGS[@]}" "$@"
