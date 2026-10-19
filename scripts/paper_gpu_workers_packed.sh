#!/usr/bin/env bash
# Several paper workers sharing one GPU (packing). Small fits (specialists,
# small outcome-pooled fits) leave a V100 mostly idle during start-up, data
# loading, validation and checkpoint writes; two to four of them side by side
# finish two to three times as many fits per hour. Large fits gain little but
# lose little. Every worker claims fits from the same to-do list as usual.
#
#   PACK=2 bash scripts/paper_gpu_workers_packed.sh     # from each GPU job
#   PACK=3 CORES=12 bash scripts/paper_gpu_workers_packed.sh
#
# PACK: workers on this GPU. Choose floor(0.85 x GPU memory / peak memory of
#   one fit); memory per fit is about the same for every regime (same batch
#   and sequence length), so read it off any running fit with nvidia-smi.
# CORES: CPU cores of this allocation (default: what the job may use). Each
#   fit runs 1 + OPERA_PAPER_NUM_WORKERS processes; keep
#   PACK x (1 + OPERA_PAPER_NUM_WORKERS) <= CORES (set NUM_WORKERS before prepare).
# STAGGER_SECONDS: pause between starts so data loading does not peak at once.
# More workers can join a GPU at any time (e.g. once only specialists remain):
#   PACK=1 bash scripts/paper_gpu_workers_packed.sh
# An out-of-memory fit is recorded as failed and retried once by the next free
# worker, resuming its last.ckpt; lower PACK if that happens repeatedly.
set -euo pipefail
cd "$(dirname "$0")/.."
PACK="${PACK:-2}"
CORES="${CORES:-$(nproc)}"
STAGGER="${STAGGER_SECONDS:-60}"
GPU="${CUDA_VISIBLE_DEVICES:-x}"
THREADS=$(( CORES / PACK ))
(( THREADS >= 1 )) || THREADS=1
# Without this each process assumes every core is its own.
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
mkdir -p logs/workers
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
for k in $(seq 1 "$PACK"); do
  name="$(hostname)-gpu${GPU}-p${k}-$$"
  WORKER_NAME="$name" WAIT="${WAIT:-1}" \
    bash scripts/paper_gpu_worker.sh > "logs/workers/$name.log" 2>&1 &
  pids+=("$!")
  echo "started $name (log: logs/workers/$name.log)"
  (( k < PACK )) && sleep "$STAGGER"
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
