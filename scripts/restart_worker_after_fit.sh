#!/usr/bin/env bash
# Restart this node's paper worker as soon as its current fit finishes, so the
# new worker sees bundles prepared after the old one started (a running worker
# only reads the stages that existed when it started). Nothing is interrupted.
#
#   nohup bash scripts/restart_worker_after_fit.sh > logs/restart_$(hostname -s).log 2>&1 &
#
# It watches logs/worker_pf_<host>.log for the next "completed", "FAILED" or
# "not estimable" line, stops the old worker at once, removes any claim the old
# worker took in the seconds before it was stopped (a fit it had not started),
# and starts a fresh worker with the same name and log.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export OPERA_PYTHON="${OPERA_PYTHON:-$repo_dir/.venv-v100/bin/python}"
host="$(hostname -s)"
worker="pf_${host}"
log="logs/worker_${worker}.log"
[ -f "$log" ] || { echo "No worker log $log on this node" >&2; exit 1; }
start_lines=$(wc -l < "$log")
echo "$(date -Is) waiting for ${worker} to finish its current fit (log at line ${start_lines})"
while true; do
  if tail -n +"$((start_lines + 1))" "$log" | grep -qE "\] (completed|FAILED|not estimable) "; then
    break
  fi
  if ! pgrep -u "$USER" -f "opera.run.paper_worker next" > /dev/null; then
    echo "$(date -Is) the worker is no longer running; starting a fresh one"
    break
  fi
  sleep 2
done
finished_at="$(date -u +%Y-%m-%dT%H:%M:%S)"
pkill -u "$USER" -f "paper_experiments worker" || true
pkill -u "$USER" -f "opera.run.paper_worker next" || true
sleep 3
pkill -u "$USER" -f "opera.run.contrastive_multicohort" || true
sleep 5
# Claims this worker took after the fit finished belong to fits it never trained.
"$OPERA_PYTHON" - "$worker" "$finished_at" <<'PY'
import json, pathlib, sys
worker, finished = sys.argv[1], sys.argv[2]
for claim in pathlib.Path("data/paper_256_v3/neural_runs").rglob(".worker_claim.json"):
    try:
        data = json.loads(claim.read_text())
    except (OSError, ValueError):
        continue
    if data.get("worker") == worker and str(data.get("claimed_at", ""))[:19] >= finished:
        claim.unlink()
        print(f"removed the old worker's fresh claim on {claim.parent}")
PY
mkdir -p logs
nohup bash scripts/run_paper_experiments.sh worker --worker-name "$worker" --execute >> "$log" 2>&1 &
echo "$(date -Is) restarted ${worker} (pid $!)"
sleep 120
grep "claimed" "$log" | tail -1
