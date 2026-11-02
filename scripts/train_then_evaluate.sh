#!/usr/bin/env bash
# One GPU, three phases, unattended:
#   1. train: the paper worker claims fits until nothing runnable is left;
#   2. evaluate: this GPU's shard of the tuning predictions, repeated until no
#      training fit is running or pending anywhere (fits finishing on other
#      GPUs are picked up by the next pass; finished predictions are skipped);
#   3. analyse (shard 0 only, CPU): once every shard has finished, the tuning
#      analysis, plot tables, task structure, paper numbers and the figures
#      that can be rendered, in a fresh analysis root.
# Tuning split only: held-out labels stay closed until the lock is signed.
#
#   nohup bash scripts/train_then_evaluate.sh <gpu> <shard> <shards> [tag] \
#       > logs/overnight_<shard>.log 2>&1 &
#
# Start one per GPU with shard 0..shards-1 and the same <shards> and tag (any
# node; coordination is through files under data/paper_256_v3). The OPERA alias
# for the analysis is OPERA_MODEL (default J_OA_OWN_L, the parsimony choice);
# if the lock selects another design, rerun phase 3 alone with
#   ANALYSE_ONLY=1 OPERA_MODEL=<alias> bash scripts/train_then_evaluate.sh - 0 <shards> <tag>
set -uo pipefail
cd "$(dirname "$0")/.."
gpu="${1:?gpu}"; shard="${2:?shard index}"; shards="${3:?shard count}"; tag="${4:-tuning_overnight}"
stage="${STAGE:-15}"
opera_model="${OPERA_MODEL:-J_OA_OWN_L}"
D=data/paper_256_v3
PY="${OPERA_PYTHON:-.venv-v100/bin/python}"
export OPERA_PYTHON="$PY"
marks="$D/predictions/.overnight/$tag"
logs="logs/overnight/$tag"
mkdir -p "$marks" "$logs"
host="$(hostname -s)"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

training_left() {
  # Fits of the training phases still running, claimed, or waiting to run.
  # Held fits (HOLD_FITS) and national fits without an encoder are not counted.
  "$PY" -m opera.run.paper_experiments queue-status 2>/dev/null | "$PY" -c '
import json, sys
text = sys.stdin.read()
report, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
left = sum(n for phase, states in report.items()
           if phase.split("/")[0] in {"router_fix", "supervision"}
           for state, n in states.items() if state in {"running", "pending"})
stale = sum(n for states in report.values() for state, n in states.items()
            if state in {"stale_claim", "claim_unreadable"})
print(left, stale)'
}

if [ -z "${ANALYSE_ONLY:-}" ]; then
  echo "$(stamp) [$shard/$shards] train on gpu $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_paper_experiments.sh worker \
    --worker-name "pf_${host}_gpu${gpu}" --execute >> "$logs/worker_${shard}.log" 2>&1
  echo "$(stamp) [$shard/$shards] no fit left to claim; evaluating shard $shard"
  deadline=$(( $(date +%s) + ${MAX_WAIT_HOURS:-20} * 3600 ))
  while true; do
    read -r left stale <<< "$(training_left)"
    CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_paper_experiments.sh evaluate \
      --stage "$stage" --split tuning --shard-count "$shards" --shard-index "$shard" \
      --execute >> "$logs/evaluate_${shard}.log" 2>&1 \
      || echo "$(stamp) evaluate pass reported an error (see $logs/evaluate_${shard}.log)"
    # The pass above started after the count, so it saw every fit counted done.
    if [ "${left:-1}" = "0" ]; then break; fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      echo "$(stamp) giving up waiting: $left training fits still open"; break
    fi
    [ "${stale:-0}" != "0" ] && echo "$(stamp) $stale stale claims (killed jobs?); run clear-stale"
    echo "$(stamp) $left training fits open; next evaluation pass in 10 min"
    sleep 600
  done
  touch "$marks/shard_${shard}_of_${shards}.done"
  echo "$(stamp) [$shard/$shards] evaluation shard done"
fi

[ "$shard" = "0" ] || exit 0
echo "$(stamp) waiting for all $shards evaluation shards"
until [ "$(ls "$marks"/shard_*_of_"$shards".done 2>/dev/null | wc -l)" -ge "$shards" ]; do sleep 300; done

root="$D/analysis/$tag"
env_file="$root/overnight.env"
mkdir -p "$root"
# The server .env plus the tuning overrides, in a fresh analysis root.
"$PY" - "$env_file" "$root" "$opera_model" "$D" "$stage" <<'EOF' || exit 1
import json, sys
from pathlib import Path
from dotenv import dotenv_values
out, root, model, data, stage = sys.argv[1:]
env = {k: v for k, v in dotenv_values(".env").items() if v is not None}
runs = json.loads((Path(data) / "bundles" / f"stage{stage}" / "bundle.json").read_text())["runs"]
alias = model.removesuffix("_L")
row = next(r for r in runs if r["name"].endswith(f"/L/fix_{alias}") and r["seed"] == 42)
checkpoint = Path(data).resolve() / "neural_runs" / row["run_directory"] / "best.ckpt"
env.setdefault("OPERA_PAPER_ANALYSIS_ESTIMAND", "cif_ipcw")  # the locked primary estimand
env.update(OPERA_PAPER_ANALYSIS_ROOT=str(Path(root).resolve()), OPERA_PAPER_PLOT_SPLIT="tuning",
           OPERA_PAPER_INVENTORY_SPLITS="tuning", OPERA_PAPER_OPERA_MODEL=model,
           OPERA_PAPER_TASK_STRUCTURE_CHECKPOINT=str(checkpoint),
           OPERA_PAPER_FIGURES_AVAILABLE_ONLY="1")
Path(out).write_text("".join(f"{k}={v}\n" for k, v in env.items()))
print("checkpoint for task structure:", checkpoint)
EOF
status=()
for action in inventory analyse plot-tables task-structure paper-numbers figure-readiness init-figures figures; do
  echo "$(stamp) $action"
  if "$PY" -m opera.run.paper_experiments "$action" --stage "$stage" --env-file "$env_file" \
       --execute > "$logs/$action.log" 2>&1; then
    status+=("$action ok")
  else
    status+=("$action FAILED (see $logs/$action.log)")
  fi
done
echo "== $(stamp) tuning analysis in $root"
printf '  %s\n' "${status[@]}"
