#!/usr/bin/env bash
# One GPU, three phases, unattended:
#   1. train: the paper worker claims fits until nothing runnable is left;
#   2. evaluate: this GPU's shard of the tuning predictions, repeated until no
#      training fit is running or pending anywhere (fits finishing on other
#      GPUs are picked up by the next pass; finished predictions are skipped);
#   3. analyse (shard 0 only, CPU): once every shard has finished, the tuning
#      analysis, plot tables, task structure, paper numbers and the figures
#      that can be rendered, in a fresh analysis root.
# EVALUATE_ONLY=1 skips training and scores existing fits only.
# Tuning split by default; SPLIT=held_out opens the test set (only after the lock
# is signed and committed).
#
#   nohup bash scripts/train_then_evaluate.sh <gpu> <shard> <shards> [tag] \
#       > logs/overnight_<shard>.log 2>&1 &
#
# Start one per GPU with shard 0..shards-1 and the same <shards> and tag (any
# node; coordination is through files under data/paper_256_v3). The OPERA alias
# for the analysis is OPERA_MODEL (default J_OA_OWN_L, the parsimony choice);
# if the lock selects another design, rerun phase 3 alone with
#   ANALYSE_ONLY=1 OPERA_MODEL=<alias> bash scripts/train_then_evaluate.sh - 0 <shards> <tag>
set -euo pipefail
cd "$(dirname "$0")/.."
gpu="${1:?gpu}"; shard="${2:?shard index}"; shards="${3:?shard count}"; tag="${4:-tuning_overnight}"
stage="${STAGE:-16}"
split="${SPLIT:-tuning}"   # held_out only after the lock is signed
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
  rm -f "$marks/shard_${shard}_of_${shards}.done" "$marks/shard_${shard}_of_${shards}.failed"
  trap 'touch "$marks/shard_${shard}_of_${shards}.failed"' ERR
  if [ -z "${EVALUATE_ONLY:-}" ]; then
    echo "$(stamp) [$shard/$shards] train on gpu $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_paper_experiments.sh worker \
      --worker-name "pf_${host}_gpu${gpu}" --execute >> "$logs/worker_${shard}.log" 2>&1
  fi
  echo "$(stamp) [$shard/$shards] no fit left to claim; evaluating shard $shard"
  deadline=$(( $(date +%s) + ${MAX_WAIT_HOURS:-20} * 3600 ))
  while true; do
    if [ -n "${EVALUATE_ONLY:-}" ]; then
      left=0; stale=0
    else
      counts="$(training_left)"
      read -r left stale <<< "$counts"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_paper_experiments.sh evaluate \
      --stage "$stage" --split "$split" --shard-count "$shards" --shard-index "$shard" \
      --execute >> "$logs/evaluate_${shard}.log" 2>&1
    # The pass above started after the count, so it saw every fit counted done.
    if [ "${left:-1}" = "0" ]; then
      "$PY" - "$D/predictions/queue_${split}_shard${shard}.json" <<'CHECK'
import json, sys
rows = json.load(open(sys.argv[1]))
allowed = {'evaluated', 'reused_verified_evaluation', 'not_estimable'}
bad = [(r['name'], r['status']) for r in rows if r.get('paper_models') and r['status'] not in allowed]
if bad:
    raise SystemExit('Evaluation incomplete: ' + repr(bad))
CHECK
      break
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      echo "$(stamp) giving up waiting: $left training fits still open"; exit 1
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
deadline=$(( $(date +%s) + ${MAX_WAIT_HOURS:-20} * 3600 ))
while true; do
  complete=0
  for ((i=0; i<shards; i++)); do
    if [ -f "$marks/shard_${i}_of_${shards}.failed" ]; then
      echo "Evaluation shard $i failed; see $logs/evaluate_${i}.log"; exit 1
    fi
    if [ -f "$marks/shard_${i}_of_${shards}.done" ]; then complete=$((complete + 1)); fi
  done
  if [ "$complete" -eq "$shards" ]; then break; fi
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "Timed out waiting for evaluation shards"; exit 1; fi
  sleep 30
done

root="$D/analysis/$tag"
env_file="$root/overnight.env"
mkdir -p "$root"
rm -f "$root/pipeline_actions_complete"
# The server .env plus the tuning overrides, in a fresh analysis root.
"$PY" - "$env_file" "$root" "$opera_model" "$D" "$stage" "$split" <<'EOF' || exit 1
import json, sys
from pathlib import Path
from dotenv import dotenv_values
out, root, model, data, stage, split = sys.argv[1:]
env = {k: v for k, v in dotenv_values(".env").items() if v is not None}
from opera.run.paper_prediction_queue import bundle_runs
from scripts.set_test_env import selected_fit
bundle, rows = bundle_runs(Path(data) / "bundles" / f"stage{stage}")
row, config_path = selected_fit(bundle, rows, model)
checkpoint = Path(data).resolve() / "neural_runs" / row["run_directory"] / "best.ckpt"
env["OPERA_PAPER_ANALYSIS_ESTIMAND"] = "cif_ipcw"
env["OPERA_PAPER_ANALYSIS_HORIZONS"] = "30,60,90,180,365,730"
env["OPERA_PAPER_TABULAR_HORIZONS"] = "30,60,90,180,365,730"
env.update(OPERA_PAPER_ANALYSIS_ROOT=str(Path(root).resolve()), OPERA_PAPER_PLOT_SPLIT=split,
           OPERA_PAPER_INVENTORY_SPLITS="tuning,held_out" if split == "held_out" else "tuning", OPERA_PAPER_OPERA_MODEL=model,
           # Lock, section 1: R_L stays learned routing in the analysis.
           OPERA_PAPER_ROUTED_MODEL="R_AB_L",
           OPERA_PAPER_TASK_STRUCTURE_CHECKPOINT=str(checkpoint),
           OPERA_PAPER_ATTRIBUTION_CHECKPOINT=str(checkpoint),
           OPERA_PAPER_ATTRIBUTION_CONFIG_DIR=str(config_path.parent.resolve()),
           OPERA_PAPER_ATTRIBUTION_CONFIG_NAME=config_path.stem,
           OPERA_PAPER_FIGURES_AVAILABLE_ONLY="1")
Path(out).write_text("".join(f"{k}={v}\n" for k, v in env.items()))
clinical = env.get("OPERA_PAPER_CLINICAL_SCORE_CONFIG")
if clinical and Path(clinical).is_file():
    from scripts.set_test_env import set_clinical_model
    from datetime import datetime
    set_clinical_model(Path(clinical), datetime.now().strftime("%Y%m%d_%H%M%S"), model)
print("checkpoint for task structure:", checkpoint)
EOF
status=()
for action in inventory analyse plot-tables task-structure paper-numbers figure-readiness init-figures figures; do
  echo "$(stamp) $action"
  if "$PY" -m opera.run.paper_experiments "$action" --stage "$stage" --env-file "$env_file" \
       --execute > "$logs/$action.log" 2>&1; then
    status+=("$action ok")
    if [ "$action" = "inventory" ]; then
      "$PY" - "$root/analysis_inventory/analysis.yaml" <<'PIN'
import sys, yaml
from pathlib import Path
path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text())
config['bootstrap_method'] = 'multiplier'
config['bootstrap_replicates'] = 1000
path.write_text(yaml.safe_dump(config, sort_keys=False))
PIN
    fi
  else
    echo "$action FAILED (see $logs/$action.log)"; exit 1
  fi
done
echo "== $(stamp) $split analysis in $root"
printf '  %s\n' "${status[@]}"

# All required launch actions succeeded; figure-readiness records optional gaps.
touch "$root/pipeline_actions_complete"
