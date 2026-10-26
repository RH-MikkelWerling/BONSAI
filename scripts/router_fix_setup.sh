#!/usr/bin/env bash
# Run once (CPU is enough), before starting the GPU workers for the router fix
# (lock addendum 2026-10-26):
#   1. prepare the stage-3 bundle (stage 2 plus the six router-fix fits);
#   2. stop unless exactly those six fits are new (every earlier fit must be
#      regenerated identically, or the workers would retrain it);
#   3. audit what the collapsed routers collapsed onto (R, RSI, disease bank);
#   4. start the tuning rehearsal from `analyse` onwards in the background.
# Then, in each GPU job: bash scripts/paper_gpu_worker.sh
#
#   bash scripts/router_fix_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/paper_256_v3

echo "== 1. stage-3 bundle"
if [[ -d $D/bundles/stage3 ]]; then
  echo "stage3 exists; keeping it"
else
  bash scripts/run_paper_experiments.sh prepare --stage 3 --execute
fi

echo "== 2. new fits in stage 3"
python - "$D" <<'EOF'
import json, sys
root = sys.argv[1]
old = {r["run_directory"] for r in json.load(open(f"{root}/bundles/stage2/bundle.json"))["runs"]}
runs = json.load(open(f"{root}/bundles/stage3/bundle.json"))["runs"]
new = [r["name"] for r in runs if r["run_directory"] not in old]
print(len(new), "new:", *new, sep="\n  ")
if len(new) != 6 or not all("/fix_" in name for name in new):
    sys.exit("Stage 3 is not stage 2 plus the six router-fix fits: do not start workers.")
EOF

echo "== 3. hub audits (aggregates only)"
for run in 0de5f3abae9725a220e5 315e06416eacbbfb8f17 3fd4c3c5b2c40c0610f8; do
  out=$D/router_decisions/hub_audit_$run.json
  python -m opera.run.audit_router_hub --run-dir "$D/neural_runs/L/seed42/$run" \
    --prediction-dir "$D/predictions/L/seed42/$run/tuning" --output "$out" > /dev/null \
    && echo "  $out" || echo "  audit $run failed"
done

echo "== 4. tuning rehearsal (background; progress in logs/rehearsal/loop.log)"
R=$D/analysis/tuning_rehearsal
mkdir -p logs/rehearsal
[[ -d $R/prediction_analysis ]] && mv "$R/prediction_analysis" "$R/prediction_analysis.failed_$(date +%m%d_%H%M)"
nohup bash -c 'for a in analyse plot-tables figure-readiness figures; do
  echo "== $a $(date +%H:%M)"
  bash scripts/run_paper_experiments.sh $a --env-file .env.rehearsal --execute > logs/rehearsal/$a.log 2>&1 \
    || { echo "$a failed"; grep -n "Error" logs/rehearsal/$a.log | tail -3; exit 1; }
done; echo "rehearsal done $(date +%H:%M)"' > logs/rehearsal/loop.log 2>&1 &
echo "  started (pid $!)"
echo "Now start the GPU workers: bash scripts/paper_gpu_worker.sh (one per GPU job)."
