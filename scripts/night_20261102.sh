#!/usr/bin/env bash
# One GPU, overnight 2-3 November 2026. Tuning data only; the test set stays closed.
#   1. now, CPU: the tuning analysis again with cross-seed contrasts (analysis/tuning_1102)
#   2. GPU: J_OA_OWN_PH on seed 43 (stage 16), then its tuning predictions
#   3. CPU: the analysis with it (analysis/tuning_1103), while
#   4. GPU: J_OA_OWN_PH on seed 44 trains (evaluated tomorrow; reporting only)
#
#   nohup bash scripts/night_20261102.sh > logs/overnight/night_1102.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/paper_256_v3
host="$(hostname -s)"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
mkdir -p logs/overnight "$D/predictions/.overnight/tuning_1103"

echo "$(stamp) 1. tuning analysis with cross-seed contrasts (background)"
if [ -d "$D/analysis/tuning_1102" ] && [ ! -d "$D/analysis/tuning_1102_first" ]; then
  mv "$D/analysis/tuning_1102" "$D/analysis/tuning_1102_first"
fi
ANALYSE_ONLY=1 STAGE=15 bash scripts/train_then_evaluate.sh - 0 4 tuning_1102 \
  > logs/overnight/analysis_1102b.log 2>&1 &
analysis_1102=$!

echo "$(stamp) 2. stage 16 and J_OA_OWN_PH seed 43"
[ -f "$D/bundles/stage16/bundle.json" ] || \
  bash scripts/run_paper_experiments.sh prepare --stage 16 --execute > logs/overnight/prepare_16.log 2>&1 \
  || { echo "prepare stage 16 failed (logs/overnight/prepare_16.log)"; exit 1; }
CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_experiments.sh worker --worker-name "pf_${host}_gpu0" \
  --max-runs 1 --execute > logs/overnight/worker_ph43.log 2>&1
echo "$(stamp)    seed 43 finished; tuning predictions"
CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_experiments.sh evaluate --stage 16 --split tuning \
  --execute > logs/overnight/evaluate_1103.log 2>&1 || echo "evaluate reported an error (see log)"

echo "$(stamp) 4. J_OA_OWN_PH seed 44 (background)"
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_paper_experiments.sh worker --worker-name "pf_${host}_gpu0" \
  --max-runs 1 --execute > logs/overnight/worker_ph44.log 2>&1 &

echo "$(stamp) 3. analysis with seed 43 of the runner-up (waits for step 1 first)"
wait "$analysis_1102"
touch "$D/predictions/.overnight/tuning_1103/shard_0_of_1.done"
ANALYSE_ONLY=1 STAGE=16 bash scripts/train_then_evaluate.sh - 0 1 tuning_1103 \
  > logs/overnight/analysis_1103.log 2>&1
for tag in tuning_1102 tuning_1103; do
  .venv-v100/bin/python scripts/lock_numbers.py "$D/analysis/$tag" > "logs/overnight/lock_numbers_$tag.txt" 2>&1
done
echo "$(stamp) done: logs/overnight/lock_numbers_tuning_1102.txt and _1103.txt"
