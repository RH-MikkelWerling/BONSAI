#!/usr/bin/env bash
# Dedicated free GPU/high-RAM allocation: evaluate saved fractions, then CPU analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
national="$PWD/data/paper_256_v3_national_20261111"
output="$national/fraction_results_20261116"
mkdir -p logs
exec 9> "$national/.fraction_results_20261116.lock"
flock -n 9 || { echo 'Fraction-result chain already running'; exit 1; }
"$python_bin" scripts/check_results_resources.py --minimum-free-gib 24 --require-no-coordinator --require-free-gpu
for split in tuning held_out; do
  "$python_bin" -u -m opera.run.paper_prediction_queue \
    --bundle-dir "$national/adapted_fits" --runs-dir "$national/neural_runs" \
    --output-dir "$national/predictions" --only data_efficiency --split "$split" \
    --num-workers 0 --report-path "$national/fraction_evaluation_${split}_20261116.json" --execute
done
if [[ ! -f "$output/analysis.yaml" ]]; then
  "$python_bin" -u -m opera.run.prepare_fraction_results --national-root "$national" --output-dir "$output"
fi
if [[ -f "$output/results/analysis_manifest.json" ]]; then
  echo "Fraction/control analysis already complete: $output/results"
  exit 0
fi
"$python_bin" scripts/check_results_resources.py --minimum-free-gib 96 --require-no-coordinator
# No distributed queue/checkpoint environment: one local worker and no helper wait.
env -u OPERA_ANALYSIS_CHECKPOINT_DIR CUDA_VISIBLE_DEVICES='' \
  OPERA_ANALYSIS_WORKERS=1 OPERA_SUMMARY_WORKERS=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$python_bin" -u -m opera.run.analyse_locked_predictions \
  --config "$output/analysis.yaml" --output-dir "$output/results"
echo "Fraction/control results complete: $output/results"
