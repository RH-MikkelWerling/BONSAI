#!/usr/bin/env bash
# User-approved exploratory J3. No scheduler submission; isolated, resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
root=data/vocabulary_control_20261120
case "${1:-preflight}" in
 preflight)
   "$python_bin" scripts/check_results_resources.py --minimum-free-gib 96 --require-no-coordinator
   exec "$python_bin" -u -m opera.run.prepare_vocabulary_control ;;
 --execute) ;;
 *) echo 'Usage: preflight or --execute'; exit 1;;
esac
mkdir -p logs
exec 9> logs/vocabulary_control_20261120.lock
flock -n 9 || { echo 'Vocabulary control already active'; exit 1; }
"$python_bin" scripts/check_results_resources.py --minimum-free-gib 96 --require-no-coordinator --require-free-gpu
"$python_bin" -u -m opera.run.prepare_vocabulary_control
"$python_bin" -u -m bonsai.run.pretrain --config-path "$PWD/$root" --config-name pretraining \
  "hydra.run.dir=$PWD/$root/pretraining"
for epoch in 10 20; do
  "$python_bin" -u -m opera.run.prepare_vocabulary_control --stage "joint$epoch"
  "$python_bin" -u -m opera.run.run_paper_queue --bundle-dir "$root/joint$epoch/bundle" \
    --runs-dir "$root/joint$epoch/neural_runs" --phase supervision --regime J --minimum-local-pretraining-epochs "$epoch" --execute
  for split in tuning held_out; do
    "$python_bin" -u -m opera.run.paper_prediction_queue --bundle-dir "$root/joint$epoch/bundle" \
      --runs-dir "$root/joint$epoch/neural_runs" --output-dir "$root/joint$epoch/predictions" \
      --split "$split" --num-workers 0 --execute
  done
done
"$python_bin" -u -m opera.run.finish_figure_comparisons --kind vocab_control --output-root "$root/matched" --execute
"$python_bin" -u -m opera.run.package_vocabulary_control
