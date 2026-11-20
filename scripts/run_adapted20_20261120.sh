#!/usr/bin/env bash
# User-launched exploratory adaptation continuation and matched joint fine-tune.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
root="data/adapted20_control_20261120"
if [[ "${1:-}" != --execute ]]; then
  echo 'Plan: verify adaptation epoch10; continue to cumulative epoch20; preserve epoch15; matched J_EA fine-tune; tuning/test evaluation; aggregate ZIP.'
  exit 0
fi
mkdir -p logs
exec 9> logs/adapted20_20261120.lock
flock -n 9 || { echo 'Adapted20 launcher already active'; exit 1; }
"$python_bin" scripts/check_results_resources.py --minimum-free-gib 16 --require-no-coordinator --require-free-gpu
"$python_bin" -u -m opera.run.prepare_adapted20 --output-root "$root"
resume_args=()
if [[ -f "$root/pretraining/version_0/last.ckpt" ]]; then
  resume_args=("paths.ckpt_path=$PWD/$root/pretraining/version_0/last.ckpt"
               paths.initialization_ckpt=null paths.initialization_vocab=null
               paths.initialization_use_source_architecture=false)
fi
"$python_bin" -u -m bonsai.run.pretrain --config-path "$PWD/$root" --config-name adaptation \
  "hydra.run.dir=$PWD/$root/pretraining" "${resume_args[@]}"
"$python_bin" -u -m opera.run.prepare_adapted20 --output-root "$root" --stage joint
"$python_bin" -u -m opera.run.run_paper_queue --bundle-dir "$root/joint/bundle" \
  --runs-dir "$root/joint/neural_runs" --phase external --regime J --execute
for split in tuning held_out; do
  "$python_bin" -u -m opera.run.paper_prediction_queue --bundle-dir "$root/joint/bundle" \
    --runs-dir "$root/joint/neural_runs" --output-dir "$root/joint/predictions" \
    --split "$split" --num-workers 0 --execute
done
"$python_bin" -u -m opera.run.package_exploratory_control --root "$root"
