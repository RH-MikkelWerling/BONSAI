#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
kind="${1:-epoch20}"
case "$kind" in epoch20|adapted20) ;; *) echo 'Expected epoch20 or adapted20'; exit 1;; esac
root="data/${kind}_matched_20261120"
"$python_bin" -u -m opera.run.finish_figure_comparisons --kind "$kind" --output-root "$root" --execute
"$python_bin" -u -m opera.run.package_epoch20_matched --kind "$kind" --root "$root"
