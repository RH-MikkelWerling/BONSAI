#!/usr/bin/env bash
# Exploratory analysis 5 only. Does not launch training or restart main analyses.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
prepared="${1:-data/diagnostics/representation_absorption_20261112/tuning_core.prepared_INTERNAL}"
metadata="${2:-}"
output="${3:-data/diagnostics/representation_absorption_20261114/pooling_tuning}"
if [[ ! -f "$prepared/prepared.json" ]]; then
  echo "Prepared exploratory snapshot not found: $prepared; keep the running main coordinator untouched." >&2
  exit 1
fi
args=(--prepared-dir "$prepared" --output-dir "$output" --split tuning)
if [[ -n "$metadata" ]]; then args+=(--cell-metadata "$metadata"); fi
# Recover the frozen metadata path from the exploratory source recipe if available.
if [[ -z "$metadata" && -f "$prepared/source_analysis.yaml" ]]; then
  metadata="$($python_bin - "$prepared/source_analysis.yaml" <<'PY'
import sys,yaml
from pathlib import Path
value=yaml.safe_load(Path(sys.argv[1]).read_text()).get('cell_metadata')
print(value or '')
PY
)"
  if [[ -n "$metadata" ]]; then args+=(--cell-metadata "$metadata"); fi
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$python_bin" -u -m opera.analysis.pooling_absorption "${args[@]}"
