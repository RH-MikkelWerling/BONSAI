#!/usr/bin/env bash
# User-launched remaining saved-prediction analyses. No training or GPU work.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
root="${OPERA_REMAINING_CPU_ROOT:-data/figure_closure_20261119}"
coverage="${OPERA_NATIONAL_COVERAGE_ROOT:-data/national_coverage_20261119}"
lane="${1:-plan}"
mkdir -p logs
case "$lane" in
  plan)
    echo 'On a free >=96 GiB CPU allocation: all (paired -> design_search -> harvest; vocabulary coverage cancelled).'
    echo 'Individual lanes: paired, design_search, vocabulary, harvest. No fitting, inference or GPU work.'
    exit 0 ;;
  all)
    failed=0
    for task in paired design_search; do
      bash "$0" "$task" || failed=1
    done
    bash "$0" harvest || failed=1
    exit "$failed" ;;
  paired)
    "$python_bin" -u -m opera.run.finish_figure_comparisons --kind core --output-root "$root" --execute ;;
  design_search)
    "$python_bin" -u -m opera.run.finish_figure_comparisons --kind design_search --output-root "$root" --execute ;;
  vocabulary)
    echo 'Vocabulary-coverage analysis cancelled by the 20 November checklist; no job launched.'
    exit 0
    "$python_bin" scripts/check_results_resources.py --minimum-free-gib 96 --require-no-coordinator
    if [[ -f "$coverage/coverage_manifest.json" ]]; then
      "$python_bin" - "$coverage" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); report=json.loads((root/'coverage_manifest.json').read_text())
h=hashlib.sha256()
with (root/'patient_metadata_INTERNAL.parquet').open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
if report.get('status')!='complete' or report.get('output_metadata_sha256')!=h.hexdigest():
    raise SystemExit('Existing coverage output is incomplete/changed; inspect before reuse')
print('Reusing verified national-coverage metadata')
PY
    else
      "$python_bin" -u -m opera.run.prepare_national_coverage_metadata --output-dir "$coverage"
    fi
    "$python_bin" -u -m opera.run.finish_figure_comparisons --kind vocabulary --output-root "$root" \
      --patient-metadata "$coverage/patient_metadata_INTERNAL.parquet" --execute ;;
  harvest)
    "$python_bin" -u -m opera.run.finish_figure_inputs --lane recover --execute --refresh-harvest --output-root "$root" ;;
  *) echo "Unknown lane $lane" >&2; exit 2 ;;
esac
