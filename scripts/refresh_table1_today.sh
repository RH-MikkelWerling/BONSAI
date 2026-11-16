#!/usr/bin/env bash
# CPU-only clinical Table 1 refresh; preserves the reviewed non-clinical rows.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
original="${OPERA_TABLE1_ORIGINAL:-data/paper_256_v2/table1/LOCAL_ONLY_internal/table1_values_long.csv}"
output="${1:-data/clinical_registry_refresh/20261111/table1_refresh_20261116}"
"$python_bin" - "$original" <<'PY'
import hashlib,sys
from pathlib import Path
path=Path(sys.argv[1])
expected='75fe32a76452de4be50b8258c1cfedd8ef8ad079b59bdcf4ef76c345e2eb7169'
if not path.is_file():
    raise SystemExit(f'Missing reviewed Table 1 source: {path}; set OPERA_TABLE1_ORIGINAL to its actual path')
if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
    raise SystemExit('Table 1 source differs from the reviewed export identified in paper/tables/table1.tex; check the input before refreshing')
PY
"$python_bin" -u -m opera.run.refresh_clinical_table1 \
  --original "$original" \
  --prepared-dir data/clinical_registry_refresh/20261111/prepared_clinical \
  --lyfo data/clinical_registry_refresh/20261111/lyfo.parquet \
  --source-analysis data/paper_256_v3/analysis/local_test_20261109_four_allocations/analysis_inventory/analysis.yaml \
  --output-dir "$output"
"$python_bin" -m opera.run.format_table1_release \
  --input "$output/table1_values_long.csv" --output-dir "$output/manuscript" \
  --disclosure-policy primary-only --latex "$output/manuscript/table1.tex"
"$python_bin" - "$output" <<'PY'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import sys
root=Path(sys.argv[1]); target=root/'table1_review.zip'
names=['table1_refresh_report.json','ipi_score_transition_counts.csv',
       'manuscript/table1_values_long.csv','manuscript/table1_pooled_display.csv',
       'manuscript/disclosure_policy.json','manuscript/table1.tex','manuscript/table1_by_disease.tex']
with ZipFile(target,'x',compression=ZIP_DEFLATED) as archive:
    for name in names:
        archive.write(root/name,name)
print(f'Download: {target}')
print('Review export: retain the two previously approved SMZL/NMZL edits when importing into the manuscript.')
PY
