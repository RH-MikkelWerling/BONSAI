#!/usr/bin/env bash
# Existing selected checkpoints only. CPU; no patient data, fitting or inference.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
output="${OPERA_TASK_STRUCTURE_OUTPUT:-data/task_structure_comparison_20261119}"
"$python_bin" scripts/check_results_resources.py --minimum-free-gib 16 --require-no-coordinator
for model in J_OA_OWN_L R_E R_EA; do
  case "$model" in
    J_OA_OWN_L) run=data/paper_256_v3/neural_runs/L/seed42/7e4429feacbfbd643a6e ;;
    R_E) run=data/paper_256_v3_national_20261111/neural_runs/E/seed42/3ea7e9d6b2970a542448 ;;
    R_EA) run=data/paper_256_v3_national_20261111/neural_runs/EA/seed42/48897758acdeebadac1f ;;
  esac
  "$python_bin" -u -m opera.run.export_opera_task_structure \
    --run-dir "$run" --output-dir "$output/task_structure_$model"
done
"$python_bin" - "$output" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import csv, sys
root = Path(sys.argv[1])
names = ['outcome_query_similarity.csv', 'outcome_correction_similarity.csv',
         'disease_correction_similarity.csv', 'outcome_specialisation.csv',
         'disease_specialisation.csv', 'task_structure_summary.json']
with ZipFile(root / 'task_structure_comparison.zip', 'w', ZIP_DEFLATED) as z:
    for model in ['J_OA_OWN_L', 'R_E', 'R_EA']:
        for name in names:
            path = root / f'task_structure_{model}' / name
            if path.suffix == '.csv':
                with path.open(newline='') as f:
                    if {'subject_id', 'patient_id', 'cpr', 'pnr'} & {c.lower() for c in next(csv.reader(f))}:
                        raise ValueError('Patient columns in task aggregate')
            z.write(path, str(path.relative_to(root)))
print(f'Download: {root / "task_structure_comparison.zip"}')
PY
