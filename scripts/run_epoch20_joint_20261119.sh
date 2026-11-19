#!/usr/bin/env bash
# Explicitly user-requested new joint fine-tune. Run on an otherwise free GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
root="${OPERA_EPOCH20_ROOT:-data/epoch20_joint_control_20261119}"
if [[ "${1:-}" != --execute ]]; then
  echo 'Plan: verify epoch10/20 hashes; clone frozen joint recipe; full encoder/head fine-tune; tuning/test evaluation; aggregate review ZIP.'
  echo 'Requires one free GPU. Use --execute to run this explicitly requested experiment.'
  exit 0
fi
mkdir -p logs
exec 9> logs/epoch20_joint_control_20261119.lock
flock -n 9 || { echo 'Epoch20 joint launcher already active'; exit 1; }
"$python_bin" scripts/check_results_resources.py --minimum-free-gib 16 --require-no-coordinator --require-free-gpu
"$python_bin" -u -m opera.run.prepare_epoch20_joint_finetune --output-root "$root"
"$python_bin" -u -m opera.run.run_paper_queue \
  --bundle-dir "$root/bundle" --runs-dir "$root/neural_runs" \
  --phase supervision --regime J --minimum-local-pretraining-epochs 20 --execute
for split in tuning held_out; do
  "$python_bin" -u -m opera.run.paper_prediction_queue \
    --bundle-dir "$root/bundle" --runs-dir "$root/neural_runs" \
    --output-dir "$root/predictions" --split "$split" --num-workers 0 --execute
done
"$python_bin" - "$root" <<'PY'
import csv,json,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from bonsai.functional.checkpointing import verified_training_selection
root=Path(sys.argv[1]); bundle=json.loads((root/'bundle/bundle.json').read_text());row,=bundle['runs']
selection=verified_training_selection(root/'neural_runs'/row['run_directory'])
report={'status':'complete','variant':'J_L_epoch20','scope':'Post-held-out exploratory control; fixed tuning checkpoint selection',
        'training_selection':selection,'run_directory':row['run_directory'],
        'note':'Review metrics only. Direct paired epoch20-minus-epoch10 CIs need a separate matched analysis on server; no locked aliases changed.'}
(root/'review_summary.json').write_text(json.dumps(report,indent=2))
with ZipFile(root/'epoch20_joint_review.zip','w',ZIP_DEFLATED) as z:
    for name in ['preparation.json','review_summary.json']:z.write(root/name,name)
    for split in ['tuning','held_out']:
        folder=root/'predictions'/row['run_directory']/split
        for name in ['direct_cr_metrics.csv','direct_cr_pooled_metrics.csv','direct_cr_family_metrics.csv','direct_cr_summary.json']:
            path=folder/name
            if path.suffix=='.csv':
                with path.open(newline='') as f:
                    if {'subject_id','patient_id','cpr','pnr'} & {c.lower() for c in next(csv.reader(f))}:
                        raise ValueError('Patient column in supposed aggregate')
            z.write(path,f'{split}/{name}')
print(f'Download: {root/"epoch20_joint_review.zip"}')
PY
