#!/usr/bin/env bash
# Extra aggregate checks only; never submits jobs, starts training or changes the lock.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
task_root="${OPERA_RECOVERY_ROOT:-data/cutoff_recovery_20261121}"
mkdir -p logs
complete() {
  python - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
from zipfile import ZipFile
p=Path(sys.argv[1]);kind=sys.argv[2]
try:
    m=json.loads((p/'manifest.json').read_text())
    assert m['status']=='complete'
    if kind in ('encoder','tabular'):
        expected=['R_EA','J_EA','R_E','J_E','J_OA_OWN_L','J_L'] if kind=='encoder' else ['R_EA','J_EA','X','XP']
        assert m['models']==expected and m['specificities']==[.75,.8,.9,.95]
        archive=p/'download_test_roc_points.zip'
    elif kind=='support':
        assert m['all_outcomes'] and m['method']=='ipcw_kernel_kish_ess_v1'
        archive=p/'download_risk_support.zip'
    elif kind=='attention':
        assert m['patients']==6446
        archive=p/'download_attention_patterns.zip'
    elif kind=='corrections':
        assert m['patients']==6446
        archive=p/'download_correction_diagnostics.zip'
    else:raise ValueError(kind)
    with ZipFile(archive) as z:assert z.testzip() is None
except (OSError,KeyError,ValueError,AssertionError):sys.exit(1)
PY
}
case "${1:-status}" in
  day)
    bash "$0" cpu
    bash "$0" attention
    bash "$0" corrections
    exec bash scripts/run_cutoff_recovery.sh interpretation-worker ;;
  cpu)
    # Sequential jobs: avoid holding multiple prepared snapshots in RAM.
    python scripts/check_results_resources.py --minimum-free-gib 96 --require-no-coordinator
    if ! complete "$task_root/exports/test_roc_encoders_75_v1" encoder; then
      python -u -m opera.run.export_test_roc_points \
        --checkpoint-dir "$task_root/analyses/core_damage/checkpoints" \
        --models R_EA J_EA R_E J_E J_OA_OWN_L J_L --specificities .75 .8 .9 .95 \
        --output-dir "$task_root/exports/test_roc_encoders_75_v1"
    fi
    if ! complete "$task_root/exports/test_roc_tabular_75_v1" tabular; then
      python -u -m opera.run.export_test_roc_points \
        --checkpoint-dir "$task_root/analyses/core_damage/checkpoints" \
        --models R_EA J_EA X XP --specificities .75 .8 .9 .95 \
        --output-dir "$task_root/exports/test_roc_tabular_75_v1"
    fi
    python scripts/analyse_task_similarity_sensitivity.py "$task_root/exports/task_structure" \
      "$task_root/exports/task_similarity_sensitivity_20261127"
    if ! complete "$task_root/exports/risk_support_all_outcomes_20261127" support; then
      CUDA_VISIBLE_DEVICES='' python -u -m opera.run.refresh_embedding_risk_support \
        --story-root "$task_root/exports/story_held_out_descriptive_v1" \
        --output-dir "$task_root/exports/risk_support_all_outcomes_20261127"
    fi
    echo 'CPU extras complete: encoder/tabular ROC, definition sensitivity and all-outcome support.' ;;
  attention)
    python scripts/check_results_resources.py --minimum-free-gib 48 --require-no-coordinator --require-free-gpu
    if ! complete "$task_root/exports/attention_patterns_v1" attention; then
      python -u -m opera.run.export_attention_patterns --recovery-root "$task_root"
    fi
    python scripts/analyse_task_similarity_sensitivity.py "$task_root/exports/attention_patterns_v1" \
      "$task_root/exports/attention_definition_sensitivity_20261127"
    echo 'Functional attention and its definition-overlap sensitivity complete.' ;;
  absolute-risk-summary)
    python scripts/check_results_resources.py --minimum-free-gib 96 --require-no-coordinator
    python -u -m opera.run.export_absolute_risk_ig_summaries --root "$task_root" ;;
  corrections)
    python scripts/check_results_resources.py --minimum-free-gib 48 --require-no-coordinator --require-free-gpu
    if ! complete "$task_root/exports/correction_diagnostics_v1" corrections; then
      python -u -m opera.run.export_correction_diagnostics --root "$task_root"
    fi ;;
  status)
    python scripts/check_subgroup_exports.py --root "$task_root" \
      --output "logs/subgroup_export_status_$(date +%Y%m%d_%H%M%S).json"
    python scripts/collect_paper_closure_status.py --recovery-root "$task_root" \
      --ig-planned-patients 6446 --output "logs/paper_day_status_$(date +%Y%m%d_%H%M%S).json"
    for item in encoder tabular support attention corrections; do
      case "$item" in
        encoder) folder=test_roc_encoders_75_v1 ;;
        tabular) folder=test_roc_tabular_75_v1 ;;
        support) folder=risk_support_all_outcomes_20261127 ;;
        attention) folder=attention_patterns_v1 ;;
        corrections) folder=correction_diagnostics_v1 ;;
      esac
      if complete "$task_root/exports/$folder" "$item"; then
        echo "$item: verified complete archive"
      else
        echo "$item: not confirmed complete; inspect its log"
      fi
    done ;;
  package)
    python scripts/package_paper_extra_checks_20261127.py --root "$task_root" ;;
  *) echo 'Use: day | cpu | attention | corrections | absolute-risk-summary | status | package' >&2; exit 2 ;;
esac
