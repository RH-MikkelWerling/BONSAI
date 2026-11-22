#!/usr/bin/env bash
# Run inside an existing allocation; never submits jobs or accesses another node.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPERA_ANALYSIS_WORKERS=1 OPERA_SUMMARY_WORKERS=1
export MPLBACKEND=Agg OPERA_THESIS_STYLE=1 OPERA_PAPER_FIGURE_MARK="" OPERA_PAPER_EXPORT_PLOT_STATE=1
unset OPERA_ANALYSIS_CHECKPOINT_DIR
export OPERA_RECOVERY_ROOT="${OPERA_RECOVERY_ROOT:-data/cutoff_recovery_20261121}"
mkdir -p logs
lane="${1:-plan}"
case "$lane" in
  plan)
    echo 'Prepare once. Two encoder allocations (cohort/adapted, 20 epochs with epoch10 saved).'
    echo 'Up to six further GPU allocations run worker; encoder allocations join worker after finishing.'
    echo 'Separate high-RAM CPU allocation runs results: early damage analysis, then registered refreshes.'
    echo 'Unchanged national/tabular/random results are reused. No jobs are submitted by this script.' ;;
  prepare)
    python -u -m opera.run.prepare_cutoff_recovery encoders --root "$OPERA_RECOVERY_ROOT" \
      > logs/cutoff_recovery_encoder_preparation_20261121.json
    python -u -m opera.run.prepare_cutoff_recovery_queue --root "$OPERA_RECOVERY_ROOT" \
      > logs/cutoff_recovery_queue_preparation_20261121.json
    cat "$OPERA_RECOVERY_ROOT/queue_summary.json" ;;
  cohort|adapted)
    bash scripts/run_cutoff_recovery_encoder.sh "$lane"
    # Reuse this allocation's remaining walltime for fine-tuning/evaluation.
    bash "$0" worker ;;
  worker)
    python scripts/check_results_resources.py --minimum-free-gib 48 --require-no-coordinator --require-free-gpu
    interpretation=()
    [[ "${OPERA_RECOVERY_IG:-1}" == 1 ]] && interpretation=(--with-interpretation)
    exec python -u -m opera.run.cutoff_recovery_worker work --root "$OPERA_RECOVERY_ROOT" --wait-hours 72 "${interpretation[@]}" ;;
  results)
    python scripts/check_results_resources.py --minimum-free-gib 96 --require-no-coordinator
    interpretation=()
    [[ "${OPERA_RECOVERY_IG:-1}" == 1 ]] && interpretation=(--wait-interpretation)
    exec python -u -m opera.run.cutoff_recovery_results --root "$OPERA_RECOVERY_ROOT" --execute --wait-hours 96 --exports "${interpretation[@]}" ;;
  status)
    python -m opera.run.cutoff_recovery_worker status --root "$OPERA_RECOVERY_ROOT"
    if [[ -f "$OPERA_RECOVERY_ROOT/pipeline_status.json" ]]; then
      cat "$OPERA_RECOVERY_ROOT/pipeline_status.json"
    fi ;;
  paper-first-preview)
    python -m opera.run.cutoff_recovery_workload --root "$OPERA_RECOVERY_ROOT" ;;
  paper-first)
    python -m opera.run.cutoff_recovery_workload --root "$OPERA_RECOVERY_ROOT" --execute ;;
  epoch20-prepare)
    python -m opera.run.cutoff_recovery_epoch20 prepare --root "$OPERA_RECOVERY_ROOT" ;;
  epoch20-opera-prepare)
    python -m opera.run.cutoff_recovery_epoch20 prepare-opera --root "$OPERA_RECOVERY_ROOT" ;;
  epoch20-status)
    python -m opera.run.cutoff_recovery_epoch20 status --root "$OPERA_RECOVERY_ROOT" ;;
  epoch20|cohort20|adapted20|opera20|cohort-opera20|adapted-opera20)
    python scripts/check_results_resources.py --minimum-free-gib 48 --require-no-coordinator --require-free-gpu
    family=()
    [[ "$lane" == cohort20 || "$lane" == cohort-opera20 ]] && family=(--family cohort)
    [[ "$lane" == adapted20 || "$lane" == adapted-opera20 ]] && family=(--family adapted)
    architecture=()
    [[ "$lane" == *opera20 ]] && architecture=(--architecture opera)
    python -u -m opera.run.cutoff_recovery_epoch20 work --root "$OPERA_RECOVERY_ROOT" "${family[@]}" "${architecture[@]}"
    bash "$0" worker ;;
  bundle)
    python - "$OPERA_RECOVERY_ROOT" <<'PY'
import sys
from pathlib import Path
from opera.run.cutoff_recovery_results import package
package(Path(sys.argv[1]).resolve())
PY
    ;;
  *) echo 'Use: plan | prepare | cohort | adapted | worker | results | status | bundle | paper-first-preview | paper-first | epoch20-prepare | epoch20-opera-prepare | epoch20-status | epoch20 | cohort20 | adapted20 | opera20 | cohort-opera20 | adapted-opera20' >&2; exit 2 ;;
esac
