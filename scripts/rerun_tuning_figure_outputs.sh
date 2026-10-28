#!/usr/bin/env bash
# Recover a failed tuning analysis without scanning the inventory or retraining.
# Run from the repository root after syncing source files.
set -euo pipefail
plot_root="${1:?Usage: bash scripts/rerun_tuning_figure_outputs.sh RUN_ROOT}"
plot_python="${OPERA_PYTHON:-.venv-v100/bin/python}"
plot_root="$("$plot_python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$plot_root")"
plot_env="$plot_root/overnight.env"
"$plot_python" - "$plot_root" "$plot_env" <<'PY'
from pathlib import Path
import sys
from opera.run.paper_experiments import load_environment, path_value
root, env_file = map(Path, sys.argv[1:])
env = load_environment(env_file)
if path_value(env, 'OPERA_PAPER_ANALYSIS_ROOT').resolve() != root:
    raise SystemExit('Environment analysis root does not match the requested run')
if env.get('OPERA_PAPER_PLOT_SPLIT') != 'tuning':
    raise SystemExit('This recovery script is for tuning previews only')
if not (root / 'analysis_inventory/analysis.yaml').is_file():
    raise SystemExit('Missing existing inventory; run inventory first')
PY
plot_stamp="$(date +%Y%m%d_%H%M%S)"
plot_logs="$plot_root/logs/recovery_$plot_stamp"
mkdir -p "$plot_logs"
# Retain partial results and all earlier figure renders for comparison.
if [[ -e "$plot_root/prediction_analysis" ]]; then
    mv "$plot_root/prediction_analysis" "$plot_root/prediction_analysis_before_$plot_stamp"
fi
for plot_action in analyse embedding-tables plot-tables figure-readiness; do
    echo "$(date +%H:%M:%S) $plot_action"
    "$plot_python" -m opera.run.paper_experiments "$plot_action" \
        --env-file "$plot_env" --execute > "$plot_logs/$plot_action.log" 2>&1
done
if [[ -e "$plot_root/figures" ]]; then
    mv "$plot_root/figures" "$plot_root/figures_before_$plot_stamp"
fi
echo "$(date +%H:%M:%S) figures"
"$plot_python" -m opera.run.paper_experiments figures --env-file "$plot_env" \
    --execute \
    > "$plot_logs/figures.log" 2>&1
for plot_action in paper-numbers supplement-tables; do
    echo "$(date +%H:%M:%S) $plot_action"
    "$plot_python" -m opera.run.paper_experiments "$plot_action" \
        --env-file "$plot_env" --execute > "$plot_logs/$plot_action.log" 2>&1
done
echo "Finished. Inspect $plot_logs, including skipped inputs in figures.log."
