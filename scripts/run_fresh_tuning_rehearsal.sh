#!/usr/bin/env bash
# Fresh tuning inventory + analyses/figures. Never opens held-out predictions.
# Run from a GPU allocation. Existing trained fits/base exports are fixed inputs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${OPERA_PYTHON:-.venv-v100/bin/python}"
stage="${STAGE:-16}"
model="${OPERA_MODEL:-J_OA_OWN_L}"
base_env="${OPERA_ENV_FILE:-.env}"
job="${1:-data/paper_256_v3/analysis/fresh_tuning_$(date +%Y%m%d_%H%M%S)}"
# Refuse to overwrite an earlier rehearsal; no source-freshness scan.
"$PY" - "$base_env" "$job" "$model" "$stage" <<'PY'
import json, sys
from pathlib import Path
from opera.run.paper_experiments import load_environment, path_value
base, destination, model, stage = sys.argv[1:]
env = load_environment(base)
root = Path(destination).resolve()
source = root / 'inputs'
if root.exists():
    raise FileExistsError(f'Use a new rehearsal directory: {root}')
paper = path_value(env, 'OPERA_PAPER_ROOT')
if not (paper / 'bundles' / f'stage{stage}' / 'bundle.json').is_file():
    raise FileNotFoundError(f'No stage{stage} bundle; set STAGE to your current completed bundle')
source.mkdir(parents=True)
env.update(OPERA_PAPER_ANALYSIS_ROOT=str(source), OPERA_PAPER_OPERA_MODEL=model,
    OPERA_PAPER_ROUTED_MODEL='R_AB_L', OPERA_PAPER_ANALYSIS_ESTIMAND='cif_ipcw',
    OPERA_PAPER_ANALYSIS_HORIZONS='30,60,90,180,365,730',
    OPERA_PAPER_TABULAR_HORIZONS='30,60,90,180,365,730',
    OPERA_PAPER_INVENTORY_SPLITS='tuning', OPERA_PAPER_PLOT_SPLIT='tuning',
    OPERA_PAPER_TABULAR_INVENTORY_CACHE=str(source / 'tabular_inventory_cache'))
if env.get('OPERA_PAPER_CLINICAL_SCORE_CONFIG'):
    clinical = path_value(env, 'OPERA_PAPER_CLINICAL_SCORE_CONFIG')
    if not clinical.is_file():
        raise FileNotFoundError(f'Configured clinical recipe is missing: {clinical}')
    env['OPERA_PAPER_CLINICAL_SCORE_CONFIG'] = str(clinical)
path = source / 'overnight.env'
path.write_text(''.join(f'{k}={json.dumps(str(v))}\n' for k,v in env.items()
    if k.startswith(('OPERA_', 'BONSAI_'))))
path.chmod(0o600)
print(f'Tuning rehearsal: {root}\nPrimary: {model}; stage {stage}\nInputs: {source}', flush=True)
PY
"$PY" -u -m opera.run.paper_experiments inventory --stage "$stage" \
  --env-file "$job/inputs/overnight.env" --execute
"$PY" -u -m opera.run.rerun_full_tuning_rehearsal \
  --analysis-root "$job/inputs" --output-dir "$job/build" --defer-national
