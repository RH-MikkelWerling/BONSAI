#!/usr/bin/env bash
# One encoder per existing GPU allocation. Repeat exactly to resume this release.
set -euo pipefail
cd "$(dirname "$0")/.."
family="${1:?Usage: bash scripts/run_cutoff_recovery_encoder.sh cohort|adapted}"
case "$family" in cohort|adapted) ;; *) exit 2 ;; esac
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset OPERA_ANALYSIS_CHECKPOINT_DIR
root="${OPERA_RECOVERY_ROOT:-data/cutoff_recovery_20261121}"
root="$(cd "$root" && pwd)"
mkdir -p logs
claim="$root/$family/encoder.claim"
mkdir "$claim" || { echo "Encoder already claimed; inspect $claim before restarting"; exit 1; }
printf '%s %s\n' "$(hostname)" "$$" > "$claim/owner"
trap 'rm -f "$claim/owner"; rmdir "$claim"' EXIT
python scripts/check_results_resources.py --minimum-free-gib 96 --require-no-coordinator --require-free-gpu
python - "$root" "$family" <<'PY'
import sys,yaml
from pathlib import Path
from opera.run.finish_figure_inputs import sha
root=Path(sys.argv[1]);family=sys.argv[2]
receipt=yaml.safe_load((root/'encoder_preparation.yaml').read_text())[family]
config=root/family/'config.yaml'
if sha(config)!=receipt['new_config_sha256']:raise ValueError('Recovery config changed')
cfg=yaml.safe_load(config.read_text())
for field,item in receipt['inputs'].items():
    stat=Path(item['path']).stat()
    if (stat.st_size,stat.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):raise ValueError(f'Recovery source changed: {field}')
for field in ('population','vocabulary'):
    path=cfg['paths']['population' if field=='population' else 'vocab']
    if sha(Path(path))!=receipt[f'{field}_sha256']:raise ValueError(f'{field} source changed')
if cfg['training']['cutoff_date']!={'year':2022,'month':1,'day':1}:raise ValueError('Wrong cutoff')
if not cfg['training'].get('recovery_cutoff_audit'):raise ValueError('Missing runtime audit')
if cfg['paths'].get('ckpt_path'):raise ValueError('Do not resume another release')
# Existing saved Hydra config must agree; auto-resume is confined to this directory.
saved=root/family/'pretraining/hydra/config.yaml'
if saved.exists():
    from opera.run.prepare_cutoff_recovery import resolved
    old=resolved(saved)
    for section in ('model','paths','training'):
        if old[section]!=cfg[section]:raise ValueError(f'Saved recovery {section} differs')
if family=='adapted':
    from opera.run.prepare_cutoff_recovery import NATIONAL_SHA
    if sha(Path(cfg['paths']['initialization_ckpt']))!=NATIONAL_SHA:raise ValueError('National source changed')
PY
python -u -m bonsai.run.pretrain --config-path "$root/$family" --config-name config \
  "hydra.run.dir=$root/$family/pretraining"
