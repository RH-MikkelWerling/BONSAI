#!/usr/bin/env bash
# Run on the original dedicated high-RAM allocation; never stops another project.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
"$python_bin" scripts/check_results_resources.py --minimum-free-gib 96 --require-no-coordinator
"$python_bin" - <<'PY'
import fcntl,json
from pathlib import Path
import yaml
from opera.evaluation.distributed_analysis import signature
from opera.run.migrate_cpu_memory_checkpoints import migrate
root=Path('data/paper_256_v3/analysis/local_test_20261109_four_allocations')
base=root/'_cpu_checkpoints_INTERNAL'
config=yaml.safe_load((root/'analysis_inventory/analysis.yaml').read_text())
with (root/'.coordinator_launcher.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=json.loads((base/'active.json').read_text())['generation']
    if active==signature(config):
        print('Checkpoint generation already matches the current recovery implementation.')
    else:
        migrate(config,base,execute=True,fix='identity-collection')
PY
bash scripts/start_test_coordinator.sh
