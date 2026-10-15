#!/usr/bin/env bash
# Run in the activated server environment, inside a GPU allocation for --gpu.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
"${OPERA_PYTHON:-python}" - "$@" <<'PY'
import sys
import matplotlib
matplotlib.use('Agg')
import numpy, pandas, pyarrow, torch, lightning, xgboost, yaml, omegaconf
print('Core imports passed; torch', torch.__version__, 'xgboost', xgboost.__version__)
if '--gpu' in sys.argv:
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable in this allocation')
    visible = torch.cuda.device_count()
    if visible != 1:
        raise SystemExit(f'Expected one visible GPU per worker; got {visible}. Set the allocation-specific CUDA_VISIBLE_DEVICES.')
    print('Allocated GPU:', torch.cuda.get_device_name(0))
PY
exec "${OPERA_PYTHON:-python}" -m opera.run.paper_experiments check
