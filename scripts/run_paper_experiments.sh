#!/usr/bin/env bash
# Activate the intended Python environment first; .env is parsed by Python.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
exec "${OPERA_PYTHON:-python}" -m opera.run.paper_experiments "$@"
