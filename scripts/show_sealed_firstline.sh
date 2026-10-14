#!/usr/bin/env bash
# Read the sealed cohort manifest; verify the mapped first-line file it names.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"
METADATA="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
python - "$METADATA" <<'PY'
import hashlib, json, sys
from pathlib import Path
metadata = Path(sys.argv[1])
manifest_path = metadata.parent / 'manifest_INTERNAL.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
firstline = Path(manifest['firstline'])
print('Sealed metadata:', metadata)
print('Manifest:', manifest_path)
print('Mapped first-line table:', firstline)
print('First-line patients:', manifest['n_firstline'])
if not firstline.is_file():
    raise SystemExit('The manifest-named table is missing; locate its recorded SHA256 rather than choosing the newest candidate')
digest = hashlib.sha256()
with firstline.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
if digest.hexdigest() != manifest['sha256']['firstline']:
    raise SystemExit('First-line SHA256 differs from the sealed manifest; do not use this file as the locked table')
print('First-line SHA256 matches sealed manifest:', digest.hexdigest())
PY
