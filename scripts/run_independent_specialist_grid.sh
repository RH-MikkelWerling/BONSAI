#!/usr/bin/env bash
# Plan once, then run bounded restart-safe independent specialist batches.
set -euo pipefail

MODE="${1:-plan}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/setup_v100.sh"
export CUDA_VISIBLE_DEVICES="${OPERA_LOCAL_GPU:-0}"

BASE="${OPERA_CANDIDATE_DIR:-$REPO/data/results/approved_neural_candidates_router_clean_20261007_v1}/time2vec_plain_shared_bce.yaml"
ROOT="${OPERA_SPECIALIST_ROOT:-$REPO/data/models/independent_specialist_grid_20261012_v1}"
SHARD_COUNT="${OPERA_SPECIALIST_SHARD_COUNT:-1}"
SHARD_INDEX="${OPERA_SPECIALIST_SHARD_INDEX:-0}"
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ && "$SHARD_INDEX" =~ ^(0|[1-9][0-9]*)$ ]] || {
  echo "Shard count/index must be nonnegative integers, count at least one" >&2; exit 2;
}
(( SHARD_INDEX < SHARD_COUNT )) || { echo "Shard index must be less than count" >&2; exit 2; }
printf -v SHARD_NAME 'shard_%02d_of_%02d' "$SHARD_INDEX" "$SHARD_COUNT"
SHARD_ROOT="$ROOT/$SHARD_NAME"
SEQUENCE="${OPERA_SEQUENCE_DATA_DIR:-$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_with_sor_final_v1}"
CACHE="${OPERA_OUTCOME_LABEL_CACHE_DIR:-$REPO/data/results/outcome_label_cache_approved_v2_INTERNAL}"
export BONSAI_COHORT_MEMBERSHIP="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
export BONSAI_OUTCOMES_DIR="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}/outcomes"

if [[ "$MODE" == plan ]]; then
  [[ -s "$BASE" ]] || { echo "Missing base config: $BASE" >&2; exit 1; }
  args=()
  if [[ "${OPERA_SPECIALIST_EXCLUDE_CLL:-0}" == 1 ]]; then args+=(--exclude-fine CLL); fi
  python -m opera.run.prepare_independent_specialist_grid \
    --base-config "$BASE" --output-dir "$ROOT" \
    --shard-count "$SHARD_COUNT" "${args[@]}"
  echo "Review $ROOT/manifest_report.json before running the queue."
  exit 0
fi
[[ "$MODE" == run ]] || { echo "Usage: bash scripts/run_independent_specialist_grid.sh [plan|run]" >&2; exit 2; }
: "${OPERA_DAPT_CKPT:?Set OPERA_DAPT_CKPT to the same DAPT encoder as the locked router}"
REFERENCE="${OPERA_LOCKED_ROUTER_RUN:-$REPO/data/models/approved_neural_router_guarded_20261011_v1/time2vec_router_interacting_guarded_scale07_e2e}"
for input in "$BASE" "$SHARD_ROOT/specialist_manifest.csv" "$ROOT/manifest_report.json" \
             "$OPERA_DAPT_CKPT" "$REFERENCE/hydra/config.yaml" "$SEQUENCE/vocabulary.pt" "$BONSAI_COHORT_MEMBERSHIP" \
             "$BONSAI_OUTCOMES_DIR/overall_survival.parquet"; do
  [[ -s "$input" ]] || { echo "Missing or empty input: $input" >&2; exit 1; }
done
command -v flock >/dev/null || { echo "flock is required for a safe single-worker queue" >&2; exit 1; }
MAX_NEW="${OPERA_SPECIALIST_MAX_NEW:-25}"
[[ "$MAX_NEW" =~ ^[1-9][0-9]*$ ]] || { echo "OPERA_SPECIALIST_MAX_NEW must be positive" >&2; exit 2; }
python - "$BASE" "$ROOT/manifest_report.json" "$REFERENCE/hydra/config.yaml" "$OPERA_DAPT_CKPT" <<'PY'
import hashlib, json, sys
from pathlib import Path
import yaml
with open(sys.argv[1], 'rb') as handle:
    digest = hashlib.sha256(handle.read()).hexdigest()
with open(sys.argv[2], encoding='utf-8') as handle:
    report = json.load(handle)
if digest != report['base_config_sha256']:
    raise SystemExit('Base config changed since manifest planning; stop and review')
reference = yaml.safe_load(Path(sys.argv[3]).read_text(encoding='utf-8'))
if Path(reference['dapt_ckpt']).resolve() != Path(sys.argv[4]).resolve():
    raise SystemExit('Specialist DAPT checkpoint differs from the locked router')
PY
python - "$ROOT/manifest_report.json" "$SHARD_COUNT" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    report = json.load(handle)
if report['shard_count'] != int(sys.argv[2]):
    raise SystemExit('Shard count differs from the planned manifest')
PY
mkdir -p "$SHARD_ROOT/configs" "$SHARD_ROOT/runs" "$SHARD_ROOT/logs"
exec 9>"$SHARD_ROOT/.queue.lock"
flock -n 9 || { echo "Another worker is running in $SHARD_ROOT" >&2; exit 1; }

mapfile -t CELLS < <(python - "$SHARD_ROOT/specialist_manifest.csv" <<'PY'
import csv, sys
with open(sys.argv[1], newline='', encoding='utf-8') as handle:
    for row in csv.DictReader(handle):
        print('|'.join(row[key] for key in ('fine_cohort', 'parent_cohort', 'outcome')))
PY
)
echo "Specialist $SHARD_NAME: ${#CELLS[@]} planned cells; at most $MAX_NEW new attempts in this invocation. GPU=${OPERA_LOCAL_GPU:-0}."
started=0
failures=0
for cell in "${CELLS[@]}"; do
  IFS='|' read -r fine parent outcome <<<"$cell"
  [[ "$fine" =~ ^[A-Za-z0-9_]+$ && "$parent" =~ ^[A-Za-z0-9_]+$ && "$outcome" =~ ^[A-Za-z0-9_]+$ ]] || {
    echo "Unsafe manifest cell: $cell" >&2; exit 1;
  }
  name="${fine,,}_${outcome}"
  config="$SHARD_ROOT/configs/$name.yaml"
  run="$SHARD_ROOT/runs/$name"
  log="$SHARD_ROOT/logs/$name.log"
  failed="$SHARD_ROOT/logs/$name.failed"
  [[ -s "$run/training_complete.json" ]] && continue
  if [[ -e "$failed" ]]; then
    if [[ "${OPERA_RETRY_EMPTY_BATCH_FAILURES:-0}" == 1 ]] && \
       [[ -s "$log" ]] && grep -Fq 'No training batches.' "$log"; then
      backup="${failed}.before_partial_batch_fix"
      [[ ! -e "$backup" ]] || { echo "Retry backup already exists: $backup" >&2; exit 1; }
      mv -- "$failed" "$backup"
      echo "RETRY previously empty-batch fit: $name (old failure preserved at $backup)"
    else
      continue
    fi
  fi
  (( started < MAX_NEW )) || break
  if [[ ! -e "$config" ]]; then
    python -m opera.run.prepare_independent_specialist_pilot \
      --base-config "$BASE" --parent-cohort "$parent" --fine-cohort "$fine" \
      --outcome "$outcome" --max-epochs 30 --full-run \
      --label-cache-dir "$CACHE" --output "$config"
  fi
  python - "$config" "$fine" "$outcome" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
cell = c.get('specialist_run', {})
if (cell.get('fine_cohort') != sys.argv[2] or cell.get('outcome') != sys.argv[3]
        or cell.get('purpose') != 'independent_dapt_specialist_full_fit'
        or c['training']['epochs'] != 30
        or c['training']['eval_monitor_metric'] != 'val/loss'
        or c['cross_outcome']['endpoint_groups'] != {}):
    raise SystemExit('Existing config is not this approved independent specialist cell')
PY
  python -m opera.run.preflight_direct_competing_risk \
    --config-dir "$SHARD_ROOT/configs" --config-name "$name" \
    --require-existing-paths --override "dapt_ckpt=$OPERA_DAPT_CKPT" \
    --override "paths.sequence_data_dir=$SEQUENCE"
  started=$((started + 1))
  echo "START $name; $(date -Is); attempt $started/$MAX_NEW"
  if python -u -m opera.run.contrastive_multicohort \
      --config-path "$SHARD_ROOT/configs" --config-name "$name" \
      "dapt_ckpt=$OPERA_DAPT_CKPT" "paths.sequence_data_dir=$SEQUENCE" \
      "hydra.run.dir=$run" >"$log" 2>&1; then
    [[ -s "$run/training_complete.json" ]] || { echo "No completion marker: $name" >&2; exit 1; }
    printf '%s\t%s\tcomplete\t%s\n' "$fine" "$outcome" "$(date -Is)" >>"$SHARD_ROOT/status.tsv"
    echo "COMPLETE $name"
  else
    printf '%s\t%s\tfailed\t%s\n' "$fine" "$outcome" "$(date -Is)" >>"$SHARD_ROOT/status.tsv"
    printf 'Inspect %s before retrying.\n' "$log" >"$failed"
    echo "FAILED $name; inspect $log" >&2
    tail -n 30 "$log" >&2 || true
    failures=$((failures + 1))
    (( failures < 3 )) || { echo "Stopped after three failures" >&2; exit 1; }
  fi
done
echo "Batch finished: $started attempted; $failures failed. Re-run the same command to continue."
