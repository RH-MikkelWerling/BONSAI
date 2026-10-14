#!/usr/bin/env bash
# Sequential, restart-safe CLL specialist fits from one DAPT encoder.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${BONSAI_PROCESSED_DATA:?Source setup_v100.sh first}"
: "${OPERA_CANDIDATE_DIR:?Set the approved neural candidate directory}"
: "${OPERA_DAPT_CKPT:?Set the Time2Vec DAPT checkpoint}"
: "${OPERA_SPECIALIST_ROOT:?Set a fixed output root for this CLL queue}"

BASE="$OPERA_CANDIDATE_DIR/time2vec_plain_shared_bce.yaml"
SEQUENCE="${OPERA_SEQUENCE_DATA_DIR:-$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_with_sor_final_v1}"
CACHE="${OPERA_OUTCOME_LABEL_CACHE_DIR:-$REPO/data/results/outcome_label_cache_approved_v2_INTERNAL}"
export BONSAI_COHORT_MEMBERSHIP="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
export BONSAI_OUTCOMES_DIR="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}/outcomes"
for input in "$BASE" "$OPERA_DAPT_CKPT" "$SEQUENCE/vocabulary.pt" \
             "$BONSAI_COHORT_MEMBERSHIP" "$BONSAI_OUTCOMES_DIR/overall_survival.parquet"; do
  [[ -s "$input" ]] || { echo "Missing or empty input: $input" >&2; exit 1; }
done
command -v flock >/dev/null || { echo "flock is required for a safe single-worker queue" >&2; exit 1; }
mkdir -p "$OPERA_SPECIALIST_ROOT/configs" "$OPERA_SPECIALIST_ROOT/runs" "$OPERA_SPECIALIST_ROOT/logs"
exec 9>"$OPERA_SPECIALIST_ROOT/.queue.lock"
flock -n 9 || { echo "Another CLL specialist queue is running" >&2; exit 1; }

mapfile -t OUTCOMES < <(python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); excluded=set(c["cohorts"]["CLL_SLL"].get("exclude_outcomes", [])); print("\n".join(n for n in c["outcomes"] if n not in excluded))' "$BASE")
[[ ${#OUTCOMES[@]} -gt 0 ]] || { echo "No CLL outcomes in $BASE" >&2; exit 1; }
echo "CLL specialist queue: ${#OUTCOMES[@]} approved outcomes; output=$OPERA_SPECIALIST_ROOT"

failures=0
for outcome in "${OUTCOMES[@]}"; do
  [[ "$outcome" =~ ^[A-Za-z0-9_]+$ ]] || { echo "Unsafe outcome name: $outcome" >&2; exit 1; }
  name="cll_${outcome}"
  config="$OPERA_SPECIALIST_ROOT/configs/$name.yaml"
  run="$OPERA_SPECIALIST_ROOT/runs/$name"
  log="$OPERA_SPECIALIST_ROOT/logs/$name.log"
  failed="$OPERA_SPECIALIST_ROOT/logs/$name.failed"
  if [[ -s "$run/training_complete.json" ]]; then
    echo "SKIP complete: $name"
    continue
  fi
  if [[ -e "$failed" ]]; then
    if [[ -s "$log" ]] && grep -Fq 'No endpoint groups meet the prespecified tuning support threshold.' "$log"; then
      echo "SKIP unsupported tuning cell: $name (previously recorded failure)"
    else
      echo "SKIP previous failure: $name; inspect $log before retrying"
    fi
    continue
  fi
  if [[ ! -e "$config" ]]; then
    python -m opera.run.prepare_independent_specialist_pilot \
      --base-config "$BASE" --parent-cohort CLL_SLL --fine-cohort CLL \
      --outcome "$outcome" --max-epochs 30 --full-run \
      --label-cache-dir "$CACHE" --output "$config"
  fi
  python - "$config" "$outcome" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
cell = config.get("specialist_run", {})
if (cell.get("fine_cohort") != "CLL" or cell.get("outcome") != sys.argv[2]
        or config["training"]["epochs"] != 30
        or config["cross_outcome"].get("endpoint_groups") != {}
        or cell.get("purpose") != "independent_dapt_specialist_full_fit"):
    raise SystemExit(f"Existing specialist config is not this approved 30-epoch BCE-selected cell: {sys.argv[1]}")
PY
  python -m opera.run.preflight_direct_competing_risk \
    --config-dir "$OPERA_SPECIALIST_ROOT/configs" --config-name "$name" \
    --require-existing-paths --override "dapt_ckpt=$OPERA_DAPT_CKPT" \
    --override "paths.sequence_data_dir=$SEQUENCE"
  echo "START $name; $(date -Is); checkpoint=$run/last.ckpt"
  if python -u -m opera.run.contrastive_multicohort \
      --config-path "$OPERA_SPECIALIST_ROOT/configs" --config-name "$name" \
      "dapt_ckpt=$OPERA_DAPT_CKPT" "paths.sequence_data_dir=$SEQUENCE" \
      "hydra.run.dir=$run" >"$log" 2>&1; then
    [[ -s "$run/training_complete.json" ]] || { echo "No completion marker: $name" >&2; exit 1; }
    printf '%s\tcomplete\t%s\n' "$outcome" "$(date -Is)" >>"$OPERA_SPECIALIST_ROOT/status.tsv"
    echo "COMPLETE $name; $(date -Is)"
  else
    if grep -Fq 'No endpoint groups meet the prespecified tuning support threshold.' "$log"; then
      printf '%s\tunsupported_tuning\t%s\n' "$outcome" "$(date -Is)" >>"$OPERA_SPECIALIST_ROOT/status.tsv"
      printf 'No endpoint groups meet tuning support; inspect %s.\n' "$log" >"$failed"
      echo "UNSUPPORTED TUNING $name; continuing to next cell" >&2
    else
      printf '%s\tfailed\t%s\n' "$outcome" "$(date -Is)" >>"$OPERA_SPECIALIST_ROOT/status.tsv"
      printf 'Training failed; inspect %s before retrying.\n' "$log" >"$failed"
      echo "FAILED $name; inspect $log" >&2
      echo "--- Last 35 log lines for $name ---" >&2
      tail -n 35 "$log" >&2 || true
      echo "--- End log excerpt ---" >&2
      failures=$((failures + 1))
      if (( failures >= 3 )); then
        echo "Stopped after three unexpected failures; inspect logs before relaunching" >&2
        exit 1
      fi
    fi
  fi
done
echo "CLL specialist queue pass finished; $(date -Is)"
