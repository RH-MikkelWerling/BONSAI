#!/usr/bin/env bash
# Release-local, sequential one-GPU queue. Dry-run/preflight unless OPERA_LAUNCH=1.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${OPERA_RELEASE_DIR:?Set the exact approved_outcome_release_v2 directory}"
: "${BONSAI_COHORT_MEMBERSHIP:?Set the new sealed metadata parquet}"
: "${BONSAI_OUTCOMES_DIR:?Set the approved release outcomes directory}"
[[ "$BONSAI_OUTCOMES_DIR" == "$OPERA_RELEASE_DIR/outcomes" ]] || {
  echo "BONSAI_OUTCOMES_DIR does not match OPERA_RELEASE_DIR" >&2; exit 1;
}
[[ -s "$BONSAI_COHORT_MEMBERSHIP" ]] || { echo "Missing metadata parquet" >&2; exit 1; }
[[ -s "$(dirname "$BONSAI_COHORT_MEMBERSHIP")/manifest_INTERNAL.json" ]] || {
  echo "Missing sealed metadata manifest" >&2; exit 1;
}
CONFIG_DIR="$OPERA_RELEASE_DIR/generated_configs"
CONFIG_NAME=direct_cr_bce_lowrank_endpoint_groups_v1
python -m opera.run.verify_approved_outcome_training_v2 \
  --release-dir "$OPERA_RELEASE_DIR" \
  --config "$CONFIG_DIR/$CONFIG_NAME.yaml"

# Same roster, loss, validation criterion and seed. Only encoder or readout
# changes per candidate. Do not warm-start from a previous outcome head.
read -r -a variants <<< "${OPERA_VARIANTS:-fourier_factorized time2vec_factorized}"
RUN_ROOT="${OPERA_RUN_ROOT:-$REPO/data/models/approved_outcome_v2_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; exit 1; }
variant_overrides() {
  local variant="$1"
  overrides=()
  case "$variant" in
    fourier_factorized) ;;
    time2vec_factorized)
      : "${OPERA_TIME2VEC_CKPT:?Set a completed, vocabulary-compatible Time2Vec checkpoint}"
      [[ -s "$OPERA_TIME2VEC_CKPT" ]] || { echo "Missing Time2Vec checkpoint" >&2; exit 1; }
      overrides+=("dapt_ckpt=$OPERA_TIME2VEC_CKPT") ;;
    fourier_shared)
      overrides+=("competing_risk.readout_mode=shared") ;;
    fourier_family_trunks)
      overrides+=("competing_risk.readout_mode=shared" "competing_risk.head_mode=family_trunks") ;;
    fourier_outcome_attention)
      overrides+=("competing_risk.readout_mode=outcome_attention") ;;
    *) echo "Unknown candidate: $variant" >&2; exit 1 ;;
  esac
}
for variant in "${variants[@]}"; do
  variant_overrides "$variant"
  echo "Preflight: $variant"
  preflight=(python -m opera.run.preflight_direct_competing_risk
    --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME" --require-existing-paths)
  for override in "${overrides[@]}"; do preflight+=(--override "$override"); done
  "${preflight[@]}"
done
if [[ "${OPERA_LAUNCH:-0}" == 1 ]]; then
  for variant in "${variants[@]}"; do
    variant_overrides "$variant"
    run_dir="$RUN_ROOT/$variant"
    [[ ! -e "$run_dir" ]] || { echo "Run directory already exists: $run_dir" >&2; exit 1; }
    echo "Training: $variant -> $run_dir"
    python -m opera.run.contrastive_multicohort \
      --config-path "$CONFIG_DIR" --config-name "$CONFIG_NAME" \
      "${overrides[@]}" "hydra.run.dir=$run_dir"
  done
fi
echo "Queue finished (OPERA_LAUNCH=${OPERA_LAUNCH:-0})"
