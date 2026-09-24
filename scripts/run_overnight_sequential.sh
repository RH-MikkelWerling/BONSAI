#!/usr/bin/env bash
# Resume harmonized pretraining, then run the six matched readout experiments
# plus the matched multi-horizon BCE objective sequentially on the current GPU.

set -euo pipefail

: "${NEW_PROCESSED:?Set NEW_PROCESSED to the harmonized processed-data directory}"
: "${PRETRAIN_RUN_DIR:?Set PRETRAIN_RUN_DIR to the directory containing version_0}"
: "${ARCH_ROOT:?Set ARCH_ROOT to the architecture experiment output directory}"

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-$ARCH_ROOT/logs}"
PRETRAIN_MODEL_DIR="$PRETRAIN_RUN_DIR/version_0"
DAPT_CKPT="$PRETRAIN_MODEL_DIR/best.ckpt"
LAST_CKPT="$PRETRAIN_MODEL_DIR/last.ckpt"

configs=(
  direct_cr_architecture_shared_linear
  direct_cr_architecture_outcome_mlp
  direct_cr_architecture_family_trunks
  direct_cr_architecture_outcome_attention
  direct_cr_architecture_shared_attention_prototypes
  direct_cr_architecture_factorized_cohort_outcome_prototypes
  direct_cr_objective_horizon_bce
)

cd "$REPO_ROOT"
mkdir -p "$LOG_ROOT" "$ARCH_ROOT"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] Regenerating experiment configs."
"$PYTHON_BIN" -m opera.run.generate_sweep_configs

pretrain_args=(
  --config-name pretrain_harmonized_fourier_len8192_20ep
  "dataset=$(basename "$NEW_PROCESSED")"
  "paths.dir=$NEW_PROCESSED"
  "hydra.run.dir=$PRETRAIN_RUN_DIR"
)
if [[ -s "$LAST_CKPT" ]]; then
  echo "[$(timestamp)] Resuming pretraining from $LAST_CKPT"
  pretrain_args+=("paths.ckpt_path=$LAST_CKPT")
else
  echo "[$(timestamp)] No last.ckpt found; starting pretraining in $PRETRAIN_RUN_DIR"
fi

if ! "$PYTHON_BIN" -u -m bonsai.run.pretrain "${pretrain_args[@]}" \
  2>&1 | tee "$LOG_ROOT/pretraining.log"; then
  echo "[$(timestamp)] Pretraining failed; downstream runs were not started." >&2
  exit 1
fi
if [[ ! -s "$DAPT_CKPT" ]]; then
  echo "[$(timestamp)] Missing completed pretraining checkpoint: $DAPT_CKPT" >&2
  exit 1
fi
echo "[$(timestamp)] Pretraining finished successfully: $DAPT_CKPT"

echo "[$(timestamp)] Running preflight for all seven matched experiments."
for config in "${configs[@]}"; do
  "$PYTHON_BIN" -m opera.run.preflight_direct_competing_risk \
    --config-name "generated/$config" \
    --require-existing-paths \
    --override run_id=0 \
    --override dapt_ckpt="$DAPT_CKPT" \
    --override paths.sequence_data_dir="$NEW_PROCESSED"
done

sleep 10

failed=()
for config in "${configs[@]}"; do
  run_dir="$ARCH_ROOT/$config"
  echo "[$(timestamp)] Starting $config"
  if "$PYTHON_BIN" -u -m opera.run.contrastive_multicohort \
    --config-name "generated/$config" \
    dapt_ckpt="$DAPT_CKPT" \
    paths.sequence_data_dir="$NEW_PROCESSED" \
    ++training.auto_resume=true \
    "hydra.run.dir=$run_dir" \
    2>&1 | tee "$LOG_ROOT/$config.log"; then
    echo "[$(timestamp)] Completed $config"
  else
    echo "[$(timestamp)] FAILED $config; continuing to the next architecture." >&2
    failed+=("$config")
  fi
done

echo "[$(timestamp)] Sequential overnight workflow finished."
if [[ "${#failed[@]}" -gt 0 ]]; then
  printf 'Failed experiments:\n' >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "All seven matched experiments completed successfully."
