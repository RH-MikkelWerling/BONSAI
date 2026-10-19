#!/usr/bin/env bash
# Width probe: the stage-1 joint model fine-tuned from the 256-wide encoder at
# 10 epochs (its epoch_010.ckpt milestone) and from the 512-wide encoder after
# its 10 epochs (last.ckpt), same bundle, data and protocol. One GPU per probe.
#
#   bash scripts/run_width_probe.sh 256 preview   # encoder audit only, no training
#   bash scripts/run_width_probe.sh 256           # fine-tune (GPU)
#   bash scripts/run_width_probe.sh 512 preview
#   bash scripts/run_width_probe.sh 512           # fine-tune (GPU)
#   bash scripts/run_width_probe.sh compare       # tuning scores and the call
#
# PROBE_EPOCHS=N caps fine-tuning epochs (use the same N for both probes).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f setup_v100.sh ]] && source setup_v100.sh
set -a; source .env; set +a
PY="${OPERA_PYTHON:-python}"
REL="${BONSAI_DATASET:-daly_care_t100_joined_bins_with_sor_final_v2}"
MODELS="${PROBE_MODELS_DIR:-$PWD/data/models/$REL}"
BUNDLE="$OPERA_PAPER_ROOT/bundles/stage1"
OUT="$OPERA_PAPER_ROOT/width_probe"
WHAT="${1:?Usage: run_width_probe.sh 256|512|compare [preview]}"

if [[ "$WHAT" == compare ]]; then
  exec "$PY" -m opera.run.width_probe --compare --output-dir "$OUT"
fi
case "$WHAT" in
  256) CKPT="$MODELS/pretrain_scaled_time2vec_hidden256/version_0/epoch_010.ckpt" ;;
  512) CKPT="$MODELS/pretrain_scaled_time2vec_hidden512/version_0/last.ckpt"
       [[ -f "$(dirname "$CKPT")/checkpoint_metadata.json" ]] || {
         echo "The 512 run has not finished (no checkpoint_metadata.json yet)" >&2; exit 1; } ;;
  *) echo "Usage: run_width_probe.sh 256|512|compare [preview]" >&2; exit 2 ;;
esac
[[ -f "$CKPT" ]] || { echo "Missing encoder checkpoint: $CKPT" >&2; exit 1; }
[[ -d "$BUNDLE" ]] || { echo "No stage-1 bundle at $BUNDLE: run prepare first" >&2; exit 1; }
ARGS=(--bundle-dir "$BUNDLE" --encoder-checkpoint "$CKPT" --width "$WHAT"
      --label "w${WHAT}_e10" --output-dir "$OUT")
[[ -n "${PROBE_EPOCHS:-}" ]] && ARGS+=(--epochs "$PROBE_EPOCHS")
[[ "${2:-}" == preview ]] || ARGS+=(--execute)
echo "Width probe $WHAT: $CKPT -> $OUT/w${WHAT}_e10"
exec "$PY" -m opera.run.width_probe "${ARGS[@]}"
