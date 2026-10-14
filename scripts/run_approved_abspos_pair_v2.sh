#!/usr/bin/env bash
# Launch one arm of the matched 69-outcome Fourier/scaled-Time2Vec comparison.
# Use on two separate one-GPU servers after source setup_v100.sh.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != fourier && "$1" != time2vec ) ]]; then
  echo "Usage: bash scripts/run_approved_abspos_pair_v2.sh fourier|time2vec" >&2
  exit 2
fi
arm="$1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${BONSAI_PROCESSED_DATA:?Source setup_v100.sh first}"
: "${BONSAI_CONFIG_PATH:?Source setup_v100.sh first}"
# Each server exposes its own single GPU as local device 0. Never assign GPU 1
# merely because the other arm is running on another server.
export CUDA_VISIBLE_DEVICES=0

RELEASE="$REPO/data/results/approved_outcome_release_v2_20261005T200002Z"
METADATA="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
SEQUENCE="${OPERA_SEQUENCE_DATA_DIR:-$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_with_sor_final_v1}"
MODEL_ROOT="${OPERA_PRETRAIN_MODEL_ROOT:-$REPO/data/models/daly_care_t100_joined_bins_with_sor_final_v1}"
PRETRAIN_RUN_NAME="${OPERA_PRETRAIN_RUN_NAME:-final_b4_20261003}"
FOURIER_RUN="$MODEL_ROOT/pretrain_harmonized_fourier_len8192_20ep/$PRETRAIN_RUN_NAME"
TIME2VEC_RUN="$MODEL_ROOT/pretrain_harmonized_scaled_time2vec_len8192_20ep/$PRETRAIN_RUN_NAME"
if [[ "$arm" == fourier ]]; then
  CKPT="$FOURIER_RUN/version_0/best.ckpt"
else
  CKPT="$TIME2VEC_RUN/version_0/best.ckpt"
fi
RUN_ROOT="${OPERA_ABSPOS_RUN_ROOT:-$REPO/data/models/approved_outcome_abspos_v2_20261005T202202Z}"
RUN_DIR="$RUN_ROOT/$arm"

export BONSAI_COHORT_MEMBERSHIP="$METADATA"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"
for path in "$METADATA" "$CKPT" "$SEQUENCE/subject_data_train.pt" \
            "$SEQUENCE/subject_data_tuning.pt" "$SEQUENCE/vocabulary.pt"; do
  [[ -s "$path" ]] || { echo "Missing or empty input: $path" >&2; exit 1; }
done
python - "$METADATA" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
manifest = json.loads((path.parent / "manifest_INTERNAL.json").read_text())
repair = manifest.get("id_type_repair", {})
if repair.get("converted_overlap", 0) < 10000:
    raise SystemExit("Metadata has no verified physical-subject ID type repair")
PY
[[ ! -e "$RUN_DIR" ]] || { echo "Run directory already exists: $RUN_DIR" >&2; exit 1; }

python -m opera.run.verify_abspos_pair_v2 \
  --fourier-run "$FOURIER_RUN" --time2vec-run "$TIME2VEC_RUN" \
  --sequence-data "$SEQUENCE"
python -m opera.run.verify_approved_outcome_training_v2 \
  --release-dir "$RELEASE" \
  --config "$RELEASE/generated_configs/direct_cr_bce_lowrank_endpoint_groups_v1.yaml"
python -m opera.run.preflight_direct_competing_risk \
  --config-dir "$RELEASE/generated_configs" \
  --config-name direct_cr_bce_lowrank_endpoint_groups_v1 \
  --require-existing-paths \
  --override "dapt_ckpt=$CKPT" \
  --override "paths.sequence_data_dir=$SEQUENCE"

echo "Validated $arm; local GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CKPT"
echo "Sequence data: $SEQUENCE"
echo "Run directory: $RUN_DIR"
if [[ "${OPERA_LAUNCH:-0}" != 1 ]]; then
  echo "Dry run only. Set OPERA_LAUNCH=1 to train."
  exit 0
fi
python -u -m opera.run.contrastive_multicohort \
  --config-path "$RELEASE/generated_configs" \
  --config-name direct_cr_bce_lowrank_endpoint_groups_v1 \
  "dapt_ckpt=$CKPT" \
  "paths.sequence_data_dir=$SEQUENCE" \
  "hydra.run.dir=$RUN_DIR"
