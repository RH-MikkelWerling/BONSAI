#!/usr/bin/env bash
# One candidate per GPU server. Preflight only unless OPERA_LAUNCH=1.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_approved_neural_candidate_v1.sh VARIANT" >&2
  exit 2
fi
VARIANT="$1"
case "$VARIANT" in
  shared|time2vec_grouped_shared_bce|fourier_plain_shared_bce|time2vec_plain_shared_bce|fourier_plain_factorized_bce|time2vec_plain_factorized_bce|family_trunks|time2vec_family_trunks|router_additive_e2e|time2vec_router_additive_e2e|router_interacting_e2e|time2vec_router_interacting_e2e|time2vec_router_disease_only_e2e|time2vec_router_outcome_only_e2e|time2vec_router_no_sources_e2e|time2vec_router_interacting_scale02_e2e|time2vec_router_interacting_scale03_e2e|outcome_attention) ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${BONSAI_PROCESSED_DATA:?Source setup_v100.sh first}"
: "${OPERA_FAMILY_DIR:?Set the successful approved_training_families_v1 output directory}"
: "${OPERA_CANDIDATE_DIR:?Set the new approved_neural_candidates_v1 directory}"
RELEASE="${OPERA_RELEASE_DIR:-$REPO/data/results/approved_outcome_release_v2_20261005T200002Z}"
METADATA="${OPERA_METADATA_PATH:-$REPO/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet}"
SEQUENCE="${OPERA_SEQUENCE_DATA_DIR:-$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_with_sor_final_v1}"
MODEL_ROOT="${OPERA_PRETRAIN_MODEL_ROOT:-$REPO/data/models/daly_care_t100_joined_bins_with_sor_final_v1}"
PRETRAIN_RUN_NAME="${OPERA_PRETRAIN_RUN_NAME:-final_b4_20261003}"
CKPT="$MODEL_ROOT/pretrain_harmonized_fourier_len8192_20ep/$PRETRAIN_RUN_NAME/version_0/best.ckpt"
TIME2VEC_RUN="$MODEL_ROOT/pretrain_harmonized_scaled_time2vec_len8192_20ep/$PRETRAIN_RUN_NAME"
if [[ "$VARIANT" == time2vec_* ]]; then
  CKPT="$TIME2VEC_RUN/version_0/best.ckpt"
fi
EXTERNAL_DAPT_CKPT="${OPERA_EXTERNAL_DAPT_CKPT:-}"
if [[ -n "$EXTERNAL_DAPT_CKPT" ]]; then
  CKPT="$EXTERNAL_DAPT_CKPT"
fi
RUN_ROOT="${OPERA_NEURAL_RUN_ROOT:-$REPO/data/models/approved_neural_v1_$(date -u +%Y%m%dT%H%M%SZ)}"
SAMPLING_ABLATION="${OPERA_SAMPLING_ABLATION:-none}"
SAMPLING_OVERRIDES=()
CACHE_OVERRIDES=()
if [[ -n "${OPERA_OUTCOME_LABEL_CACHE_DIR:-}" ]]; then
  CACHE_OVERRIDES=("+paths.outcome_label_cache_dir=$OPERA_OUTCOME_LABEL_CACHE_DIR")
fi
case "$SAMPLING_ABLATION" in
  none) RUN_DIR="$RUN_ROOT/$VARIANT" ;;
  ipw_event_horizon_bce)
    [[ "$VARIANT" == time2vec_router_interacting_scale03_e2e ]] || {
      echo "IPW sampling pilot is restricted to the clean interactive scale-0.3 candidate" >&2
      exit 2
    }
    RUN_DIR="$RUN_ROOT/${VARIANT}_ipw_event_horizon_bce"
    SAMPLING_OVERRIDES=(
      "training.batch_sampling.type=ipw_event_horizon_bce"
      "+training.batch_sampling.horizons_days=[30,60,90,180,365,730]"
      "+training.batch_sampling.event_fraction=0.5"
      "+training.batch_sampling.min_unique_events=10"
      "+training.batch_sampling.min_unique_controls=20"
      "+training.batch_sampling.seed=42"
    )
    ;;
  *) echo "Unknown OPERA_SAMPLING_ABLATION: $SAMPLING_ABLATION" >&2; exit 2 ;;
esac
ROUTER_STRESS_OVERRIDES=()
case "${OPERA_ROUTER_STRESS:-none}" in
  none) ;;
  loose_scale07)
    [[ "$VARIANT" == time2vec_router_interacting_scale03_e2e && "$SAMPLING_ABLATION" == none ]] || {
      echo "Loose-router pilot requires the clean time2vec scale-0.3 variant without sampling ablation" >&2
      exit 2
    }
    RUN_DIR="$RUN_ROOT/time2vec_router_interacting_loose_scale07_e2e"
    ROUTER_STRESS_OVERRIDES=(
      "competing_risk.adapter_residual_scale=0.7"
      "competing_risk.adapter_home_floor=0.02"
      "competing_risk.adapter_event_tau=0.0"
      "+competing_risk.adapter_shared_logit_init=0.0"
    )
    ;;
  guarded_scale07|guarded_scale07_rank16)
    case "$VARIANT" in
      time2vec_router_interacting_scale03_e2e|time2vec_router_disease_only_e2e|time2vec_router_outcome_only_e2e|time2vec_router_no_sources_e2e) ;;
      *) echo "Guarded router requires an approved Time2Vec interacting source-bank variant" >&2; exit 2 ;;
    esac
    [[ "$SAMPLING_ABLATION" == none ]] || {
      echo "Guarded-router comparison requires natural patient sampling" >&2
      exit 2
    }
    RANK=8
    if [[ "${OPERA_ROUTER_STRESS}" == guarded_scale07_rank16 ]]; then RANK=16; fi
    RUN_DIR="$RUN_ROOT/time2vec_router_interacting_guarded_scale07_e2e"
    if [[ "$RANK" == 16 ]]; then
      RUN_DIR="$RUN_ROOT/time2vec_router_interacting_guarded_scale07_rank16_e2e"
    fi
    if [[ "$VARIANT" != time2vec_router_interacting_scale03_e2e ]]; then
      [[ "$RANK" == 8 ]] || {
        echo "Source-bank controls must match the locked guarded rank-8 router" >&2; exit 2;
      }
      RUN_DIR="$RUN_ROOT/${VARIANT}_guarded_scale07_e2e"
    fi
    ROUTER_STRESS_OVERRIDES=(
      "competing_risk.adapter_residual_scale=0.7"
      "competing_risk.adapter_home_floor=0.05"
      "competing_risk.adapter_event_tau=10.0"
      "+competing_risk.adapter_shared_logit_init=0.0"
      "competing_risk.adapter_bottleneck=$RANK"
    )
    ;;
  *) echo "Unknown OPERA_ROUTER_STRESS: ${OPERA_ROUTER_STRESS}" >&2; exit 2 ;;
esac
case "${OPERA_ROUTER_BORROWED_UPDATES:-0}" in
  0) ;;
  1)
    case "$VARIANT" in
      time2vec_router_disease_only_e2e|time2vec_router_outcome_only_e2e|time2vec_router_interacting_scale03_e2e) ;;
      *) echo "Borrowed-update comparison requires at least one source bank" >&2; exit 2 ;;
    esac
    [[ "${OPERA_ROUTER_STRESS:-none}" == guarded_scale07 && "$SAMPLING_ABLATION" == none ]] || {
      echo "Borrowed-update comparison requires guarded rank 8 with natural sampling" >&2; exit 2;
    }
    RUN_DIR="${RUN_DIR}_borrowed_updates"
    ROUTER_STRESS_OVERRIDES+=("competing_risk.readout_mode=unanchored_adapters")
    ;;
  *) echo "OPERA_ROUTER_BORROWED_UPDATES must be 0 or 1" >&2; exit 2 ;;
esac
case "${OPERA_ROUTER_CURRICULUM:-inherit}" in
  inherit) ;;
  off)
    [[ "${OPERA_ROUTER_STRESS:-none}" == guarded_scale07 && \
       ( ( "$VARIANT" == time2vec_router_interacting_scale03_e2e && "${OPERA_ROUTER_BORROWED_UPDATES:-0}" == 1 ) || \
         -n "${OPERA_ROUTER_ADAPTER_EPSILON_REFERENCE:-}" ) ]] || {
      echo "Curriculum-off requires the borrowed-update comparison or a matched epsilon reference" >&2; exit 2;
    }
    RUN_DIR="${RUN_DIR}_no_curriculum"
    ROUTER_STRESS_OVERRIDES+=("competing_risk.curriculum.enabled=false")
    ;;
  *) echo "OPERA_ROUTER_CURRICULUM must be inherit or off" >&2; exit 2 ;;
esac
if [[ -n "${OPERA_ROUTER_ADAPTER_EPSILON:-}" ]]; then
  case "$VARIANT" in
    time2vec_router_disease_only_e2e|time2vec_router_outcome_only_e2e|time2vec_router_interacting_scale03_e2e) ;;
    *) echo "Adapter-epsilon comparison requires at least one source bank" >&2; exit 2 ;;
  esac
  [[ "${OPERA_ROUTER_STRESS:-none}" == guarded_scale07 && "$SAMPLING_ABLATION" == none ]] || {
    echo "Adapter-epsilon comparison requires guarded rank 8 with natural sampling" >&2; exit 2;
  }
  [[ "$OPERA_ROUTER_ADAPTER_EPSILON" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || {
    echo "Adapter epsilon must use a positive numeric notation" >&2; exit 2;
  }
  RUN_DIR="${RUN_DIR}_adaptereps_${OPERA_ROUTER_ADAPTER_EPSILON}"
  ROUTER_STRESS_OVERRIDES+=("++training.adapter_optimizer_epsilon=$OPERA_ROUTER_ADAPTER_EPSILON")
fi
if [[ -n "$EXTERNAL_DAPT_CKPT" ]]; then
  RUN_DIR="${RUN_DIR}_external_bonsai_dapt"
fi
export CUDA_VISIBLE_DEVICES="${OPERA_LOCAL_GPU:-0}"
export BONSAI_COHORT_MEMBERSHIP="$METADATA"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"

for path in "$METADATA" "$CKPT" "$SEQUENCE/subject_data_train.pt" \
            "$SEQUENCE/subject_data_tuning.pt" "$SEQUENCE/vocabulary.pt"; do
  [[ -s "$path" ]] || { echo "Missing or empty input: $path" >&2; exit 1; }
done
if [[ "${OPERA_RESUME:-0}" == 1 ]]; then
  [[ -s "$RUN_DIR/last.ckpt" ]] || {
    echo "Resume requested but last.ckpt is missing or empty: $RUN_DIR/last.ckpt" >&2
    exit 1
  }
  [[ ! -e "$RUN_DIR/training_complete.json" ]] || {
    echo "Run is already marked complete; inspect it before resuming: $RUN_DIR" >&2
    exit 1
  }
  echo "Resume requested from $RUN_DIR/last.ckpt"
else
  [[ ! -e "$RUN_DIR" ]] || {
    echo "Run directory already exists: $RUN_DIR (set OPERA_RESUME=1 to resume from last.ckpt)" >&2
    exit 1
  }
fi

python - "$METADATA" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads((Path(sys.argv[1]).parent / "manifest_INTERNAL.json").read_text())
if manifest.get("id_type_repair", {}).get("converted_overlap", 0) < 10000:
    raise SystemExit("Metadata lacks verified physical-subject ID repair")
PY
if [[ -n "$EXTERNAL_DAPT_CKPT" ]]; then
  python -m bonsai.run.audit_dapt_to_opera \
    --checkpoint "$CKPT" --vocabulary "$SEQUENCE/vocabulary.pt"
else
  python -m opera.run.verify_abspos_pair_v2 \
    --fourier-run "$MODEL_ROOT/pretrain_harmonized_fourier_len8192_20ep/$PRETRAIN_RUN_NAME" \
    --time2vec-run "$TIME2VEC_RUN" --sequence-data "$SEQUENCE"
fi

python -m opera.run.verify_approved_outcome_training_v2 \
  --release-dir "$RELEASE" \
  --config "$RELEASE/generated_configs/direct_cr_bce_lowrank_endpoint_groups_v1.yaml"
python -m opera.run.prepare_approved_neural_runs_v1 \
  --release-dir "$RELEASE" --family-dir "$OPERA_FAMILY_DIR" \
  --output-dir "$OPERA_CANDIDATE_DIR" --verify-variant "$VARIANT"
PREFLIGHT_SAMPLING_OVERRIDES=()
for override in "${SAMPLING_OVERRIDES[@]}"; do
  PREFLIGHT_SAMPLING_OVERRIDES+=(--override "$override")
done
for override in "${CACHE_OVERRIDES[@]}"; do
  PREFLIGHT_SAMPLING_OVERRIDES+=(--override "$override")
done
for override in "${ROUTER_STRESS_OVERRIDES[@]}"; do
  PREFLIGHT_SAMPLING_OVERRIDES+=(--override "$override")
done
if [[ -n "${OPERA_ROUTER_MATCH_REFERENCE:-}" ]]; then
  PREFLIGHT_SAMPLING_OVERRIDES+=(--borrowed-update-reference "$OPERA_ROUTER_MATCH_REFERENCE")
fi
if [[ -n "${OPERA_ROUTER_CURRICULUM_REFERENCE:-}" ]]; then
  PREFLIGHT_SAMPLING_OVERRIDES+=(--curriculum-off-reference "$OPERA_ROUTER_CURRICULUM_REFERENCE")
fi
if [[ -n "${OPERA_ROUTER_ADAPTER_EPSILON_REFERENCE:-}" ]]; then
  PREFLIGHT_SAMPLING_OVERRIDES+=(--adapter-epsilon-reference "$OPERA_ROUTER_ADAPTER_EPSILON_REFERENCE")
fi
python -m opera.run.preflight_direct_competing_risk \
  --config-dir "$OPERA_CANDIDATE_DIR" --config-name "$VARIANT" \
  --require-existing-paths \
  --override "dapt_ckpt=$CKPT" \
  --override "paths.sequence_data_dir=$SEQUENCE" \
  "${PREFLIGHT_SAMPLING_OVERRIDES[@]}"

echo "Validated $VARIANT on local CUDA device $CUDA_VISIBLE_DEVICES"
echo "Sampling ablation: $SAMPLING_ABLATION"
echo "Router stress pilot: ${OPERA_ROUTER_STRESS:-none}"
if [[ ${#ROUTER_STRESS_OVERRIDES[@]} -gt 0 ]]; then
  printf 'Router override: %s\n' "${ROUTER_STRESS_OVERRIDES[@]}"
fi
echo "Run directory: $RUN_DIR"
if [[ "${OPERA_LAUNCH:-0}" != 1 ]]; then
  echo "Dry run only. Set OPERA_LAUNCH=1 to train."
  exit 0
fi
python -u -m opera.run.contrastive_multicohort \
  --config-path "$OPERA_CANDIDATE_DIR" --config-name "$VARIANT" \
  "dapt_ckpt=$CKPT" "paths.sequence_data_dir=$SEQUENCE" \
  "${SAMPLING_OVERRIDES[@]}" "${CACHE_OVERRIDES[@]}" "${ROUTER_STRESS_OVERRIDES[@]}" \
  "hydra.run.dir=$RUN_DIR"
