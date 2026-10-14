# Independent specialist timing pilot (not a performance result)

The pilot narrows the approved plain shared-BCE config to one fine disease and
one outcome, predicts all six horizons, and initializes only from the local
Time2Vec DAPT encoder. It does **not** inherit pooled supervised weights.
Five epochs measure data setup and per-epoch runtime; do not use its AUC as a
scientific specialist comparison. Choose one high-support cell first, then
repeat with a small and medium cell after inspecting train/tuning support.

On the secure server, after syncing code:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh

CANDIDATES="$PWD/data/results/approved_neural_candidates_router_clean_20261007_v1"
MODEL_ROOT="$PWD/data/models/daly_care_t100_joined_bins_with_sor_final_v1"
DAPT="$MODEL_ROOT/pretrain_harmonized_scaled_time2vec_len8192_20ep/final_b4_20261003/version_0/best.ckpt"
SEQUENCE="$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_with_sor_final_v1"
CACHE="$PWD/data/results/outcome_label_cache_approved_v2_INTERNAL"
PILOT_DIR="$PWD/data/results/independent_specialist_pilot_configs_20261009"
PILOT="$PILOT_DIR/dlbcl_overall_survival.yaml"
RUN="$PWD/data/models/independent_specialist_timing_20261009/dlbcl_overall_survival"

test -s "$CANDIDATES/time2vec_plain_shared_bce.yaml" || exit 1
test -s "$DAPT" || exit 1
test -s "$SEQUENCE/vocabulary.pt" || exit 1
test ! -e "$PILOT" || { echo "Pilot config already exists" >&2; exit 1; }
test ! -e "$RUN" || { echo "Pilot run already exists" >&2; exit 1; }

python -m opera.run.prepare_independent_specialist_pilot \
  --base-config "$CANDIDATES/time2vec_plain_shared_bce.yaml" \
  --parent-cohort DLBCL_like --fine-cohort DLBCL \
  --outcome overall_survival --max-epochs 5 \
  --label-cache-dir "$CACHE" --output "$PILOT"

# CPU-only: prepares only this specialist's train/tuning labels.
python -m opera.run.warm_outcome_label_cache \
  --config "$PILOT" --cache-dir "$CACHE" \
  --output-json "$PILOT_DIR/dlbcl_overall_survival_cache_report_INTERNAL.json"

python -m opera.run.preflight_direct_competing_risk \
  --config-dir "$PILOT_DIR" --config-name dlbcl_overall_survival \
  --require-existing-paths --override "dapt_ckpt=$DAPT" \
  --override "paths.sequence_data_dir=$SEQUENCE"

time python -u -m opera.run.contrastive_multicohort \
  --config-path "$PILOT_DIR" --config-name dlbcl_overall_survival \
  "dapt_ckpt=$DAPT" "paths.sequence_data_dir=$SEQUENCE" \
  "hydra.run.dir=$RUN"
```

The approved base config still defaults to the older harmonized-v4 sequence
directory. The explicit sequence override is required because the Time2Vec
checkpoint above was trained with the final-v1 vocabulary; a path-existence
preflight alone does not check vocabulary/checkpoint shape compatibility.

Inspect the log's `Multi-cohort data setup seconds`, GPU epoch times,
peak memory and total wallclock. A warm cache does not avoid loading and
deserializing sequence tensors. For a full approved multi-outcome run, the
same cache can be warmed with:

```bash
python -m opera.run.warm_outcome_label_cache \
  --config "$CANDIDATES/time2vec_plain_shared_bce.yaml" \
  --cache-dir "$CACHE" \
  --output-json "$PILOT_DIR/multioutcome_cache_report_INTERNAL.json"
```

Then set `OPERA_OUTCOME_LABEL_CACHE_DIR="$CACHE"` when using the approved
neural launcher. The cache contains patient-level labels: keep it on the
secure server, never sync it into the repository or share its files. Source
path/size/mtime and label rules determine invalidation; the approved input
release should remain immutable. The cache never reads held-out labels.
