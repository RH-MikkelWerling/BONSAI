# CLL specialist support roster and two timing pilots

The completed CLL × overall-survival five-epoch timing run is one data-rich
cell, not a performance comparison. Before launching other CLL specialists,
select one medium-support and one scarce-but-estimable outcome using only
training and tuning labels. Keep non-estimable cells in the roster with a
reason; do not silently drop them from the scientific comparison.

After syncing the updated `audit_cohort_outcome_support.py` to the secure
server, run:

The approved training YAML, not the older 87-outcome registry, defines the
outcome roster and cohort-specific exclusions. The registry supplies only
the fine-to-grouped disease mapping and legacy fallback settings. This matters
because the approved release contains new endpoints absent from that registry.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
export BONSAI_COHORT_MEMBERSHIP="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet"
export BONSAI_OUTCOMES_DIR="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z/outcomes"

CANDIDATES="$PWD/data/results/approved_neural_candidates_router_clean_20261007_v1"
SUPPORT="$PWD/data/results/cll_specialist_support_20261010_INTERNAL"
python -m opera.run.audit_cohort_outcome_support \
  --outcomes-config "$CANDIDATES/time2vec_plain_shared_bce.yaml" \
  --fine-cohort CLL --horizon-days 365 --train-tuning-only \
  --output-dir "$SUPPORT"
```

The roster is `cohort_outcome_support.csv`. Verify that
`support_summary.json` says `held_out_loaded: false` and that the CSV has no
held-out columns. Select a medium-support cell around the median of eligible
CLL training-event counts and a scarce-but-estimable cell near the low end,
using training events, tuning events and IPCW cases/controls together. Do not
select from validation AUC or held-out support. Review all six horizons before
calling an outcome estimable across all six; this 365-day roster is a pilot
selection screen, not a final all-horizon eligibility declaration.

For each selected outcome, use a fresh `PILOT` and `RUN` path:

```bash
MODEL_ROOT="$PWD/data/models/daly_care_t100_joined_bins_with_sor_final_v1"
DAPT="$MODEL_ROOT/pretrain_harmonized_scaled_time2vec_len8192_20ep/final_b4_20261003/version_0/best.ckpt"
SEQUENCE="$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_with_sor_final_v1"
CACHE="$PWD/data/results/outcome_label_cache_approved_v2_INTERNAL"
PILOT_DIR="$PWD/data/results/independent_specialist_pilot_configs_20261010"
OUTCOME="REPLACE_WITH_SELECTED_OUTCOME"
PILOT="$PILOT_DIR/cll_${OUTCOME}.yaml"
RUN="$PWD/data/models/independent_specialist_timing_20261010/cll_${OUTCOME}"

test -s "$CANDIDATES/time2vec_plain_shared_bce.yaml" || exit 1
test -s "$DAPT" || exit 1
test -s "$SEQUENCE/vocabulary.pt" || exit 1
test ! -e "$PILOT" || { echo "Pilot config already exists" >&2; exit 1; }
test ! -e "$RUN" || { echo "Pilot run already exists" >&2; exit 1; }

python -m opera.run.prepare_independent_specialist_pilot \
  --base-config "$CANDIDATES/time2vec_plain_shared_bce.yaml" \
  --parent-cohort CLL_SLL --fine-cohort CLL --outcome "$OUTCOME" \
  --max-epochs 5 --label-cache-dir "$CACHE" --output "$PILOT"
python -m opera.run.warm_outcome_label_cache \
  --config "$PILOT" --cache-dir "$CACHE" \
  --output-json "$PILOT_DIR/cll_${OUTCOME}_cache_report_INTERNAL.json"
python -m opera.run.preflight_direct_competing_risk \
  --config-dir "$PILOT_DIR" --config-name "cll_${OUTCOME}" \
  --require-existing-paths --override "dapt_ckpt=$DAPT" \
  --override "paths.sequence_data_dir=$SEQUENCE"
time python -u -m opera.run.contrastive_multicohort \
  --config-path "$PILOT_DIR" --config-name "cll_${OUTCOME}" \
  "dapt_ckpt=$DAPT" "paths.sequence_data_dir=$SEQUENCE" \
  "hydra.run.dir=$RUN"
```

Record startup, per-epoch wall time, peak GPU memory, and train/tuning support
for all three pilot cells. Project the full CLL queue from the slowest relevant
cell, then schedule only the prespecified estimable roster. The five-epoch
pilots are for compute planning, not model selection or paper AUC claims.
