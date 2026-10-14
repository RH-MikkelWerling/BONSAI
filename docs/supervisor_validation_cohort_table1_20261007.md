# Meeting-priority validation panels, cohort flow and Table 1

## Model-level tuning scoreboard

After the paired prediction comparison has written `tuning_forest_estimates.csv`,
render the matched model-level overview with:

```bash
python -m opera.run.plot_neural_model_scoreboard \
  --estimates "$FOREST/tuning_forest_estimates.csv" \
  --baseline time2vec_plain_shared_bce \
  --output "$FOREST/neural_model_scoreboard.png"
```

Set `FOREST` to the **plain-BCE paired-comparison** output directory described
under "Revised main forest" below, not the historical six-model forest whose
baseline is `shared`. The scoreboard refuses a missing baseline rather than
silently relabeling that group-weighted run. The script
also writes a compact `.csv` with outcome-macro, family-balanced, and paired
outcome-macro delta values. Panel dots show individual outcomes or family
means; diamonds show descriptive macro means. **Family-balanced is not the
Lightning 52-endpoint-group selection metric.** The selected-checkpoint tuning
scoreboard has no aggregate uncertainty interval and does not correct
checkpoint/model selection; replicated seeds and a final held-out evaluation
are needed for a confirmatory comparison.

The current neural checkpoints support **tuning-only** figures now. Do not
launch held-out evaluation for the meeting. Run the aggregate comparison first;
the predictability panel below reads validation metrics already logged during
training, so it needs no extra GPU or patient-level export. It uses the exact
approved 69-outcome registry and display groups, not the historical 87-outcome
plot defaults. Wait for a run's `training_complete.json` before calling its
plot final. A preliminary panel from a completed Fourier run can be made now.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
RUN_ROOT="$PWD/data/models/approved_neural_v1_20261006T054836Z"
RELEASE="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
FAMILY="$PWD/data/results/approved_training_families_v1_20261006T052120Z"
RUN=router_additive_e2e
test -s "$RUN_ROOT/$RUN/training_complete.json" || { echo "Run incomplete" >&2; exit 1; }
test -s "$FAMILY/outcome_display_groups_primary_v2.yaml" || { echo "Missing approved display groups" >&2; exit 1; }
python -m opera.run.plot_neural_predictability_prototype --help | grep -q -- '--run' || { echo "Sync the updated plotting file first; server version lacks --run" >&2; exit 1; }
FIGURE="$PWD/data/results/supervisor_${RUN}_predictability_20261007.png"
python -m opera.run.plot_neural_predictability_prototype \
  --run "$RUN_ROOT/$RUN" \
  --registry "$RELEASE/experiment_registry_primary_v2.yaml" \
  --display-groups "$FAMILY/outcome_display_groups_primary_v2.yaml" \
  --output "$FIGURE"

# Optional design comparison: retain all points and add one descriptive
# OLS line per horizon (AUC versus log10 training events) in panel A.
# This writes a separate figure and a sibling *_trend_summary.csv.
python -m opera.run.plot_neural_predictability_prototype \
  --run "$RUN_ROOT/$RUN" \
  --registry "$RELEASE/experiment_registry_primary_v2.yaml" \
  --display-groups "$FAMILY/outcome_display_groups_primary_v2.yaml" \
  --horizon-trends \
  --output "${FIGURE%.png}_horizon_trends.png"
```

Once the resumed runs complete, use the same commands with
`RUN=time2vec_router_interacting_e2e` for the most relevant second panel and
run the aggregate multi-model comparator in
`docs/supervisor_preliminary_findings_20261007.md`. Its reader now combines
Lightning `version_0` and `version_1` logs after checkpoint resume and reports
a common-epoch contrast. The plot shows outcome-level CIF-AUC versus training
event support and clinical-group summaries at the run's selected tuning epoch.
It also writes a sibling `*_plot_data.csv` with outcome, horizon, display group,
CIF-AUC, and training/tuning event counts for aggregate local design iterations.
Its group bars are endpoint spread, **not confidence intervals**. For the
meeting, show the architecture contrast table, one completed-run predictability
panel, and the source-type ablation schematic; avoid a many-model figure wall.

The same command now also writes `supervisor_${RUN}_predictability_20261007_outcome_atlas.png`:
all 69 outcomes by all six approved horizons, grouped clinically, with tuning
CIF-AUC values and 365-day tuning event counts. Grey cells mean that the
selected checkpoint did not log an available AUC; they are **not zero**. The
`*_plot_data.csv` beside the figures retains those unavailable cells and can
be used for local design iteration after the aggregate-output review.

If calibration or paired patient-level comparisons are needed, run
`opera.run.evaluate_direct_competing_risk` only for the final two or three
shortlisted checkpoints, explicitly with `--split tuning` and each run's
`hydra/config.yaml`. It writes internal patient-level predictions; keep those
on the secure server. The log-based panel above is faster and sufficient for a
preliminary discussion.

```bash
# Only after that RUN has finished, on a free GPU server:
RUN=router_additive_e2e
export BONSAI_COHORT_MEMBERSHIP="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"
CUDA_VISIBLE_DEVICES=0 python -m opera.run.evaluate_direct_competing_risk \
  --checkpoint "$RUN_ROOT/$RUN/best.ckpt" \
  --config-dir "$RUN_ROOT/$RUN/hydra" --config-name config \
  --split tuning --device cuda \
  --output-dir "$PWD/data/results/supervisor_${RUN}_tuning_eval_20261007"
```

This is optional before the meeting; it is more expensive and its output
includes patient-level files. Never omit `--split tuning` because the CLI's
default is held-out.

## Paired tuning-bootstrap forest for key outcomes

Yes: evaluate fixed completed checkpoints on **tuning**, then resample the
same patients across models. This provides conditional 95% patient-bootstrap
intervals for each model's 365-day CIF-AUC and for paired differences. It does
not correct checkpoint/model selection on tuning and is not a held-out estimate.
Keep prediction Parquets internal; only review disclosure-safe aggregates for
presentation. Run one evaluation per model, preferably on separate free GPU
servers, after each model has `training_complete.json`. Use a fresh `EVAL_ROOT`.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
RUN_ROOT="$PWD/data/models/approved_neural_v1_20261006T054836Z"
RELEASE="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
export BONSAI_COHORT_MEMBERSHIP="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"
EVAL_ROOT="$PWD/data/results/supervisor_neural_tuning_predictions_20261007"
RUN=time2vec_router_interacting_e2e  # repeat with the other shortlisted runs
test -s "$RUN_ROOT/$RUN/training_complete.json" || { echo "Run incomplete" >&2; exit 1; }
CUDA_VISIBLE_DEVICES=0 python -m opera.run.evaluate_direct_competing_risk \
  --checkpoint "$RUN_ROOT/$RUN/best.ckpt" \
  --config-dir "$RUN_ROOT/$RUN/hydra" --config-name config \
  --split tuning --horizons 365 --device cuda \
  --output-dir "$EVAL_ROOT/$RUN"
```

For the matched 2-by-3 encoder/readout forest, evaluate `shared`,
`family_trunks`, `router_additive_e2e`, `time2vec_grouped_shared_bce`,
`time2vec_family_trunks`, and `time2vec_router_additive_e2e`, with each run's
own `hydra/config.yaml`. Leave the near-identical interactive router out of
this primary forest pending its implementation audit. Then run the CPU-only
comparison in a new terminal. The outcomes below are illustrative
clinical anchors, not a performance-picked list; adjust them before looking
at intervals and keep only outcomes with adequate primary-event/control support.

```bash
EVAL_ROOT="$PWD/data/results/supervisor_neural_tuning_factorial_20261006_v1"
BOOT="$PWD/data/results/supervisor_neural_paired_tuning_factorial_20261006_v1"
FAMILY_MAP="$PWD/data/models/approved_neural_v1_20261006T054836Z/shared/hydra/config.yaml"
test -s "$FAMILY_MAP" || { echo "Missing resolved family config: $FAMILY_MAP" >&2; exit 1; }
python -m opera.run.compare_approved_neural_tuning_predictions \
  --run "shared=$EVAL_ROOT/shared/direct_cr_predictions.parquet" \
  --run "family_trunks=$EVAL_ROOT/family_trunks/direct_cr_predictions.parquet" \
  --run "router_additive_e2e=$EVAL_ROOT/router_additive_e2e/direct_cr_predictions.parquet" \
  --run "time2vec_grouped_shared_bce=$EVAL_ROOT/time2vec_grouped_shared_bce/direct_cr_predictions.parquet" \
  --run "time2vec_family_trunks=$EVAL_ROOT/time2vec_family_trunks/direct_cr_predictions.parquet" \
  --run "time2vec_router_additive_e2e=$EVAL_ROOT/time2vec_router_additive_e2e/direct_cr_predictions.parquet" \
  --baseline shared \
  --outcome overall_survival --outcome treatment_failure \
  --outcome sepsis --outcome neutropenia_g3plus \
  --outcome aki_first_documented \
  --family-map "$FAMILY_MAP" \
  --horizon 365 --n-bootstrap 400 --output-dir "$BOOT"
python -m opera.run.plot_outcome_forest \
  --estimates "$BOOT/tuning_forest_estimates.csv" \
  --contrasts "$BOOT/tuning_forest_contrasts.csv" \
  --horizon 365 --cohort pooled --evaluation-label tuning \
  --interval-label '95% paired patient-bootstrap interval' \
  --metric-label '365-day CIF-AUC' --max-outcomes-per-page 12 \
  --title 'Selected outcomes; tuning only, fixed selected checkpoints' \
  --output-dir "$BOOT/forest"
```

The bootstrap CSVs already include `outcome_family`; do not pass the full
resolved Hydra config to the plotter as `--family-map` on older server copies,
because resolving unrelated config interpolations can fail. The plotter now
reads YAML maps without resolving those fields, but no map is needed here.

For the full approved-panel forest on **one tall page**, sync the updated
`opera/run/plot_outcome_forest.py`, `opera/visualization/outcome_forest.py`, and
`opera/visualization/style.py`, then rerun only the plotter against the existing
bootstrap CSVs with `--all-outcomes-one-page` and a fresh output directory.
This adds equal-outcome-weighted family mean points to each shaded family row
and writes `forest_family_macro_summary.csv`. Family rows have **no** invented
confidence interval; endpoint points retain their paired-bootstrap intervals.
The same model color and marker are used in both absolute and contrast panels.

```bash
python -m opera.run.plot_outcome_forest \
  --estimates "$BOOT/tuning_forest_estimates.csv" \
  --contrasts "$BOOT/tuning_forest_contrasts.csv" \
  --horizon 365 --cohort pooled --evaluation-label tuning \
  --interval-label '95% paired patient-bootstrap interval' \
  --metric-label '365-day CIF-AUC' --all-outcomes-one-page \
  --title 'Approved outcomes; tuning only, fixed checkpoints' \
  --output-dir "$BOOT/forest_all_one_page_v2"
```

### Revised main forest: plain multi-horizon BCE is the reference

The historical six-model forest uses `shared` as its baseline. That run uses
endpoint-group macro weighting and must **not** be relabeled plain BCE. For
the main Time2Vec architecture forest, use `time2vec_plain_shared_bce` as the
actual comparator, with grouped shared, family trunks and additive router as
the matched candidates. The newly cleaned interactive router can be added
only after its fresh run completes and its checkpoint passes the interaction
tensor/code-provenance audit. These are tuning-only contrasts.

After syncing `compare_approved_neural_tuning_predictions.py`,
`plot_outcome_forest.py`, `visualization/outcome_forest.py`, and
`visualization/style.py`, obtain tuning prediction Parquets for the four
Time2Vec runs. Reuse existing predictions only if their checkpoint, split,
horizon and patient keys match; otherwise evaluate the fixed `best.ckpt` on a
free GPU with `evaluate_direct_competing_risk --split tuning --horizons 365`.
For a fresh self-contained evaluation root, run this **after** a GPU becomes
free (not while the router training jobs occupy it):

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
RELEASE="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
EVAL_ROOT="$PWD/data/results/supervisor_neural_tuning_plain_bce_20261007_v1"
RUN_ROOT="$PWD/data/models/approved_neural_v1_20261006T054836Z"
export BONSAI_COHORT_MEMBERSHIP="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"
mkdir -p "$EVAL_ROOT"
for RUN in time2vec_plain_shared_bce time2vec_grouped_shared_bce time2vec_family_trunks time2vec_router_additive_e2e; do
  test -s "$RUN_ROOT/$RUN/best.ckpt" || { echo "Missing checkpoint: $RUN" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES=0 python -m opera.run.evaluate_direct_competing_risk \
    --checkpoint "$RUN_ROOT/$RUN/best.ckpt" \
    --config-dir "$RUN_ROOT/$RUN/hydra" --config-name config \
    --split tuning --horizons 365 --device cuda \
    --output-dir "$EVAL_ROOT/$RUN"
done
```

The comparator then takes the approved registry directly—no shell array of
69 `--outcome` flags:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
RELEASE="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
FAMILY="$PWD/data/results/approved_training_families_v1_20261006T052120Z"
EVAL_ROOT="$PWD/data/results/supervisor_neural_tuning_plain_bce_20261007_v1"
BOOT="$PWD/data/results/supervisor_neural_paired_tuning_plain_bce_20261007_v1"
for RUN in time2vec_plain_shared_bce time2vec_grouped_shared_bce time2vec_family_trunks time2vec_router_additive_e2e; do
  test -s "$EVAL_ROOT/$RUN/direct_cr_predictions.parquet" || { echo "Missing tuning predictions: $RUN" >&2; exit 1; }
done
python -m opera.run.compare_approved_neural_tuning_predictions \
  --run "time2vec_plain_shared_bce=$EVAL_ROOT/time2vec_plain_shared_bce/direct_cr_predictions.parquet" \
  --run "time2vec_grouped_shared_bce=$EVAL_ROOT/time2vec_grouped_shared_bce/direct_cr_predictions.parquet" \
  --run "time2vec_family_trunks=$EVAL_ROOT/time2vec_family_trunks/direct_cr_predictions.parquet" \
  --run "time2vec_router_additive_e2e=$EVAL_ROOT/time2vec_router_additive_e2e/direct_cr_predictions.parquet" \
  --baseline time2vec_plain_shared_bce \
  --registry "$RELEASE/experiment_registry_primary_v2.yaml" \
  --family-map "$FAMILY/outcome_training_families_primary_v1.yaml" \
  --horizon 365 --n-bootstrap 400 --skip-under-supported \
  --output-dir "$BOOT"
python -m opera.run.plot_outcome_forest \
  --estimates "$BOOT/tuning_forest_estimates.csv" \
  --contrasts "$BOOT/tuning_forest_contrasts.csv" \
  --model time2vec_plain_shared_bce \
  --model time2vec_grouped_shared_bce \
  --model time2vec_family_trunks \
  --model time2vec_router_additive_e2e \
  --reference-model time2vec_plain_shared_bce \
  --horizon 365 --cohort pooled --evaluation-label tuning \
  --interval-label '95% paired patient-bootstrap interval' \
  --metric-label '365-day CIF-AUC' --all-outcomes-one-page \
  --title 'Approved outcomes; tuning only, fixed checkpoints' \
  --output-dir "$BOOT/forest_plain_bce_v1"
```

The redesigned figure uses the predictability-map typography, a wider outcome
label field, a compact aligned legend, a separate `n / events` column, and a
contrast title naming the true baseline. Keep a focused subset for slides and
the 69-outcome page as a zoomable PDF/supplement. Check the comparator manifest
for support exclusions and valid bootstrap counts before citing it.

The comparator refuses mismatched patient/outcome/cohort/time/event rows and
under-supported outcomes rather than silently comparing different risk sets.
Its forest points and differences share the same patient resamples; the
censoring KM is re-estimated in each draw. Read `tuning_forest_manifest_INTERNAL.json`
and check valid-bootstrap counts before using the figure. The original forest
renderer still defaults to a held-out caption for existing workflows; the
explicit `--evaluation-label tuning` is required here.

## Directed router borrowing map

The ordinary predictability panel works for router and non-router checkpoints
identically: it plots their tuning CIF-AUC metrics. Only a router can also
export route weights. Its route for one eligible patient–target pair has one
shared weight, one weight per fine-disease adapter and one per outcome adapter.
With 24 fine diseases and 69 outcomes, this is **94 weights** per pair. Routing
is target-specific and patient-dependent, but the same route feeds all six
hazards/horizons; the heatmap has no extra horizon axis.

For a completed router run on a free GPU server:

```bash
RUN=time2vec_router_interacting_e2e
EVAL="$PWD/data/results/supervisor_${RUN}_route_eval_20261007"
CUDA_VISIBLE_DEVICES=0 python -m opera.run.evaluate_direct_competing_risk \
  --checkpoint "$RUN_ROOT/$RUN/best.ckpt" \
  --config-dir "$RUN_ROOT/$RUN/hydra" --config-name config \
  --split tuning --device cuda --export-routes --output-dir "$EVAL"
python -m opera.run.plot_router_borrowing \
  --routes "$EVAL/source_route_weights.parquet" \
  --config "$RUN_ROOT/$RUN/hydra/config.yaml" \
  --output-dir "$PWD/data/results/supervisor_${RUN}_borrowing_20261007"
```

The overview is two directed heatmaps: *target fine disease → source disease
adapter* (24×24) and *target outcome family → source outcome-adapter family*
(12×12). A full target-outcome → source-outcome 69×69 heatmap and CSV are
supplementary. Rows borrow from columns, so the matrices need not be symmetric.
The home disease and home outcome diagonal is omitted: the model reserves
home-source mass by design, and that is not cross-task borrowing. Output also
includes shared/own/borrowed bank mass by target outcome and support counts.
Small rows are masked at fewer than 20 tuning patient–target records. These
are **mean gating weights, not estimated performance contributions**; source
support shrinkage and differing eligibility can affect them. Patient-level
route Parquet and predictions remain in the internal evaluation directory.
The revised overview adds a route-budget panel and gives both heatmaps one
per-source-adapter color scale. It averages individual source-outcome weights
within each family for visual comparison; `family_to_family_mean.csv` still
contains the summed family mass. Consult the budget panel before concluding
that one source type dominates, and remember that gates are not effects.
The full target-outcome by source-outcome map is
`router_borrowing_outcome_full.png`; the new target-outcome by disease-source
map is `router_borrowing_outcome_to_disease.png`. These retain individual
outcome targets, unlike the family overview. The latter averages over home
disease and does not by itself estimate an outcome-by-home-disease interaction.

## Router movement version of the five-panel embedding figure

This is a separate, opt-in version of the original five-panel censoring-aware
plot. It replaces only panel B (time to observed status) with sparse local-mean
arrows from the shared pooled patient vector to the selected outcome's actual
post-router vector. Panels A/C/D/E remain status, IPCW kernel risk, calibration
residual, and effective support. The same tuning patients and same 2D PCA axes
are used in every panel. The router vectors and predictions are internal patient
data; do not move the NPZ or Parquet off the secure server.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
RUN_ROOT="$PWD/data/models/approved_neural_v1_20261006T054836Z"
RELEASE="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
export BONSAI_COHORT_MEMBERSHIP="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"
RUN=time2vec_router_interacting_e2e
TARGET=overall_survival
EVAL="$PWD/data/results/supervisor_${RUN}_${TARGET}_router_map_eval_20261007"
FIGURE="$PWD/data/results/supervisor_${RUN}_${TARGET}_router_map_20261007.png"
CUDA_VISIBLE_DEVICES=0 python -m opera.run.evaluate_direct_competing_risk \
  --checkpoint "$RUN_ROOT/$RUN/best.ckpt" \
  --config-dir "$RUN_ROOT/$RUN/hydra" --config-name config \
  --split tuning --device cuda --horizons 365 \
  --export-routed-outcome "$TARGET" --output-dir "$EVAL"
python -m opera.run.plot_censoring_aware_embeddings \
  --input "$EVAL/direct_cr_predictions.parquet" \
  --router-vectors "$EVAL/routed_patient_vectors_INTERNAL.npz" \
  --target-outcome "$TARGET" --time-col time_days --event-col event \
  --predicted-cif-col cif_365d --horizon-days 365 \
  --title "$TARGET, tuning patients" --output "$FIGURE"
# Matched original panel B, with identical patients and fitted PCA axes:
python -m opera.run.plot_censoring_aware_embeddings \
  --input "$EVAL/direct_cr_predictions.parquet" \
  --router-vectors "$EVAL/routed_patient_vectors_INTERNAL.npz" \
  --target-outcome "$TARGET" --router-panel time \
  --time-col time_days --event-col event \
  --predicted-cif-col cif_365d --horizon-days 365 \
  --title "$TARGET, tuning patients" \
  --output "$PWD/data/results/supervisor_${RUN}_${TARGET}_original_map_20261007.png"
```

Choose a supported target outcome name from the approved config. The PCA is fit
once to the shared tuning patient vectors; post-router vectors are projected
onto those **same axes**. Arrow direction is therefore meaningful within this
figure, but arrow length is magnified solely for legibility and the magnification
is printed on panel B and in the adjacent JSON. The JSON also reports median
unmagnified shift in 2D and in the full hidden space. Local arrows are shown
only for cells with at least 20 patients. Neither these PCA shifts nor the
kernel surface are causal effects. Do not compare absolute PCA positions across
different models, whose hidden spaces are not aligned.

For a more legible **exploratory** companion, rerun only the CPU plot command
with `--router-projection umap --router-arrow-bins 9
--router-arrow-min-count 12` and a new `--output` path. This fits UMAP once to
the shared vectors and transforms the adapted vectors through that same map;
the arrows describe displacement in a nonlinear layout, not hidden-space
effect sizes. Keep the PCA panel as the quantitative displacement reference.
Either plot command can add `--plot-data-output PATH_INTERNAL.npz` to save only
gridded risk, support, calibration, and local-mean arrow fields for design
iteration. The raw per-patient vectors and predictions stay in the secure
evaluation directory. Treat even aggregate NPZ/CSV exports as internal until
their release status has been reviewed.

### Denser movement check (no new checkpoint evaluation)

Once the target's `$EVAL` directory above exists, run the following on a CPU
login node. It uses more, smaller local arrows and suppresses cells with poor
agreement of individual shift directions. The per-cell coherence is
`|mean shift| / mean |shift|`, so a high value means the arrows point broadly
together; it is not a statistical confidence interval.

```bash
RUN=time2vec_router_interacting_e2e
TARGET=sepsis
EVAL="$PWD/data/results/supervisor_${RUN}_${TARGET}_router_map_eval_20261007"
test -s "$EVAL/routed_patient_vectors_INTERNAL.npz" || { echo "Evaluate target first" >&2; exit 1; }
for PROJECTION in pca umap; do
  python -m opera.run.plot_censoring_aware_embeddings \
    --input "$EVAL/direct_cr_predictions.parquet" \
    --router-vectors "$EVAL/routed_patient_vectors_INTERNAL.npz" \
    --target-outcome "$TARGET" --time-col time_days --event-col event \
    --predicted-cif-col cif_365d --horizon-days 365 \
    --router-projection "$PROJECTION" --router-arrow-bins 10 \
    --router-arrow-min-count 12 --router-arrow-style compact \
    --router-arrow-min-coherence 0.30 \
    --plot-data-output "$EVAL/${TARGET}_dense_${PROJECTION}_plot_data_INTERNAL.npz" \
    --title "$TARGET, tuning patients" \
    --output "$PWD/data/results/supervisor_${RUN}_${TARGET}_dense_${PROJECTION}_20261007.png"
done
```

Repeat with `TARGET=overall_survival` after its own evaluation exists. Compare
the adjacent JSON files: number of arrows, median local coherence, 2D shift,
full-vector shift, and (for PCA) explained variance. The projection and arrow
scale are fitted separately for each outcome, so cross-outcome apparent arrow
lengths are **not** directly comparable. PCA gives a linear projection of the
actual vector change; UMAP can reveal local structure but its arrow directions
and lengths are layout-dependent. The shared input vectors for matched patients
can be identical even while target-specific adapted vectors differ, so similar
looking arrows alone do not establish absence of disease borrowing.

## Cohort flow and Table 1 are not yet final

Current release reconciliation (server manifest and input inventory, 2026-10-07):
37,557 sealed first-line patients; 37,531 in the released overall-survival
population, split into train 29,105, tuning 1,980, and held-out 6,446. The
remaining 26 sealed patients have neither a survival-file row nor a split.
Table 1 is scoped to the 37,531 supervised patients. Only its patient-count
row has been refreshed; all other numeric rows are historical layout values.
The precise clinical reason for the 26 exclusions must not be asserted without
a source-level audit.

The two inspected files omit sex and birth date, but the secure-server
`load_dataset("patient")` table supplies `patientid`, `sex` (M/F), and
`date_birth`. After syncing `opera/R/export_table1_patient_demographics.R`,
extract only the released supervised subjects and add that one-row-per-patient
file to the Table 1 join. The extract refuses missing patient matches,
duplicate IDs, unexpected sex codes, and overwrites. Keep it on the server.

```bash
Rscript opera/R/export_table1_patient_demographics.R \
  "$RELEASE/outcomes/overall_survival.parquet" \
  "$PWD/data/results/table1_patient_demographics_20261007_INTERNAL.parquet"
```

Then produce an on-server inventory draft. It is **not** a complete Table 1
because performance status and resolved disease-specific risk category still
need audited quality-registry metadata.

```bash
python -m opera.run.export_table1 \
  --population "$META/population_metadata.parquet" \
  --metadata "$PWD/data/results/table1_patient_demographics_20261007_INTERNAL.parquet" \
  --overall-survival "$RELEASE/outcomes/overall_survival.parquet" \
  --restrict-to-overall-survival \
  --output-dir "$PWD/data/results/table1_supervised_demographics_20261007"
```

The current sealed metadata manifest reports **37,557 first-line patients**.
The diagram still needs current upstream pass/exclusion counts and separate pretraining
denominators. The older V4/V4 flow numbers (ending at 37,540) were removed
from the manuscript diagram because they cannot be spliced into the newer
CLL-V4/MM-V5 release. Outcome-specific risk sets are a separate downstream branch.

The manuscript Table 1 is a **historical 37,467-person layout draft except for
its refreshed 37,531-person patient-count row**, not an updated characteristics
result. The input inventory was inspected with:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
META="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1"
RELEASE="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
python -m json.tool "$META/manifest_INTERNAL.json"
python -m opera.run.export_table1 \
  --population "$META/population_metadata.parquet" \
  --overall-survival "$RELEASE/outcomes/overall_survival.parquet" \
  --output-dir "$PWD/data/results/table1_inspect_20261007" --inspect-inputs
```

The agreed Table 1 denominator is the released supervised survival/split
population. It can be exported with `--restrict-to-overall-survival`; the exporter records the
excluded IDs/count and refuses missing split labels. Use a **new** output
directory and review `release_review/` only after its institutional disclosure
check; `LOCAL_ONLY_internal/` stays on the server.

Performance status and prognostic risk categories come from disease-specific
quality registries, not complete population-wide EHR fields. The current
metadata builder preserves multiple IPI records as JSON rather than choosing
one row, and the Table 1 exporter does not automatically turn those JSON
records into a single risk category. Check `metadata_inventory.csv` (or the
`--inspect-inputs` columns) to see whether exact PS/risk fields already exist
as one-row-per-patient columns. If not, construct a disease-aware, explicit
quality-registry summary and pass it as `--metadata`; do **not** pick the first
of 207 multi-IPI patients or combine different scoring systems as though their
categories were equivalent. Show observed denominators and missingness by
temporal split, ideally also diagnosis year and disease, so the fall in later
coverage is visible. Describe it as changing registry availability only after
checking year-specific source coverage and join success; do not treat missing
scores as low risk or silently carry them forward.

## Recalculate pre-index history and care-setting rows

The current Table 1 medication/diagnosis/laboratory/history values are historical.
Use the existing `--meds-dir` exporter with the **event-level MEDS** root,
after checking its code namespaces with `opera.run.inspect_table1_meds`.
Its train/tuning folders are physical pretraining shards; Table 1 split labels
come from the sealed supervised metadata, not the MEDS directory names.
`opera/configs/table1_meds_prefixes.json` is for `MEDICATION//`,
`DIAGNOSIS//`, and `LABTEST//`; do not assume another MEDS version has identical
namespaces or source-event multiplicity. These are distinct complete *codes*,
not necessarily distinct active substances. All events are strictly before
index. Use a new output directory when rerunning the exporter.

The following contact-registry extraction was an earlier alternative, **not**
the preferred Table 1 route. Use the raw MEDS event stream for specialty and
region as well as medication and diagnosis counts after inspecting its exact
code prefixes; do not mix source definitions in the final table. Retain the
command below only for a possible concordance audit against MEDS:

```bash
Rscript opera/R/export_table1_contact_context.R \
  "$META/population_metadata.parquet" \
  "$RELEASE/outcomes/overall_survival.parquet" \
  "$PWD/data/results/table1_contact_context_20261007_INTERNAL.parquet" 1095
```

This counts unique nonmissing `hovedspeciale_ans` values and selects the modal
nonmissing `region_ans` from `SDS_kontakter` in the 1,095 days strictly before
first-line index. A region tie is unresolved and flagged; no eligible contacts
or no recorded specialty yield missing values, not zero. Review the printed
region-status counts before displaying the region breakdown. The contact
extract can be added to `export_table1` with another `--metadata` argument.
The contact-based region is **not** necessarily the treating hospital's region;
label it as predominant recorded contact region and consider keeping this
care-setting context in the supplement if attribution or missingness is weak.
