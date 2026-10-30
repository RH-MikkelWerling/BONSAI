# Clinical and recent-record updates: implementation and CPU refresh

30 October 2026. The new modes are implemented and locally tested. The active
server processes retain the code/recipes they loaded. No test data were opened,
no predictive model was trained, and no server output was edited locally.

## Recent records

The comparison now accepts `--horizons 30 60 90 180 365 730`. One multinomial
patient-bootstrap draw is shared across horizons, restrictions and outcomes.
Censoring is re-estimated within each draw as before. Batched count-weighted
KM/rank calculations replace expanded duplicate rows and repeated sklearn
calls. Baseline AUC work is reused across the four restrictions.

At each horizon it reports every outcome, the overall equal-outcome mean,
and the clinical display families in `outcome_display_groups_paper_v1.yaml`.
These are the clinical domains used in the figures, not the 52 loss-weighting
families. Diseases are pooled within each outcome. These means therefore
have a different weighting from the headline disease-by-outcome cell average.
Unmapped endpoints, if present, receive an explicit Unmapped family.

The point-estimable outcome roster is fixed within each horizon's macro
bootstrap. Undefined member AUCs make that draw's macro mean undefined rather
than silently changing the outcome mix. Intervals require at least 80% valid
draws. Requested/evaluable outcome counts and valid draw counts are exported.
These may vary across horizons and should be audited.

Outputs include an overview across all horizons and three family pages,
two horizons per page. Family pages use the same axis limits across horizons.
Means and intervals are from the paired patient resamples, not averages of
individual confidence limits. Layouts retain the original paper style.

Full multi-horizon tables use `*_by_horizon.csv`. Legacy filenames retain
only the one-year rows when that horizon exists. This prevents the existing
one-year attribution plots and paper-number code from averaging horizons
together. The review assembler now collects the multi-horizon pages/tables.

Future full completions request all six horizons. Existing completed one-year
stages are not silently invalidated. The separate refresh below extends them
from already completed prediction exports, without another GPU evaluation.

In a local synthetic timing check with 2,000 patients per outcome, two
outcomes, four restrictions and 100 draws, the vectorised comparison took
0.107 seconds versus 2.252 seconds for the previous expanded-row comparison.
This is about 21-fold in that example, not a promised server runtime.

## Clinical comparison

Use the explicit mode `--grouping-mode cohort_size_matched` when refreshing.
The default `current` mode preserves the existing recipe until a protocol
choice is made. The new mode does the following:

1. Allocates the evaluation cohort to model strata using predictions and
   baseline native index category counts only. It preserves ties and empty
   native categories. The closest attainable allocation is reported when
   exact counts are impossible. This is a cohort-based rank benchmark, not
   a fixed numeric individual-classifier threshold.
2. Renders the existing clinical plates with those strata, their native
   category names, and an explicit baseline-only size-matching annotation.
3. Exports paired horizon comparisons in
   `size_matched_survival_comparisons.csv`. The primary comparison bootstrap
   resamples patients jointly and repeats baseline-only rank matching in
   each draw. The refresh recipe requests 1,000 draws. Existing curve bands
   remain conditional on the displayed groups; their uncertainty definition
   differs from the paired rank-rematching differences in this table.
4. Separately fits the largest supported development model Low group whose
   KM survival is at least native Low survival. It uses all follow-up records,
   including early-censored patients, and keeps ties together. It requires
   at least 20 native/model Low patients and 10 still at risk at the horizon.
   These are explicit support settings, not a precision guarantee. It does
   not impose support requirements for arbitrary intermediate model strata.
5. Evaluates that frozen Low cutoff in
   `km_matched_low_groups.csv`, including group sizes, deaths, survival,
   paired differences and intervals, and additional Low patients per 1,000.
   Empty or unsupported groups remain explicit. KM matching applies to
   death-inclusive endpoints, not nonfatal competing-risk outcomes.
6. Fits the high-specificity classifier independently. Its confusion metrics
   are in `classification_at_anchors.csv`; every size-matched stratum boundary
   also has binary metrics in `classification_at_thresholds.csv`. The headline
   detection comparison uses the independent classifier, not the rank-matched
   model High stratum.

For this tuning rehearsal, the existing recipe still supplies **apparent
training predictions** for the two outcome-based development anchors. Their
provenance is retained explicitly. The primary size-matched comparison does
not depend on training-label threshold fitting. The final source for frozen
KM/specificity thresholds remains a protocol decision; using existing tuning
predictions before held-out evaluation is a different, available development
choice. Nothing here makes apparent training matches out-of-sample estimates.

All-cause death uses KM survival. Nonfatal outcomes use cumulative incidence
for the size-matched comparison. Zero-event percentile bootstrap intervals
can be degenerate; report tables flag this limitation. Similar point estimates
or overlapping intervals do not establish equivalence or safety of changing
treatment.

The paper-number builder keeps the two questions separate. Equal-size strata
do not fill the additional-Low-patient slots. Those slots require the separate
KM-matched table. Clinical review packaging includes both new tables and
excludes individual patient metadata/predictions.

## Files to sync

Sync the updated repo files as usual. The direct refresh dependencies changed
in this work are:

- `opera/run/compare_input_restrictions.py`
- `opera/run/refresh_recent_record_gaps.py` (new)
- `opera/run/refresh_clinical_figures.py`
- `opera/run/render_clinical_score_panels.py`
- `opera/run/clinical_figure_provenance.py`
- `opera/evaluation/clinical_risk_groups.py` (new)
- `opera/visualization/paper_figures.py`
- `opera/run/package_figure_review.py`

The completion driver, review assembler, paper-number builder and parallel
audit packager were also updated for future runs and inclusion of the new
aggregate tables. No source-manifest command or freshness opt-in is needed.

## CPU commands using the existing full workspace

Run when CPU cores are available. These commands use new sibling output
directories and leave the running rehearsal's outputs alone. They need no
additional GPU allocation. Change the first path if the live workspace differs.

```bash
opera_full_root=data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123
opera_refresh_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_gap_refresh="$opera_full_root/recent_record_horizons_$opera_refresh_stamp"
opera_clinical_refresh="$opera_full_root/clinical_rank_groups_$opera_refresh_stamp"
mkdir -p logs

nohup env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv-v100/bin/python -u -m opera.run.refresh_recent_record_gaps \
  --workspace "$opera_full_root/analysis" \
  --output-dir "$opera_gap_refresh" \
  --horizons 30 60 90 180 365 730 --draws 1000 \
  > "logs/recent_record_horizons_$opera_refresh_stamp.log" 2>&1 &

nohup env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv-v100/bin/python -u -m opera.run.refresh_clinical_figures \
  --workspace "$opera_full_root/analysis" \
  --output-dir "$opera_clinical_refresh" \
  --grouping-mode cohort_size_matched \
  > "logs/clinical_rank_groups_$opera_refresh_stamp.log" 2>&1 &

echo "Recent-record log: logs/recent_record_horizons_$opera_refresh_stamp.log"
echo "Clinical log: logs/clinical_rank_groups_$opera_refresh_stamp.log"
echo "Recent-record ZIP: $opera_gap_refresh/recent_record_review.zip"
echo "Clinical ZIP: $opera_clinical_refresh/clinical_scores_review.zip"
```

The two ZIPs are the intended aggregate review downloads. Do not zip the
clinical refresh's entire parent directory: it also holds server-side patient
metadata needed for rendering. Clinical plots and tables remain candidates
for empirical review, not an automatic scientific sign-off.

## Local verification

The mathematical regression tests compare the count-weighted calculation
with the original expanded bootstrap, including censoring, tied risks, zero
counts at event times and exact-horizon cases. Multi-horizon tests confirm
that separate and joint calls use identical draws and estimates. Tests also
cover label/split mismatch rejection and preservation of one-year consumers.

Clinical tests exercise the actual native train/tuning adapters, thresholds,
binary metrics, primary plots, paired size comparisons and independent KM
matching end to end. KM prefix selection is checked against exhaustive search.
There are tests for source population preservation, missing support, empty
native strata, tied scores and aggregate-only packaging. The main existing
figure regression suite passed after the plot changes.

Synthetic layouts were visually inspected at
`scratch/paper_reading_20261028/clinical_and_record_gap_preview_20261030/index.html`.
These are explicitly synthetic layout previews, not the new server results.
The automatic layout audit reported no collision candidates or legend overflow
for the four inspected previews. Server counts, missingness and long labels
still require the normal empirical figure review.
