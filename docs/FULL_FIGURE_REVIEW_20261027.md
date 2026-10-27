# Complete figure review before the next tuning run

Scope: all 40 rendered figure IDs in the union of the full results recipe,
attribution recipe and five clinical outputs, plus the six static TeX inputs
used in the manuscript. There are 39 rendered IDs in the current manuscript;
`fig_tabular` is an additional registered plate. Every rendered ID has a
synthetic design proof; the previously missing hierarchy proof was added.
The proofs and four contact sheets are in `scratch/full_figure_review/`.
They test layout and schema, not the truth or completeness of server results.

## Cross-figure conclusions and fixes

1. Production replay reproduced the stable router and joint predictions; metric
   recomputation agrees. There is no demonstrated inference bypass or reason to
   retrain solely for plotting.
2. Define the estimand beside each number: patient-pooled AUC, supported
   disease-balanced mean, equal-cell mean, median/IQR and conditional bootstrap
   intervals are different. IQRs are heterogeneity, not confidence intervals.
   Pooled horizon marker 0 is a sentinel, not a prediction horizon.
3. Fixed horizon plots now omit sentinel 0. The headline absolute scatter now
   selects one-year estimates, as its axes state; it previously selected pooled
   estimates by default. The support map describes the one-year training-support
   proxy instead of calling it events by zero days.
4. Source-removal plots describe other-disease replacement with the outcome bank
   retained. They now identify cross-disease dependence, distinguish shrunken
   pairs from unshrunk totals, and provide an optional raw pair view. Captions
   for these and the router-usage/horizons figures were reconciled with code.
5. Bank heatmaps normalize per task before averaging. Mean-gate entropy is
   labelled as such and is not patient-level entropy. Source mass is not the
   amplitude or predictive benefit of a correction.
6. Embedding joins use canonical IDs. Conflicting duplicate predictions and
   mismatched arrow checkpoint/outcome/split inputs are rejected. Smooth fields
   retain footprint/support masks; arrow rows use one labelled magnification.
7. Missing-input rebuilds remove stale derived tables. The pass also caught and
   fixed stale atlas, reliability and selected-strategy companion outputs.
8. Local render preflight now draws the canvas (not just constructs axes), emits
   aggregate warning/error reports, and records renderer/figure/style hashes.
   Saved figure manifests record these hashes and the preview watermark.
   Readiness can use the actual recipe and includes registered extra figures.
   Attribution preflight now honors --validate-render instead of executing a
   normal write when that flag is supplied.

## Remaining gates before interpreting regenerated results

- **Bootstrap:** strict fixed-roster AUC panels have zero valid draws in the
  returned report. Preserve missing intervals; do not replace them with zero.
  The CPU audit's supported/sparse zero-gain counts must guide the reporting
  policy. New diagnostics expose unstable-cell and valid-draw counts without
  silently redefining the estimand.
- **Model identity:** the atlas, headline and several clinical inputs explicitly
  use J, while the selected OPERA alias currently identifies R. These can be
  useful joint-model figures, but cannot be described indiscriminately as the
  selected OPERA model. The CPU audit now reports atlas directory and clinical
  model alignment. Either identify J explicitly in captions or deliberately
  prepare the selected-model inputs and matching plots. No silent substitution.
- **Clinical:** confirm `ipi_computed_group` exists, endpoint/death convention,
  eligible score populations and frozen tuning thresholds. Existing rehearsal
  failure is not a rendering success.
- **Attribution:** finish the failed GPU work and summary checks before publishing
  source shares, concepts or restriction comparisons. Synthetic tracks remain
  explicitly illustrative, separate from real patient data.
- **New fits/external encoder:** completed training is not completed evaluation.
  Include new fits only after checkpoint identity, prediction completion and
  analysis inventory are verified. External panels need all stated arms.
- **Fresh outputs:** the renderer intentionally refuses an existing output
  directory. Archive/move an old figure directory before actual regeneration,
  or use a new output path. Validation does not require this or write figures.

## Figure-by-figure critique and resolution

“Proof passed” below means a local synthetic construction/render was exercised.
It never certifies the real server table. Shared gates above still apply.

| Figure | Critique | Resolution / server acceptance check |
|---|---|---|
| `fig_headline` | Dense five-panel overview; reference ladder is a distribution of cell gains, not pooled-patient AUC. J is explicitly the primary model in this implementation. One-year scatter and pooled sentinel were inconsistent. | Fixed scatter selection and log-horizon filtering. Keep model named J unless deliberately rebuilding for OPERA. Clinical source must match the claimed model; missing scores must not look like no benefit. Proof passed. |
| `fig_atlas` | Patient-pooled outcome AUC versus training events is not a fine-cell performance summary. It can be mistaken for selected OPERA although the documented input is the joint run. | Verify atlas run/split provenance; retain explicit joint attribution or supply a deliberate selected-model version. Stale atlas deletion fixed. Proof passed. |
| `supp_atlas_matrix` | Many labels, grey gaps, and a clipped color range can obscure coverage and model identity. Fine-cell matrix uses J while pooled side panel comes from atlas input. | Require both panels to use the same run and horizon conventions. Read at full supplementary width; grey is undefined, not AUC zero. Proof passed. |
| `fig_pooling` | Cell medians and smoother bands can be mistaken for a population effect or posterior inference. Small-cell scatter is intentionally restricted. | Keep common-cell pairing and scatter eligibility explicit. Distinguish descriptive fallback bootstrap from hierarchical posterior curves; do not interpret IQR as CI. Proof passed. |
| `supp_pooling_cells` | Hundreds of unadjusted interval dots invite overclaiming; order/support bars must match cells. | Caption retains unadjusted exploratory interpretation; preserve unestimable grey cells and train-only ordering. Inspect missing-CI counts before interpreting dots. Proof passed. |
| `fig_tabular` | Registered but not placed separately in current manuscript; duplicates parts of benchmarks. Its horizon line could include pooled marker 0. | Fixed exclusion of marker 0. Keep as supplementary/diagnostic alternative unless manuscript placement is intended. Verify pooled/registry XGBoost references are genuinely available. Proof passed. |
| `fig_benchmarks` | Combines discrimination, calibration and examples; slope 1 and bias 0 have different ideals. OPERA aliases and J must be distinct. | Verify selected alias and matched cells; reliability examples follow declared training-support selection, not visual attractiveness. Treat failed calibration fits as missing. Proof passed. |
| `fig_calibration` | Domain medians hide severe individual miscalibration; slope and bias are different quantities. | Retain cell inventory, uncertainty type and diagonal/ideal references. Companion reliability table is now invalidated when its build fails. Compare selected-model versus J labels. Proof passed. |
| `fig_patient_map` | Attractive colored regions can suggest calibrated risk or causal structure. Some panels are relative observed risk, whereas residual is absolute observed minus predicted. | Explicitly distinguish scales, model and example outcome selection. Fixed ID join and masked smoothing; simulation-informed defaults preserve holes and islands. Proof passed. |
| `fig_external` | National, adapted and cohort encoders, two readouts and sample-efficiency curves can be conflated. | Check every origin/fraction/readout arm and endpoint roster. Do not extrapolate efficiency intersections outside evaluated fractions. External arrival remains a data gate. Proof passed. |
| `fig_hierarchy` | Own correction, borrowed correction and total contrast are different comparisons, not a causal decomposition of AUC. Sparse-support trends are unstable. | Keep design AB/O explicit and use the same references across cells; show IQR honestly. Added its missing full-suite design proof. Proof passed. |
| `supp_routing` | Previously selected wrong metric/horizon in factorial panel; original controls and stable final router are different experiments. | Exact metric/horizon selection fixed; label original controls. Do not infer equivalence from missing confidence bounds. Proof passed. |
| `supp_router_usage` | Shared-slot weight near zero was read as an absent encoder. Averaging before normalization biased within-bank shares. | Fixed normalization and labels; D now mean-gate borrowed-source breadth. Excludes own/shared slots. Does not claim patient-specific specialization. Proof passed. |
| `fig_borrowing_map` | Tiny shrunken pair values beside larger unshrunk totals look like a unit error; positive-share ratios can look substantive despite tiny absolute signal. | Confirmed x100 upstream exactly once; explicit shrinkage and cross-disease labels, optional raw view, duplicate guards. C is descriptive concentration, not correlation or proof of lineage dependence. Proof passed. |
| `supp_borrowing_detail` | Mixes within-model replacement with separately trained controls; tiny source-alone and leave-one-out effects need not sum to a total. | Corrected cross-disease scope and caption references. Mean replacement is primary, zero replacement sensitivity. Keep raw/shrunken view explicit and avoid causal language. Proof passed. |
| `supp_horizons` | Previously omitted router and caption described only three panels. Twelve domains/six horizons make a dense display. | Four-panel design now includes R versus J; caption distinguishes comparators and IQR. Positive horizons only; shared axis avoids visual gain inflation. Proof passed. |
| `supp_coverage` | Grey, not-applicable, no-event and fit-failed cells are scientifically different. | Keep distinct statuses and complete outcome roster; do not remove failures from the denominator. Verify new fits update coverage. Proof passed. |
| `supp_support` | Pooled marker 0 was printed as a real horizon; missing values could be mistaken for known zero events. | Corrected one-year proxy label. Require complete training metadata before interpreting cumulative distribution; grey means no usable positive count. Proof passed. |
| `supp_rescue` | Defining harm and showing gains on tuning is descriptive selection on the same data, not independent validation. | Tuning watermark/text must state that limitation. Held-out version uses tuning-defined strata without reselection. Keep paired cohorts and missing-control status. Proof passed. |
| `supp_strategy` | Per-cell selection must not use test outcomes; candidate models and loss criterion matter. | Confirm tuning-only selection and identical horizon inventories; selected-strategy companion table now removed after missing-input rebuild. Preview is exploratory. Proof passed. |
| `fig_who_benefits` | Pooled subgroup AUC and patient-weighted Brier effects answer different questions; subgroup boundaries could drift between panels. | Shared bin/row definitions and missing-effect handling tested. State estimands and endpoint/horizon composition; use paired uncertainty. Proof passed. |
| `supp_national_where` | National coverage/support/year contrasts may mix different populations or missing arms. | Match national comparisons, preserve subgroup counts, use common labels and explicitly patient-weighted Brier units. External readiness gate remains. Proof passed. |
| `supp_regions` | Schematic tiles can be mistaken for geography; map estimates conceal uncertainty and small counts. | Use supplied real geometry when available; otherwise label schematic. Retain count/interval/hatching information and identical region definitions for models. Proof passed. |
| `supp_embedding_comparison` | Independently fitted columns do not permit directional movement comparisons. Colors and support masks must be consistent. | Shared color norms, fixed fitting protocol and train/tuning-only fit. Keep contours and masked smooth metadata; interpret geometric resemblance descriptively. Proof passed. |
| `supp_calibration_maps` | Previously missing arrows, extrapolated squares and failed ID joins; per-outcome arrow scaling could disguise magnitude differences. | Separate arrow row; matched checkpoint/outcome/split and shared projection required. Common labelled magnification, adaptive grid and support mask. Need real vector exports to enable row. Proof passed. |
| `supp_embedding_gallery_1` | Domain headings can overlap with long outcome names; low-event relative risk is noisy. | Fixed IDs/masks, common relative-risk scale, eligible-patient baseline. Read full-size; do not interpret blank support as no events. Proof passed. |
| `supp_embedding_gallery_2` | Same risks; page boundaries must not duplicate or omit endpoints. | Stable domain/outcome ordering and page/per-page options; inspect roster across all six pages. Proof passed. |
| `supp_embedding_gallery_3` | Same risks; censoring and competing-event controls differ by endpoint. | Use outcome-specific follow-up/eligibility and IPCW estimates on fixed coordinates. Proof passed. |
| `supp_embedding_gallery_4` | Same risks; rare outcomes can show dramatic ratios from little absolute risk. | Preserve common scale and show each outcome's baseline incidence; no interpretation without local event support. Proof passed. |
| `supp_embedding_gallery_5` | Same risks; small eligible subsets reduce visible footprint. | Outcome-specific support masking is intentional, not a rendering failure. Check eligible counts before changing bandwidth. Proof passed. |
| `supp_embedding_gallery_6` | Last page may have fewer than 12 outcomes; forcing a full grid would create empty-looking panels. | Unused slots remain absent; verify all eligible outcomes are represented exactly once. Proof passed. |
| `fig_risk_groups` | Learned cutoffs and IPI comparisons require correct endpoint/death handling and the same score-available population. | Freeze on tuning, retain counts/risk tables, confirm J versus selected OPERA identity. Missing computed IPI is an upstream blocker. Clinical proof passed. |
| `fig_clinical_score` | AP/PR and survival panels have different targets and censoring conventions; operating threshold must match index specificity. | Preserve endpoint convention, threshold provenance and explicit score category rank; do not imply AP is a time-dependent AUC. Clinical proof passed. |
| `supp_clinical_score_age` | Small age-by-risk cells invite unstable estimates; predicted and observed values must not be merged conceptually. | Separate counts/predictions/observed risk with common cells, suppress small estimates and retain uncertainty. Clinical proof passed. |
| `fig_decision_curve` | Net benefit depends strongly on threshold units, censoring and clinical decision interpretation. | Probability threshold versus percentage label and per-1000 scale must agree; retain treat-all/treat-none references and paired comparator bands. No clinical utility claim from tuning alone. Proof passed. |
| `supp_score_overview` | Many index/cohort comparisons become unreadable when shrunk; indices have different recording populations. | Minimum-patient rule, exact index/cohort labels, fixed order and full supplementary-width placement. Recheck page height when actual number of indices is known. Clinical proof passed. |
| `fig_attribution` | IG magnitudes are not causal effects; synthetic patient illustration can be mistaken for empirical evidence. | Separate real aggregate attribution/restriction inputs from labelled synthetic tracks. Require completeness, baseline and faithfulness diagnostics before interpreting. Proof passed. |
| `supp_attribution_atlas` | All outcomes and many concepts are dense; source/time shares need a defined denominator. | Keep shared scales and source/window definitions, identify shrunken quantities, inspect at full PDF size. If actual labels exceed space, use domain subsets via renderer rather than unreadable scaling. GPU-summary gate remains. Proof passed. |
| `supp_attribution_structure` | Similarity depends on normalization and missing outcomes; restriction gains depend on matching checkpoint/patients. | Maintain symmetric outcome matrix and same restriction baseline, state agreement measure, reject failed restrictions. GPU-summary gate remains. Proof passed. |
| `supp_attribution_patient` | A curated illustrative history is not representative, and log-odds contribution is not probability change. | Keep explicit synthetic label and units; no private patient timeline exported. Synthetic illustration is an intentional exception to real aggregate inputs. Proof passed. |

## Static manuscript figures

| TeX input | Critique and acceptance condition |
|---|---|
| `modeling_paradigms` | Conceptual scope diagrams must match the actual specialist/disease/outcome/joint training definitions. They do not establish empirical superiority. |
| `information_axes` | Data-origin and supervision axes must remain distinct; external pretraining is pending evidence, not a plotted result. |
| `supervision_grid` | Same disease/outcome symbols and scope definitions as the main figure; horizons are outputs, not independent patient splits. |
| `model_inputs` | Keep pre-index observation versus prediction/follow-up boundary explicit; do not imply post-treatment tokens enter baseline prediction. |
| `cohort_flow` | Counts need the release-specific server inventory, including the documented exclusions; historical labels are not refreshed by plotting. This is a server count gate. |
| `source_router` | The base path stays active; gates allocate correction terms, own/global paths may be outside the softmax, and disease/outcome banks are distinct. Verify diagram matches selected fixed router configuration before final draft. |

The manuscript compiled in two passes with the revised captions and the current
40 synthetic figure PDFs; bibliography citations still need the normal complete
bibliography build. This does not certify a new clinical count release.

The broad figure/clinical/readiness test pass completed successfully (69 tests),
with further targeted regression tests for the fixes above. An additional real
XGBoost-training integration test cannot run in this local environment because
libomp.dylib is missing; this does not invalidate the plotting tests or indicate
a server training failure. No dependency installation or model retraining was
attempted as part of this review.

## Server sequence after syncing

1. Run the CPU router/cell audit already requested; send its aggregate JSON.
   It now includes model alignment and raw-versus-shrunken borrowing diagnostics.
2. Check completed new fits and prediction coverage; do not retrain for plot fixes.
3. Resolve interval policy and J/OPERA input identity before rebuilding statistics.
4. Rebuild only affected analyses into fresh output directories, with the exact
   inventory/checkpoints recorded. Clinical metadata and attribution failures
   need their producer repaired first.
5. Rebuild embedding/plot tables. Missing tables must stop the complete run.
6. Run readiness and both render preflights below; review warning counts and
   missing inputs. These commands do not write a new figure set.

```bash
bash scripts/run_paper_experiments.sh figure-readiness --env-file .env.rehearsal --execute
bash scripts/run_paper_experiments.sh figures --env-file .env.rehearsal --execute \
  --validate-render --validation-report logs/rehearsal/results_figure_validation.json
bash scripts/run_paper_experiments.sh attribution-figures --env-file .env.rehearsal --execute \
  --validate-render --validation-report logs/rehearsal/attribution_figure_validation.json
```

Do not use available-only mode to certify a complete run. Clinical figures are
produced by clinical-scores, outside these two recipes, and must have successful
clinical manifests. Actual rendering requires fresh result/attribution figure
output directories. Keep prior figures for a before/after review. The final
review must use real regenerated tables; synthetic proofs cannot settle missing
intervals, current patient counts or outcome-specific interpretability.
