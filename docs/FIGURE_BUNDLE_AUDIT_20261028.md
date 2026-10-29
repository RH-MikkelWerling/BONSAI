# Review of the downloaded figures and results

Source: `scratch/paper_reading_20261028/paper_review_download_20261028_182155.zip`.
Extracted into the same-named directory without the ZIP extension. Archive paths
were checked before extraction. All entries listed in the bundle manifest passed
SHA256 verification. There are 69 PNG figures (plus PDF counterparts), inspected
via six contact sheets and enlarged key router figures. This is tuning evidence,
not held-out confirmation. No empirical values in the archive were changed.

## Principal findings

This is a mixed-generation review package, not a complete rerun of all analyses.
The router is not simply the original model relabelled: across all 18,024 finite
aligned metric entries, R_L, R_AB_L and OPERA have identical estimates. The old
R_ORIG_L differs in 17,880 of those entries. Source-removal provenance identifies
R_AB_L, scientific hash `79d39ad1e7cb514720265e271266a3385a713e7c6fbd7f730974da053c059e95`,
with fresh borrowing-inference inputs in the completed local workspace.

That verifies the identity reported by these exports, not the entire raw
checkpoint or all gate/embedding inputs. Original raw predictions, checkpoints,
route export manifests and per-outcome arrow arrays were deliberately not
included. Cross-seed stability cannot be established from this bundle.

## Numerical direction of the results

Equal-cell summaries across supported disease–outcome–horizon combinations from
`aggregates/figure_inputs/plot_tables/panel_effects.csv`:

| Contrast | AUC difference | 95% interval |
|---|---:|---:|
| Disease-pooled vs specialist | +0.120 | +0.107 to +0.133 |
| Outcome-pooled vs specialist | +0.027 | +0.015 to +0.040 |
| Joint vs specialist | +0.119 | +0.104 to +0.133 |
| Primary routed vs joint | +0.009 | +0.003 to +0.015 |
| Primary OPERA vs per-task XGBoost | −0.028 | −0.036 to −0.019 |
| Joint vs per-task XGBoost | −0.037 | −0.045 to −0.028 |
| Disease-pooled XGBoost vs per-task XGBoost | +0.012 | +0.006 to +0.017 |
| Original router vs primary router | +0.001 | −0.002 to +0.004 |

These rows report 3,004 cell–horizon combinations, not 3,004 independent tasks
or patients. Summaries are rounded as exported. The current story is strong
benefit from disease sharing within neural models, a smaller outcome-sharing
gain, modest additional routed benefit, and persistent tabular superiority.
Pooled XGBoost also benefits from sharing. The primary router's predictive cost
relative to the original is small in this summary; the equivalence label relies
on the declared practical margin and is not evidence of training-seed stability.

The source-removal workflow uses different weighting and re-estimated analyses.
Its disease-balanced full-router vs joint estimate is +0.00862 with interval
−0.00149 to +0.01966. Do not substitute that interval for the equal-cell interval
above or call their differing sign-certainty a plotting error.

## Router audit

### Routing benefit and disease-course concern

The negative disease-course result exists in the exported numerical contrasts,
not only the figure. Across its 261 supported cell–horizon combinations, the
unweighted mean R1 gain is −0.01797; at one year it is approximately zero across
64 combinations. Early horizons are substantially more negative. Richter
transformation has mean −0.3855 across 12 combinations and strongly influences
the summary. Other diseases differ, including positive CLL and MCL means.
This is not evidence that routing helps disease course most in this fit.
Excluding Richter diagnostically changes the unweighted domain mean from
−0.01797 to −0.000258 across the remaining 249 combinations. The twelve
Richter entries comprise three nested endpoints at 30, 60, 90 and 180 days,
with 5–7 cases and 6–8 controls per entry. These are repeated related estimates,
not twelve independent replications. Excluding Richter is a useful diagnostic,
not a replacement primary analysis.

Cardiovascular estimates have just 46 supported cell–horizon combinations:
only one at 30 days, two at 60 days and eleven at one year. Apparent trajectories
can change because both the task mix and estimates change. Use fixed common
cells for a horizon diagnostic and show support, without hiding the full result.

### Disease-source removal

Mean-ablation disease-balanced cross-disease dependence is **0.459 AUC points**
(−0.179 to +1.039), equivalent to **0.00459 AUC**. It is not decisively positive.
Disease-source removal retains the outcome bank and shared/global paths, so it
does not measure the total value of all sharing.

The fitted shrinkage pulls most individual source-alone effects extremely close
to zero. The displayed mean-ablation heatmap spans roughly 0.00003 AUC points
in its pooled display, yet uses the full red–blue range. This is an interpretive
problem: colours make negligible posterior means look substantial. Raw effects
are noisier and larger; shrinkage is not proof that the corrections are inactive.

The lineage panel normalises positive shrunken values that are nearly zero.
Its proportions and effective source counts should not carry a biological
interpretation. Remove this panel from the main narrative or explicitly show
absolute support and magnitude before interpreting normalised shares. Display
raw estimates with uncertainty beside any shrunken view; do not turn off
shrinkage merely to produce a more striking map.

### Gate usage and arrows

The gate-usage plot shows broad allocation and similar mean profiles. Its
entropy is calculated from average gates, not a distribution of individual
patient entropies. Broad means can conceal patient-specific selection, so this
plot alone establishes neither collapse nor cross-seed stability. Allocation
between banks also depends on bank size (69 outcome vs 24 disease sources).

**Arrows are present** in `review/main/supp_calibration_maps.png`, row B,
with a common displayed magnification of ×2.2. They are faint and very similar
across outcomes. Whether that similarity is real requires a compact server-side
numerical check of per-outcome routed displacement, not a different arrow style.
The figure has overlapping left-side row labels and too much top whitespace.
It is unsuitable for publication as currently laid out.

The hierarchy plot uses a separately trained global-only/own-only/full-router
comparison, not the fixed-model source-removal contrasts. Its median/IQR
summaries are still explicitly descriptive. A fixed approximately 84-event
quadratic crossing is too prominent without uncertainty. Its actual
`hierarchy_cells.csv` input hash is in the renderer manifest, but the table was
omitted by the packager. That table is now included in future bundles.

## Confirmed stale or missing families

- **Clinical:** the package used the original `local_figures.../clinical_scores`
  folder, not a clinical-only refresh. Its inventory has two cutoffs even for
  NCCN-IPI, tuning-named thresholds, no training threshold provenance, and some
  boundaries separated by only 0.000001. Classification-at-thresholds CSV is
  missing. These are the old coarsened/tuning-derived groups. Do not use their
  low-risk expansion figures to support the revised training-derived approach.
- **Clinical headline:** computed IPI remains the manuscript-marked comparison;
  NCCN-IPI is marked false. Thus the main clinical figure selection is outdated.
- **Recent weeks:** the restriction summary contains older source restrictions,
  not the final 7/14/21/28-day gaps. Rendering failed with “No recent-record gap
  results available.” This is missing analysis output, not merely bad spacing.
- **Attribution:** the atlas and structure plots exist. The main plate and
  single-patient plate were skipped because episode inputs were absent. The
  bundle does not establish that the fresh IG run finished or supplied these
  summaries. Check that refresh manifest before attributing them to it.
- **National:** the main transfer figure was skipped for missing estimates.
  `supp_national_where` is an empty axes/legend placeholder. It is not a result.
- **Strategy:** the selected-strategy evaluation panel is unavailable. Do not
  leave a blank panel in a final figure or infer the missing comparison.

## Figure disposition

All are preview figures; “retain” means a promising design after final data and
caption/provenance verification, not submission readiness.

| Family | Disposition |
|---|---|
| Pooling figure and cell-level pooling maps | Retain. Clear central evidence; fix missing patient-count labels (`nan`), support annotation and tiny labels. |
| Tabular comparison and calibration | Retain. Strong, candid benchmark; keep descriptive calibration medians distinct from paired effect means. |
| Outcome atlas | Retain in supplement. Absolute-performance overview, not proof of a sharing gain. |
| Outcome embedding galleries 1–6 | Retain as descriptive supplement. Updated patient-point rendering is present. |
| Patient map | Retain descriptively; observed-risk and prediction colour meanings must remain explicit. |
| Router-vs-joint domain figure | Retain with support and current identities. Its right panel explicitly shows original bank controls and should be separated from the primary-router argument. |
| Hierarchy / rescue | Revise. Different control baselines, same-data tuning harm, descriptive IQRs and uncertain crossover need clearer treatment. |
| Borrowing map/detail, mean and zero | Redesign. Near-zero auto-scaled shrinkage maps and normalised lineage summaries currently overstate interpretability. Verify subgroup patient-count source too. |
| Gate usage | Diagnostic supplement. Add per-patient dispersion and cross-seed evidence before making stability/selectivity claims. |
| Arrow/calibration maps | Redesign layout and inspect per-outcome vectors numerically. Arrows exist. |
| Clinical figures, every index | Refresh inputs before layout iteration. Three-category/tuning-derived version remains in this bundle. |
| Clinical PR panels | After input refresh, move annotations away from curves and separate operating-point summaries. Current overlapping labels remain. |
| Embedding comparison | Revise. Still uses broad contours and smoothed covariates, unlike the updated galleries; IPI mean categories have limited meaning. |
| Coverage/support | Retain in supplement; distinguish model-fit availability from evaluation support. |
| Regional and patient-subgroup figures | Retain after endpoint/weight checks; simplify density and small labels. |
| Headline | Rebuild selection/labels. Current panels A/B describe joint model, despite manuscript caption framing OPERA; clinical input remains old. |
| IG atlas/structure | Retain provisionally, pending fresh-run provenance. Atlas is too dense at normal reading size; structure input-restriction labels overlap. |
| National/subset and strategy placeholders | Omit until actual estimates exist. |

Every figure also contains a literal `\\u2014` watermark escape rather than an
em dash. Fix mark normalisation once in the rendering pipeline.

## What is justified now?

The bundle supports a coherent tuning story about information sharing. It does
not support selective biological borrowing, clinical superiority under the new
threshold design, or a national-pretraining conclusion. It also does not show
that the primary router is the wrong checkpoint. We need less interpretation
of tiny individual-source effects and stronger verification of the diagnostic
exports before revising model selection.

## Next actions, in order

1. Keep this archive immutable as the reviewed snapshot.
2. Retrieve the small aggregate `hierarchy_cells.csv`, plot-table manifest and
   the selected-model/inference identity summary. Check the own-only reference
   checkpoint recorded under RDH in source-removal provenance against RBH_AB.
3. On the server, summarise per-outcome arrow arrays numerically (hashes,
   pairwise equality, displacement magnitudes and gate differences), and
   report gate dispersion per patient and across the already trained seeds.
   Export aggregate diagnostics, not patient arrays. No retraining is implied.
4. Finish the clinical-only refresh and select NCCN-IPI as intended. Finish
   recent-week restrictions and fresh IG summaries, then assemble explicit
   fresh directories rather than defaults pointing to the original workspace.
5. Fix plot selection, scale and layout using these completed inputs. There is
   no reason to rerun all mean/zero inference just to change the heatmap.
6. Record the final local evaluation specification and proceed to held-out
   evaluation after these provenance checks. Freeze primary contrasts and
   supported-cell rules before inspecting held-out results.

Supporting local artifacts: `audit_contact_sheets/sheet_1.jpg` through
`sheet_6.jpg`, and `audit/tuning_panel_results.csv`,
`audit/router_domain_horizons.csv`, `audit/tuning_auc_by_horizon.csv` inside the
extracted review directory.
