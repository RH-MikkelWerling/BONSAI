# Deep tuning rehearsal review: figures, routers, manuscript numbers and test opening

Reviewed 28 October 2026. This extends `FIGURE_BUNDLE_AUDIT_20261028.md` with a
per-file review, full results ledger, historical router identity/health evidence,
corrected aggregate-only previews and an explicit test-opening recommendation.

## Evidence and scope

The immutable source is `scratch/paper_reading_20261028/paper_review_download_20261028_182155.zip`.
Its hashes passed verification. Every one of its 69 PNGs was inspected in contact
sheets, with key router, borrowing, clinical and performance figures enlarged.
Four requested figures were absent according to renderer manifests. The local
review dashboard provides one assessment for each exported figure and each
missing figure; a useful design is not the same as a publication-ready result.

Additional evidence: the author-supplied `~/Downloads/router_audit_current.json`
(27 October) and `router_cell_distribution (1).json` (28 October). These are
historical diagnostic exports, explicitly identified and hashed in the new
review output. They refer to the same primary/original fit directories as the
current bundle. Server checkpoints and patient-level prediction arrays were
not independently reread. No test data, national results or new training were
used. No new bootstrap intervals were invented or formed by adding endpoints
of intervals from different contrasts.

Artifacts are in `scratch/paper_reading_20261028/deep_rehearsal_review_20261028/`:

- `figure_review.html`: filterable visual review of all 69 figures and four missing figures.
- `all_figures_review.csv`: machine-readable per-file disposition and SHA256.
- `equal_cell_ledger.csv` and `disease_balanced_ledger.csv`: all exported AUC,
  IPA and Brier panel estimates, with uncertainty and strata retained.
- `tuning_manuscript_numbers.json/.tex`: 576 available tuning macros,
  pooled and one-year, derived from tuning only. No stale clinical macros included.
- `absolute_auc_common_roster.csv`: descriptive absolute performance on a fixed
  roster; not pooled-patient AUC or an interval estimate.
- `router_domain_horizon_comparison.csv`, `disease_course_influence.csv`.
- `support_sensitivity_descriptive.csv`: stricter case/control cutoffs, with
  changing task composition and no newly estimated uncertainty.
- `router_identity_and_health.json`: historical diagnostics and fit identities.
- `shrinkage_reproduction.json`: direct reproduction of the saved mean-ablation
  shrinkage and a raw-AUC versus AUC-point unit check.
- `hierarchy_cells_reconstructed.csv`: AB controls reconstructed from aggregate
  contrasts, replacing a table omitted by the archive allowlist.
- `corrected_previews/`: explicit OPERA headline, mean/zero raw and shrunken
  borrowing maps, borrowing detail, reconstructed hierarchy and a clear router
  control comparison. These are review designs, not new clinical results.
- `opera_manuscript_rehearsal_review.pdf`: a separately watermarked reading
  copy with the current tuning contrasts and reviewed figures. Clinical,
  attribution and national gaps remain explicit. It does not replace the
  held-out manuscript or turn planned test captions into completed analyses.

## 1. Every figure as a rehearsal

### What the rehearsal already establishes

There are 35 clinical PNGs built from the old clinical directory, not the
training-threshold refresh. There are 30 main-recipe PNGs, two borrowing-zero
PNGs and two attribution PNGs. Missing figures are the national transfer main
plate, recent-week gaps, main attribution plate and patient attribution plate.
Two *existing* figures also have missing panels/data: national subgroup results
and selected-strategy evaluation. The absence of a file is not the same as a
null result, and presence of a PNG is not proof of completed analysis.

The principal conceptual plots should have distinct roles:

| Scientific purpose | Required model identity |
|---|---|
| Value of disease/outcome supervision | Fixed S/D/O/J comparisons. Keep J; substituting OPERA would conflate supervision with corrections. |
| Headline clinical/benchmark performance | Explicit primary OPERA, accompanied by joint and national OPERA variants when available. |
| Absolute predictability atlas | Either explicitly fixed J as a reference, or a labelled primary OPERA atlas plus a companion difference view. Never silently rename J. |
| Clinical score comparison | The declared routed fit with native score categories and frozen training thresholds; per-comparison populations. |
| Routing mechanism | Primary AB, its matched global/own-only controls and separately labelled original router. |
| National transfer | Joint local/national/adapted and routed local/national/adapted, kept separate and evaluated on common supported rosters. |
| Representations and attribution | Checkpoint identity stated. No geometry comparison across independently fitted projections as evidence of movement. |

The current primary headline mixes identities: its A/B/E are joint, its clinical
panel uses routed predictions. This is a real presentation error. The renderer
now has an explicit J/OPERA focus, fails when the requested absolute model is
absent, and uses direct primary-versus-specialist/joint/XGBoost contrasts.
Its horizon panel can show joint, primary, original and national routed variants
without substituting national joint-model effects. The original router is now
exported in model-cell tables. Missing national terms are omitted, not invented.
A national winner must not be retrospectively selected on held-out performance.

The clinical panel now requires model and training-threshold provenance in the
headline export. The clinical exporter writes those fields. Old coarsened
clinical tables therefore produce “Clinical refresh pending” rather than being
relabelled as the updated evaluation. The ZIP omitted subgroup aggregate tables
needed to reproduce panel E; it is explicitly unavailable in the corrected
local preview. The next package includes `patient_subgroups.csv` and the
hierarchy table. Raw patient arrays remain excluded.

### Presentation and theme

The palette and sans-serif type are largely coherent. The major remaining
problems are excessive density, long clinical-domain labels, overlapping PR
annotations/row labels, automatic heatmap scaling and inconsistent statistical
summaries. A theme should not make a tiny effect look large or a missing panel
look like a completed negative result.

Specific decisions:

- Keep pooling, tabular and calibration comparisons. Simplify redundant panels
  after final findings are known, rather than dropping unfavourable contrasts.
- Keep updated patient-point galleries as descriptive supplementary views.
  Observed-risk kernel surfaces and predicted-risk points must retain different
  labels. The older smoothed-covariate comparison needs redesign independently.
- Rework arrow figures into a dedicated larger row/plate, with readable shared
  scaling and a compact numerical displacement diagnostic. Arrows exist now.
- Rework mean/zero borrowing maps with a common interpretable scale. A minimum
  full scale of one AUC point corresponds to the existing 0.01 AUC practical
  margin. This is a display choice, not a threshold for significance. The
  original automatic scale stretched a few ten-thousandths of an AUC point
  across the whole colour map.
- Replace the lineage-share panel with raw-versus-shrunken absolute effects in
  the default map. Relative shares of near-zero posterior means are not useful
  biological evidence. The old panel remains an explicitly selectable diagnostic.
- Keep descriptive medians/IQRs for absolute atlases or distributions if clearly
  labelled. Paired effect means/CIs, hierarchical intervals, IQRs and seed
  variation must not be presented interchangeably.
- Treat the approximately 84-event quadratic crossing as exploratory. It is not
  an estimated clinical threshold or a precise transition justified by this sample.
- Suppress empty national/strategy panels in the final paper until inputs exist.
- Fix the literal `\\u2014` mark once centrally. Watermarks now normalise it, and
  the renderer preserves recipe marks when no environment override is set.

Do not retrofit the test design to make plots cleaner. Common-roster diagnostics,
count annotations and fixed scales can reveal instability without removing it.

## 2. What happened to the routers?

### Identity is substantially resolved

The current performance exports identify R_L and OPERA as R_AB_L:
`79d39ad1e7cb51472026`. Their 18,024 aligned finite absolute metric entries are
identical. The discarded R_ORIG_L is `0de5f3abae9725a220e5`, with 17,880 differing
entries. The historical diagnostic reports matching config/checkpoint/prediction
hashes for those runs. It also identifies the own-only RBH_AB control as
`6e2936e25eccab82590e`, matching the RDH path in the current source-removal
provenance. “RDH” there is a level-label key; it does not establish the wrong
control was used.

The earlier diagnostic already used the revised fit. Consequently unchanged
plots across these recent packages can be expected: another render or fresh
source-removal computation on the same fixed model and patients need not change
the result. Some families remain stale, especially clinical inputs, but a
stable router figure is not itself evidence of stale model selection.

### Health versus selectivity

Historical *per-patient/target* route diagnostics, seed 42:

| Diagnostic | Original | Revised AB |
|---|---:|---:|
| Effective sources (reported overall summary) | 1.595 | 70.803 |
| Top gate weight | 0.899 | 0.072 |
| Borrowed disease mass | 0.0004 | 0.1736 |
| Borrowed outcome mass | 0.8863 | 0.6962 |
| Most common top source | OS, 99.82% of rows | DLBCL, 14.71% of rows |

AB passed the recorded support-weighted health checks: top-source share 15.07%,
six distinct top borrowed sources across diseases and median effective-source
count 76.018. This supports restored allocation and absence of the previous
single-source collapse. It does not prove selective biological borrowing, good
calibration or replication across seeds. The paper describes a second seed, but
its numerical health report is not in this bundle. Obtain that report to support
“stable across seeds.” Also distinguish stable scalar diagnostics from identical
source rankings across seeds; non-identifiable source allocations may vary.

A broad gate mixture can be healthy. It may implement diffuse sharing rather
than choosing a few sources. Mean profiles also conceal patient-level variation.
Outcome mass is partly affected by having 69 outcome versus 24 disease sources.
Gates are weights, not the hazards contributed by their weighted adapters.

### Predictive performance

Equal-cell, all horizons, supported roster:

- AB vs joint: +0.00868 AUC (0.00313 to 0.01458).
- Original vs joint: +0.00950 (0.00403 to 0.01497).
- Original vs AB: +0.00082 (−0.00208 to 0.00372).
- Original vs AB IPA: −0.00425 (−0.01047 to −0.00082).

At one year, common-roster mean absolute AUC is 0.71828 for AB and 0.71800 for
the original across 600 cells. That difference has no newly computed interval.
The old model is not globally superior. Its small all-horizon AUC advantage is
weight/horizon dependent; probability accuracy favours AB in the equal-cell
summary. Do not pick a model based on which summary happens to favour it.

Disease-balanced original-minus-AB AUC is +0.00386 (−0.00177 to +0.00989),
consistent with the earlier quoted cost of approximately 0.004. This resolves
why earlier selection descriptions and the new +0.00082 figure differ.
Intervals reflect patient uncertainty conditional on fitted models, not a full
training-seed noise distribution. The phrase “within the noise floor” should
refer to the actual separate repeat-fit evidence, not merely a CI crossing zero.

The original router remains a valuable sensitivity model. Gate collapse does
not invalidate its predictions: its shared path and own-task floors still
operate. It weakens interpretation of its borrowed-source allocation. The
revised primary can be justified by a documented development amendment,
restored allocation and a small predictive trade-off. It cannot be justified
solely by producing desired maps.

### Three different contributions must stay separate

| Comparison | Equal-cell AUC | Disease-balanced AUC |
|---|---:|---:|
| Own-only minus global-only | +0.00739 (0.00137, 0.01373) | −0.00502 (−0.01913, 0.00901) |
| Full AB minus own-only | +0.00453 (0.00041, 0.00888) | +0.01379 (0.00354, 0.02556) |
| Full AB minus global-only | +0.01191 (0.00672, 0.01759) | +0.00877 (−0.00095, 0.01897) |

These are separately trained architectures. Disease-source mean replacement is
another estimand: it retains outcome/global/shared components and measures
fixed-fit reliance. Its all-disease estimate is +0.00459 AUC (−0.00179 to
+0.01039). Weak per-disease source evidence is compatible with a useful full
architecture. It does not establish that learned non-uniform gate selection is
necessary: there is no matched uniform-mixture training comparator in this
bundle. State that limit rather than launching an open-ended search for a
more selective router. A fixed-fit uniform-gate check could be a declared
secondary diagnostic, but would not replace a retrained comparator.

I also reproduced the saved shrinkage for all 452 finite off-diagonal,
all-outcome pairs in the mean-ablation export. Maximum numerical disagreement
was 2.1e-19 AUC points. Converting both effects and SEs to raw AUC before
fitting, then converting back, gave the same result. The fitted prior SD was
0.00064 AUC points, with median absolute raw z-score 0.65. Median absolute
raw effect was 0.060 AUC points, maximum 3.11, while maximum absolute shrunken
effect was 0.000065 AUC points. The near-blank corrected map reflects the
fitted shrinkage, not an extra factor-of-100 error in that step. This is weak
evidence for specific source pairs, not proof that every true source effect
is zero. Shared-patient dependence also limits a simple independent-effects
interpretation of the empirical prior.

### Disease-course and cardiovascular concerns

Disease-course R1 mean is −0.01797 over 261 cell–horizon entries. Removing RT
*diagnostically* gives −0.000258 over 249. Twelve RT entries repeat three nested
outcomes over 30/60/90/180 days, with 5–7 cases and 6–8 controls. Early-horizon
losses and repeated labels explain much of the pooled appearance. The original
router's disease-course mean is also negative (−0.01433), so returning to it
would not recover the remembered domain-wide benefit. At one year AB's mean
is approximately zero. This is a real fit/population result, not an axis flip.

Cardiovascular R1 mean is −0.01990 over only 46 entries. Only one cell is supported
at 30 days and two at 60 days; the original is worse (−0.03193 overall). Show
support and task composition rather than describing the curve as a stable
trajectory. Do not remove RT or rare cardiovascular cells to improve headlines.

## 3. What can the manuscript currently say numerically?

### Core sharing results

Equal-cell AUC differences, all supported horizons:

| Contrast | Difference (95% interval) | Interpretation |
|---|---:|---|
| Disease-pooled vs specialist | +0.11966 (0.10679, 0.13315) | Large disease-sharing gain within this neural model family. |
| Outcome-pooled vs specialist | +0.02734 (0.01472, 0.04038) | Smaller AUC gain; IPA is −0.00499 (−0.01146, 0.00082). |
| Joint vs specialist | +0.11852 (0.10426, 0.13317) | Large gain, but not larger than disease pooling alone. |
| Joint vs outcome-pooled | +0.09118 (0.08074, 0.10084) | Disease supervision adds substantially when outcomes are already shared. |
| Joint vs disease-pooled | −0.00114 (−0.00754, 0.00526) | No demonstrated additional discrimination from outcome supervision after disease pooling. |
| Pooling interaction | −0.02848 (−0.04220, −0.01495) | Benefits are not additive; not proof of a biological antagonism. |

This is more informative than “more sharing is better.” Additional supervision
can be redundant or constrained by training. The equal-cell and disease-balanced
conclusions agree that disease pooling is the dominant observed contribution.
Disease-balanced joint-minus-disease-pooled is −0.00958 (−0.02490, 0.00376).

### Strong alternatives and probability accuracy

- Primary OPERA vs per-task XGBoost: AUC −0.02803 (−0.03633, −0.01929),
  IPA −0.03596 (−0.05323, −0.02469), Brier improvement −0.00371
  (−0.00661, −0.00064). The deficit is not confined to ranking.
- Joint vs pooled XGBoost: AUC −0.04824 (−0.05575, −0.04057).
- Disease-pooled neural vs pooled XGBoost: −0.04709 (−0.05352, −0.04080).
  This is particularly important: sharing scope is comparable, so the neural
  deficit cannot be explained simply by its comparator seeing more diseases.
- Pooled vs per-task XGBoost: +0.01153 (0.00561, 0.01717).
- XGBoost vs logistic regression: +0.07517 (0.06724, 0.08477).

On the same 600 one-year cells, descriptive mean absolute AUCs are:
S 0.59418, D 0.71065, O 0.62905, J 0.70749, AB 0.71828,
original router 0.71800, XGBoost 0.74982, pooled XGBoost 0.75434,
logistic regression 0.66749. These are means of cell AUCs, not an AUC obtained
by mixing patients across diseases/outcomes. Weak neural specialists magnify
the apparent sharing gain. This is precisely why the simpler baselines matter.

### Counts, scarcity and support

- One-year OPERA AUC is higher than XGBoost in 189/600 cells (31.5%).
- One-year AB is higher than joint in 351/600 (58.5%).
- Joint is higher than specialist in 453/600 (75.5%).
- With horizons pooled within each cell, the corresponding counts are
  212/675, 390/675 and 523/675. Do not mix these denominators with the
  3,004 cell–horizon count used for the main pooled effect.
- In diseases with fewer than 500 training patients, OPERA-minus-XGBoost AUC
  is −0.003 (−0.024, 0.018). This is uncertain, not established equivalence.
  Joint-minus-specialist is +0.173 (0.134, 0.211).

The >=5 case/control rule is an evaluation support policy, not a property of
training. It selects a more observable task distribution and reduces, but does
not eliminate, instability. It does not favour a model by giving it a different
roster within a paired contrast. It does restrict generalisation to extremely
sparse tasks. The current implementation gates *all* metrics at this threshold,
including Brier and IPA, not just AUC. The exported Brier summaries do not
automatically cover the additional single-class cells. If broader Brier coverage
is wanted, declare it as a separate secondary population before test evaluation;
do not silently change the primary paired roster.

As a rehearsal sensitivity, raising the minimum cases **and** controls gives:

| Minimum | Supported cell–horizons | Disease-pooled minus specialist | Joint minus specialist | OPERA minus joint | OPERA minus XGBoost |
|---|---:|---:|---:|---:|---:|
| 5 | 3,004 | +0.1197 | +0.1185 | +0.0087 | −0.0280 |
| 10 | 1,802 | +0.0866 | +0.0777 | +0.0157 | −0.0356 |
| 20 | 1,131 | +0.0708 | +0.0629 | +0.0149 | −0.0348 |
| 50 | 451 | +0.0233 | +0.0185 | +0.0089 | −0.0386 |

These are descriptive means, without new intervals. They compare progressively
different task populations, not the causal effect of adding events to the same
tasks. The broad pattern survives removal of the sparsest evaluated cells.
Pooling gains shrink, while the routing gain persists and the XGBoost gap
remains. The revised router overtakes the original on these >=10 and >=20
rosters (original minus revised: −0.0016 and −0.0022). This further weakens a
claim that the original is consistently better. Do not choose the primary
support rule from these results. A larger test set will change the supported
task mix as well as precision, so compare split-level summaries with a
common-roster sensitivity before interpreting their difference.

### Claims not yet available

The current clinical figures do not support the revised training-matched risk
groups. Do not promote old NCCN-IPI low-risk counts as new results. Clinical
claims need the native four categories, training cutoff manifest, achieved
held-out rates/specificities, paired uncertainty and per-index populations.
MIPI already illustrates why “beats every index” cannot be assumed from the
old tuning overview. Overlapping index comparisons cannot be summed into a
unique-patient count.

National transfer and label efficiency have no results in this bundle. Recent
7/14/21/28-day gaps are missing. Existing attribution/source-restriction numbers
are provisional until the fresh fit/sample manifest is checked. The older
restriction analysis is pooled over outcomes across patients and is not the
same estimand as the cell-level contrast ledger.

## 4. Open the test set now or wait?

Recommendation: **finish this finite rehearsal and freeze the development
choices now; open the test set after the national arms have completed their
tuning-based selection, if that is still imminent.** This is about keeping
selection independent, not hoping for better results or needing more sample size.
Do not wait to implement cosmetic changes or to obtain a selective borrowing map.

Opening local test results now is defensible only if *all* national configurations,
selection metrics/rules, adaptation choices, thresholds and planned contrasts
are already fixed, and nobody uses local test findings to revise those choices.
The same patients and related outcomes connect the arms. Seeing local failures
can influence national development even before national test predictions exist.
We cannot establish from this archive that the national selection is locked.
The current symbolic contrast manifest still labels itself planned and requires
an actual execution/analysis lock; do not confuse the template with that lock.

A finite release checklist:

1. Fix primary AB and original sensitivity identities, their controls and seeds.
   Retain original sensitivity irrespective of test direction. Obtain the second
   seed's aggregate health report if claiming cross-seed stability.
2. Record national encoders/readouts, checkpoint selection on training/tuning,
   adaptation schedules and full/partial-label comparisons. No test-based winner
   selection under an unspecified “OPERA” alias.
3. Freeze endpoints, horizons, native-score versions, threshold population,
   eligibility, >=5 support rule, averaging weights, paired bootstrap, contrasts,
   practical margins and handling of absent models. Identify primary vs secondary.
4. Verify one tiny end-to-end rehearsal for fresh clinical, IG, gap and plot
   exports with checkpoint hashes. Geometry/cosmetic work can continue later.
5. Save the actual locked configuration, file hashes, timestamp and amendment
   log, then evaluate all frozen local and national arms with the same pipeline.

Fixing a genuine coding error after test opening is still necessary, but record
what changed and which results had been seen. Do not treat repeated test-guided
model/analysis changes as independent confirmation. Do not withhold negative
results. Once the above conditions are met, waiting longer confers no scientific
benefit: the held-out results should become the main paper evidence.

## Implementation and validation

Changed the headline model selection and labels, original/national model-cell
export, clinical headline provenance fields, borrowing scale/diagnostic panel,
watermark normalisation and preservation, and the bundle's aggregate inputs.
Fixed J comparisons remain fixed. No model training, server inference or test
evaluation was launched. Corrected previews use the source numbers unchanged.
Scientific comparisons were not reselected for favourable outcomes.

Validation includes model-identity and stale-clinical rejection tests, small-effect
heatmap scale checks, plot-table/hierarchy integration and renderer tests. The
targeted suite passed 53 tests. The reading PDF compiled successfully. Its
figure-height limits were reduced to leave room for long supplementary captions.
File hashes are recorded with the working changes.

Remaining server exports are finite: native-category training-threshold clinical
outputs with operating-point metrics, second-seed router health if claiming
replication, per-outcome arrow-vector summaries to rule out duplicated arrays,
fresh attribution provenance, and the recent-week restriction rows. National
results and the frozen selected-strategy comparison should be added when ready.
None requires changing the primary local router merely to obtain a nicer map.

The recommendation to finish model selection before opening the test is
consistent with [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378).
The temporal test should also be described accurately: use of a patient's
pre-2022 history during unsupervised pretraining is not supervised test-label
training, but it does not demonstrate generalisation to patients absent from
all pretraining data. Document the permitted record dates and overlap.

## Graphical follow-through

See `FIGURE_GRAPHICAL_CHANGELOG_20261028.md` for the subsequent implementation
and every requested figure-family change. The graphical comparison is now
`graphical_audit_20261028/figure_review.html`. The factual results companion is
`graphical_audit_20261028/opera_tuning_results_snapshot.pdf`. Updated real
tuning artwork is linked into the rehearsal manuscript. Figures requiring
unavailable real inputs remain placeholders there, with their old artwork
preserved in the before/after gallery. No synthetic layout proof is inserted
into either results document.
