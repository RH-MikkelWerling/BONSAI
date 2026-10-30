# Parallel audits: interpretation and release status

30 October 2026. Reviewed `scratch/parallel_audits_review.zip`, including its
nested mechanism ZIP. This is **tuning, seed 42**. All recorded audit stages
completed. Completion is not a blanket test-release sign-off.

Local graphical/numerical review:
[review page](../scratch/parallel_audits_review_20261030/review/index.html).
The reproducible aggregate analysis is
`scratch/parallel_audits_review_20261030/analyse_review.py`. Its output includes
CSV summaries, input hashes and a machine-readable checklist. No patient-level
predictions or checkpoints were needed for this local review.

## Decision now

The AKI refit proposal is withdrawn. The selected revised router is present and
its prediction algebra checks out. Its benefit is associated with own-task and
borrowed patient-specific corrections. These results do **not** support the
idea that the global adapter supplies most of the discrimination improvement.
They provide weaker evidence that tailoring the precise source proportions
to each patient is essential.

The remaining work is clinical-comparison refresh, figure/model consistency,
the newer pooled-population diagnostics, and the planned national downstream
fits/selection. There is no AKI-driven reason to repeat training or the long
cross-family metric audit.

## 1. AKI resolution and existing locks

The author's server reproduction, logged at 19:30 in
`ANALYSIS_LOCK_DRAFT_20261015.md`, identifies the saved prototype as the table's
producer. It reproduced **86,829 onsets in 31,213 patients with no differences**.
The hypothetical look-ahead implementation changed onsets in 14,068 patients.
The strict seven-day boundary changed 22 patients. Retain the actual prototype's
365-to-7-day historical mean, **both boundaries inclusive**. Do not introduce the
strict-boundary version as an unneeded repair.

The active AKI source is used by both outcome derivation and tabular features.
This specific look-ahead concern is resolved for both. The outcome roster,
release, feature matrices, fitted models and IPW sensitivity roster remain as
they were. This is not a claim that every unrelated clinical-definition question
has been settled. The raw server reproduction outputs are not in this ZIP;
the author and dated analysis record supply that resolution.

Updated the two active agendas and the data-definition question list, which
still incorrectly called AKI an active blocker. Historical repair documents
remain marked superseded. No running jobs or data releases were changed.

## 2. Correct router, predictions and metric implementation

The selected model is **R_AB_L**, analysed under `R_L`/OPERA:

- Fit: `seed42/L/fix_R_AB`; run `79d39ad1e7cb51472026`.
- Checkpoint SHA256: `2d58ab74d124065eb08b719237418dcc0b1507f2ce7b5fafda82b53b87bc1524`.
- Global adapter enabled; two router warm-up epochs; both source banks.
- Best epoch 12, selected by endpoint-group macro horizon BCE.
- Frozen config, selected checkpoint and evaluation receipts agree.
- Route exports and selected prediction directory agree. The original router
  has a distinct identity, `R_ORIG_L`, rather than being substituted silently.

The mechanism audit covers **1,980 patients, all 69 outcomes and six horizons**.
Reconstructed probabilities match all **806,580 prediction rows**, with zero
mismatched rows and maximum probability error `8.94e-7`. This strongly supports
the tested readout/export implementation. It does not prove generalisation or
identify a biological mechanism.

The cross-family audit independently reconciles **18,249 prediction artifacts**:
one joint, one OPERA, 8,843 per-disease XGBoost and 9,404 pooled-XGBoost artifacts.
There are no missing saved estimates or finite-status mismatches. The largest
native-versus-analysis metric discrepancy is `5.27e-15`. Saved and recomputed
finite cell estimates agree to `1.11e-16`.

These checks establish consistency under the existing prediction/label
contract. They do not independently validate clinical source definitions or
turn tuning estimates into external validation.

## 3. What the router contributes

All differences below are **AUC percentage points**: raw AUC difference ×100.
For example, +0.87 points means raw AUC +0.0087, not +0.87 AUC.

The separately trained models share 3,004 supported disease/outcome/horizon
cells in the packaged control ladder:

| Separately trained model | Mean cell AUC | Difference from joint, points |
|---|---:|---:|
| Joint | 0.72445 | 0 |
| Global-only control, N_A | 0.72122 | −0.324 |
| Own-task-only control, RBH_AB | 0.72860 | +0.415 |
| Full revised router | 0.73313 | +0.868 |

Full versus own-task-only is +0.453 points. Full versus global-only is +1.191
points. These are descriptive comparisons of separately trained fits, not
additive causal allocations of improvement to architectural parts.

The independently paired frozen-path audit gives full OPERA versus joint
**+0.868 points (95% CI 0.357 to 1.458)** over all horizons. At one year, the
main cross-family analysis gives **+1.079 points (0.339 to 1.895)** across 600
supported disease/outcome cells. The frozen audit uses 400 multiplier draws;
the cross-family table uses 1,000. Their point estimates agree.

Within the selected checkpoint, replace a correction by its disease/target
tuning mean, retaining the average shift and removing its patient variation:

| Patient-specific correction removed | Full minus altered AUC, points | 95% CI |
|---|---:|---:|
| Global | +0.009 | −0.085 to +0.086 |
| Own disease/outcome | +0.506 | +0.239 to +0.767 |
| Other diseases | +0.204 | −0.009 to +0.433 |
| Other outcomes | +0.180 | +0.025 to +0.322 |
| All borrowed corrections | +0.489 | +0.172 to +0.835 |

This answers the global-adapter concern. Removing its patient variation barely
changes aggregate discrimination; removing own-task or combined borrowed
variation lowers it. The encoder/head-only path and the path with its global
correction are both below the separate joint fit. The task/source corrections
bring the full model above it. Frozen paths are co-trained parts of OPERA, not
substitutes for separately trained comparators.

Freezing **all gate weights** within each disease/target lowers AUC by only
**0.145 points (−0.072 to +0.366)**. Freezing just relative borrowed-source
proportions lowers it by **0.121 points (−0.087 to +0.346)**. Adapter outputs
still vary by patient in both controls. This distinguishes useful borrowed
predictions from strong evidence that every patient needs a different mixture.
The gate substitution has a small Brier improvement in favour of full OPERA
(`0.000639`, CI `0.000030` to `0.001177`), so do not describe every measured
effect of personalised gates as null.

The five-outcome vector audit's 1.54% outcome-specific correction energy does
not identify the global adapter. The exact head-space component audit spans
all 69 outcomes and shows a smaller global contribution than own/borrowed
components. Similar latent arrows are compatible with the findings above.
The separate all-outcome geometry/risk-rank sidecar is not in this archive.

The original router is not uniformly better: its all-horizon point gain over
joint is +0.950 points versus +0.868 for the revised fit. At one year the
original-minus-revised difference is **−0.028 points (−0.422 to +0.363)**.
There is no one-year evidence here that the revised fit sacrifices a clear
performance advantage. The data support retaining a verified interpretable
architecture, without claiming it wins every training comparison.

## 4. Pooled cohort: XGBoost advantage survives matched populations

The archive's common-population table has equal patient, event-case and control
counts across all four models for every shared outcome/horizon. `common_n` is
also identical across models in the coverage report. No asymmetric denominators
are apparent in these aggregate results.

At one year, all 69 outcomes have finite pooled AUCs. Equal-outcome descriptive
means are:

| Model | Mean pooled-outcome AUC | Mean Brier | Mean IPA |
|---|---:|---:|---:|
| Joint | 0.74995 | 0.09506 | 0.10631 |
| OPERA | 0.76494 | 0.09150 | 0.12227 |
| X, disease-specific XGBoost portfolio | 0.80862 | 0.08650 | 0.17200 |
| XP, disease-pooled XGBoost | 0.79172 | 0.08718 | 0.17475 |

OPERA has higher one-year pooled AUC than joint for **53/69 outcomes**, than X
for **4/69**, and than XP for **14/69**. Its mean paired AUC difference is
+1.499 points versus joint, −4.368 versus X and −2.678 versus XP. These averages
are descriptive. Do not average individual confidence limits to invent an
interval for the average, or count unadjusted per-outcome intervals as a new
confirmatory hypothesis family.

The X advantage is not confined to AUC: its Brier/IPA results above are also
stronger on this population. There is no basis here for presenting local OPERA
as the best overall predictor or assuming national pretraining will reverse it.

Population caveats remain:

- The common population is the four-model intersection. It is a fair paired
  comparison on that population, not proof of equal prediction coverage.
- At one year, OPERA loses an average 0.50% of its eligible patients per
  outcome to parity exclusions (median zero); the largest exclusion is 10.48%.
  At 30 days, rare outcomes can lose far more. One loses 712/1,947 patients.
- `next_malignancy` and `next_hematological_malignancy` have **no shared roster
  at 30 days**. The old coverage CSV shows zero common patients, but its paired
  contrast coverage omits these tasks. A third task has insufficient cases/
  controls. Thus 66/69 30-day AUCs are finite, not all 69.
- Current evaluator code already adds explicit absent-model coverage,
  `population_by_disease.csv` and `auc_pair_composition.csv`. Those newer
  outputs are absent from this archive; do not call them reviewed or recompute
  the expensive cross-family audit just to obtain them.
- Pooled AUC compares cases and controls from different diseases as well as
  within diseases. It answers a different question from the primary macro
  average of disease-specific cells. X is a portfolio, XP one fit per outcome/
  horizon. Their cross-disease score ordering needs the decomposition for
  interpretation, even when the arithmetic and paired populations are sound.

For one-year mortality specifically, the common population is all 1,980
patients with 266 deaths. OPERA minus joint is −0.596 points (CI −1.679 to
+0.427), versus X −3.576 (−5.805 to −1.354). Benefits in the aggregate do not
imply a mortality benefit over joint.

The primary fine-disease OPERA-versus-X pooled contrast table is not packaged.
Do not substitute these whole-cohort means for that headline estimand.

## 5. Clinical scores: available, but this is the older grouping snapshot

All **10 score types** are in `score_inventory.csv` and in the independent
binary-anchor classification table. Only five render ordered model-group
plates with the old rule: IPI, NCCN-IPI, FLIPI2, R-IPI and R-ISS. CLL-IPI,
IPSSWM, MALT-IPI and MIPI have crossed low/high anchors. IPS has two native
groups but incompatible independently fitted low/high cutoffs. Those are
grouping-rule failures, not five missing clinical datasets.

The snapshot was copied at 12:05. It uses **apparent training predictions**
and `risk_matched_low`. It does not contain the later cohort-size matching,
KM-matched Low comparison or their paired survival tables. Follow
[the CPU refresh instructions](CLINICAL_AND_RECORD_GAP_REFRESH_20261030.md).
The final development source for outcome-based frozen thresholds still needs
the explicit protocol choice described there.

NCCN-IPI reconciliation in DLBCL: 476 predicted patients, 343 with an available
score at index, 133 unavailable/late. Native groups contain 24/86/150/83
patients, preserving all four categories. Model Low contains 26 versus native
Low 24, with **one death in each**. This is two additional patients, not a
persuasive large clinical expansion on the tuning sample.

The frozen high cutoff transported from training gives specificity 0.870 and
recall 0.525, versus NCCN-IPI specificity 0.824 and recall 0.559. These are not
matched specificity in the evaluation cohort. Precision is 0.456 versus 0.398;
MCC 0.374 versus 0.338. Describe each operating point correctly. Some other
scores' model Low groups have higher observed mortality than native Low after
transport. Do not claim matching training risk guaranteed matching tuning risk.

Recorded-score native category definitions remain marked unverified in the
config audit. Preserve that narrower open issue rather than interpreting an
available category as fully validated clinical applicability.

## 6. Figures and remaining sign-off

Viewed all three packaged mechanism PNGs. The three-panel AUC figure is
readable, coherent and within the established style; no obvious clipped labels
or intervals. The IPA companion and multi-metric sensitivity plot are also
readable. File hashes match both the packaged figure receipts and source
summary tables. Captions must distinguish frozen interventions from trained
controls and state tuning/conditional patient-bootstrap uncertainty.

The atlas environment input still points to joint-model pooled metrics while
clinical/maps use selected OPERA. This may be deliberate. Resolve the identity
explicitly: keep a clearly labelled joint atlas or select OPERA consistently
for a primary-OPERA atlas. It is not evidence that the mechanism audit used the
wrong router. The archived audit reports this mismatch accurately.

The disease-pair map is legitimately almost blank after shrinkage: across 460
off-diagonal pairs, median absolute raw effect is 0.058 AUC points, but the
median shrunk effect is `1.60e-6` points and maximum `6.51e-5`. Correct-model
provenance is present. This does not establish that every analysis is perfect,
but switching colour scales cannot make those posterior estimates support a
strong pair-specific borrowing narrative. Prefer aggregate reliance and its
uncertainty. Retain raw pair estimates/intervals descriptively in the supplement
if useful, without presenting noisy extremes as discovered clinical relations.

Next actions, in order:

1. Keep the current release/fits. No AKI correction or repeat training.
2. Refresh clinical comparisons on CPU with the declared size/KM rule and
   inspect the resulting paired tables before fixing the final protocol.
3. Collect current whole-cohort disease/pair decomposition outputs and the
   existing primary `prediction_analysis/effect_summaries.csv` (plus its
   `effect_summaries_disease_balanced.csv` sensitivity). These are
   needed to distinguish the headline disease-cell comparison from pooled
   cohort ranking. Also collect the all-outcome geometry sidecar if run.
4. Make the atlas model identity explicit and prefer a mechanism figure over
   an uninformative pairwise map for the main borrowing explanation.
5. Finish planned national downstream training/tuning decisions. Confirm final
   model/analysis inputs and coverage, then release the test evaluation.

The main rehearsal's remaining IG/removal/record-gap receipts and final figure
assembly are not certified by this parallel audit ZIP. This review does not
require repeating completed checks with matching scientific inputs, and does
not change a model, threshold, estimand or test-set release on the server.
