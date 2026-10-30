# Comparable model risk groups without predictive model retraining

30 October 2026. This is a candidate protocol for review, not a change to the
running server recipe or the final test lock. No test outcomes were used.

**Later discussion:** [same size versus same survival](CLINICAL_SIZE_AND_SURVIVAL_MATCHING_20261030.md)
considers a simpler primary grouped benchmark: match native category sizes
using baseline categories and model rankings, then compare KM survival.
It distinguishes that outcome-free cohort rule from direct KM-matched
development cutoffs. This document's IPCW low-anchored full partition remains
an optional candidate, not a silently activated primary analysis.

## What the HemaSphere paper actually did

The published article is [Werling et al., *Machine learning enhances risk
stratification and treatment failure prediction in diffuse large B-cell
lymphoma*, HemaSphere 2026, e70336](https://onlinelibrary.wiley.com/doi/10.1002/hem3.70336).
Its Baseline comparisons subsection separates binary classification from
four-category stratification.

For stratification, the stated requirements were a model Low group with an
event rate comparable to NCCN-IPI Low and adequately populated groups. The
reported numerical boundaries were 0.10, 0.30 and 0.65. Matching the highest
group's specificity was not a stated requirement for constructing those four
categories. The paper does not specify which data selected those boundaries.
Its endpoint was two-year treatment failure, including death, relapse or
second-line treatment. Those numerical boundaries cannot be imported into
OPERA's one-year mortality comparisons.

The local thesis `papers/paper2/paper2_addendum.tex` describes a later, stricter
operationalisation. It adds a high-specificity anchor and intermediate-size
matching and uses new out-of-fold fits. That operationalisation is not the
published grouping rule. Its reported sensitivity results also warn that the
size of the Low group depends materially on threshold development. The
underlying addendum result tables were not independently verified here, so
they should not be presented as newly confirmed empirical results.

The addendum implementation also falls back to a minimum-size Low group if
no prefix meets its risk target (`paper2_addendum_analyses.py:461`). That
fallback can fail the stated event-risk constraint. We should not copy it.

## Two separate reasons our present groups can look wrong

**An unnecessarily restrictive combination of objectives.** The current
`risk_matched_low` rule independently finds a largest low-risk prefix and a
high-specificity cutoff. It rejects their combination when they cross. That
does not show that every coherent joint partition is impossible. A smaller
Low prefix might meet the risk constraint while leaving space below the fixed
High threshold. For a two-category index, there is one boundary: the proper
joint question is whether that boundary satisfies both requirements, not
whether two independently optimised thresholds happen to be identical.

A constrained version retaining both targets is a defensible alternative.
It would maximise the Low group **subject to** a supported, ordered partition
and the independently fixed high-specificity boundary. However, some high
anchors may themselves leave fewer than the required number of patients in
High. Small samples and incompatible targets can still make that rule fail.
It should not be slipped in as a numerical fix to the current protocol.

**Threshold transport and clinical target transport.** Apparent training
predictions are not predictions on independent patients. Moreover, native
index group risks can change over calendar time and with score availability.
In the last downloaded report, NCCN-IPI's training Low risk was 0.76%, whereas
its tuning Low group had one death among 24 patients, or 4.17%. These are
historical rehearsal aggregates, not the result of the current server run.
Sampling variation and temporal changes can both contribute. A larger
training denominator cannot guarantee that its target or the model's numeric
threshold will reproduce a contemporary group's risk.

## Recommended candidate: low-risk-anchored, ordered model strata

Keep the published paper's useful clinical question, but make the remaining
choices explicit rather than choosing attractive-looking boundaries by hand.

1. Fix each index's eligible disease population, recorded native categories,
   endpoint and horizon. Use exactly the same score-available patients to
   compare the model with that index. Preserve four NCCN-IPI categories and
   four CLL-IPI categories where those categories are present in the frozen
   registry data. Do not manufacture a missing native category.
2. Use **existing tuning predictions** for threshold development. The same
   frozen predictive model produced them without training on those labels.
   Tuning was already used in model selection and remains development data.
   Fit both the native Low target and model cutoffs on the same paired tuning
   population. This avoids mixing a historical training risk target with a
   different contemporary threshold-development population. It does not
   remove small-sample uncertainty.
3. Estimate the native Low group's horizon event risk using the existing
   CIF censoring-weighting convention. Among attainable boundaries in sorted
   model scores, choose the largest Low group whose estimated event risk
   does not exceed that target **and which leaves a supported full partition**.
   Never split tied predictions or substitute an unsupported cutoff when
   there is no match.
4. Divide the remaining patients into K-1 ordered model strata. Target the
   native non-Low category proportions, conditional on being non-Low. Choose
   the attainable boundaries closest to those cumulative proportions subject
   to the same support requirements. This is a transparent descriptive choice.
   It does not impose equal group event risks or equal group sizes overall.
5. For the prototype, support means at least the greater of 20 known-status
   patients and 5% of the development population **per model group**. These
   are reviewable feasibility settings, not a guarantee of precision or
   clinical usefulness. Report event counts and effective sample sizes too.
   With four groups, this requires at least 80 known-status patients. Ties
   can make a partition impossible even above that count. Small native
   reference groups still give uncertain target risks.
6. The High boundary is the final boundary of that same ordered partition.
   **It is not required to match specificity.** Independently fit a binary
   specificity-matched cutoff for the clinical detection comparison. Show
   its achieved specificity, sensitivity, precision and event counts on the
   evaluation cohort. Do not use this independent cutoff to redefine High
   in the stratification panel.
7. Freeze the rule, population, support settings and numeric thresholds
   before test evaluation. Apply the same rule separately to every declared
   model variant. Do not recompute percentiles or adapt groups to test labels.

This produces model-derived strata analogous to the index's categories, not
interchangeable categories with demonstrated equivalent absolute risk. Label
them as **model risk strata**, with their own colour key. Middle/high risks
are observed results, not built-in equivalence claims. A model whose observed
risks do not increase across its score groups must retain that finding. Do
not merge groups after seeing test outcomes to make the plots look orderly.

The original clinical scores still retain their native definitions and labels.
We are constructing an interpretation of the model score, not replacing or
redefining NCCN-IPI, CLL-IPI or another instrument.

Matching the event rates of every native category is a different, more
demanding proposal. Several noisy category-specific targets do not necessarily
admit one ordered partition of another score. Anchoring High to its event
risk rather than its specificity would at least use a risk-stratification
criterion consistently, but it needs a declared joint optimisation and can
still be infeasible. I would not add those constraints merely to make every
colour appear to have the index's meaning. The low-anchored candidate is
closer to the published precedent and leaves the other risks to be evaluated.

## Small samples: what this does and does not solve

This solves the logical crossing of independent boundaries. It does not create
more independent events. A native Low group with no observed deaths does not
establish a true risk of zero. A development point estimate below its target
is not proof of safety or noninferiority. Likewise, no observed deaths in the
model Low group can leave a broad confidence interval.

Before locking, bootstrap the **whole threshold-development procedure**:
resample development patients, re-estimate the index Low target, and refit
the thresholds. Report the fraction of resamples with no supported partition,
the variation in cutoffs and group sizes, and any group-risk inversions. This
requires no predictive model training. It measures development instability,
not independent predictive performance. If using cross-validation of the
threshold procedure within tuning, the neural model stays fixed, and this
still does not undo the earlier use of tuning for model selection.

For the test analysis, use frozen thresholds and paired patient bootstrap
intervals for achieved group risks, risk differences, patient counts and
classification metrics. Report calibration separately on raw predictions.
No recalibration or supervised model fitting is part of this proposal.

The desired sentence is an empirical result to establish, not a threshold
definition: the model identified X additional Low patients per 1,000, with
observed risk A versus B and uncertainty for the difference. Development
matching alone never justifies saying that evaluation risks are equal.

## What the audit needs to show

- For every index: native reference counts/events, model group counts/events,
  observed CIF or survival and intervals, cutoffs and development provenance.
- For every category boundary: ordinary known-status and censoring-weighted
  TP/FP/FN/TN, recall, precision, specificity, NPV, MCC, accuracy and F1. Keep
  undefined quantities undefined. Use the already implemented metrics.
- Separately: the independent specificity-matched binary operating point.
- A low-group size versus observed-risk curve on development data, with the
  selected cutoff marked. It should expose whether the selection sits in a
  noisy interval rather than relying only on an attractive survival plot.
- Failed or unstable classifications retained in the report. Continuous
  model-versus-index evaluation remains available even if stratification fails.

The actual new thresholds and group risks require the existing patient-level
predictions on the server. The downloaded aggregate CSVs cannot identify them.
No patient-level export or additional GPU allocation is needed for this audit.

## Implementation prepared for review

`opera/evaluation/clinical_risk_groups.py::fit_low_anchored_groups` implements
the candidate as an isolated helper. It searches all attainable Low boundaries,
checks that the tail can support the remaining categories, and finds the
closest supported tail proportions by dynamic programming. It returns group
counts, weighted risks, effective sample sizes and an inversion flag.

Call it on the positive-weight known-status subset returned by the existing
`clinical_weights` function. Pass `model_risk`, `label`, `ipi_group`, the
registered `group_order` and `ipcw_weight`. Keep early-censored patients out
of the threshold-fitting labels. Apply the returned numeric cutoffs to all
eligible evaluation patients, retaining the existing survival/CIF estimators.

It is deliberately not registered as a renderer default or inserted into the
active rerun. Its tests include exhaustive enumeration of small tied-score
samples, impossible risk matches, two- and four-category indices, weighted
risk constraints and misleading risk-ordering examples. This verifies the
algorithm, not the clinical adequacy of any particular server result.
