# LDH manuscript: morning review, 30 October 2026

Read the [editorial PDF](../paper/render/ldh_edition_20261029/opera_ldh_editorial.pdf)
or the [single-column review PDF](../paper/render/ldh_edition_20261029/opera_ldh_review.pdf).
Both use the same independently edited text. The original manuscript and thesis
remain unchanged. The edition directory retains its original 29 October name.

## This editing pass

The opening now places prognosis at treatment initiation, where disease control
and serious complications matter together. The introduction brings joint
learning, task-specific corrections, and pretraining into the same argument
about sharing information. Its three questions remain explicit.

Methods explain the supervised grid, the 52 loss-weighting families, the
correction controls, and the meaning of source removal. They retain the
training-derived clinical cutoffs, native score categories, common-patient
comparisons, censoring adjustment, and conditional bootstrap uncertainty.
Classification measures at the boundaries include recall, precision,
specificity, negative predictive value, Matthews correlation coefficient,
accuracy, and F1.

Results give estimates and counts. Interpretation now sits mainly in the
Discussion. Repeated administrative reminders have been collected into one
visibly unfinished results paragraph. All 15 numerical comparison macros were
retained, including the four-decimal positive interval for full versus own-only
corrections. No new performance estimates or confidence intervals were fitted.

The Discussion develops three distinctions:

- Disease sharing helps within the neural family, while XGBoost remains ahead
  in the available local comparison.
- Little average change from outcome sharing could support consolidation,
  but does not establish non-inferiority or rule out losses for particular tasks.
- Better performance of the correction architecture does not establish that
  selective, patient-dependent routing explains the gain.

The clinical discussion uses achieved risk and specificity. It does not convert
a larger low-risk group into a claim of equivalent prognosis. Caption wording
has been shortened, the Research in context separator occupies its own line,
and the clinical figure's explanatory note is separated from its footer.
Paragraphs are also protected against single orphaned lines at page breaks.

The main text is approximately 2,700 words, with a 246-word Summary and 27
references. This leaves room for the national, calibration, and final clinical
findings within the journal's main-text allowance. The evidence search and
author declarations remain explicitly unfinished.

## Review and release sequence

First settle the clinical framing and the interpretation of the modest
correction gains. Then reconcile the arriving rerun reports with the model
identities, paired metrics, clinical denominators and thresholds, source-removal
outputs, and refreshed attribution/input-restriction figures.

National pretraining needs its checkpoint and vocabulary/configuration receipts,
followed by the planned tuning comparisons. Freeze the resulting model choices
and evaluation rules before the reserved test evaluation. Weak routing effects
or a national model that does not improve performance are valid results.
Unresolved provenance or inconsistent estimates require correction.

The existing [router and test-release agenda](AGENDA_20261029_ROUTER_AND_TEST_RELEASE.md)
remains the operational guide. Tomorrow is a reasonable target for reviewing
the audits and deciding on release. It is not a reason to bypass unresolved
findings. Submission administration, such as completing author declarations,
can proceed separately from the scientific analysis lock.

The [subsequent code audit](CODE_RELEASE_AUDIT_20261030.md) found corrections
to the clinical threshold search. Replace the paper's provisional clinical
counts from a fresh CPU report before drawing clinical conclusions. The
existing fitted predictions can be reused.
