# Clinical risk groups: training-derived cutoffs

**30 October review:** this describes the implemented rule, not a resolved
final protocol. See [the design reconsideration](CLINICAL_RISK_GROUP_DESIGN_RECONSIDERATION_20261030.md)
for the distinction between independent clinical operating points and one
ordered classification, and the limitations of apparent training predictions.
The running recipe has not been silently changed.

Implemented 28 October 2026 following the agreed training-development approach.
The clinical renderer defaults to threshold_split: train. The completion workflow
exports all training predictions from the selected frozen checkpoint, registers
these under clinical_threshold_artifacts (not the main evaluation inventory),
and generates score groups before evaluation on tuning or held-out patients.
No model retraining is required.

The low-risk cutoff selects the largest candidate group (minimum 20 known-status
patients) with IPCW event risk no higher than the index's low-risk training
risk. The high-risk cutoff approximately matches the index's censoring-weighted
training specificity. Intermediate groups follow the index's relative middle
group sizes. Cutoffs are fixed before evaluation, using >= for positive model
classification. Independently selected anchors that cross, coincident middle
cutoffs, empty model groups and missing anchor groups are reported as not
estimable under this rule. A crossed pair is not proof that no constrained
joint solution exists. No size-matched or constrained-optimisation fallback
is silently substituted. Thresholds retain full numerical precision.

For two-category indices, a partition has only one cutoff. If the independently
chosen low/high anchors differ, they are evaluated separately, rather than
constructing an invalid extra model category. `classification_at_anchors.csv`
contains both binary comparisons when fitted, even if the joint risk-group
figure is unavailable. `cutoff_diagnostics.csv` identifies the actual failure
and `score_population_flow.csv` accounts for scores by disease and split.
Expected matching failures have status `not_estimable`. They leave a documented
gap without terminating other valid score jobs as a runtime error.

The predictions used for development are apparent training predictions. They
can yield optimistic matching targets, so matching is not presumed to hold in
new patients. Tuning results remain exploratory because tuning informed model
selection. Out-of-fold training predictions are an optional future sensitivity,
not the implemented primary procedure.

The 29 October report demonstrates imperfect transport of the matching
targets. More model low-risk patients do not imply matching observed mortality
on tuning. Report achieved risk and specificity for both systems alongside
the training targets. See `CLINICAL_POPULATION_AND_THRESHOLDS_REVIEW_20261029.md`.

classification_at_thresholds.csv reports both systems at every non-low boundary.
It contains threshold provenance, TP/FP/FN/TN, recall/sensitivity, precision/PPV,
specificity, NPV, MCC, accuracy, F1 and flagged fraction. Known-status counts
exclude early censoring and count competing death as negative. cif_ipcw rows
use disease-specific censoring weights and contain weighted counts. Undefined
metrics are NaN. These are point estimates, not bootstrap confidence intervals.

Use a new completion workspace after syncing changes. Missing training exports
block clinical rendering rather than using tuning data as a silent fallback.
report_manifest.json hashes development and evaluation prediction inputs, metadata
and recipes. Raw prediction exports stay within the secure environment.
