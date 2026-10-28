# Clinical risk groups: training-derived cutoffs

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
classification. Crossed, coincident or empty-group thresholds and missing anchor
groups are reported as not estimable. No size-matched fallback is substituted.

The predictions used for development are apparent training predictions. They
can yield optimistic matching targets, so matching is not presumed to hold in
new patients. Tuning results remain exploratory because tuning informed model
selection. Out-of-fold training predictions are an optional future sensitivity,
not the implemented primary procedure.

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
