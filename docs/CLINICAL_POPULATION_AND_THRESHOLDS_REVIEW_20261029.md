# Clinical population and threshold review — 29 October 2026

## Inputs and scope

Reviewed the new `scratch/score_inventory.csv`, `scratch/headline_scores.csv`
and `scratch/config_audit.json` against the 28 October downloaded review.
Copies, SHA-256 receipts and numeric comparisons are preserved in
`scratch/paper_reading_20261028/clinical_reconciliation_20261029/`.
These are tuning results. No patient-level data or new server predictions
were available locally.

## The NCCN-IPI denominator changed deliberately, but was not made clear enough

| Quantity | Previous review | New review |
|---|---:|---:|
| Comparison population | All subtypes with computed components | DLBCL |
| Patients with model predictions in that population | 1,973 | 476 |
| Patients with the required score | 909 | 343 |
| Score missing in the comparison population | 1,064 | 133 |
| Group scale | Three coarsened categories | Four native categories |
| Model cutoff-development split | Tuning | Training |
| Model low-risk cutoff | 0.068939 | 0.015246 |
| Model high-risk cutoff | 0.305845 | 0.355328 |

Historical recipe `108a837:opera/run/prepare_available_plot_configs.py`
used `cohorts='all'` for both computed IPI and NCCN-IPI. The current NCCN-IPI
recipe uses `cohorts=['DLBCL']`. The broad IPI comparison still has exactly
1,973 eligible and 909 scored patients in the new report. The evidence points
to excluding other subtypes from the NCCN-IPI comparison, not general score
loss. This accounts for 566 fewer scored patients in that comparison.
The aggregate reports cannot establish patient-by-patient identity.

The original NCCN-IPI was developed for DLBCL
([Zhou et al., Blood 2014](https://pubmed.ncbi.nlm.nih.gov/24264230/)).
Returning to the broader population to recover a larger sample would change
the clinical applicability question. Keep the DLBCL main comparison and
label its denominator explicitly.

Restoring four groups only splits the former intermediate category. The
native low category is still 0–1. It does not by itself explain the loss of
low-risk patients. From the implemented age points, a patient older than 60
at diagnosis already has at least two NCCN-IPI points. An empty native low
category at older diagnosis ages is consistent with that score. Plot age is
age at treatment, which can differ from age at diagnosis.

## Low-risk expansion is much smaller in the current DLBCL comparison

The old broad-population headline implied 409 model low-risk patients versus
50 for NCCN-IPI. The new DLBCL comparison has **26 versus 24**. Both new groups
have one observed death by one year, giving risks of 3.85% and 4.17%,
respectively. This is two additional patients, or 8.3% more than the native
low-risk group, with considerable uncertainty. It is not the previously
suggested large expansion.

The training population for this threshold comparison has 7,008 scored DLBCL
patients. The native low group's training IPCW risk is 0.7619%, which anchors
the new, lower model threshold. Moving threshold fitting from tuning to
training changes the operating point. The target risk must not be presumed
to transport unchanged to tuning or test patients.

## Matching on training does not imply matching on tuning

The following are ordinary known-status tuning results reconstructed from
the uploaded aggregate fractions, not censoring-adjusted group estimates.

| Index | Index low n | Model low n | Index low mortality | Model low mortality |
|---|---:|---:|---:|---:|
| NCCN-IPI, DLBCL | 24 | 26 | 1/24 (4.17%) | 1/26 (3.85%) |
| IPI, broad scored population | 214 | 595 | 5/214 (2.34%) | 38/595 (6.39%) |
| FLIPI2 | 14 | 47 | 0/14 | 2/47 (4.26%) |
| R-IPI | 34 | 81 | 1/34 (2.94%) | 1/81 (1.23%) |
| R-ISS | 60 | 244 | 1/60 (1.67%) | 13/244 (5.33%) |

For NCCN-IPI high-risk classification, the index identifies 33 of 59 deaths
and the model 31 of 59. Specificity is 82.4% versus 87.0%. These are two
different tuning operating points, although the model threshold was chosen
to match the index's specificity on training. FLIPI2 shows a particularly
large specificity difference, 83.3% versus 57.8%.

Do not describe these as additional patients at matching *evaluation* risks
or specificity. Report the frozen training targets and both achieved tuning
quantities. Whether to add a separate evaluation-standardised ROC comparison
or use out-of-fold development predictions is a protocol decision to make
before opening test results. Neither should be silently substituted because
the current results are less favourable.

## Five risk-group jobs failed, not five model fits

CLL-IPI, IPS, IPSSWM, MALT-IPI and MIPI failed the ordered cutoff check. Their
threshold-development samples contain 760–1,843 training patients. The
failure cannot simply be attributed to too few tuning patients. The current
report does not retain enough detail to distinguish all underlying cases.

Confirmed implementation and interpretation issues:

1. The low-risk and high-specificity thresholds are selected independently.
   If the largest feasible low cutoff is above the high cutoff, they do not
   define a disjoint ordered partition. Each binary operating point can
   still be meaningful. This is not proof that no constrained joint solution
   exists. New diagnostics count feasible low candidates below the high
   threshold, without silently changing the declared matching rule.
2. IPS has only two recorded categories in the supplied audit. Two categories
   require one partition cutoff. The old routine constructed two anchor
   cutoffs regardless of category count. It now explicitly identifies this
   conflict, allows a shared cutoff if the anchors coincide, and otherwise
   exports the two binary comparisons separately.
3. Rounding cutoffs to six decimals before evaluation could manufacture ties
   or move patients between groups. Thresholds now retain full precision.
4. Historical code explicitly forced cutoffs to increase by `1e-6`. The old
   CLL-IPI, IPS, MALT-IPI and MIPI inventories contain precisely that spacing.
   Those old figures are not evidence that the new constraints were met.
   This artificial adjustment must remain removed.

## Code changes and next check

- `cutoff_diagnostics.csv` records anchor values, index low risk, target and
  achieved training specificity, candidate counts and distinct failure codes.
- `classification_at_anchors.csv` retains recall, precision, specificity,
  MCC and confusion counts for each fitted binary anchor even when the
  combined risk-group partition is unavailable. It distinguishes ordinary
  known-status and IPCW estimates and does not label crossed anchors as a
  valid partition.
- `score_population_flow.csv` records scored and unscored counts by actual
  disease before the recipe's disease restriction, separately for training
  and evaluation. It contains no patient identifiers.
- Expected cutoff failures are labelled `not_estimable`, rather than ending
  the entire renderer as a runtime failure. Completion still records gaps.
  Unexpected data/configuration errors remain blocked.
- The three new aggregate tables are included by review assembly and the
  existing review packager. No model is refit and no prediction file changes.

Run one fresh CPU clinical refresh after syncing the updated code and
checksum manifest. Reuse the original completed training-prediction workspace.
Return the three new aggregate tables. The reports already received settle
the denominator change and current observed counts. The new diagnostics are
needed to assign an exact cause to each remaining ordered-group failure.

Validation: 48 targeted tests passed, including real native-export rendering,
crossed-anchor binary evaluation, the two-category case, preservation of
small distinct thresholds, population-flow accounting, completion, assembly
and packaging. The server diagnostics remain to be checked on actual data.

## Additional diagnostic snapshots received before the full rerun

The population-flow screenshot confirms the DLBCL restriction directly:
the DLBCL tuning row has `in_comparison_population=True`, 476 predicted,
343 scored and 133 unavailable. Other disease rows are marked False while
their computed scores remain present. This is exclusion from this comparison,
not deletion of those scores. The DLBCL training row has 7,008 scored patients.

The independent-anchor screenshot shows crossed anchors in at least:

| Index | Low-risk anchor, approximately | High-specificity anchor, approximately |
|---|---:|---:|
| CLL-IPI | 0.5737 | 0.3175 |
| IPS | 0.0791 | 0.0187 |
| IPSSWM | 0.3423 | 0.0786 |

These are independently selected binary thresholds. A patient can fall below
the first and above the second. They cannot be interpreted as disjoint low
and high categories of a single ordered partition. This observation alone
does not establish whether a constrained joint solution exists, why these
training optima arose, or whether the model's ranking is poor. The exact
cutoff diagnostics remain useful for those questions.

NCCN-IPI high-boundary confusion counts in the screenshot are internally
consistent with the uploaded headline table:

| System | TP | FP | FN | TN | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Model | 31 | 37 | 28 | 247 | 52.5% | 87.0% |
| NCCN-IPI | 33 | 50 | 26 | 234 | 55.9% | 82.4% |

The model flags fewer false positives and also misses two more deaths. This
is not a comparison at equal achieved tuning specificity. Identical
known-status and IPCW rows are consistent with all patients in these displayed
score populations having known one-year status. These are two estimands of
the same patients, not duplicate patients to sum together.

The displayed clinical threshold values 1, 2 and 3 are explicitly labelled
zero-based group ranks. For native NCCN-IPI groups they mean scores at least
2, 4 and 6, respectively. They are not a substitution of a 0–3 numeric NCCN-IPI
score. The high-specificity anchor and low-risk anchor remain separate
operating points in the newly added export.

A fresh downstream rerun can complete the figures and diagnostics. Repeating
the fixed inputs and matching rule is expected to reproduce these cohort and
transport effects. The eventual threshold protocol and claims still need
review before held-out evaluation.
