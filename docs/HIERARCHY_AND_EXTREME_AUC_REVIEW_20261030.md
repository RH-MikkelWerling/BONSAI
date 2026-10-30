# Hierarchy contrasts and extreme AUC gains — 30 October 2026

This review checks current source and the **28 October tuning download**.
It does not report outputs from the ongoing recovered run, national fits or
held-out patients. Supporting aggregate extracts are saved under
`scratch/code_audit_20261030/` as `extreme_auc_comparisons.csv` and
`extreme_auc_summary.json`.

## Which router fix the run should contain

The primary model is the revised R_AB fit, exposed as R_L, with a global
adapter outside the routed mixture and two router warm-up epochs. Previous
server replay verified its identity and active borrowed corrections. The
original collapsed router remains a sensitivity model. Recovery retains the
existing fit and completed stages. It now accepts synced source code by default,
recording the current code fingerprint. It does not refit a router.

A distinct evaluation-gradient fix changes the warm-up detachment condition
to `self.training and self.router_frozen`. It corrects integrated gradients,
not ordinary prediction values, AUC, arrows or source-removal inference.
A full run started with that source computes corrected IG. To certify this
particular recovered run, inspect its source/attribution receipts. Do not
infer deployment from a locally corrected file. See the
[implementation review](ROUTER_IMPLEMENTATION_RECHECK_20261029.md).

## Exact hierarchy contrasts

`analyse_locked_predictions.router_decomposition` defines these separately
trained seed-42 fits for the AB design:

| Display | AUC difference | What remains available |
|---|---|---|
| Own correction | RBH_AB_L minus N_A_L | Own-only retains shared encoder/head, global adapter and own disease/outcome corrections. |
| Borrowing | R_AB_L minus RBH_AB_L | Full R_AB retains own corrections and can additionally use other disease/outcome adapters. |
| Both | R_AB_L minus N_A_L | Full correction architecture versus shared encoder/head plus global adapter only. |

N_A_L is **not** the plain J_L joint model. RBH_AB_L is **not** a specialist
trained solely on one disease–outcome cell. Its shared path learns jointly.
Its own disease adapter learns across outcomes in that disease, and its own
outcome adapter learns across diseases for that outcome. Each source adapter
acts on the current patient's record representation rather than retrieving
another patient's record.

"Borrowing" in this figure is the *incremental performance contrast* from
permitting other-task adapters beyond that own-only control. It is not the
AUC of an isolated borrowed-only predictor and is not the contribution of
borrowed vectors held fixed in one checkpoint. Training, gates and shared
parameters can adapt differently in the two fits. No matched trained static
mixture control isolates the value of patient-dependent source selection.

Within a matched cell, the three differences telescope exactly. The downloaded
maximum numerical discrepancy is 1.25e-16. Across supported cell–horizons,
equal-cell mean differences are own +0.007386, borrowing +0.004529, and both
+0.011915. Medians in panel C do **not** have to add, even though the
underlying cellwise contrasts do.

Fixed-checkpoint disease-source mean removal answers another question. It
retains the outcome bank, global adapter and own disease while replacing
other-disease terms with tuning-reference means at unchanged gates. Its
"none" condition is not RBH_AB_L and does not remove all borrowing.

## The curves and what they can establish

`paper_figures._log_support_fit` fits ordinary least squares:

```text
AUC difference = a + b log10(target training events)
                   + c [log10(target training events)]²
```

There are no knots or spline penalty. The ribbon is the 2.5th–97.5th
percentile of **200 row-bootstrap fits**, seed 0. It is a descriptive curve
band, not a patient-bootstrap confidence interval for a clinical effect.
Rows share patients, repeated horizons and nested endpoints. Resampling them
independently does not account for this dependence or training-fit variation.

The default recipe supplies no horizon override. The figure therefore uses
all supported disease–outcome–horizon rows, with the training event count at
each row's horizon. It does not restrict panel B to one year or first average
within disease–outcome pairs. The current main-manuscript caption's one-year
description needs correcting. Points require at least ten training events
and an interval width no greater than 0.4, but **all finite eligible rows enter
the fits**. This display filter also needs to be explicit.

In the old AB download, the quadratic curves cross near 70 training events
and again near 3,627. The code labels only the first grid crossing. These are
features of a descriptive quadratic, not estimated universal sample-size
thresholds. Remove the precise automatic crossover annotation before sign-off
or replace it only with a separately justified analysis. Do not present the
row-bootstrap ribbon as confirmatory uncertainty.

The broader pattern is also present in direct summaries:

| Target training events | Cell–horizon rows | Mean own contrast | Mean borrowing contrast |
|---|---:|---:|---:|
| <20 | 141 | −0.0194 | +0.0222 |
| 20 to <100 | 1,084 | +0.0001 | +0.0079 |
| 100 to <500 | 1,265 | +0.0152 | +0.0019 |
| ≥500 | 514 | +0.0110 | −0.0011 |

This is compatible with more reliable task-specific fitting when data are
abundant and greater usefulness of other-task corrections when data are
scarce. It does not establish that explanation causally. Different diseases,
outcomes, horizons and event prevalences populate the bands. The x-axis is
also not the exact support count used by each disease/outcome adapter, whose
supervision spans multiple tasks. Panel C's median/IQR describes heterogeneity
across these rows, not uncertainty in their pooled mean.

## The +0.8 AUC gains

The old primary comparison contains 3,004 supported disease–outcome–horizon
rows. Joint versus specialist has **eight** gains of at least +0.80, spanning
five disease–outcome pairs, all TCL. OPERA versus specialist also has eight
such rows. Disease-pooled versus specialist has ten, in TCL and TRANSFORMED_FL.
Floating-point equality at 0.80 is counted with tolerance 1e-12. The smaller
class has only five to seven patients in every one of these extreme rows.

Examples:

| Disease / outcome / horizon | Cases / controls | Training events | Specialist AUC | Joint AUC | Difference |
|---|---:|---:|---:|---:|---:|
| TCL / anaemia G3+ / 60 days | 6 / 8 | 8 | 0.125 | 1.000 | +0.875 |
| TCL / anaemia G3+ / 365 days | 7 / 7 | 23 | 0.143 | 1.000 | +0.857 |
| TCL / thrombocytopenia G3+ / 365 days | 5 / 9 | 20 | 0.156 | 1.000 | +0.844 |

These are absolute AUC differences. They are not routing gains. For the
revised router, the largest full-versus-own-only cellwise difference in this
download is +0.267, and the equal-cell mean is +0.004529.

Requiring at least ten evaluation cases **and** ten controls removes every
joint-minus-specialist gain of at least +0.70:

| Evaluation minimum in each class | Rows | Mean joint minus specialist | Largest gain |
|---|---:|---:|---:|
| 5 | 3,004 | +0.1185 | +0.875 |
| 10 | 1,802 | +0.0777 | +0.583 |
| 20 | 1,131 | +0.0629 | +0.508 |
| 50 | 451 | +0.0185 | +0.261 |

These cutoffs change the task roster. They do not estimate what would happen
if the same models received more training data. A larger test sample should
improve evaluation precision, but does not repair an unstable trained
specialist or guarantee that extreme differences disappear.

The canonical paired evaluator requires identical patients, labels, release
contracts and IPCW weights, and checks follow-up parity. The comparison is
designed to change supervised sharing scope while keeping the evaluation
population fixed. The factorial builder also uses the same pretrained
initialisation, architecture and checkpoint-selection policy. Its budget is
equal epochs, **not** equal optimiser steps or compute. Additional supervised
data therefore also change training exposure. This estimates the benefit of
sharing under the specified training recipe rather than a compute-matched
algorithmic advantage. That makes a large gain possible without an unfair test
denominator. It does not make five cases/five controls a precise estimate,
nor establish superiority over strong practical alternatives. Conditional
bootstrap intervals do not quantify training instability or resolve sparse
evaluation by themselves.

One finding deserves an explicit targeted server check: TCL specialists are
below chance in 78 of 103 supported rows, with median AUC 0.36. The seven
TRANSFORMED_FL rows have median specialist AUC 0.04. These correlated rows
cannot be treated as independent replications. Small-data fitting can create
systematically poor rankings, but it would be premature to attribute all of
this to sampling noise. Inspect those existing fits' cohort/endpoint identity,
checkpoint selection, learning histories, score direction and saved-versus-live
predictions before final sign-off. Do not reverse predictions or exclude cells
to improve the result. The previous server replay checked J_L and R_L, not
these extreme specialists.

Keep absolute performance and the strong tabular comparisons visible. The
saved OPERA-minus-XGBoost mean is −0.0280 AUC. Pooling's benefit over a weak
neural specialist does not establish an advantage over XGBoost.
