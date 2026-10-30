# Explaining the router's gain over joint learning

30 October 2026. National pretraining is at epoch 7 of 10 and the recovered
tuning run is active, as reported by the user. The calculations below use
the **28 October downloaded tuning aggregates**, not that active run or
national/held-out results. No active-run code was changed for this review.

**Implementation added:** the separate CPU readout audit is now available.
See [server command and outputs](ROUTER_MECHANISM_SERVER_AUDIT_20261030.md).
It implements the component and gate substitutions below without changing
the active rehearsal. Its real-server results are still pending.

## What we can already distinguish

I reconstructed the control ladder directly from the physical model names in
`prediction_analysis/estimates.csv`, rather than assuming the OPERA alias.
The four models have exactly the same 3,004 supported cell–horizon rows.
Each row is one disease, outcome and horizon. Equal-row averaging gives:

| Separately trained model | Mean AUC | Difference from joint |
|---|---:|---:|
| Joint, J_L | 0.72445 | Reference |
| Global correction only, N_A_L | 0.72122 | −0.00324 |
| Global plus own-task corrections, RBH_AB_L | 0.72860 | +0.00415 |
| Full revised OPERA, R_AB_L | 0.73313 | +0.00868 |

At one year on the same 600 cells, the corresponding differences from joint
are −0.00249, +0.00561 and +0.01079. These are descriptive point estimates.
No new confidence interval was calculated for global-only versus joint or
own-only versus joint. The primary full-versus-joint saved paired interval
is +0.00313 to +0.01458 across all supported horizons.

The existing sequential contrasts are also consistent:

```text
global-only − joint         −0.003236
own-only − global-only      +0.007386
full − own-only             +0.004529
full − joint                +0.008679
```

The differences telescope exactly within each cell. This arithmetic does not
make them a causal allocation of the full improvement to three components.
The models are separately trained and their encoders, heads, global functions
and gates can adapt differently. In particular, no percentage of the AUC gain
should be called "explained by borrowing" from this ladder alone.

Nevertheless, global correction alone does not reproduce the full model's
observed gain in these fits. Own-task corrections improve the mean comparison,
and permitting other-task adapters improves it further. This is much more
specific than "the router is healthy" and makes a testable interpretation of
the architecture possible. It does not rule out extra nonlinear capacity as
part of the explanation.

Reproducible aggregate outputs:
`scratch/paper_reading_20261028/router_mechanism_audit_20261030/`
contains `control_ladder_cells.csv`, `control_ladder_summary.csv` and
`receipt.json`. These contain no patient-level records. The all-horizon
absolute means differ from the one-year headline because their rosters and
horizons differ.

## Why diffuse mixing can still help

For a given patient, an adapter computes a small nonlinear transformation of
that patient's record representation. It does not retrieve another patient's
record or substitute the average outcome rate of another disease. Its learned
parameters carry supervision from its source disease or source outcome.

The joint readout applies an outcome-specific linear head to its learned
patient representation. OPERA inserts a global nonlinear correction and a
weighted combination of source-specific nonlinear corrections before that
head. Even nearly constant mixture weights can produce useful **patient-varying
corrections**, because the adapter functions themselves depend on the patient.
A broad mixture can act as a collection of supervised correction features
without selecting one dominant donor for each task.

Own-only is already a shared model. Its disease adapter is learned across
outcomes within that disease, its outcome adapter across diseases for that
outcome, and its encoder/head context jointly. The full-minus-own contrast
concerns additional other-task adapter access, not all information sharing.

Three quantities must remain separate:

1. Allocation: the weights assigned to sources.
2. Correction: the weighted vectors after source-support shrinkage.
3. Predictive usefulness: whether the resulting scores improve ranking or
   probability accuracy for the target.

Gate entropy answers the first question. Arrow length and vector norms are
closer to the second. Neither establishes the third. Identical displacements
can have different effects under different outcome heads, and nonzero source
vectors can cancel. These properties already have implementation tests.

## The remaining empirical questions

We need to distinguish learned correction functions from patient-specific
allocation, and source-bank reliance from their combined vector magnitude.
The active run's raw-vector report helps separate genuine outcome similarity
from projection/averaging. Its disease-source mean/zero removals quantify a
narrower intervention. Neither isolates the global adapter, removes the
entire outcome bank, or tests the need for patient-varying gates.

The existing replay's 46 patients establish active paths and numerical
reproduction. They are not a sufficient performance sample for a component
benefit analysis. Use the full available tuning roster for that analysis,
or a selection fixed without looking at labels and described as a diagnostic.

## Smallest useful follow-up after the active run

This is a frozen-checkpoint readout analysis, not an architecture search.
Saved pooled encoder vectors can avoid another medical-record encoder pass.
The existing routed-vector exports contain those vectors for their exported
patient roster. Disease identity, outcomes and follow-up come from the same
frozen evaluation inputs. Coverage must be reported if this roster differs
from the complete prediction population.

Before interventions, replay the full readout/head from the saved vectors and
compare its CIFs with the original saved predictions. Check ordered sources,
patient alignment and the selected checkpoint. This is a direct numerical
reconstruction check, not a blanket source-freshness requirement.

The implementation also compares the router checkpoint's encoder/head-only
and global-only paths directly with J_L using paired patient uncertainty.
This can distinguish an advantage already present in the router-trained shared
path from an advantage that appears when its correction terms are retained.
It remains a counterfactual reliance comparison rather than a causal partition
of the training process.

### A. Identify which components change useful predictions

At fixed encoder vectors, source functions, head and original gates, separate:

- Global correction.
- Own disease and own outcome corrections.
- Other-disease corrections.
- Other-outcome corrections.

Project each through both target-event and competing-event head coefficients
for every interval. Their terms add exactly in the **unclamped head outputs**.
The subsequent hazard clipping, exponentiation and cumulative-incidence
integration are nonlinear. Check full reconstruction before attributing terms.

For component removal, retain the original gates and all other terms. Evaluate
the full model and each removal on the same patients, including patient-level
risk changes, AUC, Brier/IPA and paired uncertainty. Use tuning-reference mean
replacement as well as zero replacement to assess sensitivity to removing
the component's average shift. Aggregate effects on the fixed paper roster
and show support/domain summaries. Do not add the risk/AUC changes as if they
were additive contributions.

This distinguishes a global correction explanation from own-task and borrowed
bank reliance. It also covers the outcome bank, which the current primary
disease-removal maps retain. A large vector that does not affect relevant
head outputs or predictive accuracy is not a useful component merely because
its norm is large.

### B. Test what patient-dependent gates add in this trained fit

Use two transparent diagnostic substitutions, with reference weights estimated
on tuning patients without their outcome labels:

1. Replace the complete gate vector with the mean gate vector for that disease
   and target outcome. This preserves task-dependent allocation while removing
   allocation differences between patients within that task. Adapter outputs
   remain patient-dependent and the global correction is unchanged.
2. Freeze only the **relative composition among borrowed sources** to its
   within-task reference mean, retaining each patient's original total borrowed
   mass and own/null weights. This separates source choice from how much the
   model borrows. Define zero-borrowed-mass handling explicitly.

Report risk changes and paired accuracy contrasts, not just distances between
weight vectors. Neither substitution should renormalise a removed component
into the surviving original terms. These substitutions are exploratory
within-fit reliance tests, not separately trained static-gate comparators.
Any later held-out use must retain tuning-fitted reference weights and a
predeclared missing-task fallback.

If the substituted weights preserve performance, the selected fit may be
using a useful broad mixture of task-trained functions with little need for
individual source selection. If performance worsens, that supports reliance
on its learned allocation. It still does not establish that a model trained
from the outset with static gates could not perform equally well.

### C. Explain representative successes and failures

Use the existing tuning-only gain/neutral/loss outcome-selection rule rather
than selecting attractive arrows. For each probe, show which component changes
the predicted risk, whether the change improves ranking/probability accuracy,
and how it varies with disease and training support. Clinical interpretation
of a named source requires a detectable effect and correct source identity.
Broad mixture benefit can coexist with weak individual disease-pair effects.

## What counts as sufficient understanding

Before calling the mechanism explained, we should be able to state:

1. Which trained controls reproduce or lose the advantage over joint learning.
2. Which terms in the selected fit actually alter its head outputs and risk,
   and whether their removal worsens accuracy on matched patients.
3. Whether patient-specific allocation matters in that fit, or whether the
   evidence supports primarily a broad mixture of learned correction features.
4. Where the benefit occurs, where it fails and what uncertainty remains.

The answer does not have to be selective biological donor discovery. It does
have to go beyond a non-collapse pass and beyond attractive geometry.
If a claim that adaptive routing is necessary is retained, a matched trained
static-gate comparator remains necessary evidence. Otherwise the paper should
describe the demonstrated value of shared learning and task-specific
corrections, and report selective-routing evidence with its actual limits.

The active run can finish unchanged. The reconstructed control ladder is
already available. The frozen-checkpoint component and gate substitutions
are implemented in the separate audit, **not outputs already produced by
the full rehearsal**. Their results are not yet completed server verification.
