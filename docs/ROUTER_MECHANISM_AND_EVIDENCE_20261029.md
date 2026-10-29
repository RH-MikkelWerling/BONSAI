# What the routers implement and what the results establish

29 October 2026. This review uses the current source, the preserved server
replay report and the downloaded tuning aggregates. The fitted checkpoints
and patient-level arrays remain on the server. No training, test inspection
or change to prediction/runtime code was made for this review.

## Assessment

The reviewed prediction path is internally coherent, and sampled server
replay strongly supports the intended revised fit being used. The confirmed
evaluation-gradient bug affects IG and has already been fixed. The present
evidence favors a working correction architecture with diffuse allocation,
modest aggregate predictive gains and weak identification of particular
disease-source effects. It does not establish that patient-specific selective
routing causes those gains, or that the global adapter dominates them.

There are substantial limits in what the diagnostic labels suggest. In
particular, “healthy routing” means passing a non-collapse rule. It does not
mean useful, selective, stable-across-seeds or biologically interpretable
transfer. A new production-function counterexample passes the recorded health
rule with every adapter correction exactly zero.

## The actual prediction, in order

The encoder first creates a patient representation `h` from the pretreatment
record. The revised primary fit is R_AB, exposed as R_L/OPERA in the paper
inventory. It has 24 disease adapters and 69 outcome adapters, a global
adapter, interacting queries, rank 8 and 5% own-task gate floors. The global
adapter and two training warm-up epochs distinguish it from the original
collapsed router. R_AB does not have own adapters at full strength outside
the mixture. That is the separate R_O design.

Each source adapter is a small nonlinear function of this same patient's
representation. It is not retrieving another patient's record. In the
anchored design, disease adapter parameters receive supervised gradients
from patients of that disease across outcomes. Outcome adapter parameters
receive supervised gradients from that outcome across diseases. Borrowing
targets can update their gates and shared representation, but do not directly
update the borrowed adapter parameters. This statement concerns supervised
gradient ownership, not an absence of ordinary optimizer momentum/decay.
The shared encoder/head context has already learned across tasks.

The query uses the patient representation, known fine-disease identity and
the identity of the target outcome. The target's observed future label is
not a query input. Interacting queries permit the source preference to depend
jointly on those inputs. Permission to learn an interaction is not evidence
that a large interaction was learned.

Schematically, the primary readout is:

```text
adapted(h, outcome) = h + scale × [global(h)
                                 + sum_source gate(h, disease, outcome, source)
                                              × shrunk_adapter_source(h)]
```

There is one softmax over the 93 sources and one zero-correction slot.
The own disease and own outcome receive reserved floors when supported.
The global correction sits outside that softmax. The patient vector `h`
is always present at coefficient one. A small “shared gate weight” therefore
does not mean the model has stopped using the shared encoder. It measures
allocation to the zero-correction option.

An outcome-specific linear head maps the adapted vector to event and
competing-event log hazards across intervals. Integration gives cumulative
incidence at each horizon. Identical hidden-vector changes can have opposite
effects under different outcome heads. Cumulative-incidence changes do not
add linearly even when log-hazard contributions do.

## What broad allocation looks like without learning

Using the actual readout, flatten all 94 logits, retain the same bank counts,
set all source support multipliers to one and retain the two 5% own floors.
This gives an analytic comparator, not a newly trained model:

| Gate allocation | Flat-logit comparator | Recorded revised router average |
|---|---:|---:|
| Own disease | 5.96% | 6.22% |
| Own outcome | 5.96% | 6.04% |
| Other diseases | 22.02% | 17.36% |
| Other outcomes | 65.11% | 69.62% |
| Zero-correction slot | 0.96% | 0.75% |

The outcome bank receives 71.06% in total before learning because it has
69 members, compared with 24 disease members. The recorded outcome-bank total
is about 75.66%. More outcome mass is not, by itself, evidence that outcome
sources are more informative. The recorded aggregate masses are relatively
close to this flat-logit comparator. They cannot establish how individual
patients' gates vary. Actual support shrinkage also differs across sources.

The flat-logit comparator has an effective named-source count of about 83.
The revised fit's health statistic has median 76.018; its separate overall
effective-source summary is 70.803. These have different aggregation
conventions and must not be treated as identical statistics. Both revised
summaries support a broad mixture rather than a few sharply selected sources.

Reproduction: `scratch/paper_reading_20261028/router_mechanism_audit_20261029/allocation_baseline.json`.

## A health pass can coexist with no correction at all

The new counterexample uses the real readout and `routing_health` function.
Every source up-projection and the global up-projection is zero. Its adapted
vectors equal the shared vector exactly. Four equally represented diseases
have borrowed-source logit preferences of only 0.001, with no patient-dependent
or outcome-dependent learned query variation. Own floors remain active.

Nevertheless, the health rule passes. The most frequent top source has share
25%, four different borrowed sources win across diseases, and the effective
source count exceeds 1.5. The tiny preferences determine the winning names,
although none of these sources changes a prediction. Thus a pass is a useful
check against one-source allocation collapse, not evidence of meaningful
contribution or selectivity. The stage-3 rule uses gate × support shrinkage,
not actual patient-specific vectors or risk effects. The later stage-4 norm
proxy is closer to contribution magnitude, but still omits vector direction,
cancellation and alignment with the outcome head.

This is a diagnostic limitation, not a demonstration that the fitted R_AB
corrections are zero. Server replay establishes that they are active.

## Why the arrows and source maps can look disappointing

The arrows include the global term and the complete source mixture. A global
term is shared exactly across outcomes. Similar weighted mixtures of reused
source functions can also be shared in practice. The two-dimensional
projection and local averaging can conceal further differences. No arrow is
defined as a path toward a high-risk region. Predicted-risk maps are the
appropriate companion; observed event markers are noisy evidence of risk.

A new test demonstrates identical global displacements producing opposite
logit changes through two heads. Another demonstrates that nonzero, equally
weighted adapter vectors can cancel exactly. Gate weight, vector norm and
predictive effect are separate quantities. Neither similar arrows nor a large
vector norm proves that the global adapter determines predictions.

The primary source-removal map asks a narrower question. It replaces
other-disease terms by tuning means, while retaining the own disease,
the entire outcome bank, the global correction and the shared encoder.
That removes the disease sources' patient-specific variation while retaining
their average log-hazard contribution. It does not remove all cross-task
learning or all corrections.

There are also two different shrinkage operations:

1. Training-support shrinkage scales adapter outputs inside the predictor.
   Disease event counts sum across outcomes, while an outcome's count concerns
   that outcome. Nested definitions can count a patient repeatedly. These
   are support proxies, not directly comparable effective sample sizes or
   posterior uncertainty estimates.
2. Post-hoc empirical shrinkage pools noisy disease-pair effect estimates
   toward zero for the map. It never changes the fitted model's predictions.

The downloaded all-outcome map's tiny posterior means reproduce under current
code to approximately 2e-19 AUC points. Its maximum absolute raw effect is
3.11 AUC points, while the maximum absolute shrunken effect is 0.0000651
AUC points. This is not a factor-of-100 plotting error. Most individual pair
estimates are weak relative to their uncertainty. It is not proof that every
true pair effect is zero. Pair estimates share patients and are dependent.
The displayed shrunken intervals use posterior mean ±1.96 posterior SD,
rather than exact mixture quantiles. Treat them as approximate descriptive
intervals, not a source-discovery guarantee.

Diffuse contributions, similar source functions, cancellation, prediction
head null directions and retained outcome/global terms can all weaken an
individual disease-removal effect. Which dominates in the fitted model is
not yet identified by the aggregate download.

## What the comparative results actually support

Saved tuning summaries over 3,004 supported disease–outcome–horizon entries,
with equal-cell averaging, are:

| Contrast | AUC difference | Saved 95% interval |
|---|---:|---:|
| Revised router minus joint | +0.00868 | +0.00313 to +0.01458 |
| Revised router minus trained own-only control | +0.00453 | +0.00041 to +0.00888 |
| Revised router minus trained global-only control | +0.01191 | +0.00672 to +0.01759 |
| Original router minus revised router | +0.00082 | −0.00208 to +0.00372 |

These are conditional on the selected fitted models. Correlated horizons are
not independent experiments. The original-minus-revised equal-cell IPA contrast
is −0.00425, with saved interval −0.01047 to −0.00082, favoring the revised fit
for probability accuracy in that summary. The original has not established a
clear general performance advantage. Neither crossing-zero AUC intervals nor
one repeat fit establish formal equivalence or a full training-noise distribution.

Full versus joint combines adapters, extra conditional capacity, global
correction and differences in fitting. Full versus trained own-only supports
the architecture that permits borrowing, but also changes the softmax
competition and home weights during training. It does not isolate a single
source, pure information transfer or the value of patient-dependent gates.
The source-removal experiment instead measures reliance of one fixed fit,
conditional on its existing shared and retained correction paths.

No matched trained uniform-mixture or patient-independent gate comparator is
in the reviewed bundle. We cannot claim that selective routing is necessary
for the observed gain. A post-hoc gate substitution would only test reliance
of this trained fit; it is not a substitute for training that comparator.
That limitation can be stated without launching another architecture search.

## Implementation findings and remaining empirical checks

No additional prediction-path bug was found in this review. Fine-disease
ordering, outcome selection, gradient ownership, fixed-term source removal,
training/export parity and reconstruction have direct tests. Server replay
used the intended R_AB checkpoint and reproduced sampled probabilities within
tolerance. The known checkpoint warm-up bug concerned evaluation input
gradients and requires corrected IG, as already documented.

Warm-up deserves precise wording: detaching gate logits freezes the gate
parameter gradients. It does not hold the numeric allocation constant if
the patient representation changes. A new test demonstrates that property.
The readout's older “starting distribution” wording is stronger than the
implementation warrants. This is a specification clarification; no model
behavior was changed here.

The remaining high-value checks use existing fitted artifacts:

- Ordered source labels and second-seed health. A stable entropy does not
  establish identical source rankings across seeds.
- Raw outcome-specific correction vectors before projection. The overnight
  vector audit is already configured for this.
- Exact global and source-mixture vectors from the same frozen checkpoint,
  their head-projected log-hazard terms, and their effects on horizon risks.
  The current raw-vector audit alone does not isolate the global adapter.
  These calculations can reuse saved encoder vectors without re-encoding
  medical records. Keep them separate from a retrained global-only control.
- Patient/target variation relative to flat, population-average and
  disease-average gates, using actual correction/head effects as well as
  allocation. Similarity of averages alone cannot resolve this.

None requires a more visually appealing router as an endpoint. If the checks
confirm broad sharing and weak individual-source identification, the proper
claim is that shared learning with task-specific corrections improves some
predictions, while evidence for selective disease-pair borrowing is limited.
The earlier disease-course and cardiovascular anomalies are also strongly
affected by sparse supported cells and repeated nested outcomes. The original
router does not restore the remembered domain-wide disease-course gain in
the current tuning population.

## Verification and effect on tonight's run

Five new interpretation tests cover bank-size allocation, a zero-correction
health pass, warm-up allocation changes, opposing head effects and vector
cancellation. The targeted suite passed 94 tests, with four CUDA-only cases
skipped locally. Its result is preserved in
`scratch/paper_reading_20261028/router_mechanism_audit_20261029/tests.xml`.
No runtime, configuration or checksum-manifest changes were needed. Tonight's
previously supplied command remains valid; this review does not add another
required GPU job or reopen the model-selection policy.
