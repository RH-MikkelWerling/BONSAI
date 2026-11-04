# What the test results would mean (written 3 November 2026, before the test set is opened)

Companion to `docs/LOCK_CLOSURE_DRAFT_20261101.md`. What is fixed here is which
comparison, with which label, supports which claim; the wording of the
paragraphs is a draft and will be rewritten around the actual numbers. Nothing
here changes the lock: the primary model, contrasts and labels are the lock's.
The mechanism is reported from the test-set decompositions; the tuning values
below are orientation only. Test set, primary estimand (paired IPCW AUC per disease–outcome cell,
pooled over horizons, every evaluable cell equal), OPERA = the J_OA_OWN design, seed 42, with the encoder chosen by lock section 4b.

## Labels (from the analysis lock)

**Better**: 95% interval above 0. **Worse**: below 0. **Equivalent**: interval
within ±0.01. **Inconclusive**: otherwise. **Above margin**: better, with an
estimate above 0.01. Every statement below refers to the evaluable cells (at
least five cases and five controls), which come mostly from the larger diseases
and common outcomes; the disease-balanced summary is reported beside it, and
where the two disagree in label, both are stated.

## The three comparisons that define the scenario

- `O2`: final OPERA against local J_L (total-system comparison).
- `O2_MATCHED`: final OPERA against J on the selected encoder (same-encoder
  architecture comparison). Both are central; only the second isolates the
  architecture when national pretraining is selected.
- `OAOWN_AB`: OPERA against learned routing (R_AB, the secondary model).
- `O1`: OPERA against per-task XGBoost.

## What tuning showed (for orientation, not a prediction)

Gains over the joint model, paired tuning AUC, seed 42 (seed 43 similar):
global correction only −0.003; own corrections inside the router +0.006; each
outcome's own reading only +0.005; own disease and outcome corrections +0.008;
reading with only the outcome or only the disease correction +0.008 each; own
corrections with learned routing (R_AB) +0.011; OPERA +0.013. Four-seed OPERA:
+0.012 over J, +0.004 over own corrections, +0.0007 over R_AB, −0.031 against
XGBoost (tuning, optimistic for XGBoost).

## Mechanism (reported in every scenario)

Whatever the headline, the paper reports where the signal sits, from the two
prespecified decompositions: OPERA's ladder (own corrections; + each outcome's
own reading) with its ablations (without the disease correction, without the
outcome correction), and the router's (global only; own corrections; + learned
routing), with the routing summary (own share, entropy, learned against frozen).
The questions they answer: is the gain in the task-specific corrections, in the
reading, or in borrowing between tasks; do the disease and outcome corrections
both carry it; is any of it in what all tasks share (the global correction).

## Scenario A (expected): O2 better; OAOWN_AB equivalent or inconclusive; O1 worse

Specialising each prediction improves on borrowing uniformly. The gain sits in
the task-specific corrections, for the patient's disease and for the outcome,
and in each outcome's own reading of the record; a correction shared by all
tasks adds nothing. Learning which other tasks to borrow from is a second way to
specialise that gains about as much on top of the same corrections; its routes
reduce to the task's own correction plus an even mix of the others, so learned
routing amounts to partial pooling rather than targeted borrowing. Tuned
per-task XGBoost remains more accurate, most of all for disease course, where
it draws on recent counts of functional-status contacts.

## Scenario B: O2 better and OAOWN_AB better

As A, except that the fixed reading design is better than learned routing, not
only equal to it. Routing is still reported as improving on own corrections
where it did; the conclusion is that reading the record per outcome is the
stronger of the two ways to specialise, not that borrowing between tasks harms.

## Scenario C: O2 equivalent or inconclusive

The tuning gain from specialising does not replicate on the test set. The
robust result is borrowing itself (P1, P5): learning across diseases and
outcomes against one model per task. The design search becomes a documented
negative result: no specialisation of the shared model, learned or fixed,
reliably improved on the joint model. The mechanism decompositions are still
reported, as descriptions of what the tuned models did.

## Scenario D: O1 equivalent or better

The gap to per-task XGBoost closes on the test set, consistent with the
optimism of per-task tuning on the tuning set and with the temporal shift. The
headline changes: the sequence model matches tuned tabular models while
predicting every task from one model. Combines with A, B or C for the rest.

## Seeds (alongside every scenario)

Reported beside the seed-42 results: OPERA over its four seeds (mean and
range), the seed-averaged contrasts (`OAOWN4_*`) and the ladder on seed 43. If
seed 42 and the seed mean carry different labels for a comparison, both are
stated and the interpretation rests on the seed mean; the primary result stays
seed 42 as prespecified.

## Clinical indices (alongside every scenario)

For each index with its own endpoint and patients: whether OPERA's size-matched
risk groups separate survival at least as well as the index's categories
(lowest and highest groups), with NCCN-IPI in DLBCL as the main comparison and
the count of indices for which this holds. The model is OPERA (J_OA_OWN). The
statement is about stratification on the patients with a recorded index, not
about clinical benefit.

## National arm (written when its results exist)

Two lines, added before its results are opened: whether the national encoder
narrows the gap to XGBoost, mainly in laboratory outcomes and rare cells, for
OPERA and for learned routing alike.


Follow-up clarification (3 November): scenario wording about specialisation
requires O2_MATCHED, not O2 alone, if OPERA uses E/EA. National pretraining and
architecture may contribute differently. Clinical-index comparisons are
estimation-focused with pointwise paired intervals, outside the confirmatory
Holm family. Report both group sizes and achieved survival/event risks; a
non-negative estimate is not evidence of non-inferiority. The secondary
balance uses disease → approved loss_endpoint → variant → horizon.


Clinical KM matching implementation (3 November, before test evaluation):
alongside equal-size comparisons and development-frozen anchors, show largest
low/high ranked groups meeting the observed evaluation-cohort clinical KM
survival, with paired target re-estimation and boundary search in every draw.
These descriptive operating points are separate binary selections and can
overlap. Export achieved survival and valid matching-draw fractions; no formal
superiority, non-inferiority or deployment-cutoff claim follows from matching.
The exact support and matching rules are recorded in closure section 6c.
