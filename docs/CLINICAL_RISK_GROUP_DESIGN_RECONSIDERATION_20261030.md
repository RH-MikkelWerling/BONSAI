# Clinical risk groups: separate the questions before freezing the protocol

Status: methodological recommendation, not a change to the running recipe.
The independent allocation snapshots the current clinical diagnostics. It does
not fit a replacement threshold rule. The current primary rule remains
`risk_matched_low`, with apparent training predictions used for development.

## What is awkward about the current rule

The implementation separately finds the largest low-risk prefix with observed
event risk no higher than the index's low-risk group, and a high-risk threshold
matching the index's highest-group specificity. It then invents the middle
model boundaries from the index's middle-group proportions. These are three
different criteria. There is no guarantee that their cutoffs define a single
ordered, disjoint classification.

For example, a low cutoff of 0.57 and a high cutoff of 0.32 would label risks
between the two as both low and high. This is an incompatibility between the
independent operating points, not by itself evidence of a broken neural
forward pass. Lowering the low cutoff to enforce order would change the
"largest feasible low-risk group" objective. A jointly constrained solution
may exist, but is a different explicitly defined rule, not a numerical repair.

The middle model groups have no independently established clinical meaning.
Giving them the index's names can imply that the two systems share an absolute
risk scale or constitute interchangeable clinical instruments. The native
index's labels should remain intact. Model operating points should be named
by what was matched and the development population, not just inherited labels.

Searching every risk cutoff for the largest prefix whose observed event rate
passes a constraint also optimises against noisy labels. Prefix event rates
can fluctuate rather than rise monotonically. A minimum of 20 people alone
does not make a very low event-rate estimate precise. Increasing the training
denominator does not eliminate this selection issue.

## The separate problem with apparent training predictions

The score targets can sensibly be estimated from the large training registry
population. But these patients also trained the neural model. Their model
predictions can look better separated than predictions in new patients.
Choosing model cutoffs using that separation can produce optimistic development
matches and poor transport of numeric thresholds. Apparent model performance
and resampling validation are different, as described by
[Collins and colleagues, BMJ 2024](https://www.bmj.com/content/384/bmj-2023-074819).

This does not make a completely untouched test evaluation invalid. Thresholds
are part of development and can be frozen before testing. Test results would
honestly evaluate the entire fitted system, including a poorly transporting
threshold. It does mean that using more apparent training predictions did
not solve the threshold-estimation problem as fully as the original move
away from sparse tuning data suggested. Rebootstrapping the same fitted
training predictions does not make them out of sample.

The last clinical report already demonstrates unequal achieved tuning risks
and specificities. Matching a development target is not the same as matching
the observed event rate in the evaluation cohort. We must not advertise the
latter unless the evaluation results and uncertainty support it.

## Recommended primary presentation

Keep the clinical indices in their native categories and assess the model's
continuous predictions. Report paired discrimination, calibration and the
clinical operating comparisons separately. Categorisation loses information,
and the rationale for thresholds should be explicit rather than driven only
by statistical convenience. See
[TRIPOD-Cluster's explanation of risk groups](https://www.bmj.com/content/380/bmj-2022-071058).

Use two distinct clinical benchmark questions:

1. **Low-risk identification:** how many patients does the frozen model cutoff
   identify, and what is their observed event risk, alongside the index's
   native low-risk group? Record the development risk target, achieved
   evaluation risks, uncertainty and additional patients identified. A failure
   to meet the target in evaluation is a result, not permission to move the
   cutoff using evaluation labels.
2. **High-risk identification:** at a development cutoff targeting the index's
   specificity, how many events does the model capture? Report both systems'
   achieved evaluation specificity, recall, precision, MCC, counts and paired
   uncertainty. Do not claim equal achieved specificity unless it holds.

The two operating points need not form one common risk classification. If
they overlap, that must be explicit. Do not use them together to assign every
patient to apparently disjoint low/intermediate/high groups. Do not suppress
an otherwise valid binary comparison because the combined partition failed.

Evaluation-standardised ROC or event-rate comparisons are possible separate
performance summaries, but selecting an evaluation cutoff is not a frozen
deployable classifier. Such analyses must be declared and labelled separately.

## Where model thresholds should come from

My preference is out-of-sample development predictions. Out-of-fold predictions
over training patients could retain the larger denominator. To obtain them,
the supervised fitting pipeline must exclude each patient's labels in the
fit predicting that patient. Merely exporting the current full-fit training
predictions again, bootstrapping them, or cross-validating only a calibration
layer does not achieve that. This would require additional supervised fits.
Fold-trained and final-fit score scales may still differ, so cross-fitting
does not guarantee temporal transport or equality of evaluation risks.

The practical alternative using existing models is to estimate thresholds
from the existing tuning predictions, acknowledging that this is a development
set already used for model selection. The test set remains the untouched
evaluation. Sparse score populations may make some cutoffs too imprecise or
unavailable. Estimate uncertainty and declare support requirements in advance.
Returning to tuning must be an explicit protocol decision with its small-sample
trade-off, not a silent fallback for scores whose training rule looks poor.

There is no cost-free way to obtain a large, independent labelled threshold
development sample from the current single fitted model. The honest choices
are additional cross-fitting, smaller development data, or retaining the
apparent-training rule as a clearly qualified benchmark. Changing the source
of predictions is distinct from deciding whether two operating points should
be forced into one partition.

## If we still want complete model risk groups

For descriptive plots, development quantiles matching the index's category
shares give a clear, label-light size-matched comparison. It asks whether
the model separates outcomes better at comparable group sizes. Ties and absent
native strata still need an explicit handling rule. Frozen numeric quantiles
will not necessarily preserve group sizes in a later cohort. Such a plot
cannot establish that the model identifies more low-risk patients at the
same risk, because comparable development group size is built into its design.

For an eventual clinical classification, clinically justified absolute-risk
boundaries would be preferable, with adequate calibration and prospective
assessment. We currently have no agreed action-linked boundaries, so inventing
universal percentages would add unsupported clinical meaning.

## Decision before test release

First collect the current cutoff diagnostics and transport results. My
recommendation is to stop making the full matched partition the gatekeeper
for reporting a score. Retain the continuous comparison and separate clinical
operating points. Decide explicitly whether threshold development remains
apparent-training based, moves to tuning, or merits cross-fitting. Update the
methods, figure labels, uncertainty rules and analysis lock consistently before
the held-out run. None of this requires rebuilding frozen outcomes or choosing
a new predictive architecture to improve a figure.
