# Proposed thesis argument and replacement passages — 29 October 2026

These are review drafts, not edits applied to the thesis. Numerical passages
are explicitly tuning-only and must not become held-out conclusions by removing
their status labels. Locations refer to the current `thesis/kappa/kappa.tex`.

## Central argument

Clinical prediction often divides a limited resource into many separate
learning problems. Each disease has its own patients, each outcome its own
labels, and each model must recover predictive patterns from a fraction of the
available experience. Some of those patterns extend across diseases and
outcomes. Others do not. This thesis asks which information can be shared to
improve prediction, when sharing instead moves errors between tasks, and what
can be gained by serving many clinical questions with one model. The aim is to
establish the value and limits of sharing, including when a simpler or more
specialised model remains preferable.

## Shorter research questions

**RQ1. What does a richer patient record add to clinical risk prediction?**
Paper II compares an established clinical index with models using continuous
variables and broader longitudinal information. It examines the contribution
of pooling lymphoma subtypes and the clinical consequences of the resulting
rankings and risk groups.

**RQ2. When does sharing across diseases and outcomes improve, preserve or
degrade prediction?** Paper III compares otherwise matched supervision scopes
from a common pretrained starting point. It distinguishes the contribution of
other diseases from that of other outcomes and reports their benefits and
costs across tasks, rather than only their average.

**RQ3. How can shared learning be adjusted to the target?** Paper III tests
whether disease- and outcome-specific corrections improve a joint model, what
is gained from borrowed corrections beyond own-task adjustment, and what the
fitted model's source dependence permits us to interpret.

**RQ4. What does learning beyond the specialty add?** Paper III compares local
and national pretraining, directly and after adaptation, at full and reduced
label availability. These comparisons ask whether information learned from
patients outside hematology improves specialty prediction or reduces its
data requirements.

Across these questions, we also evaluate whether a consolidated model can
replace a portfolio of predictors with acceptable performance, coverage and
computational costs. Building reusable data, definitions and evaluation tools
is a parallel objective, addressed by Paper I and carried through the later work.

Editorial option: retain the thesis's existing RQ1/RQ2 numbering and make the
last three questions RQ2a–c. The key change is that each has an explicit answer
and permits an unfavourable result.

## Framework addition: transfer, interference and consolidation

Place after the partial-pooling explanation and before the detailed model
families. Cite Caruana for multi-task learning, Efron–Morris for shrinkage, and
Standley/Yu for task compatibility and interference. The latter studies provide
context, not a diagnosis of this model.

> Sharing is useful when what is learned outside a target task improves its
> predictions. This is positive transfer. It becomes negative transfer when
> the shared model predicts the target less well than the appropriate
> restricted model. Both are statements about a comparison. A shared model
> can improve over an isolated model yet lose to a model that shares along
> only one axis.
>
> The distinction matters in the disease–outcome grid. Learning other outcomes
> within one disease may help a specialist, but those same outcomes may add
> little once the model already learns from patients across diseases. The
> two sources of supervision may supply overlapping information. Their
> benefits need not add, and a smaller incremental benefit is not by itself
> evidence that sharing has harmed prediction.
>
> An average also hides who gains and who loses. A near-zero difference can
> mean that most tasks are unchanged, or that improvements in some balance
> losses in others. We therefore examine the distribution of paired effects,
> their uncertainty and their relation to training support. Failure to show
> an average improvement does not establish that important losses are absent.
>
> Sharing can have value even when it does not improve the average prediction.
> One model may answer many questions that previously required separate
> predictors. Whether this is a useful trade depends on the losses it incurs,
> the coverage it provides and its costs in training and use. Consolidation
> is a separate benefit to measure, not a substitute for reporting predictive
> performance.

## Framework correction: what partial pooling does and does not imply

> Partial pooling provides a useful statistical analogy. A group with little
> evidence of its own relies more strongly on the pattern estimated across
> groups, while a well-supported departure can remain distinct. This can
> reduce aggregate estimation error, but it does not guarantee that every
> group improves. If the assumed common pattern is inappropriate, shrinkage
> can introduce bias where the local evidence is weakest.
>
> The neural models in this thesis do not implement the normal hierarchical
> model directly. They share a learned representation and allow different
> forms of task-specific adjustment. Their behaviour has to be established
> through the prediction comparisons. Source-removal estimates use shrinkage
> for a different purpose, to limit overinterpretation of noisy effects in
> the analysis.

## Methods clarification: common evaluation and different learning problems

> The neural scope comparisons begin from the same pretrained encoder. A
> specialist is restricted to one disease and outcome during supervised
> fine-tuning, although its initial representation has already learned from
> the broader pretraining population. Disease-pooled models add patients
> from other diseases, outcome-pooled models add supervision from other
> outcomes, and the joint model adds both. These comparisons estimate the
> incremental contribution of supervised sharing.
>
> Neural and tabular predictions are evaluated on matched risk sets using
> the same outcome definitions, competing-death treatment, censoring weights,
> support criteria and paired patient resampling. Their training procedures
> and representations differ. The tabular models use aggregates over
> lookback windows and tune each prediction task separately, while the
> sequence models learn from ordered events and can share their parameters
> across tasks. Their comparison is between complete prediction pipelines.

Only describe these as verified server procedures after the pending provenance
checks pass. Code-level parity and aggregate-table consistency have been checked.

## Provisional discussion opening using the tuning evidence

> The development results show that the value of sharing depends on what is
> shared. Within the neural comparisons, pooling patients across diseases
> produced the largest discrimination gain. Sharing outcomes within a disease
> produced a smaller gain, and added little on average after diseases were
> already pooled. These effects were not uniform across tasks. A broad model
> can therefore preserve the average while changing which predictions it
> makes well.
>
> This is an important qualification to the idea that related tasks should
> be learned together. Shared supervision can recover information that a
> sparse task cannot estimate alone, but some of that information may
> already be available through another route. In this study, even the neural
> specialists began from a representation pretrained across the cohort.
> Additional supervision across diseases and outcomes was measured on top
> of that existing sharing.

For a more numerical tuning-only paragraph:

> Across 3,004 supported disease–outcome–horizon combinations, disease pooling
> increased AUC by 0.120 relative to neural specialists, while outcome pooling
> increased it by 0.027. Joint learning differed from disease pooling by
> −0.001 (95% interval −0.008 to 0.005). This near-zero aggregate did not mean
> that every task was unchanged. At one year, 258 of 600 cells had estimated
> losses exceeding 0.01 AUC and 230 had gains exceeding 0.01. These counts
> describe point estimates rather than established harms or benefits. They
> motivate reporting the distribution of effects alongside the mean.

## Discussion addition: strong tabular performance

> Sharing and model class answer different questions. Disease pooling also
> improved XGBoost, so the evidence for sharing does not depend on sequence
> modelling. The stronger performance of the current tabular pipeline shows
> that access to a more detailed representation does not ensure that a model
> learns more useful information from it. Aggregation discards some temporal
> detail, but it also supplies structure that may be valuable when labels
> are limited. Conversely, a shared sequence representation may simplify
> prediction across a broad panel even where it is not the most accurate
> predictor for each target. These possibilities require separate evidence
> about accuracy, coverage and cost.

Avoid concluding that tabular performance proves which mechanism is responsible.
History windows, tokenisation, sample size, optimisation and selection all differ.

## Discussion addition: routing without a selective disease map

> The routed model tests whether task-specific departures improve a shared
> predictor. Its overall comparison with the joint model combines the effects
> of additional capacity, own-task adjustment and borrowed corrections. The
> controls separate these contributions more closely. A modest improvement
> over the joint model is therefore evidence for the fitted correction
> system, not by itself evidence that the model has learned a clinically
> meaningful map of relationships between diseases.
>
> Routing weights describe allocation within that system. Their interpretation
> depends on the size and direction of the corrections they weight. Diffuse
> weights can coexist with useful adjustments, and a large weight can attach
> to a correction that changes predictions little. Source removal addresses
> dependence more directly, but describes the fitted model under the chosen
> ablation. It does not identify which source would improve a new model if
> supplied in isolation.

Retain the numerical conclusions as pending until the user's provenance/replay
reports confirm the fit identities and the interpretation of each export.

## Discussion correction: Paper II clinical claims

> The addendum strengthens Paper II's evidence for improved ranking while
> qualifying its low-risk headline. On the same patients, the model retained
> a clear advantage over the corrected NCCN-IPI in discrimination. The size
> of the low-risk group was more dependent on threshold development. The
> several-fold expansion obtained with the published cutoffs did not persist
> when cutoffs were developed from out-of-fold training predictions using
> the stated criterion. This distinction matters clinically. Better ranking
> creates opportunities for risk stratification, but the number of patients
> assigned to a group and their achieved risk must be evaluated with
> independently developed thresholds.

If giving 81 versus 79, preserve the different model/index denominators from
the addendum and do not call that table a matched-patient superiority test.
For OPERA, describe the actual apparent-training threshold development honestly
and report achieved evaluation risks rather than promising identical rates.

## Clinical and engineering contributions

> Learned representations change where expertise enters model development.
> They reduce the need to prescribe every predictive feature, but they do
> not remove choices about what a recorded event means, when it was known,
> who is eligible for an endpoint or how a laboratory finding corresponds
> to a clinical grade. The outcome and code audits in this work made these
> choices concrete. Clinical review remains necessary throughout development,
> especially when a standard definition is approximated from routine data.
> Shared pipelines make those choices inspectable and reusable, which is
> part of their scientific value.

## National pretraining: result-contingent interpretations

Use one of these only after the relevant comparisons are available.

- **If it improves broad held-out performance:** learning from patients outside
  the specialty can improve specialty predictions under the evaluated
  pipeline. Identify whether source availability or vocabulary changed too.
  Do not attribute the gain solely to more patients without those controls.
- **If it mainly helps with fewer labels:** its main value is data efficiency
  rather than a higher full-data ceiling. Quantify within the measured range
  of the learning curves and report uncertainty.
- **If it adds little or loses:** the evaluated national pretraining procedure
  did not improve these tasks over the local alternative. This does not negate
  the disease-sharing comparisons or demonstrate that population pretraining
  cannot help under other data, objectives or adaptation strategies.

Do not compare the unfinished epoch-3 run with a completed local encoder as if
that were the prespecified final national arm.

## Proposed closing argument

> The boundaries used to classify and treat diseases need not be the boundaries
> used to learn their prognosis. The useful question is how to use the shared
> information while retaining differences that matter for prediction. The
> answer need not be one architecture for every task. It may combine a shared
> representation, task-specific adjustment and simpler models where they are
> stronger. What makes those choices defensible is a common evaluation of
> their benefits and costs, supported by data and definitions that others can
> inspect and reuse.

Add the actual held-out answer immediately before this paragraph. It should
close a demonstrated argument rather than replace a result with a general claim.
