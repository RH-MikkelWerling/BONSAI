# Thesis review against the current OPERA evidence — 29 October 2026

## Overall assessment

The thesis has a coherent subject: how to make more of the information already
available for clinical prediction, including information learned outside the
target disease and outcome. Its strongest contribution is a connected empirical
argument, supported by reusable infrastructure. It does not need every step
towards a larger or more shared model to improve discrimination.

The current draft is already more cautious than a simple claim that deep
learning wins. It acknowledges strong tabular alternatives and the possibility
of negative transfer. Its problem is that many individual passages still tell
the earlier, more optimistic story: richer records, then more diseases, then
more outcomes, then selective routing, then national pretraining. The current
tuning evidence supports an uneven pattern rather than that progression.

The most useful revision would make **positive transfer, negative transfer,
redundant supervision and consolidation of predictions** explicit possible
answers to the research questions. Route interpretability and national
pretraining then become tests of how sharing can work, not results the thesis
needs in order to succeed.

The finding that disease pooling helps within both neural and tabular families
is especially valuable. It means the sharing argument does not depend on the
sequence models beating XGBoost. Conversely, the current XGBoost advantage is
important evidence about how effectively the pipelines use their inputs.

## Scope and status of this review

Reviewed the active `thesis/main.tex`, all of `thesis/kappa/kappa.tex`, the English
and Danish abstracts, preface, additional-paper introduction, Paper I addendum,
Paper II addendum, OPERA import wrappers, relevant included-paper sections,
the synthesis figure generator and the thesis task list. Compared these with
the current BONSAI manuscript and downloaded tuning aggregates. Inspected the
rendered analogy and landscape figure in `thesis/main.pdf`.

This is a synthesis, scientific-interpretation and consistency review. It is
not an independent reproduction of the original Paper I/II patient-level
analyses, a full reference-by-reference literature audit, or a graphical audit
of all 355 thesis pages. Published reprints should retain their historical text.
Corrections belong in addenda and in the thesis's own synthesis.

No source files in the thesis directory were edited. The review and proposed
replacement passages are concrete drafts in BONSAI. No test results were opened.
All new numerical work uses the existing aggregate tuning download.

Supporting artifacts:

- `scratch/paper_reading_20261028/thesis_review_20261029/analyse_transfer.py`
- `scratch/paper_reading_20261028/thesis_review_20261029/transfer_review_numbers.json`
- `scratch/paper_reading_20261028/thesis_review_20261029/outcome_transfer_auc_cells.csv`
- `scratch/paper_reading_20261028/thesis_review_20261029/primary_transfer_summaries.csv`
- `scratch/paper_reading_20261028/thesis_review_20261029/joint_minus_disease_domain_summaries.csv`
- `scratch/paper_reading_20261028/thesis_review_20261029/outcome_transfer_review.pdf`
- `docs/THESIS_PROPOSED_ARGUMENT_AND_PASSAGES_20261029.md`

The script checks contrast means against the saved effect summaries, verifies
unique cell keys and the support rule, and records source-file SHA-256 hashes.

## 1. What the current results actually support

All results below are development evidence from tuning, conditional on the
selected fitted models. The 3,004 entries are supported disease–outcome–horizon
combinations, not independent observations or separately fitted models.

| Question | Contrast | Equal-cell AUC difference, saved 95% interval | Interpretation |
|---|---|---|---|
| Do other diseases help an outcome model? | Disease-pooled − specialist | +0.11966 (+0.10679 to +0.13315) | Large gain within this neural family. |
| Do other outcomes help within a disease? | Outcome-pooled − specialist | +0.02734 (+0.01472 to +0.04038) | There is an average discrimination gain in this contrast. |
| Do outcomes add once diseases are already pooled? | Joint − disease-pooled | −0.00114 (−0.00754 to +0.00526) | Little average AUC change, with uncertainty. |
| Do the two gains add? | Pooling interaction | −0.02848 (−0.04220 to −0.01495) | The incremental gains overlap or differ by context. This is not itself proof of harmful transfer. |
| Do routed corrections improve joint learning? | Revised OPERA − joint | Approximately +0.0087 (+0.003 to +0.015) | A modest average gain, not proof of selective borrowing between identifiable disease pairs. |
| Is local OPERA the strongest current predictor? | OPERA − per-task XGBoost | −0.02803 (−0.03633 to −0.01929) | The current tabular pipeline is stronger on this roster. |

The outcome-pooling gain over specialists does not extend clearly to probability
accuracy: its IPA difference is −0.00499 (−0.01146 to +0.00082). The thesis should
distinguish ranking from absolute-risk prediction.

One further distinction matters throughout: **the neural specialists already
share information through their common pretrained encoder**. They are isolated
in supervised fine-tuning, not models that have learned only from their target
cell since initialisation. The S/D/O/J comparisons estimate the additional
contribution of supervision across tasks on that pretrained starting point.
They do not compare all borrowing against no borrowing. This is precisely why
pretraining belongs in the same story but needs its own contrast.

### The important correction to “almost no negative transfer”

The useful insight is that one shared model may replace many outcome-specific
models with little **average** performance change. The current results do not
show that individual outcomes suffer almost no negative transfer.

At one year, in 600 supported cells, **joint versus disease-pooled** has:

- 311 negative point estimates, 268 positive and 21 exact ties.
- 258 losses greater than 0.01 AUC, and 230 gains greater than 0.01 AUC.
- Mean difference −0.00316 and median −0.00211.

These are descriptive point-estimate counts, not counts of established harms.
The 0.01 cut is an illustrative magnitude, not a newly justified clinical
non-inferiority margin. Sparse cells are noisy and outcomes are correlated.
Nevertheless, the pattern is incompatible with presenting the near-zero mean
as evidence that every task keeps its performance.

There is also a domain pattern worth taking seriously. Joint minus
disease-pooled AUC for **lymphocyte/eosinophil outcomes** is −0.06834
(−0.09072 to −0.04676) across 127 cell–horizon combinations in 14 diseases.
At one year it is −0.06906 (−0.09789 to −0.04069), across 24 cells in 13
diseases. The direction is negative at all six horizons. These are correlated
comparisons and unadjusted domain intervals, not six replications or a
confirmed mechanism. But this is a much stronger reason to investigate
redistribution than a handful of extreme sparse-cell estimates.

Tightening the case/control threshold does not turn joint-minus-disease-pooled
into a clear descriptive gain:

| Minimum cases and controls | Cell–horizons | Mean joint − disease-pooled AUC |
|---|---:|---:|
| 5 | 3,004 | −0.00114 |
| 10 | 1,802 | −0.00888 |
| 20 | 1,131 | −0.00790 |
| 50 | 451 | −0.00487 |

These thresholds select different task populations. They are not an experiment
on increasing sample size and have no new confidence intervals.

### A better interpretation of consolidation

Use three distinct claims:

1. **Positive transfer:** the shared system predicts a target better than the
   appropriate restricted system.
2. **Preserved performance:** losses are bounded tightly enough to exclude a
   justified practical margin, at the declared aggregate or task level.
3. **Consolidation:** one model provides many predictions instead of a
   portfolio, with measured costs, coverage and performance trade-offs.

The current evidence supports a consolidation hypothesis and little average
incremental AUC change from outcomes after disease pooling. It does not yet
establish taskwise preserved performance. The equal-cell interval is compatible
with an average loss greater than 0.005, and the disease-balanced estimate is
−0.00958 (−0.02490 to +0.00376). Weighting matters.

A positive consolidation claim needs the actual number of fitted models
replaced, storage, inference time for the full prediction panel, and training
cost. Model count alone does not establish lower training cost. Do not count
six neural horizons as six separately trained models.

Task compatibility and the accuracy/computation trade-off are established
multi-task concerns, rather than a new statistical principle. This strengthens
the framing when cited honestly. [Standley et al., 2020](https://proceedings.mlr.press/v119/standley20a.html)
study task grouping and inference trade-offs; [Yu et al., 2020](https://arxiv.org/abs/2001.06782)
study detrimental gradient interference. These are precedents, not evidence
that gradient conflict caused the observed OPERA losses.

## 2. The argument the thesis should make

The central question should be **what information can usefully be shared, under
which constraints, and with what costs**, rather than how far more information
can keep increasing AUC.

The papers then have clear roles:

- **Paper I:** makes linked information accessible across projects and diseases.
  Its contribution is demonstrated reuse and usable definitions, not merely a
  prerequisite for Paper III.
- **Paper II:** tests whether a richer patient description improves an
  established clinical prediction problem. Its addendum shows why comparison
  populations and threshold development change the clinical interpretation.
- **Paper III:** tests the sources and limits of sharing across a much broader
  prediction panel. It distinguishes disease sharing, outcome sharing,
  task-specific correction, population pretraining and strong tabular alternatives.
- **Paper IV:** asks whether clinical tables can be made useful as supervision
  for another kind of model. It is a separate extension of information reuse,
  not validation of the hematology predictors.

The thesis can make a stronger contribution with mixed results. It would show
that disease boundaries need not be prediction boundaries, that auxiliary
outcomes need not add benefit after disease pooling, and that a broad sequence
model can still lose to carefully aggregated tabular information. That is a
substantive account of where information becomes useful and where its
representation or sharing fails to deliver it.

### Organise the questions as tests, not stages of expected progress

Retain RQ1 for Paper II, but split the current sprawling RQ2 into explicit parts:

- **RQ2a — Across which axes does sharing improve or degrade prediction?**
  Disease and outcome supervision, their conditional contributions and
  distribution of gains/losses.
- **RQ2b — What does task-specific adjustment add?** Routed versus joint,
  own-task versus global-only, full versus own-task, and fixed-fit removal.
  Each comparison has a different interpretation.
- **RQ2c — What does population pretraining add beyond local pretraining?**
  Full-data performance, label efficiency and adaptation, with provenance of
  available data and vocabulary.
- **RQ2d — What can a consolidated model deliver clinically and operationally?**
  Strong baseline comparisons, native index endpoints, achieved risk-group
  outcomes, coverage and the cost of the model portfolio.

These parts need not become four new chapters. They should be visible in the
aims, the answer section and the conclusions. Avoid making positive results a
condition of each question having an answer.

## 3. Chapter-by-chapter changes

### Abstract and Danish resumé

Sources: `chapters/abstract.tex`, especially the Paper II paragraph and the
claim about a model reading the full record.

Keep the clinical motivation and the link from infrastructure to prediction.
Replace the current general contrast between indices and learning from more
information with a question that permits both transfer and interference.
Differentiate clinical inputs, sharing supervision and pretraining explicitly.

The 160-versus-one older low-risk headline must be qualified or removed from
both abstracts. The Paper II addendum reports that the large low-risk expansion
depends on the published cutoffs. Its training out-of-fold cutoff analysis gives
81 low-risk patients versus 79 for corrected NCCN-IPI in the reported populations.
That is not evidence for a general several-fold expansion under independently
developed cutoffs. It also compares 897 model patients with 869 index patients
in that risk-group table, so the 81-versus-79 counts are not a fully paired
clinical superiority estimate either. The paired ranking gain is stronger:
ROC-AUC +0.071 (0.042 to 0.102), PR-AUC +0.119 (0.073 to 0.169).

The English abstract labels itself approximately 300 words, but English and
Danish text together currently contain about 907 body words by TeXcount.
There is room to shorten by removing detailed protocols and the broad claim
that every prediction is explained. Attribution decomposition and clinical
plausibility are narrower than a clinically validated explanation.

Describe local and national pretraining separately. Keep national efficacy
pending until it is measured. No results from the unfinished national encoder
should enter a factual thesis conclusion yet.

One additional addendum sentence needs correction: the calibration paragraph
says the preceding comparisons depend only on ranking. AUC/PR-AUC do, but the
same table also contains classification metrics at a fixed probability cutoff
of 0.5. Those are not invariant to recalibration. Also use “patients flagged”
or “number needed to evaluate” for screening efficiency, rather than calling
extra detections a treatment NNT without an intervention-effect estimate.

### Synopsis

Source: `kappa/kappa.tex:13`, `:94`, `:131`, `:179`.

The house/lighthouse analogy is memorable and explains scarcity well. Keep it
if it sounds like the author. It currently occupies too much of the opening
and returns repeatedly. Trim its five-item list and later reprises so the
clinical question arrives earlier. The contrast with housing valuations and
the later Danish housing-tax aside distract from the actual prediction problem.

Do not imply that clinicians themselves use only a checklist. The limitation
concerns formal prognostic tools, not all clinical judgement. Also distinguish
the original IPI's binary questions from the refined NCCN-IPI categories.

The synopsis should contain an early sentence that says that useful sharing
can improve performance, preserve average performance across a broader set of
outputs, or fail. That sets up the eventual mixed results.

Replace “one source at a time” for Paper II with a shorter, accurate summary.
The later framework already acknowledges that the original sequence of models
did not isolate all axes. The completed 2×2 addendum is the cleaner evidence.

The statement that infrastructure is “the larger contribution” need not be
repeated as a defensive ranking of the thesis's achievements. Give its reuse
evidence once, then let the synthesis show both contributions.

### Background

Sources: `kappa/kappa.tex:387`, `:418`, `:512`, `:546`.

This chapter establishes the clinical setting well. Keep the distinction
between prediction and treatment benefit and the reason for borrowing when a
national cohort cannot easily recruit more patients within the same task.

Remove the hospital 50-petabyte/97%-unused statistic, which the text itself
says has no stated source. It adds no evidence the thesis needs.

Soften repeated absolutes. A single-disease model can learn shared physiological
patterns from its own patients; it cannot use additional training examples from
other diseases. An index can use continuous components, and tabular models can
be pooled or multi-output. Sequence models do not eliminate human choices:
event timing, vocabularies, adaptive truncation, bins, history limits, targets,
masking and weighting are all consequential choices in this project.

Do not present tabular aggregation simply as information destroyed. It is also
an inductive bias that can improve data efficiency. The current tabular lead
makes this distinction central. A sequence representation has access to order
within its truncation window; that is an opportunity, not demonstrated use of
all clinically relevant temporal information.

The clinical score examples should motivate a fair benchmark rather than
portray outdated tools as the only alternative. Explicitly introduce pooled
XGBoost here. It shares across diseases too.

Also avoid classifying all indices as “no pooling”. The original IPI was
developed across aggressive non-Hodgkin lymphoma histologies and institutions.
The contrast is with broader and explicitly tested sharing, not a claim that
clinical scoring has never pooled diseases. [The original IPI study](https://www.nejm.org/doi/full/10.1056/NEJM199309303291402)
documents that development population.

### Framework

Sources: `kappa/kappa.tex:641` through `:1487`.

This is the thesis's intellectual centre. It needs a better account of why
sharing can preserve the mean while redistributing performance, rather than
more general background about ever larger models.

Keep the five sources as an organising device, but stop treating them as
independent experimental axes or an ordering of increasingly powerful models.
National pretraining changes patients, representation and possibly available
sources/vocabulary. More horizons reuse correlated observations; they do not
multiply the number of independent patients or events.

Add a short section immediately after partial pooling that defines:

- Positive and negative transfer relative to a named comparator and metric.
- Average transfer versus the distribution across targets.
- Redundant supervision versus harmful sharing.
- Preserved performance versus a non-significant superiority test.
- Consolidation of models versus improvement of an individual prediction.

Use the S/D/O/J contrasts as the concrete example. State that O−S and J−D
answer different questions. Do not say globally that “outcomes do not help”.

Retain Efron–Morris and Caruana. **Both are already cited in the active thesis.**
The improvement needed is precision, not simply adding their names.
Efron–Morris concerns aggregate estimation risk under a particular statistical
setting, not a guarantee for every clinical target.
[Efron and Morris, 1975](https://doi.org/10.1080/01621459.1975.10479864)
is an appropriate foundation for shrinkage, while
[Caruana, 1997](https://doi.org/10.1023/A:1007379606734) is the relevant
multi-task-learning foundation.

Separate three uses of pooling: shared representation learning, the OPERA
architecture inspired by task-specific departures, and shrinkage of noisy
analysis estimates. They are related ideas but not one fitted hierarchical
Bayesian model. A smooth, diffuse router is not automatically an empirical-Bayes
posterior over related tasks.

### Aims

Source: `kappa/kappa.tex:1495`.

The current RQ2 asks whether using everything keeps improving prediction and
whether learning what to borrow “makes sharing work”. The first builds in a
progress narrative, while the second makes the router sound necessary for
sharing. Recast as the explicit questions above. Keep infrastructure as a
parallel objective with its own observable contribution.

### Methods

Sources: `kappa/kappa.tex:1636`, `:1661`, `:1757`.

Keep this chapter as a cross-paper account of design decisions and comparability,
not another full OPERA methods section. Use the current paper for configuration
detail and include the missing supplements in the thesis package.

Correct the following before filling final results:

- Outcome-specific readouts remain in the joint model. It does not use the
  same prediction rule for every outcome.
- The deployed revised router has a global correction outside the routed
  mixture. The older schematic and prose must describe the selected architecture.
- Additivity is on the appropriate linear model scale; cumulative incidence
  is a nonlinear transformation. Source corrections do not add linearly to
  the predicted probability of an event.
- A separately trained all-sources-masked control is not numerically the same
  fitted joint model. Equality of an architecture in a special case and
  equality of predictions from separately trained weights are different claims.
- A difference between two fits is a repeatability observation, not a measured
  distribution or a universal “noise floor”.
- Replace the claimed hierarchical Bayesian primary summary with the actual
  paired patient-bootstrap estimand. Identify equal-cell and disease-balanced
  summaries separately. Shrinkage is still relevant to source-removal summaries.
- Remove the decision-curve subsection and later claims about its results,
  unless that analysis is deliberately reinstated. The present manuscript
  removed it and the user has not authorised new clinical-utility conclusions.
- Update clinical thresholds from tuning-derived to the actual training-derived
  procedure. Current OPERA code records `apparent_training` predictions; do not
  describe those as out-of-fold. The Paper II addendum's out-of-fold procedure
  is distinct. Achieved rates and specificity on evaluation patients can differ
  from the development targets and must be reported.
- All 69 outputs remain separately predicted, with 52 endpoint families used
  for loss weighting. Explain the difference in one sentence and cross-reference
  the details.
- National training steps, epochs, vocabulary and source coverage must come
  from the completed run. “Every encoder is pretrained for 10 epochs” is not
  an established property of the unfinished national run.

Check the status of the promised sensitivity using diagnoses available before
treatment. The thesis permits final subtype assignment from codes recorded
14 or 90 days after treatment starts. Distinguish using that retrospective
classification for evaluation strata from supplying it as an input to the
router. If the latter relies on information unavailable at prediction time,
the primary clinical interpretation needs an available-at-index analysis or
an explicit restricted-use qualification. This review has not established
the actual server-side availability of each assigned subtype. Do not describe
the sensitivity as completed without its results.

The recent evaluation audit is useful here. The **paper evaluator** uses common
risk sets, censoring weights, support rules and paired draws for neural/XGBoost
contrasts. The training procedures and representations differ. Do not turn
evaluation parity into a claim that all model families were trained identically.

Clarify the statement that evaluation “corrects” the known-label training issue.
IPCW adjusts the evaluation estimator under censoring assumptions. It does not
recalibrate the fitted predictions or undo bias introduced during training.
Laboratory follow-up ending at the last measurement is also potentially
informative, rather than automatically administrative. A marginal censoring
model is not guaranteed to resolve dependence on patients' measured or
unmeasured health. This is a limitation to describe, not a new endpoint rerun.
[Gerds and Schumacher, 2006](https://pubmed.ncbi.nlm.nih.gov/17240660/)
distinguish the censoring assumptions behind these Brier estimators.

### Discussion

Sources: `kappa/kappa.tex:1937`, `:2010`, `:2146`, `:2305`, `:2501`, `:2711`.

The discussion currently answers RQ2 through a long sequence of placeholders,
then gives a landscape interpretation, then repeats the framework through
partial pooling, novelty and human knowledge. Consolidate these into fewer
arguments that answer the questions.

An effective order would be:

1. What information improved prediction in Paper II, including what changed
   after its addendum.
2. What moved between tasks in Paper III: dominant disease-sharing gains,
   conditional outcome redundancy and heterogeneous costs.
3. What routing adds beyond ordinary joint learning and own-task capacity.
4. Whether national pretraining changes accuracy or label efficiency.
5. What the tabular comparison says about representation and inductive bias.
6. What follows for a practical prediction system and for reusable infrastructure.
7. The limitations that change those interpretations.

The “bitter lesson” section becomes more interesting if it confronts the current
results. Aggregation may provide a useful bias at this scale. Broader pretraining
may or may not overcome that, and even a national gain would need a fair
comparison before being credited specifically to population breadth. Avoid
“the lesson has not yet arrived”, which makes tabular success sound like a
temporary inconvenience to an inevitable result.

Avoid saying that clinicians no longer need to be involved in the middle of
development. The outcome/code audit gives concrete counterexamples: CTCAE
adaptations, heterogeneous LDH assays, broad procedure definitions and molecular
prefixes. Their role changes, but clinical validation of definitions remains
part of the entire development cycle. This is a stronger, evidence-based
engineering/clinical collaboration section than speculation about replacing
clinical judgement with model capacity.

Keep the distinction between prognosis and intervention benefit. Tighten the
claims about low-risk R-CHOP patients “supporting keeping them on it”: observed
outcomes under received care do not establish comparative treatment benefit.
Reduced monitoring is also an intervention with costs and risks, not a use
that is automatically validated by accurate ranking.

Infrastructure is a major independent contribution and should stay substantial.
It should also include what the current work has taught about contracts:
time stamps, endpoint versions, vocabulary meaning, fit identity and a common
evaluator. These are practical mechanisms by which a shared resource produces
comparable science, rather than only a list of repositories.

### Conclusions and perspectives

Source: `kappa/kappa.tex:2923`.

State results at their actual strength. Do not repeat the large older low-risk
expansion as an unconditional conclusion. Give an explicit answer for sharing
and its limits, then a separate answer for national pretraining once available.

The closing claim can be that disease labels need not define isolated learning
problems. It cannot yet be that the best model learns a clinically meaningful
map of who should borrow from whom. If weak, diffuse source dependence persists,
that is a boundary of the interpretation rather than a failed thesis.

The future-work section should distinguish a query model predicting new codes
from a model answering any clinical endpoint. A composite, graded or eligibility-
restricted endpoint still requires a definition and validation. MEDS harmonises
format, not clinical semantics or source availability.

### Paper IV and front matter

Keep Tables2Traces as a clearly labelled additional paper. Its present bridge
already avoids claiming to validate OPERA, which is appropriate. Frame it as
reuse of clinical information through another representation. Its exam-category
gains do not independently establish clinical reasoning mechanisms or
transportability of the hematology models.

The preface's OPERA author list is outdated: it lacks Anish Simhal, uses the old
form/order of several names, and retains a coauthor roster inconsistent with
the current article. Copy the final article roster after it is agreed. Preserve
the published author lists for Papers I and II.

## 4. Specific errors and overstatements to fix

Locations refer to the current source at review time.

| Location | Problem | Required change |
|---|---|---|
| `kappa/kappa.tex:232` and `:1265` | A single-disease model “cannot see” common physiology. | It cannot learn from additional patients with other diseases. It can learn common patterns within its own data. |
| `:491` | Deep learning removes decisions about what the model sees. | It shifts decisions to representation, tokenisation, chronology, truncation, labels and evaluation. |
| `:931` | Representation-learning sample requirements are stated without their conditions. | Present the theory as conditional on suitable shared structure and the learning setting. |
| `:916` and `:1690` | Isolated learning sounds as though the fitted neural specialist learns its entire representation from its own events. | Distinguish a conceptual model trained from scratch from the actual specialist fine-tuned from a shared pretrained encoder. |
| `:1008` | Partial pooling means a rare cohort “cannot be outvoted”. | Rare groups are usually shrunk more strongly. Partial pooling offers a model of differences, not immunity from mis-specification or poor subgroup performance. |
| `:775` | IPCW evaluation sounds as though it corrects bias in the trained predictions. | It adjusts estimation of evaluation performance under censoring assumptions, without changing the predictions. |
| `:1040` and `:2081` | One readout shared by every disease and outcome, reading representation identically. | Joint shares an encoder and prediction architecture while retaining outcome-specific outputs and conditional information in the representation. |
| `:1065` | Linear readout means additive changes in predicted risk. | Additive components on the linear/log-hazard scale do not imply additive cumulative incidence. |
| `:1077` | Near-zero adapters mean the fitted router starts as the separately fitted joint model. | This is an architectural initialisation, not the same fitted encoder/head trajectory. |
| `:1128` | National population contains the cohort, so the national encoder saw everything local pretraining saw. | Nested patient populations do not imply identical events, calendar coverage, vocabulary or local EHR sources. |
| `:1141` | National minus local isolates what the rest of the population adds. | It compares complete pretrained systems unless sources, vocabulary, architecture, training objective and compute are controlled. |
| `:1190` | Foundation models should answer more or less any clinical question. | Broader adaptability does not imply arbitrary endpoint prediction without labels or validation. |
| `:1245` | Paired patients alone make a gain attributable to sharing alone. | Also describe the matched architecture/training design and its remaining differences. Pairing alone is insufficient. |
| `:1254` | Only a task-balanced summary can support a rare-disease claim. | Report rare-disease strata and uncertainty directly. Equal-cell weighting alone does not guarantee adequate rare-group evidence. |
| `:1456` | Reclassified patients “had the outcomes it predicted”. | Report the actual observed risks and intervals, under the threshold-development procedure. |
| `:1483` | Sequence modelling counts as progress only if it beats the tabular approach. | Better accuracy is one form of progress. Consolidation, coverage, efficiency or reuse may also count if demonstrated and losses are explicit. |
| `:1679` | All encoders have an established 10-epoch pretraining history. | Use completed checkpoint/step provenance and distinguish local, national and adapted arms. |
| `:1700` and `:1826` | Masked-source control computes exactly the joint fit's predictions and defines its noise floor. | Distinguish architectural equivalence, a separately trained control, fixed-fit ablation and replication variability. |
| `:1706` | Outcome adapters were expected to add little because outcomes have their own heads. | A head does not prevent interference in a shared encoder. Remove retrospective expectations or identify them as documented hypotheses. |
| `:1788` and `:2857` | Decision curves described as an analysis the paper reports. | Remove unless actually completed and retained in scope. |
| `:1815` and `:2883` | Primary effects are described as hierarchical Bayesian/posterior quantities. | Match the actual frequentist paired-bootstrap summaries and reserve shrinkage descriptions for analyses that use it. |
| `:1982` | Disease-course outcomes expected to have the greatest router benefit. | Replace the expectation-led placeholder with the observed domain results and their sparse support. |
| `:2050` | Domain differences in model AUC establish intrinsic foreseeability or the condition for transfer. | They describe this model, information and case mix. Similar absolute AUCs do not establish shared predictive mechanisms. |
| `:2087` | Diffuse gates mean each disease correction is shrunk towards the average correction. | Gate mass is not contribution size; inspect the support-scaled readout corrections and fixed-fit effects. |
| `:2076` | Clinical indices are placed at an absolute “no pooling” endpoint. | The original IPI already pooled aggressive lymphoma histologies. Describe the actual degree and form of sharing. |
| `:2258` | The Paper III sequence/tabular comparison uses the same features. | It uses matched evaluation populations/targets, with different feature representations and history windows. |
| `:2412` | No work in any clinical field has separated these axes. | Narrow to the supported scope or substantiate with a dedicated search. This review does not establish that broad priority claim. |
| `:2854` | Clinical thresholds were estimated on tuning. | Update to actual training-based development and say apparent versus out-of-fold explicitly. |
| `:2919` onward | Conclusion still carries a threshold-dependent low-risk headline. | Lead with verified paired discrimination and qualify risk-group findings. |

## 5. Figure and build problems

### Landscape figure: actual data would currently hide negative transfer

In `figures/kappa/evidence_map.py`, the domain bars use
`max(row.diseases, 0)` and `max(row.outcomes, 0)`. A negative effect is therefore
rendered as no bar. This is a genuine figure-logic error for the intended
positive/negative transfer question, even though the current B/C data are
explicitly labelled synthetic. Replace these with signed axes, visible zero,
uncertainty intervals and declared contrasts. Do not simply feed new results
into the existing code.

The same generator clamps domain AUCs to 0.55–0.90. Current real tuning estimates
can lie outside that range. Display the full range or mark every clipped value.
Its silent fallback to synthetic CSVs also needs to fail in a release build
when real inputs are required.

Panel A labels the poorer Waldenström result “biologically distinct”. That is
an untested explanation of one result. Remove the annotation; discuss possible
differences as hypotheses, alongside data, support and optimisation explanations.

Panel C translates AUC gains into multiples of training patients from learning
curves, with extrapolation. Keep this only when a comparator learning curve
actually identifies the conversion within a supported range and its uncertainty
is propagated. A plateau or overlapping/noisy curve does not identify a useful
sample-size equivalent. Direct label-efficiency curves are safer and clearer.
Do not infer a value for routing from a joint-only label-fraction curve.

The analogue figure also visually promises that unrelated sources are excluded.
Present this as the goal of selective sharing rather than the verified behaviour
of the current diffuse router. The new transfer review plot deliberately
preserves the negative effects.

### Current reader PDF needs a clean final build

The current `thesis/main.pdf` contains 355 pages and **272 literal `??` markers**
in extracted text. Its matching root `main.log` reports 294 undefined-reference
warnings. The separate `build/main.log` reports a 355-page output with no
undefined-reference warnings. The two PDFs have different file sizes.

This is evidence of inconsistent build products, not proof of 294 source-level
reference mistakes. After revisions, use the documented `build_mac.sh` chain
and distribute its copied final `main.pdf`. Recheck the actual PDF, not only an
older log or the README's 343-page status. Existing logs also contain overfull
boxes; a full graphical pass remains necessary after the text and figures are
settled. This review visually inspected two synthesis figures only.

### OPERA is shared with the article, but its supplements are incomplete here

`papers/opera/opera.tex` imports the live BONSAI manuscript body and result
macros. The synthesis, abstract and preface remain separate text. Updating the
article does not fix their old conclusions.

`papers/opera/opera_appendix.tex` imports supplementary figures only and
explicitly omits the two methods supplements. As detail has moved out of the
article's main Methods, this leaves the thesis reader without important data,
outcome and model definitions. Include them as thesis-compatible supplementary
sections or appended PDFs with their references intact, or provide a clearly
identified, permanent full supplement. Do not count a source-code link as a
substitute for the methods needed to assess the thesis.

The current prototype figure files and number placeholders must remain visibly
development-only. A thesis rehearsal can include tuning evidence if the status
is unmistakable. Final conclusions must use the frozen held-out analysis.

## 6. Work order and what is genuinely still needed

### Do now, while server work runs

1. Fix synthesis-level contradictions with Paper II's completed addendum.
   Preserve the published paper, qualify the older low-risk headline and use
   paired estimates in the thesis's own claims.
2. Rewrite the aims and the opening of the discussion around transfer,
   heterogeneity and consolidation. Replace expectation-led placeholders.
3. Correct the pooling/readout/pretraining/evaluation descriptions and remove
   analyses no longer in scope.
4. Correct the landscape generator before importing real results. Keep full
   signed effects, matched domains, explicit uncertainty and no synthetic fallback
   in release mode.
5. Plan inclusion of the missing OPERA supplements and refresh the preface.
6. Use one tuning evidence table as the source for all provisional thesis
   numbers, with contrast definitions and denominators attached.

### Investigate from existing tuning outputs, without another architecture sweep

- The lymphocyte/eosinophil joint-versus-disease-pooled loss. Inspect outcome-
  specific performance and loss histories, support, label consistency and
  family weighting. These are possible implementation/optimisation explanations,
  not a demonstrated disease mechanism. Keep it as a declared diagnostic, not
  a reason to remove those outcomes from the primary test panel.
- The distribution of gains/losses after each sharing step, including stricter
  support and disease-balanced summaries. Use the current saved bootstrap for
  uncertainty when available. Point-estimate counts are not confirmed harms.
- Whether OPERA improves the domains where joint learning loses to disease
  pooling. OPERA-versus-joint alone cannot answer that. This can be a descriptive
  tuning cross-tab; freeze any held-out comparison before inspection.
- The actual model portfolio, compute, storage and inference costs, so a
  consolidation claim can be numerical rather than rhetorical.

These checks do not require changing the frozen outcome definitions, rerunning
the entire training programme, or delaying every other writing task.

### Wait for the relevant evidence

- The router provenance/replay and source-removal reports the user is running.
- Refreshed native clinical indices and training-derived threshold manifests,
  with achieved evaluation risks, specificity and counts.
- Completed national/local/adapted models and the prespecified label-efficiency
  arms. Pretraining completion alone is not a downstream result.
- Recent-week omission and attribution validation with the correct checkpoint,
  token mapping and reference. Do not call ambiguous molecular prefixes verified.
- Frozen held-out results for confirmatory thesis claims.

### Before opening held-out results

If consolidation/non-inferiority will be an inferential claim, define its
comparator, target population, metric, weighting and justified margin now.
Do not pick the margin merely because it fits the tuning confidence interval.
Keep subgroup loss reporting alongside any aggregate test. Choosing an aggregate
non-inferiority question does not establish non-inferiority for every cell.

Keep the current primary analyses and add the new interpretation transparently
as tuning-motivated. Do not rewrite the historical rationale as though every
new question had been specified before the rehearsal. Test inspection should
not drive router selection, outcome removal, margin choice or vocabulary fixes.

## 7. Length, emphasis and final judgement

TeXcount reports 22,228 body words, 411 heading words and 953 caption words in
the active kappa. Its framework and discussion dominate the text. There is no
need to force a journal-style length on a thesis, but the repeated explanation
of borrowing dilutes the argument.

A useful editorial target is approximately a 15–20% reduction in synthesis
prose, chiefly by shortening the synopsis and removing repeated framework,
novelty and human-prior explanations from the discussion. Use the recovered
space for concrete results, a fair account of negative transfer and the
limitations of the interpretation. Keep one clear partial-pooling explanation
and refer back to it.

The thesis is stronger if it can say which assumptions survived its experiments
and which did not. Disease sharing can matter greatly while task-specific
tabular models remain stronger. Outcome sharing can permit a broad model while
redistributing performance. A router can be mechanically sound without revealing
a selective clinical map. National pretraining can be useful, redundant or
insufficient at this scale. Each is an answer to the same question about using
information, rather than a separate rescue or failure of the thesis.
