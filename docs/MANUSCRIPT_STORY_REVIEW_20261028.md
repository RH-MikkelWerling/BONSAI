# Manuscript assessment and Lancet Digital Health comparison

28 October 2026. Review of the current `paper/opera_body.tex`, its main figure captions, Table 1 and the recently audited supplement. This is an editorial and scientific assessment, not a rewrite or a certification of pending results. Server figures and final held-out/national outputs were not available for inspection.

## Overall assessment

The manuscript has improved substantially. Its opening now identifies a clinical problem, the disease–outcome grid explains the controlled comparisons, and it distinguishes gains over neural specialists from superiority over strong tabular methods. Partial pooling provides a more defensible rationale than a claim that routing discovers which biologically related disease to borrow from. The discussion now recognises that accurate prognosis under observed treatment does not establish the safety of changing treatment.

Lancet Digital Health remains a reasonable target for the research question and study resource. The present draft is not ready for submission, and journal fit cannot yet be decided from tuning previews and placeholders. The main risks are an unfocused narrative, insufficiently established incremental clinical value, temporal-only validation, unstable small-subgroup estimates, and unresolved implementation provenance. A large national encoder may improve the results but does not itself resolve these issues.

The strongest framing is: **Can prognosis in a data-limited specialty improve by learning beyond a single disease and outcome, and when does that additional information help?** OPERA is an instrument for answering this question. Whether it becomes the principal named clinical model depends on the final evidence.

## Journal comparisons

These are a purposive comparison, not a systematic review. Full indexed article text was available for the RETFound comparison and substantial sections of TRisk. ECG-CLIP and METRIC-AF were assessed from their published abstracts. Some direct publisher pages and the journal-specific author PDF returned access errors. Exact current word, reference and display-item limits must therefore be checked from the submission system rather than inferred from third-party summaries.

| Published study | Editorial lesson for this manuscript |
|---|---|
| [Rao et al., TRisk, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12935155/) | Begins with an existing clinical decision and reports consequences in patient counts. Our strongest clinical example should similarly show what changes per 1,000 patients, while avoiding an unsupported treatment-benefit inference. |
| [Yew et al., RETFound comparison, 2026](https://doi.org/10.1016/j.landig.2026.101031) | Finds task- and sample-size-dependent advantages rather than universal foundation-model superiority. A mixed outcome can be publishable when it produces a useful, well-validated answer. Their external datasets are an important strength we do not currently match. |
| [Ko et al., ECG-CLIP, 2026](https://pubmed.ncbi.nlm.nih.gov/42680677/) | Connects broad task performance, independent external data and label efficiency. National pretraining here should likewise be judged by what it adds, including performance with fewer labels, rather than population size alone. |
| [Bedford et al., METRIC-AF, 2025](https://pubmed.ncbi.nlm.nih.gov/40930926/) | Keeps one clinical prediction problem central and connects calibration and validation to possible future prophylaxis studies. Our breadth needs an equally understandable use case. |

The journal examples use clinically anchored introductions, explicit study designs, structured summaries with Background/Methods/Findings/Interpretation/Funding, and quantified findings. The RETFound article includes Research in context and a Role of the funding source statement. The [Lancet family guidance](https://www.lancet.com/pb/assets/raw/Lancet/authors/lancet-information-for-authors.pdf) also describes the three-part Research in context panel. Our Key findings box does not replace that panel. Its evidence-search paragraph must be based on a documented search that was actually performed.

Do not copy the strongest deployment language in another paper. Our evidence should determine our claims. Retrospective discrimination, calibration and operating-point analyses are not prospective clinical benefit.

## Is the story coherent?

Yes, provided three levels remain distinct:

1. **Clinical breadth:** one system addresses disease control and complications across a specialty, with comparison against clinical indices and tuned tabular models.
2. **Controlled learning question:** matched neural models estimate the contribution of supervision from other diseases and other outcomes. These are the most interpretable sharing experiments.
3. **Extensions:** task-specific corrections and national pretraining test whether the shared model can improve further. Source removal and attribution help describe the fitted system.

The manuscript currently lets level 3 occupy too much space. A small router gain plus broad, weakly differentiated borrowing should not support a headline about discovering selective transfer. Nor should high gains over weak specialists establish that the resulting model is the best available predictor. Pooled XGBoost is a particularly important comparator because it tests whether neural gains reflect access to more labelled patients rather than a unique architectural advantage.

If national OPERA beats strong tabular methods, the paper can combine a clinical advance with controlled evidence about sharing. If it helps only in sparse tasks, that narrower result can be valuable if supported by fixed populations, calibration and precision. If it does not close the XGBoost gap, the story should become a candid evaluation of the benefits and limits of shared sequence learning. That would be a harder editorial case without external validation, but not an incoherent study.

## Read-through by section

### Title and abstract

The title is clear and does not depend on a model winning. A design descriptor such as “a nationwide model development and temporal validation study” could clarify the evidence type.

The abstract now starts from the value of prognosis, but spends too much space motivating and enumerating models before reaching a result. Keep a short clinical premise, the unresolved sharing question, design/population, two or three main numerical findings, and one restrained interpretation. Model acronym expansion and all three OPERA variants need not fit in the summary. State the evaluation population, uncertainty and strongest comparator directly. Do not give national joint-model effects the OPERA label.

Replace the Key findings box with the journal's Research in context structure when preparing submission. Do not invent search dates or retrospectively describe an undocumented search as prespecified.

### Introduction

This is the strongest improvement. The clinical opening is preferable to a broad claim that medicine has learned to separate diseases. The sequence from limited events to related tasks to wider pretraining now makes sense.

It can still be tightened. The partial-pooling and foundation-model paragraphs overlap in their explanation of sharing. The examples of record regularities need only one sentence. Keep the four questions as a drafting scaffold, but shorten their explanations. Source-removal and attribution details do not need to be previewed individually.

“Performance grows with data” should remain a motivation rather than a guarantee that other diseases supply useful information. Define data-limited tasks by their available events, not by describing the Danish health system as a low-resource setting. Partial pooling is an inspiration here, not proof that a neural router implements the inferential behaviour of a hierarchical statistical model.

### Methods

The supervision grid is now understandable. The distinction between disease-pooled, outcome-pooled and joint is useful and should remain in the main text.

The temporal-design paragraph still says every modelling choice used tuning and the test set “was evaluated once.” This conflicts with the revised supplement and unfilled final analysis. State the design and actual status accurately, then identify the final lock/amendments from manifests.

About 230 words on disease-correction removal is too much relative to the main evaluation description. Keep the fixed-model nature, mean replacement and retained components in two or three sentences, and move the three ablation conditions to the supplement.

Important scientific qualifications must survive shortening:

- Death is competing only for non-fatal outcomes, not for the overall-survival outcome itself.
- Clinical groups use fitted-model training predictions, not out-of-fold predictions. More training patients reduce small-sample instability but do not eliminate apparent-fit optimism or threshold uncertainty.
- A training-matched event rate or specificity is not guaranteed to remain matched in tuning or test patients.
- National versus local pretraining compares strategies that differ in vocabulary, source deliveries and exposure as well as population size.
- The main comparison population changes when a specialist cannot be fitted. Report coverage and paired accuracy separately.
- Source-removal averages use tuning patients. Document the fallback for target diseases without tuning patients and distinguish it from training-derived clinical thresholds.

A Role of the funding source statement is missing. Attribution/input-restriction methods also need a concise introduction in the main text if their results retain a large main-text section.

### Results: cohort and predictability

The temporal-shift paragraph is improved and appropriately brief. Clarify the antecedent of “Of these” after introducing the national population: 37,531 derives from the local DALY-CARE population, not the entire national pretraining population.

Table 1 still intentionally collapses risk categories into three bins. Its footnote discloses this, but it can be confused with the newly native clinical comparisons. Prefer availability summaries in the main table and a separate score-specific table, or label the descriptive harmonisation very prominently. Do not alter the approved aggregate export by inventing native counts.

The predictability atlas risks delaying the principal finding. One short paragraph can establish heterogeneity and support, with the full atlas moved to the supplement if it does not carry the central conclusion.

### Results: clinical and tabular comparisons

This should be the first substantive result after cohort description. Put the final model's XGBoost comparison and the main clinical-index finding together so readers do not have to choose between two apparently different success stories.

For DLBCL, give low-risk group counts, observed mortality and its interval, plus high-risk recall/precision and achieved specificity. Training-matched groups must not be described as proven to have equal held-out risk. An inconclusive difference is not equivalence.

“Beats every clinical index” requires named, supported comparisons, compatible endpoints/horizons, paired populations and uncertainty. The current shared one-year survival benchmark is distinct from showing superiority on every index's native endpoint. State the narrower claim if that is what was tested. Missing score availability limits generalisation. Show XGBoost on the same score-available patients where possible, or make its absence explicit.

The large Table 2 is useful as a complete inventory but burdens the main narrative. Keep a short table of the core contrasts and move the full portfolio to the supplement. A model-count or efficiency claim needs actual training/inference/storage measurements. One checkpoint is not automatically a cheaper clinical system.

### Results: sharing and routing

The matched supervision experiments are central. State which axis helps most and by how much, then quantify variation with training support and negative transfer. Confirm that comparisons combined in a sentence use the same cells or explicitly report different populations.

Specialist failure, absence of a tuning population and inability to estimate AUC are different conditions. The NMZL example should explain that distinction rather than implying it has no training patients. Five cases and five controls is a reporting support rule, not evidence that a cell is clinically validated.

Keep routing proportional to its observed benefit. Distinguish separately trained own-only/global-only models from mean/zero ablations of one fitted router. These answer different questions. A two-fit difference is a limited measure of fit variation, not a universal noise floor. Avoid a precisely interpreted crossover in training events unless it is stable and its uncertainty is reported.

The borrowing map should move to the supplement unless it adds a substantive, supported finding beyond broad dependence. Shrunken tiny source effects and lineage-normalised shares can look more biologically meaningful than their absolute size warrants.

### Results: national pretraining

This may become the principal extension, but reserve judgement. Report local, national and adapted national joint and routed models on common supported cells, with paired estimates. Distinguish fixed full-data performance from label efficiency.

A reduction to 10–50% of training patients does not necessarily create the same number of positive labels in every task. Report retained events as well as patient fraction. Interpolation between three fractions should not be presented as a precisely observed data-saving threshold. National pretraining also has a cost, so “label efficient” and “compute efficient” are different claims.

### Results: attribution, recent records and robustness

This is presently a second large story. Keep one clinically intelligible recent-record result and a concise validation statement in the main text. Most of the full attribution atlas, correlation maps and synthetic-patient display can sit in the supplement.

The reference contribution is not “timing alone.” It retains age, subtype, positions and a mean-content construction. Its covariance share is a signed decomposition relative to that reference, not a causal proportion of risk attributable to care timing. Do not interpret a mean IG contribution as the effect of a clinical finding on the outcome. A synthetic patient's invented attributions illustrate a display but are not empirical validation of an explanation method.

The last robustness paragraph is still an analysis checklist. Convert it to one actual finding with uncertainty, referring readers to the supplementary checks.

### Discussion

The discussion is much clearer and more restrained, but too much still follows the plan rather than the findings. The opening should contain the final two or three results, including the strongest comparator deficit if present. Remove anticipation of what national pretraining might show from the submitted version.

Keep one paragraph on clinical meaning, one on the sharing experiments and partial pooling, one on national transfer if informative, then limitations and implications. “For the first/second/third question” can disappear once the ordering itself is clear.

The limitations still say the small tuning year limits risk-group threshold estimation, although the implemented clinical cutoffs now use training data. Replace that with the actual issues: small evaluation strata, apparent training predictions, threshold uncertainty and model selection on tuning data. Include preprocessing chronology if the artifact audit reveals exposure beyond the stated boundary.

The conclusion should state what information sharing achieved and its limits. It should not require an OPERA-versus-XGBoost victory to make grammatical or scientific sense.

## Format, length and display items

TeXcount estimates, excluding figure captions but including draft placeholders, are approximately 625 words for the Introduction, 1,518 for Methods, 2,259 for Results and 1,089 for Discussion: about 5,500 words across those sections. Main captions add about 1,900 words. Macro expansion, unfilled results and the separately included table make these estimates unsuitable as final submission counts.

The body includes ten main figures and two main tables once the included Table 1 is counted. The reading layout works as a review document but is not yet a compact journal article. This is an editorial concern independently of the exact current journal limits, which were not verified.

A working shorter version could retain five or six figures: study/shared-learning design; clinical and tabular performance; controlled pooling/routing effects; national transfer and label efficiency; and, only if informative, recent-record dependence or one explanation figure. Detailed routing diagrams, full borrowing maps, outcome atlases and embedding galleries can remain in the supplement. This is a proposed editorial target, not a quoted journal cap.

Several captions still say median/IQR despite the intended paired mean/CI comparison summaries. Fix captions against actual refreshed outputs rather than globally replacing every median: descriptive distributions may legitimately retain medians and IQRs. The transfer figure still mixes means, medians and an interpolated crossing. Use one clearly defined estimand for each claim.

## Highest-value next steps

1. Collect the aggregate manifests and final figures, then write a one-page results ledger: contrast, split, population, cell set, estimate, interval and allowed claim. Resolve provenance before polishing the headline.
2. Choose one principal conclusion and two supporting conclusions from those results. Keep national pretraining and routing proportional to what they add.
3. Finish the NCCN-IPI case study and all supported native-score comparisons. Prioritise precision and achieved operating points over a large list of classification metrics.
4. Freeze the development and evaluation boundary. Record all tuning-informed amendments transparently and stop adapting the analysis to final test findings.
5. Strengthen transportability evidence if feasible. A genuinely independent cohort is more informative than another internal model variant. Region-specific evaluation of a nationally trained model is not external validation. If external data are unavailable, keep claims explicitly temporal and Danish.
6. Shorten the main article around those findings, replace the Key findings box with Research in context, and complete funding, contributors, interests and data/code access statements.

No manuscript text was rewritten in this review. The aim is to choose the evidence-supported story before another broad prose pass.

## Follow-up implemented after author feedback

The author clarified that the unifying story is information sharing: whether,
what and how to borrow. The abstract, introduction, question outline and
discussion now use that framework. The introduction and attribution results
were shortened. The outcome atlas, full disease-borrowing map and attribution
plate were moved to the supplement with their references retained. The main
text now distinguishes pending test evaluation, training-derived thresholds,
reference-dependent attribution and comparison-specific cell populations.
Research in context and the funding-role statement have explicit unfilled
components rather than invented search or funding details. Patient-count
reporting is specified per comparison, alongside achieved rates and specificity,
without double-counting overlapping index populations.

The existing scientific review above records the draft before those changes.
Pending numerical claims remain pending. Server access is available only to the
author. `docs/PAPER_REVIEW_BUNDLE_20261028.md` gives a CPU-only assembly and ZIP
workflow for the currently completed outputs and later refreshed bundles.
