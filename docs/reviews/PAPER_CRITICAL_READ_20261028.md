# Critical manuscript read — 28 October 2026

Read against `paper/opera_body.tex`, `paper/supplement_evaluation_details.tex`, `paper/supplement_model_details.tex`, the current figure captions and relevant analysis/number-export definitions. This is an editorial and internal-consistency review. It does not verify clinical literature claims or certify the still-running empirical results. Locations refer to the current source and will move after editing.

> Follow-up edit: the manuscript now explicitly identifies local OPERA as R_AB_L/R_L, distinguishes J and the controls, uses O1 for OPERA-versus-XGBoost prose, corrects joint-model table labels and makes both atlas panels describe J. Those findings below record the pre-edit review; other editorial issues remain.

> Author direction after this review: retain all scientific questions and streamline their organization; target The Lancet Digital Health; keep clinical comparisons descriptive and late-diagnosis checks proportionate. See AUTHOR_DIRECTION_AND_ESTIMAND_20261028.md for the proposed balanced estimand and the distinction from the running analysis.

## Overall assessment

There is a substantial paper here. Its strongest contribution is a controlled study of sharing supervision across a disease-by-outcome grid in a coherent clinical setting, with strong tabular comparators and recorded clinical indices. That contribution survives a modest routing gain, a null external-encoder result, or tabular models winning some tasks. The paper does not need all its optional branches to succeed.

The current text is an ambitious manuscript scaffold, not yet a convincing completed article. This is more than the X placeholders: several parts describe different generations of the model and analysis. Its argument also accumulates claims faster than it discriminates between demonstrated performance, descriptive mechanisms and future clinical possibilities. These are repairable problems, but replacing X with numbers alone will not repair them.

My central recommendation is to make this the paper about **when sharing helps, what it costs, and how one shared model compares with practical alternatives**. Routing is an important controlled extension, not a result that has to rescue the paper. Population pretraining is another empirical branch, not the premise on which the paper's value depends.

## What already works

1. **The disease-by-outcome grid.** The S/D/O/J design translates a general machine-learning question into experiments that a clinical reader can understand. It provides a better contribution than merely naming a new architecture.
2. **A coherent prediction moment.** First-line treatment initiation is intelligible and clinically relevant. Keeping the prediction time common across outcomes makes the panel a defined problem rather than an arbitrary collection of benchmarks.
3. **A demanding comparator portfolio.** Per-task XGBoost, pooled tabular models, registry-enriched models and available indices allow different explanations to be tested. Keep these comparisons prominent even when they do not favour OPERA.
4. **The distinction between allocation and benefit.** Source weights versus source-removal effects, and a within-fit ablation versus training an own-source control, are valuable distinctions. The text already acknowledges many of them, although not consistently.
5. **The limitations are unusually substantive.** Pretraining overlap, single fits, temporal rather than geographic validation, registry delay, score availability and prediction under observed care are all discussed. These should be integrated into the relevant claims rather than left to neutralize stronger claims at the end.
6. **Useful negative results are allowed.** The factorial design can show that disease pooling matters more than outcome pooling, or that routing adds little. Those are publishable scientific answers if comparisons are fair and estimates precise enough.
7. **The prose is often clear and concrete.** The examples of infections, kidney function and common treatment records make the sharing argument accessible. Preserve this style while shortening the repeated explanations.

## Issues that must be reconciled before importing results

### 1. Which model does each reported number describe?

The manuscript says OPERA is the selected scope/routing/encoder configuration, while fixed-arm contrasts answer the scientific questions. This is a sound distinction. However, `final_contrasts_20261014.yaml` defines T2 = gain(J_L,X), T6 = gain(J_L,XP), and T9 = gain(J_L,XR). `build_paper_numbers.py` directly exports these contrast numbers; its table labels call them OPERA comparisons. The abstract and benchmark Results use T2/T6/T9 while naming OPERA. Separate OPERA contrasts O1/O2/O3 also exist. This is a concrete identity mismatch, not merely ambiguous writing.

Before filling the abstract or Table 2, make an identity ledger: displayed claim, exact contrast expression, model checkpoint/alias, split, horizon and weighting. Use explicitly named OPERA contrasts for the selected model and retain J comparisons under their true names. Do not silently relabel the fixed J estimand. The atlas also needs a deliberate decision: a J-derived atlas cannot be captioned OPERA if OPERA is R.

### 2. The router in Methods is not consistently the selected router

Main Methods around lines 287–325 say disease adapters were selected, outcome adapters added little, and the locked model uses disease adapters only. The supplement's general formulation repeats disease-only selection, then the variants section introduces the selected R_AB configuration and a global adapter outside the mix. Current analyses refer to disease and outcome banks and an own-only AB reference. These descriptions do not give a reader one unambiguous architecture.

Write one current architecture paragraph for the actual primary model. Then summarize the original screen and amendments as selection history in the supplement. A compact table should name primary R_AB, original router, global-only, own-only, joint and any other displayed controls; state banks, global path, warm-up, source restrictions and seed. Make the equation describe the selected model rather than requiring a reader to retrofit it several pages later.

### 3. The clinical comparison has changed

The new recipe compares every recorded index with OPERA on all patients with that index, at a common one-year overall-survival endpoint. It harmonizes display groups to Low/Intermediate/High. Main Methods and clinical Results still promise disease-specific populations, native endpoints, original groups, two-year outcomes, and DLBCL-only IPI manuscript panels. The supplement contains a separate native-endpoint/adaptive-horizon design. These are different analyses.

For the tuning draft, describe the current common-endpoint benchmark exactly. If native-endpoint analyses are retained later, distinguish them as another analysis with its own recipe and table. A category-only index collapsed from four strata to three is a coarsened comparator: retain original ordinal categories or numeric scores as a discrimination sensitivity where available. Keep three groups for harmonized displays if desired, but do not call that an exact reproduction of each index. The current IPI manuscript illustration pools all patients with recorded IPI; the old DLBCL-only captions must change unless an explicitly labelled DLBCL secondary illustration is intentionally added.

### 4. The draft reads as if the final test evaluation has happened

The abstract, key findings, temporal Methods, Results and discussion repeatedly state that patients treated from 2023 were evaluated once after every decision was fixed. The current run is a tuning rehearsal. A page banner is insufficient when the main prose states completed held-out findings. Keep a clean distinction between observed tuning results and the planned held-out design. The historical lock/amendment record should be factual and dated, not reconstructed as a single pristine prespecification.

### 5. Weighting and horizon language need one clear contract

Headline differences weight disease–outcome–horizon cells equally; router selection and borrowing are disease-balanced; figures often show a median of disease–outcome gains pooled over horizons; some panels need one year; endpoint groups combine grade variants; hierarchy/curve summaries use still other weights. Each is defensible for a different question, but moving between them can reverse an impression of benefit.

State the primary finite-panel estimand plainly. Put the headline mean with its patient interval beside the median/IQR and coverage in a small summary table, and identify every figure's weighting. Report how many cells and horizons each contrast includes. A pairwise common risk set does not guarantee a common cell set across contrasts. Use a common-roster sensitivity for the principal comparisons. The headline B caption still says pooled horizons even though the absolute-AUC scatter implementation uses a single horizon. Match it to the actual panel.

### 6. Intervals must describe the implemented uncertainty

The prose says 1,000 patient resamples. The refreshed implementation uses a frequentist positive multiplier bootstrap with IPCW weights held fixed. State both choices. The intervals condition on fitted models, selected hyperparameters, clinical cutoffs and estimated censoring weights. They do not include training, selection, cutoff or censoring-model uncertainty. A single no-source-versus-J contrast is not an estimate of the distribution of refitting variation; call it a diagnostic, not a general noise floor.

### 7. Statistical and practical importance are being blended

The claim rule first calls an interval above zero “better,” then marks the estimate above a practical margin if the point estimate exceeds 0.01. That does not establish that the true difference exceeds the practical margin; an interval wholly within the equivalence region can also be statistically positive. Present direction/evidence and practical compatibility separately. Do not make “better” sound synonymous with clinically meaningful. Explain which hypotheses actually enter the Holm family, how unavailable external/clinical comparisons are handled, and whether its size was fixed. The draft's pending analysis-lock fields need resolving before claiming a finalized family. Use net-benefit differences as net true-positive equivalents under a threshold trade-off, not observed additional events prevented or patients helped.

## The core argument: sharpen rather than expand it

The introductory story currently runs through fragmentation, multi-task learning, meta-learning, experts, foundation models, a clinical-score critique, the Danish resource and national transfer. Several paragraphs return to the same question. The opening also makes disease-specific modelling sound more uniformly naive than it is. Disease-specific definitions and care pathways have good reasons; the relevant gap is quantitative evidence for sharing under matched prediction conditions.

Reduce the introduction to four or five paragraphs:

1. A clinically defined problem: many treatment-start risks, few events per disease/outcome, existing tools cover a narrower set.
2. Sharing can help or hurt; existing evidence does not resolve the two supervision axes under this evaluation design.
3. Why this cohort and prediction point offer a useful test.
4. The exact controlled questions and comparators. Add national pretraining here only as a distinct question.

The detailed parade of large models can become a shorter synthesis with the most relevant citations. The resource description belongs largely in Methods. Novelty claims such as “has not been tested” and “first” need a dedicated citation check; this review has not independently established them.

Suggested contribution sentence:

> We tested whether sharing supervision across lymphoid cancers and clinical outcomes improves prediction at first-line treatment, separating the two axes of sharing and comparing shared models with task-specific tabular models and recorded prognostic indices.

This stays true if some sharing strategies fail.

## Abstract and title

The current title is clear, broad enough for the design and does not oversell architectural novelty. Keep it provisionally. The abstract is overloaded: cohort/outcome/fits counts, factorial effects, routing, national pretraining, label savings, missing indices, risk-group reclassification and attributions all compete. It asks the reader to remember too many denominators before the primary result appears.

Build the final abstract around one question, one design sentence, three numerical findings and a restrained implication. Prefer:

- the main shared-model versus strong baseline result with interval and cell coverage;
- the principal factorial finding, including an informative null;
- one clinical comparison on its actual scored population, or the transfer result if that is central and complete.

Move the total number of experimental fits, detailed router decomposition, individual attribution finding and extended label-efficiency result out unless one is the primary contribution. “69 outcomes for every patient” needs qualification for outcome eligibility and monitoring. The key-findings box currently repeats the abstract and promises the best possible version of seven results; remove it or reduce it to three verified statements.

## What the factorial design establishes—and what it does not

All supervised arms share cohort pretraining. Thus S is a supervision specialist, not a model that has learned nothing from other diseases: the pretrained representation already includes cross-disease information. Explain this early. D−S and O−S estimate the effect of changing supervised scope conditional on the shared pretraining and training protocol.

Likewise, factorial contrasts change the amount/diversity of supervised data and optimization conditions. They do not uniquely identify biological information transfer or distinguish it from regularization and representation learning. If exposure/optimization budgets differ, document that, rather than calling the arms otherwise identical without qualification. The broader record and different tabular feature windows also make the neural comparison a pipeline comparison, which the limitations correctly acknowledge.

The common comparison roster should be primary for claims comparing axes. Then report added evaluable coverage separately. Related grade thresholds and composite endpoints are correlated and partly redundant: patient bootstrap pairing addresses sampling dependence, but equal weighting still gives some biological phenomena more influence. Retain the endpoint/domain-balanced sensitivity and make its conclusion easy to find.

## Routing and borrowing: what would convince a reviewer?

The question should be “Does routing add useful predictive performance beyond a shared model, and what component of the fitted model is responsible?” It should not presuppose that selective clinical borrowing must emerge.

The three most useful pieces of evidence are:

1. R versus J on the same supported cells, with absolute AUC, paired difference, interval and calibration consequences.
2. Selected R versus appropriate global-only/own-only controls, with the differences in parameterization clearly stated.
3. Fixed-model cross-disease ablations, labelled as dependence under that intervention.

Do not subtract results from independently trained controls and conclude that a residual “came from borrowing” as a causal decomposition. These models have different optimized encoders and may interact. Similarly, the two train-time corrections and fixed-fit disease-only ablation answer different questions when the outcome bank remains active.

The retained global encoder and outcome bank must be described whenever “no borrowing” is used. Prefer “other-disease corrections replaced by tuning means.” A mean ablation removes patient-specific deviations relative to that mean; it is not an experiment removing all information about those source diseases. A small single-source AUC effect can reflect redundancy, saturation or metric insensitivity. Broad gates do not prove broad useful signal; broad ablation effects are different evidence. Partial pooling is an analogy supported by a shrinkage mechanism and a support pattern, not a demonstration that the router computes a hierarchical posterior.

The candidate-pair plan remains draft. Main Methods now says the pairs are descriptive, but the introduction and discussion still promise that pairs/negative controls were tested and that expected pairs suggest real shared clinical signal. Remove that contradiction. Do not promote or retire hypotheses according to whether the observed gate pattern fits the desired story. An amended exploratory analysis is legitimate; a retroactive confirmatory label is not.

The tiny shrunken heatmap values need direct interpretation. Show an unshrunk companion and state absolute scales. Shrinkage concentration is not evidence of biological absence, and normalized lineage shares can look substantial when all underlying gains are tiny.

## Clinical claims need a narrower bridge from prediction to use

The score-available same-patient comparison is the right contrast. Availability in a research registry is not the same as availability to a treating clinician. Late registry submission can make a score absent from the extract even when it was used in care. The finding that 58% lack a recorded category therefore does not by itself establish that those patients had no clinical risk assessment. Separate no dedicated index, no recorded score, no calculable score, model prediction available and model prediction validated.

The numerical output of a model for every cell is not validated clinical coverage. Some cells cannot be evaluated at all; five cases and five controls is a minimum analysis rule, not sufficient validation for patient management. Show three layers: predictions can be produced; performance can be estimated; estimates are precise enough to support use. Unsupported cells should remain explicitly unknown.

Low-risk groups at matched observed risk are a useful descriptive comparison, but optimizing their cutoffs on the same tuning patients makes the tuning gain optimistic. Report that prominently and do not claim held-out reclassification until evaluated. Where feasible, include the original ordinal score/numeric-score comparison to distinguish richer information from category coarsening.

For decision curves, mortality risk is not itself the benefit of closer monitoring. Define the action, eligibility, resource use and threshold interpretation; decision curves express utility under those assumptions, not observed benefit. The discussion's statement that monitoring/supportive care “asks for little” understates workload, false-positive harm and intervention uncertainty. Risk under standard care cannot identify patients who can safely receive less treatment. Avoid statements implying de-escalation or causal treatment guidance.

## Outcome ascertainment, prediction time and censoring

The fixed treatment-start time is a strength, but verify that the operational information time matches it. Disease assignment up to 90 days after treatment can encode information unavailable at prediction, particularly when routing consumes the fine disease label. The later-diagnosis sensitivity is essential, not merely one optional robustness check. State whether the input label, analysis subgroup or both use that assignment.

The laboratory monitoring risk set also deserves a conspicuous explanation: if eligibility uses future monitoring, the result is conditional on ascertainment, rather than unconditional toxicity risk for every treated patient. A recent-record gap does not eliminate selection due to how outcomes are observed. Show the selection/coverage diagram and retain monitoring-related limitations.

The preprocessing description says numerical bins/code counts use a random 90% of patients unrelated to the temporal split, while the vocabulary is restricted to pre-2022 events. Verify the date scope of the bin-fitting events before asserting that no future information shaped preprocessing. This is an audit question, not a conclusion that outcome leakage occurred.

Known-status BCE training versus full-risk-set IPCW evaluation is not inherently an implementation error, but the mismatch should be described precisely. The limitation that it “tends to overestimate risk” is too general without assumptions about censoring and the composition of known-status patients. Natural-proportion sampling does not ensure calibration either. Replace guarantees with the actual objective and report long-horizon calibration and the sampling sensitivity. Censoring weights held fixed give conditional intervals; avoid describing them as all sources of patient-population uncertainty.

## Population pretraining: valuable, but currently a separate promise

The discussion already admits that the encoders differ in population, vocabulary, data delivery and training budget. Main Methods nevertheless says the comparison measures what the rest of the population adds. That is not isolated by this design. It estimates transfer of the delivered encoder/pipeline, with adaptation as another intervention. Use that wording consistently.

Also verify pretraining calendar cutoff, vocabulary construction dates, inclusion of cohort individuals and post-index records, architecture compatibility and label-efficiency subset construction. A negative membership-inference diagnostic is not proof of no memorization or no leakage. Its scope should be described as limited evidence about the tested attack.

Do not allow an expected encoder arrival to dictate an unfinished central claim. If matched downstream results and calibration are not ready, keep the local paper coherent and mark the transfer branch as pending in the internal draft. A null transfer result can strengthen the message that specialized transfer is not automatic. A positive result still does not establish portability to every specialty.

## Interpretability and patient representations

Integrated gradients plus perturbation checks are stronger than attractive heatmaps alone. Keep the checks. But event removal can produce implausible records; a randomized-model check and baseline sensitivity do not establish clinical correctness. Restrict conclusions to model reliance under the chosen baseline and perturbation.

“Variance carried by” a source and “mean odds ratio” derived from attribution contributions require careful definitions: correlated sources and a constructed baseline do not provide a unique causal decomposition or conventional exposure odds ratio. Put the formula/normalization next to the display or in a clearly referenced supplement.

The synthetic patient is appropriately labelled. Treat it as an explanation of the interface, not empirical evidence that real individual explanations are faithful. Real aggregate examples and diagnostics supply the empirical evidence. The embedding maps are descriptive illustrations; cluster separation or projected correction arrows do not demonstrate improved risk prediction. These figures belong in the supplement unless they answer a question essential to the central argument.

## Structure and length

The main body currently contains ten result-figure calls in addition to the conceptual material and tables. Pooling, routing, hierarchy and borrowing each repeat portions of the same story; clinical comparisons appear in headline panels, their own section and several detailed figures. The external and interpretability branches then each restart a large question. This is too much simultaneous narrative burden.

Recommended main order:

1. Cohort, evaluable target coverage and the exact selected model.
2. Overall performance versus strong baselines, with discrimination and calibration.
3. What supervised disease/outcome sharing adds, including harm and support dependence.
4. What routing adds beyond shared learning, plus one restrained borrowing analysis.
5. Clinical index comparisons on scored patients.
6. Transfer only if complete and central; otherwise a compact secondary section.
7. A short robustness summary; then Discussion.

A practical main-figure set is study/grid design, headline comparison, factorial pooling, router/control comparison, clinical comparison, and optional transfer. Merge or supplement the absolute atlas, detailed hierarchy, source maps, attribution galleries and embeddings. Keep a master table of primary contrasts and uncertainty; let the supplement carry exhaustive cells.

The current Results asks questions repeatedly in headings. Those are useful while drafting, but the final headings should state supported findings. Do not write declarative headings until results are checked. Put the clinical illustration after the reader understands the sharing comparison, unless clinical superiority is genuinely the main finding.

## Discussion: less repetition, more decision

Several Discussion paragraphs restate the Introduction's fragmentation argument. The most valuable Discussion would answer: what empirically changed; why the axes behaved differently; what routing did or did not add; what coverage means; why transfer succeeded or failed; and what prevents clinical use now.

Cut the suggestion that disease-specific registries/classifications rest on a single mistaken assumption about prognosis. The study does not challenge their diagnostic or treatment role. Replace “boundaries need not be the boundaries” as a broad manifesto with a focused interpretation of the observed contrasts.

Do not portray choosing a lower-scoring router as incapable of selection bias. It may avoid selecting the highest point estimate, but post-result choice changes the target model and interpretability criterion. Transparency, frozen amendments before held-out inspection and reporting all relevant variants are the defensible argument.

The conclusion should survive three possible results:

- If shared neural models match rather than beat XGBoost: emphasize consolidation, supported coverage, calibrated risks and maintenance only to the extent measured; do not market superiority.
- If routing adds little or borrows broadly: sharing may matter more than sophisticated source selection. That is an informative result.
- If national pretraining adds nothing: transfer is conditional on domain/vocabulary/optimization, not automatically valuable because of scale.

## What evidence is still missing from the readable article?

These are decision checks, not a request to launch a new experimental programme.

| Question | Minimum needed | Already implemented or separate? |
|---|---|---|
| Which model and which patients produced the headline? | Model/contrast ledger, common-roster summary, supported-cell coverage | Existing outputs plus identity reconciliation |
| Is improvement practically meaningful? | Absolute AUC/calibration alongside paired differences and CIs, not only a percentage of positive cells | Existing metrics; improve display/prose |
| Does routing add beyond capacity and identity? | Clearly named controls, comparable risk sets; limited matched-seed evidence if available | Existing controls; do not infer general seed variability from one contrast |
| Does source removal reflect the selected model? | Verified inference provenance, fresh CPU summaries, mean/zero distinction | Targeted refresh after current run |
| Is the clinical contrast fair? | Scored-patient counts, original versus harmonized categories, exact endpoint/horizon, availability over time | Clinical pipeline plus an original-score sensitivity if needed |
| Are signals confined to recent diagnostic work-up? | Fixed-index 7/14/21/28-day gap sensitivity, stated as such | Implemented separately; plotting does not launch inference |
| Are findings driven by outcome ascertainment/censoring? | Monitoring/risk-set counts, long-horizon stability, known-status sensitivity | Existing analyses conditional on metadata/config; verify completion |
| Is late disease information influential? | Exclude later-assigned diagnoses, audit router disease-label timing | Existing sensitivity plus provenance audit |
| Does national transfer work? | Compatible checkpoint plus matched downstream results and calibration | Separate branch; no conclusion until done |
| Is consolidation actually cheaper? | Clearly defined deployment portfolio, inference/pretraining/downstream costs, uncertainty about maintenance | Compute inventory; do not substitute total experimental fit count |

The cost denominator matters: `narrow_fits` currently counts specialists plus disease-pooled and outcome-pooled fits. Calling that number “separately trained specialists replaced” is misleading. Report the number of actual task-specific models needed for the stated deployment target separately from all scientific experimental arms. GPU-hours and CPU-hours should not be compared as interchangeable walltime, and pretraining costs need an explicit amortization choice.

## Proposed editing sequence

**First: identity and claim reconciliation.** Resolve model names/macros, actual router formulation, clinical recipe, bootstrap, horizons and tuning-versus-test tense. These are blockers for trustworthy reading of the results.

**Second: structural edit.** Shorten Introduction; choose one principal contribution; merge routing/hierarchy/borrowing exposition; reduce main figures; move screen history into the supplement. Write an abstract skeleton without promises of every optional success.

**Third: result integration.** Import verified aggregates; replace each placeholder with an estimate, a justified statement that the analysis is inconclusive, or deletion. Each Results paragraph should contain finding, supporting comparison, size/uncertainty, and one limitation relevant to that finding.

**Fourth: discussion rewrite.** Start from what happened, including the strongest null/harmful finding. Cut repeated motivation, unsupported national generalization and untested clinical-action claims. Keep the useful limitations, but connect them to claims where they arise.

**Fifth: submission completion.** Check novelty/citations, TRIPOD+AI checklist, definitions/count release, ethics details, author/funding/involvement statements, repository/archive/license, compute/availability and figure/table crosswalk. This review is not a literature or legal audit.

## Suggested replacement language

Instead of “The model learned what to borrow, and from whom” as an established result:

> The router allowed predictions to depend on source-specific corrections. We tested whether this improved performance over the shared model and examined the fitted model's dependence on those corrections.

Instead of “OPERA predicted every outcome for every patient”:

> The shared model produced risks across the outcome panel; performance was evaluated on outcome-specific eligible patients, with unsupported disease–outcome cells reported separately.

Instead of “the national comparison measures what the rest of the population adds”:

> We compared transfer from the delivered population-pretrained encoder with cohort pretraining, and evaluated continued cohort pretraining as a separate adaptation step.

Instead of “one fit-versus-fit difference sets the noise floor”:

> Matched control fits assessed whether architectural changes alone could explain the observed difference. Patient-bootstrap intervals condition on the fitted models and do not quantify general training variability.

Instead of “the clinical index was missing, so no tool existed for those patients”:

> Recorded index availability varied across patients and periods. Comparisons were restricted to patients with each index recorded; absence from the registry extract does not establish absence of clinical assessment.

## Bottom line of the review

The study is worth finishing. Its best argument is the controlled comparison and the clinical evaluation, not the number of models, architectural complexity or an assumed win on every axis. The most urgent work is to make every part of the paper describe the same actual experiment. Then a restrained result—including null and harmful comparisons—will be much more persuasive than a manuscript that appears to have anticipated every success.
