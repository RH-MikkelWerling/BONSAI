# Lancet Digital Health: manuscript and figure review

29 October 2026. Fresh review of five full research articles, the current OPERA manuscript source, a newly compiled reading copy, and the available figure designs and tuning outputs.

## Overall assessment

The manuscript has improved substantially. It now has one recognisable argument: whether prognosis benefits from information learned outside the disease–outcome task being predicted, and how to retain useful specialisation while sharing. The strongest features are the matched supervision comparisons, the clinically coherent treatment landmark, the national cohort, strong tabular comparators, and evaluation of several risks that clinicians consider together.

The paper is a plausible Lancet Digital Health submission, but the current materials do not establish that it is ready or that this is its strongest eventual venue. The remaining challenge is evidential as well as editorial. The reader needs to see a clinically worthwhile result, understand what the controlled comparisons establish, and recognise where sharing fails. A large outcome panel and an elaborate router are insufficient by themselves. Independent validation is also a substantive weakness relative to several of the examples below.

My preferred central claim is conditional on the held-out results:

> Prognostic information need not follow diagnostic boundaries. Controlled comparisons across lymphoid cancers show which forms of sharing improve prediction, where a common model preserves useful specialisation, and how those predictions compare with practical alternatives.

This remains a coherent question if XGBoost wins. It becomes a weaker clinical-model adoption story in that case. National pretraining may strengthen the result, but should not be treated as the event that makes the existing argument valid.

## What was reviewed, and important version limits

- Read the full main text of the five articles below, including their methods, results and discussion, and examined their figure presentations. This is a purposive editorial comparison, not a systematic literature search or proof of novelty.
- Read `paper/opera_body.tex`, including its abstract, Research in context panel, Introduction, Methods, Results, Discussion and back matter. Checked relevant supplementary evaluation methods and figure captions, the current Table 1, and the preceding router-mechanism audit.
- The open reading PDF was compiled on 28 October; the shared manuscript body changed on 29 October. Compiled the current source successfully into `scratch/ldh_article_review_20261029/reading_current/opera_manuscript_reading.pdf`, without overwriting the user's PDF or modifying manuscript sources.
- The reading wrapper points to `paper/figures/tuning_preview/` and `paper/tables/tuning_preview/`. Both are empty. The newly compiled PDF therefore still contains result-figure and numerical placeholders. It is a current **prose** reading copy, not a completed results bundle. Its 79 pages include the supplement despite the wrapper's comment describing main text and references only.
- Examined the available redesigned headline, benchmark, pooling, hierarchy, national-pretraining, clinical, patient-map, borrowing, routing, support, attribution and calibration-map images in `scratch/paper_reading_20261028/graphical_audit_20261028/redesign_20261029/`. Their statuses differ. The national, native-category clinical example, patient map, attribution atlas/structure, recent-record-gap and calibration-map examples inspected here say **SYNTHETIC LAYOUT CHECK — not results**. They demonstrate layouts, not observed effects. The headline and several pooling/router/support plots contain downloaded tuning summaries and say **TUNING REHEARSAL**.
- The separately embedded definitions PDF has its own build history. Recompiling the manuscript does not prove that this embedded appendix reflects every subsequent code-label clarification. Check its receipt when assembling the final bundle.

The review does not inspect the test set, change endpoint definitions, select a new router, or alter the ongoing server run. Article downloads, text extractions, word-count inputs and the isolated build are under `scratch/ldh_article_review_20261029/`. A reading receipt and figure plan accompany this document.

## Five journal articles and the useful comparisons

### 1. Foresight — Kraljevic and colleagues, 2024

[Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11220626/), DOI 10.1016/S2589-7500(24)00025-6.

A broad EHR forecasting study with a clear platform identity. Patient timelines and concrete forecast examples make the technical capability legible. The main figures explain the pipeline, predictions and errors rather than presenting an exhaustive model portfolio. However, models are developed and tested within the constituent datasets; this should not be described as transporting one frozen model across them. Next-concept forecasting is also different from calibrated cumulative incidence after treatment.

**Application to OPERA:** one model predicting many outcomes is established territory. Our more distinctive contribution is the controlled separation of supervision sources, coupled to specialty-wide treatment prognosis and strong comparators. Borrow the concreteness of the examples, not expansive claims about clinical deployment or causal understanding.

### 2. TRisk — Rao and colleagues, 2025

[Full text, accepted manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12935155/), [published article record](https://pubmed.ncbi.nlm.nih.gov/40461349/), DOI 10.1016/j.landig.2025.03.005.

The clinical selection problem anchors the paper. Calibration and risk distributions accompany discrimination, and reclassification is made tangible through people selected, events captured and events missed. Validation includes held-out general practices, which is meaningful separation but does not amount to a wholly independent health system. The accepted manuscript and final abstract have small numerical differences, so this review does not transfer their counts into our draft.

**Application to OPERA:** present additional low-risk patients, their observed mortality, high-risk events captured and false positives together. Our richer XGBoost comparators are a strength to preserve. Unlike a prevention setting with an established intervention framework, retrospective prognostic reclassification here cannot establish that reducing treatment is safe.

### 3. Breast-cancer mortality — Clift and colleagues, 2023

[Published full text](https://pure-oai.bham.ac.uk/ws/portalfiles/portal/280232967/CliftAK2023Predicting.pdf), DOI 10.1016/S2589-7500(23)00113-9.

The paper is organised around predicting a consequential clinical endpoint, rather than demonstrating neural-network superiority. Regression models perform best. Internal–external validation, calibration and clinical context carry the contribution. Performance and calibration are reported explicitly across relevant populations, and further impact evaluation remains necessary.

**Application to OPERA:** an unfavourable neural comparison can be a scientifically useful result if the paper answers a valuable clinical question and evaluates alternatives fairly. This does not mean that any complex model losing to XGBoost is publishable. We must explain what the sharing experiment teaches and what practical value remains. A compact performance table plus readable calibration is more persuasive than another dense embedding panel.

### 4. GSU-Pulmonary Score — GlobalSurg/STARSurg, 2024

[Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12056793/), DOI 10.1016/S2589-7500(24)00065-7.

A specific preoperative prediction time and clinical purpose make the study easy to follow. Similar performance from more complex modelling supports a practical simpler score. Distinct validation cohorts and transparent calibration weaknesses matter more than a dramatic discrimination claim. The calibration figure makes deterioration between cohorts visible rather than concealing it within one overall statistic.

**Application to OPERA:** state when and for what a prediction could be used, show failure as clearly as success, and distinguish discrimination from probability accuracy. Its independent validation is stronger evidence of transportability than our regional stratification. We should not copy its outcome-handling choices merely because the presentation is effective.

### 5. STAMPEDE multimodal prognostic model — Parker and colleagues, 2025

[Published full text](https://eprints.gla.ac.uk/357809/1/357809.pdf), DOI 10.1016/j.landig.2025.100885.

A locked model is evaluated in an important advanced-prostate-cancer setting using trial cohorts. The most useful visual comparison is added prognostic separation within familiar clinical groups. Selection into the biomarker cohort is disclosed. Retrospective prognostic separation is not the same as demonstrating treatment benefit. Some figures are extensive, so these examples do not support a rule that every Lancet figure must be minimalist.

**Application to OPERA:** make NCCN-IPI recognisable, then show what the model adds within and across its native categories. A simple display of group sizes and observed risks may communicate this better than age-stratified heatmaps. Borrow that clinical anchoring without introducing new, test-derived cutoffs.

### What these examples jointly suggest

This is my editorial inference from the selected papers, not a journal rule. Their strongest common feature is a legible connection between the clinical problem, the evaluated comparison and the conclusion. The model architecture serves that connection. Validation, calibration and the consequences of classification receive substantial attention. A new architecture is neither necessary nor sufficient.

Our opportunity is unusually good experimental separation within one specialty. Our disadvantages are the unfinished empirical account, weak source-specific interpretation, limited rare-cell precision and lack of an independent validation setting. Graphic polish can improve communication, but cannot remove those disadvantages.

## What has already improved

The previous review should not be applied unchanged. The current draft already:

- Uses the information-sharing story throughout and enumerates three questions.
- Explains the treatment landmark, competing events, the supervision grid and partial-pooling inspiration more clearly.
- Distinguishes cohort and national pretraining and acknowledges that their comparison is not a pure population-size intervention.
- Has a Research in context panel, albeit with unfinished evidence-search and findings sections.
- Describes training-derived clinical thresholds and native categories. It no longer relies on fitting these groups to the small tuning year.
- States that predictions have not been recalibrated on test data.
- Moves the representation, attribution and detailed borrowing maps to the supplement, leaving seven main figures.
- Separates clinical-index comparisons from XGBoost performance, and explicitly rejects treatment de-escalation claims from retrospective low-risk classification.

These are substantial improvements. The next revision should build on them rather than reopen every wording decision.

## Does the current figure set answer the questions?

| Question | Evidence that answers it | Current presentation | Main concern |
|---|---|---|---|
| Does additional supervision help? | Matched specialist, disease-pooled, outcome-pooled and joint fits | Fig 5; full contrasts in Table 2 | This is central evidence but the actual four-model supervision grid is supplementary. |
| Are these predictions competitive? | Same-patient and paired common-cell comparisons with per-task and pooled XGBoost, plus calibration | Figs 2–3 | Repetition, incomplete headline panels, and caption/image mismatches obscure the answer. |
| What does this mean clinically? | Native clinical-index comparisons, observed group risks, counts and achieved operating points | Fig 4 and score supplement | Synthetic redesigned examples are not the refreshed clinical results. Low-risk expansion is not yet established generally. |
| What does specialisation add? | Joint versus full corrected model; separately trained global-only, own-only and full controls | Fig 6 and router supplement | The full-versus-joint difference is called a routing effect although several components differ. |
| Does borrowing beyond haematology help? | Local/national/adapted joint and routed arms; matched label fractions | Fig 7 | The inspected image is synthetic. The label-efficiency experiment concerns joint models, not every OPERA arm. |
| What information does the fitted model use? | Fixed-fit restrictions, attribution checks and carefully scoped source removal | Supplement; brief Results account | These describe dependence, not a causal mechanism or an independent validation of sharing. |

### Figure 1: retain a design figure, change its centre of gravity

The current figure is clean and substantially more readable than the model descriptions it replaces. However, it spends much of a page teaching clinical scores versus tabular versus sequence modelling. The paper's central experiment is the change in supervision scope, and that four-model grid remains in the supplement.

Bring the specialist/disease/outcome/joint grid into the main design figure. Add the treatment landmark and temporal split, then show local versus national pretraining as different sources feeding the same downstream experiment. A small schematic of shared and task-specific corrections is sufficient for OPERA.

Concrete issues in the current source/render:

- Tabular lookback labels are 365, 180 and 30 days, while Methods specifies 90 days, one year and three years. Match them or explicitly label the schematic illustrative.
- The tabular panel says one task, while the paper evaluates pooled XGBoost. Sharing is a modelling choice available to tabular methods too.
- The scaling panel implies more information produces higher attainable performance. The paper specifically tests when additional information is useful or harmful. Avoid a guaranteed improvement arrow.
- A generic three-group clinical illustration should not be mistaken for the native NCCN-IPI categories used later.

### Figure 2: headline — simplify the claim and reconcile the actual image

The tuning image is visually accessible, and the OPERA-versus-XGBoost scatter gives an honest impression of the comparison. It is not consistent with the current caption. Panel A shows gains over three references across horizons, while the caption promises one-year absolute AUC for every available model on a common roster. Its annotations name 95% intervals but the numeric intervals are not printed beside the means. Panels C and E are missing pending clinical and subgroup exports.

Choose one role: the overall performance and coverage of the competing strategies. Include relevant joint and OPERA variants and both XGBoost strategies, but avoid turning every development control into a headline model. Reserve the original router for a clearly identified sensitivity analysis. Report paired effects with intervals in a compact main table. State the common roster and how much coverage it excludes. An all-model intersection must not silently remove difficult tasks or change the population every time another model is added.

Clinical reclassification deserves its own figure. Recorded-history effects can remain supplementary. Horizon curves need uncertainty or an explicit descriptive label. Do not use the fraction of cells with a positive point estimate as a fraction of proven successes.

### Figure 3: tabular benchmarks — important evidence, currently partly redundant

The domain forest and the support plot are useful. In the inspected image the domain forest says mean and 95% CI; the manuscript caption says median and IQR. The legend also omits a registry-variable contrast that the caption promises. These are substantive definition mismatches, not spacing issues.

The two reliability examples currently show a single visible XGBoost bin for two ALT definitions in AITL. That does not demonstrate comparative calibration, despite the caption promising three models. It might reflect missing prediction exports or insufficient bin support. Use the existing fixed selection rule and report any inability to populate it honestly. If the rule itself needs amendment, settle it on the tuning rehearsal before test inspection. Two nested definitions from one small disease are a poor clinical illustration unless they are the necessary result of a justified rule.

Calibration slope medians can conceal serious overprediction and underprediction in different cells. Retain a readable conventional predicted-versus-observed figure with uncertainty and risk distribution, plus slopes and calibration-in-the-large in the numerical supplement. Pooled-cohort calibration can be helpful, but case mix can conceal within-disease problems. Embedding-space calibration surfaces are supplementary diagnostics, not substitutes.

Combine overlapping overall benchmarking material from Figures 2 and 3. Keep a separate calibration figure only if merging would make it too small to read.

### Figure 4: clinical indices — preserve this main-text role

This is the most direct bridge to clinical practice. The redesigned four-category NCCN-IPI layout has legible legends, line styles and risk tables, and correctly preserves the native categories. Its inspected numbers are synthetic, so they cannot resolve the latest patient-count concerns or support the Results narrative.

The main display should establish the comparison population, show model and index group sizes, observed one-year mortality with uncertainty, and the achieved high-risk sensitivity, precision and specificity. A compact within-index reclassification panel can show added separation. Age-specific heatmaps and the full precision–recall annotation stack belong in the supplement if they crowd out these answers.

The current real reconciliation is a warning against assuming the hoped-for headline: among 343 NCCN-IPI-scored tuning patients, the index has 24 low-risk patients and the model 26, with one observed death in each. The broader computed-IPI comparison has 214 versus 595 low-risk patients, but 5 versus 38 deaths. Those raw fractions do not establish matching evaluation mortality. High-risk achieved specificity also differs after freezing thresholds. These are rehearsal results, not a reason to retune to force an attractive test result.

Training matching defines an operating rule. It does not guarantee equal observed risk or specificity in a later population. Include uncertainty and describe the achieved comparison. Do not add patient counts across overlapping indices as if they represented distinct people. Comparisons outside an index's native endpoint or indication must be labelled accordingly.

### Figure 5: pooling — strongest current figure for the central scientific question

The image shows a useful combination of disease size, domain effects and training support. Panel B now has paired uncertainty, resolving the earlier omission. Disease sharing appears to account for much more of the gain than outcome sharing, while the support plot makes the low-data question concrete.

Keep this experiment prominent. Reduce duplication between its four panels if needed, but do not lose the factorial comparison. Make the same-cell roster explicit when comparing axes. A difference between two separately averaged contrasts on different rosters is not the incremental effect of adding one source.

The four scopes also answer different transfer questions. With specialist S, disease-pooled D, outcome-pooled O and joint J, D−S and O−S describe adding each source alone. J−O describes disease supervision when outcomes are already shared, and J−D describes outcome supervision when diseases are already shared. These conditional effects are particularly important for the observation about preserving performance across outcomes. The factorial interaction J−D−O+S is a performance contrast, not an additive decomposition of biological information. All require a common supported roster for direct comparison.

The heatmap uses domain medians and the forest uses means. This can be defensible, but it needs a simple explanation. The support curve is a kernel smoother in tuning and a proposed hierarchical fit on test. Settle and rehearse the final analysis before opening test data. A smooth trend across noisy cells does not establish a clinically meaningful event-count threshold. Grey cells and the evaluation denominator are part of the result, not visual clutter.

### Figure 6: corrections and routing — useful, but the interpretation needs restraint

The control comparisons are worth keeping. Their support dependence could be an interesting secondary result: local corrections may help more when a task has sufficient supervision, while broader corrections can be more useful when it has little. The current plotted quadratic crossing near 84 events is much more precise-looking than the evidence warrants.

The tuning rescue panel classifies cells as harmed and evaluates recovery using the tuning rehearsal. That is not held-out evidence of rescue. The planned test version, using tuning-defined strata, answers a stronger question. Mark the distinction prominently.

The lower forest shows median and IQR, which describes heterogeneity rather than precision of the average effect. Use paired mean intervals for the main inferential claim and retain the distribution where useful. Explain that the global-only, own-only and full controls are separately trained fits. These are not three exact additive slices of one final prediction, and AUC differences cannot be interpreted as additive patient-level contributions.

Most importantly, call the joint-versus-full result an effect of the corrected architecture. The reviewed bundle does not isolate the benefit of patient-dependent source selection against a matched trained uniform-mixture model. Do not make that stronger claim. A new architecture search is unnecessary if the claim is appropriately limited.

### Figure 7: national pretraining — right question, entirely pending evidence

The synthetic design usefully separates local, direct national and adapted national encoders, with joint and routed variants. It belongs in the main paper if the completed comparison has an interpretable result, whether positive, null or harmful.

Lead with paired differences and intervals on a declared common roster. The current mix of absolute mean AUC, median AUC, median paired differences and IQR summaries makes the panels harder to compare. Label the reduced-training experiment as joint-model label efficiency. Prefer estimates at the actual 10%, 25%, 50% and 100% fractions to a highlighted interpolated claim of matching with approximately 37%. Equality of point estimates is not equivalence.

Retain the existing qualification that national versus local pretraining changes vocabulary, record delivery and exposure as well as population scope. The experiment tests a transfer strategy, not population size alone. The filename `fig_external` should not lead readers to mistake national pretraining for external validation.

### Supplement: keep explanation proportional to evidence

- **Support/coverage:** keep the complete grid and marked estimability rule. The support figure is much better at showing the actual small-data setting than a matrix restricted to successful cells. Five cases and five controls permit an estimate; they do not validate a clinically dependable tool. Report the fraction of tasks and patients represented.
- **Router usage:** broad outcome-bank mass is partly a consequence of 69 outcome versus 24 disease sources and the gate floors. Include a flat-allocation reference. Gate weight is not risk contribution or predictive usefulness. The zero-correction slot is not the entire shared encoder. Avoid a legend that merges these ideas as “own sources/shared”.
- **Borrowing maps:** the mean-replacement map retains outcome and global terms. Its scope is other-disease patient-specific corrections, not all borrowing. Show unshrunk uncertainty alongside shrinkage. Weak pair identification is a meaningful limitation, not something to repair by exaggerating the colour scale or selecting attractive pairs.
- **Representations/arrows:** no main-text mechanism claim should rest on these. An identical hidden-vector shift can have opposite effects through different outcome heads. An arrow need not point towards an observed high-risk region. Independently fitted projections cannot be read as comparable coordinates. Actual final predicted risk is a better companion than event markers alone.
- **Attribution atlas:** useful audit material, but the inspected redesigned atlas is synthetic and too tall for one readable ordinary appendix page. Split by domain with repeated legends, retaining one coherent overall summary. Labels such as odds ratios describe changes in model output, not epidemiological associations or causal effects. Preserve uncertainty about truncated Danish codes rather than inventing a fine-grained concept.
- **Attribution structure:** the agreement matrix and input-restriction analysis answer different questions. They can be separate figures. Similar attribution within nested laboratory definitions is partly expected and is not independent confirmation of a discovered biological structure.
- **Recent-record gaps:** a clinically relevant robustness check with a simple paired-effect display. Removing the last weeks of records at the same treatment landmark does not establish performance at an earlier prediction date or under retraining on shorter histories. The inspected equal-effect layout is synthetic; do not interpret its apparent uniformity across domains.

## Section-by-section critique of the current draft

### Title, abstract and Research in context

Use a clinical and design-specific title. A working direction is **“Sharing prognostic information across lymphoid cancers and outcomes: a nationwide temporal validation study”**. This is a framing option, not a confirmed final title. Name OPERA in the title only if the model itself, rather than the controlled sharing question, becomes the main contribution.

The opening now establishes the value of prognosis clearly. The abstract still spends too much space explaining three forms of borrowing and the architecture before presenting evidence. Convert it into the journal's five-part Summary. Allocate roughly 40 words to Background, 80–90 to Methods, 110–125 to Findings and 30–40 to Interpretation, plus Funding. Final numbers must replace drafting instructions rather than being added on top.

The Findings should contain the principal supervised-sharing effect, the competitive baseline result, the most meaningful clinical-index comparison and the national-pretraining result. Do not list every model arm. If outcome sharing mainly preserves performance, state its estimated difference and uncertainty rather than calling it beneficial by assumption.

The Research in context structure is now correct. Its evidence-before section still requires a reproducible search. Reading these five journal examples does not complete that task. Added value should identify a demonstrated finding, not merely describe the experimental programme.

### Introduction — retain most of it, sharpen the final map

The opening clinical paragraph is much stronger. The explanation of why more same-disease patients may be unobtainable is persuasive. The injury example usefully foreshadows national pretraining, although it can be compressed. Positive and negative transfer are now introduced through concrete examples rather than an extended tutorial.

The three questions work, but “whether” and “what” overlap. Keep the memorable wording while giving each a distinct experimental answer: whether supervised sharing helps, which sources add information, and whether task-specific correction/adaptation improves its use. End with those questions. The subsequent list of source removal, attribution, temporal and subgroup analyses can move out of the final paragraph.

Be fair to clinical indices. Their purpose includes portability and use with a few routinely available variables. Their developers did not necessarily assume that all other diseases were uninformative. The opportunity is to test what richer records and shared supervision add, not to imply that earlier clinical reasoning was naive.

The present introduction is about 516 prose words. It is not the main source of excess length.

### Methods — accurate direction, more separation of essential design from implementation

Keep the population, treatment landmark, predictor availability, temporal split, competing-event estimand, supervision scopes, fair comparators, clinical cutoff provenance and primary summary/uncertainty method in the main paper. Explain the 69 outcomes versus 52 loss-weighting families in one sentence, with the exact aggregation in the supplement.

Important precision points:

1. Every matched local sequence model begins with the same cohort-pretrained encoder. A specialist is therefore not a no-borrowing model in an absolute sense. The factorial experiment measures additional cross-task **supervision during fine-tuning**, conditional on shared pretraining.
2. OPERA is inspired by partial pooling. Do not imply that its gate weights implement classical uncertainty-derived hierarchical shrinkage or establish an optimal bias–variance trade-off.
3. Clinical eligibility should follow each index's validated disease setting, not merely whether a field happens to be recorded. The current broad wording at body lines 281 onwards needs alignment with the audited configuration.
4. The bootstrap description should match the implemented patient multiplier/weighting scheme. Patient-sampling intervals conditional on fitted checkpoints do not include all fitting, model-selection or training-threshold uncertainty.
5. Thresholds derived from apparent training predictions are permitted by the present design but can be optimistic. Their evaluation must be external to that derivation, with achieved operating characteristics reported. Do not imply the larger training population eliminates this issue.
6. A primary summary across equally weighted cells is not the AUC of a randomly selected patient across the whole cohort. State the target of the summary. An all-cohort analysis is complementary and can be affected by between-disease case mix.

The source-removal main-text subsection can be folded into a short secondary-analysis paragraph. The exact mean replacement, ownership conventions, shrinkage, classification metric catalogue and selection rules belong in the supplement. Move Role of the funding source to the end of Methods as the guide requests.

### Results — move from a comparison inventory to a sequence of answers

The current draft is an honest scaffold, not a finished empirical account. Many placeholders still instruct the writer how to interpret future findings. Do not leave these as additional paragraphs after the numbers arrive.

Suggested order within the three-question story:

1. Population and evaluation coverage, briefly. Distinguish the local pretraining population from the supervised cohort and the national source.
2. Does sharing supervision improve prediction, and which axis supplies the improvement? Lead with the matched factorial result and uncertainty.
3. Are the resulting models competitive and clinically informative? Give XGBoost/pooled-XGBoost results and then native-index comparisons in patient terms.
4. What do task-specific corrections add? Separate the overall comparison from own/global/full controls and any support dependence.
5. Does national pretraining extend the available information? Include only completed model arms and the correctly labelled joint label-efficiency experiment.
6. One short robustness/input-dependence paragraph containing the most important finding. Move the rest to the supplement.

This order makes the source of the claimed sharing effect clear before asking the reader to judge adoption. A different order can work, but the main paper should not begin with an underdeveloped predictability atlas and then march through every contrast identifier.

Specific corrections needed regardless of result direction:

- Body lines 463 onwards say “at 90% specificity” although thresholds are frozen on tuning. Report the tuning target and achieved evaluation specificity separately.
- The NMZL example at lines 602–605 links zero tuning patients to having no model. Table 1 has five training patients, zero tuning and 22 test patients. Explain the actual fitting/selection eligibility failure. Lack of tuning patients is not the same as absence of a training cohort.
- “Routing changed AUC” at lines 633 onwards should describe the corrected model versus joint. It does not isolate the routing algorithm.
- “All borrowed corrections switched off” at lines 680 onwards overstates the disease-bank mean-ablation scope.
- Do not require a largest source-pair discovery in the prose if the estimates remain weak. “No individual source was reliably identified” is a possible informative result.
- Use a single fixed display/analysis convention for final support curves. Rehearsal should test the procedure planned for the test set, not a materially different one.

Table 2 currently contains 15 contrasts, seven columns and an extended inference-rule caption. Keep approximately six to eight core contrasts in a readable main table and move the complete portfolio to the appendix. Retain a row for national OPERA when available, without substituting a national joint-model effect. Report model counts and resource savings only when measured on comparable terms, including the cost of pretraining/adaptation where relevant.

### Discussion — stronger now, still too repetitive and partly anticipatory

The distinction between improved sharing and beating XGBoost is excellent and should remain. So should the distinction between a favourable retrospective prognosis and safe treatment reduction. The diagnostic-boundaries argument is the most interesting broader implication if the held-out contrasts support it.

Open with the actual principal findings, then discuss the three questions in the same order as the Introduction. Avoid spending another paragraph restating that everything is borrowing. Use the results to explain what changed in our understanding.

The outcome-sharing observation deserves more attention if confirmed. Preserving performance across many outcomes with one model can be useful even without an average AUC increase. However, intervals spanning zero do not show equivalence or absence of negative transfer. Report the prespecified margin, uncertainty and distribution of harmful effects where supported. Consolidation is a possible practical benefit, not evidence of cost savings until resource measurements are available.

The router paragraph should focus on the measured balance of shared and task-specific corrections. Broad averaging can be useful. The study does not need selective, biologically recognisable borrowing to be scientifically coherent. Conversely, non-collapse and attractive arrows do not establish that the chosen router is clinically preferable.

A positive national result would support the evaluated transfer strategy. A null result would limit claims about the sufficiency of scale and increase the importance of domain fit. Neither result establishes that all specialties will benefit or that national pretraining supplies a biological explanation.

Keep three central limitations: transportability, outcome/monitoring measurement, and precision/selection uncertainty. Give the specific consequence of each. The pre-2022 records of later test patients in self-supervised pretraining are already disclosed appropriately. A temporal treatment-episode evaluation is not a completely person-disjoint test. Detailed code-level qualifications can stay in the supplement, including the frozen broad outcome definitions already audited.

### Supplement and back matter

The supplement should mirror the three questions, with a short contents page and a route from each main claim to its supporting tables. Separate core definitions and evaluation rules from exploratory diagnostics. Avoid treating every exploratory source-pair plot as a coequal result.

Complete the actual protocol/analysis-lock history, amendments, model receipts and executed-analysis inventory. Do not backdate prespecification. The remaining submission items include the documented evidence search, data/code access particulars, funder role, contributors, actual data-verification statement, patient/public involvement, interests and an accurate AI-assistance declaration. These require factual author input where it is not recorded; they should not be invented to fill a template.

## Scientific claims to keep separate

| Finding, if supported | What it permits | What it does not establish |
|---|---|---|
| Disease-pooled beats specialist | Extra disease supervision helps these fitted models on the stated population. | Rich histories beat five-variable indices because of sharing alone. |
| Multi-outcome model retains similar performance | Potential consolidation with bounded measured loss if sufficiently precise. | No negative transfer in every cell, or equivalence because a CI crosses zero. |
| Full OPERA beats joint | The corrected architecture improves the stated summary. | Patient-dependent source selection is necessary or optimal. |
| Source mean-ablation changes predictions | The fixed fit relies on that patient-specific correction conditional on retained paths. | Causal disease relationships, or necessity after retraining. |
| More model-low-risk patients | Broader favourable-prognosis identification under observed treatment, at measured risks. | Safe treatment reduction or automatically matched later event rates. |
| National pretraining helps | Benefit of the evaluated national transfer strategy. | Population size alone caused the gain, or external validation has been achieved. |

## Recommended final figure architecture

An editorial target of five main figures would work, with a sixth if keeping corrections separate is necessary for readability. This is **not a verified journal figure cap**.

1. **Clinical landmark and experimental design:** temporal split, pretraining sources, four supervision scopes, simple corrected model.
2. **Performance and calibration against practical alternatives:** selected available local/national strategies, per-task and pooled XGBoost, explicit coverage, paired effects and readable calibration. Full portfolio in the appendix.
3. **Clinical meaning:** NCCN-IPI in DLBCL, native categories, patients classified, observed mortality, achieved high-risk operating points. Other indices in a consistent supplementary series, with a concise cross-index summary where supported.
4. **What supervised sharing contributes:** disease/outcome factorial effects, training support, harms as well as gains; a compact correction-control panel or a separate sixth figure.
5. **Borrowing beyond haematology:** direct/adapted national transfer, joint versus corrected models and measured label-efficiency fractions with uncertainty.

The complete design grid, a clinical example and calibrated comparisons should be understandable without reading the router appendix. Keep the same model colours, names, difference direction, clinical-domain order and AUC units throughout. Use IQR for heterogeneity and confidence intervals for precision, with explicit labels. Check typography at final print size; a large PNG viewed on screen is not a readability test for a small journal panel.

## Length and journal requirements

The supplied August 2026 [Information for authors](https://www.thelancet.com/pb-assets/Lancet/authors/tldh-info-for-authors.pdf) was checked from the local PDF; the publisher endpoint rejected automated retrieval. The Article limits are 3,500 main-text words and 30 references, with a Summary of at most 300 words headed Background, Methods, Findings, Interpretation and Funding. Research in context has its own three sections and should not contain citations. The guide also specifies reporting and submission materials, editable figures and a separate organised appendix.

The current section-aware TeX count, treating result keys as non-prose, is:

| Section | Approximate prose words now | Suggested budget |
|---|---:|---:|
| Introduction | 516 | 450 |
| Methods | 1,449 | 950 |
| Results | 1,867 | 1,400 |
| Discussion | 1,094 | 700 |
| Total | **4,926** | **3,500** |

Headings add approximately 90 words. These counts include drafting instructions and placeholder text but exclude captions, abstract, Research in context and back matter. They are planning counts, not a final submission word count. The draft is about 1,400–1,500 words over the limit, requiring roughly a 29–30% reduction of current main text. Results will change length when instructions are replaced by findings.

There are 54 unique citation keys in the four main sections, not 54 necessarily needed main references. Move implementation-specific references with the methods they support into the separately referenced supplement, preserving the clinical-index and central sharing literature. Do not cut evidence merely to retain an architectural tutorial.

The current PDF is a reading format. Its typography is not itself a reason for rejection. Superscript reference formatting, final file organisation and submission declarations need conversion, but they are lower priority than the evidence and narrative. There is no basis here for inventing a five-figure limit or banning Results subheadings.

## Priority actions

### Before interpreting the final test outputs

1. Assemble one versioned tuning reading bundle with actual figures, numerical macros and a manifest. Keep synthetic examples separate. Reconcile plot labels with captions and link every result to its model, split, roster and threshold provenance.
2. Freeze the final estimand/display choices: paired rosters, main summaries, uncertainty, calibration-example rule, support curves, native-index eligibility and treatment of unsupported risk cutoffs. Finish rehearsing the already planned checks rather than choosing favourable-looking test displays later.
3. Keep the router claim within the evidence. The mechanism audit supports active, diffuse corrections and modest tuning gains, not selective source discovery. Review the rerun receipts before calling it stable or interpretable.

### Once the pending tuning and national analyses arrive

4. Create a claim ledger with one row per principal result: estimate, interval, population, coverage, practical size, qualification and main/supplement location. Populate the manuscript from this ledger rather than from whichever figure is easiest to read.
5. Produce the clinical counts and observed-risk comparisons without assuming low-risk expansion or matched evaluation specificity. Include failures and unsupported groups.
6. Replace the abstract and Discussion scaffolds with the actual answers. Cut approximately 1,400 words and simplify the main comparison table. Keep a clearly labelled tuning manuscript separate from the eventual test manuscript.

### What would most strengthen the eventual submission

The highest-value additions are credible independent validation and a precise account of clinical consequences, if feasible within the study. Neither means launching an indefinite set of new models. The existing planned clinical operating-point and temporal analyses can already improve the paper considerably. A clinically justified external application would address a more important limitation than another embedding gallery.

If the final results show meaningful cross-disease gains, little harmful multi-outcome transfer, useful probability accuracy and credible clinical stratification, the paper can make a substantial argument about the unit on which prognostic models should be trained. If the result is predominantly that weak specialists improve while richer tabular models remain clearly preferable, then the methodological lesson is narrower. The writing should acknowledge that outcome rather than inflate routing or breadth to compensate.

The next level is therefore a shorter, clinically anchored account of a well-controlled sharing experiment, with visible limits. The draft now has the conceptual structure to support it. The final held-out evidence must determine how ambitious the conclusion can be.
