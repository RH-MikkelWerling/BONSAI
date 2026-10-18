# Manuscript review after integrated-gradients integration — 18 October 2026

## Editorial judgement

The study has a strong, coherent question: can clinically useful predictors be
built by sharing labelled experience across diseases and outcomes, and does a
nation-scale starting representation improve that strategy? The novelty is the
controlled evaluation of these sources of experience in an unusually broad,
clinically defined panel. A router is an optional way of making sharing more
selective. It is not the reason the study matters, and a null router result
would not invalidate the central study.

The draft is substantially developed but is **not submission-ready with only
numbers missing**. Several claims still need provenance or design decisions,
not prose. The result placeholders are a reporting template, not evidence of
benefit. Preserve the ambition, but let both positive and negative results set
the final abstract, title and conclusion.

## The argument readers should retain

1. Patients face several risks at treatment start; conventional disease-course
   indices cover only part of this need, and their availability is uneven.
2. Related diseases and outcomes offer potentially reusable experience, but
   neither transfer benefit nor protection from negative transfer is assured.
3. The disease–outcome square defines four training scopes. Their paired
   contrasts estimate the incremental predictive value of each supervision axis
   under a common development protocol.
4. Tabular pipelines and applicable clinical indices test whether that sharing
   produces useful improvements over realistic references.
5. Nation-scale pretraining and local adaptation test whether an existing broad
   encoder adds value beyond local learning, including when labels are scarce.
6. Calibration, harm, coverage and cost determine where the approach is useful.
   Explanation and representation figures help audit the model, but do not
   establish these performance claims.

## Changes made in this pass

- Abstract distinguishes missing routine predictors from the claim that no
  predictor could ever be developed. Removed “complete” history and a causal
  reading of “without raising their risk.” Result direction remains pending.
- Introduction distinguishes published scores, clinical adoption and recording
  in these registries. MZL-IPI is acknowledged as an externally validated score,
  rather than used to support a blanket lack-of-validation claim.
- Retained the clinical importance of prognostic scores but separated prediction
  improvement from evidence for changing IPI-based treatment decisions.
- Replaced a broad priority claim with the precise unresolved empirical question.
- Clarified that matched register sources do not mean identical information:
  tabular three-year aggregates and truncated event sequences differ in history,
  resolution and representation. The contrast compares full pipelines.
- Clarified that excluding curated diagnostic LDH from the quality registry
  does not exclude pre-index LDH results from the laboratory history.
- Distinguished the delivered nation-scale encoder comparison from a controlled
  experiment on pretraining population size. Equal epochs are not equal exposure.
- Made post-index laboratory observation requirements visible in main Methods
  and limitations, and added dated-pre-index disease-identity sensitivity wording.
- Disclosed full-period rare-code merging as transductive preprocessing. It is
  not made strictly prospective by restricting later vocabulary/gradient fitting.
- Separated cell-weighted table summaries from disease/endpoint-weighted
  hierarchical estimands, and point-estimate wins from uncertainty-supported wins.
- Clarified metric-scale pooling interaction and descriptive cell-map intervals.
- Added an explanation Methods paragraph and Results home, replaced the stale
  “no patient-level explanations” limitation, and expanded IG methods formally.
- Labelled IPS versus treatment failure as endpoint-approximate, separate from
  endpoint-matched overall-survival comparisons; explicitly requested tabular
  results on the same score-available patients.
- Corrected the appended model-reporting draft's choice of IPI versus NCCN-IPI
  as the principal example and removed an unverified assumption that the national
  arm necessarily uses vocabulary remapping.
- Added original IG, sanity-check, CTCAE v5 and BONSAI references, and updated
  the numeric-encoding reference to its published ICML proceedings record.

## Highest-priority clinical review issues

| Issue a reviewer will raise | Evidence/action needed | Additional neural training? |
|---|---|---|
| Is the predicted population identifiable when treatment starts? | Laboratory eligibility uses an index-to-day-30 measurement or early death, so report eligibility fractions and explicitly conditional risk sets. Consider baseline-only identifiable eligibility as a sensitivity; assess whether its labels can be meaningfully ascertained before changing the training cohort. | Usually re-evaluation first; a changed training population is a separate decision. |
| Was disease identity available, especially for routing and specialist selection? | Export diagnosis/evidence date, treatment index, source, rescue flag and an independently documented baseline diagnosis where available. Re-evaluate all relevant models in the dated-pre-index subset; excluding only diagnoses after day 14 does not answer the question. Later coding is not proof of clinical ignorance, but treatment specificity alone does not verify the exact fine subtype. | No for subset evaluation. |
| Are treatment lines and complications credible? | Supervisor sign-off on the cohort and Table 1; archive source grading functions and line rules; report agreement with independent registry information where available, with discordance and subgroup counts. Seeded CLL/MM dates are not independent validation of their own reconstruction. Some therapy-like episode rules require repeated later records; quantify whether retrospective confirmation excludes short episodes or early deaths. Index-day events/deaths are excluded under the date-resolution convention. | No. |
| Are the outcomes really CTCAE adverse events? | Keep version-specific/local definitions and caveats. Separate persistent abnormality, incident event, worsening and antimicrobial-use proxy. Do not call the full panel adjudicated toxicity caused by treatment. | No. |
| Is the index comparison fair? | Inventory exact index/version, intended population, endpoint and horizon; restrict score timing; compare sequence and tabular models in identical score-available patients. Distinguish IPS's progression endpoint from second-line/death. Distinguish ISS from R-ISS. Report score missingness. | No if the corresponding fitted predictions exist. |
| Does a larger low-risk group establish safety? | Report achieved test risk and its interval, attained specificity, group size and events; tuning matching does not establish non-inferiority on test. Do not use “no higher risk” as a guaranteed conclusion without a justified margin and adequate precision. | No. |
| Are benefits actionable? | Give one concrete supportive-care use case and absolute numbers at a tuning-fixed operating point. Decision curves require an explicit action, plausible thresholds and appropriate competing-risk treatment; do not add arbitrary net-benefit plots just to complete a checklist. | No. |

The planned “eight subtypes without a routinely used index” stratum needs clinical
review against contemporary guidance, particularly given MZL-IPI and published
Richter/PCL indices. Absence of a registry score is measurable; absence of routine
clinical use requires separate evidence. Keep its definition and chronology in the
analysis lock, and call any change after test inspection exploratory.

## Highest-priority ML/statistical review issues

| Issue | Required resolution | Cost |
|---|---|---|
| Two-axis claims | Use the same complete four-model cell roster for disease pooling, outcome pooling and interaction. Report conditional and combined contrasts, not just joint versus specialist. Interaction depends on metric scale; it is not a biological mechanism. | Existing predictions. |
| Unequal optimisation | Report effective batch, patients/tokens, updates, selected epoch, early stopping and parameter counts. The effect includes the specified optimisation/selection regime. Equal epoch caps do not isolate added labels at fixed computation. | Run logs. |
| Fragile specialist comparator | A frozen-encoder per-cell regularised probe is the most useful inexpensive additional baseline if not already available. A frozen **joint** readout does not by itself answer whether tiny specialists should be frozen. Export representations once, then fit CPU probes. | One embedding export + small fits. |
| Training-run uncertainty | Patient bootstraps condition on a fitted model. If gains are marginal, repeat a small prespecified set of decisive contrasts with additional seeds before strong superiority/equivalence claims. Do not imply bootstrap precision includes training variability. | Targeted fits, not a repeat of every arm. |
| Censoring estimand | The paper's primary estimand is CIF/IPCW. `.env.paper.example` explicitly selects `cif_ipcw`; raw analysis templates/CLI fallback still allow known-status analysis. Check the **generated** analysis manifest, not the template alone. Keep sensitivity outputs separate. | Configuration/provenance check. |
| IPCW assumptions and intervals | Current locked analysis estimates marginal censoring survival within disease/outcome/split and reuses those weights in patient bootstrap draws. Report this conditional approximation and weight tails; consider re-estimating censoring weights within draws for decisive sparse/long-horizon results. Monitoring-related selection is not fixed by IPCW. | CPU only. |
| Panel summaries | Label arithmetic cell means, medians and hierarchical disease/endpoint averages separately. Use a shared roster for comparisons across arms. A panel-average benefit does not validate every sparse cell. | Existing outputs. |
| Bayesian summary | Lock metric-scale priors, meaningful-effect margins, support bins, spline complexity and covariance regularisation. Keep raw estimates visible. Prior predictive, posterior predictive and convergence checks must be actual outputs, not assurances in prose. Avoid extra nesting/interactions unless a question needs them. | Existing analysis. |
| Multiple correlated tasks | Distinguish primary families/horizon from exploratory cells. Hierarchical shrinkage and correlated bootstrap do not automatically provide family-wise error control. Raw cell dots are pointwise intervals, not discovery guarantees. | Reporting. |
| National provenance | Reconcile native national-data tokenisation, any vocabulary alignment, and the statement that identifiers cannot link across extracts. Document how downstream target patients are associated with national-format records, without inferring that pretraining identity exclusion is possible. Report architecture, deliveries, vocabulary, overlap, missingness and compute for each encoder. | Artifact audit; no automatic retraining. |
| Prospective versus retrospective validity | Full-period rare-code coarsening uses later aggregate information. Either keep and disclose this transductive design or create a strict pre-cutoff preprocessing sensitivity; a strict replacement may require rebuilding/pretraining and must be justified by likely impact. | Disclosure now; retraining only if needed. |
| Test lock | Archive what was fixed, when, and what test results had been seen. Development chronology must support “evaluated once”; do not retroactively describe iterative test-informed decisions as prespecified. | Provenance. |

## Results and figure hierarchy

Keep the existing information-axes/model-paradigms figure. The compact scientific
arc should read: cohort/coverage → overall predictive value → two-axis
contribution and negative transfer → matched clinical consequences → incremental
nation-scale/adaptation value. Absolute predictability belongs beside the main
performance findings. Routing, strategy selection, regional/calendar analyses and
explanations support that arc rather than becoming independent headline stories.

One small multi-outcome risk-profile illustration could help a clinician see
what the output actually is: several absolute risks at declared horizons, with
a reference population. Use a clearly synthetic illustration or disclosure-safe
aggregates, not an attractive real-patient example selected after inspecting its
outcome. This needs existing predictions or a schematic, not another model fit.

The current main text contains many short question headings and repeats XGBoost
results in the overview and benchmark section. Once values are available, give
the overview one principal number per question; reserve detailed comparisons for
the corresponding section. Combine short robustness/subgroup sections when they
share a figure. Each result paragraph should state the population/target,
comparison, effect with interval, and interpretation/exception. Avoid claims of
clinical safety, mechanistic transfer or deployment readiness derived from AUC.

The main-text figure sequence currently gives the full clinical story to IPI.
Add the same-population tabular mortality result to that narrative/table; beating
a coarse index is weaker evidence than improving on a tuned tabular predictor.
Keep the complete score inventory and per-score risk panels in the supplement.
No fixed rule requires a cohort-flow diagram to be Figure 1 for every venue.

Keep representation galleries supplementary and descriptive. Projected proximity
is neither validated clinical similarity nor an uncertainty estimator. Explanation
plates should follow the core analyses, with quality checks alongside attribution
patterns. Do not introduce three explanation plates at the beginning of the
supplement and claim numbering follows main-text first mention.

## Explainability: rigorous, computationally reusable

The user clarification supersedes the earlier suggestion to defer IG in
`REVIEW_ROUND2_20261018.md`: do the chosen analysis properly. Limit the scientific
questions and cache the expensive calculation, rather than dropping checks.

Use the tuning-selected primary model, a declared outcome/horizon set, fixed
sampling/reference choices, numerical completeness, alternative-reference and
partial-randomisation checks, and event replacement/deletion versus random
controls. Export signed token credits, full/reference probabilities, time/event
identity, sampling weights and quality metadata once. Existing exports support
source, event, concept, disease and arbitrary lookback summaries without new GPU
attribution. They also support internal patient-level timelines or additive
risk decompositions: include reference risk, selected signed event credits,
all remaining credit and the numerical residual so the display reconciles to
the predicted risk. Do not silently retain only favourable examples.

Three outcomes and 300 patients are an initial descriptive scope, not a guarantee
of adequate precision. Inspect pilot numerical reliability and runtime before
locking the completed scope. Rare subgroup/concept summaries require enough
carriers; ten sampled patients per subtype does not justify a full fine-subtype
atlas. Show failure rates and uncertainty, and withhold unsupported panels.

Keep these distinct:
- signed sums explain risk relative to the reference;
- absolute shares describe allocation of attribution magnitude;
- withholding recent history tests the prediction with altered input;
- a true earlier-origin prognosis needs a separate longitudinal design;
- plausible directions do not validate biological mechanisms;
- explaining a single joint model does not explain its gain over specialists.

Fixing clinical code names, changing aggregation, choosing an additional plot
from the same target/patient exports and updating captions are CPU-only. A new
checkpoint, patient sample, target or reference requires new extraction for that
scientific change. Do not promise that cached IG can answer arbitrary future
counterfactuals without more inference.

## What is still required before submission

1. Verified cohort/Table 1, treatment-line and outcome provenance, code lists,
   source coverage and timing; final approvals and score inventory.
2. Completed paired predictions and numerical tables, with explicit estimands,
   common cell rosters, missing/failed fits and negative-transfer findings.
3. Locked Bayesian settings and actual diagnostics; score cutoffs and adequate
   follow-up; explanation mapping review and server pilot quality outputs.
4. Exact national/local architecture/data/compute provenance; report cost and
   missingness alongside data-efficiency gains.
5. Actual test-lock chronology, governance, data/code-access, licensing,
   authorship, funding and patient/public-involvement statements.
6. A final result-driven abstract and Discussion. Do not fill in a positive
   conclusion merely because the synthetic figures are compelling.

This review did not launch server training or silently change scientific locks.
Remaining issues are evidence needs, not claims that the corresponding analyses
have already been run. Citation verification and build evidence are recorded in
`REFERENCE_AUDIT_20261018.md` and the closing verification entry below.

## Verification of this editing pass

Both the standard manuscript (72 pages with empirical figure placeholders) and
synthetic-figure preview (90 pages) build successfully in
`paper/render/paper_review_20261018/`. All 80 cited keys resolve, no duplicate
bibliography keys were found, and neither final main build has undefined
citations/references or overfull boxes. The source-table column padding was
reduced to remove its overflow. Two existing PGF node-border warnings remain.
The embedded definition and model-reporting supplements were rebuilt; the
standalone definition document retains a minor 1.1 pt text overflow warning.
The IG methods page was visually inspected. Displayed cohort-flow/split totals
reconcile arithmetically, which does not replace clinical sign-off.

A stale root `.bbl` was initially picked up by the standard manuscript build in
a fresh output directory. Explicitly regenerating that output's bibliography
resolved its missing citations; the final PDF and log are clean. If repeating
this isolated-directory build, use BibTeX on the output's `.aux` if latexmk
picks up an older bibliography from the source directory.

`git diff --check` passed and the attribution YAML parses with unchanged compute
settings. This pass changed manuscript/documentation and configuration comments,
not attribution algorithms; the full test suite was not rerun. The previous
focused IG release suite remains recorded separately (39 tests).
