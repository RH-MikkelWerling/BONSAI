# Author direction and proposed estimand — 28 October 2026

This updates the editorial recommendations in PAPER_CRITICAL_READ_20261028.md after the author's response. All scientific questions are retained. This document proposes an analysis-summary revision; it does not change the running jobs, primary analysis code or analysis lock.

## Agreed direction

- Organize the whole study as one programme of shared prediction, retaining supervised pooling, routing, population pretraining, clinical comparisons, coverage and interpretation. Detail can be distributed between main text and supplement without removing a question from the paper.
- Target The Lancet Digital Health. This is a plausible aspirational target, conditional on convincing completed results and a concise clinical narrative; it is not a prediction of acceptance. The current official author PDF could not be retrieved (403), so exact submission limits are not treated as verified here. Organize for clinical readability now; verify the live official submission specifications before final formatting.
- Make the introduction shorter through synthesis, not by dropping the population-pretraining or clinical questions. Organize Results as performance/coverage, sources of improvement, then interpretation/robustness. The methodological branches should support one clinical question rather than appear as separate new papers.
- Chosen model identities and historical selection descriptions are reconciled in the manuscript. The tuning/test prose is a preview issue for now; do not let another global tense rewrite delay result production.
- Keep clinical comparisons descriptive. Retain them because they link the risk estimates to familiar clinical stratification. Decision curves are optional and should not drive the paper unless an actionable decision and utility assumptions are well specified.
- Score missingness demonstrates computability limitations of the recorded score pipeline even with few required fields. It does not establish that clinicians lacked the underlying information or that missingness is model architecture failure. Quantify recorded-score availability and model-output coverage separately.
- Predictions in sparse or unestimable cells remain a useful demonstration of scope; do not describe them as validated performance. Test data should support more estimates than tuning, which the coverage table can establish.
- Treat late diagnosis as a bounded sensitivity, not a central suspected failure. The diagnosis-informed baseline addresses benefit from explicit diagnosis. It does not by itself test use of future-derived disease assignment. Existing exclusion analyses can settle whether overall and rare-subgroup results move materially; no large new experiment is proposed.

## Recommended primary performance question

For a typical disease in the specialty, how much does method A improve discrimination over method B across distinct clinical endpoints and the six specified prediction windows?

Use within-disease/outcome/horizon IPCW AUC differences, averaged with equal weight for diseases, endpoint families within diseases, grade variants within a family and horizons within an outcome. Families must be the frozen loss/endpoint families, not a grouping chosen after seeing effects. The 12 broad clinical domains serve a separate domain-balanced sensitivity; they are not interchangeable with the approximately 52 endpoint families that collect grade variants.

For disease d, endpoint family g, outcome variant o and supported horizon h:

Delta_macro = mean_d mean_g mean_o mean_h [AUC_A(d,o,h) - AUC_B(d,o,h)].

This is a finite-panel, disease-balanced discrimination estimand conditional on the supported target inventory. It is not expected patient benefit, average effect in the Danish treatment population, a causal transfer effect, or performance in the unobserved sparse targets. Equal horizon weighting describes six selected windows; it is not an integral over calendar time.

The current main summary instead gives every supported disease-outcome-horizon row equal weight. That can upweight diseases with more evaluable tasks, outcomes with more evaluated horizons, and findings represented by several grade variants. It is not mathematically incorrect; it targets a different average. Current router-selection summaries already balance diseases/outcomes/horizons, but do not automatically implement endpoint-family balancing. Adoption of the new primary summary needs an explicit dated analysis amendment before held-out inspection, justified by the question, not the model ranking.

## Why this aggregation rather than the alternatives?

- Do not pool all patient/outcome rows into one AUC. Scores for different outcomes are different questions. Even disease-pooled AUC for one outcome can improve through between-disease risk differences without improving within-disease ranking.
- Do not weight the primary result by disease size or event count: the common diseases and common outcomes would dominate the question of learning for scarce targets. A patient-frequency-weighted summary is useful as a distinct population-oriented sensitivity.
- Do not use inverse-variance weighting for the primary finite panel: precision weighting changes the target towards easier, larger cells. Hierarchical estimation can be supplementary without redefining the headline population of tasks.
- Use a mean paired difference, not a median, for the primary incremental effect. The mean captures magnitudes as well as direction and respects declared weights. Median/IQR and fraction positive remain descriptive distribution summaries.
- Keep the cell support threshold (>=5 cases and controls) separate from summary weights. Sparse but supported cells retain their scientific weight; their imprecision belongs in the uncertainty. Unsupported cells have coverage status, not zero effects.

## Roster and uncertainty contract

Freeze one supported cell-horizon inventory for each prespecified comparison family before bootstrapping. For the pooling-axis comparison, use S/D/O/J common supported cells. For OPERA/J/X comparisons, use their common support when comparing their incremental gains. Show pairwise maximal-coverage results separately if useful. Do not require the intersection of every optional external/control model; that can unnecessarily discard large parts of the core panel.

Hold the point-estimate weights and roster fixed in all paired patient multiplier replicates. Do not select new cells or renormalize to a different subset of tasks in each replicate. Report included/excluded diseases, endpoint families and horizons. The current disease-balanced router_decision.panel helper can renormalize over finite tasks per replicate; it cannot simply be reused without checking this fixed-roster requirement.

Report the same weighted absolute AUC for each model alongside the paired difference. All-six-horizon summaries are accompanied by horizon-specific effects, especially one year. Report flat-cell and patient-frequency-weighted sensitivities, endpoint/domain sensitivity where relevant, median/IQR, and supported coverage. If these disagree, explain what disease/task composition drives it instead of choosing the most favourable estimate.

## Probability accuracy and calibration

AUC is the primary discrimination metric, not a universal measure of prediction quality. Clinical risk claims require probability performance. Retain paired IPCW Brier differences, IPA and calibration as key secondary results with explicit weighting. IPA standardizes by a null-model error but can be unstable when that denominator is small in rare cells; a single relative improvement must not conceal that. Do not trade an AUC gain against a probability-accuracy harm silently.

Relevant methodological discussion: Van Calster et al., The Lancet Digital Health 2025, DOI 10.1016/j.landig.2025.100916 (https://pubmed.ncbi.nlm.nih.gov/41391983/), distinguishes discrimination, calibration, overall performance, classification and utility. No one metric answers all of these questions.

## Clinical endpoint/survival plan

Reuse exported native follow-up and predictions; model outputs define strata and observed event times define survival estimates. Use Kaplan-Meier for overall survival and Aalen-Johansen for non-fatal outcomes with competing death. This is principally analysis/rendering, not retraining.

One-year overall survival is a useful common comparison. Distinguish native endpoint from native development horizon: many indices were developed at longer horizons. A progression index compared with treatment failure is an explicitly approximate-endpoint comparison, not an exact reproduction. Keep the endpoint mapping fixed by score identity and clinical rationale. Risk-group cutoffs selected and evaluated on tuning patients are descriptive and optimistic; final evaluation uses frozen cutoffs on held-out patients.

## Next steps without interrupting current runs

1. Review the current flat-cell results and verify completeness/model identity.
2. Decide and version the primary summary contract before held-out inspection.
3. Compute balanced and sensitivity summaries from existing per-cell estimates and paired draws where available; this ordinarily needs CPU aggregation rather than inference/retraining.
4. Complete descriptive score comparisons and survival curves from available follow-up.
5. Organize the manuscript around the three nested questions while preserving every branch.

## Follow-up: metrics versus aggregation

The author correctly notes that Brier and IPA are already used throughout the manuscript. The remaining decision is not to add probability accuracy to an AUC-only study; it is to define how each metric is summarized across heterogeneous tasks and which estimand leads which claim. The introduction now ends with the agreed three questions and a map to the corresponding Results analyses.

For cell c and shared null Brier error B0_c, IPA_m,c = 1 - Brier_m,c / B0_c. The paired IPA gain is (Brier_B,c - Brier_A,c) / B0_c. Brier and IPA therefore share the same underlying error comparison within a cell; their agreement is not independent validation. Averaging their differences across cells produces different weighting because IPA normalizes each cell by its baseline error.

Mean cell-level IPA gain = sum_c w_c * DeltaBrier_c / B0_c. Dividing the average Brier gain by the average null Brier error instead implicitly weights cell-level IPA gains by w_c * B0_c. Those are different estimands. The former gives declared task weights after normalization; the latter gives more weight to tasks with larger baseline uncertainty. Neither is automatically the correct summary of all clinical utility.

AUC quantifies ranking within a risk set; Brier quantifies probability error on an absolute scale; IPA quantifies probability-error reduction relative to a task-specific null prediction. For a rare outcome, a near-zero predictor can have a small Brier error without strong discrimination. IPA addresses the baseline-error scale but ratios can be unstable in small, very rare cells. AUC is independent of prevalence under pure prevalence changes with fixed case/control score distributions, but it remains dependent on case mix and spectrum. No metric makes disease/task weighting irrelevant.

The proposed next step is to inspect matched-roster AUC, Brier and IPA under identical candidate weight schemes, without selecting the scheme according to OPERA's rank. Decide which question each answers: discrimination across the specialty; normalized risk prediction across tasks; or prediction error for the actual patient/task mix. Keep absolute calibration visible. Do not introduce a numerical composite that trades AUC against IPA with arbitrary conversion factors.

A possible reporting contract is a declared primary discrimination estimand, a key probability-accuracy estimand under the same roster and hierarchy, and descriptive distribution/coverage summaries. Whether IPA is key secondary or co-primary is still a substantive design decision, not settled by this note. The current running analysis remains unchanged.
