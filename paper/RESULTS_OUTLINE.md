# Results outline to fill after the locked analysis

The order is population → pooling → routing and operational alternative →
tabular comparison → nation-scale pretraining/DAPT → clinical interpretation →
secondary heterogeneity. Section titles remain questions until real findings
justify declarative titles. Synthetic values must never enter the prose.

For each comparison specify the metric/estimand, horizon, common cell roster,
number of patients, uncertainty method, seeds and weighting. Lead with a paired
absolute difference and its interval, then describe heterogeneity. Report
unsupported cells separately, not as zero effects. Keep known-status comparisons
and full-risk-set IPCW summaries explicitly distinct. No numerical tolerance,
minimum support or clinical threshold is invented by this outline.

## 1. Study population and the landscape of predictable outcomes

Establish who was studied and what could be evaluated before comparing methods.

- Confirm first-line and supervised totals, exclusions and temporal split counts
  against approved Table 1 and cohort flow. Distinguish SSL patients from
  supervised patients and from endpoint risk sets.
- Report age, sex, disease composition, follow-up and pre-index history from the
  fresh export. Give the 24-disease/69-outcome scope, structural exclusions and
  endpoint eligibility counts.
- Report evaluable cell counts at 365 days and all horizons, train event support,
  cells with an unfittable specialist, and score availability by disease/year.
- Describe the J model's absolute predictability atlas with AUC and probability
  accuracy/calibration; show weak as well as strong clinical domains. AUC alone
  does not establish utility or benefit from sharing.

Fill-in: “Of [source N], [first-line N] entered the first-line cohort and
[supervised N] the temporal analysis. At 365 days, [N/common denominator] cells
were evaluable. Discrimination ranged from […] and probability accuracy […].”

Figures: cohort flow, Table 1, atlas. Supplement: risk sets, support and coverage.

## 2. Separate and combined contributions of disease and outcome pooling

Answer which supervised axis helps, holding encoder and architecture fixed.

- Disease pooling D−S; outcome pooling O−S; combined J−S.
- Conditional effects J−O and J−D; interaction J−D−O+S.
- Paired primary-horizon effects and uncertainty, complementary Brier/log-loss
  effects, seed variation and estimable support for each contrast.
- Describe disease/endpoint heterogeneity and observed support dependence,
  including harmful cells. Raw estimates accompany any Bayesian summaries.
- Report predictions/accuracy in cells without an estimable specialist as a
  coverage result, without claiming superiority to an absent comparator.

Fill-in: “Disease pooling changed [metric] by [effect, interval]; outcome
pooling by […]; joint learning by […]. The conditional effects […] and
interaction […] indicated [complementarity/redundancy/uncertainty].”

Main: pooling plate. Supplement: cell maps, six horizons, statistical checks.

## 3. Incremental value of conditional routing

Answer R−J, not whether multi-outcome learning itself works.

- Report the tuning-locked policy and R−J effect with interval, practical scale,
  seed variability and lower-tail degradation.
- Report the matched neither/disease/outcome/both bank contrasts and whether
  either bank provides an incremental benefit. Keep tuning screens distinct
  from replicated held-out estimates.
- Describe test performance in tuning-defined negative-transfer strata; these
  are not strata selected retrospectively from test harm.
- State null findings plainly. A wide interval is uncertainty, not equivalence.
  Gate usage is descriptive and cannot identify a transfer mechanism.

Fill-in: “Relative to J, the locked router changed […] by […]. Disease and
outcome bank increments were […]. In cells identified as harmed on tuning, […].”

Main: promoted routing effect plate. Supplement: gate usage, rescue and screens.

## 4. One joint model versus separate per-cell choices

Test the practical alternative of maintaining a portfolio of narrower models.

- Compare J with the S/D/O procedure selected per cell on tuning, using the same
  prespecified selection rule and common evaluable cells on test.
- Report paired effect and uncertainty, frequencies of S/D/O selection and seed
  stability. Do not select the best method retrospectively on test predictions.
- Report fits, parameter/storage burden, total training and inference costs,
  plus coverage. “One model” does not imply lower compute without these data.
- Claim equivalence only with a prespecified margin and appropriate uncertainty.

Fill-in: “Against tuning-selected cell-specific models, J changed […] by […],
while requiring [fits/storage/compute]. The selection procedure chose S/D/O in
[…]. Unsupported targets remained […].”

Main: promoted strategy plate, potentially combined with pooling in final layout.

## 5. Comparison with strong tabular prediction

Answer whether the complete neural pipeline improves over usual alternatives.

- S−X, J−X and R−X, plus X−logistic as baseline-strength evidence.
- Report matched patients, eligibility, labels/horizons, feature access and
  tuning budgets; identify any enriched-input comparison explicitly.
- Give paired discrimination and probability-error differences, harmful cells,
  calibration slope/intercept or calibration-in-the-large as actually defined,
  and reliability examples chosen by a declared rule rather than favourable plots.
- Explain variation across diseases/outcomes/horizons without attributing the
  whole difference to event order or pretraining alone.

Main: tabular effects with compact calibration summary. Supplement: full
calibration plate and six-horizon tables. The compact composite remains to be made.

## 6. Nation-scale pretraining and domain adaptation

Separate origin from adaptation under fixed shared/routed readouts.

- E−L (external direct); EA−E (local DAPT); EA−L (total package), each for J/R.
- Report routing-by-origin and adaptation-by-readout interactions with their
  uncertainty; do not reselect the router using nation-scale-model test results.
- Describe vocabulary coverage, target support and calendar strata. Report
  actual architecture, source checkpoint, token/budget and compute differences.
- A gain from EA−L alone cannot establish a DAPT benefit; a package comparison
  with different architectures cannot establish a pure model-scale effect.

Fill-in: “Nation-scale initialization changed […] by […]; continuation on local
records added […]. The total effect […] differed between readouts by […].”

Main: nation-scale pretraining plate. Supplement: coverage/support/calendar strata and provenance.

## 7. Clinical interpretation through established risk groups

Use the prespecified NCCN-IPI/DLBCL example rather than merging clinical scores.

- Report applicable score-available patients, missingness and follow-up support.
- State model cutoffs locked on tuning and their rationale: comparable low-risk
  event rates with clinically meaningful group sizes. Check the historical
  0.10/0.30/0.65 candidates; do not assume they satisfy this in the new cohort.
- Report held-out two-year event-free survival and uncertainty, group sizes,
  low-risk yield and cross-classification on identical patients.
- Include death in the treatment-failure composite; use KM for that example.
  For nonfatal endpoints with competing death use AJ incidence instead.
- Report precision/recall at a tuning-locked operating threshold, achieved test
  specificity and age-stratified patterns. Do not call a descriptive age pattern
  proof that bias was removed or that treatment decisions would improve.

Main: risk-group plate. Supplement: detailed clinical score/age panels and other
scores separately in their intended disease populations.

## 8. Secondary heterogeneity, robustness and resource use

Summarise prespecified checks, with full reporting in the supplement.

Patient history, age/sex and score availability; horizons and calendar time;
late-coded identity exclusion using the same fits; censoring estimands; seed
variability; hierarchy/prior checks; actual training/inference costs. Report
which findings survive these checks and which remain uncertain. Descriptive
embedding maps are supplementary, not evidence of predictive superiority.

End the Results with what the comparisons establish and the main residual
uncertainty. Reserve prospective clinical consequences for Discussion.

## Prose implementation

The manuscript now follows this outline as connected Results prose. Each
section progresses from its comparison to the paired effect, heterogeneity and
clinical/operational interpretation. The atlas establishes absolute performance;
pooling, routing and the strategy portfolio establish what shared learning adds;
tabular and nation-scale comparisons test the alternatives; risk groups make the
probabilities interpretable. Figure captions describe displays, while prose
will state the findings. Replace bracketed fields only with locked analysis
outputs; do not repeat every panel or turn null estimates into equivalence.

The 14 October full-pass layout keeps this narrative order but places detailed
routing and portfolio plots in the supplement. Calibration is now a main figure
within the tabular-benchmark sequence. The primary three questions govern main
space. All comparisons, including null routing effects, remain in Results.
