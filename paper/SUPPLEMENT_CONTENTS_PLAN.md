# Supplementary evidence beyond the main figures

Figures explain the pattern; exact tables make it auditable. Provide tidy CSV
exports alongside readable summary tables, rather than hundreds of printed pages.
Synthetic figures are layout proofs and cannot fill these empirical tables.

## Required tables and documentation

1. Cohort/data sources and definitions: first-line reconstruction, disease
   assignment, second-line rules, all outcomes, baseline eligibility,
   ascertainment, censoring, competing death and caveats. The styled definitions
   supplement is included in the assembled PDF; its traceability/open questions
   remain review companions.
2. Outcome risk sets and coverage: disease, outcome, horizon, split, eligible N,
   primary/competing events, unknown horizon status, exclusions and why a metric
   or specialist cannot be estimated.
3. Absolute model performance: model, seed and seed summary, disease, outcome,
   horizon, estimand, N/events, AUC, Brier/log loss and calibration where estimable,
   with uncertainty and unavailable-value reasons. Give readable domain summaries
   plus complete machine-readable cell tables. Do not average seed confidence
   intervals or mix known-status metrics with full-risk-set IPCW metrics.
4. Paired contrasts: S/D/O/J factorial, R−J and source banks, tabular comparisons,
   external/DAPT increments and tuning-selected strategy. Include the common
   evaluation roster, effects/intervals, weighting and seed variation; identical
   cells/patients must underpin each paired comparison.
5. Models and selection: resolved configurations, architectures, parameter
   counts, initialization/vocabulary hashes, optimizer/budgets, seeds and tuning
   selection rules/screens. Clearly separate tuning results from held-out evidence.
6. Robustness: horizons, censoring analyses, calendar strata, late-coded disease
   sensitivity, patient history/subgroups, seed variation and any Bayesian
   assumptions, diagnostics and justified sensitivities actually used.
7. Clinical-score comparisons: score availability by disease/time, missingness,
   intended populations, cutoffs and tuning rationale, risk-group N/events and
   survival/incidence with intervals, reclassification and numbers at risk.
8. Compute and provenance: training/inference time, memory/storage, hardware,
   fit counts and external/domain-adaptation provenance. Report actual measures.

## Confusion matrices: selected operating points, not every task

A confusion matrix is meaningful only after fixing an outcome, horizon and
threshold. Our primary survival/competing-risk predictions are continuous and
have endpoint-specific censoring; a matrix alone discards that information.

Provide TP/FP/TN/FN, sensitivity, specificity, PPV/NPV and uncertainty for the
prespecified clinical-score comparisons or other explicitly proposed decisions.
Freeze thresholds on tuning data and report actual held-out specificity. State
whether counts use known-status patients; report excluded early-censored patients
and never classify them as negatives. Do not present weighted IPCW quantities
as literal patient counts. There is no need for an exhaustive matrix wall for
all 69 outcomes × diseases × horizons unless a claim depends on those thresholds.

## Optional material

Exploratory readout catalogue (formerly main Figure 11), gate-use summaries,
embedding maps and probes remain supplementary, with their descriptive status
explicit. Include them only when they explain a reported comparison; do not
imply every catalogue architecture is a completed primary arm.

## Detailed neural reproducibility tables

Use actual resolved configurations and saved checkpoint metadata, not defaults
or planned settings. Distinguish confirmed architecture, planned budgets and
completed training. The local candidate is 256 wide, four layers/four heads,
scaled Time2Vec and context 8,192; verify these against the delivered checkpoint.

### Transformer and representation specification

- Encoder architecture/class and causal attention mask; tokenization modalities,
  vocabulary size/hash, numeric binning and missing/unknown-token treatment.
- Hidden width, depth, attention heads and head width, feed-forward width,
  activation, normalization placement/type, attention/residual dropout and biases.
- Temporal encoding formulation and dimensions; positional encoding, age/time
  units, embedding combination and scaling; maximum context/window policy.
- Sequence ordering, ties, padding/masks, truncation and patient pooling/readout.
  State exactly which records are permitted before the prediction origin.
- Trainable versus total parameters for encoder, heads and each adapter bank.
- Hazard intervals/cause parameterization, CIF conversion and horizon objective;
  grade/endpoint-group weights, censoring masks and structural exclusions.
- Router rank, query width/mode, keys/embeddings, residual scale, home floors,
  support shrinkage, shared path and initialization. Describe detached borrowed
  parameters versus live encoder/query gradients, and anchored/unanchored policy.

### Training specification for each phase

Separate local SSL, direct external export, DAPT and supervised fitting.
For each, report optimizer, learning rate, schedule, warmup, weight decay,
clipping, microbatch size, gradient accumulation, effective batch size, precision,
seed, sampling, loss weights and frozen parameters. Report planned maximum and
actual completed epochs, optimizer steps, processed windows/tokens or labelled
exposure, checkpoint-selection metric/direction, validation coverage and selected
checkpoint epoch. Explain stopping/resume behaviour and completion criteria.
Equal epochs across S/D/O/J do not mean equal compute or exposure.

For external transfer, give the source model's actual width/depth/heads/objective,
pretraining population/time coverage and available training budget; mark
unavailable provenance explicitly. Record remapped token overlap, local-only
initialization, vocabulary and checkpoint hashes, and that DAPT starts from the
same direct export. Document size/budget mismatches rather than calling them a
controlled scale comparison.

For XGBoost/logistic, report feature recipes/windows, missingness/encoding,
scaling, hyperparameter search spaces/budgets, tuning objective, final settings,
seeds, class weighting and horizon-label construction. Clinical-score comparisons
need original score definitions, applicability, availability, model group cutoffs
and their tuning provenance.

## Additional figures, ranked by scientific need

**Required evidence; generate from existing logs/predictions where possible:**

- Training and tuning trajectories for the primary fits, marking selected
  checkpoints and resumed segments. Compare 64/256 only with declared matching
  conditions and both optimizer-step and wall-clock axes where available.
- Seed stability of headline contrasts, showing fitted-seed variation separately
  from patient-bootstrap intervals; do not treat seeds as independent patients.
- Follow-up/censoring support by split and horizon, especially two-year held-out
  risk groups. Include numbers at risk and observed/unknown horizon-status counts.
- Complete calibration/reliability and discrimination/probability-error summaries
  across horizons, with uncertainty and support. Keep full-risk-set and
  known-status estimands distinct.
- Temporal performance/calibration drift and pre-index history strata already
  specified in the analysis plan; do not select favourable strata on test.
- External transfer coverage and DAPT trajectories, alongside the existing
  vocabulary/support/calendar effect plate.
- Same-fit late-coded disease sensitivity and an availability/count audit;
  compare the same prediction artifacts after the prespecified exclusion.
- Tuning-only source/rank/query selection evidence, clearly separated from
  replicated held-out contrasts and old 64-wide experiments.

**Conditional on a claim or model actually used:**

- Bayesian prior/posterior predictive checks, raw-versus-pooled effects and the
  small set of justified sensitivity analyses. No figure needs to exist for an
  unused model extension; diagnostics do not substitute for raw paired evidence.
- Decision curves/net benefit only for explicit actions, thresholds and an
  appropriate survival/competing-risk estimand. This requires an agreed clinical
  decision setting, not merely another attractive plot.
- Controlled label-scarcity learning curves, route-removal interventions or
  equal-compute controls only if those causal/mechanistic claims are made. These
  require additional training or intervention analysis and are not prerequisites
  for the current observational support-stratified conclusions.

## Release package and remaining locks

Provide numbered supplementary Methods, a table/figure contents page, a frozen
run inventory and a file manifest tying figures/tables to tidy sources, code
versions, analysis configurations and reviewed releases. Include package versions,
hardware and reproduction commands. Protect patient-level data; release only
approved aggregate artifacts/code. Keep clinical/data-definition open questions
visible until resolved; supervisor Table 1 sign-off is still pending.

Before filling empirical tables, lock the primary metric/censoring estimand,
main horizon, panel weighting, support/estimability criteria, contrast roster,
practical margins and any multiplicity policy. Synthetic prototypes do not
resolve these scientific decisions. The supplement should make every headline
claim auditable, not maximise the number of plots.

## Material added in the current writing pass

`supplement_evaluation_details.tex` is integrated into the full manuscript as
readable prose covering matched risk sets, intended-endpoint score benchmarks,
probability mappings, age/operating-point comparisons, censoring/calibration,
robustness and representation provenance. Eight additional CSV templates cover
the missing numerical/provenance exports. `render_supplement_tables` renders
reviewed aggregate summaries with hashes and rejects empty templates; the main
manuscript automatically includes their combined LaTeX file when supplied.
Scientific estimates and all underlying export adapters still require completion.
