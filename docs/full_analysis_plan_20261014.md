# OPERA full analysis plan and question hierarchy

This is the prospective plan for the approved 69-outcome / 24-fine-disease
release and the new 256-dimensional encoder. It supersedes the older
router-first paper narrative. It records proposed analyses, not completed
experiments or empirical findings. Cohort/Table 1 still require clinical signoff.

## 1. Headline questions: the paper must answer these

| Priority | Question | Required fits | Primary contrast family |
| --- | --- | --- | --- |
| 1 | Does supervised learning across diseases and outcomes help, separately and together? | Local S, D, O, J | P1–P6 |
| 2 | Does the resulting prediction system outperform conventional tabular modelling? | Cell-specific XGBoost and logistic regression; matched neural predictions | T1–T4 |
| 3 | Does external pretraining improve on local pretraining, and does local domain adaptation contribute? | L, E, EA crossed with shared J and locked readout R | E1–E4, A1–A2 |

S is one fine disease × one outcome; D is all applicable diseases × one
outcome; O is one fine disease × all applicable outcomes; J is the full panel
with shared multi-horizon BCE. Each fit predicts all six horizons. Rows are
diseases and columns are outcomes. R uses the same full-panel supervision as J
and changes the readout. Source eligibility never defines a supervision scope.

For positive-oriented gain g(A,B), the pooling effects are D−S, O−S, J−O,
J−D and J−S; the interaction is (J−D)−(O−S), equivalently (J−O)−(D−S).
These are related factorial estimands. Their sum identities hold per cell;
they are not six independent discoveries. The comparison measures the training
procedures, including their available labels, patients and optimization exposure.
It does not identify a biological causal interaction or equal-compute advantage.

The tabular question needs both J versus X and R versus X, plus S versus X to
separate sequence-model capacity from pooling. Compare X versus logistic as a
reference for the strength of the tabular baseline. Registry-enriched predictors
are a separate comparator unless the same inputs are available to every model.
Use the same index, eligible patients, label definitions and prospective split.
Include clinically available disease identity in tabular predictors where
applicable; a cell-specific baseline already has fixed disease identity.

For encoder origin compare E−L (direct external transfer), EA−E (DAPT), and
EA−L (complete external package), for both J and R. These require four additional
downstream fits per seed after the local J/R fits, not a repeat of all specialists.
If external architecture/corpus/parameter count differ, describe the benefit of
that package; do not attribute it solely to pretraining scale. If E is incompatible
with the native encoder, audit and resolve the architecture before launch.

## 2. Required attribution and qualification

Routing is subordinate to the pooling question. Compare R−J. A no-source
adapter N must agree with shared J within an explicitly locked tolerance before
interpreting bank effects. Use matched N/RD/RO/RDO source controls at the locked
rank, query and update policy. Their bank interaction is
(RDO−RD)−(RO−N). An outcome bank can be redundant with task-specific heads;
a weak bank increment does not imply outcome pooling is ineffective.

Use tuning only for a bounded method decision: first source bank at rank 8 and
interacting query width 128; then rank 8 versus 16 for the candidate bank; then
additive 128 versus interacting 128 and 512 at that rank. Every fit uses seed
42; no seed replication is planned. Prefer the simpler configuration when
performance is within the locked practical tolerance. A shared or disease-only
model is an acceptable final result. Confirm anchored versus borrowed-update
policy and optimizer epsilon from the chosen reference; do not mix policies
inside a bank contrast. The prepared branches are alternatives, not instructions
to run the entire Cartesian sweep. A bank decision made at rank 8 is conditional
on that screen; final controls can reveal rank-dependent conclusions, which
must be reported rather than prompting held-out reselection.

Characterize coverage and negative transfer: which cells have estimable
comparisons, which gains matter in absolute terms, and which cells deteriorate.
Distinguish absent endpoints, single-class evaluation, no training events,
unsupported estimates and failed fits. Never silently intersect predictions to
make a comparator look complete. Cells lacking S cannot enter the full factorial;
full-panel coverage beyond that subset remains valuable but is a separate result.

## 3. Required secondary questions: reuse locked predictions

These require analytical outputs, not additional neural training:

| Question | Contrast/output | Interpretation constraint |
| --- | --- | --- |
| Who benefits from pooling? | P1–P6 by training patient/event support, disease, endpoint group | Rarity association, not causal scarcity |
| Can routing rescue negative transfer? | R−J within strata defined by tuning J−S | No held-out oracle strata |
| Is pooling complementary or redundant? | P6 across cells and support | Metric-specific interaction |
| Are improvements calibrated? | Brier/log loss, calibration plots/bias/ECE | No held-out recalibration |
| How stable are gains over follow-up? | All six horizons with paired draws | Correlated horizons, not independent replications |
| Do patients with little recorded history benefit? | Paired patient loss gains by preindex history, age and covariates | Prespecified bins and source provenance |
| Where does external transfer work? | E−L, EA−E by disease, rarity, calendar year and domain mismatch | Descriptive subgroup heterogeneity |
| Does DAPT change the need for routing? | (R_EA−J_EA)−(R_E−J_E) | Locked readout; no external retuning |
| Is the benefit worth the complexity? | Performance versus GPU time, parameters, memory and labels | Equal epochs are not equal cost |
| Are routes plausible and variable? | Gate/source use versus training support and target | Attention/gates are not evidence of source benefit |
| Does delayed coding affect conclusions? | Date-lag audit and same-fit exclusion sensitivity | Diagnosis date is a coding proxy; verify exact subtype at index |

Run the patient bootstrap with common subject draws across models, outcomes,
horizons and training seeds. Average seed effects within each draw; report seed
range/SD separately. This interval conditions on the fitted seed roster, and does
not magically estimate uncertainty over all possible training runs. The runner
exports shared draw vectors for dependence-aware summaries. A cell estimate must
retain its support, missingness and uncertainty reason.

## 4. Statistical backbone and hierarchical analysis

The evidential foundation is raw patient-matched effects and their uncertainty.
Show these alongside the hierarchical plot. Partial pooling is useful for
summarizing noisy cell heterogeneity and avoiding rankings dominated by tiny
cells. It is not mandatory for a valid model comparison, and a visually smooth
curve is not evidence of model adequacy.

Use a simpler crossed disease/outcome/cell model with a linear log-event-support
trend as the reference; compare the existing spline model, flat versus nested
outcome effects and reasonable prior scales. Declare whether disease-group and
clinical-family labels organize the model or only organize figures. Parents
encode an exchangeability assumption, not a claim that their children share
biological mechanisms. A common curve with varying offsets does not allow every
disease to have its own scarcity-response shape.

Before formal Bayesian intervals: prior predictive checks on each metric scale,
posterior predictive checks, convergence/ESS/divergences, sensitivity to nesting
and smoothness, and dependence-aware uncertainty using the shared patient draws
or a validated joint covariance model. The existing rarity model's independent
observation likelihood is not sufficient for correlated cells/endpoints/seeds.
The question-aligned linear and covariance-aware implementation is now in
`opera.analysis.bayesian_contrasts`, with independent contrast bases and exact
posterior decomposition identities. The new runner saves prior/posterior checks,
raw-versus-shrunken estimates and finite-panel summaries. Production fits still
require real-data model and sensitivity checks; keep paired effects primary.
See `bayesian_question_analysis_20261014.md` for model assumptions and execution.

Use 365 days as the main display and retain all horizons. The current secondary
runner evaluates known-horizon-status probabilities; these are distinct from
the existing full-risk-set CIF/IPCW evaluation. Do not mix their estimates or
label one as the other. Evaluate competing-risk calibration and censoring
assumptions with the full-risk-set pipeline before making population-risk claims.

**Analysis lock still required before held-out inspection:** primary metric and
censoring estimand; main weighting (recommended equal fine diseases and endpoint
groups, with patient-weighted sensitivity); minimum support; clinically relevant
tolerance; multiplicity across headline contrast families; seed roster; tuning
method-selection rule; subgroup bins; reportable/disclosure-cleared outputs.
The checkpoint criterion is already implemented: complete tuning-set
endpoint-group-macro known-status horizon BCE, lower is better, one checkpoint
per fit. Do not replace it with cell-specific held-out best epochs.

## 5. Interesting extensions: only if a stronger claim needs them

1. A prespecified external S/D/O/J subset tests whether external pretraining
   changes the pooling axes themselves. Choose informative cells using training
   support and clinical coverage before inspecting external held-out gains.
2. Training-data downsampling separates scarcity association from an actual
   learning-curve intervention. This requires new training and is not essential.
3. Source removal/shuffling or matched extra-capacity controls test a routing
   mechanism. Retrained removal identifies a different intervention from
   inference-time source disabling; neither is obtained from gate weights alone.
4. Equal-compute comparisons test efficiency independently of the scientific
   pooling contrasts. Keep them separate from the matched-epoch primary study.

These extensions must not delay the three headline questions or the required
no-new-training analyses. Embedding maps are descriptive, supplementary and
optional; they cannot establish transfer benefit.

## 6. Execution and manuscript contracts

The exact symbolic contrasts are in
`opera/configs/manifests/final_contrasts_20261014.yaml`; formulas and estimands
are expanded in `contrast_plan_20261014.md`. The prediction schema and existing
analysis outputs are in `locked_prediction_analysis_20261014.md`.

`pretraining_finish_handoff_20261014.md` describes bundle preparation, phased
execution and direct external export. `paper/FIGURE_OUTLINE.md` maps questions to
figures and artifacts. `paper/opera_manuscript.tex` follows this hierarchy;
placeholders must be replaced only by locked, approved results.

## 7. Manuscript gap plan

`paper/MANUSCRIPT_GAP_PLAN.md` inventories every `\placeholder` and
`\plotplaceholder` in `paper/opera_manuscript.tex`, plus the promised but
unmarked items. For each one it names the exact result needed, the pipeline
stage and output table (`prediction_analysis/*.csv`, `bayesian/*/posterior_*.csv`,
Table 1 exports), and whether a script already produces it. It also lists the
scripts still to be written (compute inventory, cell and patient metadata
builders, coverage reasons, external vocabulary coverage, pretraining corpus
summary, absolute-metric intervals). It gives the order of operations from
pretraining completion to text, and the decisions that must be locked before
held-out evaluation. Primary numbers come from the `cif_ipcw` analysis run and
known-status numbers from a separate `binary_known_status_unweighted` run,
each under its own `OPERA_PAPER_ANALYSIS_ROOT`.
