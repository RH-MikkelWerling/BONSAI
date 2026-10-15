# OPERA contrast plan: pooling, tabular prediction, external pretraining and DAPT

This is an analysis plan, not evidence of results or a launch manifest. The
symbolic inventory is `opera/configs/manifests/final_contrasts_20261014.yaml`.
Execution gates are in `final_experiment_gates_20261014.md`. All new comparisons
use the approved release and matched256 protocol; historical64 findings inform
bounded method choices but do not substitute for matched256 controls.

## Notation and common evaluation

- L: local256 self-supervised encoder; E: external pretrained encoder directly;
  EA: the same external encoder after local domain-adaptive pretraining (DAPT).
- S: one fine disease/one outcome; D: all applicable diseases/one outcome;
  O: one fine disease/all applicable outcomes; J: all/all shared BCE.
- R: the locked router under all/all supervision. RD/RO/RDO: disease,
  outcome, both source banks. N: neither bank, verified against shared J.
- X: tuned cell-specific EHR XGBoost; LR: logistic reference. XR: optional
  registry-enriched XGBoost, reported separately.

Subscripts name encoder origin. Every contrast is evaluated on identical target
disease/outcome/horizon patients with identical labels/known-status masks.
For AUC define `gain(A,B) = AUC(A)-AUC(B)`; for Brier/log loss define
`gain(A,B) = loss(B)-loss(A)`. Thus positive always favors A. Interactions are
differences of gains, on that metric's scale, not latent biological effects.
365 days is the main display; all six horizons and supported cells remain
available. Report paired uncertainty, training support, evaluation support and
seed variability. Use one pooled model checkpoint per fit, not per-cell
held-out checkpoint selection. Common full-tuning BCE selection is specified
in `supervision_factorial_20261014.md`.

## A. Essential contrasts for the three research questions

### 1. How much do the two supervised-pooling axes contribute?

| ID | Contrast | Question |
| --- | --- | --- |
| P1 | gain(D_L,S_L) | Disease pooling with outcome fixed |
| P2 | gain(O_L,S_L) | Outcome pooling with disease fixed |
| P3 | gain(J_L,O_L) | Disease pooling when outcomes already pooled |
| P4 | gain(J_L,D_L) | Outcome pooling when diseases already pooled |
| P5 | gain(J_L,S_L) | Combined pooling benefit |
| P6 | gain(J_L,D_L)-gain(O_L,S_L) | Pooling interaction; equal to P3-P1 |

P5 decomposes along either path: P1+P4 or P2+P3, for each common target
cell on a common metric scale. These are related summaries, not six independent
discoveries. They measure supervised-training procedures at fixed pretraining;
different patient/label volume, loss exposure and optimization are part of the
intervention. Equal epochs do not mean equal compute.

Evaluate every contrast across train event/patient support, fine disease,
clinical endpoint group and horizon. Keep natural rarity and precision separate.
Key hypotheses: one axis helps more in some regions; their combination is
complementary in some cells and redundant or harmful in others. Report negative
transfer explicitly. Missing specialists restrict complete factorial coverage;
pooled-model coverage outside that subset is a separate result.

### 2. Does routing add value beyond ordinary sharing?

| ID | Contrast | Question |
| --- | --- | --- |
| R1 | gain(R_L,J_L) | Incremental value of selected routing |
| R2 | gain(RD_L,N_L) | Disease-bank effect alone |
| R3 | gain(RO_L,N_L) | Outcome-bank effect alone |
| R4 | gain(RDO_L,RD_L) | Outcome-bank increment with disease sources |
| R5 | gain(RDO_L,RO_L) | Disease-bank increment with outcome sources |
| R6 | gain(RDO_L,N_L) | Combined source-bank effect |
| R7 | gain(RDO_L,RD_L)-gain(RO_L,N_L) | Source-bank interaction |

R2-R7 require a matched source-factorial policy/rank/query, not mixing whichever
individual arm had its best exploratory score. RD/RO retain disease/target query
context and all/all supervision. If N and J are protocol- and prediction-identical,
reuse one fit after verification. R can equal RD or RDO; selection is on tuning.
Outcome-only null results do not rule out R4 being positive.

Pre-launch tuning contrasts: rank16 versus rank8; interacting versus additive;
query512 versus query128; plus the recorded anchored/borrowed, matrix epsilon
and curriculum contrasts where evidence is still needed. Change one factor at
a time. These select the method; they are not held-out exploratory searches.

### 3. Is the neural pipeline better than tabular prediction?

| ID | Contrast | Question |
| --- | --- | --- |
| T1 | gain(R_L,X) | Selected local neural pipeline versus usual tabular baseline |
| T2 | gain(J_L,X) | How much is available without routing? |
| T3 | gain(S_L,X) | Sequence/pretraining advantage without pooled supervision |
| T4 | gain(X,LR) | Strength of the tabular comparator |
| T5 | gain(R_EA,X) and gain(J_EA,X) | Adapted external neural pipelines versus the same baseline |

T3 uses existing specialist outputs, with no new fits, and helps distinguish a
sequence/pretraining effect from a pooling effect. It remains a pipeline
comparison, not a pure architecture effect with identical inputs/compute.
XR and clinical scores are secondary complete-case comparisons; rerestrict all
models to identical eligible patients, and do not pool these rows with EHR-only
full-population results. Tabular models fit train and select on tuning. Reuse X
predictions across encoder-origin comparisons when the task contract is unchanged.

### 4. Does external pretraining help, and what does local DAPT add?

The mandatory external-origin matrix is {L,E,EA} x {J,R}. The SAME external
source checkpoint, remapping policy and downstream protocol link E and EA.
Direct E still receives supervised finetuning; only local SSL adaptation is
absent. DAPT is a separate pre-supervision stage, not router/contrastive training.

| ID | Contrast | Question |
| --- | --- | --- |
| E1 | gain(J_E,J_L) | External initialization value under shared supervision |
| E2 | gain(R_E,R_L) | External initialization value under selected routing |
| A1 | gain(J_EA,J_E) | Local domain-adaptation value under shared supervision |
| A2 | gain(R_EA,R_E) | Local domain-adaptation value under routing |
| E3 | gain(J_EA,J_L) | Adapted external versus locally achievable shared model |
| E4 | gain(R_EA,R_L) | Adapted external versus locally achievable final model |
| I1 | gain(R_E,J_E)-gain(R_L,J_L) | External pretraining x router interaction |
| I2 | gain(R_EA,J_EA)-gain(R_E,J_E) | DAPT x router interaction |
| I3 | gain(R_EA,J_EA)-gain(R_L,J_L) | Adapted-external pipeline x router interaction |

E3=E1+A1 and E4=E2+A2 on identical target cells; an external advantage can
be present directly, arise mainly after adaptation, or diminish after adaptation.
EA versus L alone cannot establish that DAPT helped. Audit architecture,
vocabulary/numeric/time correspondence, unmatched-token frequencies, source
pretraining dates/patient overlap and temporal provenance for E/EA. Record
external pretraining and DAPT token/GPU budgets separately. External-versus-local
is a package comparison unless scale is the only changed ingredient.

Run local DAPT using only the permitted SSL subjects/events, with checkpoint
selection on the appropriate SSL validation view, never supervised held-out
outcome metrics. Record DAPT objective, vocabulary, cutoff, optimizer, epochs,
windows and initialization hash. DAPT may improve representation alignment,
learn remapped local-only tokens, and provide extra optimization; A1/A2 measure
that combined stage. They do not isolate domain matching from extra training
compute. A matched extra-SSL control is needed only for that stronger claim.

This requires one external DAPT encoder fit per chosen pretraining seed, not
one DAPT fit per downstream disease/outcome. The E/EA matrix adds four downstream
fits per downstream seed if local J/R already match. Replicate downstream seeds;
if only one encoder/pretraining/DAPT seed exists, describe uncertainty as
conditional on that encoder rather than full pretraining-seed uncertainty.
Direct E local-vocabulary export still needs implementation/audit; never replace
it with a zero-epoch DAPT run. Hold the locally locked router fixed across origins.

## B. Interesting analyses from the mandatory saved outputs

| Question | Analysis | Additional training |
| --- | --- | --- |
| Do the two pooling axes help in different scarcity regimes? | P1-P4 versus train events/patients and clinical groups; partial pooling with measurement uncertainty | None |
| Does routing rescue tasks hurt by ordinary pooling? | P5 and R1 jointly, plus direct R_L-S_L; validate tuning-defined negative-transfer strata on held-out | None |
| Does adaptation primarily repair a domain mismatch or help already strong tasks? | A1/A2 versus train-defined token overlap, EHR source mix, history density and endpoint family | None; descriptive mechanisms only |
| Does DAPT reduce the need for routing? | I2, with per-cell effects and uncertainty | None |
| Which patients have better probability predictions? | Paired Brier/log-loss contributions versus pre-index history density, age, sex and baseline-defined atypicality | None; observed prediction gain, not an individual causal effect |
| Are gains discrimination or calibration gains? | AUC, Brier/log loss and calibration side by side, before/after identical tuning-only calibration rules | Calibration fitting only if prespecified |
| Do methods reach cells where specialists cannot be fitted? | Estimability/coverage table separate from paired-performance table | None |
| Is performance stable across eras/horizons? | Prespecified held-out calendar strata and six horizons; follow-up/ascertainment support alongside estimates | None; no retuning by era |
| Does router use resemble its predictive contribution? | Source weights, correction magnitudes and post-hoc removal sensitivity together | Extra inference; weights alone do not prove benefit |
| Can one pooled model match a bank of independently selected strategies? | Tuning-selected best of S/D/O per cell versus J/R on held-out; preserve selection provenance | None, if tuning artifacts are saved |
| What is the cost of the gain? | Training compute, inference throughput, parameter/storage burden and coverage versus performance | Bounded benchmark |

Prioritize the first four: they connect the three study questions without a
new architecture grid. Secondary figures should show uncertainty, unsuccessful
cells and coverage; do not select attractive held-out cells or subgroups and
present them as prespecified evidence. Patient gain analyses must enforce
prediction/label parity first: existing convenience helpers may inner-join
inputs and are not themselves parity certification.

## C. Bounded extensions worth considering, explicitly optional

1. **Does broad pretraining replace supervised pooling?** Repeat S/D/O/J on a
   prespecified support/clinical-coverage subset with EA, reusing local subset
   outputs. Compare each P1-P6 between EA and L, especially
   `gain(J_EA,S_EA)-gain(J_L,S_L)`. This is the strongest additional scientific
   question, but needs external specialists and intermediate-scope fits. It
   supports a subset claim, not a full-panel one. No need to repeat on E as
   well unless specifically studying DAPT x pooling.
2. **Is cross-disease routing useful beyond disease-specific capacity?** Train
   a home-only disease-source control with the same bank/rank and joint labels,
   then compare learned disease routing against it. A second prespecified static
   cross-source mixture distinguishes learned patient-dependent allocation from
   fixed sharing. Hold gradient ownership/shrinkage/compute policies matched;
   separate those two questions rather than treating post-hoc removal as a fit.
3. **Is sharing especially useful when target labels disappear?** On a small
   prespecified task panel, reduce target-cell training supervision while keeping
   other-cell supervision fixed. Compare S/D/O/J at each label fraction. Mask
   every supervised use of withheld target labels (loss, sampling, null hazards,
   support/curriculum grids); retain permissible other outcomes. This explicitly
   tests a scarcity intervention. No large full-panel fraction sweep is needed.
4. **Does joint training help as initialization even when a specialist wins?**
   Compare a J-initialized cell specialist with the independently pretrained S
   on a subset. The former already receives both pooling axes; record its extra
   training stage/budget. This asks whether pooled initialization plus local
   specialization helps, not whether independent specialists isolate sharing.
5. **Did external pretraining improve frozen features or optimization?** Fit
   matched standardized frozen probes for L/E/EA on the same targets and compare
   their encoder-origin gains with full finetuning. This tests representation
   accessibility under a fixed probe class; it does not uniquely identify the
   cause of every finetuning gain.
6. **Is pretraining itself necessary?** One architecture-matched randomly
   initialized J and optionally R provides a contextual control. It is not
   required for external-versus-local, but is required for an explicit
   pretraining-versus-no-pretraining claim. Freeze its training budget and
   acknowledge that fixed-budget random initialization may be less converged.

Do not queue all optional work. My recommended additions are (1) if affordable,
and (2) if the manuscript makes a learned-borrowing mechanism claim. Most of the
other interesting results should first come from the mandatory predictions.

## Analysis lock and artifacts

Prespecify headline contrasts/metrics, endpoint-group/fine-disease aggregation,
practical tolerance and support criteria before held-out evaluation. Decide
whether diseases receive equal weight or population weight and report the
alternative as sensitivity. Show AUC and Brier separately; avoid a post-hoc
combined score. Related P/E/R contrasts are algebraically dependent: a large
number of positive cell estimates is not that many independent discoveries.
Use a prespecified multiplicity family for formal cell-level claims; exploratory
subgroup/mechanism findings remain labelled exploratory. An inconclusive
interval is not proof of equivalence. No alpha/tolerance is silently locked here.

Save full tuning and held-out prediction artifacts keyed by origin, regime,
source/query/rank policy, seed, subject ID, outcome, horizon and eligibility,
plus config/checkpoint hashes and support/selection inventories. Keep stable
run identities distinct from overloaded legacy `model_family` names. Bootstrap
patients jointly across compared methods and relevant endpoints; account for
training seeds separately. Single-class cells cannot contribute AUC but can
contribute probability-loss/coverage summaries with appropriate interpretation.
Known-status BCE selection and IPCW evaluation are different estimands; do not
silently label the unweighted selector censoring-adjusted population log loss.

Suggested main figures: (i) pooling axes and rarity, (ii) paired tabular/neural
performance and heterogeneity, (iii) L/E/EA by shared/router showing DAPT and
pretraining interactions. Route mechanisms, six-horizon details, specialists,
coverage, calibration and compute belong in supporting panels/supplements.
