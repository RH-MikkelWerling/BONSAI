# Validation-stage experiment review

## Follow-up audit: 30 September 2026 bundle and outcome exports

Inputs: `recent_validation_20260930T084626Z.tar.gz` and
`outcome_export_audit_20260930T085117Z.csv`. The bundle contains completed,
unlocked `encoder020`, `routerinteraction`, and `weightinggroups` entries.
`encoder020` has two metric logger versions because training resumed; these
must be interpreted as one training trajectory rather than two independent
runs. Its best *logged* endpoint-group monitor value is 0.745729 at zero-based
epoch 10 (version_0); version_1 peaks at 0.731950 at epoch 24 and finishes
at 0.727975 at epoch 29. The bundle excludes checkpoint weights, so it does
not certify which epoch the server's `best.ckpt` contains after resume.

At each run's best logged endpoint-group epoch, `encoder020` versus
`weightinggroups` gives endpoint-group CIF-AUC 0.745729 versus 0.754358,
overall macro 0.761033 versus 0.771430, 365-day mean across 87 outcomes
0.736587 versus 0.746413, and mean 365-day CIF Brier 0.08045 versus
0.07928 (lower is better). The 0.20 encoder multiplier therefore does not
improve this one-seed validation comparison. At 365 days it exceeds
`weightinggroups` on 21/87 outcomes, with median paired delta -0.00982.
The interacting router finishes at 0.748904 endpoint-group and 0.743602
365-day outcome mean, also below `weightinggroups` on those scores; its
source/freeze policy differs and this is not an isolated router effect.

The outcome export audit lists exactly the 87 registry outcomes. It finds no
duplicate or missing IDs, missing index/censor dates, invalid event chronology,
invalid split labels, or split-date mismatches in any file. This establishes
structural integrity only, not correct source grading, feature timing,
observation eligibility or clinical validity. Row counts vary by endpoint
eligibility (13,546 to 37,467); they must not be summed as unique patients.
The XGBoost feature-profile metadata summary has not yet been supplied, so
RKKP inclusion in the completed pooled baseline remains unverified.

## Audited bundle: 2026-09-29 16:17 UTC

### Updated audit: complete archive at 19:09 UTC plus XGBoost validation CSV

Inputs: `recent_validation_full_20260929.tar.gz` and
`xgboost_validation_aggregates.csv`. The previously omitted `weightinggroups`
run is **completed** and now changes the shortlist. Both files are
validation-only. No held-out predictions were opened.

The completed endpoint-group-weighted low-rank BCE run selects zero-based epoch
21 on its configured endpoint-group monitor: macro AUC **0.771430**, endpoint-
group AUC **0.754363**, family macro **0.736430** and 365-day pooled outcome
mean **0.746414** (87 outcomes). It starts from the same DAPT source checkpoint
as weightingreference. At the corresponding selected epochs, reference has
0.766486 / 0.748679 / **0.744696** / 0.744126, and the frozen-encoder
additive router has 0.765491 / 0.748904 / 0.739711 / 0.743602. Thus
weightinggroups leads the first, second and 365-day metrics, but reference
leads family macro. This is a substantive objective tradeoff, not an across-
the-board victory. The router source and freeze policy differ, so this is
not an isolated architecture ranking. All are single seed 42.

For clarity, the **unweighted mean of 87 pooled outcome-specific neural CIF
AUROCs at 365 days**, taken at each run's configured-monitor-selected epoch,
is 0.746414 weightinggroups, 0.744126 weightingreference, 0.743602 router,
and 0.735104 shared BCE. The incomplete BCE-attention history gives 0.737354
at its best currently logged monitor epoch. These are not 0.771/0.766/0.765
multi-horizon macro values; nor are they disease-balanced means.

At 365 days, weightinggroups exceeds weightingreference for 56/87 outcomes
(median paired outcome change +0.00626), but loses notably on second-line
treatment (-0.0582), next hematological malignancy (-0.0847), and treatment
failure transformation (-0.0342). Rare outcome deltas need uncertainty and
event support before interpretation. Weightingreference's better disease-control
and treatment-trajectory behavior may matter more clinically than overall macro.

The XGBoost CSV contains 87 distinct ALL_HEMATOLOGY outcomes, all with
`split=tuning`, `evaluation_regime=fixed_horizon`, seed 42. Mean unweighted
per-outcome AUROC is **0.775204** (median 0.784163); 11 outcomes have fewer
than 20 positives, including two with fewer than five. Representative values:
overall survival 0.8376 (260/1967), treatment failure 0.7050 (471/1733),
second-line treatment 0.6601 (290/1733), all-definition infection 0.7104
(937/1925), anemia G3 0.8078 (575/1927), creatinine G2 0.8750 (390/1924).
These are fixed-horizon binary AUROCs with outcome-specific eligible sets.

The apparent 365-day neural CIF-AUC means (best weightinggroups 0.7464;
XGBoost binary AUROC 0.7752) are **not a matched estimate**. The metrics can
differ in handling competing death, early censoring, prediction target and
eligible controls. Numerically, XGBoost exceeds weightinggroups on 73/87
outcomes, but that is descriptive and cannot establish superiority on the
same estimand. A validation-only paired evaluation of both score sets on the
same patients/eligibility and event/censoring definitions is now the highest-
value comparison. If score predictions remain only on server, compute paired
aggregate metrics/uncertainty there and export aggregates, not patient rows.

The new archive still lacks interacting-router, encoder multiplier and other
post-bundle experimental results. BCE outcome-attention remains incomplete in
the metadata, and the duplicate root3 BCE-MLP entry still has no metrics.
Before new training, inventory `opera_independent_v4` and any newer roots once
again; do not infer that absent runs failed.

**Revised shortlist:** weightinggroups and weightingreference as co-leading
low-rank policies; router as mechanistic candidate; shared BCE as simpler
neural control; XGBoost as essential strong tabular control. Do not prioritize
the MLP or older piecewise readout variants. Resolve the matched baseline and
final MEDS release before interpreting <0.01 AUC differences as robust gains.

**Still-unfinished experimental questions:** (1) an interacting router is
configured but absent from the bundle; (2) the additive router improvement
does not identify learned routing because its shared head/adapters train, so
fixed-routing and continued-shared-BCE controls are needed; (3) a comparable
router warm-start/source and trainable-encoder ablation are not completed;
(4) outcome/positive balancing and minimum-unique-patient safeguards are not
implemented/tested in this multi-outcome objective; (5) disease/outcome
mixture and pooled-vs-single-disease studies remain the central scientific
tests; (6) repeat seeds and final MEDS replication are still pending.
The independent launcher has `routerinteraction`, `encoder005`, `encoder020`
and `encoder100` task names, but a task name is not evidence of a completed run.

### Recovery and final-data priorities (study-team follow-up)

The existing server results must be inventoried before commissioning duplicate
runs. The archive's exclusion manifest confirms 1,218 excluded artifacts under
`root1/pooled_xgboost_validation_365d`, including aggregate metrics and reports.
The study team located that run under
`data/models/outcome_readout_pilot_harmonized_v4/pooled_xgboost_validation_365d`
and located the omitted `weightinggroups` suite under
`data/models/endpoint_weighting_pilot_v1`. The previous bundle roots did not
include this latter suite. `data/models/opera_independent_v4` and
`data/models/horizon_bce_readout_combinations_v4_verified` are also relevant.
The bundler labels every `evaluation/` path sealed, regardless of its split;
thus absence from the bundle is NOT absence from the server. The neural shared
BCE run also has 154 excluded evaluation artifacts. Their split must be checked
before recovering aggregates; do not lift the filter wholesale or export raw
predictions/held-out outputs. `weightinggroups` is absent from both included
members and exclusions because its suite root was not supplied.

`opera.run.export_validation_evaluation_aggregates` now reads only
`evaluation/<outcome>/result.jsonl`, requires an explicit validation split and
matching outcome name, then exports an allowlist of scalar aggregate fields.
It refuses to write anything if no validation-labelled rows are present.
The default `opera.run.evaluate` split is held-out, so a directory named
"validation" is not sufficient proof. The transfer CSV still needs routine
small-cell disclosure review before external sharing.

Priority order:

| Priority | Work | Deliverable / decision gate |
|---|---|---|
| P0 | Recover existing neural and XGBoost validation aggregates; locate weightinggroups and any newer runs | Complete run registry, actual best-checkpoint identity, common-panel validation shortlist; no duplicate training |
| P0, parallel | PHAIR compatibility preflight now, before large training finishes | Exact architecture/config/vocabulary/preprocessing contract; small early checkpoint loads and runs on target environment |
| P1 | Freeze final MEDS release and audit against v4 | Immutable data/version hashes, patient/split/endpoint counts, vocabulary and sequence drift, numeric-value transforms; do not assume changes are negligible |
| P2 | Reproduce selected pooled XGBoost and shared BCE/low-rank candidates on final data | Same data, masks, horizons, selection panel and seed; this establishes final-data baseline, not a full historical sweep |
| P3 | Minimum-support inverse-square-root rebalancing ablation on the strongest simple BCE setup | Baseline vs one balanced condition, equal optimizer steps, calibration/AUPRC and rare/common endpoint changes |
| P4 | Interacting router vs additive/fixed-routing/continued-source controls; then encoder LR multipliers | Source/warm-start/freeze parity, bounded training budget; test only combinations supported by earlier evidence |
| P5 | Pooled vs single-disease and selected outcome/disease mixtures | Paired disease x outcome metrics and Bayesian rarity/forest-plot evidence; replicate finalist effects across seeds |
| P6 | Large PHAIR encoder vs current encoder, with identical selected downstream head(s) | Isolate encoder gain; treat DAPT as a separate ablation if needed; memory feasibility first |
| Final | Lock model and analysis; one held-out evaluation | No test-driven selection or subsequent retuning |

Final MEDS is the intended primary release; v4 results remain labelled pilot
evidence. Preserve the frozen split assignment unless an explicit protocol
revision is made. The encoder's original token IDs and numerical normalization
must not be overwritten by a newly generated final-data vocabulary/scaler.
New codes require an explicit mapping/unknown-token or versioned adaptation
strategy, and unknown-code frequency must be measured.

Rebalancing must be gated by DISTINCT training patients within each relevant
positive/negative pool, not summed events across tasks/horizons. The router's
NMZL source has five training patients, so it is not a reasonable aggressive
oversampling pool. Prespecify a support threshold after inventory, keep low-N
cells naturally sampled or pooled/shrunk, cap patient probabilities and repeated
draws, mix with uniform sampling, and log unique patients/repetition/effective
sample size. These safeguards are a proposed experiment contract, not an
implemented sampler.

PHAIR handoff checklist:

- Checkpoint plus exact BONSAI commit, resolved architecture/model config,
  training cutoff/data provenance, and environment/package/CUDA versions.
- Exact vocabulary token-to-ID mapping and special tokens; numeric binning or
  normalization assets, units, temporal encoding/origin and truncation rules.
- Export consolidated model weights readable on CPU, rather than requiring
  source-cluster-specific distributed shards or custom training services.
- Verify strict encoder reconstruction/state loading; accept only intentional
  new downstream heads. Never use unrestricted strict=False as compatibility.
- Compare deterministic synthetic-sequence encoder outputs in both environments,
  then a one-batch downstream forward/backward on the real target hardware.
  SDPA backend and FP16/FP32 fallback must be tested if the PHAIR run uses
  FlashAttention/BF16 unsupported on the target V100.
- Check VRAM at actual hidden size/depth/sequence length. Architecture
  compatibility does not guarantee the large model fits for end-to-end tuning.
- The local manifest requires torch>=2.10 whereas earlier server tracebacks
  showed torch 2.7.1. Reconcile the actual target environment explicitly;
  do not assume editable installation or cross-version checkpoint loading.

Local native reconstruction already supports `bonsai-native-rope-v1` with
explicit value/time encoding metadata and strict backbone loading. This is
infrastructure, not a compatibility certificate for the not-yet-supplied PHAIR
configuration/checkpoint. An early tiny checkpoint plus config is the fastest
way to settle the remaining uncertainty before expensive pretraining finishes.

Source: `recent_validation_20260929T161712Z.tar.gz`, read directly without
extracting or executing bundled code. Validation only; no held-out predictions
were inspected. The bundle contains 12 inventory entries, 11 metric histories,
and 10 completed entries. One root1 BCE-MLP entry has no metrics; the root2
BCE-attention history ends at epoch 19 with no completion marker.

The table reconstructs the epoch maximizing each run's configured monitor and
reports all metrics from that same epoch, NOT independently maximized metrics.
It does not prove the serialized best checkpoint corresponds to that epoch:
completion metadata lacks the saved checkpoint epoch/score.

| Run | Monitor epoch (zero-based) | Macro AUC | Family macro AUC | Endpoint-group AUC |
|---|---:|---:|---:|---:|
| Shared piecewise-exponential | 25 | 0.744310 | 0.725461 | unavailable |
| Piecewise + outcome MLP | 15 | 0.747544 | 0.727874 | unavailable |
| Piecewise + family trunks | 10 | 0.743413 | 0.728474 | unavailable |
| Piecewise + outcome attention | 25 | 0.745858 | 0.729387 | unavailable |
| Piecewise + attention prototypes | 18 | 0.744457 | 0.729344 | unavailable |
| Piecewise + low-rank prototypes | 23 | 0.745217 | 0.727266 | unavailable |
| Shared horizon BCE | 16 | 0.757338 | 0.729126 | unavailable |
| Horizon BCE + outcome MLP | 10 | 0.755023 | 0.726971 | unavailable |
| Horizon BCE + attention (incomplete) | 18 | 0.761158 | 0.732396 | unavailable |
| Horizon BCE + low-rank reference | 17 | 0.766486 | 0.744696 | 0.748679 |
| Horizon BCE + additive frozen-encoder router | 29 | 0.765491 | 0.739711 | 0.748904 |

Old runs monitor family macro; the final two monitor endpoint-group macro.
Macro AUC averages finite outcome x horizon metrics pooled across cohorts; it
is not a disease-balanced score or a 365-day-only score. Endpoint-group selection
uses support admission and has 413 included outcome x horizon cells in both new
runs. Thus old/new monitor scores are not interchangeable. Old runs must be
rescored under a frozen common panel to make a definitive ranking.

### What moves performance

- Horizon BCE is the most consistent apparent improvement: shared BCE exceeds
  shared piecewise by 0.013028 macro AUC at their monitor epochs. Historical
  competing-risk implementation differences remain a confound; rerun a matched
  baseline before attributing the full difference solely to the objective.
- Low-rank BCE reference improves another 0.009148 over shared BCE, but changes
  grouping/selection as well as readout. Do not call this a clean rank ablation.
- Attention gives a smaller promising BCE improvement (+0.003820), but the
  run is not marked complete. Extra MLP capacity does not help: train epoch loss
  falls to 0.2208 while validation loss rises to 0.2412 and macro AUC falls to
  0.7372 at epoch 29 (from an epoch-13 macro peak of 0.7561).
- Router macro increases from 0.757381 at epoch 0 to 0.765491 at epoch 29;
  endpoint-group AUC increases from 0.739474 to 0.748904. Both peak at the final
  logged epoch, so the 30-epoch cap, rather than a demonstrated plateau, ends it.
  This supports more adaptation time, not indefinite continuation.
- Router vs low-rank reference is effectively unresolved: group AUC differs by
  +0.000225, macro by -0.000994, and family macro by -0.004984. Single seed,
  different starts/freeze/loss aggregation; no superiority conclusion.
- Router pooled 365-day AUC improves for 65/87 outcomes relative to the original
  shared BCE source at its monitor epoch; median paired outcome change +0.008746.
  Larger gains include next hematological malignancy (+0.0574), neutropenia G3
  (+0.0414) and thrombocytopenia G3 (+0.0370). Larger losses include AST G2
  (-0.0364), fibrinogen G3 (-0.0341) and myocardial infarction (-0.0309).
  These are exploratory, correlated endpoint comparisons without uncertainty;
  sparse-support extremes must not be interpreted as reliable clinical effects.
- Pooled 365-day outcome means: source 0.735104,
  router 0.743602, low-rank reference 0.744126. Multi-horizon headline macro
  scores should not be pasted next to 365-day XGBoost AUROC.
- Macro Brier at monitor epochs: source 0.068163, router 0.066706, low-rank
  0.067128. This is reassuring but not a substitute for per-endpoint calibration.
  Different loss reductions make the absolute validation loss between router
  and reference unsuitable as a direct quality ranking.
- Route entropy rises 4.1254 -> 4.1588 while shared weight falls 0.1225 ->
  0.1083. This does not establish useful specialization: adapters and shared
  head also train. Fixed-routing and continued-source controls are required.

### Missing evidence

1. `weightinggroups` (~0.771 previously reported) is absent. Its precise metrics
   and config are the highest-priority addition. No interacting-router run is
   included; no encoder 0.20/1.00 sweep or cohort/outcome-mixture run is included.
2. XGBoost validation results are absent. Need pooled and disease-specific
   baselines evaluated on identical outcome/horizon eligibility and censoring
   conventions, plus the same supported selection panel.
3. Disease-stratified validation metrics, paired uncertainty, AUPRC and detailed
   calibration are absent. Logs supply pooled outcome/horizon AUC and Brier,
   not the disease x outcome comparisons needed for the OPERA sharing claim or
   Bayesian rarity/forest-plot analyses.
4. Save best checkpoint epoch/monitor score, resolved data/population hashes,
   and historical code revision. Current-checkout code is not historical proof.
5. Source training support is available for the router, but reports only 5
   training patients for NMZL. Its 34 training events span outcomes and do not
   imply 34 independent patients. Assess patient support before strengthening
   rare disease adapters or inverse-frequency sampling.

### Prioritized next experiments (not launched)

1. Recover weightinggroups and completed attention artifacts. Re-evaluate the
   shortlist and XGBoost on a common validation-only panel. Retain held-out data.
2. Use the best compatible source with warm-start parity. Low-rank weights are
   not directly interchangeable with a shared-head router source; compare
   shared-source warm starts and separately supervised low-rank baselines.
3. Test additive vs interacting router, same source, with fixed-routing and
   continued-shared-source controls. Use a bounded longer adaptation budget
   because the present router still improves at epoch 29. Keep encoder frozen
   initially to isolate adaptation; then test 0.05/0.20/1.00 on the winner with
   matched unfrozen controls. Fine-cohort labels must be prediction-time valid.
4. First rebalance on the simpler winning BCE readout, not simultaneously with
   a router redesign. Establish baseline + tempered inverse-square-root +
   effective-number alternative, equal training steps, same selection panel.
   Repeat the shortlist across seeds rather than expanding every combination.
5. Only then run pooled-vs-single-disease and outcome/disease-mixture studies;
   they test the information-sharing hypothesis and enable the rarity analysis.

### Collaborator's sampling suggestion

The collaborator pointed to upstream BONSAI. Its inspected main commit
`e2deb33b4ffe54baeabff2b8ce859e4da670f7a3` implements
`bonsai.functional.sampling.inverse_sqrt`: patient weight is 1/sqrt(n_class),
and WeightedRandomSampler draws len(labels) samples with replacement. The local
fork has the same inverse_sqrt formula. This is the concrete available recipe,
although the collaborator's actual configuration remains unverified:
https://github.com/FGA-DIKU/BONSAI/blob/e2deb33b4ffe54baeabff2b8ce859e4da670f7a3/bonsai/functional/sampling.py .
For a 1:99 class ratio, expected sampled minority fraction is 1/(1+sqrt(99)),
about 9.1%, not 50%. Upstream fine-tuning prohibits combining sampling and loss
weighting. Its separate sqrt loss helper sets pos_weight=sqrt(n_negative/n_positive).
The sampler assumes one class label per patient and is not wired to OPERA's
multiple outcomes/horizons. No sampler experiment was implemented or launched.
All supplied configs use random minibatches and class_balanced=false; that flag
alone is not proof of how the absolute BCE loss would be reweighted.

Prefer a tempered training-only proposal, with floors/caps and a uniform-mixture
component. Counts must be eligible known positive/negative labels at the chosen
outcome/horizon, not censored-as-negative patients. Multi-label patients have
one patient sampling probability: naively summing rare positives can massively
oversample co-morbid/high-observation patients and correlated grade endpoints.

Separate (a) outcome/group importance, (b) positive-negative balance within a
task, and (c) patient sampling. They answer different questions. For absolute
risk, enriched sampling requires explicit inverse-proposal correction of the
patient loss (and natural-distribution censoring/support estimation). Without
correction it changes the target distribution and can damage calibration; a
weighted objective can be studied deliberately, but must be labelled as such.
Do not double-balance by using uncorrected enrichment and inverse-frequency
loss weights together. The current competing-risk sampler guard rejects
non-random enriched sampling; adding a config flag alone is not sufficient.

An established comparator is inverse effective-number class weighting rather
than raw inverse frequency: https://arxiv.org/abs/1901.05555 . This supports an
ablation candidate, not evidence that it will improve these specific models.

The remaining sections preserve the pre-bundle plan/history; the audited
results above supersede earlier user-reported running values where available.

## Current evidence (reported by the study team, not yet audited)

| Condition | Reported macro AUROC | Interpretation pending bundle review |
|---|---:|---|
| Low-rank endpoint-group weighting (`weightinggroups`) | ~0.771 | Leading reported candidate |
| Low-rank hierarchical-support reference (`weightingreference`) | ~0.765 | Changes aggregation and family priorities together |
| Additive anchored router, frozen encoder (`routergroups`) | ~0.760 at epoch 14 | Improving; not merely stuck, but below the leading candidate |
| Interacting router (`routerinteraction`) | Not established | Implementation ready; do not treat as completed evidence |

These are user-reported running/peak values, not selected-checkpoint scores.
Confirm macro definition, horizon coverage, eligible patients and source identity
before ranking. The router uses a compatible shared-BCE source and a frozen
encoder; low-rank pilots start from DAPT with encoder fine-tuning. Their difference
does not isolate routing quality. The recent pretraining-population/indexed
population distinction must also be documented.

## Compact extraction contract

`opera.run.bundle_recent_validation` accepts multiple suite roots. Includes:

- Frozen configs, overrides, checkpoint metadata and completion status.
- Sanitized training/validation metric histories; epoch and step retained.
- Router shared mass, entropy, residual magnitude and source weights if logged.
- Training interval support, null hazards and source training support.
- Run inventory and per-metric minima/maxima/latest observations.
- Checksummed contents and a clearly labelled current-checkout code snapshot.
- Tabular metadata only when its evaluation split explicitly says validation.

Never exports raw predictions, embeddings, patient tables, weights or logs.
Paths mentioning held-out/test/evaluation are excluded. Metric histories retain
only train/val columns. Logged summaries could still be disclosure-sensitive;
use the institutionally approved transfer channel. Files from active runs are
not an atomic snapshot; export again after completion. The bundle does not
evaluate models or recover missing per-outcome metrics. A metric peak is not
necessarily the result at the checkpoint selected by the configured monitor.
Tabular console metrics cannot be reconstructed from metadata; retain that gap
explicitly and generate validation aggregates on-server in a subsequent step.

## Review sequence and decision gates

1. Audit completion, source checkpoint, dataset/vocabulary, splits, readout,
   objective, weighting, optimizer, encoder freezing and code provenance.
2. Reconstruct epoch validation histories. Plot epoch training losses only;
   batch/step losses are not epoch averages. Inspect curriculum and LR timing.
3. Evaluate selected checkpoints on identical validation eligibility cells if
   existing metrics do not provide them. Include shared-source baseline and
   XGBoost using the same censoring-aware estimand, not unmatched ordinary AUC.
4. Compare paired outcome x horizon x fine-cohort effects, calibration and
   support. Do not equate small aggregate changes with uniform changes.
5. For routers, inspect residual magnitude and checkpoint gradient magnitudes;
   compare learned routing, fixed home routing and shared-only predictions.
   Shared-only at the adapted checkpoint is not the original source: its head
   has also trained. Evaluate both explicitly.
6. Select at most one mechanistically justified final router protocol, plus a
   matched shared-model control. Keep any interacting-query/frozen run separate
   from staged/unfrozen changes. Freeze the final analysis protocol before
   opening held-out results.

## Disease-level Bayesian transfer forest

The primary transfer contrast is all-disease training minus local-disease
training, evaluated on the SAME patients of each fine disease, at the same
outcome and horizon. Use matched model families (XGBoost vs XGBoost, neural vs
neural), features/eligibility and validation/tuning rules. Clearly distinguish
global zero-shot use from global-initialized local fine-tuning; they are separate
contrasts. A neural-vs-XGBoost contrast answers a different question.

Existing `bayesian_rarity.py` has disease/group effects, outcome/family effects,
rarity splines and `cell_theta` joint posterior draws in `posterior.nc`.
It exports `posterior_cells.csv`, but not a disease-average forest estimand yet.
Derive a disease summary at EACH posterior draw by taking a prespecified
weighted average of its predicted cell effects, THEN compute posterior mean,
95% credible interval and probability of benefit. Do not average interval bounds
or plot a disease random-effect coefficient as its total expected benefit.

Prespecify endpoint-group/family weights and a primary horizon. For comparisons
between diseases, prefer a common clinically applicable task panel; alternatively
show actual-task-mix estimates and label their differing denominators. Report
cell coverage and train event counts. Never turn clinically inapplicable or
unsupported tasks into observed zero effects. Very sparse cells can contribute
through partial pooling without making their standalone estimates reliable.

Two different summaries should not be confused:

- Expected benefit at each disease's ACTUAL rarity: combines rarity and residual
  disease-specific differences, answering which diseases benefit in this dataset.
- Benefit standardized to a common rarity/task distribution: addresses residual
  differences after adjusting for those factors, requiring overlap/support.

Use a main disease forest for the first estimand, alongside the rarity curve.
Use an outcome-by-disease delta heatmap or faceted supplementary forests for
heterogeneity, not a huge main forest with every combination. Export plot-ready
tables: contrast, model family, split, fine/grouped disease, horizon, metric,
weighting/standardization definition, posterior mean/lower/upper, P(delta>0),
P(delta>prespecified meaningful threshold), task coverage and support.

The existing rarity likelihood uses cell-level standard errors; patient overlap
across outcomes/horizons can induce dependence. Joint draw aggregation preserves
the model's posterior dependence, but cannot repair dependence omitted from the
likelihood. Audit that assumption, use patient-level paired bootstrap covariance
or sensitivity analyses before interpreting disease-level intervals as final.
Natural rarity associations are not causal effects of adding patients; controlled
training-size experiments address that separately.
