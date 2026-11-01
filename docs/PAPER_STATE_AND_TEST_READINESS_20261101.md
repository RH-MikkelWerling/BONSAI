# Paper state and test readiness — 1 November 2026

Reviewed HEAD `7d5f9c6`, the current manuscript sources, experiment record,
closing-note draft, figure plan, model/queue implementations and the author's
training-log screenshot. No server access, new empirical prediction exports,
national checkpoint receipt or held-out evaluation was available in this review.
The author confirmed that tuning rehearsals remain the proving ground while
we prepare the test release. No test evaluation was launched.

## Assessment

The study has a clearer prospective organising idea, but the new empirical
story is ahead of the manuscript and final analysis handoff. The new model
components have useful tests. We are ready for the final **tuning rehearsal
of the new design**, not yet for an unattended full test pipeline.

This requires closing specific decisions and exercising the new path, not
another architecture search or a repeat of every historical audit. Manuscript
polish and final panel spacing do not themselves need to delay fixed test
predictions. Outcome/metric/threshold/model choices do.

## What has changed since the parallel-audit review

| Earlier focus | Current direction | Evidence and qualification |
|---|---|---|
| Sharing representations and routing task corrections | Share an encoder, let each outcome attend to its token states, add own disease/outcome corrections | Stages 9/12/13, model code and tuning logs. |
| OPERA defined as the revised AB router | J_OA_OWN is the simplest leading candidate | Primary selection remains unfilled in the closing note. |
| Better routing might recover more transfer | Many alternative correction-sharing designs stay near the existing router in log AUC | Requires paired analysis under the primary estimand before declaring no added benefit. |
| Similar correction arrows left the mechanism unclear | A model without routed sources can improve through outcome-specific reading plus own corrections | Distinct from a frozen removal of borrowed terms in a router trained to use them. |
| Pairwise borrowing maps as the main explanation | Pooling ladder, reading/correction comparisons, outcome attention and task-parameter structure | Retire source-route maps for a primary without sources; retain relevant historical controls in the supplement. |
| National pretraining might change the representation | Still planned; must carry the selected readout into both direct/adapted national arms | Code supports this; finished pretraining alone is not a downstream result. |

A suitable argument is: **learn the common record representation across
diseases and outcomes, let outcomes use it differently, and allow small
task-specific departures.** Explicit routing of other tasks' corrections is
one tested option, rather than the definition of borrowing itself.

Keep the distinctions:

- Joint encoder learning is already borrowing. A small extra gain from routing
  is not evidence that information sharing did nothing.
- A frozen router can depend on borrowed terms while a separately trained,
  simpler design learns an equally good or better predictor without that route.
- Outcome-specific attention plus corrections improving over J does not isolate
  a benefit of outcome sharing in the encoder. That would require the relevant
  matched single-outcome readout control.
- The selected architecture may use outcome queries and disease corrections,
  but near-equal alternatives do not establish a general clinical law that
  diseases never need different attention to the record.
- New readout ingredients are not individually new statistical principles.
  The substantive contribution remains the clinical setting, controlled sharing
  comparisons, strong alternatives, temporal evaluation and population transfer.

Describe the many tuning-driven design additions as model development. The
confirmatory test protocol can be fixed now; that does not make the entire
historical search prespecified. Similarly, a small average outcome-sharing
difference is not proof of no negative transfer in every task. Use the declared
margin, interval and cell-level harm summaries for that claim.

## How to read the screenshot

`scripts/monitor_router_runs.py` reads **`val/cr_cif_auc_macro`**. It averages
finite outcome/horizon AUCs from pooled validation patients. Checkpoints are
selected by **`val/cr_horizon_bce_endpoint_group_macro`**, not best AUC.
The table's final `best` column is not the selected predictor.

| Fit | Log AUC at loss-selected checkpoint |
|---|---:|
| J | 0.7711 |
| R_AB | 0.7836 |
| R_AB seed 43 | 0.7834 |
| J_OA_OWN | 0.7916 |
| J_OA_OWN_PH | 0.7912 |
| J_OA_LR_OWN | 0.7910 |
| J_OA_OWN seed 43 | 0.7865 |
| Shrinkage variants shown | 0.7905–0.7916 |

The leading log difference is +0.0080 versus R_AB and +0.0205 versus J,
or +0.80 and +2.05 AUC percentage points. This is encouraging, not the final
primary-paper contrast. The primary analysis ranks within fine-disease cells
with the declared patient matching, censoring convention, support and weighting.
Do not carry these log differences directly into the results macros.

The reading model's shown seed gap is 0.0051, much larger than the AB router's
0.0002. The two shown reading seeds average 0.78905 on this log metric, still
above R_AB, but this does not replace the planned paired analysis or establish
sampling/training uncertainty. The planned stage-14 s44/s45 and second J seed
are not shown in the screenshot. Preserve seed 42 as the prespecified primary;
do not choose the best seed after observing all four.

The seed-range parsimony rule is a development heuristic, not a confidence
interval or formal equivalence test. Report its definition honestly. Complete
the closing note from paired tuning estimates, including the sensitivity to
cell versus disease balancing, before freezing the selected design.

## The manuscript has not yet caught up

There are several distinct sources:

- `paper/opera_manuscript_reading.tex` still inputs the default `opera_body.tex`.
- `paper/opera_manuscript_consolidated_review.tex` selects
  `opera_body_consolidated.tex`, but points to the October rehearsal figures
  and number macros.
- `paper/opera_manuscript_ldh_submission.tex` uses `paper/ldh_edition/`.

They are not interchangeable current versions. Since `2790dfb`, the consolidated
body has gained only a 14-line **comment**, which does not change the rendered
argument. Its abstract, Methods, how-to-borrow Results and Discussion still
describe an expert-routing primary. The reading-copy body also remains routed.
The new experiment record and figure plan are more current than either text.

Required editorial pass once the primary design is fixed:

1. State the sharing question in the abstract; describe the actual readout and
   reserve routing for the comparator/sensitivity result. Change the OPERA
   expansion only after the design choice is recorded.
2. Replace the routed Methods with shared token states → outcome attention →
   own corrections → risk heads. Keep the matched S/D/O/J factorial explicit.
3. Rebuild the how-to-borrow Results around J, reading alone, own corrections
   alone and their combination, with support/clinical-domain variation. Do not
   claim synergy from marginal point gains without the matched factorial
   contrast and uncertainty.
4. Make the pooling ladder the central figure. Put the historical design search
   in the supplement. Source removal is not a mechanism of a source-free model.
5. Keep the XGBoost comparison frank. The earlier pooled analysis favoured it;
   no current paired new-OPERA result establishes reversal. National pretraining
   is a pending experiment, not an expected rescue.
6. National comparisons vary vocabulary/code retention and laboratory-value
   resolution as well as pretraining population size. The current consolidated
   Methods correctly acknowledges this; retain that qualification.
7. Keep clinical score strata size-matched without outcome-based boundary
   selection. The separate survival-matched Low analysis needs its own frozen
   development rule. Neither justifies a blanket superiority statement before
   the paired clinical results are available.

Older prose currently says five cases/controls is the minimum needed to estimate
an AUC. It is the study's minimum reporting-support rule, not a mathematical
requirement. Also preserve the distinction between the 52 loss-weighted endpoint
families and the broader clinical display groups; inspect the generated config
before choosing either phrase in Methods/Discussion.

## Concrete release status

| Item | Status | Required next evidence/action |
|---|---|---|
| AKI provenance | Resolved | Keep the existing release. No AKI-driven refits. |
| New attention/correction forward, gradients and config reconstruction | Tested locally | Finished selected-fit receipts and production inference still need rehearsal. |
| Primary design and seed | Draft | Complete pending fits used by the selection rule; paired tuning results; fill/sign the closing note. |
| Metric/weighting parity | Existing implementation audited; new fit not yet reconciled | Run the new exports through the same primary cell and disease-balanced analyses. |
| National readout generation | Implemented/tested | Verify exported encoder/vocabulary/config/cutoff, then direct and adapted national J/OPERA fits. |
| National route export | **Fixed in this review** | Route exports now follow readout capability, not R_E/R_EA aliases. Sync this small change. |
| Inherited routing inputs | **Fixed in this review** | Inventory templates discard old route paths; only actual matching exports repopulate them. A reading primary cannot inherit the earlier router's routes from the template. |
| Local/national selected model identity | Supported by aliases | Use the same chosen readout and fixed learning-rate rule. Rebuild inventory in a new workspace. |
| Cross-seed noise/range comparisons | **Separate tuning diagnostic added** | Main paired analysis groups by training seed. Use audit_reading_seed_rehearsal for cross-seed comparisons and actual training-seed provenance; do not read missing main-table contrasts as zero. |
| Best/last checkpoint sensitivity | **Not yet wired through the queue** | Queue hardcodes best.ckpt and has no last option/parallel sensitivity inventory. Rehearse explicit, separately identified outputs before promising this analysis. |
| Test model roster | **Unresolved scope** | Earlier notes keep design search tuning-only; closing note section 3 proposes every candidate at best/last. User asks to continue proving on tuning; no expanded test scope inferred. |
| Test CLI | Documentation corrected | The accepted split is held_out, not test. No test command executed. |
| Full explanatory/rehearsal driver | **Still contains router-only steps** | Replace/skip routed vectors, frozen-source mechanism and source removals for an attention primary; use the new task-structure/attention analyses. |
| Clinical strata/classification | Supported; final new-model run pending | Recompute new-model groups, KM/specificity anchors and recall/precision/MCC with explicit split/threshold provenance. |
| Fine-disease availability at prediction | Existing bounded sensitivity, still relevant | Carry the earlier diagnosis-timing sensitivity to the selected disease-conditioned model; do not silently interpret retrospective subtype assignment as a baseline feature. |
| Final figures | Some replacements implemented, not all integrated | Test new recipes against actual new-model tables. Cosmetic iteration can continue after scientific inputs are fixed. |

The queue's held-out filter currently tests whether a fit has a canonical alias.
All the new `router_fix` candidates have aliases, so **that filter is not a
signed final-model allowlist**. An unrestricted stage-14 held-out evaluation
would include the search designs. Use explicit retained fit names/a frozen
manifest once the scope is agreed; do not assume the label "paper model" closes
this decision automatically.

The closing note also has an earlier highest-AUC rule followed by a later
seed-range parsimony rule. The later amendment should be expressed as one
unambiguous selection algorithm in the final signed version. National fits can
be tested later under a genuinely frozen design/training/selection rule, but
local test results must not guide further national-model redesign.

The closing note's two-seed average of the selected design is available without
new fits only if that actual design has both replicas. Stages 12/14 repeat
J_OA_OWN; they do not automatically supply a second seed for every possible
stage-13 winner. Keep that conditional statement explicit.

For pretraining provenance, distinguish supervised test-patient exclusion from
the pre-2022 record cutoff. Pretraining can legitimately include earlier records
of people treated in the later test period under the existing protocol. Verify
the agreed cutoff for records and fitted vocabulary/bin statistics rather than
quietly imposing a different patient-exclusion policy.

## Figure interpretation check

The new task-organisation plot used separate normalisation ranges for its two
heatmaps but only one colourbar. Fixed it to use a shared symmetric scale and
added a regression test. This is a display correction; no model outputs changed.

Query similarity and adapter `up @ down` similarity are parameter diagnostics.
The latter is not the complete nonlinear function `up · GELU(down · x)` or an
observed patient-risk correction. The former is not a causal attribution of
which clinical events matter. Use per-head attention/IG or removal checks for
patient-event claims. Show endpoint-family labels/ordering as display context,
not evidence by itself that the model discovered the taxonomy.

The current task exporter uses validation-case counts for the outcome plot and
sums training events over outcomes for its disease `size` field. The latter is
not a unique patient count. Do not label it "patients"; obtain actual training
patient denominators if that is the chosen explanatory axis. Neither issue
blocks risk prediction, but both matter for the proposed mechanism claims.

## Finite next rehearsal

1. Finish the already registered seed/ablation/design fits needed by the current
   rule. Stop adding new architecture stages.
2. Produce completed-fit tuning predictions for J, R_AB, the reading/correction
   candidates and required seed/ablation controls. Use best.ckpt selected by
   loss. Verify actual checkpoint receipts, not the monitor's best-AUC column.
3. Apply the primary paired cell estimator, disease-balanced sensitivity and
   coverage rules; select the primary by the agreed rule. Populate the closing
   note. Keep uncertainty conditional on fitted parameters separate from seed
   variability.
4. Create one clean tuning inventory with the new OPERA alias. Run paired
   results, probability/calibration checks, clinical groups/metrics and the new
   task-structure/interpretation exports. Assemble the intended figure set,
   explicitly excluding retired router analyses.
5. Verify the national handoff and prepare its matched downstream fits. Freeze
   their rules before looking at local test results if released separately.
6. Freeze the exact test model/checkpoint/analysis roster, confirm whether the
   last-checkpoint sensitivity is retained and exercise its output paths, then
   run the same pipeline with held_out in a separate output tree.

No blanket source-file freshness check is proposed. Existing valid pretraining,
independent baseline fits and scientific input locks remain reusable.

For the seed comparison, after the new tuning inventory contains the completed
replicas, use the CPU-only sidecar (replace the inventory path):

```bash
.venv-v100/bin/python -m opera.run.audit_reading_seed_rehearsal \
  --analysis-config /actual/new_tuning_inventory/analysis.yaml \
  --output-dir "data/paper_256_v3/analysis/reading_seeds_$(date -u +%Y%m%dT%H%M%SZ)" \
  --replicates 1000
```

It reads declared tuning native-CIF artifacts only, refuses unequal patient/task
rosters, and pairs replica aliases in a separate analysis context. Outputs keep
left/right training seeds, use both declared weighting schemes, and include
absolute common-roster cell AUCs for the seed mean/range. Seed-zero in these
tables is explicitly a pairing context, not a training seed. Missing replica
aliases are listed. No model is chosen automatically. Primary-analysis seed
columns and artifacts are not rewritten.

## Validation in this review

**90 tests passed** across stages 9–14, encoder-rate configurations, task-structure
export/figures, paired seed diagnostics and the paper workflow. This includes
eight readout-aware route export cases, stale route-template checks, cross-seed
pairing checks and a shared-colourbar regression test added here. Existing warnings
concern plotting deprecations and a test's tensor-to-scalar conversion, not test
failures. `git diff --check` passes. The tests use local synthetic fixtures;
they are not a substitute for a selected production-checkpoint tuning rehearsal.
No model training or predictions were changed by these fixes.
