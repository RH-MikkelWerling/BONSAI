# Tuning-data draft by 31 October 2026

Working plan agreed from the 27 October conversation. Codex handles local code,
analysis planning and manuscript integration. Mikkel runs commands in the secure
server environment and returns aggregate status/results. No SSH access from
Codex is available. Do not export patient-level files.

## Deliverable and dates

Aim for a coherent, readable local-model draft on **Friday 30 October**, with
**Saturday 31 October** for corrections, supervisor packaging and external
results that actually finish. The core draft must be possible without the
external encoder. It contains real tuning results, explicitly marked throughout,
a filled local Results/Discussion/abstract, approved population descriptions,
main local figures and Table 2, and a compact usable supplement. Outstanding
external analyses and author-supplied details have a short explicit list.

Tuning estimates reuse data involved in checkpoint/model/threshold selection.
Describe them as preliminary/descriptive; intervals do not remove selection
optimism, and they do not establish held-out superiority. Threshold-selected
clinical groups are especially exploratory on the same tuning patients.
Keep planned test evaluation distinct in Methods and prose. Use
`paper/opera_manuscript_tuning_preview.tex` for the assembled draft.

## What we know and what remains unverified

| Work | Evidence now | Next action |
| --- | --- | --- |
| Earlier local factorial/benchmark analyses | Mikkel reports much of stage-1 rehearsal completed | Inventory actual model/metric/figure coverage and failed steps |
| Sampling fits | `fix_J_IPW`, `fix_R_AB_IPW` running; expected 28 Oct morning | Verify completion, export tuning predictions, add IPW_J/IPW_R sensitivities |
| Integrated gradients | Some work completed | Check cache, numerical checks, covered targets and model identity before extending |
| Gap/subgroup additions | Code checked locally | Sync; refresh active recipes; four gap passes and CPU subgroup analysis |
| Clinical scores, borrowing, embeddings, calibrated figures | Renderers exist; server completion unknown | Inspect artifacts/logs and repair only missing or invalid steps |
| External encoder | Expected approximately Thu 29 Oct | Prepare inputs now; arrival is the start of a pipeline, not a final model |
| Tuning manuscript assembly | Wrapper and generated-number machinery exist | Export reviewed aggregates/PDFs, integrate numbers, write interpretations, build |

Existing local tests pass for the latest additions (84 passed, one excluded
because this Mac cannot load the XGBoost OpenMP runtime). This establishes code
readiness, not real server-output completion.

## First status return from the server

Run from the repo root in the activated environment. These commands inspect
status; figure-readiness can return nonzero while inputs are missing.

```bash
qstat -u "$USER"
tail -n 50 logs/rehearsal/remaining.log
bash scripts/run_paper_experiments.sh queue-status --env-file .env.rehearsal --execute
bash scripts/run_paper_experiments.sh status --stage 5 --split tuning --env-file .env.rehearsal --execute
bash scripts/run_paper_experiments.sh figure-readiness --env-file .env.rehearsal --execute
```

If a path/config differs, report the actual path. Return the failed-step summary,
training/prediction status of the two IPW runs, and the missing-figure list.
Where a rehearsal step failed, send its error tail, not full patient-level logs.
Before launching new work, inspect the active environment/configuration to verify
which model, split and analysis root it uses. `.env.rehearsal` and generated
rehearsal copies may differ. Never print the complete private environment file.

## Dependency order and server jobs

### 1. Establish one current tuning snapshot (27–28 October)

Use a fresh analysis root; preserve existing training/prediction artifacts.
Include all six horizons. Set `OPERA_PAPER_INVENTORY_SPLITS=tuning`,
`OPERA_PAPER_PLOT_SPLIT=tuning`, `OPERA_PAPER_FIGURE_MARK=TUNING SET`,
`OPERA_PAPER_ANALYSIS_ESTIMAND=cif_ipcw`, and explicitly resolve the selected
OPERA alias and primary routed alias (`R_AB_L` mapped to analysis R_L).
The attribution recipe separately needs `run.split: tuning`; any Bayesian
recipe separately needs `split: tuning`. Do not rely on the CLI default split.

After the IPW fits finish, evaluate their completed best checkpoints on tuning.
Retain them as sampling sensitivities under the current protocol; completion
does not automatically replace the primary model. The stage-5 bundle contains
the new fits. Check which predictions already exist before evaluating more.
Collect the complete local S/D/O/J, routed/control, X/logistic/XP/XR rosters and
report unsupported cells as such. Do not rerun completed training.

Then inventory -> analysis -> clinical-score outputs -> plot tables ->
figures/numbers/supplement tables. Independent GPU inference and clinical-score
work can proceed while CPU analyses run, subject to actual node resources.
If the two IPW fits slip, freeze the existing local core first and merge their
sensitivity outputs later; do not hold all writing for these two rows.

### 2. Complete targeted interpretation (28–29 October)

GPU priority after the current fits: (i) missing essential tuning predictions;
(ii) four 7/14/21/28-day gaps; (iii) source-removal inference if its main-figure
result is absent. Reuse completed full/restricted passes only under the same
checkpoint, split and restriction recipe. Reuse existing valid IG exports and
regenerate CPU summaries. Expand IG only if the selected illustrative outcome
lacks sufficient reliable coverage; an exhaustive attribution sweep is not a
prerequisite for the draft.

CPU priority: current factorial/benchmark/calibration summaries and Table 2;
clinical-score comparison and headline panel D; subgroup AUC/Brier; sampling
sensitivity; gap comparison; borrowing map/summary if the text uses it.
A short numerical/coverage check precedes interpreting each result. The small
tuning population will not support every rare cell or clinical index; blank
unsupported entries are acceptable and must stay visible.

The gap question is sensitivity to removing recent records at a fixed treatment
index, not forecasting at earlier visits. The borrowing map describes fitted
model dependence; the latest protocol makes the named pairs descriptive.

### 3. Clinical and manuscript prerequisites (start 27 October)

Confirm that the secure clinical-score recipe has actual index identifiers,
correct intended endpoints, score components/categories, model alias and review
provenance. The tracked example contains unresolved fields and defaults to J_L;
it cannot simply be treated as a completed OPERA comparison. Use the intended
endpoint comparisons and one strong DLBCL illustration already supported by the
data. Do not force every rare index into a full plot.

Check the corrected pre-index history summary and current Table 1/flow against
the approved population. Reconcile analysis weighting across generated Table 2,
figure captions and prose. Keep exact source/estimand/horizon/denominator with
all numbers. Resolve source-model details from saved configs/metadata rather
than old defaults. Author/affiliation/funding details can be collected in parallel.

Use the current single-column manuscript as the editorial source. Write Results
from the actual local figures in this order: predictability/coverage, benchmarks,
pooling, routing/borrowing, clinical interpretation, sensitivity. Write the
abstract/conclusion after those paragraphs. State null and harmful results.

### 4. External branch (inputs now; checkpoint expected 29 October)

Prestage the matching external vocabulary, our patients' external-format
sequences, architecture/RoPE/version information, and available pretraining
provenance. Check the import/export path, coverage and resource allocations now.
Do not assume the delivered encoder will outperform the local one.

On arrival: compatibility/provenance audit -> export -> smoke -> local DAPT ->
matched J/R downstream fits from direct E and adapted EA -> tuning prediction
exports -> incremental inventory/analysis/figures. Direct downstream fitting can
in principle run alongside DAPT on separate GPUs, but the current bundle
preparer requires both E and EA paths. Early direct-only launch would need a
small explicit orchestration change; do not point EA at E to bypass the gate.
If the external run is included, use the primary R_AB readout consistently.

At the first smoke/pilot, measure steps/time and remaining GPU-hours from the
actual hardware; include queue and CPU analysis time. Make a go/no-go decision
for this draft from that estimate. Preserve the planned training budget and
selection rules rather than shortening runs to obtain a result by a date.
Four downstream fits plus DAPT may not finish by 31 October.

If only the direct branch completes, report direct transfer and mark DAPT as
pending. If no valid external comparison completes in time, include a concise
pending-results section and the fixed comparison design. The local draft still
ships. Avoid rebuilding every local analysis/attribution around the new encoder
at the last minute; update the selected-model narrative only once its actual
completed tuning comparison has been reviewed.

## Daily milestones

| Date (Copenhagen) | Server work | Writing/integration | Exit condition |
| --- | --- | --- | --- |
| Tue 27 Oct | Status audit, preserve caches, update active recipes; leave two fits running | Freeze core figure/story roster, collect metadata, prepare Results skeleton | Concrete missing-artifact list and job queue |
| Wed 28 Oct | Finish IPW tuning exports; local snapshot and clinical/subgroup/calibration analyses; gaps/source removal as needed | Fill local Results and Table 2; inspect main figure layouts | Core local numbers and figures available, known gaps assigned |
| Thu 29 Oct | Complete interpretation summaries; accept/audit external checkpoint and benchmark its pipeline | Complete local Results/Discussion; integrate score and sensitivity outputs | Readable local draft; external finish forecast grounded in throughput |
| Fri 30 Oct | Finish only essential missing local outputs; external branch continues if feasible | Abstract, conclusion, captions, references, tuning labels, compiled PDF review | Complete first local tuning draft |
| Sat 31 Oct | Export any completed valid external aggregate results | Corrections, external insertion if ready, supervisor package | Dated draft PDF, sources, provenance and short outstanding list |

## Things to check/fix in local code during integration

- `scripts/rehearse_remaining.sh` resolves stage-3 R_L, not necessarily the
  latest primary alias, and sets `concept_mapping_reviewed: true` to exercise
  code paths. Do not blindly reuse it as the publication extraction recipe.
- Generated analysis/figure/attribution recipes do not inherit later template
  updates automatically. Refresh the active copies before subgroup/gap runs.
- `build_paper_numbers.py` currently looks for attribution numbers under
  `attributions/summary/`; the normal summary action writes `summary_365d/`.
  Reconcile this path before expecting attribution macros to populate.
- The tuning wrapper switches figure/number paths, but the evaluation supplement
  still includes `tables/generated/supplement_tables.tex` directly. Wire it to
  the tuning package and correct captions that still say test/one year when
  their source is tuning/pooled horizons.
- Verify Bayesian `split: tuning` if those analyses are included. Fit only the
  models needed for the displayed claims and inspect diagnostics. Unfinished
  hierarchy/sensitivity plots can remain pending in this first draft.
- Check clinical-score output and `headline_scores.csv` before the headline
  figure; otherwise panel D is a genuine dependency blocker.

## Definition of done

Every displayed number/figure has an identifiable tuning source and appropriate
support; the shared patient roster/estimand/weighting is stated; every figure is
marked TUNING SET; the PDF prose distinguishes observations from planned test
claims. The main narrative has no unexplained local-result placeholders. External
and author-input gaps are listed explicitly. No synthetic layouts masquerade as
results, and no held-out data are needed. Bundle the PDF, editable TeX, reviewed
aggregate tables/PDFs and a provenance/missing-items list for the supervisor.

Optional galleries, broad IG sweeps, additional exploratory router designs,
extra seeds, and exhaustive supplementary variants should not delay the core
local draft. Existing sensitivity results are retained; new experiments need a
specific missing claim and enough time to finish.
