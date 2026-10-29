# Agenda — 29 October 2026

Aim: finish a finite tuning rehearsal, establish which router is being used
and what it contributes, then freeze the analysis before opening held-out
results. Existing endpoint definitions stay fixed. No outcome reconstruction
or architecture sweep is planned.

## Full fresh rerun requested on 29 October

After syncing the complete updated source and `code_manifest.sha256`, run
from the repository root inside a GPU allocation:

```bash
opera_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_full_root="data/paper_256_v3/analysis/full_tuning_rehearsal_$opera_stamp"
mkdir -p logs

nohup .venv-v100/bin/python -u -m opera.run.rerun_full_tuning_rehearsal \
  --analysis-root data/paper_256_v3/analysis/local_figures_20261028_143127 \
  --output-dir "$opera_full_root" \
  > "logs/full_tuning_rehearsal_$opera_stamp.log" 2>&1 &

echo "Run directory: $opera_full_root"
echo "Main log: logs/full_tuning_rehearsal_$opera_stamp.log"
```

Monitor with:

```bash
tail -f "logs/full_tuning_rehearsal_$opera_stamp.log"
```

The launcher verifies runtime source checksums before expensive work. It
copies the locked inventory declarations into a fresh workspace and recomputes
the paired analysis from the original base prediction exports. The model
checkpoints, original base prediction/encoder-embedding exports, metadata,
vocabulary and frozen outcome definitions are inputs, not refitted here.
There is no test evaluation and no model training.

Fresh stages include router replay, paired metrics/bootstrap analysis,
training-prediction export and clinical matching, all configured arrows,
mean/zero source-removal inference and analysis, recent-record gaps, IG and
its checks/summaries, the full configured input-restriction analysis, embedding
and plot tables, figures, manuscript numbers, supplementary tables, tabular
metric reconciliation and final review assembly/packaging. IG and restriction
summaries transferred into the unified workspace come only from this rerun.
The national model is not added by rediscovering or selecting new fits.

Outputs below `$opera_full_root`:

- `full_rehearsal.json`: stage status, inputs and paths.
- `logs/`: one log per stage.
- `analysis/clinical_scores/`: updated score figures and diagnostics.
- `review/`: the final assembled figure suite, including fresh IG/restrictions.
- `paper_numbers/` and `supplement_tables/`: final aggregate manuscript outputs.
- `paper_review.zip`: the existing allowlisted figure/aggregate review bundle.

The ZIP is not a full analysis backup and does not include the separate final
manuscript-output directories. Existing clinical partitions that fail the
declared matching rule remain explicit gaps. A successful process exit does
not by itself certify all release checks or scientific interpretations.
Inspect `full_rehearsal.json`, the score inventory and review missing list.
Do not sync changed runtime code into the checkout while this job is running.
Use sufficient scheduler walltime. `nohup` does not extend that limit. If the
allocation ends, retain the workspace and logs for stage-specific recovery
rather than deleting them or starting a second fresh run automatically.

**National encoder status, updated 29 October:** pretraining stopped at epoch
3 overnight. Relaunch is planned for tonight, with completion expected on
30 October. Completion and downstream tuning results are not yet verified.
Today's local router/provenance and figure checks can proceed independently.
Before held-out evaluation, fix the national downstream/adaptation choices as
well as the local ones. A resumed run should retain the saved optimiser and
scheduler state where applicable, and its actual steps/tokens must be recorded.

The test set is ready to open once implementation and selection decisions are
fixed. A visually striking source map or a positive routing result is not a
release requirement. A reproducible weak or diffuse borrowing result is a
valid finding and should narrow the claim.

**First server report reviewed, 29 October:** the existing router replay
**passed**. R_L is the intended revised `fix_R_AB` fit; saved checkpoint and
prediction hashes match, all 46 sampled patients replay within tolerance at
all six horizons, seed-42 health passes, and neural AUC/Brier reconciliation
agrees to numerical precision. Borrowed corrections affect predictions.
This completes the first replay check for the reported inputs. It does not
complete ordered bank-label verification, second-seed stability, tabular
reconciliation or the clinical/national release checks. See
`docs/ROUTER_RELEASE_REPORT_REVIEW_20261029.md`. Do not repeat the GPU replay
for unchanged inputs merely because another report is still pending.

## 1. First job: verify the existing router and its predictions

Sync the reviewed source and checksum manifest to the server. Run inside a
scheduler allocation, from the repository root. The expected completed
source-removal workspace is below. Check that it is the intended workspace
rather than relying on whichever `.env` is active.

```bash
opera_python=.venv-v100/bin/python
opera_analysis=data/paper_256_v3/analysis/local_figures_20261028_143127
opera_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_audit="data/paper_256_v3/analysis/router_release_audit_$opera_stamp"

test -f "$opera_analysis/overnight.env"
test -f "$opera_analysis/analysis_inventory/analysis.yaml"
mkdir -p "$opera_audit"

"$opera_python" -m opera.run.verify_router_predictions \
  --env-file "$opera_analysis/overnight.env" \
  --device cuda \
  --patients-per-cohort 2 \
  --output-dir "$opera_audit/replay"
```

This existing command verifies bundle/config/checkpoint/prediction hashes,
checks joint/router patient and follow-up alignment, recomputes metrics from
saved tuning predictions, and replays a deterministic, label-blind sample from
each fine cohort. It also compares the full router against its encoder/head
alone and against zero borrowed corrections. It runs inference, not training.
The default replay tolerance is `atol=1e-5, rtol=1e-4`. A passing replay is a
sampled implementation check, not proof of benefit across all patients.

**Return `replay/report.json` for review.** Inspect subprocess logs on the server
if it fails. Keep `selected_subjects_INTERNAL.json`, prediction parquets and
patient representations on the server. The report contains aggregate checks.

The expected identities from the reviewed download are:

| Role | Scientific run identity | Required interpretation |
|---|---|---|
| Current primary local OPERA / R_L alias | R_AB_L, `79d39ad1e7cb51472026` | Revised router with the recorded non-collapsed routing |
| Original sensitivity router | R_ORIG_L, `0de5f3abae9725a220e5` | Original collapsed router, retained as a sensitivity |
| Matched own-task control | RBH_AB_L, `6e2936e25eccab82590e` | Same revised model family without borrowed corrections |

These are scientific/run identifiers, **not checkpoint-content hashes**.
The actual hashes must be checked by the server report. The same selected
checkpoint must feed performance, route summaries, source removal, arrows,
clinical comparisons and attributions. Changing an environment alias does
not rewrite an existing inventory.

If no GPU allocation is available yet, an optional CPU preflight is:

```bash
"$opera_python" -m opera.run.audit_tuning_router \
  --env-file "$opera_analysis/overnight.env" \
  --analysis-root "$opera_analysis" \
  --verify-artifacts --route-diagnostics \
  > "$opera_audit/artifacts_and_health.json"
```

The replay command already includes this provenance audit and full metric
reconciliation, so the preflight is not another required computation.

### Additional check: tabular versus sequential evaluation

The router replay's default metric reconciliation covers neural models only.
Before release, also reconcile XGBoost and pooled XGBoost with the same paper
estimates. This added read-only CPU check accepts the canonical tabular tables.
It neither refits models nor opens the test set:

```bash
"$opera_python" -m opera.run.audit_tuning_router \
  --env-file "$opera_analysis/overnight.env" \
  --analysis-root "$opera_analysis" \
  --recompute-metrics --metric-models X XP J_L OPERA \
  > "$opera_audit/cross_family_metrics.json"
```

Every requested model must resolve to tuning artifacts. The updated audit resolves `OPERA`
through `opera_model`; with the earlier audit version, request `R_L` instead
of `OPERA` for the current primary fit. Check every reconciliation
entry for zero missing saved estimates, zero finite-status mismatches and
numerical agreement of native, recomputed and saved AUC/Brier (approximately
`1e-10` or better). An existing saved `analysis_manifest.json` must declare
`estimand: cif_ipcw` and the intended support and bootstrap settings. The
command reports discrepancies for review rather than treating its exit code
alone as a pass. It does not independently audit the raw feature extraction.

Also inspect the existing tabular feature alignment report and feature manifests.
Primary X/XP must use `sequence_matched`, not the separate registry-augmented
XR arm. Confirm that raw feature lookback windows end before treatment and
that patient membership/index dates match the sequence cohorts. Recomputing
metrics cannot establish absence of post-index information in source features.
The detailed local findings are in
`docs/EVALUATION_PARITY_AUDIT_20261029.md`.

## 2. Routing health and source interpretation

First obtain the existing stage-3 router-fix decision report and the revised
router's second-seed health. The expected report is usually
`data/paper_256_v3/router_decisions/router_decision_router_fix_rule.json`.
Verify its fit names and phase rather than treating its filename as proof.
If the second-seed health is absent, recompute it from the existing tuning
route/support exports. This does not require fitting another model.

The existing stage-3 rule is support-weighted, per patient/target:

- No single source is the top contributor in more than 50% of rows.
- At least three different sources are the modal top borrowed source across
  target diseases.
- Median effective contributing-source count is greater than 1.5.

The newly received server replay report confirms the seed-42 AB health pass:
top-source share 15.07%, six distinct top borrowed sources and median effective
count 76.018. This reproduces the earlier diagnostics for the selected fit.
It does not establish stability across seeds. The original router shows the
earlier single-source collapse.

Before interpreting source labels, verify the **ordered** disease/outcome
source lists against the saved checkpoint/model configuration and
`source_route_support.json`, including fine-disease indexing. The aggregate
usage table has 24 disease sources, 69 outcome sources and one shared slot.
Count equality alone does not establish correct label ordering.

Keep three questions separate:

1. **Implementation:** correct fit, labels, active correction path and replay.
2. **Allocation:** non-collapsed patient-level gates, including second-seed
   behaviour if claiming stability. Entropy of average gates is insufficient.
3. **Predictive contribution:** paired results for full versus joint, full
   versus own-only, and fixed-model mean/zero replacement. These are different
   contrasts. Raw gate mass or projected arrow length does not measure benefit.

Review AUC and probability accuracy together, on the same supported roster and
with the same averaging weights. Revisit disease course and cardiovascular
outcomes at the cell level, with case/control counts and uncertainty. A
diagnostic restricted to better-supported cells can expose fragility, but it
does not replace the frozen >=5 primary support rule.

The current tiny shrunken pair estimates reproduce the saved estimator. If
they remain tiny after identity checks, report weak evidence for individual
source effects. Do not keep modifying the router until the map looks selective.

## 3. Refresh the figures from completed analyses

**Clinical review update, 29 October:** the refreshed report has 343 scored
DLBCL patients for NCCN-IPI, compared with 909 across subtypes in the old
recipe. The broad IPI comparison still has 909. NCCN-IPI low-risk counts are
26 for the model versus 24 for the index. The old large expansion is not the
same comparison. IPI and R-ISS low-risk expansion does not preserve matching
tuning mortality. Do not claim matching evaluation risks/specificity merely
because thresholds matched training targets. Review
`docs/CLINICAL_POPULATION_AND_THRESHOLDS_REVIEW_20261029.md` before release.
Five group partitions need the new cutoff diagnostics. Their independent
binary anchors are retained even when a valid ordered partition is unavailable.

After router identity is verified, regenerate with the corrected code. Reuse
the completed mean/zero source-removal outputs. Do not rerun those GPU passes
because source columns or heatmap labels changed.

For the original workspace's plot tables, figures and manuscript numbers:

```bash
OPERA_PYTHON="$opera_python" bash scripts/rerun_tuning_figure_outputs.sh \
  "$opera_analysis" --render-only
```

This updates general plot tables and renders, with backups of the preceding
figures. **It does not rerun clinical-score evaluation or replace the existing
`clinical_scores` images.** It does not redo the full paired analysis. In
particular, the expanded
S/D/O/J/OPERA strategy bars require the newly generated
`strategy_selection_inclusive.csv`. If that file is absent, a CPU analysis
refresh is needed for those bars; the render-only launcher cannot create it.
That refresh reuses saved predictions and does not train models.

Do not blindly rerun `finish_local_figure_review.sh`: it creates new clinical
and IG workspaces and launches those computations. First check which previous
stages actually finished.

### Refresh the clinical family separately

The reported NCCN-IPI image with “1 years”, three groups and tuning-derived
cutoffs is an old render. Running the general `--render-only` command leaves
that image untouched. After syncing the current source and checksum manifest,
run this from the repository root inside a GPU allocation:

```bash
opera_python=.venv-v100/bin/python
opera_analysis=data/paper_256_v3/analysis/local_figures_20261028_143127
opera_clinical_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_clinical_root="data/paper_256_v3/analysis/clinical_training_groups_$opera_clinical_stamp"

"$opera_python" -m opera.run.complete_local_figures \
  --analysis-root "$opera_analysis" \
  --output-dir "$opera_clinical_root" \
  --clinical-only

opera_clinical_dir="$opera_clinical_root/clinical_scores"
```

This exports training predictions from the selected fitted model, develops
thresholds on training patients, evaluates on tuning patients and renders the
current clinical figures. It does not retrain models, run IG or repeat source
removal. Check `completion.json` and `clinical_scores/score_inventory.csv` for
gaps. Open **the new** `$opera_clinical_dir/fig_clinical_score.pdf`, not the
unchanged image under the original `local_figures_20261028_143127` directory.
The computed NCCN-IPI comparison retains Low, Low-intermediate,
High-intermediate and High categories. Already-coarsened registry categories
cannot be reconstructed without numeric totals and remain a separate audit
limitation.

If a **completed clinical workspace already contains training predictions**,
avoid repeating inference. Rebuild its reports on CPU instead, using that
workspace's parent directory (the directory containing `analysis_inventory`,
not its `clinical_scores` subdirectory):

```bash
# Set opera_completed_clinical_workspace to that existing clinical workspace.
"$opera_python" -m opera.run.refresh_clinical_figures \
  --workspace "${opera_completed_clinical_workspace:?Set the completed clinical workspace}" \
  --output-dir "$opera_clinical_root"

opera_clinical_dir="$opera_clinical_root/clinical_scores"
```

These are alternatives. Each uses a fresh output directory. The CPU command
refuses to proceed if the training prediction export is absent. Do not resume
an older completion workspace after syncing changed scientific code.

**Recovery for `Missing canonical prediction fields: ['data_contract']`:**
the training export was registered without the release contract, and the
shared reader also lacked explicit permission for training rows. Both are
fixed. Sync the updated source and checksum manifest, then reuse the export
from the failed clinical workspace on CPU. For the reported run:

If the same traceback recurs with `read_artifact` returning
`normalize_predictions(frame, estimand)` without `allow_training`, or the
refresh subprocess still at line 51, the server is running the pre-fix source.
Sync all six changed modules, not just the refresh entry point:
`opera/evaluation/secondary_analysis.py`,
`opera/run/analyse_locked_predictions.py`,
`opera/run/clinical_figure_provenance.py`,
`opera/run/complete_local_figures.py`,
`opera/run/refresh_clinical_figures.py` and
`opera/run/render_clinical_score_panels.py`, together with
`code_manifest.sha256`. Before rerunning, check the modules that the server
interpreter actually imports:

```bash
.venv-v100/bin/python - <<'PY'
import inspect
from opera.run import analyse_locked_predictions as reader
from opera.run import render_clinical_score_panels as renderer
from opera.run import refresh_clinical_figures as refresh
from opera.evaluation.secondary_analysis import normalize_predictions
assert 'allow_training' in inspect.signature(reader.read_artifact).parameters, 'Old prediction reader'
assert 'allow_training' in inspect.signature(normalize_predictions).parameters, 'Old prediction normalizer'
assert 'training_artifact_with_contract' in inspect.getsource(renderer.main), 'Old clinical renderer'
assert 'training_artifact_with_contract' in inspect.getsource(refresh.main), 'Old refresh command'
print('Clinical contract fix present:', refresh.__file__)
PY
```

Proceed after that prints `Clinical contract fix present`. Use a new timestamp
for each retry because failed refresh directories are retained.

```bash
opera_python=.venv-v100/bin/python
opera_clinical_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_clinical_root="data/paper_256_v3/analysis/clinical_report_refresh_$opera_clinical_stamp"

"$opera_python" -m opera.run.refresh_clinical_figures \
  --workspace data/paper_256_v3/analysis/clinical_training_groups_20261029_123537 \
  --output-dir "$opera_clinical_root"

opera_clinical_dir="$opera_clinical_root/clinical_scores"
```

This copies the matching model/seed's registered tuning release contract into
the new training-artifact declaration and retains the `train` split. It does
not relabel training patients as tuning, modify prediction parquets, bypass
identity checks or repeat inference. Training rows remain rejected by the
general analysis reader unless explicitly enabled by the clinical caller.

Clinical renders now record renderer hashes, evaluation/threshold splits,
score definitions and figure hashes. Review assembly rejects an explicitly
selected stale clinical folder. Without an explicit folder, it omits stale
clinical outputs and reports the missing family. Packaging also rejects
legacy clinical images without a current receipt. This checks freshness and
the declared comparison settings, not clinical validity or upstream leakage.

Assemble completed clinical, attribution and omission outputs explicitly:

```bash
# Set these to the existing completed workspaces before running.
# opera_clinical_dir=.../clinical_scores
# opera_attribution_dir=.../attributions
# opera_restriction_dir=.../input_restrictions

"$opera_python" -m opera.run.assemble_figure_review \
  --analysis-root "$opera_analysis" \
  --clinical-dir "${opera_clinical_dir:?Set the completed clinical_scores directory}" \
  --attribution-dir "${opera_attribution_dir:?Set the completed attributions directory}" \
  --restriction-dir "${opera_restriction_dir:?Set the completed input_restrictions directory}" \
  --output-dir "$opera_audit/figure_review"

"$opera_python" -m opera.run.package_figure_review \
  --review-dir "$opera_audit/figure_review" \
  --analysis-root "$opera_analysis" \
  --output "$opera_audit/paper_review.zip"
```

Review the stage manifests and missing-family list before calling this
complete. Assembling does not launch missing inference. A partial bundle is
still useful to return. Follow the existing disclosure review before export.

Required tuning rehearsal checks:

- Real clinical figures use native score categories and the intended frozen
  training-derived thresholds. Compare achieved event rates/specificity as
  well as extra low-risk patients and events found. Export recall, precision,
  specificity, MCC and confusion counts at each threshold.
- Real arrow exports identify the same checkpoint, outcome and split as the
  base patient vectors. Both vectors use the same projection. Missing or
  near-zero arrows remain missing or near-zero rather than being manufactured.
- Attribution concepts use the saved adaptive mapping and lab-bin metadata.
  Provisional or ambiguous molecular labels do not silently become verified
  held-out findings. Re-summarise completed IG where possible rather than
  repeating the costly attribution calculation for a label change.
- The recent-record omission analysis is present and identifies its checkpoint
  and hidden windows. It is distinct from attribution-faithfulness checks.

## 4. While the server audits run

Locally, review the 29 October gallery and sign off the figure structure,
counts, captions and interpretation. Finish the tuning results version and
the limitations from the frozen outcome-code audit. Keep factual results
separate from explanations in the Discussion.

Check the national arms' current state. Pretraining completion alone is not
enough: downstream tuning-based checkpoint/model selection and any cohort
adaptation choices must be fixed before seeing held-out results. If those
choices are already fixed, remaining computation need not delay a lock.

## 5. Lock and open the test set

Create the actual dated release record, not just a draft/template:

- Model/encoder/seed roster, primary OPERA identity, original-router sensitivity,
  matched controls, checkpoint hashes and tuning-only selection rules.
- National direct/adapted arms and any national-vs-local naming/selection rule.
- Frozen endpoints, horizons, eligibility, support rule, score categories and
  threshold artifacts. No held-out threshold fitting.
- Primary estimand and averaging weights, paired uncertainty method, contrast
  definitions and handling of unavailable models/unstable cells.
- Software/config/data-release fingerprints, known limitations, timestamp and
  amendment history. Essential rehearsal stages either pass or are explicitly
  excluded from confirmatory claims.

Then create a **fresh held-out workspace** and run the frozen local and national
roster through the same analysis. The completion/recovery launchers above are
explicitly tuning-only; do not simply change their environment to held_out.
Test execution commands should be prepared from the signed-off roster and
reviewed before execution. The commands above do not launch held-out evaluation.

After opening, fix genuine implementation errors with a recorded amendment.
Do not use the results to change the primary router, score thresholds,
comparison population or national-model selection. Cosmetic figure work can
continue without treating each render as a new model-development round.
