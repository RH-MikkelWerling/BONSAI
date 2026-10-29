# Router implementation recheck — 29 October 2026

Some server jobs are running; their exact commands have not been supplied.
The earlier claim that the full tuning launcher had started was an inference,
not a verified job status. A full fresh rehearsal is now planned overnight.
Local edits below have not been deployed there. Leave a checkout used by active
jobs unchanged until they finish, or prepare a separate updated checkout.
The national encoder is also running. Its completion and downstream tuning
results remain pending.

## One confirmed bug, with a narrow computational impact

`SourceAnchoredReadout` reconstructs `router_frozen=True` when the saved
configuration specifies warm-up epochs. That Python flag is not saved in
the state dict. Training updates it as epochs advance, but evaluation loading
does not replay that hook. Previously, `route_weights` detached gate logits
whenever this flag was true, including in evaluation mode.

Consequently, the loaded model produced the correct prediction values but
incorrect input derivatives. Integrated gradients omitted the derivative of
the patient-dependent routing weights. This was reproduced with state-identical
models whose only difference was the warm-up configuration. Forward outputs
were exactly equal, while input gradients differed. A separate numerical
finite-difference check also failed before the fix.

The local fix applies the detachment only when both `self.training` and
`self.router_frozen` are true. Training warm-up retains its original behavior.
Evaluation and IG retain the complete gate derivative, independently of the
reconstructed warm-up flag. Both new regression tests pass after the fix.

Affected outputs: IG values and ranks, IG-selected event-deletion checks,
randomisation/reference comparisons based on IG, attribution summaries,
figures and manuscript numbers derived from them. These need fresh caches.
The existing attribution intent fingerprint includes the readout source, so
resuming an old cache with the corrected implementation should be rejected.

Unchanged numeric outputs: ordinary predictions, AUC/Brier/IPA, clinical score
thresholds, route weights, mean/zero source removal, arrows and ordinary
input-restriction predictions. The fix does not require retraining a model.
It also does not explain nearly identical arrows across outcomes.

## Implementation checks

The reviewed paths and regression tests support the following:

- Data module and readout both sort the fine-disease roster. Outcome indexes
  are shared with the risk head and objective. Arrow export slices the
  explicitly requested outcome, rather than always selecting the first one.
- Training, gradient-cache training, validation and prediction export use the
  competing-risk log-hazard path. Reconstruction and prediction replay have
  separate regression coverage.
- Borrowed adapter parameters are detached for other targets. The patient
  input and routing paths remain differentiable outside training warm-up.
  The global correction is outside the routed mixture.
- Fixed-source mean/zero ablation preserves gates and the other source terms.
  It retains the own-disease source. Outcome/global corrections stay in the
  base term. This differs from the diagnostic that masks and renormalises gates.
- The primary revised fit is `fix_R_AB`, with both banks, home floor 0.05,
  a global adapter and two warm-up epochs. It is not the alternative design
  with an unrestricted own correction outside the mixture.

The existing server report already identified R_L as scientific run
`79d39ad1e7cb51472026`, replayed 46 patients at six horizons within tolerance,
and reconciled neural AUC/Brier to numerical precision. Seed-42 allocation
passed the recorded health rule. The original router remained collapsed.
See `ROUTER_RELEASE_REPORT_REVIEW_20261029.md` for exact numbers and hashes.

These checks do not certify the server's complete ordered bank labels or
second-seed behavior. They also do not establish that individual sources have
clinically meaningful effects. The reviewed tiny pair estimates can be a real
result of diffuse sharing and shrinkage.

## Fresh overnight run versus recovery of an earlier run

A fresh full rehearsal started after syncing the fix computes corrected IG
automatically. It also includes the raw-vector audit and whole-cohort estimates.
Use the command in `AGENDA_20261029_ROUTER_AND_TEST_RELEASE.md`. No separate IG
refresh is required afterward unless that stage fails or a new bug is found.

The commands below are recovery instructions for a run that already computed
IG with the old source. They are not extra mandatory work after a corrected
fresh run.

First retain its logs and `full_rehearsal.json`, then sync the updated source
and `code_manifest.sha256`. Do not start another complete tuning rerun merely
for the IG fix. Use the actual completed full-rerun path below.

```bash
opera_python=.venv-v100/bin/python
# Replace this value with the directory printed by your running launcher.
opera_full_root="${opera_full_root:?Set the actual full_tuning_rehearsal directory}"
opera_post_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_fixed_ig="$opera_full_root/ig_gate_gradients_$opera_post_stamp"
opera_fixed_review="$opera_full_root/review_gate_gradients_$opera_post_stamp"

"$opera_python" -m opera.run.prepare_attribution_refresh \
  --analysis-root "$opera_full_root/analysis" \
  --output-dir "$opera_fixed_ig" --execute
```

This is the GPU step. It recomputes IG and its configured checks, then
summarises them. It uses the same frozen checkpoint and a fresh directory.
Do not point it at the old `ig` directory or its patient caches. If the active
run did not create `analysis/completion.json`, recover that failed stage first
or use the verified original completed workspace for `--analysis-root`.

The following checks use CPU and saved prediction/vector outputs:

```bash
"$opera_python" -m opera.run.audit_router_vectors \
  --vectors-root "$opera_full_root/analysis/routed_vectors" \
  --output "$opera_full_root/router_vectors_$opera_post_stamp.json"

"$opera_python" -m opera.run.evaluate_whole_cohort_outcomes \
  --analysis-config "$opera_full_root/analysis/analysis_inventory/analysis.yaml" \
  --models OPERA J_L X XP --split tuning --horizons 30 60 90 180 365 730 \
  --output-dir "$opera_full_root/whole_cohort_$opera_post_stamp"
```

The vector audit checks outcome markers, checkpoint/split, aligned patient
rosters and shared-vector agreement. It reports pairwise raw-delta equality,
cosines and norms, plus variation after subtracting each patient's mean across
exported outcomes. It exports aggregates only. The subtracted mean is not an
exact isolation of the global adapter. High raw similarity supports a shared
correction explanation. Distinct raw vectors with similar plotted arrows
point toward projection/averaging. Equal nonzero raw vectors warrant checking
the target queries and export provenance before interpretation.

Whole-cohort results retain disease-conditional censoring weights and include
small disease cells before pooling. They report each model's available
population and an explicit intersection across all requested models. A missing
specialist prediction can reduce that common population. Coverage tables make
this visible. Each outcome/horizon is evaluated separately, with paired patient
multiplier uncertainty. Pooled AUC includes between-disease risk differences
and must accompany, rather than replace, within-disease results. Calibration
slopes/intercepts are diagnostic fits and do not recalibrate predictions.
Run a separate fixed OPERA/J_L roster if a full neural-cohort comparison is
needed in addition to the four-model common roster, and report both populations.

After corrected IG succeeds, rebuild the review and attribution-dependent
numbers in new directories. This preserves the old rehearsal unchanged:

```bash
"$opera_python" -m opera.run.assemble_figure_review \
  --analysis-root "$opera_full_root/analysis" \
  --attribution-dir "$opera_fixed_ig/attributions" \
  --restriction-dir "$opera_full_root/ig/input_restrictions" \
  --output-dir "$opera_fixed_review"

"$opera_python" -m opera.run.build_paper_numbers \
  --analysis-root "$opera_full_root/analysis" --split tuning \
  --attribution-summary "$opera_fixed_ig/attributions/summary_365d" \
  --output-dir "$opera_full_root/paper_numbers_gate_gradients_$opera_post_stamp"

"$opera_python" -m opera.run.package_figure_review \
  --review-dir "$opera_fixed_review" \
  --analysis-root "$opera_full_root/analysis" \
  --output "$opera_full_root/paper_review_gate_gradients_$opera_post_stamp.zip"
```

`build_paper_numbers` now accepts an explicit fresh attribution summary and
records that source. If training-support coverage counts are needed, also
pass the existing cell-metadata path from `analysis/overnight.env` using
`--cell-metadata`. The earlier full rerun's coverage numbers remain available.
The corrected review assembler uses the explicit IG directory for its renders
and bundled attribution summaries. Existing clinical validation still applies.
If input restrictions failed, retain that gap rather than silently substituting
an older directory.

Return the new review ZIP, raw-vector JSON, whole-cohort aggregate CSVs/manifest,
cross-family metric report, clinical diagnostics and corrected paper-number
outputs for review. The vector NPZs and attribution patient caches stay on the
server. The standalone whole-cohort outputs and separate paper-number directory
are not automatically included in the figure ZIP.

## Remaining release decisions

The router's implementation no longer has an identified unresolved forward-pass
bug in this audit. Test release still requires the ordered-label and second-seed
reports, cross-family metric reconciliation, corrected IG if used for claims,
clinical threshold/population review, and national checkpoint/vocabulary/config
compatibility with downstream choices fixed on tuning data. Crossed clinical
anchors remain explicit non-estimable partitions, not a reason to invent cuts.
Strong-looking arrows or positive source-pair effects are not release criteria.

## Local verification record

The targeted regression run passed 186 tests, with four CUDA-only cases skipped
on the local machine. It covered the readout, routing controls and both axes,
training/export prediction parity, source removal, usage/projection, decision
rules and dynamics, checkpoint reconstruction, token attribution and real-model
IG, the full rehearsal/completion launchers, metric reconciliation, the new raw
vector and whole-cohort evaluators, attribution refresh preparation and paper
number generation. The two new gradient tests failed before the readout fix
and passed afterward. Whole-cohort tests include missing-model coverage,
fine-disease censoring weights, case-mix discrimination and split isolation.
The raw-vector tests cover alignment, identical/orthogonal/zero corrections and
incompatible exports. Corrected-number tests confirm that explicit fresh IG
replaces stale defaults without overwriting them.

All tracked OPERA/BONSAI source checksums verified after the update, and
`git diff --check` passed. This is local verification, not a claim that the
updated sources or corrected IG have already run on the server.

No held-out predictions were read by these local tests. The new whole-cohort
CLI defaults to tuning, and its test confirms that a held-out artifact declared
in the same inventory is not opened when tuning is requested. The current
completion/refresh launchers remain tuning-only. Prepare held-out commands from
the final dated release roster instead of changing their environment variables.
