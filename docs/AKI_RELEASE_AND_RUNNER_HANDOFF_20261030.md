# AKI correction: release locks and existing runners

30 October 2026. AKI stays in the paper. This is a documented correction to one
endpoint before held-out evaluation. The existing disease roster, 69 outcome
names/order, endpoint groups, clinical families, horizons, splits, selected
architectures and optimisation settings are preserved. The original release
and its fitted models remain identifiable development artifacts.

## What changes at each lock

| Existing object | Correction handoff |
|---|---|
| Sealed cohort, index dates and splits | Unchanged. Changed eligible AKI IDs, index dates or splits stop the repair for investigation. |
| Original AKI measurements/episodes | Retain. Regenerate a separate episode export with only the annual-window inequality corrected, across complete histories. |
| Outcome release | New v2 revision with parent manifest hash. Replace AKI only. Copy the other 68 files and registry byte for byte. Re-audit structure. |
| Outcome lockdown/analysis record | Record the known-error amendment. Do not repeat support-based screening of the outcome roster. |
| Frozen paper bundle | Same retained designs/seeds. Fits using AKI acquire new config hashes/run directories. Independent unaffected fits retain exact config bytes, hashes and run directories. |
| Worker/checkpoint locks | Existing worker uses the refit-only bundle and original neural-runs root. Old completion/non-estimability records cannot satisfy new identities. |
| Label caches | Corrected AKI has a new absolute path, changing its existing cache contract. Other targets retain original paths. No blanket cache deletion. |
| Tabular fits | AKI-only refits, same features/internal tuning, fresh output root. Other targets remain intact. |
| Prediction/analysis inventory | Combined bundle collects new affected fits and original independent fits. Explicit tabular links select old/new directories. New data-contract identity and analysis workspace. |
| Clinical thresholds and interpretation | Re-estimate model thresholds on the declared development population. Recompute explanations for corrected checkpoints. Native clinical scores and figure templates remain reusable. |

These are data/configuration identities, not source-file synchronisation checks.
The tools do not scan the repository for newer source files. They check actual
outcome bytes and frozen configuration identities at the handoff.

## 1. Regenerate episodes and compare the endpoint

The source owner must run the original AKI writer into a separate output
location with the corrected condition:

```r
datetime >= datetime[i] - days(365) & datetime < datetime[i] - days(7)
```

Keep the mean, other detection branches and episode rules unchanged for this
bug-only comparison. Do not crop histories at treatment index. The original
writer and corrected episode table are not available locally yet. The new
Python tools **do not regenerate AKI from creatinine**.

The preferred route reuses the original lab-coverage audit and survival
follow-up. First it reconstructs the original endpoint from original episodes.
Every eligible patient's event, index and censor dates must match the frozen
endpoint. It then applies the same export rules to corrected episodes. This
avoids regenerating other outcomes or using today's mutable registry extract.

Find `aki_first_documented__audit.parquet` in the original outcome-export
directory, generally not the approved release's `outcomes/` directory. The
original episode CSV was supplied through `OPERA_AKI_TABLE` and is tab-delimited.
Replace the three `/actual/...` paths below. Confirm the parent release against
the frozen bundle; the example is the documented original release path.

```bash
opera_python="$PWD/.venv-v100/bin/python"
opera_parent_release="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
opera_old_episodes="/actual/path/AKI_table_all_dalycare_simplified.csv"
opera_new_episodes="/actual/path/corrected/AKI_table_all_dalycare_simplified.csv"
opera_coverage_audit="/actual/original/export/aki_first_documented__audit.parquet"
opera_aki_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
opera_aki_audit="$PWD/data/results/aki_correction_audit_$opera_aki_stamp"

"$opera_python" -m opera.run.audit_aki_correction \
  --old-outcome "$opera_parent_release/outcomes/aki_first_documented.parquet" \
  --overall-survival "$opera_parent_release/outcomes/overall_survival.parquet" \
  --old-episodes "$opera_old_episodes" \
  --corrected-episodes "$opera_new_episodes" \
  --coverage-audit "$opera_coverage_audit" \
  --max-gap-days 180 \
  --output-dir "$opera_aki_audit" \
  --write-patient-differences
```

`180` is the current R lab engine's observation-gap rule. Exact reproduction
must succeed with the original policy. A missing last draw uses administrative
follow-up, matching R's `pmin(..., na.rm=TRUE)`.

Outputs: corrected `aki_first_documented.parquet`, `comparison_INTERNAL.json`,
`horizon_changes_INTERNAL.csv`, and optional `patient_differences_INTERNAL.parquet`.
Keep these on the secure server. Patient differences contain IDs/dates. Exact
counts and server paths in the reports also require disclosure review before
download.

The comparison reports training/tuning event additions/removals, shifted dates,
censor changes and horizon-status changes at 30/60/90/180/365/730 days, accounting
for competing deaths. These are diagnostic calendar-day states. Actual training
and evaluation still use the existing datamodule's full contract. No predictive
metric or held-out performance is calculated. Structural checks and old-export
reproduction cover the full export.

Alternatively supply `--new-outcome PATH` instead of the three episode/coverage
arguments if the co-author exports a corrected **final five-column endpoint**
through the original lab engine. Simply replacing event dates is insufficient
because losing an event can change censoring. If the original coverage audit is
missing, use the original-engine route rather than guessing coverage.

Review examples of additions, removals and shifted onsets clinically. Equal
event totals do not establish equal supervision. If all development inputs are
identical, inspect all training/selection dependencies before deciding no refit
is needed. The tooling conservatively prepares affected refits even then.

## 2. Publish a one-endpoint release amendment

After reviewing the comparison and intended definition:

```bash
opera_aki_release="$PWD/data/results/approved_outcome_release_v2_aki_$opera_aki_stamp"

"$opera_python" -m opera.run.build_aki_corrected_release \
  --parent-release "$opera_parent_release" \
  --corrected-outcome "$opera_aki_audit/aki_first_documented.parquet" \
  --comparison "$opera_aki_audit/comparison_INTERNAL.json" \
  --output-dir "$opera_aki_release" \
  --reason 'Corrected the reversed historical-baseline window in the AKI episode writer. Other detection branches and episode rules retained. OPERA uses first documented post-index episode onset, with original coverage and censoring.'
```

This writes `revision: aki_baseline_window_correction_v1`, parent manifest hash,
the 68 unchanged endpoint identities and a new structural audit. It remains a
candidate pending normal clinical review, not automatic clinical sign-off. The
old support report is not relabelled as new. Portable release-local config
templates are copied where present; the paper bundle is amended separately.
No need to repeat the original 87-to-69 selection, ALT/creatinine replacements
or composites. No current composite depends on this AKI endpoint.

## 3. Prepare selective refits and a carry-forward inventory

```bash
opera_aki_bundle="$PWD/data/paper_256_v3/bundles/aki_repair_$opera_aki_stamp"

"$opera_python" -m opera.run.prepare_aki_repair_bundle \
  --bundle-dir "$PWD/data/paper_256_v3/bundles/stage5" \
  --release-dir "$opera_aki_release" \
  --output-dir "$opera_aki_bundle" \
  --models S_L D_L O_L J_L R_AB_L RBH_AB_L N_A_L
```

`--models` is the reported model roster, not all historical screens. These
aliases cover the minimal local factorial/mechanism set. Add every other
retained control, national arm or second-seed alias before preparing the bundle.
Repeat `--bundle-dir` for designs in additional frozen bundles. Missing requested
aliases cause an error. All seeds represented by an alias are retained, but a
second-seed router may have a separate alias such as `R_AB_s43_L`. List it
explicitly if retained. The final reported roster still needs checking.

`dependency_inventory.csv` lists `refit`/`carry_forward`, old/new hashes, original
bundle and run directories. AKI S/D, O fits including AKI, J and retained routers
and controls get new identities. Independently initialized other-outcome S/D
keep exact old configs. Supervised-parent initialisation is rejected for a
separate dependency review. SSL checkpoints are reused.

`$opera_aki_bundle/refit_bundle` contains affected fits only, for GPU workers.
`$opera_aki_bundle` contains all retained fits, for prediction and inventory.
Do not combine original and corrected bundles in one worker invocation.

## 4. Existing worker and tabular commands

Run on each independent GPU allocation after normal environment setup:

```bash
"$opera_python" -m opera.run.paper_worker next \
  --bundle-dir "$opera_aki_bundle/refit_bundle" \
  --runs-dir "$PWD/data/paper_256_v3/neural_runs" \
  --wait
```

Claims coordinate as usual. Resume applies only to a corrected fit's own
interrupted run. The old-label checkpoint is under a different hash. Preserve
encoder audits/selection rules. An existing `HOLD_SPECIALISTS` file will keep S
fits parked until that deliberate hold is removed.

On a suitable CPU allocation, reuse the actual existing prepared-feature root:

```bash
opera_aki_tabular="$PWD/data/paper_256_v3/tabular_runs_aki_$opera_aki_stamp"

"$opera_python" -m opera.run.paper_tabular_queue \
  --bundle-dir "$opera_aki_bundle" \
  --runs-dir "$PWD/data/paper_256_v3/neural_runs" \
  --features-dir "$PWD/data/paper_256_v3/features" \
  --output-dir "$opera_aki_tabular" \
  --outcome aki_first_documented \
  --execute
```

Change `--features-dir` if the existing location differs. Parallel allocations
can use existing `--shard-count N --shard-index I`. Outcome selection happens
before sharding. Retained registry-enriched XGBoost also needs an AKI-only rerun
using its original feature root, `--feature-profile registry_augmented`,
`--per-cell-models xgboost`, and a separate fresh output root.

Create a single root for the existing analysis inventory:

```bash
"$opera_python" -m opera.run.prepare_aki_tabular_overlay \
  --bundle-dir "$opera_aki_bundle" \
  --old-tabular-root "$PWD/data/paper_256_v3/tabular_runs" \
  --corrected-tabular-root "$opera_aki_tabular" \
  --output-dir "$opera_aki_bundle/tabular_overlay"
```

These are directory links with untouched original provenance. AKI always points
to the new root; missing corrected fits remain missing. Repeat for registry-
enriched roots when retained.

## 5. Corrected tuning evaluation and sign-off

```bash
"$opera_python" -m opera.run.paper_prediction_queue \
  --bundle-dir "$opera_aki_bundle" \
  --runs-dir "$PWD/data/paper_256_v3/neural_runs" \
  --output-dir "$PWD/data/paper_256_v3/predictions" \
  --split tuning \
  --execute
```

Existing independent predictions retain their identities; new shared fits
produce new predictions for all their outcomes. Rebuild `paper_analysis_inventory`
with the combined bundle, tabular overlay, a new data-contract identity and fresh
workspace. Set **`--splits tuning` explicitly**. The inventory defaults to both
splits. Preserve the estimand, horizons, paired patient/cell rules and chosen
router mapping. Reusing fits does not mean reusing old pooled comparison tables.

The full-rehearsal wrapper discovers original stage bundles and workspaces.
Do not assume it will discover the correction automatically. Use the direct
queues and explicit combined inventory until its inputs are deliberately
configured for this release. Source-removal and attribution exports must use
the corrected checkpoint, not the original rehearsal workspace.

National raw-record pretraining continues. Prepare downstream national fits
against corrected labels. After corrected tuning audits, the final model roster,
clinical thresholds and analysis inputs are signed off, record the final release
and proceed to held-out evaluation under the fixed protocol.

The remaining server inputs are the original/corrected episode CSVs and frozen
coverage audit (or a corrected final endpoint). No patient-level AKI data are
available locally. Local tests establish integration, not the size of the
phenotype change or clinical validity of the corrected definition.

## Local validation

Eleven AKI repair tests pass, including old-export reproduction before export,
onset/censor changes, development-only comparisons, competing-death status,
unchanged registry/68 outcome files/display groups, the existing v2 release
verifier, existing prediction/tabular job readers, preserved independent fit
identities, new affected fit identities, supervised-resume rejection and no
fallback to old AKI tabular fits. A broader run of existing worker, workflow,
label-cache, tabular and release tests passed 68 checks before the final
resume/display-verifier additions; the expanded AKI suite then passed separately.
All four new command-line entry points load successfully. No server model fit,
actual phenotype export or held-out performance evaluation was run here.
