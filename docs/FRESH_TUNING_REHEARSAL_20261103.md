# Fresh figure rehearsal — 3 November 2026

Purpose: one complete figure/number build from the frozen inputs, followed by
review and closure. It is not another model-selection round. No held-out data
are read by this tuning-only entry point.

## Freshness boundary

Retain trained checkpoints, registered base prediction exports, metadata,
outcome definitions and approved configuration. Regenerate paired metrics and
bootstrap summaries, clinical training exports/evaluation/KM matching, borrowing
inference and mean/zero analyses, outcome correction vectors, integrated
gradients, input restrictions, whole-cohort and cross-family metrics, plot
tables, task-organisation/correction-arrow exports, figures and paper numbers.

This does not test a fresh forward pass for every specialist/XGBoost base
prediction: those are explicit fixed inputs. It does test the downstream build
without previous analysis/figure caches. Fresh explanatory inference uses new
output directories. No previous result images or explanatory aggregates are
imported from the source workspace. Transferring outputs freshly computed
within this same run into the review folder is permitted and recorded.

## Changes made to the existing full rehearsal

- Added task-structure and correction-arrows generation before final assembly.
- Corrected primary-versus-router fit roles: clinical/gaps use selected OPERA;
  borrowing uses the registered routed comparator and its own-only control.
- Patient vector exports now support the selected outcome-attention model,
  pairing its own shared summary with its own final outcome representation;
  secondary-router vectors are not projected onto a different model's map.
- Explicit --clinical-config can supply the approved active recipe rather than
  silently regenerating a shared-OS clinical recipe. The selected model is
  retargeted without replacing endpoint mappings.
- Whole-cohort results are now explicitly passed into review assembly.
- Export figure_checklist.csv with every renderer figure, including missing
  inputs and rendering failures. full_rehearsal.json records step status.
- Incomplete runs retain their gallery/ZIP and exit nonzero rather than looking
  successful. --defer-national allows only missing-input skips for fig_external,
  supp_national_where and supp_embedding_comparison; other missing figures and
  rendering failures still block. Result is complete_local_national_pending,
  not complete. Deferred figures and reasons remain in the checklist.
- The clinical KM figures/tables, Holm table and current figure recipes are
  included through the updated renderers/review packaging.

## Server launch (updated 4 November)

Run from the repository root on a GPU allocation after syncing all changed
and new files. No existing completed analysis folder or manual source path is
needed: the launcher reads the configured server .env and creates a fresh
input inventory, a fresh tabular-conversion cache and a sibling build folder.
The stage-16 bundle must contain the current completed local fits and their
registered base prediction exports. Set STAGE differently only if your current
full bundle has another stage number.

```bash
export OPERA_PYTHON="$PWD/.venv-v100/bin/python"
export OPERA_ENV_FILE="$PWD/.env"
export STAGE=16
export OPERA_MODEL=J_OA_OWN_L

opera_rehearsal="data/paper_256_v3/analysis/fresh_tuning_$(date +%Y%m%d_%H%M%S)"
opera_log="logs/$(basename "$opera_rehearsal").log"
mkdir -p logs
nohup bash scripts/run_fresh_tuning_rehearsal.sh "$opera_rehearsal" \
  > "$opera_log" 2>&1 &
opera_pid=$!
printf 'PID: %s\nRun: %s\nLog: %s\n' "$opera_pid" "$opera_rehearsal" "$opera_log"
tail -f "$opera_log"
```

The configured OPERA_PAPER_CLINICAL_SCORE_CONFIG is carried into the new run
and retargeted to the selected model without changing endpoints. Do not replace
it with the example's unresolved null mappings. If no active config is set,
the older generated shared-survival recipe remains a rehearsal fallback, not
proof of correct native-endpoint matching; audit the per-index roster.

No source-manifest verification is enabled. Existing checkpoints and base
prediction exports remain explicit fixed inputs, not newly scored models.
No fitting and no held-out prediction reading is launched. Large inference and
bootstrap jobs may be quiet between reported progress updates.

In another terminal, or after stopping only the tail command, inspect:

```bash
cat "$opera_rehearsal/build/full_rehearsal.json"
```

Once complete, review:
- `$opera_rehearsal/build/figure_checklist.csv`
- `$opera_rehearsal/build/review/`
- `$opera_rehearsal/build/paper_numbers/`
- `$opera_rehearsal/build/paper_review.zip`

`complete` or `complete_local_national_pending` is the successful pipeline status.
`finished_with_gaps` retains available figures and the ZIP but exits nonzero;
inspect failed/blocked steps and the figure checklist. The inventory step occurs
before full_rehearsal.json is created. Do not start another full-figure job at
the same time: completion uses a shared lock in the configured paper root.

## Review and decision

Inspect full_rehearsal.json, figure_checklist.csv, the review folder and
paper_numbers/results_macros.tex. Audit counts, paired coverage, endpoint and
proxy labels, both KM extremes and their valid bootstrap fractions, primary
versus matched-encoder contrasts, layouts and missing-output reasons.
Unsupported clinical matches remain reported, not silently dropped or counted
as model failures. Any decision to change support or matching rules happens on
tuning and is documented before held-out evaluation.

Sign off the available local scope if the only deferred work is the named
national arm. Use the same scientific definitions, core evaluators and figure
recipes on held-out data, with its new prediction exports. This rehearsal
entry point itself remains tuning-only; it is not a test-set launcher.

Validation: 60 local orchestration, figure-completion, attribution-refresh,
assembly and packaging tests passed before the additional patient-vector
compatibility check. Server-scale inference and the full figure review are
still performed by the rehearsal itself.

Final expanded checks: 95 tests passed across patient-vector export, stage-9
model behavior, fresh orchestration, completion, attribution setup, assembly
and packaging. `git diff --check` passed. No held-out data were read.


4 November launch verification: 88 targeted tests passed across the new
inventory-first launcher, full orchestration, primary/router fit selection,
attribution setup, patient vectors, failure handling, Holm and paper workflow.
A focused regression also verifies the fresh conversion-cache argument. Shell
syntax and git diff whitespace checks passed. Server data remain uninspected.
