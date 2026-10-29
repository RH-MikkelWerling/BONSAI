# Clinical figure refresh — 29 October 2026

## Confirmed cause

The user ran `scripts/rerun_tuning_figure_outputs.sh --render-only` against
`local_figures_20261028_143127`. That command refreshes the general figure
suite, not the clinical-score evaluation stage. The old clinical files remain
in that workspace. The supplied image still has “1 years”, three NCCN-IPI
categories and tuning-derived cutoffs. These differ from the current clinical
renderer and recipe.

The review assembler previously copied clinical images and consumed clinical
headline tables without checking render freshness. Rerunning general plots
or packaging could therefore circulate the old clinical comparison again.

## Changes

- Clinical rendering writes a versioned receipt containing renderer source
  hashes, evaluation/threshold splits, score definitions and output hashes.
- Assembly checks that receipt before consuming clinical headline tables or
  copying clinical images. An explicitly selected stale clinical directory
  fails before assembly starts. An implicit stale directory is excluded and
  recorded as a missing family.
- Review packaging rejects stale or unversioned clinical images, including
  old assembled reviews that bypass the new assembler.
- The general render launcher explicitly states that it does not refresh
  the clinical family.
- `refresh_clinical_figures` rebuilds recipes, native groups where numeric
  totals exist, evaluation and plots on CPU when training predictions have
  already been exported. It requires a fresh directory and refuses to run
  without that export.
- Clinical completion requires its render receipt as well as its headline
  and classification tables.

## Server action and inspection

Follow the clinical-only command in
`docs/AGENDA_20261029_ROUTER_AND_TEST_RELEASE.md`. Use a fresh sibling workspace
and pass its `clinical_scores` directory explicitly to review assembly. This
uses the fitted selected model for training-prediction export and evaluation.
It does not retrain models or rerun IG or source removal.

The new primary computed NCCN-IPI plot should show its four native categories,
training-derived cutoffs and “1 year”. Check the completion report and score
inventory before treating the clinical family as complete. Recorded registry
scores whose categories were already collapsed remain a separate limitation.
The receipt is a freshness/comparison-contract check, not an audit of all
clinical applicability or source data.

## Validation

31 targeted tests passed across provenance, review assembly, packaging,
clinical panels, recipe generation and completion. These include detection of
overwritten figures, extra legacy images, changed renderers, wrong selected
models, tuning-derived cutoffs and a three-group primary NCCN-IPI definition.
The CPU refresh was checked for refusal without training predictions and for
reconstruction of native categories from numeric totals while reusing an
existing training export. The server's actual new figures still need visual
inspection after the run.
