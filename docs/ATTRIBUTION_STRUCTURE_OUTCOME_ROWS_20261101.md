# Attribution structure: individual outcome rows

Implemented the agreed revision on 1 November 2026, retaining the existing
indigo/sienna palette, typography and clinical-domain order.

- A shows individual-outcome attribution agreement, with numbered outcome
  labels, family headings and matching column numbers. The heading now reads
  “Agreement in episode attributions”.
- B shows the actual individual-outcome input-removal AUC changes in precisely
  the same row positions. Its overall average is a separate strip above the
  grid, beneath the column headings. Individual cells are not overprinted with
  numbers.
- Family header rows and gaps line up across both panels. Colourbars have
  separate axes so they cannot shrink one panel and break row alignment.
- Unavailable estimates remain grey. An outcome present in either input table
  remains in both panels, with unavailable values indicated explicitly.
- Exact AUC effects and bootstrap intervals remain in the existing CSV tables.
  Duplicate outcome/restriction rows from multiple horizons are rejected rather
  than silently averaged. Family averages cannot substitute for endpoint data.

## Inputs and rendering

The renderer and its wrapper accept `outcome_restrictions`, wired into
`opera/configs/attribution_figures.example.yaml` as
`${OPERA_PAPER_INPUT_RESTRICTION_DIR}/input_restriction_auc.csv`.
The separate overall strip still reads `input_restriction_summary.csv`.
Both are already produced by `compare_input_restrictions`; no new inference or
model fitting is required for this layout change.

Review assembly and ZIP packaging now include the endpoint-level CSV, alongside
the summary. The manuscript figure caption describes the new layout and the
direction of the AUC changes. Other attribution figures retain their inputs.
The synthetic generator uses a separate random stream for the added endpoint
effects, preserving its existing summary and patient examples.

## Preview and verification

Open
`scratch/paper_reading_20261028/attribution_structure_outcomes_20261101/index.html`.
The main preview uses the historical rehearsal's 69 outcome names, with entirely
synthetic values. It is a layout proof, not updated empirical evidence.
A 95-outcome stress preview and `layout_audit.json` are alongside it.

The subsequent style pass uses the standard manuscript axes-title and export
helpers, aligning each panel heading with its full panel including row labels.
The width, text hierarchy and colourbar sizes follow the other attribution
figures. The overall summary now sits immediately under the column headings,
removing the detached summary band and reducing the blank space above A.
The standalone explanatory footer is left to the caption, consistent with
the other plates.

The panels also use distinct colour schemes for their different measurements.
A retains indigo/sienna for attribution correlation. After reviewing the
pooling, regional and clinical figures, B uses the existing terracotta/teal
ramp (`BANK_CMAP`), whose endpoint hues also appear in the pooling contrasts.
Terracotta is an AUC decrease, white is zero change and teal is an increase.
This replaces the rejected ochre/teal calibration palette. Its summary strip
shares B's exact scale. Missing values stay grey in both panels.

The 50 targeted tests covering the figure, review packaging, restriction
comparisons and attribution workflow passed. New checks protect row alignment,
individual effects, missing estimates and prevention of cross-horizon averaging.
`git diff --check` passed.

A separate broader attribution-pattern test exposed an unrelated manuscript
number-fixture mismatch: `test_every_attribution_number_in_the_manuscript_is_produced_and_formatted`
constructs its inputs from the current manuscript's cited keys, then requires
`ig_var_reference` even when that key is absent. This test was not changed as
part of the figure revision.
