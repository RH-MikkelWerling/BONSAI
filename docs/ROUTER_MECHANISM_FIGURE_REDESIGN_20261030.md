# Router mechanism figure redesign — 30 October 2026

The original three-column list of 14 intervention identifiers was replaced
with a question-led figure. Its estimates, intervals, weighting and scientific
outputs are unchanged.

## Main figure

- **A: Where does the gain over joint appear?** Three paths through the same
  frozen OPERA checkpoint are compared with the independently trained joint
  model. The full-OPERA row is highlighted. These are not a causal waterfall
  or the separately trained own/global controls.
- **B: Which corrections help?** Global, own-task, other-disease,
  other-outcome and all-borrowed terms occupy five readable rows. Filled
  diamonds show mean replacement, with open circles for zero replacement.
- **C: Does individual routing help?** Two contrasts separate fixing every
  gate from fixing borrowed-source proportions while retaining borrowing
  amount. The adapter functions still depend on each patient.

B and C share the same x-axis limits within a metric, so a small gate effect
cannot appear as large as a correction effect merely through magnification.
Machine-roundoff-only differences do not generate an axis at 1e-17. Raw CSV
values are retained. Negative effects and all saved finite intervals remain
visible without truncation.

The main view uses AUC. A matching companion uses IPA. A separate sensitivity
figure retains all 14 interventions, grouping paired zero/mean replacements
into nine rows and showing AUC, Brier and IPA. The HTML opens with the scientific
questions and plots, with numerical tables and definitions in expandable
sections. Type sizes and palette come from the existing OPERA style system.

## Preview and verification

`scratch/paper_reading_20261028/router_mechanism_redesign_20261030/index.html`
contains the preview, visibly marked **synthetic layout preview, not study
results**. Main AUC and sensitivity panels were inspected graphically. Text,
legends, intervals, tick labels and footnotes have room at the exported size.

The audit tests passed **22 tests**, including rerendering from aggregates
while prohibiting model loading and metric computation. Tests verify that
scientific CSVs and `report.json` remain byte-for-byte unchanged after rerendering,
and that missing joint comparisons are labelled rather than invented.
Source lint and `git diff --check` passed.

The existing audit command generates this layout automatically after syncing
the renderer. For a completed/running audit using the previous renderer, let it
finish and use the [render-only command](ROUTER_MECHANISM_SERVER_AUDIT_20261030.md).
That refreshes figures and ZIP without repeating the analysis.
