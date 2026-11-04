# Clinical index summary: style revision

Updated only `figure_clinical_index_summary` in `opera/visualization/paper_figures.py`.
The co-author's evaluation, ordering, counts, verdicts and uncertainty are unchanged.

The first revision did not sufficiently match the manuscript. A second visual
comparison against `fig_headline`, `fig_hierarchy` and `supp_strategy` identified
the actual mismatch: this forest inherited the per-index DejaVu font override,
while those figures use the shared sans-serif/Arial-family settings. This
renderer now follows the shared font family, `LABEL_SIZE` and `TICK_SIZE`;
its interval widths, opacity, round caps and diamond size match the existing
forest helper. Other clinical renderers are unchanged. The disease column uses
normal-weight type, with more room between it and the index labels. Explicit
two-line axis labels retain readable type in all four columns.

- Lighter, normal-weight group headings with thin rules, replacing the large
  bold header band. Methodological subtitles and the both-ends count remain
  in `opera_caption_notes` rather than competing with the plot headings.
- Slimmer count strips, restrained diamonds/intervals and lighter separators,
  with quiet vertical guides and consistent clinical text sizes.
- Restored “OPERA better →” above each forest at the author's request. Fully
  unavailable rows have one continuous grey band masking the grid and zero
  lines beneath a left-aligned explanation. Individual unavailable cells use
  a centred label with a white backing so no reference line crosses the text.
- Compact two-column legend below the forests, with sufficient space for the
  axis labels and synthetic-proof mark.
- Retained every clinical index, pooled population, unavailable row and reason.

Preview: `scratch/paper_reading_20261028/clinical_summary_style_20261103/index.html`.
The original is available in the expandable comparison beneath the revised
figure. Both use the original synthetic aggregate inputs, not test results.

Validation: all eight existing clinical-index-summary tests passed, including
the derivation, table/macros, recipe and figure checks. All 14 preview rows and
all four effects and interval bounds match the saved original summary. Visual
inspection and the layout audit found no text-collision candidates or legend
overflow. `git diff --check` passed. Real-data labels still need inspection when
those outputs are available.
