# Follow-up to returned figure screenshots

Confirmed issues and implemented corrections:

- Headline history panel used `history`, while actual patient tables use
  `history_event_count`; it also expected display contrast codes instead of P5/T2.
- Coverage inventory counted early-censored IPCW rows as controls. Performance
  eligibility already correctly used positive-IPCW rows. This changes coverage
  diagnostics, not evidence that predictions or fitted models were wrong.
  Plot-table rebuilding repairs legacy inventories using saved estimate counts.
- Headline and horizon figures were using descriptive medians despite the agreed
  paired-mean headline estimand. The plot-table builder now carries saved panel
  and domain means/95% intervals into paired_cells.csv. These two figures use
  them; older input tables retain the explicitly labelled descriptive fallback.
  Other median/IQR panels remain descriptions of cell distributions and must not
  be described as paired-mean confidence intervals.
- Horizons now have independent horizontal scales so the router contrast is not
  visually compressed by joint/specialist differences.
- The clinical headline removes a disease-specific duplicate only when the same
  index has a pooled comparison, and gives probability-one markers space. This
  does not certify which scores exist in server exports or their denominators.
- Empty subgroup AUC panels explicitly state no supported estimates; the audit
  inspects source stratified AUC files to distinguish missing conversion from
  missing numerical estimates. The source files are required for diagnosis.
- Gallery defaults to final-model patient risk points (fitted router included in
  probabilities), with shared [0,1] scale. Observed IPCW surfaces remain available
  via mode: observed_surface. A shared embedding remains a pre-readout location;
  point colours and coordinates answer different questions.
- Strategy preview unavailable text shortened to stay inside its panel.

Do not drop specialists or sparse cells because their results are unfavourable.
Report model/prediction coverage separately from evaluation support. Keep >=5
cases and controls as the declared main AUC-comparison roster; it is not a law
that AUC is undefined below five (only a single-class AUC is undefined). AUC for
1-4 cases can be reported separately as exploratory and unstable; never replace
undefined AUC by zero. Brier can be defined without both classes, so broader
prediction-accuracy coverage is a separate useful estimand, not an interchangeable
replacement of the supported AUC panel. IPA has a degenerate null denominator
in single-class populations. Eligibility based on observed evaluation outcomes
restricts the estimand toward common/evaluable tasks; equal/disease balancing
within that roster does not undo selection into it.

Router improvements in an aggregate tuning selection metric do not imply
improvement at every horizon/domain, or the same ordering under another
weighting. Inspect saved paired means, exact checkpoint identity, and matching
cell rosters before interpreting. We cannot establish the server result from
screenshots alone. Old disease contours, old endpoint highlights and overlapping
calibration titles in these screenshots do not match current defaults. Source
and input hashes in figure_manifest.json distinguish rerenders from old PDFs.

Borrowing figures still use the source_removal tables copied by the overnight
runner; refreshing paired analysis does not recompute those ablations. Gate
summaries, router-versus-joint contrasts and source-removal effects are distinct
analyses; do not label all three as stale without checking their provenance.

## Server sequence after syncing

First record the current files (read-only audit, no bootstrap or model work):

```bash
plot_root=data/paper_256_v3/analysis/plot_rebuild_20261028_080223
.venv-v100/bin/python -m opera.run.audit_figure_results \
  --analysis-root "$plot_root" --output "$plot_root/figure_results_audit.json"
```

Inspect/share that aggregate JSON to resolve missing subgroup AUC and score
inventory. Rebuild only plotting tables to add the saved mean/CI columns and
repair coverage diagnostics, then render:

```bash
.venv-v100/bin/python -m opera.run.paper_experiments plot-tables \
  --env-file "$plot_root/overnight.env" --execute
bash scripts/rerun_tuning_figure_outputs.sh "$plot_root" --render-only
```

This does not rerun analysis/bootstrap. It does not refresh cached borrowing,
create missing clinical scores or export unavailable router vectors. Those
remaining items require their own provenance/input diagnosis.
