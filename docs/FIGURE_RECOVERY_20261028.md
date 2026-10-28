# Figure recovery, 28 October 2026

The reported analysis failed in calendar-year subgroup labelling before writing
its outputs. Most missing figures are downstream consequences, not independent
plot failures. The failed analysis did not save paired bootstrap results; those
must be recomputed. Existing predictions, inventory, clinical-score results and
embedding inputs can be reused.

Changes:
- Calendar-year strata accept valid integer years, including registries spanning
  more than 20 years; arbitrary high-cardinality covariates still require bins.
- Subgroup specifications are checked before bootstrap computation.
- Stratified reanalysis labels only patients in the relevant prediction roster.
- Embedding exports record the configured highlight order. The patient map uses
  it, with the selected five clinical endpoints as a fallback for older tables.
- Disease-map KDE contours are opt-in; default points show actual overlap.
- Calibration titles wrap within columns, with room for the legend and row labels.
- Missing router vectors are labelled as unavailable, never as a shared readout.

After syncing the updated source, from the repository root on the server:

```bash
mkdir -p logs
nohup bash scripts/rerun_tuning_figure_outputs.sh \
  data/paper_256_v3/analysis/plot_rebuild_20261028_080223 \
  > logs/figure_recovery_20261028.log 2>&1 &
tail -f logs/figure_recovery_20261028.log
```

Use the actual run directory if the timestamp differs. This script uses that
run's overnight.env, keeps its inventory, archives the previous analysis and
figure directories, and runs analysis, embedding tables, plot tables, readiness,
figures, paper numbers and supplement tables. It stops on subprocess errors.
Rendering remains available-only: inspect figures/figure_manifest.json and the
recovery logs for skipped/failed plates. Completion does not certify that all
required figures exist. Do not run concurrently with another writer to this root.

Missing routed vectors require genuine per-target exports before an arrow row
can be produced. These changes do not add fabricated arrows, change predictions,
or establish whether every unseen figure is correct. Old cached source-removal
results are not recomputed by this recovery script.

## Continue after successful analysis stopped at readiness

Readiness is advisory for available-only previews. The recovery script now continues
to rendering after reporting gaps. Preserve the completed analysis and plot tables:

```bash
nohup bash scripts/rerun_tuning_figure_outputs.sh \
  data/paper_256_v3/analysis/plot_rebuild_20261028_080223 --render-only \
  > logs/figure_render_continue_20261028.log 2>&1 &
```

This skips analysis, embedding export and plot-table generation. It archives old
figure renders, renders available plates, and updates paper numbers and tables.
Missing external origin estimates, clinical plates and synthetic illustrative
patient timelines remain separately reported gaps.
