# Overnight tuning rebuild

After syncing the current source to the server, run from the repository root on
an allocation with enough remaining CPU walltime:

```sh
mkdir -p logs/rehearsal
nohup .venv-v100/bin/python -u -m opera.run.run_tuning_plot_rebuild \
  --env-file .env.rehearsal \
  > logs/rehearsal/overnight_plot_rebuild.log 2>&1 < /dev/null &
```

Follow progress with `tail -f logs/rehearsal/overnight_plot_rebuild.log`.
`nohup` protects against terminal disconnects, not allocation expiry.

The runner prints its fresh `overnight_tuning_YYYYMMDD_HHMMSS` directory beside
the existing analysis root. It preserves the existing analysis, refreshes the
tuning inventory and analysis, and uses at least five cases and five controls
with a frequentist multiplier bootstrap. It attempts clinical scores, embedding
and plot tables, figure validation, rendering, paper numbers and supplement tables.
It launches no training, inference or GPU attribution. Cached aggregate
source-removal/attribution inputs are copied and explicitly recorded as reused.
Missing inputs or a failed stage do not stop independent later stages.

In the morning inspect `overnight_run.json`, `results_validation.json`,
`attribution_render_report.json`, `figures/`, and individual `logs/` files inside
that directory. A successful available-only renderer does not certify that all
required figures were present; inspect the per-figure statuses and warnings.
The runner reports `finished_with_gaps` when any stage failed or was blocked.

To resume after interruption, substitute the directory printed in the log:

```sh
.venv-v100/bin/python -u -m opera.run.run_tuning_plot_rebuild \
  --env-file .env.rehearsal --output-dir /absolute/printed/overnight_directory --resume
```

Completed stages are retained; failed stages are retried. Changed runner/source
or base environment requires a fresh run. There is no reliable runtime estimate
yet: inventory filesystem scans and the new bootstrap dominate the CPU work.
