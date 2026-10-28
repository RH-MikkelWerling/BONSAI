# Router domain follow-up

Remaining median/IQR summaries of absolute predictability and within-tile outcome
spread are descriptive. The main pooling disease-level comparison is now a mean;
its heatmap retains a labelled within-domain median. Neither median/IQR nor
across-cell IQR should be called a confidence interval. Missing training support
labels now say unavailable rather than nan; missing support is not zero patients.
The strategy preview panel headings are wrapped to avoid overlap.

The horizons router panel uses the saved R1 (R_L minus J_L) domain means and
patient-bootstrap intervals. A regression with deliberately positive disease
course and negative cardiovascular effects confirms that their markers retain
the correct row labels. This does not establish the numerical correctness of
server prediction inputs or reconcile earlier runs with different model pairs.

Run the targeted aggregate-cell diagnostic after syncing:

```bash
plot_root=data/paper_256_v3/analysis/plot_rebuild_20261028_080223
.venv-v100/bin/python -m opera.run.diagnose_router_domains \
  --analysis-root "$plot_root" --output "$plot_root/router_domain_diagnostic.json"
```

It reports mean, median, equal-disease mean, support, and five lowest/highest cells
per domain and horizon, and compares per-horizon exported means with the raw
analysis summaries and cell means. No patient records, inference or bootstrap
rerun is needed. A false source/export match flag is a concrete inconsistency to
investigate. True matches show the plot reflects those source gains, not that the
model selection/prediction source is necessarily the intended historical fit.

The h=0 plot data pool horizons within disease-outcome cells, while the raw
all-horizon source summary weights disease-outcome-horizon cells. They are
explicitly not checked for equality. The diagnostic does not count OPERA-J and
R1 twice when they are aliases. Cardiovascular domains contain few distinct
endpoints; support and influential cell effects must be inspected before deciding
whether broad intervals indicate a bug or limited information.
