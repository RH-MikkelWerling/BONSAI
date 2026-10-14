# Clean versus loose routers and preliminary CLL specialists

All commands below use **tuning** patients and each run's selected `best.ckpt`.
Do not use held-out outcomes while choosing the adapter configuration.

## 1. Paired predictive comparison

On a GPU allocation, after syncing the current code:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
bash scripts/compare_clean_loose_router_tuning.sh
```

The script exports six-horizon predictions for both checkpoints, reusing only
exports whose manifest matches the checkpoint and tuning split. It then makes
an exact patient/outcome/label-matched, paired-bootstrap 365-day CIF-AUC
comparison. Review
`data/results/router_clean_loose_tuning_20261011_INTERNAL/paired_comparison/tuning_forest_contrasts.csv`
and the manifest's unsupported-outcome inventory. The run-summary AUC gain is
not a paired patient-level interval.

If an export directory is partial, inspect it and choose a new
`OPERA_ROUTER_PAIR_DIR`; the script does not delete or overwrite it.

## 2. Borrowing routes from the loose checkpoint

For the *same four prespecified cells* as the clean figure:

```bash
bash scripts/make_loose07_router_borrowing_figure.sh
```

For all fine disease x outcome cells, optionally run:

```bash
bash scripts/make_loose07_router_full_atlas.sh
```

The first command writes the selected-cell PNG/PDF and per-patient tuning route
exports.

The updated allocation figure can reuse the existing route exports without
rerunning model inference. It shows absolute source gate weights and within-bank
enrichment, rather than basis-point contrasts against the displayed rows:

```bash
OPERA_ROUTER_FIGURE_DIR="$PWD/data/results/router_scale03_cell_allocation_20261011_v2_INTERNAL" \
  bash scripts/make_scale03_router_borrowing_figure.sh
OPERA_ROUTER_FIGURE_DIR="$PWD/data/results/router_loose07_cell_allocation_20261011_v2_INTERNAL" \
  bash scripts/make_loose07_router_borrowing_figure.sh
```

The scripts refuse to overwrite an existing figure directory. The full
per-source means and within-bank enrichment are saved to
`all_source_weights_by_cell.csv`; cross-cell preference contrasts remain in a
separate diagnostic CSV. A gate weight is not an estimated predictive benefit.

For the patient-level selectivity revision, reuse the same route exports again
with a new output directory:

```bash
OPERA_ROUTER_FIGURE_DIR="$PWD/data/results/router_scale03_cell_allocation_20261011_v3_INTERNAL" \
  bash scripts/make_scale03_router_borrowing_figure.sh
OPERA_ROUTER_FIGURE_DIR="$PWD/data/results/router_loose07_cell_allocation_20261011_v3_INTERNAL" \
  bash scripts/make_loose07_router_borrowing_figure.sh
```

In the v3 diagnostic figure, whiskers are *patient 10th–90th percentiles*, not uncertainty in the
mean. `route_selectivity_by_cell.csv` distinguishes an almost-uniform route for
each patient (`median_patient_peak_enrichment` near 1) from heterogeneous
patient-specific routes that merely average out (`mean_route_peak_enrichment`
near 1 but patient peak substantially above 1). It also reports the median
effective fraction of sources, where 1 is a uniform within-bank allocation.

The publication-style revision moves these technical diagnostics out of the
artwork. It displays one dot per named source (mean gate weight) and a faint
even-split reference within each off-home bank. The outcome names come from
`opera/visualization/outcome_names.py`; these are presentation labels only.
Replot from the same exports in fresh directories:

```bash
OPERA_ROUTER_FIGURE_DIR="$PWD/data/results/router_scale03_cell_allocation_20261011_v4_INTERNAL" \
  bash scripts/make_scale03_router_borrowing_figure.sh
OPERA_ROUTER_FIGURE_DIR="$PWD/data/results/router_loose07_cell_allocation_20261011_v4_INTERNAL" \
  bash scripts/make_loose07_router_borrowing_figure.sh
```

### Patient-level distribution diagnostic

The paper figure summarizes mean source use. To check whether similar means
conceal very different patient routes, reuse the already-exported tuning routes
without model inference:

```bash
bash scripts/make_router_weight_distribution_diagnostics.sh
```

This defaults to the loose 0.7 checkpoint and the four prespecified cells.
Each cell receives separate disease-source and outcome-source PNG/PDF pages
showing **every patient at every source** (faint dots), median (diamond), and
patient 10th–90th percentiles (line). The dotted rule is an equal share of that
cell's total disease or outcome bank weight; own sources are shown first.
`source_weight_distributions.csv` retains all means, quantiles, ranges and
standard deviations. This is an internal diagnostic, not the main paper plot;
source weight does not measure predictive benefit. To run the same check for
the clean checkpoint, set all three paths explicitly:

```bash
OPERA_ROUTER_RUN="$PWD/data/models/approved_neural_router_clean_20261007_v1/time2vec_router_interacting_scale03_e2e" \
OPERA_ROUTER_EXPORT_ROOT="$PWD/data/results/router_scale03_cells_tuning_20261010_INTERNAL" \
OPERA_ROUTER_DISTRIBUTION_DIR="$PWD/data/results/router_scale03_patient_weight_distributions_20261011_INTERNAL" \
  bash scripts/make_router_weight_distribution_diagnostics.sh
```

Compare the two checkpoints on identical patients using:

```bash
CLEAN="$PWD/data/models/approved_neural_router_clean_20261007_v1/time2vec_router_interacting_scale03_e2e"
LOOSE="$PWD/data/models/approved_neural_router_loose_20261010_v1/time2vec_router_interacting_loose_scale07_e2e"
python -m opera.run.compare_router_cell_routes \
  --clean-run "$CLEAN" --loose-run "$LOOSE" \
  --clean-export-root "$PWD/data/results/router_scale03_cells_tuning_20261010_INTERNAL" \
  --loose-export-root "$PWD/data/results/router_loose07_cells_tuning_20261011_INTERNAL" \
  --cell CLL:overall_survival --cell CLL:sepsis \
  --cell DLBCL:overall_survival --cell MM:aki_first_documented \
  --output-dir "$PWD/data/results/router_clean_loose_paired_routes_20261011_INTERNAL"
```

`paired_route_similarity.csv` reports paired-patient total variation, mean
route variation, and within-cell patient variation. Zero total variation means
identical gate distributions. `paired_source_weight_changes.csv` retains every
named source. A larger adapter residual or weaker support shrinkage can change
prediction effects *without* changing gate weights; these route outputs alone
do not quantify predictive contribution.

## 3. Preliminary independent CLL specialists

The specialist queue's `v2` runs must be complete. Start with three clinical
outcomes, or set `OPERA_CLL_OUTCOMES` to a space-separated list of completed
CLL specialist outcomes:

```bash
bash scripts/compare_cll_specialists_to_routers.sh
```

The script evaluates specialist, clean router, and loose router on the same
fine CLL cohort and outcome. It requires exact patient and event-label identity
before comparison. For each of six horizons it reports paired CIF-AUC estimates
and contrasts if at least 20 cases and controls exist; otherwise it explicitly
marks the horizon unsupported. Inspect `comparison/cll_specialist_contrasts.csv`
and `comparison/cll_specialist_estimates.csv` in each outcome directory under
`data/results/cll_specialist_router_comparison_20261011_INTERNAL`.

The specialists were independently fine-tuned from DAPT and selected by
single-outcome tuning loss. Pooled routers used a different prespecified
endpoint-group AUC selection rule. This is a preliminary *comparison of
training strategies*, not a capacity- or checkpoint-selection-matched router
ablation. Do not claim a win for cells where the specialist AUC is undefined.

## Candidate end configuration (not yet chosen)

Event-support shrinkage is a sensible default protection against noisy rare
sources; removing it was an exploratory stress test, not a recommendation to
permanently discard it. If paired results support the high-capacity run without
calibration or negative-transfer concerns, the first guarded finalist to test
would be: residual scale `0.7`, own-source floor `0.05`, event tau `10`, and
shared-logit initialization `0`. This retains source shrinkage and enough home
gradient while allowing more flexible routing than the original `0.3/0.1/20/3`
configuration. It is a *proposal*, not an empirically established optimum.
The current loose `0.7/0.02/0/0` remains a separate exploratory arm.

The guarded candidate is now runnable, with adapter rank held at 8:

```bash
bash scripts/run_router_guarded_scale07.sh preflight
bash scripts/run_router_guarded_scale07.sh launch
# Only if interrupted and last.ckpt exists:
bash scripts/run_router_guarded_scale07.sh resume
```

It writes to `data/models/approved_neural_router_guarded_20261011_v1/`
by default and does not submit a scheduler job. The guarded run differs from
the clean `0.3/0.1/20/3` arm in four router settings; it is a proposed
finalist, not a one-factor causal ablation. Rank 16 should be tested only
after this rank-8 run, changing rank alone while holding the guarded settings
fixed. Compare tuning predictions, individual cells, calibration and source
routes before deciding which configuration merits held-out evaluation.

If a second GPU is available, the rank-16 capacity check can train in parallel
with guarded rank 8. It changes only `adapter_bottleneck` in the configuration:

```bash
bash scripts/run_router_guarded_scale07_rank16.sh preflight
bash scripts/run_router_guarded_scale07_rank16.sh launch
```

The rank-16 run uses its own root directory and does not submit a scheduler
job. The same seed does not imply identical initialization of all downstream
parameters when adapter shapes change; treat one-seed differences as
exploratory and repeat promising rank comparisons across seeds before a strong
capacity claim. Keep held-out evaluation closed until the finalist is selected.
