# Scale-0.3 router borrowing atlas: all tasks and source banks

Use the **completed clean Time2Vec interacting scale-0.3 checkpoint** and its
saved resolved `hydra/config.yaml`. Read only tuning route exports. Do not use
held-out outcomes to choose figure cells, sources, or display parameters.

## Primary full-map export

The primary atlas covers every configured fine disease × outcome task, not a
hand-picked subset. It streams tuning patients through the checkpoint and
accumulates counts, source-weight sums and squared sums directly; it does not
write millions of patient-target route rows. Each cell includes patients with
an eligible label for that target outcome. For task `(d,o)` and source `s`,
the estimand is the unweighted patient mean route gate
`mean_i w(i,o,s) | fine_disease(i)=d, eligible(i,o)`. Cells with fewer than 20
eligible tuning patients are white in the display but remain in the cube/count
inventory. This threshold concerns display stability, not AUC estimability.

Two aligned heatmaps show off-home disease-bank mass and off-home outcome-bank
mass for *all* task cells, on a common color scale. A second page shows
directed disease-to-disease and outcome-family-to-family source preference,
averaging task cells equally (the family view is per individual source
adapter, avoiding family-size inflation). A compressed array retains the
complete fine-disease × target-outcome × named-source means and counts, so any
task can be drilled into later. No static 2-D panel can legibly show all four
indices simultaneously; the paired pages and cube preserve that information.

On a spare GPU allocation:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
bash scripts/make_scale03_router_full_atlas.sh
```

Defaults: `OPERA_ROUTER_RUN` is the clean scale-0.3 run directory and
`OPERA_ROUTER_ATLAS_DIR` is
`data/results/router_scale03_full_borrowing_atlas_20261010_INTERNAL`.
The script checks its checkpoint, saved config, metadata and outcome-release
paths, refuses to overwrite an output directory, and exports only train-derived
normalization grids plus **tuning** patients; held-out labels stay closed.
Use the PDF for full-page inspection. The PNG is a preview; outcome labels
will be small at ordinary screen scale because the entire panel is present.

## Optional clinical zoom figure

For a prespecified fine disease `d` and target outcome `o`, each eligible
tuning patient contributes one route vector. The five-part budget is the
patient-average gate mass assigned to shared path, own disease adapter, other
disease adapters, own outcome adapter, and other outcome adapters. A named
off-home source bar is the patient-average weight for that adapter. This is a
conditional *router preference* among observed tuning patients, not a
prediction effect, source-data influence estimate, or proof of clinical
information transfer. The patient bootstrap interval is conditional on one
fitted checkpoint; it does not cover seed/checkpoint uncertainty.

The first panel gives the full weight budget, so a bright source does not hide
the shared path. The two adjacent panels show the top four off-home disease
and outcome adapters for that exact disease × outcome cell. Home adapters are
separate because they are partially protected by a configured floor. All
off-home source weights, not just plotted leaders, are exported to CSV. Labels are
"where the router places weight", never "how much this source improves AUC".

Select three or four cells **before examining this figure**: one common target,
one clinically plausible cross-disease borrowing target, one cross-outcome
target, and one scarce CLL target if it has enough eligible tuning patients.
Do not choose only cells with the most visually striking route weights. These
examples are explanatory, while the complete matrices remain in supplementary
material. The figure rejects cells with fewer than 20 tuning patients by
default; that is a display privacy/stability threshold, not an AUC threshold.

## Generate the first version

The optional zoom option exports only four clinically prespecified tuning cells
and renders them. Run on a spare GPU allocation, not concurrently with the CLL
training queue on the same GPU. It reuses completed cell exports, refuses
partial export directories and existing figure directories, and checks export
checkpoint provenance before plotting:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
bash scripts/make_scale03_router_borrowing_figure.sh
```

Optional environment variables are `OPERA_ROUTER_RUN`,
`OPERA_ROUTER_EXPORT_ROOT`, `OPERA_ROUTER_FIGURE_DIR`, `OPERA_LOCAL_GPU`, and
`OPERA_ROUTER_CELLS` (space-separated `FINE_DISEASE:outcome` strings). The
defaults are CLL–overall survival, CLL–sepsis, DLBCL–overall survival and
MM–first documented acute kidney injury. These are fixed clinical examples,
not sources selected by their gate weights. The source bars within each cell
are exploratory leaders from tuning data.

The following older manual patient-row export is not needed for the aggregate
full atlas. Use it only if patient-level route diagnostics are required.

If a tuning `source_route_weights.parquet` already exists for this exact
checkpoint, reuse it. Check the matching run's `best.ckpt` and saved config
paths, then use the command below with clinically chosen `--cell` arguments:

```bash
find data/results -name source_route_weights.parquet -print
```

Inspect each candidate's sibling `direct_cr_summary.json`: it must name the
scale-0.3 `best.ckpt` and `split: tuning`. If none exists, export routes on a
free GPU (the 365-day horizon is sufficient because gate weights do not vary
with prediction horizon):

```bash
source setup_v100.sh
RUN="$PWD/data/models/approved_neural_router_clean_20261007_v1/time2vec_router_interacting_scale03_e2e"
export BONSAI_COHORT_MEMBERSHIP="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet"
export BONSAI_OUTCOMES_DIR="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z/outcomes"
EVAL="$PWD/data/results/router_scale03_routes_tuning_20261010_INTERNAL"
CUDA_VISIBLE_DEVICES=0 python -m opera.run.evaluate_direct_competing_risk \
  --checkpoint "$RUN/best.ckpt" --config-dir "$RUN/hydra" \
  --config-name config --split tuning --horizons 365 \
  --device cuda --export-routes --output-dir "$EVAL"
```

Then render the preference figure:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
RUN="$PWD/data/models/approved_neural_router_clean_20261007_v1/time2vec_router_interacting_scale03_e2e"
ROUTES="$PWD/data/results/router_scale03_routes_tuning_20261010_INTERNAL/source_route_weights.parquet"
OUT="$PWD/data/results/router_scale03_cell_borrowing_20261010_INTERNAL"
test -s "$RUN/best.ckpt" || exit 1
test -s "$RUN/hydra/config.yaml" || exit 1
test -s "$ROUTES" || exit 1

python -m opera.run.plot_router_cell_borrowing \
  --routes "$ROUTES" --config "$RUN/hydra/config.yaml" \
  --checkpoint "$RUN/best.ckpt" \
  --cell CLL:sepsis --cell CLL:aki_first_documented \
  --output-dir "$OUT"
```

Those cell names are examples, not final chosen examples. The command writes
`router_cell_borrowing.png`, a vector PDF, full source CSV, route budget CSV,
and an estimand/provenance JSON. Keep the per-patient route Parquet INTERNAL.
The script refuses a non-interacting or non-0.3 config, non-tuning rows,
missing cells, duplicate patient-target rows, or invalid route sums.

## Separate prediction-dependence analysis

The existing evaluator accepts `--remove-source disease:...` or
`--remove-source outcome:...` and writes paired full-versus-masked CIFs. Its
masking path *renormalizes* the remaining route weights. A future panel can
show the mean or distribution of `CIF_full − CIF_masked` for a named source at
each horizon, on exactly the same tuning patients. This is a within-model
intervention on a trained router, not the gain from having trained with that
source. It should not share a color scale or caption language with raw gate
weights. The source-type factorial models answer the separate question of
whether learning with a source bank improves predictive performance.
