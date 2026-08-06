# OPERA representation diagnostics

## Endpoint dependencies

`opera/configs/endpoint_dependencies.yaml` is the auditable label-dependency
graph. Composite endpoints and alternate definitions are excluded from scaffold
discovery. G2+/G3+ pairs are detected automatically, retained as
`nested_threshold_control`, and excluded only as direct scaffold edges.

## Competing-risk gradient diagnostic

The diagnostic separates the exact likelihood into components with a common
valid-patient denominator:

```
full likelihood = exposure + primary event + competing death + smoothness
```

For scaffold discovery, use a conditional permutation null. Gradient rows are
permuted independently by endpoint within cohort, observed status, and follow-up
quartile. `excess_alignment` is observed cosine minus the mean conditional-null
cosine; it is therefore the preferred discovery quantity, not raw cosine.

Example:

```bash
python -m opera.diagnostics.representation_gradient_conflict \
  --checkpoints "$DIRECT_CKPT" \
  --config-name generated/direct_cr_full_panel \
  --objective competing_risk \
  --gradient-components full_likelihood exposure primary_event competing_death \
  --null-permutations 16 \
  --batches 32 \
  --batch-size 8 \
  --logical-batch-size 256 \
  --num-workers 6 \
  --output-dir "$DIAGNOSTIC_ROOT/direct_cr_components" \
  --override hardware.num_workers=6
```

New artifacts include:

- `endpoint_dependencies.csv`: every pair's dependency/control classification.
- `gradient_components_batches.csv`: observed, conditional-null, and excess
  alignment by component and logical batch.
- `gradient_components.csv`: pair-level aggregate.
- `gradient_cosine_<component>.csv`: square raw component matrices.
- `atomic_scaffold_candidates.csv`: ranked eligible atomic pairs.
- `atomic_scaffold_matrix.csv`: square excess-alignment discovery matrix;
  excluded cells remain missing rather than being interpreted as zero.

## Censoring-aware embedding panel

The plotting CLI accepts the coordinates CSV written by
`opera.run.plot_patient_embeddings`. Supplying the original primary and
competing-outcome tables ensures deaths are not treated as administrative
censoring.

```bash
python -m opera.run.plot_censoring_aware_embeddings \
  --input "$PLOT_ROOT/patient_umap_coordinates.csv" \
  --x-col component_1 \
  --y-col component_2 \
  --outcome "$BONSAI_OUTCOMES_DIR/anemia_g3plus.parquet" \
  --competing-outcome "$BONSAI_OUTCOMES_DIR/overall_survival.parquet" \
  --horizon-days 90 \
  --covariates age_at_index treatment_year sequence_length sex cohort_grouped \
  --title "Grade 3+ anaemia" \
  --output "$PLOT_ROOT/anemia_g3plus_censoring_aware_90d.png"
```

The five panels show observed status, time to status, IPCW kernel cumulative
incidence, cross-fitted nuisance-adjusted enrichment, and effective local sample
size. Risk/enrichment regions below `--min-effective-support` are masked.
