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
# Vocabulary-learning diagnostics

The vocabulary atlas is descriptive and is not, by itself, evidence that
clinical concepts were or were not learned. Run the exposure-stratified audit
before interpreting it:

```bash
python -u -m opera.run.diagnose_vocabulary_learning \
  --checkpoint "$BONSAI_PRETRAIN_CKPT" \
  --vocabulary "$BONSAI_PROCESSED_DATA/daly_care/vocabulary.pt" \
  --subject-data "$BONSAI_PROCESSED_DATA/daly_care/subject_data_train.pt" \
  --subject-data "$BONSAI_PROCESSED_DATA/daly_care/subject_data_tuning.pt" \
  --output-dir "$BONSAI_RESULTS_ROOT/vocabulary_learning_pretrain64" \
  --batch-size 4 \
  --num-workers 0 \
  --max-loss-batches 32 \
  --max-context-batches 32 \
  --logit-chunk-size 64 \
  --attention-backend sdpa
```

The command writes:

- `vocabulary_frequency_coverage.csv`: the fraction of vocabulary and actual
  event mass retained by candidate minimum-frequency thresholds;
- `token_learning_diagnostics.csv`: exposure, norm, frequency stratum, and
  sampled target-level code loss;
- `frequency_stratified_neighbors.csv` and
  `neighbor_coherence_by_frequency.csv`: an exact-neighbour audit on a bounded,
  frequency-stratified token sample;
- `contextual_sensitivity_summary.csv`: representation movement after masking
  numeric values, shifting the calendar by five years, and reversing event
  chronology;
- `contextual_linear_probes.csv`: held-out-subject linear decodability of
  normalized values and calendar time from raw versus contextual event states.

If an actual initialization checkpoint was saved, pass it with
`--initial-checkpoint`. This adds exact row-wise movement from initialization.
Do not reconstruct a random checkpoint after training and describe it as the
initial state: model construction order and RNG state make that comparison
non-identical.

The neighbour audit intentionally caps its quadratic calculation at 5,000
frequency-stratified tokens. Raise `--max-neighbor-tokens` only if memory and
runtime permit. The current `token_family` is a source/namespace family, not a
clinical ontology; merge curated token metadata in a subsequent analysis for
clinical-family precision.
