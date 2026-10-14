# Tonight's OPERA checklist for the 7 October supervisor meeting

This is a **tuning-only, single-seed** preview. Keep held-out closed and keep
patient-level exports on the secure server. A run is ready for its final figure
only after `training_complete.json` and `best.ckpt` exist; resumed Lightning
versions should be read together. Do not interrupt training to produce plots.

The completed aggregate comparison (common epoch 30) provisionally ranks
Fourier additive router at 0.74225 endpoint-group-macro CIF-AUC, Time2Vec
additive router at 0.73958, and Time2Vec interacting router at 0.73955.
These are descriptive tuning scores, not a final generalization ranking.
Next compare the matched 2-by-3 encoder/readout grid on fixed tuning patients:
Fourier and Time2Vec, each with grouped shared, family trunks, and additive
router. Keep the near-identical interactive router out of the primary forest
until its deferred implementation audit is complete.

## Ordered workflow for tonight

1. **Choose a provisional model on tuning.** Inventory completed runs, then
   compare overall endpoint-group-macro CIF-AUC at best, last, final-five, and
   common completed epochs. Evaluate fixed checkpoints on the same tuning
   patients and produce a paginated paired-bootstrap 365-day forest across the
   approved 69 outcomes, recording sparse exclusions explicitly. Highlight
   five clinical anchors in the presentation. Review family trade-offs and
   valid bootstrap counts. The forest describes uncertainty conditional on selected
   checkpoints; it does not turn tuning into held-out validation.
2. **Characterize the selected model.** Export pooled patient embeddings once
   and plot a global atlas colored by available demographics, disease, and
   treatment fields. Inspect metadata columns and denominators before choosing
   overlays; do not silently interpret missing registry fields as negatives.
   Produce censoring-aware five-panel plots for **five** clinically important,
   supported outcomes rather than batching all 69. If the winner is a router,
   add target-specific post-router movement panels for those five: these
   require one target-specific vector export each, whereas the global atlas
   reuses one pooled-vector export. Use router-versus-matched-baseline gains as
   a descriptive prioritization criterion only after clinical relevance and
   support are established. Record the selection rule and all 69 comparisons;
   do not present selected gains as unbiased evidence of a routing effect.
   These maps show associations and geometry, not feature importance or a
   causal explanation of what the model uses.
3. **Inspect router borrowing.** For each completed router, export tuning route
   weights and inspect directed disease-to-disease and outcome-family-to-family
   heatmaps, shared/own/borrowed mass, and support. Keep the 69-by-69 map as a
   supplement. If the overall winner is not a router, show this separately as
   a mechanism analysis of the strongest router, not an explanation of the
   winning non-router model.
4. **Plot the selected model's 69-outcome predictability atlas.** Use the
   approved registry/display groups and the selected run's logged tuning
   metrics. Label its support, endpoint-group aggregation, and tuning-only
   status. A matched simpler reference is useful if it remains legible.

The executable validation, forest, router, and predictability commands are in
`supervisor_validation_cohort_table1_20261007.md`. Keep these four steps in
this order; plotting can run in parallel after the provisional model is fixed.

## Deferred audit, not a blocker for the preliminary figures

- [ ] Revisit the full 69-outcome forest artwork against the **actual** server
  render. The paired tuning estimates/contrasts are available, but the layout
  is not yet publication-ready. Refine overall spacing, family-row hierarchy,
  label/legend readability, and the presentation of family macro estimates;
  keep the same color and marker for each model in both panels. Do not confuse
  descriptive equal-outcome family means with bootstrap family confidence
  intervals. This polish does not block the embedding and router figures.
- [ ] Investigate why additive and interacting router logs appear identical or
  one epoch shifted. Compare resolved configs, interaction parameters and
  gradients, checkpoint hashes, and full-precision predictions on identical
  patient-target rows. Do not claim an interaction benefit until this is
  resolved. If prediction arrays are exactly identical, treat the arms as one
  effective model in the forest rather than independent evidence.

## Step 1 details: the scientific story

- [ ] Inventory every current run: completed versus still running, resolved
  encoder (Fourier or scaled Time2Vec), head, loss/family policy, checkpoint,
  best epoch, and final epoch. Record failed or missing arms explicitly.
- [x] Run the aggregate neural comparator from
  `supervisor_preliminary_findings_20261007.md` once all selected runs finish.
  Compare best, last, final-five-epoch, and **common completed epoch** AUC;
  show family trade-offs. Do not choose an architecture from best-epoch AUC alone.
- [ ] Once the winner is selected, make its 69-outcome **predictability atlas**
  and optionally one matched simpler reference. These are log-derived; no GPU
  is needed. Check 69 outcomes/52 endpoint groups before presenting.
- [ ] Save a compact architecture trajectory figure or table: shared BCE,
  family trunks, additive router, interacting router, with encoder named.
  Distinguish early learning from eventual performance using common epochs;
  GPU-time comparison is a follow-up unless runtime logs are comparable.
- [ ] Evaluate each fixed matched checkpoint on the same tuning patients and
  make a paginated 365-day paired patient-bootstrap forest across all 69
  approved outcomes, excluding only prespecified low-support cases with reasons
  in the manifest. Highlight five clinical anchors on slides. The commands are in
  `supervisor_validation_cohort_table1_20261007.md`. Label it **tuning only**:
  the intervals condition on selected checkpoints and do not correct selection
  bias or estimate held-out generalization. If GPU time is short, show the
  log-derived point comparison instead, explicitly without intervals.

## Steps 2 and 3 details: embedding and routing figures

- [ ] Choose **one completed router checkpoint** and five clinically important,
  adequately supported target outcomes. Initial anchors for review are
  `overall_survival`, `treatment_failure`, `sepsis`, `neutropenia_g3plus`, and
  `aki_first_documented`; replace an anchor only after checking the approved
  outcome definition and tuning event/control support. They span disease
  course, infection, hematologic toxicity, and renal toxicity. Among eligible
  anchors, a larger paired router gain can prioritize panel order, but label
  any performance-guided selection exploratory. Do not choose by whichever
  embedding map happens to look most striking.
- [ ] Export tuning predictions plus routes; make the directed 24×24 disease
  and 12×12 outcome-family borrowing heatmaps. Inspect the full 69×69 matrix
  only as a supplement. Check row/column orientation, unsupported cells, and
  the home-source exclusion. Gate mass is **not** performance contribution.
- [ ] Export each target's shared and post-router vectors and render the **new
  five-panel movement version**. Compare each side by side with the original
  five-panel status/time/risk/calibration/support layout. Panel B should show
  sparse local-average arrows; verify magnification and median unscaled shifts
  in the JSON. Keep the best *legible* version, not necessarily the busiest.
- [ ] Visually review alignment and interpretation: same tuning patients,
  same target/horizon, PCA fit only to shared vectors, post-router transformed
  on those axes, plausible censoring/support masks, no decorative arrows over
  unsupported regions. If movement nearly vanishes in 2D, report that and the
  full-hidden-space shift instead of amplifying it without context.

The evaluation/plot commands are in
`supervisor_validation_cohort_table1_20261007.md`. The route export and routed
vector export can be requested in **one** evaluation invocation for the same
run and target (`--split tuning --export-routes --export-routed-outcome "$TARGET"`),
then passed to their respective plotting scripts. Use a fresh output directory
and each run's `hydra/config.yaml`, not `.hydra`.

## Meeting-ready if time allows

- [ ] A one-page slide: approved panel and cohort; architecture ladder with
  common-epoch/final-five caveat; predictability atlas; selected outcome points;
  router source-type diagram; one borrowing heatmap and one movement panel.
  State clearly that
  these are preliminary tuning observations, not held-out or replicated effects.
- [ ] Check the current sealed cohort manifest, flow exclusions, and Table 1
  source fields. Quality-registry performance status/risk have era-dependent
  availability. Do **not** relabel the historical Table 1 or old cohort-flow
  counts as current; show audited denominators or mark them pending.
- [ ] Queue the matched source-type arms (no sources, disease only, outcome
  only, both) only after verifying identical configs/checkpoint rules. Their
  planned 2×2 contrast is tomorrow's hypothesis, not a result tonight.
- [ ] Keep the tabular comparison as a separate workstream until its feature
  matrices, patient IDs, index dates, splits, outcomes, and censoring horizons
  match the neural release. Do not show an unaligned XGBoost comparison.

## Stop conditions before calling any figure final

- Wrong or uncompleted checkpoint; old 87-outcome defaults; mixed Fourier and
  Time2Vec labels; mismatched patient splits; `.hydra` rather than `hydra`;
  patient-level data exported off-server; low-support heatmap cells; maps whose
  legends imply 2D kernel risk is calibrated model risk; or arrows whose
  display multiplier is omitted. Any of these means **preliminary diagnostic**,
  not presentation-ready evidence.
