# Real-data censoring maps

`opera.run.plot_censoring_aware_map` joins existing coordinates and CIF predictions
by subject ID. It does not train a model or rerun UMAP. Use coordinates from the
same model and split as the predictions; prefer tuning during development.

```bash
python -u -m opera.run.plot_censoring_aware_map \
  --coordinates "$COORDINATES" \
  --predictions "$TUNING_PREDICTIONS" \
  --outcome creatinine_increased_g2plus \
  --cohort-level cohort_fine --cohort DLBCL \
  --membership "$BONSAI_COHORT_MEMBERSHIP" \
  --split tuning --horizon 365 \
  --output-dir "$MAP_ROOT/creatinine_DLBCL_365d"
```

Set the variables to existing exports. The membership file must contain
`subject_id` and the requested `cohort_fine`/`cohort_grouped` column. It is optional
when predictions or coordinates already have complete membership. Older pooled
prediction exports can have entirely empty `cohort_fine`; membership fills that
gap without rerunning model inference. Contradictory identities fail explicitly.

Selectors:

- `--cohort-level all`: omit `--cohort`; pool patients in the supplied exports.
- `--cohort-level cohort_grouped --cohort DLBCL_like`: one grouped cohort.
- `--cohort-level cohort_fine --cohort DLBCL`: one fine cohort.

"All" means all available exported patients for that outcome and split; a
specialist-only prediction export cannot represent all cohorts. Fine selection
subsets the existing model's predictions: it does not make the model a specialist.
Keep one pooled projection for comparable anatomical locations across cohort
subsets. However, map bandwidth is estimated per selected subset unless supplied,
and censoring weights are estimated in that subset. Common coordinates alone do
not make smoothing/weights identical. Inspect support before interpreting small
cohorts. Outputs are PNG, PDF and JSON recording the selection and diagnostics.

Boundary/smoothing checks: the evaluation grid extends four kernel bandwidths
beyond patient coordinates to avoid cropping the support contour. This changes
the displayed domain, not the estimator bandwidth. Use `--bandwidth-scale 0.5`,
`0.75`, and `1.0` in separate output directories for a sensitivity comparison;
do not choose a setting merely for apparent separation. Narrower kernels show
more local structure but lower ESS and increase sampling noise. CLI residual
maps use a shared +/-0.20 probability range and mark colorbar overflow. For the
pooled batch script use `BANDWIDTH_SCALE=0.5 bash scripts/plot_pooled_bce_censoring_maps.sh ...`.
Risk is the incidence of the coded primary event: for overall survival the risk
panel is mortality probability, not probability of survival.
