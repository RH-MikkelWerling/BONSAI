# Locked experiment intent: supervision pooling and source routing

For the ordered pre-launch checks, tabular comparison and reduced external
pretraining matrix, see `final_experiment_gates_20261014.md`.

## Study state confirmed on 14 October

The architecture-matched 256-wide pretraining is running with the collaborator
specifications. Completed 64-wide source-arm, borrowed-update and specialist
fits remain matched supplemental evidence; they are not controls for 256-wide
models. Tabular feature matrices/baselines are today's separate workstream.
Cohort and Table 1 await supervisor sign-off. Preserve the endpoint-definition
caveats in reporting; the panel must not be described as uniformly full CTCAE.

## Final model decision

Multi-outcome supervision with disease-source routing is an acceptable final
model. Outcome-source routing is optional and requires incremental predictive
benefit under the matched 256 protocol. Do not keep it merely because it was
part of the original design, and do not keep searching until both banks win.
Select on tuning before opening held-out results. Compare disease-only against
both using paired AUC and Brier, support and heterogeneity. If inconclusive,
prefer the simpler disease-only source bank and report uncertainty. An outcome
bank mask retains outcome-conditioned queries and full multi-outcome training.

Outcome-only resembling neither does not prove no cross-outcome learning:
both jointly supervise the encoder. Small, nearly patient-invariant outcome
corrections can be approximately absorbed by existing outcome heads. Anchored
outcome corrections train through only their own head; borrowed usefulness is
not directly optimized. Optimization attenuation is a further hypothesis,
not a universal explanation inferred from performance parity.

## Supervision factorial: shared architecture throughout

| Regime | Disease scope | Outcome scope | Number of fits before exclusions |
| --- | --- | --- | ---: |
| S | One fine disease | One outcome | 24 x 69 |
| D | All applicable diseases | One outcome | 69 |
| O | One fine disease | All applicable outcomes | 24 |
| J | All applicable diseases | All applicable outcomes | 1 |

Use the same audited 256 pretrained checkpoint, trainable encoder, shared
linear hazard heads, six-horizon BCE, temporal split and eligibility rules.
No router, contrastive objective or pooled-supervised initialization. Keep
competing-death labels/files for single-outcome fits: death handling is part of
that endpoint's definition, not an extra supervised outcome head. Fine-disease
filters must use `cohort_fine`, not the grouped parent. Preserve structural
cohort/outcome exclusions. The generator includes every available fine disease,
including CLL, and writes a manifest; availability does not imply estimability.

The generator retains endpoint-group-macro training and grade grouping, with
groups restricted to the selected heads. It disables support curriculum in all
four regimes to avoid scope-dependent task activation. Learning rates, batch
settings, numeric representation, smoothness and other base settings are
inherited identically. Budget: 30 complete epochs, natural sampling, seed 42
initially, then matched seed replication. Equal epochs preserve nominal patient
passes but do not match steps, supervision density, gradient dilution or compute.
Record actual exposure, optimizer steps and GPU hours; effects are of these
training procedures, not pure information content at equal compute.

## Comparable checkpoint selection

All new factorial fits select the minimum
`val/cr_horizon_bce_endpoint_group_macro`: known-patient BCE mean per horizon,
then horizon mean, grade mean and endpoint-group mean over the complete tuning
split. Primary events are positive from their event time onward; competing death
is known negative; early administrative censoring is unknown beyond follow-up.
Single-class horizons are usable. Entirely unknown horizons are omitted; zero
usable labels fails. No AUC support floor is used for BCE selection. AUC/Brier
remain reporting metrics. The selector excludes hazard smoothness penalties.

This is the same rule over each fit's supervised scope, not the same tuning
population. J/D get one selected checkpoint each, reused for all their disease
evaluation cells; do not choose a pooled checkpoint separately for every cell.
Sparse or missing tuning evidence must be classified explicitly. Existing
specialists using `val/loss`, and J/router fits using AUC, are historical protocols.
Do not resume them with this changed selector. For new matched router comparisons,
set the same BCE monitor and preserve all other chosen router/shared settings.
Architecture/capacity exploration may retain its recorded tuning-AUC screen;
final comparison fits use this common BCE policy. No held-out-driven selection.

## Contrasts and evaluation

Evaluate S/D/O/J on exactly the same target disease/outcome patients and labels,
at every estimable horizon. Use paired patient resampling and report event and
control support, AUC and Brier, uncertainty and negative transfer. Do not silently
drop failed/unsupported specialists; publish coverage and reasons. Broader fits
with predictions in unsupported specialist cells do not enter complete factorial
contrasts, but remain visible in separate predictive summaries.

For higher-is-better score M, report D-S (disease pooling alone), O-S (outcome
pooling alone), J-O (disease pooling conditional on outcome pooling), J-D
(outcome pooling conditional on disease pooling), J-S (combined), and
J-D-O+S (interaction on M's scale). Reverse improvement signs for Brier. A
representative prespecified subset supports only a sampled-cell claim; full
landscape claims require the whole estimable specialist panel.

Separately run matched J supervision with neither, disease-only, outcome-only
and both source banks. These measure architecture/source-bank effects, not
removal of disease/outcome supervision. Learned relationship-mechanism claims
need an additional matched fixed/home-only-routing comparator.

## Materialize on the secure server after the checkpoint audit

Use the approved release's shared endpoint-group BCE candidate, not legacy
checked-in generated configs. Confirm it points to the finalized sequence store,
vocabulary, metadata and release. Choose batch settings appropriate to width256
in this base before generating all arms. The checkpoint argument is explicit;
the generator does not certify checkpoint width or architecture.

```bash
python -m opera.run.prepare_supervision_factorial \
  --base-config /secure/approved256/shared_endpoint_group_bce.yaml \
  --encoder-checkpoint /secure/pretrain256/version_0/best.ckpt \
  --output-dir /secure/factorial256/seed42/configs --epochs 30 --seed 42

python -m opera.run.preflight_direct_competing_risk \
  --config-dir /secure/factorial256/seed42/configs \
  --config-name D_all_diseases__overall_survival --require-existing-paths

python -m opera.run.contrastive_multicohort \
  --config-path /secure/factorial256/seed42/configs \
  --config-name D_all_diseases__overall_survival \
  hydra.run.dir=/secure/factorial256/seed42/runs/D_all_diseases__overall_survival
```

The historical entry-point name does not enable contrastive learning: these
configs explicitly use direct horizon BCE only. Run each manifest config in
a distinct output directory and preflight before training. Use fresh directories
per seed/backbone. No jobs are launched by materialization. Old specialist queue
wrappers enforce their historical config protocol; use the new configs directly
rather than passing them through those wrappers.
