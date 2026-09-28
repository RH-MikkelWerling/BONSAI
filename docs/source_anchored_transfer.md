# Source-anchored OPERA transfer pilot

This pilot asks whether a patient-specific route can borrow information from
other diseases and outcomes while keeping the source adapters anchored to their
own supervised examples. It does **not** treat routing weights as diagnoses or
causal effects. The encoder remains the pretrained/shared-BCE representation;
fine disease (24 names) and target outcome (87 names) identify sources.

The shared horizon-BCE head is loaded from `OPERA_SHARED_BCE_CKPT`. Each source
has a rank-8 bottleneck residual. A router combines the patient embedding,
fine-disease identity, and target-outcome identity, once for all six horizons.
The shared path is always available. Borrowed source parameters are detached
but their outputs remain differentiable through the patient embedding and
router. A home disease and a home outcome have a fixed 0.1 route floor when
supported; only their own eligible labels update their parameters. Training-only
event counts shrink low-support residuals toward zero. Unknown fine identity
uses the shared path. The anchored and unanchored versions differ only in
whether borrowed adapter parameters receive gradients.

Four named conditions should be selected on tuning data, not held-out data:

| Condition | Generated config |
|---|---|
| Shared BCE | `direct_cr_objective_horizon_bce` |
| BCE + factorized cohort/outcome prototypes | `direct_cr_objective_horizon_bce_factorized_cohort_outcome_prototypes` |
| Unanchored source adapters | `direct_cr_objective_horizon_bce_unanchored_adapters` |
| Source-anchored adapters | `direct_cr_objective_horizon_bce_source_anchored_adapters` |

The adapter conditions begin from the **same** completed shared-BCE checkpoint,
retain its shared head and encoder, and initially freeze the encoder. This
staged comparison is not a matched-from-scratch parameter/compute comparison;
report both the shared checkpoint's training cost and the adapter-stage cost.
Use a separate matched-budget run if that distinction matters to a claim.
The cohort/outcome prototype control uses `cohort_grouped`; the adapters use
`cohort_fine`. Do not interpret their routing weights as directly comparable.

Example (server environment and `.env` already loaded):

```bash
export OPERA_SHARED_BCE_CKPT=/absolute/path/to/shared_bce/best.ckpt
python -m opera.run.preflight_direct_competing_risk --config-name generated/direct_cr_objective_horizon_bce_source_anchored_adapters
python -m opera.run.contrastive_multicohort --config-name generated/direct_cr_objective_horizon_bce_source_anchored_adapters
python -m opera.run.evaluate_direct_competing_risk \
  --checkpoint /absolute/path/to/anchored/best.ckpt \
  --config-name generated/direct_cr_objective_horizon_bce_source_anchored_adapters \
  --split tuning --export-routes --remove-source disease:DLBCL \
  --output-dir /absolute/path/to/tuning_evaluation
```

The evaluator writes standard predictions and metrics, a patient/outcome route
table, and extra `removed_source_cif_*` columns. Removal is a **diagnostic**:
it suppresses a named source and renormalizes routes without retraining, so it
is not an unbiased estimate of the source's causal contribution. The checkpoint
stores the ordered source names and train-only support buffers; the training
directory also contains `source_training_support.csv`. Save the resolved config,
checkpoint hash, split, and source checkpoint path with exported comparisons.

For the morning, test encoder learning-rate multipliers `0.05`, `0.20`, and
`1.00` on the winning readout, using identical splits and a tuning-only selection
rule. Do not run this sweep as part of the present pilot.

Before server launch, run the focused tests and a GPU cost check:

```bash
python -m pytest -q tests/test_source_anchored_readout.py tests/test_experiment_registry.py
python -m scripts.benchmark_source_adapters --device cuda --batch-size 16
```
