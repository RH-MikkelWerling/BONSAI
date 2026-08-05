# OPERA run readiness and follow-up register

This document records the checks required for the current full-panel OPERA run
and the scientific questions that must remain visible after it finishes. It is
a decision register, not evidence that an unchecked item has passed.

## Before starting the production run

- [ ] Synchronize the complete modified files rather than merging individual
  snippets, especially `OperaContrastiveModule.py`.
- [ ] Regenerate configs with `python -m opera.run.generate_sweep_configs`.
- [ ] Confirm the resolved config uses physical batch 8, logical train and
  validation batches 256, `mean_last_128`, hierarchical-support aggregation,
  FiLM numeric integration inherited from the checkpoint, anchor weight 0.02,
  and DAPT pair-weight floor 1.0.
- [ ] Confirm the DAPT embedding store was built from the same pretrained
  checkpoint, pooling rule, subject population, and sequence construction.
  Training-store coverage must be at least 99%.
- [ ] Confirm every configured outcome belongs to exactly one of the 11 outcome
  families, and AMYLOIDOSIS, BL/LBL, and HCL exclude the three outcomes that
  require a valid second-line-treatment definition.
- [ ] Run a five-train/five-validation-batch smoke test in a fresh output
  directory. It must complete training, logical validation, checkpoint writing,
  and CSV logging without warnings that indicate a model/data failure.
- [ ] Verify the logged head and encoder learning rates rise on every logical
  optimizer step and reach `5e-5` and `5e-6`, respectively, after the two-epoch
  warm-up. Do not resume checkpoints produced by the earlier epoch-stepped
  warm-up implementation.
- [ ] Use a fresh production output directory so completion markers and
  scheduler state from an earlier run cannot be reused.

## During the run

- [ ] Monitor `train/loss_epoch`, `val/loss`, contrastive KL/excess loss and
  target entropy, competing-risk loss, anchor loss, and both parameter-group learning rates with
  `python -m opera.run.monitor_training <run-directory>`.
- [ ] Check that losses are finite and validation does not immediately diverge.
  Anchor loss is a drift diagnostic and is not expected to decrease from its
  near-zero initialization.
- [ ] Inspect family losses and active-outcome counts. Improvement must not be
  confined to one common/easy family; sparse outcomes should not dominate.
- [ ] The legacy ever-event logistic probe is disabled in the production
  config. If explicitly enabled, treat it only as a diagnostic: it ignores
  censoring time and is not a survival estimand. Use train-fitted, tuning-scored
  Cox and fixed-horizon IPCW/CIF probes for representation validation.
- [ ] Record wall time, peak GPU memory, physical/logical batch sizes, and the
  number of logical batches. These are needed to compare future context-length
  and model-capacity runs fairly.

## Immediate post-run acceptance checks

- [ ] Identify best, final, and initial checkpoints and retain their resolved
  configs and metadata sidecars.
- [ ] Plot total, contrastive, competing-risk, family, and selected outcome
  train/validation curves against both optimizer step and epoch.
- [ ] Quantify encoder parameter displacement and patient-representation cosine
  drift from the original BONSAI checkpoint. Determine whether learning occurred
  mainly in the projection/competing-risk heads or in the encoder.
- [ ] Compare frozen probes for the original BONSAI and OPERA checkpoints using
  identical splits and preprocessing. Report per-outcome results, family macro
  summaries, and uncertainty—not only their unweighted grand mean.
- [ ] Run the representation-gradient-conflict diagnostics and inspect whether
  outcome families exert systematically opposing gradients.
- [ ] Treat `outcome_batch_diagnostics.csv`, `outcome_diagnostics.csv`, and
  `family_diagnostics.csv` as the primary contrastive audit artifacts. They
  report target entropy, learnable KL, headroom, effective-pair support, event
  composition, and representation-gradient norm without widening Lightning's
  training CSV.
- [ ] Compare `family_gradient_conflict.csv` within-family and between-family
  cosines before enabling family projection heads. A family head is warranted
  only when conflict is reproducible across logical batches and the clinical
  grouping is at least as coherent as an empirical grouping.
- [ ] `normalization_references.json` contains median initial-checkpoint KL per
  outcome. To test fixed loss-scale calibration, set
  `cross_outcome.outcome_scale_mode=initial_kl` and
  `cross_outcome.outcome_reference_scale_file=<that JSON>`. These scales stay
  fixed during training; do not recompute them from the adapted checkpoint.
- [ ] Reject or qualify the run if the best checkpoint is driven by only one
  family, if most outcomes are inactive, if validation worsens throughout, or
  if downstream results do not beat the original checkpoint.

## Central DLBCL experiment

The primary clinical endpoints are overall survival and treatment failure
(death or second-line treatment). The 730-day analysis is the prespecified
paper-aligned horizon; other horizons are secondary.

- [ ] Use one locked DLBCL definition (`cohort_fine == DLBCL` for the primary
  analysis; `cohort_grouped == DLBCL_like` only as a named sensitivity analysis).
- [ ] Compare tabular baseline, random initialization plus fine-tuning, original
  BONSAI frozen probe, BONSAI plus fine-tuning, OPERA frozen probe, and OPERA plus
  fine-tuning on identical patient splits.
- [ ] Run exact cached Cox survival fine-tuning and fixed-horizon IPCW analyses.
  For non-death endpoints, distinguish cause-specific/net-risk evaluation from
  cumulative-incidence evaluation with death as a competing event.
- [ ] Report C-index, horizon AUROC/AUPRC, Brier score, and calibration with
  paired patient-bootstrap confidence intervals. Do not infer value from UMAP
  separation alone.
- [ ] Verify index dates, strictly future outcomes, censoring, competing events,
  split isolation, and treatment-failure construction against the earlier DLBCL
  paper contract.

## Numeric-value validation

- [ ] Confirm from checkpoint metadata—not parameter-count inspection—that the
  encoder uses `value_embedding_mode: film` and that value tensors reach it.
- [ ] Run matched models with: full FiLM values; the same events with values
  masked during both training and evaluation; laboratory concepts without
  values; and laboratory events removed.
- [ ] Prioritize value-dependent endpoints: AKI/creatinine, anemia,
  neutropenia, thrombocytopenia, potassium, sodium, calcium, albumin, and related
  grade-2/grade-3 thresholds.
- [ ] Compare against simple last-observed-value tabular baselines.
- [ ] Test patient-value shuffling within each lab concept, controlled value
  perturbations, threshold monotonicity, and relevant gradients/attributions.
- [ ] Prevent leakage: inputs must end before the prediction index, outcomes
  must be strictly future, same-time measurements must not reveal labels, and
  all normalization/statistics must be learned from training data only.

## Input-domain and context-length ablations

- [ ] Pretrain a PHAIR-light-style diagnosis/procedure/medication-only model
  from scratch. Filtering the current full-input checkpoint only at OPERA or
  fine-tuning time is not a valid comparison.
- [ ] Optionally test diagnosis/procedure/medication plus lab concepts without
  numeric values to separate code-domain coverage from value integration.
- [ ] Measure sequence-length quantiles separately for every input policy:
  median, p90, p95, p99, p99.5, p99.9, maximum, and truncation fractions at
  3,372, 4,096, 8,192, 12,288, and 16,384 tokens.
- [ ] Select context length from those distributions and downstream tail-stratum
  performance. The collaborators' <0.1% truncation statement applies to their
  restricted domains and cannot be transferred directly to the full event set.
- [ ] Retrain when changing maximum context; do not override a checkpoint's
  maximum sequence length after pretraining.
- [ ] Benchmark unpadding and mixed-window training for memory, throughput, and
  numerical equivalence on the V100 environment. Confirm no hidden dependency
  on FlashAttention when SDPA is selected.
- [ ] Revisit adaptive-truncation `n=100` only with explicit coverage tables.
  Keep collaborator compatibility as a named setting rather than assuming that
  the same `n` is optimal for a smaller dataset.

## Model capacity and optimization

- [ ] Keep the current hidden-64/layer-4/head-4 model as the small baseline.
- [ ] After fixing input domains and context policy, test a medium model first:
  hidden 128, six layers, and four or eight heads. Consider hidden 256/eight
  layers only if the medium model gives a justified downstream gain.
- [ ] Run a short learning-rate sensitivity comparison: encoder multipliers
  0.1 versus 0.3, with 1.0 only if stable. Compare validation/downstream gains,
  representation drift, and gradient norms—not training loss alone.
- [ ] Test anchor weight 0 versus 0.02 and, if needed, a modest stronger value.
  The anchor should prevent destructive forgetting without keeping the encoder
  numerically fixed.
- [ ] Test logical-batch sensitivity (for example 128 versus 256) while keeping
  the physical batch memory-safe. Gradient caching removes memory dependence,
  not the statistical dependence of pairwise objectives on logical batch size.
- [ ] Keep effective-pair fraction as a diagnostic rather than multiplying the
  primary loss by it. The hierarchical global support shrinkage is the primary
  reliability control; compare the former scaling only as a sensitivity run.
- [ ] Report contrastive cross-entropy as target entropy plus excess KL. Use KL
  for checkpoint comparison because target entropy varies with batch composition;
  verify headroom before interpreting outcomes with diffuse targets.
- [ ] Treat family weights and `support_tau_locations=100` as prespecified
  primary choices but run uniform-macro and alternative support-shrinkage
  sensitivity analyses before making strong scientific claims.
- [ ] Keep `model.projection_mode=shared` as the primary model until gradient
  diagnostics justify the `family` ablation. Family mode uses one shared MLP
  trunk and one final projection per outcome family; the pooled encoder state,
  not any family projection, remains the downstream foundation embedding.

## Diagnostic-first branch decision

Run the gradient diagnostic on the original BONSAI checkpoint and the partial
OPERA checkpoint with identical logical batches. Use physical batch 8 and
logical batch 256. Each checkpoint directory now contains long-form outcome
diagnostics, fixed initial-KL reference scales, and within/between-family
conflict summaries in addition to the cosine matrices.

For an encoder-only BONSAI checkpoint, the diagnostic loads the pretrained
encoder and deterministically initializes the configured shared projection
head from `--seed`. Thus its KL references describe the actual start of OPERA
adaptation, rather than pretending the original pretraining learned an OPERA
projection that did not yet exist.

```bash
python -m opera.diagnostics.representation_gradient_conflict \
  --checkpoints "$BONSAI_CHECKPOINT_ROOT/daly_care_pretrain/best.ckpt" \
                "$BONSAI_CHECKPOINT_ROOT/daly_care_joint_opera_partial/best.ckpt" \
  --config-name generated/joint_opera_full_panel \
  --output-dir "$BONSAI_RESULTS_ROOT/opera_representation_diagnostics" \
  --batch-size 8 --logical-batch-size 256 --batches 16 --num-workers 0
```

Choose the next run only after inspecting those artifacts:

- Shared projection + no scaling is the conservative default.
- Shared projection + fixed initial-KL scaling is the scale-control ablation;
  it is appropriate when gradient norm is strongly explained by initial KL.
- Family projection + fixed initial-KL scaling is justified only when stable
  between-family conflict exceeds within-family conflict and several families
  retain meaningful headroom.
- Do not normalize dynamically by current KL or headroom. That creates a moving
  objective and can explode weights for solved or nearly uniform outcomes.

The generator writes three complete training configs:

- `generated/joint_opera_full_panel`: shared projection, no outcome scaling.
- `generated/joint_opera_initial_kl`: shared projection, fixed initial-KL scaling.
- `generated/joint_opera_family_initial_kl`: shared trunk plus family heads,
  fixed initial-KL scaling.

For either calibrated config, point the environment variable at the reference
artifact from the *original* checkpoint diagnostic before launching:

```bash
export OPERA_KL_REFERENCE_FILE="$BONSAI_RESULTS_ROOT/opera_representation_diagnostics/00_best/normalization_references.json"
```

The calibrated run fails closed if that artifact lacks any configured outcome
or contains a non-positive/non-finite scale. If 16 logical batches do not cover
the complete panel, rerun the diagnostic with more batches; do not fill missing
rare-outcome scales with an arbitrary constant.

For the cheap projection-only comparison, add `model.freeze_encoder=true` to
the shared calibrated and family calibrated smoke runs and keep their data seed,
logical batches, optimizer budget, and validation batches identical. This still
encodes each physical batch, but isolates whether extra projection geometry is
useful before spending a full encoder-adaptation run. A cached-head benchmark is
an efficiency improvement, not required for the scientific comparison.
- [ ] Check `torch.compile`/graph-break behavior separately; do not assume the
  graph-safe competing-risk expressions imply the entire model compiles.

## Longer-term representation direction

- [ ] Evaluate task-conditioned/direct prediction inspired by EveryQuery as a
  separate architecture after the current BONSAI-versus-OPERA comparison is
  established. It should not retroactively replace the baseline experiment.
- [ ] Diagnose residual treatment-year and sequence-length signal with adjusted
  probes and stratified downstream evaluation. Temporal structure in UMAP is
  not itself failure, but performance should not arise solely from calendar or
  utilization shortcuts.
- [ ] Cache reusable subject datasets with a version key covering source files,
  cohort membership, outcomes, splits, maximum length, and preprocessing config
  so repeated smoke tests avoid safe-but-expensive regeneration.
- [ ] Run multiple seeds for final comparisons and preserve configs, software
  versions, checkpoint hashes, split hashes, and complete evaluation artifacts.
