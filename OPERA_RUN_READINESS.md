# OPERA run readiness and follow-up register

This document records the checks required for the current full-panel OPERA run
and the scientific questions that must remain visible after it finishes. It is
a decision register, not evidence that an unchecked item has passed.

## Locked direct competing-risk adaptation ladder

The universal contrastive geometry is now a diagnostic/legacy arm rather than
the assumed primary adaptation mechanism. The generator writes four matched
direct-likelihood configs, all initialized from the same BONSAI checkpoint,
using the same 87 outcomes, natural-distribution batches, weak anchor, pooling,
hierarchical family normalization, optimizer, and validation objective:

- `generated/direct_cr_full_panel`: linear outcome heads; primary baseline.
- `generated/direct_cr_family_trunks`: residual family prediction trunks.
- `generated/direct_cr_curriculum`: five training-support quantiles introduced
  progressively, with a smooth two-epoch ramp for every newly active tier.
- `generated/direct_cr_family_trunks_curriculum`: interaction arm; run only
  after the two one-factor ablations unless compute is abundant.

The family curriculum changes training weights only. Validation always scores
the complete 87-outcome objective, and active family weights are renormalized
to sum to one. This prevents a lower loss caused merely by omitting difficult
families from being mistaken for improved validation.

Do not divide the encoder loss by 87 on top of hierarchical aggregation. The
current loss already allocates a normalized budget across families and then a
normalized support-smoothed budget within each family. A further constant
division only changes the effective learning rate. Before adding GradNorm or
gradient surgery, measure per-family encoder-gradient norms and conflicts under
the direct likelihood; loss magnitude alone is not the quantity to equalize.

## Pretraining audit and locked ablations

The current DALY-CARE checkpoint uses causal autoregressive pretraining. At
position `t`, the encoder receives only tokens and numeric values through `t`
and predicts the code and (when present) normalized numeric value at `t+1`.
The numeric target is shifted independently and is not exposed through FiLM.
Calendar censoring is exclusive at 2022-01-01, split files are separate, causal
attention is enforced for AR datasets, Fourier absolute time stays finite in
mixed precision, and ehr2meds' train-fitted normalized values are validated in
`[0, 1]`. No direct future-value leakage was found in this path.

The audit nevertheless identifies four scientific limitations:

- Tail-only windows emphasize recent calendar time, utilization, and long-record
  structure; this is a plausible contributor to the observed UMAP geometry.
- Next-token prediction can exploit deterministic ordering among simultaneous
  events. It is a valid sequence objective but not proof of clinical semantics.
- Code cross-entropy and numeric MSE are separately averaged and then added.
  `value_regression_loss_weight=1.0` is explicit, but does not equalize their
  encoder-gradient contributions.
- A hidden-64, four-layer model may encode useful prediction signal without
  producing visually disease-clustered mean-pooled embeddings.

Two matched pretraining configs are now available in addition to the observed
tail baseline:

- `daly_care_pretrain_no_values`: identical event sequence and FiLM parameter
  structure, but all scalar inputs/targets are masked and numeric loss is zero.
- `daly_care_pretrain_mixed_window`: 50% tail and 50% random clinical windows in
  training, with deterministic tail validation.

Use downstream frozen probes and fine-tuning—not pretraining loss or UMAP
alone—to select among these checkpoints. The decisive numeric experiment must
also include future lab endpoints, a last-observed-value baseline, and a
within-concept value-permutation control materialized independently within each
data split. Do not shuffle values globally across lab concepts or across
train/tuning/test boundaries.

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
# Gradient-aligned scaffolding roadmap

The next transfer experiment should treat scaffolding as a measured training
intervention, not as another representation objective.  For target outcome
`t`, train with `L_t + lambda * sum_s w[s,t] L_s`, where the auxiliary weights
are estimated before the scaffolded run from representation-gradient cosine
on training-fold patients.  This resolves the apparent circularity: use a
fixed starting checkpoint to estimate the scaffold map, freeze that map, and
then start a fresh scaffolded run from the same checkpoint.  The held-out test
split must never determine weights.  Online EMA weights are a later ablation,
not the first implementation.

Required controls are target-only, all-auxiliary, gradient-aligned,
support-matched random, and anti-aligned auxiliaries.  Positive cosine is a
hypothesis about optimization compatibility, not evidence of transfer until
these controls improve target survival metrics.

Analyse scaffolds hierarchically:

1. outcome-to-outcome gives maximum resolution but is noisy;
2. outcome-family-to-target is the stable, interpretable default;
3. source-disease-by-family-to-target is the central cross-disease transfer
   map and directly tests the data-scarcity/scaffolding thesis;
4. patient-specific gates and encoder-layer-specific maps are deferred until
   the static maps are validated.

Estimate each cell over multiple logical batches and report mean cosine,
bootstrap interval, event support, joint patient support, and gradient norm.
Shrink noisy outcome cells toward their family and disease-family means.
Candidate weights should use only stable positive alignment (for example the
positive lower confidence bound) multiplied by a capped support-reliability
term, then be normalized so the total auxiliary gradient scale is controlled.

The existing representation diagnostic now accepts
`--objective competing_risk`; run the conflict verdict on its pair-batch output
before choosing auxiliaries.  Disease-by-family restriction and the frozen-map
training consumer remain explicit implementation gates before claiming a
scaffolded model result.

# Remaining validation and ablations

- Finish the direct competing-risk run and select by survival validation loss;
  do not infer convergence from three or four epochs.
- Run random initialization versus standard pretraining versus direct CR for
  DLBCL treatment failure and for a deliberately numeric-dependent laboratory
  endpoint.
- Run value ablations (FiLM/numeric values on versus off) and masked-value
  pretraining; verify numeric value perturbations change predictions in the
  expected direction.
- Compare linear shared head, family-specific prediction trunks, and the
  support-tier curriculum.  The curriculum ranks individual outcomes by
  training event-location support: high support first, then medium, then all.
  It no longer hand-picks infection, hematologic, or electrolyte families.
- Add functional preservation as an optional teacher-student penalty over the
  heads' full native interval distributions, with a low default coefficient.
  Do not reduce each head to arbitrary 30/90-day risk tokens.
- Measure discrimination and calibration with survival-native metrics by
  outcome and compact family summaries; retain AUROC only as a secondary probe.
- Quantify sequence truncation, compare clinical-code-only versus numeric-full
  input, and test a larger encoder only after the low-level representation and
  numeric checks pass.
