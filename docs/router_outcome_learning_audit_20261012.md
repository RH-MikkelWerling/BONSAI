# Outcome-source learning audit

The source arms can differ in useful flexibility even when both axes receive many labels. More patients reduce uncertainty; the horizon BCE objective averages over patients/horizons and then over endpoint groups, so data counts do not translate directly into larger gradients. In the balanced special case, an outcome adapter's own loss has about one endpoint group's objective weight; a disease adapter sees each endpoint group's contribution restricted to its own patients. Neither bank has a general data-count advantage.

The code deliberately detaches borrowed source parameters. A source is trained on its own task domain: disease adapters on eligible outcomes within their disease, outcome adapters on eligible patients for their outcome. Borrowing updates the router and patient representation, but not the borrowed adapter. This preserves source ownership, but it does not train every outcome correction to be useful under other outcomes' risk heads. Disease corrections are exposed to many risk heads during home training; outcome corrections are exposed to one outcome's risk head. This is a substantive functional asymmetry.

With almost constant routing and approximately linear adapter activation, an outcome-dependent correction has form `h_o = (I + A_o)h`. Its prediction `W_o h_o + b_o` can be represented by another ordinary per-outcome head `W'_o = W_o(I + A_o)`. Nonlinear and patient-dependent routing can exceed this equivalence, but small initialization starts near this redundant regime. Disease-conditioned corrections change `A` with patient disease, providing additional conditional flexibility. Outcome-only resembling no-sources is consequently compatible with working gradient paths.

The default base configuration also enables a five-tier support curriculum. With warmup=1, stage=1, ramp=2, the lowest-support tier starts at zero-based epoch 4 and reaches full weight at epoch 5. The saved server config must confirm these settings. An outcome source for a disabled target has no home gradient during that period; disease sources can train from already enabled targets. Later all tiers have full objective weight.

Other attenuation mechanisms are the home route weight (about 0.064 for 69 outcome sources versus 0.088 for 24 disease sources at uniform logits and home floor 0.05), small adapter initialization, and Adam epsilon 1e-6. Epsilon can damp very small gradients; actual source gradient statistics are required to quantify that effect. Event shrinkage is `events / (events + 10)` in guarded runs. Disease counts pool events across outcomes, so that shrinkage is asymmetric, but common outcome sources with hundreds/thousands of events should not be strongly shrunk.

## Confirmed diagnostic artifact

The existing `effective_residual_magnitude` subtracts a standard `Linear` baseline from routed `einsum + bias` logits. Under mixed precision, the routed multiplication rounds before adding the float32 bias, while `Linear` returns a rounded result including bias. This yields a nonzero metric even when source corrections are exactly zero. A local explicit FP16 example with zero correction produced about 0.00011 with ordinary head biases and about 0.00080 with initial hazard bias -2.3. Small logged residuals therefore cannot establish adapter use.

Mixed precision can additionally erase actual small corrections when adapted embeddings are cast to half precision before the risk projection. A controlled example erased about 94% of embedding-coordinate changes of size 1e-5. This confirms the mechanism exists, not its magnitude in the server checkpoints.

## Audit actual checkpoints

After syncing, on a GPU allocation run:

```bash
bash scripts/audit_router_source_learning.sh outcome_only
bash scripts/audit_router_source_learning.sh disease_only
bash scripts/audit_router_source_learning.sh none
```

Each arm uses the same seed to sample up to 256 training patients across the configured cohorts. The encoder is evaluated in FP32; cached representations feed both FP32 and the configured mixed-precision risk projection. The output records the checkpoint epoch, true projected correction, fraction of eligible logits unchanged, and the legacy metric artifact with all adapters absent. No optimizer step occurs. No held-out metrics or patient identifiers are exported. `source_diagnostics.csv` reports source support, home gates, curriculum multipliers, parameter magnitudes, and FP32 objective gradient magnitudes relative to optimizer epsilon. These gradients diagnose the mathematical objective; they do not measure the AMP GradScaler or the optimizer's accumulated moments.

Interpret the outcomes in order: verify the resolved modes, curriculum stage, and positive home gradients; compare FP32 correction with mixed-precision correction; then assess redundancy/transfer through paired predictions and source-removal diagnostics. Checkpoint epochs must be considered when comparing arm magnitudes. Existing training runs keep their original behavior during this audit.

## Matched experiment permitting borrowed updates

The returned server audits at epochs 9–10 confirm valid fine identities, correctly masked source banks, and nonzero outcome corrections. Disease-only mean projected correction was 0.27073; outcome-only was 0.002032 (about 133-fold smaller). The outcome bank received 98.66% of routing mass. At its checkpoint about 6.02% of eligible logits were exactly unchanged by the correction under configured precision; this does not support complete numerical erasure. The no-source arm's true correction was zero and the legacy metric artifact was about 0.000552. Curriculum is explicitly enabled in the saved server configs, but all five tiers have full weight by these epochs. A matched curriculum-on/off benefit has not been established by these results.

One additional outcome-only arm changes `readout_mode` to `unanchored_adapters`, leaving scale, rank, shrinkage, sampling, encoder, initialization, loss, curriculum, and training budget fixed. Disease sources remain disabled. Outcome sources can now receive gradients from any target that routes to them. It writes a separate `_borrowed_updates` run directory.

```bash
bash scripts/run_guarded_outcome_borrowed_updates.sh preflight
bash scripts/run_guarded_outcome_borrowed_updates.sh launch
# If interrupted with last.ckpt, replace launch with resume.
```

Routing still describes which learned components are used, but an outcome-named component is no longer trained exclusively on that outcome. Shared embeddings remain an additive baseline regardless of the shared route coefficient, so gate mass is not a percentage of total predictive contribution. A curriculum-off comparison should be its own experiment after this matched comparison, rather than changing two factors at once.

## Locate the optimizer bottleneck before inferring clinical transfer

The 133-fold projected correction ratio does not establish either a wiring bug or a biological absence of related outcomes. Related grades share an endpoint-group loss denominator, but their source parameters and risk-head weights remain separate. Forward/gradient tests establish connectivity, not adequate optimization or transferable correction directions.

Run `bash scripts/inspect_router_source_optimizer.sh` after syncing. This is CPU-only: it loads saved checkpoints, inspects source parameter magnitudes and Adam moment states, and exports an aggregate JSON without reading patient data. It reports the adaptive update implied by saved moments/LR and the multiplicative attenuation from epsilon. The output file also includes primary-risk-head weight cosines for outcomes in the same endpoint group. That cosine measures parameter alignment, not proof of clinical similarity or transfer benefit. If stored optimizer tensors cannot be uniquely associated with the source parameter by shape, the inspector fails rather than assigning an incorrect state.

Use this with `source_diagnostics.csv` from the training-patient audit to locate the gap: smaller raw corrections/parameter updates indicate an optimization bottleneck; comparable raw correction norms but much smaller projected corrections indicate head geometry or mixture cancellation. Low gradients alone cannot establish small Adam updates, so inspect the saved moments. A diagnosis still requires the returned server statistics; local tests cannot identify which of these mechanisms dominates the real fits.

## Full interacting comparison with borrowed parameter updates

The prioritized experiment now enables borrowed gradients in **both** source banks of the full guarded rank-8 interacting model:

```bash
bash scripts/run_guarded_router_borrowed_updates.sh preflight
bash scripts/run_guarded_router_borrowed_updates.sh launch
```

One GPU is sufficient. The wrapper performs Bash syntax checks on the server and uses a separate `_borrowed_updates` run directory. The default run root is `data/models/approved_neural_router_borrowed_updates_20261012_v1`; an explicitly set `OPERA_NEURAL_RUN_ROOT` takes precedence. Use `resume` after an interruption with `last.ckpt`. The launcher composes the new config and requires the scientific settings to match the completed guarded reference except `readout_mode` changing from `source_anchored_adapters` to `unanchored_adapters`. The check includes curriculum, rank, scale, seed, sampling, encoder, objectives, training budget, source banks, inputs, and hardware configuration. Output locations and label-cache location are operational fields and may differ. Curriculum therefore stays enabled in this matched experiment; its benefit remains unestablished. Sources with zero home event support remain masked by the existing support rule.

Tests exercise the real horizon-BCE loss: one disease/target-outcome loss sends nonzero down/up gradients to a borrowed disease and a borrowed outcome, and Adam changes both. Initial forward outputs agree with the anchored implementation within float32 tolerance. Reference-contract tests reject additional changes to curriculum, rank, sampling, seed, or sequence input. The source-readout implementation itself has not been edited, preserving the source-code hash needed to resume existing anchored runs.

## Matched curriculum removal

The curriculum-off comparison is explicitly requested as a simplification test:

```bash
bash scripts/run_guarded_router_no_curriculum.sh preflight
bash scripts/run_guarded_router_no_curriculum.sh launch
```

It uses the full borrowed-update router and changes only `competing_risk.curriculum.enabled=false`. Scale 0.7, rank 8, event tau 10, home floor 0.05, seed, ordinary sampling, optimizer, and training budget remain matched. The preflight compares against the saved curriculum-on borrowed-update config and rejects any additional scientific setting change. Reference training need not have finished; its saved config must exist. If its run was placed under a custom root, set `OPERA_BORROWED_ROUTER_RUN` to that run directory. The new output name ends in `_borrowed_updates_no_curriculum`, so existing training is not overwritten. Use `resume` after a scheduler interruption. One GPU is sufficient; no job is submitted by these scripts.
