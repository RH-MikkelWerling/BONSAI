# Source-bank effects versus routing-context effects

This audit traces approved candidate generation, launch overrides, fine-disease identity and label loading, native encoder inputs, source routing, hazard prediction, horizon BCE, gradient caching, optimizer grouping, checkpoint reconstruction, evaluation, and route export. It reviews repository behavior and the returned server diagnostics; it cannot independently reread the secure server's complete results.

## What the completed factorial identifies

| Arm | Learned query context | Disease source bank | Outcome source bank | Supervised outcome panel |
| --- | --- | --- | --- | --- |
| Neither | Query constructed but inactive | Off | Off | Full |
| Disease sources only | Patient + fine disease + target outcome | On | Off | Full |
| Outcome sources only | Patient + fine disease + target outcome | Off | On | Full |
| Both source banks | Patient + fine disease + target outcome | On | On | Full |

`adapter_source_mode` controls source availability. It does not switch query inputs, outcome heads, or outcome losses. Both query contexts remain constant across the active-router arms. Consequently these are valid conditional source-bank interventions, not a factorial for presence/absence of disease and outcome information throughout the model. A disease-source-only router can produce different patient representations for different target outcomes. That is outcome-conditioned use of disease experts, not an enabled outcome-expert bank.

Use comparisons of disease sources versus neither and outcome sources versus neither for bank effects without the other bank; use both versus outcome sources and both versus disease sources for conditional bank effects. Their differences/interactions include changed routing allocation, capacity, and training adaptation. Do not equate them with pure biological information contributions or assume additive AUC gains.

## Trace and verification

- Candidate generation clones one interacting full configuration and changes only source mode for the source controls. The guarded launcher applies the same rank, residual scale, home floor, tau, and shared-logit initialization. The source-mode flag consumes no additional random draws, so initial parameter inventories and prediction-head weights match.
- The data module builds the same pooled patient and outcome roster for every source arm. Fine identities are mapped separately from grouped cohort identifiers. Collation preserves the scalar fine index. Missing outcome records are represented by negative label/time/event values and excluded by the loss mask.
- Native BONSAI encoder inputs are clinical sequence fields; observed outcome labels are not inserted as routing query inputs. The target query is the identity of the outcome being predicted, not its observed value. The shared encoder is supervised by the full outcome panel in all four arms.
- Source mode masks disabled columns before the common softmax and disables that bank's live home-gradient corrections. Fine and target embeddings remain in the additive and interacting query branches. Enabled adapters modify `(patient, target outcome, hidden)` representations, which are projected by the corresponding outcome head.
- Both ordinary and gradient-cached training pass fine identity into the source readout. Gradient caching detaches encoder representations for replay, not live adapter parameters. Adam includes trainable source parameters in its head group.
- Anchored source parameters receive gradients only from home supervision. Unanchored source parameters receive gradients from borrowing targets too. Source support buffers and source-name indexing agree with the prediction rows. The returned audits establish nonzero outcome corrections and correctly disabled banks in the actual source runs.
- Horizon BCE masks unknown labels, normalizes observed patient/horizon losses, and averages endpoint-group losses. Related grades share a loss group denominator; they do not share adapter matrices or head weights by that fact alone.
- Validation and checkpoint selection use routed hazards converted to CIFs on a fixed label-derived selection inventory. The flawed residual diagnostic does not supply the selection AUC. Checkpoint reconstruction uses saved model-init settings; evaluation calls the same source readout and risk projection.

Tests exercise actual OPERA forward prediction with a controlled encoder, all four source masks, horizon loss, source/context gradients, and CIFs. In particular, disease-source-only has a live target-query gradient while its outcome source columns/parameter gradients are zero; the reciprocal property holds for outcome-source-only. Initial states match across source modes (including identical undefined/null-hazard buffers). A further test demonstrates that zero query embeddings alone leave identity-specific home-floor routing.

## Problems and remedies

| Finding | Status | Remedy and interpretation |
| --- | --- | --- |
| Short labels suggest all disease/outcome effects are removed | Confirmed interpretation problem | Label results “disease sources only” and “outcome sources only”; state that query context and joint training are held fixed. Existing fits remain valid for bank comparisons. |
| Zeroing a query embedding does not remove all explicit identity conditioning | Confirmed architectural property | For a learned-query ablation, mask that embedding in both query branches and clearly retain home floors. For fully identity-independent routing, remove the corresponding home-floor dependence too. These answer different questions. |
| A single softmax assigns more aggregate initial mass to a larger bank | Confirmed normalization property | At flat logits with 24/69 supported sources and current settings, disease bank mass changes from 96.2% alone to 28.0% with both banks. Inspect trained mass before attributing performance loss. Bank-level/within-bank normalization or count-adjusted logits are possible new designs, not retrospective corrections to these fits. |
| Weak outcome parameter updates and strong Adam epsilon attenuation | Confirmed by server checkpoint diagnostics | Test the already agreed borrowed-update rule and a matched adapter-only epsilon alternative. Connectivity tests do not establish sufficient optimization. |
| Anchored borrowing blocks borrowed-parameter co-adaptation | Intentional restriction | Unanchored comparison permits joint learning; named source provenance then changes. It is not a proof that every related outcome should improve performance. |
| Support shrinkage uses non-equivalent event-count units on the two axes | Confirmed counting convention | Disease counts sum across outcomes; grades can count the same patient repeatedly. Treat this as a support proxy rather than an estimated uncertainty/credibility weight. A revised count or uncertainty model is a separate methodological change. |
| Curriculum suppresses low-support targets early | Confirmed configuration behavior | The matched curriculum-off run activates all supported targets from epoch zero; no change to existing runs. At later fully activated epochs, this is a learning-history issue rather than current zero weighting. |
| Legacy residual metric mixes projection arithmetic | Confirmed diagnostic bug | Use the audit's same-projection and FP32 projected-correction measurements. A future logging fix must use a new/documented metric definition; do not silently reinterpret historical curves. |
| Route mass is interpreted as fraction of total representation/predictive benefit | Incorrect interpretation | The patient vector is always an additive baseline. Gate mass allocates correction weight; interpret source contribution through predictions and explicitly defined removals. |
| Source-removal export redistributes remaining gate weights | Confirmed diagnostic convention | Current removal renormalizes, measuring withdrawal plus redistribution. For a fixed-mixture contribution estimand, zero the correction while holding other weights fixed. Keep the two estimands distinct. |

No source-bank leakage or mismatch of outcome/source indices was found in the reviewed path. This is narrower than claiming the entire repository is bug-free or that the existing architecture optimizes both banks fairly.

## If explicit query-input effects are needed

Hold a nonempty source inventory, training roster, heads, objective, and optimizer fixed. A separate query-context factorial would compare patient-only, patient+disease, patient+outcome, and patient+disease+outcome learned queries. There is currently no configuration switch for these four explicit contexts. A correct implementation must mask inputs in both additive and interaction branches, preserve parameter/RNG inventories, and specify whether deterministic home-floor context is retained. Removing explicit disease context still leaves disease information in clinical patient histories; removing outcome-query context still leaves target-specific heads and multi-outcome supervision.

This second factorial is necessary only for a claim about the value of explicit routing context. It is not required to interpret the completed source-bank factorial, and it should not be introduced merely to rescue an underperforming outcome bank.
