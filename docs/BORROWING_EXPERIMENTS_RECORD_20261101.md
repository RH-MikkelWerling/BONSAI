# How to borrow: experiment record, 26 October – 1 November 2026

One place for every router, borrowing and readout experiment run for Paper III
after the router lock: what each asked, why, what it showed on tuning data, and
what is still open. Formal definitions, contrasts and the selection rule are in
`docs/ANALYSIS_LOCK_DRAFT_20261015.md` (amendments 26 Oct – 1 Nov). Numbers here
are **training-log macro AUC at the selected checkpoint, single seed, tuning
data**: a sanity check, not the paired analysis. The test set is closed.

Reference points: joint model (J) **0.7711**; R_AB (primary router) **0.7836**;
R_AB seed 43 0.7834 (noise floor ≈ 0.0002–0.0005); own-only control 0.7832.

## The question

The joint model borrows everything: one representation for every disease and
outcome. Specialists borrow nothing. The paper asks how a model should borrow
in between. Our first answer was a router that lets each prediction borrow
corrections from other diseases' and outcomes' adapters.

## Stages and rationale

| Stage | Date | What | Why | Result (tuning, training logs) |
|---|---|---|---|---|
| 3–5 | 26–27 Oct | R_AB (global adapter, warm-up), own-outside variants, controls, IPW sampling | Original router collapsed onto one source | R_AB 0.7836; own-only 0.7832: the gain over J comes from **own corrections**, borrowing adds ~0 |
| 6 | 30 Oct | R_PP: partial-pooling router (own weight n/(n+τ), soft-normalised sources, learned temperature) | Make routing selective and auditable | Collapsed to one source per prediction (patient term dominated; concentration rewarded) |
| 7 | 31 Oct | R_PT5, R_PT3V: task-level routing, bounded logits; R_PP_FROZEN | Fix the collapse | Stable routes, but ~0.004 below R_AB: own corrections were capped (soft-normalised, ×w_own); τ fell, scale rose |
| 8 | 31 Oct–1 Nov | 25-fit sweep: own corrections exactly as R_O; only the borrowed part pooled. Routing variants, hierarchy over lineage/endpoint groups (R_PH), learned groups (R_PL8), live sources, rank 32, 10% data arm | Every design contains the own-only model; test every reasonable way to borrow | All within ~0.001 of R_AB (routing 0.7830, hierarchy 0.7835, learned groups 0.7838, rank 32 0.7825). At 10% data: hierarchy +0.0013, learned groups +0.0023 over the floor |
| 9 | 1 Nov | Task-conditioned pooling: outcome (or disease × outcome) queries attend over the encoder's token states; with and without own corrections; tied outcome queries; lineage-tied diagnosis terms | Every correction so far acts on one summary vector made before the outcome is known | Reading alone: +0.003 (outcome) to +0.009 (diagnosis × outcome tied to lineage, 0.7800) over J, still below R_AB. Gates settled at ~0.07 (max ~0.15), gradients healthy (no damping). Own-correction variants: pending |
| 10 | 1 Nov | J and R_AB with the encoder fine-tuned at 1.0× (and R_AB at 0.3×) the head's learning rate, instead of 0.1× | All readouts hit the same ceiling: is the representation the limit? | Pending |
| 11 | 1 Nov | J_FPF_PH_OWN_PH and J_OA_OWN with the encoder at 1.0× | Reading attends over token states, so a harder-trained encoder may matter more for the reading models than for J and R_AB | Pending |

## What the toys showed (scratchpad `router_toy/`, 3 seeds each)

- **One outcome, 20 diseases in lineages.** Uncapped own corrections beat the
  capped R_PT design; routing added 0–0.016 on data-poor diseases; known
  lineage groups added +0.10 when data were scarce. Free rein (logit bound 10)
  collapsed 92–97% of tasks onto one source with no gain; the patient gate
  overfitted.
- **Multi-outcome (12 diseases, 9 outcomes in families), checkpoint chosen on
  validation.** Hierarchy best or tied best in every scenario; R_AB-style strong
  with ample data, weakest when scarce.
- **Why routing adds little:** an oracle router with routes fixed to the true
  relatives still fell short of the hierarchy. Routing mixes other tasks'
  separately fitted corrections (their quirks included); a hierarchy fits one
  shared correction on all members' data. Penalising the own correction or
  cross-fitting the router made routes cleaner but not more accurate.
- **Token-level toy (shared summary = mean of tokens, lossy by design).**
  Outcome attention + own corrections far ahead (0.869 vs 0.814 hard cells);
  the size did not transfer to real data, where the trained summary already
  carries most of the signal. Attention heads specialise; head-averaged maps
  hide it.

## What we now think

1. **Sharing a representation and adding small task corrections captures the
   gain.** How tasks share beyond that (routing, hierarchy, learned groups,
   capacity) no longer matters at full data: every variant sits at R_AB's level.
2. **Task-conditioned reading helps, modestly,** and more when it is specific
   to the diagnosis and tied to the lineage. Diseases do read the record
   somewhat differently (this contradicts the toy-based "outcomes read, diseases
   adjust" asymmetry; do not claim it).
3. **Known structure helps where parameters are data-hungry:** in the
   per-diagnosis reading (+0.002) and with 10% of the data, not in the
   corrections at full data.
4. **The ceiling looks like the representation.** Hence stage 10, and the
   national arm. National pretraining changes the representation in two ways:
   more learned from far more records, and **finer lab-value resolution**:
   lab bins are set from each lab code's training counts, so national data give
   more lab codes their own tokens and more bins per code. This bears directly
   on XGBoost's likely edge (exact recent values in fixed windows).
5. **Clinical parallel (for the Discussion / thesis):** the prognostic indices
   (IPI, FLIPI, MIPI, CLL-IPI) combine largely the same measurements with
   disease-specific weights and cut-offs. Molecular markers (TP53, del(17p),
   IGHV, MYC) are in the record via Patobank, but only where tested.

## Failures and their fixes (for the methods supplement and our own practice)

| What | How it showed | Fix |
|---|---|---|
| Original router: 90% weight on one outcome's adapter within one epoch | Collapse | Global adapter, warm-up, bounded logits |
| Own adapters inside the mix grew ~20× | Router parked weight on small sources | Own corrections outside the mix (R_O) |
| R_PP: patient term at full size, unbounded logits | Collapse after warm-up | Task-level routing, bounded logits (R_PT) |
| Zero task queries normalised: ~1/eps gradients | fp16 overflow, every step skipped, silently | Random query init, eps; regression test |
| R_PT: own corrections capped and shrunk | ~0.004 below R_AB, silently | Own corrections exactly as R_O (stage 8); every design contains the own-only model |
| Learned latent groups never separated | Symmetry: identical groups under warm-up | Group routing live from step 0, bound 10 |
| Own-corrections module helper named `_apply` | Every own-correction fit failed when moved to GPU | Renamed; device/dtype-move test for every stage-9 model |

Practice adopted: integration test of every config through the real model
code; device-move test; validation logs of correction sizes; toy checks with
checkpoint selection; the GPU smoke test (`opera.run.router_smoke`) for new
model code.

## Figures (if a stage-9 design is selected)

Pooling ladder (specialists → disease-/outcome-specific → joint → + reading →
+ adjustment, by cell support); per-outcome specialisation (attention gate vs
events, by endpoint family); outcome reading map (query similarity); what each
outcome reads (per head or attribution, by concept and time); disease deviation
(own-correction size vs patients). Router maps are retired.

## Open

- Stage 9 own-correction fits (J_OA_OWN, J_OA_LR_OWN, J_OA_OWN_PH, J_FPF_OWN,
  J_FPF_PH_OWN_PH): do reading and own corrections add up?
- Stage 10: does a harder-trained encoder lift every model?
- 10% arm: J_frac10 is not in the local bundle (built with the national arm);
  compare the 10% designs with each other until it exists.
- Selection by the locked rule, then predictions, paired analysis, audits and
  figures for the selected design; the OPERA expansion ("One Pretrained Encoder,
  Read Adaptively") only if a reading design is selected.
