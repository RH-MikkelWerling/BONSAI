# Transition to an architecture-matched 256-wide encoder

Next decision meeting: `docs/agenda_20261014_router_lock.md`.
Downstream planning overlays: `opera/configs/hidden256_paper_run_plan.yaml`.
These are not yet executable downstream configs: checkpoint and the remaining
router policy decisions must be filled after review.

## Decision and remaining confirmation

Prepare a fresh local-only 256-wide, scaled-Time2Vec encoder at learning rate
3e-4 and effective batch 128. Do not launch until the collaborators supply their
resolved configuration. Current encoder: width 64, four layers, four heads
(16 dimensions per head). Width 256 with four heads means 64 dimensions per
head; width 256 with 64 heads means four dimensions per head. These are not
interchangeable. Head count and layer count are deliberately missing in the
new candidate configuration.

Also confirm causal objective, sequence length, time units/scaling, vocabulary
and value encoding, dropout, normalization/FFN settings, optimizer epsilon,
weight decay, scheduler, epoch/token budget, windowing and target-loss policy.
The candidate inherits local settings for all unconfirmed fields, including
20 epochs, epsilon 1e-6 and sequence length 8192; it is not yet certified as an
exact collaborator match. Learning rate here is the pretraining rate, not a
blanket instruction to change downstream learning rates.

## Pretraining preparation

Use `scripts/run_pretrain_hidden256.sh preflight` after exporting confirmed
`BONSAI_256_HEADS` and `BONSAI_256_LAYERS`. Review the resolved config, then set
`BONSAI_256_ARCHITECTURE_CONFIRMED=1` and use `launch` inside one GPU allocation.
Default physical batch is one and accumulation 128. Benchmark throughput and
memory before committing the full fit; increase `BONSAI_256_BATCH_SIZE` to a
divisor of 128 when safe. Effective batch is 128 on ONE GPU; multi-GPU training
requires accounting for world size. This wrapper does not submit jobs.

Use a new output directory and no 64-wide initialization. The inherited trainer
can auto-resume its own checkpoint; never change architecture in an existing
run directory. External-pretrained initialization uses the separate audited
transfer/DAPT workflow, not this local-only launcher. Source architecture and
vocabulary must be verified before importing external weights.

## Close the 64-wide stage

Do not terminate jobs automatically. Inventory finished and unfinished fits,
preserve last/best checkpoints, resolved configs, provenance and tuning metrics.
Finish the narrowly scoped epsilon comparison where practical; avoid starting
more 64-wide specialist shards once transitioning the paper's primary encoder.
Keep finished specialists as a matched 64-wide comparison, not controls for a
256-wide router. Incomplete fits must be labelled incomplete, not compared as
full-budget results. Table width, layers/heads, rank, query width, source mode,
anchoring, epsilon, curriculum, seed, completed epoch and tuning score.

## Bounded router capacity check

Source matrices already take encoder width dynamically; patient queries,
source keys and disease/outcome embeddings also follow encoder width. The
interacting-query MLP hidden dimension remains separately configurable (128
currently). Adapter rank remains separately configurable (8 currently).

On the selected 256-wide local encoder, use three tuning-only conditions with
identical source mode, anchoring, scale, shrinkage, epsilon, curriculum, seed
and training budget:

| Condition | Adapter rank | Query MLP width | Contrast |
| --- | ---: | ---: | --- |
| Reference | 8 | 128 | Existing method on the larger backbone |
| Larger correction | 16 | 128 | Rank effect |
| Larger query | selected rank | 512 | Query-capacity effect versus its 128-wide counterpart |

The third condition follows rank selection; 512 preserves the old 2x query
hidden/encoder-width ratio but is a candidate, not a theoretical requirement.
No automatic rank-32 sweep. Source matrices contain 2*S*H*r parameters, so
rank 8 at H=256 already has four times the matrix parameters at H=64. Fixed
initialization scales mean initial correction magnitudes need not be identical
across widths: audit actual FP32 projected corrections, gate concentration and
Adam epsilon factors before interpreting performance as capacity alone.

## Required paper fits after method lock

1. Local-only matched 256 pretraining, then shared multi-outcome baseline and
   selected router with identical downstream protocol.
2. Source-bank factorial on that encoder: neither, disease only, outcome only,
   both. Queries stay patient/disease/outcome-conditioned in all router arms;
   these isolate source-bank adaptation, not the availability of raw identity
   information. Check whether the shared baseline and neither arm are genuinely
   identical before reusing one fit.
3. Independent disease-by-outcome specialists from the SAME selected encoder
   checkpoint, sharded over GPUs; retain support/estimability classification.
4. XGBoost/tabular baselines on identical patients, eligibility and splits.
   Width-independent data preparation need not be repeated.
5. External pretrained 256 encoder, and external-plus-local-DAPT where the
   paper claims that distinction: shared baseline and locked router. Run the
   full source factorial and matched specialists on the selected final external
   encoder if making those claims there. Do not automatically repeat every
   exploratory sweep on every pretraining branch.
6. Replicated seeds for the primary neural comparisons, within available budget.
   If unavailable, explicitly report training-seed uncertainty as unmeasured.

## Analysis and figures

Select architecture/checkpoints on tuning only; keep held-out evaluation closed
until the method and comparisons are frozen. Use paired patient-resampled
contrasts with consistent endpoint-group aggregation, AUC and Brier score;
report disease/outcome/horizon support and negative transfer, including CLL.
Report the source-bank contrasts D-N, O-N, DO-D and DO-O. Router gate mass is
not predictive contribution: accompany borrowing summaries with normalized
within-bank preferences, per-patient distributions and ablation performance.
Recompute embedding/PCA/UMAP and route atlases for the actual selected encoder;
do not compare independently fitted projection coordinates as aligned spaces.

The old 128 config exists, but completed results have not been verified locally.
Width comparisons are capacity-only ONLY when heads/layers/time encoding/data
and training protocol are matched. If these change to match collaborators,
label comparisons as architecture changes, not isolated hidden-size effects.
Summarize parameter count, token budget, GPU time and inference cost alongside
performance. Existing 64-wide work remains useful supplemental evidence.
