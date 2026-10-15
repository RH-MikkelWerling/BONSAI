# Agenda: 14 October 2026 — close the router method

## Objective

Follow-up decisions and executable supervision-factorial preparation are in
`docs/supervision_factorial_20261014.md`. The 256 pretraining is now running
with the collaborator specification. Final comparisons use a common full-tuning
horizon-BCE checkpoint selector; disease-only routing is an acceptable selected
model if outcome sources add no supported incremental value.
The current ordered pre-launch checks and reduced external-pretraining design
are in `docs/final_experiment_gates_20261014.md`; use that order before queuing
the full grid. Source mode and additive/interacting query are separate axes.

Freeze the remaining router decisions, then rebuild the essential paper fits
on the new local 256-wide encoder. No further open-ended architecture sweeps.
Pretraining can run now independently of the downstream router decisions.

## Decisions, in order

1. Archive the 64-wide scoreboard: completed epoch, selected checkpoint, seed,
   source mode, rank/query width, anchoring, curriculum and optimizer epsilon.
   Label unfinished fits explicitly; do not delete checkpoints or stop jobs
   automatically. Review the epsilon comparison and existing borrowing-update
   and curriculum comparisons. Select one common training policy; if evidence
   is inconclusive, explicitly record a provisional choice rather than claiming
   an improvement.
2. Confirm encoder specification: hidden size 256, four heads (64 dimensions
   per head), and assumed four layers. Check the collaborator resolved config
   for the other settings before describing the pretraining contrast as exact.
3. Approve the bounded capacity study in
   `opera/configs/hidden256_paper_run_plan.yaml`: rank8/query128 versus
   rank16/query128, followed by selected-rank/query512. Freeze all other
   choices. Use tuning only, equal budgets and the same encoder checkpoint.
   Prefer smaller capacity when differences are inconclusive; verify correction
   magnitudes and optimizer damping before declaring a capacity bottleneck.
4. Freeze the primary router and exact initialization checkpoint. Record a
   config/hash and allocate separate GPU runs for source arms and specialists.

## Why these are different knobs

Encoder produces h[256]. Each source adapter computes
256 -> rank -> GELU -> 256; the weighted corrections are added to h.
Rank8 is a conservative source-specific correction capacity, not an
eight-dimensional replacement of the patient representation. It was not
derived theoretically; the 64-wide rank16 comparison did not establish a gain.
At width256, rank8 already has four times the source-matrix parameters.

The interacting query concatenates projected patient, disease and target
vectors: 768 -> query_width -> GELU -> 256, added to the ordinary additive
query. It selects sources and is separate from adapter rank. Current query
width128 does not bottleneck the whole prediction path. Query512 is a single
capacity check, not an automatic scaling requirement.

## Execution order and GPU allocation

- Now: one GPU for fresh local256 pretraining. Default physical batch1,
  accumulation128, scaled Time2Vec, LR3e-4, 20 epochs with epoch10 checkpoint.
- After checkpoint audit: two GPUs can compare rank8 and rank16 concurrently.
  Then one GPU runs the larger-query condition at the selected rank.
- After method lock: four independent source-arm runs (both, disease only,
  outcome only, neither), one GPU each, or sequentially with fewer GPUs. Reuse
  the selected both-arm fit only if its complete config/budget matches.
- Shared grouped baseline: one independent fit unless verified redundant with
  the neither-arm implementation/protocol.
- Specialists: four disjoint disease-by-outcome shards, one GPU per shard or
  sequentially. ALL diseases, including CLL unless a matching256 CLL queue
  already exists. Do not reuse64 specialists as256 controls.
- Tabular work proceeds independently. External encoder fits follow the
  audited transfer plan once its final checkpoint/vocabulary are available.

## Return to the current comparison stage

Use the approved release, sealed metadata, unchanged label eligibility, and
same sequence/vocabulary contract. Materialize downstream configs from the
approved SERVER release, not legacy generated examples. Substitute the audited
new checkpoint and separate256 output roots. Planning overlays are not launch
configs until tomorrow's policy choices are filled and preflight succeeds.
The existing checkpoint override is OPERA_EXTERNAL_DAPT_CKPT; despite its name
it selects an audited encoder checkpoint. The new pretraining checkpoint will
be under the chosen run directory's version_0/best.ckpt.

Collect paired tuning predictions, source contrasts D-N/O-N/DO-D/DO-O, AUC and
Brier by disease/outcome/horizon with support counts. Freeze decisions before
held-out evaluation. Rebuild route distributions, normalized source-preference
atlases and embedding panels on the selected256 model. Gate weights are not
proof of beneficial transfer. Retain64 results as matched supplemental work;
changing per-head width alongside encoder width is not a width-only ablation.

## Completion gate

One recorded encoder checkpoint and one locked router policy/capacity; all
required comparisons queued in disjoint output directories; no held-out-driven
tuning and no additional exploratory grid without an explicit new decision.
