# What similar router arrows mean — 29 October 2026

The user reported nearly identical correction arrows across calibration-map
outcomes and subsequently supplied the five-outcome figure. The displayed
arrow fields do look very similar. This is a visual observation, not a
numerical comparison of the original vectors. This note concerns the arrow
interpretation. The separate evaluation-gradient fix does not change arrows.

## What the arrows currently measure

`evaluate_direct_competing_risk.py` exports the shared pooled encoder vector
and `adapted[:, target_index]`, where the requested outcome supplies the
index. `build_paper_embedding_tables.py` checks the outcome marker, split,
patient alignment and equality of the exported shared vector with the map's
encoder embedding. It applies one fitted scaler/PCA/UMAP pipeline to each
outcome's adapted vectors. `figure_calibration_maps` bins the resulting
displacements and uses a common arrow scale across outcome columns.

The selected `fix_R_AB` fit in the reviewed server report has both source
banks enabled, interacting queries, home floor 0.05 and a global adapter.
Its displayed displacement includes the entire readout correction. For this
architecture, schematically:

`adapted_o = shared + scale * (global(shared) + sum_s weight_o,s * adapter_s(shared))`.

The global term is exactly shared across outcomes. Source adapter vectors are
also reused across target outcomes, with outcome-dependent mixing weights.
Consequently, similar total displacements are possible even with distinct
outcome queries. The arrows do not isolate borrowed sources, source-specific
benefit or calibration improvement. A more accurate figure label would be
“Projected readout correction”. This label can be updated after the current
run without new inference.

## Why different outcomes can look alike

- A large common global term can dominate the full correction. This is a
  hypothesis about magnitude until the actual vectors are decomposed.
- The reviewed router has diffuse allocation. Similar weighted averages of
  the same source vectors can produce similar target corrections. Entropy
  alone does not establish that target weights or outputs are identical.
- A two-dimensional projection and local averaging can hide genuine
  differences in the original hidden dimensions. UMAP is fitted on the
  shared encoder vectors, so adapted vectors may also move outside the
  distribution used to fit the map.
- Each outcome has different risk-head weights. An identical hidden-vector
  change can raise one outcome's predicted risk and lower another's. Arrow
  direction has no inherent low-risk-to-high-risk meaning.

High-risk regions can overlap because outcomes share prognostic information,
but overlapping risk surfaces do not establish why the arrow fields look
similar. The model is not explicitly trained to move patients toward a
high-risk area of a UMAP plot.

## Comparing outcomes with different high-risk regions

This is a useful visual diagnostic. If risk-coloured maps differ while arrow
fields remain similar, the arrows cannot be read as movement toward a universal
high-risk location. The supplied figure's first row shows observed event status,
however, rather than predicted-risk surfaces. Event locations alone do not
identify the model's high-risk regions, particularly for rare outcomes.

Similar arrows remain compatible with a dominant global correction, similar
routed mixtures, or different corrections whose differences disappear in the
projection. Even a precisely identical correction can have opposite effects
through different outcome-head weights. Accordingly, distinct risk regions
plus similar arrows do not by themselves establish reliance on the global
adapter.

The next useful comparison is shared encoder, global-only correction and
routed source correction from the same frozen checkpoint. Measure their
original-dimensional vectors and their effects through each outcome's head,
then compare the resulting horizon-specific risks. Vector norm dominance and
predictive contribution are separate measurements. Log-hazard contributions
can be decomposed through the linear head, but cumulative incidence is
nonlinear and also depends on competing hazards. Report fixed-component
removal effects as diagnostics without treating them as additive risk changes.
Keep the separately trained global-only control distinct from this
within-checkpoint decomposition.

## Checks possible after the full rerun

The new per-outcome `routed_patient_vectors_INTERNAL.npz` exports remain on
the server and contain the necessary shared/adapted vectors. A CPU audit can
compare aligned per-patient deltas across outcomes, reporting pairwise cosine
similarity, norms, exact-array equality and the amount of outcome-specific
variation after subtracting a per-patient mean correction. That mean is a
diagnostic common component, not an exact isolation of the global adapter.

Compare original-dimensional deltas with PCA and UMAP views, and pair them
with within-checkpoint predicted-risk changes before versus after the
readout. Exact own/global/borrowed decomposition requires applying the saved
readout to those shared vectors and fine-disease labels, but does not require
re-encoding patient histories. The separately trained joint model is a
different comparator from the router's encoder/head with adapters omitted.

Similarity already present in the raw deltas supports a shared-correction
explanation. Similarity arising mainly after projection points toward the
visualisation. Identical raw arrays across supposedly different target
exports require checking file provenance and outcome indexing. Neither the
current screenshot report nor the existing replay audit resolves all three.
