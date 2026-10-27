# Router and embedding figure audit — 27 October 2026

## Confirmed in source and local tests

The paper evaluator obtains contextual token states through the complete
transformer (`get_objective_representations` -> `_pool` -> `_encode_hidden`).
With `cls_last` it selects the last valid prediction-token state after that
encoding. This state can attend to all preceding allowed history; it is not
the raw CLS lookup embedding. Source adapters transform that patient state
before the outcome-specific linear head computes hazards. Checkpoint loading
uses saved architecture/model-init metadata and a strict state-dict load.

A new test using the real causal BONSAI transformer changes an ordinary event
code, leaves CLS/timing fixed, and confirms the contextual state and hazards
change. It also matches evaluator hazards against the explicit adapter-plus-head
calculation. This verifies the implementation, not the identity/health of a
particular server checkpoint.

Standard `patient_embeddings.npz` exports are pooled encoder states BEFORE
source adapters. A routed model has outcome-specific adapted states, so no
single shared post-router patient embedding exists. The optional routed-outcome
export is required to inspect that transformation. The rehearsal configuration
also deliberately supplies just the local fine-tuned stage; a one-column map is
not evidence that other representation stages were evaluated.

## Concrete embedding failure fixed

Production CSV reads preserve `subject_id` as strings, while Parquet exports
can retain integer IDs. The gallery and patient-map joins previously matched
no patients and silently drew blank panels with 0.0% incidence. Both plotting
boundaries now canonicalize IDs, validate uniqueness/completeness, and reject
an entirely unmatched join. Builders normalize IDs too, including metadata joins.
Patients with no horizon events/eligibility are explicitly labelled. Last pages
size themselves for the number of panels actually present, and disease-domain
labels use short names to reduce collisions.

Metadata embedding comparisons now default to individual coloured patient
points rather than interpolated means. Disease contours are off by default;
explicit `metadata_style: smooth` retains the previous continuous-field option.
Coordinates are unchanged by these display changes. Synthetic patient-map and
comparison previews were inspected; real-data visual review remains necessary.

## Why the AUC comparisons must be reconciled on the server

- R_L is an analysis alias. Unless `OPERA_PAPER_ROUTED_MODEL=R_AB_L` was applied
  when creating the inventory, it can still refer to the originally locked
  final-source router. Changing the env later does not rewrite an inventory.
- Route usage, source-removal maps and contrast figures can come from separate
  invocations; each must identify its source fit.
- The current routing forest shows medians/IQR over cells; the decision code
  averages equally by disease, then outcome, then horizon and requires at least
  five cases/controls. `effect_summaries.csv` also has equal-cell means. These
  do not necessarily agree, even for the same predictions.
- One documented +0.0138 AUC gain is R_AB versus RBH_AB (matched no-borrowing
  control), not R_AB versus J_L. This is a candidate explanation, not an
  identification of the result Mikkel recalls.
- 0.01 raw AUC equals one AUC point. Compare identical horizons, populations,
  estimands, seeds and weights before interpreting a discrepancy.
- Adapter-bank controls R2–R7 are the original bank factorial, not automatically
  matched controls for the later fixed global-adapter router. The explicit
  `router_borrow_AB` contrast is the matched R_AB–RBH_AB comparison.

## Read-only server command

Sync the updated source and checksum manifest. From the server repo, use the
EXACT environment file that generated the figures (change the example if needed):

```bash
python -m opera.run.audit_tuning_router --env-file .env.rehearsal \
  --route-diagnostics --verify-artifacts > logs/rehearsal/router_audit_current.json
```

The command does not infer, train, select a model, or overwrite analysis outputs.
It prints model-alias-to-fit mappings, readout settings, recorded evaluation
inputs and config hashes; optional verification hashes the prediction and
checkpoint files (potentially slow). It checks route-export directory identity,
reports route concentration/support-weighted health from existing exports,
includes source-removal provenance when present, and juxtaposes AUC mean/median
and decision weighting on current contrasts. It reports embedding join counts
without patient identifiers. Support-weighted health is the stage-3 rule; it
is not the stage-4 correction-magnitude-weighted health criterion.

Return the JSON, particularly R_L/R_AB/R_ORIG fit names, hash results, route
provenance, health and R1/router_borrow_AB summaries. Do not rerun training or
replace the primary model until those results identify the actual discrepancy.
If artifact mappings are wrong, rebuild inventory/analysis in a fresh root
with the intended alias; do not silently edit existing aggregate result files.
If embedding join mismatches are confirmed, the existing coordinate/outcome
tables can be rerendered with the fixed plotting code without fresh GPU work.

## Display preference clarified later on 27 October

Keep disease contours and smooth metadata fields as defaults. Clip colour fields
with a patient-cloud footprint, density and effective-support masks; do not fill
empty square corners or bridge unoccupied gaps between patient islands. This
supersedes the earlier paragraph choosing individual-point colouring by default.
The footprint is a display boundary, not an inferential confidence region.
It is applied to metadata, gallery risk, patient-map risk/residual and calibration
maps. The existing coordinate and prediction tables can be reused for rendering.

## Follow-up on borrowing and usage screenshots

The source-removal design operates on the **disease-source terms only**.
Its `none` variant retains own disease terms, the entire outcome bank, global
corrections and the base encoder/head. Primary removed terms are replaced by
their tuning means. Thus the dependence panel is cross-disease dependence, not
all-borrowing dependence. Labels now say so. The separate verification replay's
zero-borrowing intervention removes both other-disease and other-outcome gated
terms, so its differences are not the same estimand.

Borrowing-map `effect` is an adaptively shrunken posterior mean in AUC points;
`raw_effect` retains the original paired point estimate. Total-dependence and
trained-control panels are unshrunk. A nearly zero shrunken pairwise map can
coexist with measurable total dependence, especially since source effects are
not additive in AUC. Upstream pooled_effects applies x100 once; figure renderers
do not apply another factor. Tiny colorbar ticks are not evidence of a missing
x100 conversion. The map labels now state shrinkage explicitly. Both borrowing
renderers accept `pair_estimate: raw` for an unshrunk sensitivity view, and reject
duplicate target/source/domain rows instead of pivoting them into an average.
Unshrunk pair intervals do not generate significance dots for hundreds of pairs.
The CPU audit now reports raw versus shrunken magnitude summaries when available.

Usage heatmaps now normalize within each disease/outcome task before averaging,
so tasks assigning greater total mass to a bank no longer dominate its displayed
within-bank shares. This changes the weighting, not the underlying saved gates.
The renderer rejects multiple models/seeds/splits, duplicate source rows, invalid
weights and unmapped targets. It no longer fails on an empty off-diagonal scale.

Panel D now shows effective borrowed-source count (exp entropy) calculated from
**mean gates**, excluding shared and own-disease/own-outcome slots. This is not
mean patient-level effective source count; mean gates can conceal individual
specialization. Its IQR describes heterogeneity across disease summaries, not a
confidence interval. No borrowed mass gives undefined breadth, not a fictitious
one-source result. Panel C remains gate mass by bank, not correction magnitude.

Embedding arrows were enlarged by 50% in the automatic common scale target
(6% to 9% of map span), with slightly wider arrowheads and the paper's routed
model color. All outcomes still share a single labelled magnification factor.
Synthetic style proofs are in `scratch/router_review/`; no real results replaced.
