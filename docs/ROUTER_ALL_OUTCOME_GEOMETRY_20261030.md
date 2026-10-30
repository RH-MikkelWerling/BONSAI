# Distinct outcomes and the shared router correction

30 October 2026. Tuning diagnostics only. The existing rehearsal and its
mechanism audit are unchanged. No empirical results from the new audit are
available locally yet.

## What the screenshot establishes

The selected path ends in `79d39ad1e7cb51472026/best.ckpt`, the revised fit
recorded in our current router review. The report covers 1,980 patients,
256-dimensional corrections and five outcomes: anaemia, creatinine increase,
hospitalisation, infection/antimicrobial use, and overall survival.

The reported `outcome_specific_energy_fraction` is 0.0154482, or 1.54%. It
subtracts each patient's average correction across those five outcomes.
The remaining 98.46% is energy of the common within-patient component. These
are squared-vector-magnitude quantities, not fractions of predictive benefit,
patient variance explained, or an identified global-adapter contribution.

Median total correction norms are approximately 4.68 to 4.98. Together with
the high pairwise cosines, this suggests similar magnitudes as well as common
directions for this particular five-outcome set. These outcomes could share
physiology, treatment burden or general vulnerability. The screenshot does
not establish which clinical factor caused the common correction.

## What should change in the diagnostic

The five-outcome audit cannot show whether the same pattern holds across the
full endpoint panel. Analyse every fitted outcome before selecting additional
illustrations. Inspect targets from different clinical domains, such as later
treatment, lymphocyte changes, thrombotic events and electrolyte abnormalities,
when the fitted panel and support permit. No particular pair is guaranteed
to have a distinct risk pattern.

In the present gallery, patients share one coordinate map. What differs is
their outcome risk colouring, not a separately estimated embedding for each
outcome. Independently refitted UMAP coordinates would introduce arbitrary
rotations and distortions and should not be used to measure correction
similarity. Use the same patients and shared map for any new panels.

For each outcome pair, compare model-predicted patient-risk rankings using
Spearman correlation at the same horizon on patients eligible for both
outcomes. This is a reproducible diagnostic of different predicted risk
patterns. It is not observed event concordance, biological independence or
evidence of predictive accuracy. Different event prevalences alone need not
produce different patient rankings.

Then compare corrections before and after removing the **exact global
adapter**. Here `routed` means the complete weighted own-and-borrowed source
mixture. It does not mean borrowed terms alone. The frozen mechanism audit
separates own and borrowed components further.

## New CPU-only sidecar

`opera.run.audit_router_geometry` reuses the selected checkpoint head and
saved pooled encoder vectors. It computes every fitted outcome's correction
on CPU, validates the global/source decomposition against the production
readout, and validates reconstructed probabilities against the saved tuning
predictions. It does not repeat the transformer encoder or the expensive
bootstrap analyses. It does not train or recalibrate a model.

Outputs:

- `report.json`: checkpoint/reconstruction, total and source-only outcome
  residual energy, global energy, and the global/source cross term.
- `outcome_geometry.csv`: per-outcome total/global/source median norms,
  global-source direction alignment and eligible patient counts.
- `outcome_pairs.csv`: every pair's common eligibility, predicted-risk rank
  correlation, total correction cosine and source-only correction cosine.
- `different_risk_patterns.csv`: the 20 least-correlated predicted-risk pairs
  as candidates for graphical inspection. The full table remains available.
- `router_geometry_review.zip`: exactly the four aggregate reports above.

The global and source vectors can reinforce or cancel each other. Their
squared norms therefore do not give additive contribution percentages.
The report includes the cross term to make that explicit. Zero corrections
produce undefined directions, not invented cosine values. Pairs need at
least 20 common patients by default. This is a geometry-support setting,
not an event-count rule or a guarantee of clinical precision.

## Server command

Sync the new `opera/run/audit_router_geometry.py`. Its existing dependencies
are the same frozen-head/audit modules used by the parallel mechanism job.
Run from the repository root on the CPU allocation, preferably once the
currently running CPU job has released its cores.

```bash
opera_full_root=data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123
opera_geometry_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_geometry_root="$opera_full_root/router_geometry_$opera_geometry_stamp"
opera_geometry_log="logs/router_geometry_$opera_geometry_stamp.log"
mkdir -p logs

nohup env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  .venv-v100/bin/python -u -m opera.run.audit_router_geometry \
  --analysis-root "$opera_full_root/analysis" \
  --output-dir "$opera_geometry_root" \
  --horizon 365 --threads 4 \
  > "$opera_geometry_log" 2>&1 &

echo "Geometry report: $opera_geometry_root/router_geometry_review.zip"
tail -f "$opera_geometry_log"
```

Return `router_geometry_review.zip` with the mechanism ZIP. No new GPU
allocation or additional outcome-by-outcome encoder export is needed. The
existing vector export contains the shared patient representation required
to reconstruct every target's readout.

## How to interpret the two audits together

1. **Total corrections similar, source-only corrections different.** The
   global term helps explain similar arrows. Whether it helps prediction
   requires the mechanism audit's global removal and joint comparisons.
2. **Source-only corrections also similar across distinct risk patterns.**
   The weighted adapter bank itself supplies a common transformation. This
   can be useful, but does not establish selective borrowing by target.
3. **Small source vectors, substantial head/prediction effects.** Representation
   magnitude understates the effect because the prediction head is sensitive
   to those directions. Norm dominance is not performance dominance.
4. **Replacing patient-specific gates changes little.** Within this frozen
   fit, individual allocation may not explain the gain. This does not show
   that a separately trained static model would achieve identical results.

Global and source corrections are one part of the comparison with joint.
The encoder and head were also optimised differently. Use the mechanism
audit's full/encoder/global-path comparison with joint alongside component
removal. A frozen-path intervention measures reliance of the fitted model,
not how another architecture would have trained.

## Local verification

Twelve tests passed across the new geometry audit and existing raw-vector
audit. They include production-head replay, batching independence, exact
global/source separation, opposing predicted-risk rankings hidden by a common
vector direction, cancellation, eligibility, zero corrections and aggregate
ZIP contents. Tests use constructed models and do not substitute for empirical
server reconstruction or performance checks.
