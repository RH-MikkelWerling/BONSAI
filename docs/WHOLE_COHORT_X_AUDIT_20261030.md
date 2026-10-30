# Whole-cohort aggregation and XGBoost audit, 30 October 2026

## Status

The server screenshot records stage transitions, not metric values. It shows
`whole_cohort` finishing before `integrated_gradients`. No current whole-cohort
server result tables were found locally during this audit. Consequently this is
a code and statistical-definition audit, not confirmation of the suspicious
empirical X estimate.

No arithmetic error was found in the pooled AUC implementation. A coverage
reporting omission was fixed. Additional aggregate diagnostics now expose
population exclusions and the contribution of cross-disease ranking. Neither
predictions nor the evaluation estimand were changed. Nothing was trained and
no held-out data were read.

## What X and XP actually mean

`opera/run/paper_analysis_inventory.py` registers **X** from separate XGBoost
fits for each disease, outcome and horizon. Its whole-cohort prediction vector
is a portfolio assembled from those fits. It is not a single XGBoost fitted to
all patients. **XP** is the disease-pooled XGBoost comparator for each outcome
and horizon, subsequently exported into fine-disease cells for evaluation.

Both are legitimate comparators if labelled accurately. XP is the direct
tabular comparator to a whole-cohort outcome prediction model. Retaining X
also tests the portfolio of specialists. These answer different questions.

The training code can record a tabular fit as not estimable when training has
one class or tuning lacks the classes needed for AUC selection. Such cells have
no registered X predictions. Pooling cannot recover predictions from an
unfitted model. In particular, this missingness can select diseases differently
for different outcomes and horizons, including through tuning event support.

## Calculation checked

1. Native sequence CIFs and canonical tabular probabilities enter the same
   normalization and metric functions. Event 1 by the horizon is a case.
   Competing deaths by the horizon are controls. Early administrative censoring
   has zero weight.
2. Administrative censoring weights are estimated within fine disease,
   outcome, horizon, model and fitted seed. They are preserved when disease
   labels are subsequently collapsed for pooled metrics.
3. For the shared comparison, the patient roster is intersected across all
   requested models for that outcome/horizon. Censoring weights are then
   re-estimated on that shared roster, still within fine diseases. Patient,
   label, follow-up-at-horizon, data-contract and weight agreement are checked
   by the paired evaluator.
4. Outcomes and horizons remain separate. Pooled AUC is calculated from patient
   scores, not by averaging disease AUCs. Brier score uses the full risk-set
   denominator, including early-censored patients. IPA uses the pooled outcome
   reference risk, which differs from disease-specific IPA.
5. A tabular probability placeholder for an early-censored patient has zero
   IPCW weight and cannot affect AUC or losses. That patient still belongs in
   the full risk-set Brier denominator. Placeholders for known-status patients
   are rejected.
6. Uncertainty uses paired patient multiplier draws across methods and tasks,
   conditional on fitted models and estimated IPCW. It does not refit the
   censoring curves in every draw or capture training variability.

## Why the pooled X estimate can look different

The whole-cohort AUC compares cases and controls from **different diseases**
as well as from the same disease. Disease-specific score offsets do not change
a model's within-disease AUC but can substantially change the pooled AUC.
Differences in score scale across specialist fits matter in that calculation.
Differences in genuine disease-level baseline risk matter too. A decomposition
alone does not establish that either is an implementation defect.

A regression counterexample gives both models perfect AUC within both diseases.
A monotone score shift in one disease changes pooled AUC from 1.00 to 0.19.
In that fixture 82% of weighted case/control pairs cross disease boundaries.
This is synthetic evidence about the metric, not a result of OPERA.

The existing `available_population_estimates.csv` evaluates each method on
its own available population. Those rows must not be subtracted across models
to claim a paired advantage. `common_population_estimates.csv` and
`paired_contrasts.csv` use the common roster. Even this common roster is the
intersection of **all four** requested methods, not necessarily the complete
cohort or the largest roster available for an individual pair. Presence of XP
does not fill missing X fits.

## Reporting fix and new outputs

Previously `model_coverage.csv` iterated only over existing model/task rows.
If X had no predictions for an entire outcome/horizon, that missing row was
absent and no shared estimate could be produced. This could conceal why an
outcome disappeared. Coverage now includes absent model/task combinations,
union and common population fractions, missing disease names and explicit
availability status. The numerical metric estimates themselves are unchanged.

New outputs, for both available and common rosters:

- `population_by_disease.csv` records patient and event/control support,
  zero-weight rows, placeholder counts, known-status probability ranges and
  unique-score counts, and descriptive within-disease AUC. Small-cell point
  estimates here are diagnostics, not replacements for the supported primary
  analysis.
- `auc_pair_composition.csv` records pooled AUC, within-disease pair AUC,
  between-disease pair AUC and their weighted pair fractions. The fractions
  reconstruct the pooled AUC exactly. This is a point-estimate identity,
  not a causal explanation or a new significance test.

The parallel-audit ZIP allowlist now includes these two reports. It still
excludes individual predictions and patient metadata.

## Required empirical checks

For the suspicious X row, identify its outcome, horizon and population table.
Then compare all four models' common patient and case/control counts. Check
missing disease fits, score distributions and within/between AUC components.
Also reconcile X against the cross-family metric audit. A successful workflow
stage is not confirmation of these checks.

Do not describe an X estimate as covering the complete cohort if its coverage
report shows missing patients or diseases. Do not infer superior within-disease
discrimination merely from a larger pooled AUC. No automatic recalibration or
replacement of missing specialist predictions was introduced.

## CPU-only server refresh

The running rehearsal can continue. To generate these diagnostics after
syncing `opera/run/evaluate_whole_cohort_outcomes.py`, run this in the repository
on an allocation with CPU capacity. This reads existing predictions and writes
a fresh sibling directory. No GPU inference, source-removal rerun or model
training is involved.

```bash
opera_whole_root=data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123
opera_whole_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_whole_audit="$opera_whole_root/whole_cohort_audit_$opera_whole_stamp"
mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv-v100/bin/python -u -m opera.run.evaluate_whole_cohort_outcomes \
  --analysis-config "$opera_whole_root/analysis/analysis_inventory/analysis.yaml" \
  --models OPERA J_L X XP --split tuning \
  --horizons 30 60 90 180 365 730 --replicates 1000 \
  --output-dir "$opera_whole_audit" \
  > "logs/whole_cohort_audit_$opera_whole_stamp.log" 2>&1 &

echo "$opera_whole_audit"
```

When complete, that specific `whole_cohort_audit_*` directory contains only
the aggregate CSV reports and its manifest and can be returned for review.
The existing `whole_cohort` directory is already sufficient to inspect the
initial estimates and coverage without waiting for this refresh.

## Verification

Focused tests cover pooled case-mix effects, missing-model coverage, identity
and follow-up parity, disease-conditional censoring, exact weighted AUC
decomposition against brute-force pairs with ties and competing death,
zero-weight placeholders, and agreement of native sequence versus
horizon-truncated tabular exports. Existing workflow, metric and queue tests
also passed. In total, 93 relevant tests passed across the focused and dependent
suites. Current server estimates remain pending local inspection.
