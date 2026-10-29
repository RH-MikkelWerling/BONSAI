# XGBoost versus sequential models: evaluation audit, 29 October 2026

The current paper evaluator applies the same metric definitions and paired
evaluation rules to both model families. The downloaded tuning tables are
internally consistent. This is not a completed independent audit of the server's
raw predictions, feature extraction or checkpoint provenance.

## What is shared

- Paper tabular jobs inherit the specialist bundle's population, fine disease,
  outcome, competing-death file, eligibility and ascertainment gates
  (`opera/run/paper_tabular_queue.py`). The paper training command uses
  `--known_status_cif --selection_metric log_loss --tune`.
- Canonical CIF evaluation restores the complete tabular risk set. A missing
  prediction is permitted only for an early administratively censored patient,
  whose contribution has zero IPCW weight. The placeholder probability is not
  scored as a zero-risk control. Missing known-status predictions cause an error
  (`opera/run/paper_analysis_inventory.py`).
- Before paired estimation, exact patient IDs, labels, data contracts, IPCW
  weights and follow-up as seen at the horizon must agree. The evaluator does
  not silently intersect different patient sets. Competing death is a known
  non-case for cumulative incidence, identically for both families.
- `opera/evaluation/secondary_analysis.py` uses one implementation of weighted
  AUC, Brier, log loss and IPA. IPCW Brier/log loss use the full risk-set
  denominator. The support decision and patient bootstrap draws are shared
  across models. Production support defaults to five cases and five controls.
- Primary X/XP feature preparation removes outcome/bookkeeping columns and
  registry/adverse-event predictors unavailable to sequence models. XR is a
  distinct registry-augmented reference arm. This preparation rule does not
  itself prove that every upstream aggregation window ends before treatment.

## What the downloaded aggregates establish

Bundle: `scratch/paper_reading_20261028/paper_review_download_20261028_182155`.
Compared `prediction_analysis/estimates.csv` with `contrasts.csv` directly,
using the original outcome IDs and contrast IDs rather than plot labels.

For each of R_AB_L, R_L and J_L versus X, all 53,058 matched metric rows have
identical `n`, `n_cases` and `n_controls`. Equal counts alone are not proof of
identical patient IDs. The evaluator's stricter identity checks are the code
contract, which still needs provenance confirmation on the server.

For each contrast below, 17,686 AUC/Brier rows were joined to their component
estimates, including unsupported rows. There were 6,008 finite differences
across the two metrics, representing 3,004 supported cell–horizon combinations.

| Saved contrast | Models | Mean supported AUC difference | Maximum AUC/Brier subtraction error |
|---|---|---:|---:|
| O1 | OPERA − X | −0.0280318304 | 2.71e−16 |
| T2 | J_L − X | −0.0367108826 | 3.05e−16 |
| T7 | XP − X | +0.0115248647 | 3.19e−16 |

AUC gain is candidate minus reference. Brier gain is reference minus candidate,
so a positive gain means better prediction for either metric. The current
tabular advantage is present in the estimates, not introduced by a reversed
plot label or subtraction. No metric estimates were changed by this audit.

## Limits on interpretation

These are comparisons of fitted prediction systems, not a controlled test of
architecture alone. Tabular features aggregate history, while sequence models
use tokenised and potentially truncated histories. XGBoost fits and tunes each
disease–outcome–horizon task separately. Shared neural models learn across tasks
and select a shared checkpoint. Their information representations, loss
weighting and tuning opportunities differ intentionally.

Tuning performance is selected performance. Both families use this year during
development, and XGBoost's task-specific selection provides many opportunities
for selection optimism. This does not establish that its observed lead is an
artifact, nor that held-out ordering will reverse.

Five cases and five controls are an estimability rule, not a guarantee of a
precise individual-cell AUC. Aggregate paired results should be accompanied by
coverage and the existing stricter-support sensitivity analyses. Compare models
on matched cells rather than subtracting absolute means from different rosters.

Patient-bootstrap intervals are conditional on selected fitted models and fixed
estimated censoring weights. They do not include the entire uncertainty from
hyperparameter/model selection, training randomness or estimating censoring
weights. IPCW also relies on appropriate censoring assumptions. These limitations
apply to both model families.

## Remaining server checks

1. Inspect the actual `analysis_inventory/analysis.yaml` and saved
   `prediction_analysis/analysis_manifest.json`: the primary estimand must be
   `cif_ipcw`, with the intended support/aggregation/bootstrap settings. Inventory
   and launcher defaults retain a historical known-status option, so defaults
   alone are not evidence of the setting used for the downloaded run.
2. Recompute X, XP, J_L and OPERA AUC/Brier from the canonical predictions and
   compare against the saved estimates. `audit_tuning_router` now accepts
   `--recompute-metrics --metric-models X XP J_L OPERA`. Its original default
   reconciliation and the router replay remain neural-only. Missing requested
   model artifacts cause an error. Reported numerical discrepancies require
   review even if the command exits successfully.
3. Verify actual tabular training metadata (`binary_known_status_cif`, selection
   split/metric, backend, selected parameters), completion hashes, feature
   profiles and the existing membership/index-date alignment report. Confirm
   upstream lookback cutoffs. The local download deliberately lacks the raw
   features and predictions needed to certify these independently.

The older `compare_direct_cr_tabular.py` utility uses an inner patient join.
It is not the strict paired paper evaluator and should not supply primary
paper comparisons without its own population reconciliation.

## Local verification

Targeted tests exercise competing-death labels, censoring, paired metrics,
canonical tabular inventories, tuning, feature/job contracts and caching.
The initial XGBoost tests could not load macOS OpenMP. Using the already installed
PyTorch `libomp.dylib` through a process-local `DYLD_FALLBACK_LIBRARY_PATH`
resolved that dependency without changing project or system packages.

The end-to-end queue test then exposed a stale assertion: ordinary cache reuse
returns `reused_fit`, while `--verify-outputs` returns `reused_verified_fit`.
The test now explicitly checks both paths. Runtime cache behavior is unchanged.
The additional reconciliation test covers an early-censored placeholder,
competing death, an intentionally corrupted saved metric and a missing requested
model. Final verification: **96 tests passed**, with one scikit-learn
deprecation warning. `git diff --check` passed. No model was retrained on the
study data and no test-set results were opened.
