# Analysis lock — DRAFT (not locked)

Fixed before any held-out evaluation. When agreed, these values are written into
`final_contrasts_20261014.yaml`, `bayesian_contrasts.example.yaml` and
`locked_prediction_analysis.example.yaml`, and this file is renamed
`ANALYSIS_LOCK_<date>.md` and committed before `evaluate --split held_out`.

## Estimand and data
- Cumulative incidence at each horizon with censoring weights (`cif_ipcw`);
  death from other causes is a non-event (competing risk).
- Test set: patients treated from 2023, evaluated once. Tuning set (2022) used for
  every choice. Training through 2021.
- Sensitivity: known-status unweighted estimand; excluding late-assigned
  subtypes; size-matched clinical cutoffs.
- Stratum without a dedicated prognostic index (no multivariable index derived
  for the entity and externally validated; borrowed or family-level indices do
  not count): RT, tFL, LBL, NMZL, HCL, ALCL, PCL, solitary plasmacytoma, BCL and
  TCL unspecified (paper supplement, `tab:index-availability`). Sensitivity:
  without NMZL (counting the MZL-IPI); with AITL and BL added (requiring routine,
  guideline-recommended use).

## Metrics
- **Primary: IPCW AUC** at each horizon (cumulative cases vs controls at the
  horizon, censoring-weighted), on the `cif_ipcw` estimand.
- **Key secondary (accuracy): IPA** = 1 − Brier / Brier of the observed event rate, censoring-
  weighted. Same ranking as the Brier score within a cell; comparable across
  outcomes with different event rates, so suited to pooled summaries. Brier
  differences reported alongside.
- **Calibration:** slope and calibration-in-the-large (descriptive).
- **Horizons:** all six (30, 60, 90, 180, 365, 730 days) are prespecified and
  reported with their own intervals; one year is shown in the main figures.

## Unit, uncertainty and weighting
- Unit: one disease × outcome cell; models compared on identical patients.
- Paired patient bootstrap within disease, **1,000 replicates**, seed 2026, the
  same draws for every model, outcome and horizon.
- Panel summaries: every disease weighs equally; within a disease every loss
  endpoint (grade variants of one finding together; 52 endpoints) weighs
  equally. Sensitivity: equal weight per clinical domain.
- Estimability: ≥5 cases and ≥5 controls per cell, ≥80% finite bootstrap draws;
  other cells are shown but not pooled.

## Claims
- Practical margin: ±0.01 AUC, ±0.01 IPA.
- Router-null gate (R0, no-source router vs joint): interval within ±0.005 AUC
  before source-bank effects (R2–R7) are interpreted. With every source masked
  the router's correction is exactly zero, so R0 is the joint model with a
  different initialisation: it checks the routing machinery and measures the
  difference between two fits of the same model (the noise floor). It does not
  test capacity; RDH and ROH (own adapter, no borrowing) do.
- Router decisions on tuning data (added 2026-10-20, before any source-screen
  fit had finished). Paired patient bootstrap on the tuning patients, 1,000
  replicates, seed 2026, same draws for every fit; panel one-year AUC and Brier
  differences weighted as for the panel summaries. A candidate replaces the
  default (disease_only; rank 8; interacting query, width 128) only if the 95%
  interval of its AUC gain excludes 0, the point gain exceeds both 0.01 and
  |R0|, and its Brier score is not worse. Otherwise the default, or the simpler
  option, is kept. Outcome sources (outcome_only, both) must pass this against
  disease_only.
- If the R0 gate fails, the router proceeds with the defaults throughout, the
  source-bank contrasts are reported as not interpretable, and |R0| is reported
  as the noise floor for every single-seed contrast.
- Label per contrast and horizon: **better** if the pooled 95% interval excludes
  0 and P(gain > margin) ≥ 0.95; **equivalent** if the interval lies within
  ±margin; otherwise **inconclusive**.
- Primary contrasts: P5 joint vs specialist, P1 disease pooling, P2 outcome
  pooling, T2 joint vs XGBoost, E1 national vs local. Others secondary;
  C1–C5 explanatory.
- No p-value correction: interval estimates, hierarchical shrinkage, every cell
  reported.

## Bayesian layer (very weak priors)
| Metric | Mean effect | Between-cell SD | Support slope |
| --- | --- | --- | --- |
| AUC | Normal(0, 0.1) | Half-normal(0.1) | Normal(0, 0.05) |
| IPA | Normal(0, 0.2) | Half-normal(0.2) | Normal(0, 0.1) |
| Brier | Normal(0, 0.02) | Half-normal(0.02) | Normal(0, 0.01) |

Prior-only run checked before fitting; the priors cover effects several times
larger than any plausible gain, so the data dominate.

## Decision points (secondary, descriptive)
Thresholds are fixed per model, disease, outcome and horizon on that model's
tuning predictions and applied unchanged to test patients with known status:
tuning event rate (equal alert rate across models), Youden, maximum MCC, and
specificity 0.80, 0.90 and 0.95. Reported: threshold, TP/FP/FN/TN, sensitivity,
specificity, PPV, NPV, MCC and flagged share, with 95% intervals from the shared
patient bootstrap, and paired differences J_L-X, J_L-S_L and R_L-J_L.
Exported confusion counts below 5 are suppressed; full counts stay internal.
No decision-point metric is a primary or key secondary endpoint.

## Clinical scores
Each index on its own endpoint (mostly overall survival, 2 years or the longest horizon within median follow-up); every index with ≥50 patients
per subtype, each index pooled across subtypes, NCCN-IPI and computed IPI;
cutoffs `risk_matched_low` on tuning data; top 5%/10% observed risk reported.

## Open
- [x] IPCW AUC primary; IPA key secondary
- [ ] ±0.01 margins
- [ ] outcome for treatment failure and `manuscript.index` in the score config
