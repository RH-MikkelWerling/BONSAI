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
  reported with their own intervals. Headline differences and Table 2 pool all
  six horizons (each disease-outcome-horizon cell equal; `horizon_days = 0` in
  the summaries); the per-horizon table is a supplement, and one year is shown
  in the main figures (revised 2026-10-21, before the test set was opened).

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
- Tuning choices: source bank, rank, query and OPERA (applied by
  `opera.run.router_decision`; OPERA by the same rule over the final arms).
  - Rule: the **best option** on tuning data. Highest panel AUC, where the
    panel averages the AUC difference over every estimable cell (>= 5 cases
    and >= 5 controls), each disease equal, each outcome equal within a
    disease and all six horizons equal within an outcome. Full-risk-set IPCW
    estimand; one paired patient bootstrap (1,000 replicates, seed 2026,
    shared draws) for the reported intervals.
  - Ties: options within the noise floor of the best count as tied, and the
    simplest is chosen. Noise floor = |R0|, the panel AUC difference between
    the router with every source switched off and J (the same model,
    initialised differently).
  - Calibration guard: an option whose panel IPA is worse than the simplest
    option's by more than the IPA noise floor is not eligible.
  - Simplest first: sources disease_only, outcome_only, both; rank 8, 16, 32;
    query additive 128, interacting 128, interacting 512; OPERA: joint before
    router, cohort before adapted national before national encoder.
  - R0 is also a machinery check (|R0| <= 0.005 expected), reported, not used
    to choose.
  - The 0.01 margin is the practical margin for test-set claims only, not for
    tuning choices: these are hyperparameters, and the claims are tested on
    patients the choices never saw.
  - History: on 2026-10-20 the screens required an interval above zero and a
    gain above max(0.01, |R0|) to leave a default. On 2026-10-21 this was
    replaced by the rule above, after rough source-screen point estimates had
    been seen (disease_only and both within about 0.001, outcome_only lower).
    The source-bank choice therefore could depend on the noise floor; its
    consequence is limited because stage 2 trains all four banks with the
    locked rank and query and compares them on the test set. Rank, query and
    OPERA results had not been seen.
  - The tuning panel sees mainly the common diseases (rare cells are rarely
    estimable on 1,980 patients), so it cannot show gains where data are
    scarcest.
- Label per contrast and horizon (revised 2026-10-21, before the test set was
  opened), from the paired bootstrap of the pooled difference: **better** if
  the 95% interval lies above 0 (**worse** if below), additionally flagged
  "above margin" when the estimate exceeds the practical margin; **equivalent**
  if the interval lies within ±margin; otherwise **inconclusive**. Each pooled
  contrast also reports a two-sided paired bootstrap p-value (twice the smaller
  share of draws on either side of zero). The earlier rule (better only if
  P(gain > margin) ≥ 0.95) labelled a precisely estimated gain below the margin
  as equivalent, conflating statistical and practical significance. The
  hierarchical model's posterior probabilities of benefit, harm and practical
  equivalence are reported alongside, descriptively.
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
