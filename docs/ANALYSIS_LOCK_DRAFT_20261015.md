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
- Sensitivity: known-status unweighted estimand; excluding the 209 patients
  whose disease was assigned after day 14 (late-assigned subtypes);
  size-matched clinical cutoffs.
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

## Source removal: which diseases the router borrows from (added 2026-10-22, design revised the same day)
- Within the locked router (R_L) only, no retraining and no comparison with a
  separately fitted model: a fit-to-fit comparison carries the noise floor
  (~0.004 AUC on tuning), as large as the per-source effects.
- The readout is a sum of routed corrections, one per source disease, and the
  risk head is linear, so a source is dropped by removing its term; no other
  weight changes (no renormalisation, which would inflate the other sources).
- One evaluation writes every variant: full; none (own disease only, no
  borrowing); in:X (own disease and X); out:X (every source but X).
- Effects, AUC points on the IPCW estimand, one paired patient bootstrap shared
  by all variants: total borrowing = full - none; source alone (leave-one-in,
  primary) = in:X - none; source left out (leave-one-out) = full - out:X. The
  two are the extremes of the source's Shapley value. Reading: both above 0,
  main source; alone only, redundant with related sources; left out only,
  useful in combination.
- Pooling per target, source and clinical domain (the paper's 12 groups):
  estimable cells only (>= 5 cases and >= 5 controls), each outcome equal and
  each horizon equal within an outcome; "All outcomes" pools every outcome the
  same way.
- Map (descriptive, main figure): source alone with adaptive shrinkage
  (Stephens 2017; zero-centred normal scale mixture fitted per clinical domain
  over all off-diagonal pairs). Lineage is not used in the pooling, so a
  lineage pattern cannot be produced by it. Dots: pooled 95% interval above 0
  (on synthetic data the unpooled intervals marked about half false dots, the
  pooled ones almost none). Unshrunk estimates and the source left out kept
  alongside; every pair shown. Targets with few estimable cells are labelled
  grey and do not set the colour scale.
- Context (supplementary figure): three levels per target, same bootstrap:
  routing (R_L - J_L), borrowing as trained (R_L - RDH_L), dependence (full -
  none), the last also against disease size.
- Primary borrowing claim (added 2026-10-22): the prespecified pairs as a set.
  Statistic: mean unshrunk source-alone gain of the pairs minus the mean of the
  other off-diagonal pairs of their clinical domains; null: 10,000 random draws
  of the same number of pairs from the same domains (one-sided permutation).
  The same for the negative controls (expected: no difference). IPA computed
  the same way as a secondary metric; three levels also over all diseases
  (each disease equal).
- Main figure 7, kept simple: map (A); each disease's total gain from
  borrowing (dependence) on the same rows, all diseases dashed (B); the expected
  pairs and controls against the typical pair (mean of the other pairs of the
  same domains), test in the caption (C); total gain by clinical domain, all
  diseases equal (D). Supplement: the
  three levels per disease, by disease size, pairs alone vs left out, all 12
  domain maps by group. Disease groups order the map for reading only; no
  claim that borrowing follows lineage. `borrowing_lineage.csv` (within vs
  across groups per domain, paired intervals) is descriptive and not reported.
- Named claims: only the pairs in `opera/configs/borrowing_hypotheses.yaml`
  (clinically motivated and literature-referenced, graded strong / moderate /
  exploratory, with three negative controls), on the unshrunk source-alone
  estimate, with the prespecified claim labels and a 0.25 AUC point margin
  (`margin_points` in the hypotheses file; the 1 point model-comparison margin
  would call a real contribution of a few tenths "equivalent"); the source left
  out reported with each. Fixed before the test set is opened.
  DRAFT pending haematologist review.
- Interpretation: the fitted model's use of a source, not biology.
- Code: `run_paper_experiments.sh source-removal --split <split>` (one GPU
  evaluation, `predictions/<R_L run>/<split>_source_removal/`), then
  `borrowing-map --split <split>` (`opera/run/source_removal.py`; needs the
  ordinary `evaluate` predictions of J_L and RDH_L on the same split for the
  routing and trained-borrowing levels).

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
patient bootstrap, and paired differences OPERA-X, J_L-S_L and R_L-J_L (OPERA-X
  replaces J_L-X on 2026-10-22 so the text's 'OPERA' matches; the dp90 macros
  must follow OPERA's arm once it is selected).
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

## Label savings (added 2026-10-22)
- LS10/25/50: joint model (cohort encoder) on 10/25/50% of training patients vs
  specialists on all of them; LSE10/25/50: the same from the adapted national
  encoder; FD10/25/50: adapted national joint model on a share vs the cohort
  joint model on all patients (the "matched the full-data model with X%"
  claim; matched = claim label equivalent or better). Contrasts in
  `opera/configs/manifests/final_contrasts_20261014.yaml` (label_savings).

## Borrowing: mean ablation (added 2026-10-22)
- Switched-off sources are replaced by their tuning-set mean term per target
  disease and outcome (mean ablation; primary); zero ablation is a
  sensitivity. Reason: on tuning the router kept almost no weight on the shared
  slot (0.05, from 0.037 at initialisation) and spread the rest nearly evenly
  (route entropy 3.18 of a maximum ln 27 = 3.30; effective residual 0.66
  log-hazard), so zeroing borrowed terms would remove most of the correction
  mass and confound lost information with distortion. Means use tuning
  patients only; diseases without tuning patients use the all-patient mean.
- Implemented in `evaluate_direct_competing_risk --borrowing-design` (both
  ablations written, `source_removal/{mean,zero}/`) and
  `opera.run.source_removal --ablation {mean,zero}`.

## Router sensitivity fit (added 2026-10-22, before stage 2)
- control_RSI (alias RSI_L): the locked router configuration with
  `adapter_shared_logit_init` = 3.0 (code default; the locked runs use 0.0;
  about 15-18% starting weight on the shared slot with 94 sources, not most of it),
  otherwise identical; one fit, stage 2, router_controls phase. Contrast RS1 =
  gain(RSI_L, R_L), descriptive; route entropy and shared weight reported for
  both. Motivation: tuning route entropy 3.18 of max ln 27 = 3.30 and shared
  weight 0.05, i.e. near-uniform routing. Decided after seeing these tuning
  diagnostics and before any test-set result.

## Router optimizer fix and rerun of the screens (2026-10-23)
- The routing parameters (keys, queries, query interaction, shared logit) were
  trained with Adam epsilon 1e-6, which damped their updates (gradients mostly
  below epsilon; router_fit_audit). Every router parameter now uses epsilon
  1e-10 (`adapter_epsilon_scope: router`); source, rank and query screens were
  rerun from scratch before the router was locked and before any test result.
- The rerun chain selected: source bank both, rank 8, interacting query width
  512 (router_decisions.json, router_lock.json). The earlier damped chain
  (disease_only, rank 8, additive) is archived and not reported as a result.
- Added RBH (both banks, own adapters only, no borrowing) and contrast
  C6 = gain(R_L, RBH_L): the borrowing of the locked router itself.

## Initialisation rule (added 2026-10-25, before the stage-2 fits finished and before any test data were opened)
Seen at the time: training-time validation metrics of the unfinished RSI fit
(shared slot starting at logit 3.0, about 15-18% of the routing weight with 94
sources, against about 1% at logit 0) showing sharp routing (route entropy 4.17
to 0.39, shared weight to ~0) and a lower validation loss than R mid-training.
No tuning-panel AUC/IPA for R or RSI and no test results had been seen.
1. R_L (shared logit 0) stays the primary model unless RSI_L beats it on the
   tuning panel (router-decision weighting) by more than the noise floor in AUC
   and passes the IPA guard.
2. If so, one further seed of each (R and RSI) is trained; the shared-slot
   start of 3.0 becomes primary only if its advantage exceeds the noise floor
   in both seeds. Otherwise R stays primary.
3. The model not chosen is reported as the sensitivity analysis (RS1).
4. Whichever is primary, routing selectivity is reported for both: route
   entropy, effective number of sources (exp entropy), shared weight, and the
   share of weight on own versus borrowed sources (route export on tuning).

## Decision curves (added 2026-10-23)
- Action: closer monitoring and supportive care for patients above the
  threshold; outcome: death within two years in DLBCL; range 5-40%.
- Prespecified operating points: the model's low- and high-risk cutoffs from the
  risk-group rule (risk_matched_low: low = largest low-risk group with observed
  risk no higher than the IPI low-risk group; high = the IPI high-risk group's
  specificity), both fixed on tuning data. Net benefit there is written to
  `decision_curve_operating_points.csv` and marked on the figure.
