# Mock peer review, 18 October 2026

Two simulated reviews of the manuscript as it stands, read against the results
we most plausibly expect. They are meant to find weaknesses before reviewers do.
Nothing here is a result.

## Assumed results (most likely, from what we have seen)

Based on the tuning runs (joint model, width 256, tuning macro AUC about 0.79,
up from about 0.73 at width 64), earlier DLBCL work and how these designs
usually behave:

- **Joint vs specialist:** positive overall, about +0.02 to +0.04 AUC. Most of
  the gain is in cells with few training events, and it is near zero in large
  cells (DLBCL, CLL, myeloma).
- **Disease pooling vs outcome pooling:** disease pooling adds more than
  outcome pooling, and the two are partly redundant (small negative
  interaction).
- **Joint vs per-task XGBoost:** a small gain overall, about +0.01 to +0.02
  AUC, larger in rare cells. XGBoost is on par in common cells. XGBoost given
  the quality-registry variables is on par with or better than the joint
  model for DLBCL treatment failure.
- **Calibration:** reasonable slopes, but calibration-in-the-large drifts in
  2023 for some domains (treatment patterns changed).
- **Against the IPI:** clearly better stratification in DLBCL, with more
  low-risk older patients and a sharper top 5%. Results are mixed for indices
  in small populations, with wide intervals.
- **Routing:** close to equivalent to the joint model.
- **National encoder:** little or no gain used directly, a small gain after
  local adaptation, and a clearer gain at 10 to 25% of training patients.

---

## Review 1: data scientist (clinical machine learning, foundation models for EHR)

**Summary.** A well-designed, unusually careful comparison of supervision
scopes (specialist, disease-pooled, outcome-pooled, joint), a router and
population pretraining for 69 outcomes in 24 lymphoid cancers. It uses
temporal validation, competing risks and paired inference. The matched design
is the main contribution. The headline gains are modest, and several choices
need justification.

**Major comments**

1. **Effect sizes and their meaning.** A pooled AUC gain of 0.01–0.02 over
   XGBoost is small. With about 1,000 evaluable cells, many correlated, a
   panel-level interval can exclude zero while no single prediction changes
   meaningfully. Translate the gains into decision-relevant quantities per
   cell type, such as events caught per 1,000 at fixed specificity, and show
   the distribution of cell-level gains with their own intervals, not only
   medians.
2. **Tuning set size.** All model selection, early stopping, cutoffs and the
   "best model per cell" strategy use 1,980 patients treated in 2022. For
   small subtypes this is a handful of events. How stable are the
   tuning-selected cutoffs and strategies? Please quantify the selection
   uncertainty, for example by resampling the tuning set, or the IPI
   comparison may be partly an artefact of a lucky cutoff.
3. **Single training run.** All intervals are conditional on one fitted model.
   Neural models vary between seeds by amounts comparable to the reported
   gains. At least the joint model, the headline specialists and XGBoost
   should be refit with additional seeds.
4. **Comparison with other foundation models.** The paper builds on
   CORE-BEHRT and compares with XGBoost, but not with another pretrained EHR
   model or simpler baselines (for example, the same transformer with a
   frozen encoder and linear probe). Without that, it is unclear whether the
   gain comes from pretraining, fine-tuning or the multi-task head.
5. **Pretraining overlap with test patients.** Test patients' pre-2022
   history was part of self-supervised pretraining. The authors argue no
   outcomes leak, which is right, but the encoder has seen these exact
   sequences. A sensitivity analysis pretraining without test patients, or
   evaluating on test patients absent from pretraining (new residents,
   patients with no pre-2022 records), would settle it.
6. **The national encoder comparison is confounded.** It differs in
   population, register deliveries, vocabulary, training budget and possibly
   width, as the authors state. That makes "does population pretraining
   help?" hard to answer. Equal-budget or equal-vocabulary controls, even for
   one configuration, would help.
7. **Outcome panel selection.** 69 of 87 candidate outcomes were selected
   with training and tuning event counts. Please show that excluded outcomes
   would not change conclusions, or report them as a sensitivity analysis.
8. **Router gate.** The router-null equivalence gate is a good idea. Report
   its result before any router claim.

**Minor**
- Report the number of parameters, training cost and inference time per
  model against the 1,750 specialists claimed to be replaced.
- IPCW with a censoring model fitted how? Report weights.
- Calibration recalibration is excluded from the primary analysis. Fine, but
  show how much a simple intercept update on tuning data would fix the 2023
  drift.

---

## Review 2: clinician (haematologist, lymphoma and CLL, prognostic research)

**Summary.** An ambitious national study that addresses a real clinical gap:
risk estimates for complications and rare subtypes at the start of treatment.
The comparison with the IPI is the most clinically interesting part. I am less
convinced about when and how the predictions would be used, and how much of
the gain matters to patients.

**Major comments**

1. **Prediction time.** Predictions are made at the start of first-line
   treatment, but the main decisions the IPI informs (regimen, number of
   cycles, CNS prophylaxis) are taken before treatment starts. What decision
   is the model meant to support at that moment? Supportive care and
   monitoring are plausible, but the paper should say so plainly and give
   examples with lead time.
2. **The IPI comparison uses the wrong endpoint.** The IPI was built for
   overall survival at five years. Comparing it on two-year treatment failure
   favours the model. The authors mention native-endpoint analyses, and these
   should be in the main text with the same weight.
3. **What the model knows that the IPI does not.** The model does not read
   stage, LDH or performance status, the core of the IPI, yet beats it. That
   is surprising and needs explanation. Is it reading proxies, for example
   recent admissions or lab tests that track tumour burden? Without some
   explanation, clinicians will not trust it.
4. **Outcome definitions.** Infection partly defined by antimicrobial
   prescriptions, and toxicity by laboratory thresholds, are proxies.
   Prophylactic antimicrobials, common in CLL and myeloma, could inflate
   "infection". Please report how prophylaxis was handled, and validate at
   least the main complication outcomes against chart review in a sample.
5. **Treatment changed during the test period.** New first-line regimens
   (polatuzumab-R-CHP, BTK inhibitors, venetoclax, daratumumab) change
   outcome risks. Results by regimen era, or at least by year, are needed.
6. **Rare subtypes.** The strongest argument is coverage for rare subtypes,
   but there the intervals will be widest. Show the absolute performance and
   calibration for the eight subtypes without an index, with patient and
   event counts, so a reader can judge whether these estimates are usable.
7. **Usefulness.** AUC and IPA do not tell a clinician whether acting on the
   model helps. Decision curves for one or two actionable complications
   (febrile neutropenia and G-CSF, infection and prophylaxis) would show
   clinical value directly.

**Minor**
- The "per 1,000 patients" framing is excellent. Use it throughout.
- The reclassification figure should show how many patients move between
  categories in both directions, not only survival per cell.
- State where in the clinical workflow the risk profile would be shown, and
  who acts on it.

---

## How we can answer both

| # | Concern | Response | Status |
|---|---|---|---|
| DS1, C7 | Small effects, clinical meaning | Per-1,000 and decision-point numbers per cell group; decision curves for G-CSF and infection prophylaxis | Partly in place (decision points); decision curves to do |
| DS2 | Tuning-set selection uncertainty | Resample the tuning set to re-select cutoffs and strategies; report the spread of the resulting test estimates | To do (inference only) |
| DS3 | Single training run | Two extra seeds for the joint model, headline specialists and XGBoost | Planned, lower priority |
| DS4 | Source of the gain | Linear probe on the frozen local encoder (cheap); random-init joint model already in the design | Linear probe to do |
| DS5 | Pretraining overlap | Evaluate on test patients with no pre-2022 records in pretraining | To do (subgroup, no retraining) |
| DS6 | National confounding | Already stated; reframed as "encoders as delivered" | Text in place |
| DS7 | Outcome selection | Report the 18 excluded outcomes descriptively with joint-model performance | To do |
| DS8 | Router gate first | Results text orders the gate first | Text change made |
| C1 | Prediction time | Intended use states supportive care and monitoring, not regimen choice; added why prediction is at treatment start | Text change made |
| C2 | IPI endpoint | Native-endpoint comparisons in main text alongside treatment failure | Text change made; analysis in place |
| C3 | What the model reads | Occlusion by source and time window; concept-level attributions for treatment failure | To do (inference only) |
| C4 | Outcome proxies | State prophylaxis handling; report infection with and without antimicrobial-only events | Check definitions; sensitivity to do |
| C5 | Treatment era | Results by calendar year already planned; add first-line regimen groups where recorded | Year in place; regimen to do |
| C6 | Rare subtypes | Supplementary table with absolute performance, calibration and counts for the eight subtypes without an index | Table to add |
| C-minor | Reclassification counts | Count grid added under the survival grid (Figure 6D) | Done |
