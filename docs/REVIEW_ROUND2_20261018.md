# Review round 2, 18 October 2026: critique, answer, re-critique

Aim: shore the paper up for a high-impact venue. Each item gives the critique,
our answer, what a reviewer would say to that answer, and where it lands
(done, flag for decision, or text).

## 1. "The specialist is a strawman" (data scientist; strongest remaining critique)
- **Critique.** Fine-tuning a whole transformer on one cell with 20 events
  overfits, so of course pooling wins. A fair specialist would not fine-tune
  the encoder.
- **Answer.** Early stopping on tuning loss limits overfitting, and per-task
  XGBoost is a strong non-neural specialist. The frozen-encoder joint control
  (P7) shows what fine-tuning adds.
- **Re-critique.** That still compares a fine-tuned specialist with a
  fine-tuned joint model. The cleanest answer is a per-cell probe: the frozen
  pretrained encoder's patient embedding with a ridge-penalised logistic
  regression per cell and horizon. It is cheap: embed every patient once,
  then fit small linear models.
- **Lands as: FLAG.** Recommended. It is a new baseline row, the
  "specialist done right", and it pre-empts the critique entirely.

## 2. Rare subtypes: "is it better than what we have now?"
- **Critique.** The coverage argument rests on the rarest subtypes, where
  intervals are widest.
- **Answer.** Two prespecified strata (fewer than 500 training patients;
  subtypes without an index) report every comparison, including XGBoost and
  XGBoost pooled across diseases. Pooled XGBoost is the strongest tabular
  answer to "just pool". There is also a Table 2 column and a Results
  subsection.
- **Re-critique.** "What we have now" for these patients is often the IPI
  applied off-label. The pooled-IPI comparison across rare lymphomas covers
  that where the IPI is recorded.
- **Lands as: DONE** (strata, Table 2 column, Results subsection, numbers
  wired). Optional: "the related common disease's model" as a further row
  (inference only). It is a nice-to-have, now that pooled XGBoost covers
  most of it.

## 3. Temporal shift as a strength
- **Critique.** Performance may degrade over the test years.
- **Answer.** The test period spans changing first-line treatment and register
  coverage. Performance and calibration are reported by calendar year
  (prespecified), so stability becomes evidence of robustness.
- **Re-critique.** Say what changed, with dates, or the claim is vague.
- **Lands as: DONE** (Results subsection, year stratification in the config,
  Discussion), with **one input needed**: which first-line changes happened in
  Denmark and when.

## 4. A stricter infection endpoint: a 70th outcome?
- **Critique.** Infection partly defined by antimicrobial prescriptions is a
  proxy.
- **Answer.** Prophylactic agents are excluded. A stricter label (diagnosed
  infection, positive microbiology or hospital antimicrobials) can be
  evaluated with the existing infection predictions, without training a new
  target.
- **Re-critique.** If it is not trained, does it show anything? Yes: if the
  model ranks the stricter label nearly as well, the proxy is not driving
  performance. That is exactly the question a reviewer asks.
- **Lands as: FLAG (recommend evaluation only, not a 70th outcome).** A 70th
  training target adds a near-duplicate infection endpoint, changes the panel
  and loss weights, and needs retraining, for little gain. As an evaluation
  label it costs nothing.

## 5. Explainability: how important is it here?
- **Critique (clinician).** How can it beat the IPI without stage, LDH or
  performance status?
- **Answer.** The paper's claims are about how to learn (sharing, pooling,
  pretraining), not about mechanisms, so explainability is not central. One
  compact, intuitive analysis is enough to pre-empt the "black box"
  objection.
- **Proposal (lighter than full occlusion).** One table: one-year AUC of the
  joint model for three domains (death, infection, cytopenia) when the input
  is restricted. The restrictions are diagnoses only, add prescriptions, add
  laboratory results, all sources, and the last year of history only. The
  reader sees what each source adds, and the model is not retrained.
  Integrated gradients is left to a follow-up paper.
- **Lands as: FLAG.** Choose between this table, nothing, or the full design.

## 6. "What does the output look like?"
- **Critique (both).** Neither reviewer can picture the tool in use.
- **Answer.** An illustrative risk profile: one row of outcomes, with
  predicted two-year risk against the population average, for a synthetic
  example patient or for averaged archetypes (for example, a fit older
  patient with DLBCL rated high-risk by the IPI). No real patient data.
- **Lands as: FLAG.** A strong teaser panel for Figure 1 or the Discussion. It
  is a design choice.

## 7. Framing contingency
- **Critique.** If XGBoost matches the joint model overall, the headline
  falls.
- **Answer.** The paper stands on three legs: coverage (one model for every
  cell, including cells no specialist can be trained for), equivalence or
  gain against tuned tabular models, and the controlled decomposition of
  what sharing adds. The text is direction-neutral and filled by the numbers
  step, so only the abstract's verbs and the conclusion depend on the
  direction.
- **Lands as: TEXT** (already neutral). Decide the headline after the results.

## 8. Reproducibility without data
- **Critique.** Code without data or weights cannot be checked.
- **Answer.** The pipeline already runs on synthetic data (the figure
  prototypes and the tests). A released synthetic demo dataset, with one
  command that runs analysis, tables, numbers and figures, lets anyone
  execute the full pipeline.
- **Lands as: FLAG (recommended).** Small effort, high value for the review.

## 9. Smaller items fixed in this round
- "1,750 models" and "more than 10,000 tabular fits" were upper bounds for a
  full panel. They are now counted from the outputs (`narrow_fits`), and the
  tabular claim is reworded.
- Table 2 gains a rare-subtype column.
- Figure S7 marks the cells whose interval excludes zero.
- The prespecified strata are described in Methods.

## Subsequent review update

The earlier suggestion in point 5 to defer integrated gradients is superseded.
IG is now implemented with numerical, reference-sensitivity, randomisation and
perturbation checks plus reusable patient-level exports. Computational scope is
bounded without dropping those checks. The later claim/evidence audit and
remaining priorities are in `MANUSCRIPT_REVIEW_20261018.md`; citation-specific
corrections are in `REFERENCE_AUDIT_20261018.md`.
