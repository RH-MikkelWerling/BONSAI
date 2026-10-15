# Clinical scores on their intended endpoints

This is a distinct required benchmark, not merely a risk-group illustration.
For every available score/version/population, verify its development endpoint,
index and intended horizon from the original publication. Register the exact
mapping to our endpoint and document deviations. A two-year evaluation cannot
be described as reproducing a score's original longer-horizon validation.
Treatment-start evaluation may also differ from a diagnosis-time intended use.
An endpoint missing from the current panel requires deliberate construction,
training and evaluation; do not substitute treatment failure for survival or
progression without documenting a different question.

Compare the original score with XGBoost, logistic regression and prespecified
sequence models on identical score-available, endpoint-eligible patients.
Keep full-cohort machine-learning results separate so missing-score selection
is visible. Freeze model selection and any mappings on development data.
Report common patient/event counts, availability timing, follow-up and paired
uncertainty. Use all-cause survival estimators for death endpoints and competing
risk evaluation for nonfatal endpoints where relevant.

Two information comparisons are useful: score-component-only tabular models
versus the fixed score, and richer-history tabular/sequence models versus the
score. Record which score variables are actually available to each model.
A richer-history gain is a pipeline/information benefit, not architecture alone.

Raw score rankings support discrimination comparisons. They are not predicted
probabilities: never insert a raw IPI score into Brier or calibration calculations.
Use an applicable published mapping or a development-fitted, held-out-frozen
mapping. Report its origin. Risk groups retain original definitions; probability
mapping is separate from redefining those groups. Report both native ranking
and mapped-risk performance when applicable.

Required numeric table: score/version, disease/population, original/evaluated
endpoint, original/evaluated index and horizon, deviation rationale, model/input
set, common N/events, score availability, follow-up, discrimination/paired effect
and interval, mapping provenance, Brier/calibration when valid, seed summary.
Main text should show a compact intended-endpoint comparison; complete score
families and secondary endpoint extensions remain supplementary.

Existing `clinical_score_panels.example.yaml` plots one model per score and
currently illustrates treatment failure. It does not implement this multi-model
intended-endpoint benchmark. The contract below must be resolved before writing
the dedicated evaluator; current plots are not evidence that this analysis ran.
