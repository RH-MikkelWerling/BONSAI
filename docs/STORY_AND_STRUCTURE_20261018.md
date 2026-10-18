# Story and structure, 18 October 2026

## The paper in one sentence
Clinical risk prediction in haematology does not have to be built one disease
and one outcome at a time: one model, learning across 24 lymphoid cancers, 69
outcomes and the whole Danish population, predicts the risks that dedicated
tools cover, at least as well, and reaches the many they do not, while relying
on information clinicians recognise. (Final wording depends on the locked
results; every claim below has a fair fallback.)

## Why it can be a top-tier paper
- Scope: every adult treated for a lymphoid cancer in a country, 69 clinically
  defined outcomes with competing death, six horizons.
- Design: a matched decomposition of where the gain comes from (other diseases,
  other outcomes, the population), not just "a big model is better".
- Benchmarks: the strongest practical comparators (XGBoost tuned per
  prediction, with and without registry variables; every recorded index on its
  own endpoint).
- Evaluation: locked, once, on a temporally held-out test set that spans a
  change in first-line treatment.
- Mechanism: the attributions show why sharing does not blur the predictions
  (one history, read differently for each complication).
- Clinical form: an explanation per patient and per risk, in the format a tool
  could show.

## Storyline of the Results (eight steps)
| # | Section | Claim once results are in (fallback in brackets) | Display |
|---|---|---|---|
| 1 | Who was studied, and what was predicted? | National cohort, 24 subtypes, 69 outcomes | Table 1 |
| 2 | Which risks can be anticipated? | An atlas of what is predictable at treatment start [what is not] | Fig. 2 |
| 3 | Does one model outperform the methods in use? | One model beats XGBoost per task and the specialists, and matches the best per-cell choice [equivalent, with N fewer models] | Table 2, Figs 3-4 |
| 4 | Does it improve on the indices, and help where none exists? | Better than the IPI on its own endpoint, and coverage of subtypes without one | Fig. 5 |
| 5 | Where does the gain come from? | Both axes add, most where data are scarce [which axis, where it hurt] | Fig. 6 |
| 6 | What does population pretraining add? | Matches the local model with X% of patients [no direct gain] | Fig. 7 |
| 7 | What does the model rely on? | Clinically sensible findings, read differently per complication, confirmed by what the model loses without each source | Fig. 8 |
| 8 | Does performance hold over time, regions, patients? | Stable across years and regions | Supplement |

Section headings stay as questions until the numbers are in. The intended
declarative headings are noted as comments above the sections that need them.

## Display items
Main: Fig. 1 (paradigms), 2 (atlas), 3 (headline), 4 (benchmarks and
calibration), 5 (risk groups vs IPI), 6 (pooling), 7 (national pretraining), 8
(what the model relies on), Table 1, Table 2: ten items, like the HemaSphere
DLBCL paper. If the journal allows eight, move Fig. 4 (calibration detail) and
Fig. 7 panels D-E to the supplement first.

## Where explainability sits, and how much of it
- Main text: Fig. 8 at domain level, in the grammar of the other main figures
  (rows shared across panels, white cell borders): A where the signal is, B what
  each part of the record adds (model-level test, no attribution assumptions),
  C which findings raise which risks with expected effects boxed, D one
  patient's history read for three risks (agreement between outcomes in S31).
- Supplement: the outcome-level atlas (S30), the structure figure with input
  restriction (S31) and the patient's three explanation cards (S32).
- Methods: one paragraph in the main text, details in the supplement.
- Discussion: answers "why does sharing not blur the predictions", and shows
  what a clinician would see.
- Checks (removal vs random, randomisation, second reference, completeness) are
  numbers in the text, not a figure.

## Compute for the attributions
Three changes cut the cost without giving up completeness (contributions that add
up to the prediction, checked for every patient):
- Gauss-Legendre quadrature with 8 nodes instead of 32 midpoint steps: on the
  routed model it matches a 256-step reference more closely than 32 midpoint
  steps do (relative error about 3e-7 against 7e-6). Patients failing the
  completeness check are recomputed with 32 nodes.
- All 72 targets in one vectorised backward pass per step instead of 72
  separate passes (identical results, tested): same arithmetic, far better use
  of the GPU by a 256-wide model.
- Optional half precision (bf16/fp16), guarded by the same completeness check;
  compare a pilot against fp32 before using it.

Arithmetic per patient falls from about 4,600 to about 1,160 forward-equivalents.
Expected: roughly 2-8 GPU-hours on a V100 for 1,500 patients in fp32 (well under
that on an A100 or in half precision), depending mostly on history length. The
50-patient pilot records the measured seconds per patient and the projected
hours for the planned sample in `attribution_run.json`. The checks and the
input-restriction runs add little.

## Writing principles used in the rewrite
- Lead each section with what was found, then how.
- One message per paragraph, no hedging before the claim; limitations gathered
  in one place.
- Numbers through `\result` keys only, so the text cannot drift from the
  analysis.
- Bold where the design earns it (scope, locked temporal test, strongest
  comparators, decomposition), and plain about what it does not show (causal
  effects, geographic transport, decision benefit).
