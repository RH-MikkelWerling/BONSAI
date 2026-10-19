# What the model relies on: plan, 18 October 2026 (final design)

Not an explainability paper. What we want from it, in order of priority:

1. **Scientific (paper):** does the model rely on sensible information, and does
   the shared model use the same findings differently for different
   complications? This supports the sharing thesis directly.
2. **Clinical (now):** known risk factors behave as expected, and an example of
   what an explanation of one patient's risks looks like.
3. **Clinical (later):** a per-patient explanation record (the episodes that most
   raise and lower each risk) that a tool can show beside each prediction, as
   CLL-TIM shows its top factors. Produced by the same run, no extra work.

## Figures and what each answers
| Figure | Question | Panels |
|---|---|---|
| Main Fig. 8 | What does the model rely on? (domain level) | A: what makes patients differ, by source; B: AUC without each part of the record; C: findings x domains, expected effects boxed; D: one patient's history read for three risks (agreement between outcomes is in S31) |
| Atlas (S30) | The same for every outcome | A: by source; B: by time; C: findings x outcomes, 21 expected cells boxed |
| Structure (S31) | Does it use the same history differently per outcome? | A: outcome x outcome agreement, domains outlined; B: AUC change on part of the record, by domain |
| Patient (S32) | Why is each of one patient's risks what it is? | One card per outcome: average findings, the episodes that raise and lower the risk, rest of the history, prediction |

Compute and storyline: see `docs/STORY_AND_STRUCTURE_20261018.md`.

The patient figure in the manuscript stays synthetic (no real timelines can be
published): its explanation table is written by `render_paper_figure_prototypes` to
`paper/figures/synthetic_patient_explanation.csv` and used by both the main
figure recipe and S32.

Checks (removal vs random, weight randomisation, second reference,
completeness) are reported as numbers in the text, not as a figure.

Dropped after review: the checks collage, k-means routes to high risk, the
patient profile across outcomes, the single-outcome timeline build-up, the
SHAP-style concept rankings, absolute attribution shares, history-reveal traces.

## Computation (one run, cached)
- Locked best model; every outcome on log-odds (atlas, structure); death,
  serious infection and severe neutropenia also in absolute risk (patient
  explanations, removal checks).
- every held-out test patient (weight 1); a stratified sample of 1,500 (up to 40 per subtype, weights restore the mix) is the fallback if compute is short;
  checks on 200 / 100 patients. Pilot 50 patients first to measure cost.
- Reference: average training event, timing kept. `[MASK]` is untrained.
- Input restriction on the whole test set, every outcome, no IG needed.

## Extractions
| File | Level | Used in |
|---|---|---|
| `summary_365d/attribution_variance_sources.csv`, `_windows.csv` | Aggregate | Atlas A-B, `ig_var_*` |
| `summary_365d/attribution_concepts.csv` | Aggregate | Atlas C, `ig_expect_*` |
| `summary_365d/attribution_outcome_similarity.csv` | Aggregate | Structure A, `ig_similarity_*` |
| `input_restrictions/input_restriction_summary.csv` | Aggregate | Structure B, `restrict_*` |
| `summary_365d/attribution_faithfulness.csv`, `_sanity.csv`, `_quality.csv` | Aggregate | Text, `ig_faithful_*`, `ig_rho_*` |
| `summary_365d/attribution_lab_response.csv`, `_top_concepts.csv` | Aggregate | Text, review of unexpected patterns |
| `explanations_INTERNAL.parquet` (`render_individual_attribution --all`) | Internal | The clinical record format |
| `patient_<id>_INTERNAL.png/.csv` (`--subject-id`) | Internal | Review with clinicians |

Aggregates are exported after disclosure review (cells under 20 patients
suppressed). Everything `_INTERNAL` stays in the secure environment.

## Commands (server, after the model is locked)
```
bash scripts/run_paper_experiments.sh attributions --attribution-max-new-patients 50 --execute   # pilot
bash scripts/run_paper_experiments.sh attributions --execute
bash scripts/run_paper_experiments.sh attribution-summary --execute
bash scripts/run_paper_experiments.sh input-restrictions --execute    # uses the IG baseline
bash scripts/run_paper_experiments.sh attribution-figures --execute
bash scripts/run_paper_experiments.sh paper-numbers --execute
python -m opera.run.render_individual_attribution --input-dir <analysis>/attributions \
    --all --config opera/configs/attribution_analysis.example.yaml --output-dir <secure dir>
```

## Inputs still needed
- Confirm the NPU codes and clinical names in
  `opera/configs/attribution_analysis.example.yaml`, then set
  `concept_mapping_reviewed: true`.
- Check the 21 expected finding-outcome directions before the run.

---

# Earlier design notes (Codex, 18 October; superseded where they conflict)


This is a rigorous secondary model-behaviour analysis. Computational efficiency
comes from fixed scope, shared computation and reusable exports, not from
omitting reliability checks.
Use one downstream checkpoint selected on tuning, three clinically distinct
outcomes (death, serious infection and neutropenia), and 365/730-day horizons.
The default planned sample is at most 300 patients with ten per fine subtype
before uniform top-up. Sampling weights restore the source-population mix.
If mandatory subtype quotas exceed the cap, the job rejects the settings rather
than silently exceeding the requested sample. Do not choose patients on test
outcomes or attractive explanations.

## What is actually explained

The target is the model's raw cumulative-incidence probability at a fixed horizon.
IG interpolates event-content embeddings from their training-derived mean to
the observed content, holding event times, age context and ordering fixed.
Background token content may be attributed, but age carried through time/age
embeddings is fixed; this is not a decomposition of the total effect of age.
Fine-disease conditioning used by a router stays fixed too. The reference is an
artificial embedding-space construction, not a healthy or event-free patient.
Joined laboratory-bin tokens are supported; incompatible whole-token numeric
embedding modes are rejected. Clinical effects and treatment benefits are not
identified. A calibrated prediction pipeline needs separate accounting for its
calibration mapping; these targets are raw neural probabilities.

Flash-checkpoint metadata is reconstructed under float32 SDPA for input gradients,
retaining its recorded rotary layout and learned weights. Tests cover production
OPERA reconstruction with flash metadata, native/injected prediction equality,
completeness and input gradients. Actual CUDA backend equivalence still needs a
server pilot. Record and compare pilot probabilities with the locked native
prediction exports before interpreting results.

## Cost control and reuse

The encoder forward along the path is shared across targets, but each target
still needs its own backward pass. Explaining all 69 outcomes is not the default
and is not the cost of explaining one. The default uses 32 midpoint steps with
one bounded refinement to 128 if completeness needs it. Faithfulness checks use
50 sampled patients; alternate-baseline and partial parameter-randomisation
checks use 30. Replacement/deletion tests use 1/5/10 events and three random
sets per size. Fixed-index history reveals use 30 patients and inference only.

Start the SAME planned sample with a five-patient pilot, then resume it without
the pilot cap. Completed patient exports and the training reference vectors are
cached, hashed and checked. Changing scientific settings, checkpoint/data or
model/extraction implementation requires a new output directory. Changing
execution chunk size or removing the pilot cap does not. One process owns the
output directory through a file lock. Resume skips completed IG/check exports;
only the interrupted patient may need recomputation. No clock-time promise is
made: benchmark the first patients at the actual sequence lengths and GPU.

## Numerical and perturbation checks

Native inference must match the injected-embedding forward. Nonfinite outputs
fail. Every patient/target records prediction, reference prediction, signed
attribution sum, absolute completeness error, regularised relative error and
step count. Nonconverged finite explanations stay in quality reports but are
excluded from substantive summaries. Relative error divides by at least 0.001
risk units, so inspect absolute error as well for nearly zero differences.

For validation, sum signed token credit within a timestamp event, then replace
or delete highest-signed, lowest-signed and random event sets. Event numbering
excludes timestamps that contain only structural tokens, so random controls do
not waste draws on nonexistent events. These are relative perturbation checks;
deletion may be out of distribution. Positive net credit and absolute magnitude
are different rankings. Sanity checks randomise nonconstant readout parameters
and, separately, transformer parameters while retaining embeddings. This is a
partial randomisation diagnostic, not the trained random-initialisation arm.
Alternative references answer somewhat different questions; agreement is a
sensitivity result rather than a universal pass threshold.

## Faithful aggregation

1. Keep signed risk contributions in percentage points. Their sum approximates
   the risk difference from the chosen reference, not the absolute risk.
2. For source/time shares, sum absolute token credit WITHIN each patient,
   normalise within patient, then average with sampling weights. Long histories
   do not dominate simply through token counts. Zero-magnitude histories have
   undefined shares and the share denominator is reported separately.
3. Also report signed contributions and positive/negative timing totals. Positive
   and negative evidence may cancel; absolute shares must not be described as
   explained variance, independent source effects or additional clinical utility.
4. Concept means are explicitly among carriers; they are not population means.
   Latest-result analyses select the latest result first, THEN ask whether its
   bin belongs to a prespecified low/high range. Zero- and one-based bins are
   mapped from recorded vocabulary bounds. Relative bin position is not a
   clinical reference range, concentration or dose-response function.
5. One summary contains ONE horizon. Render 365 and 730 days separately from the
   same cached tokens. Patient-bootstrap intervals describe variability of the
   explanation sample with fixed fitted weights; they do not include model-fit
   uncertainty or establish biological hypothesis tests. Data-selected concepts
   and nominal direction checks remain exploratory.

Named clinical concepts/laboratory curves are omitted until
`concept_mapping_reviewed: true`. Review actual code identity, bin semantics,
units/threshold meaning and clinical applicability; do not merely flip the flag.
Source/time/quality and perturbation summaries can still run before that review.
Empty concept tables have headers and render without fabricated clinical results.
Minimum support rules and secure-output review apply to released aggregates.

## Timing and predictions over time

The source/time atlas asks when the attributed records occurred. It does not
attribute the event times themselves. Signed lookback-window summaries complement
absolute shares. The history-reveal plate restricts records to those available
through 365/90/30/7/0 days before treatment while retaining the SAME treatment
index and forecast horizon. Its paired changes are sensitivity to progressively
revealed history, not clinical predictions made at those earlier visits.
True longitudinal forecasting would require earlier origins, origin-specific
eligibility and information, and longitudinal validation; defer it for this paper.
There is no reason to compute fresh IG for every history cutoff.

## Figures and practical use

Three supplementary prototypes are included: selected-outcome source/time and
concept patterns; numerical/perturbation/reference checks; and signed timing plus
history reveals. The existing synthetic atlas illustrates a possible broader
panel, not a completed 69-outcome explanation experiment. For the final paper,
use the selected-outcome atlas and checks; keep timing as an optional supporting
plate or combine it after the actual findings are known. No main claim depends
on aesthetically appealing patient examples. Raw patient/event exports remain
internal; named examples require disclosure review.

This can reveal whether predictions concentrate on recent laboratory results,
medication or longer histories, which signed concepts recur among carriers,
and whether predictions are sensitive to recent records. It cannot by itself
explain why OPERA outperforms a comparator: performance contrasts establish that
claim. Shared representations can distribute credit among correlated records.

## Primary methodological sources checked

- [Sundararajan, Taly and Yan, ICML 2017](https://proceedings.mlr.press/v70/sundararajan17a.html): IG, reference dependence and completeness.
- [Adebayo et al., NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html): model/data randomisation sanity checks.

The earlier draft's assorted 2026 literature links are not prerequisites for
this implementation and were not used as verified evidence of clinical validity.

## Local verification, 18 October 2026

The focused release suite passed: **39 tests** across token attribution,
attribution workflow, input restriction, visualization/attribution integration,
and representation-gradient diagnostics. It includes a real OPERA checkpoint
round trip, injected/native prediction equivalence, linear-model completeness,
resume/cache corruption checks, deterministic perturbations, single-horizon
exports, latest laboratory selection and patient-weighted summaries. Scoped Ruff
checks and `git diff --check` passed. All three explanation prototypes rendered.
The synthetic manuscript compiled successfully (85 pages, no undefined references
or citations; an existing overfull table box and two PGF node warnings remain).

This does not substitute for a secure-server pilot using the actual trained
checkpoint and clinical vocabulary. Inspect probabilities, completeness failures,
wall time and memory after the five-patient pilot, then resume the unchanged
planned sample. In particular, the clinical laboratory/concept mapping remains
explicitly unreviewed in the example configuration; named clinical panels stay
disabled until that mapping is checked. No patient data or GPU jobs were run as
part of this local verification.

## Clarification after manuscript review

The sample sizes are planned descriptive budgets, not assurances of precision
for every disease or concept. Before interpreting patterns, report numerical
failures, reference dependence and event-removal controls. A pilot establishes
cost and reliability; unsupported subgroup panels remain suppressed. Arbitrary
lookback summaries, reviewed concept groupings and internal additive patient
displays can use the cached signed token credits. True earlier-origin clinical
forecasts and explanations for an uncomputed model/target need separate work.
See `MANUSCRIPT_REVIEW_20261018.md` for the claim-to-evidence boundaries.

## Lightweight vocabulary grouping for the overview

`attribution_concept_ranking.csv` is the recommended discovery table. It uses
exact NPU identifiers with all bins/text values collapsed, ATC level 3 (four
characters), and ICD-10 three-character categories while preserving the Danish
D prefix. Different NPU identifiers are not assumed equivalent. Procedures,
pathology and unrecognised codes remain intact: arbitrary prefix truncation is
not treated as a validated hierarchy.

All repeated observations contribute to a patient's signed concept sum. The
ranking is the sampling-weighted population mean absolute patient-level sum,
with absent concepts contributing zero. Signed population means, weighted
prevalence and distinct carrier counts accompany it. Minimum support is counted
in patients, not events. This descriptive discovery table has no confidence
intervals; review labels for only the leading concepts before clinical display.
It does not replace the older `attribution_top_concepts.csv`, whose carrier-only,
latest-laboratory-value summaries answer a different question. Regenerate CPU
summaries into a fresh directory to obtain the new table; no new IG is needed.

Verification: the new grouping/ranking tests and attribution integration suite
passed (17 tests); scoped Ruff and whitespace checks passed.

## Ready-to-render overview and internal individual view

The summary step now also writes `attribution_overview.csv` and
`attribution_concept_timing.csv`. The overview selects the top 15 supported
concepts using the population-weighted mean absolute patient-level net credit.
It reports weighted recording prevalence and weighted 10/25/50/75/90 percentiles
of signed patient contributions, including zero for absent concepts. These are
patient distribution ranges, not confidence intervals. They complement the
existing numerical and perturbation diagnostics rather than replace them.

The matching time map retains that exact concept order and partitions signed
credit into 0–30, 31–90, 91–365, >365 days and undated/background contents.
Population means retain noncarriers as zeros; sparse concept/window cells are
masked using the existing minimum distinct-patient threshold. Exported positive
and negative token-credit totals allow cancellation to be inspected. Missing
support is never drawn as an observed zero. Cross-outcome magnitudes are not a
universal measure of clinical importance. Data-selected top concepts are
exploratory and labels require review for only the leading codes.

`attribution-figures` renders one overview per default outcome, using those CSVs
and the normal source-hashed figure recipe. Recipes for custom outcome panels
can reuse the same function with `options.outcome`. Optional names live in
`summary.concept_labels` and are used only when the mapping-review flag is true.
Unreviewed codes still render honestly with their identifiers.

An individual chart can be produced entirely from cached exports inside the
secure environment:

```bash
python -m opera.run.render_individual_attribution \
  --input-dir /secure/paper_analysis/attributions \
  --subject-id PATIENT_ID --outcome overall_survival --horizon-days 365 \
  --output-dir /secure/paper_analysis/individual_review/example_01
```

Replace the example paths with the actual analysis root; no new server path is
required by the extraction workflow. Optional `--config` supplies the same
reviewed labels. Select examples systematically, without favouring visually
plausible or correctly predicted cases. This command verifies cached export
hashes and requires a converged patient/target explanation. Its outputs and
patient provenance are marked INTERNAL and are not added to the public recipe.

The individual chart combines a signed waterfall (top eight concepts, all
remaining credit and numerical residual) with dated event contributions for
those concepts. It reconciles reference and final risk. The waterfall order is
importance, not chronology; the event timeline is attribution at treatment start,
not an earlier-origin risk curve. Undated credit remains in the waterfall.
This is a review figure, not a clinically validated decision-support interface.

Synthetic global/local figures are registered in the prototype renderer and
included in the supplementary manuscript; they contain invented patients and
labels, not empirical findings. Main extraction, bootstrap settings and GPU
budgets are unchanged. Regenerate summaries into a fresh directory after the
extraction completes, then run the existing `attribution-figures` command.

Release verification for these views: 33 relevant tests passed, including signed
aggregation, temporal reconciliation, patient-support suppression, figure
rendering, and the individual CLI reading hash-verified cached Parquet exports.
The production recipe rendered all three outcome overviews from aggregate CSVs.
Both synthetic examples were visually inspected. The 91-page synthetic
manuscript builds in `paper/render/explanations_20261018/` without undefined
citations/references or overfull boxes (two existing PGF node warnings remain).
Scoped Ruff and whitespace checks pass. No server jobs were launched.
