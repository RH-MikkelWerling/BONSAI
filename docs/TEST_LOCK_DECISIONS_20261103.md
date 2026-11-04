# Decisions proposed after the test-lock review

Discussion proposals, not a signed amendment. No test data have been opened.

## Multiplicity and clinical success

Use Holm for a finite list of confirmatory headline hypotheses, not every
disease/outcome cell, horizon, seed and exploratory architecture. The exact
family must be enumerated before testing, including how pending national
hypotheses enter. Keep descriptive effect estimates and intervals visible
regardless of significance. Holm does not require independent tests; it does
require appropriate input p-values.

Primary methodological source:
https://www.fda.gov/media/162416/download (October 2022, multiple endpoints;
co-primary AND claims versus OR claims, and Holm under dependence). This is
statistical guidance relevant to the claim structure, not a claim that OPERA
is a regulatory drug trial.

Updated after the author's follow-up on 3 November: use estimation-focused
clinical comparisons rather than the previously proposed co-primary AND test.
Report all prespecified indices with paired pointwise intervals. Clinical
comparisons are descriptive, outside the confirmatory Holm family; favourable
point estimates are not formal non-inferiority or simultaneous superiority.

Matching test category SIZE is outcome-blind. Matching test KM by scanning
cutoffs is outcome-adaptive. The latter can be presented as a descriptive
frontier, but not as independently validated performance at a frozen cutoff.
Primary equal-size comparisons and complementary training-frozen anchors remain.
Report achieved test survival/specificity beside counts; avoid asserting exact
matching where the estimates differ. No additional clinical margin is needed
unless a formal non-inferiority claim is intended.

Counts in the current figure are index/population comparisons, including pooled
populations; they are not necessarily distinct independent clinical indices.
Non-negative estimates remain descriptive, not proof of non-inferiority.

## Weighting

The equal-cell primary is intentional and remains unchanged. Equal-outcome
versus equal-endpoint-family secondary weights are different legitimate
estimands. The discrepancy is with the protocol, not patient labels or fitted
predictions. Within one disease, two anaemia-grade heads and one mortality head
give anaemia 2/3 of equal-outcome weight, but 1/2 of equal-family weight.

Implemented: the written disease → loss endpoint → outcome variant
→ horizon weighting for the declared secondary summary, using `loss_endpoint`
from the approved training configuration, not the broader display groups.
Retain equal-outcome weighting as an additional labelled sensitivity if useful.
Existing point estimates and bootstrap draws can be reaggregated; no refit is
required. The family map must accompany the saved draws when resummarising.

## Clinical endpoint/population audit

Added `opera.run.audit_clinical_endpoint_config`. It expands the actual score
configuration against metadata and lists unresolved targets, missing native/
proxy mappings, incorrect primary models, unregistered horizons, cutoff-rule
differences, and absent applicability/provenance reviews. It reads no prediction
artifacts and writes no patient rows. It exits nonzero if definitions need work.
It checks recorded completeness, not the truth of a clinical review statement.

Run on the server using its actual paths and the selected model alias:

```bash
.venv-v100/bin/python -m opera.run.audit_clinical_endpoint_config \
  --score-config /path/to/active_clinical_score_config.yaml \
  --metadata /path/to/prepared_clinical_metadata.parquet \
  --outcomes-config /path/to/resolved_approved_training_config.yaml \
  --expected-model J_OA_OWN_L \
  --output-dir data/paper_256_v3/analysis/clinical_definition_audit_20261103
```

Use R_E/R_EA instead if the tuning-only national selection changes the primary.
The two resulting aggregate audit files can be reviewed locally.

Missing example mappings cannot all be filled with overall survival:

| Index | Original target evidence | Implication |
| --- | --- | --- |
| ISS | Survival: https://pubmed.ncbi.nlm.nih.gov/15809451/ | Overall-survival comparison |
| IPSSWM | Survival after treatment: https://pubmed.ncbi.nlm.nih.gov/19196866/ | Overall-survival comparison in the intended population |
| FLIPI2 | PFS: https://pubmed.ncbi.nlm.nih.gov/19652063/ | Explicitly document the available proxy, if exact PFS is unavailable |
| MALT-IPI | Primarily EFS: https://pubmed.ncbi.nlm.nih.gov/28720586/ | Explicitly document any proxy; do not silently call it the original endpoint |

The index must also apply to the population: recorded use in metadata alone
does not establish validity for that disease. Native-population comparisons
and pooled/off-label extrapolations must be labelled separately.

## Final model versus component tests

Keep `O1 = final OPERA − XGBoost` and `O2 = final OPERA − J_L` as total-system
comparisons, with O2 labelled accordingly. If OPERA is national, O2 includes
encoder/tokenisation changes as well as readout changes.

Add an explicit selected-encoder comparison: final OPERA minus the joint model
on the SAME encoder (J_L/J_E/J_EA). Keep the component ladder on each available
encoder: J → own corrections (R_PF_OFF) → outcome reading plus corrections.
Final minus R_PF_OFF isolates the addition of outcome-specific reading on the
same encoder with the same correction design. Locally, also show final minus
reading-only J_OA, isolating the addition of corrections; the four-fit factorial
can describe interaction rather than assuming additivity. The national partial
ladder is already planned; a full national factorial is not required for the
current claims. Do not invent a new fit requirement to answer an already
matched comparison. All intervals condition on the trained fits.

National-versus-local comparisons concern the pretraining/tokenisation pipeline,
not a clean causal experiment varying only pretraining population size.

## Close-out order

1. Clinical comparisons are descriptive. Enumerate the remaining methodological
   confirmatory family before signing.
2. Correct the declared secondary weights; add the explicit matched-encoder
   contrast; implement the frozen Holm family and align scenario interpretations.
3. Finish the final runner: no training, frozen fit roster, explicit six horizons,
   cif_ipcw, tuning plus held-out inventories, explicit bootstrap method, and
   no success marker after failed/missing required outputs. Rehearsal helpers
   may still render partial figures, but those do not certify a complete release.
4. Run the clinical definition audit and one tuning rehearsal of the resulting
   analysis/configuration. Reuse valid trained fits and predictions.
5. Sign and record the lock/configuration/fit roster, then evaluate the test set.
   Apply the national tuning-selection rule before viewing national test results.

The 3 November follow-up implements family-balanced secondary weights and
matched-encoder contrasts, and updates the draft closure. The lock remains
unsigned; no held-out evaluation has been launched.


Implementation follow-up
------------------------
`O2_MATCHED`, `O2_LOCAL`, `O_ROUTING_MATCHED`, `O_READ_MATCHED` and the available
local reading-only comparison are exported as central numbers. References are
fixed by the selected model's encoder, not chosen on test results. Corrected
national-vs-local references and encoder-origin panels to retain J_OA_OWN_L
when R_E/R_EA is the final selection; previously those helpers reverted to R_L.

Clinical configuration additions document ISS/IPSSWM survival and the original
FLIPI2/MALT-IPI targets with source links. FLIPI2/MALT-IPI outcome mappings remain
explicit nulls until an approved exact endpoint or labelled proxy is chosen.
The active server config, exact registry strings/versions, applicable diseases,
and time origin still need reconciliation. No patient outcome or fit changes.


Validation of this follow-up: 72 targeted evaluation/clinical/mechanism tests
passed, followed by 40 targeted selection/contrast/weighting/clinical tests after
the national-reference correction. No refits and no test predictions were read.

Paper-number generation and prediction workflow checks: 33 passed after adding
the required family map to the synthetic metadata fixture. YAML parsing and
`git diff --check` also passed.


Endpoint matching disclosure (3 November)
----------------------------------------
The clinical template now records `endpoint_match`, `endpoint_proxy_note` and
sources. These fields propagate to score_inventory.csv, headline_scores.csv
and the endpoint-audit roster. `same_endpoint` means the same event definition,
not identical horizons, time origin or population to the original study.
IPS is endpoint-approximate if evaluated on next treatment/death. FLIPI2 and
MALT-IPI are explicitly flagged as pending proxy mappings, not silently filled.
The broader failure/new-haematological-diagnosis composite is not an exact
transformation/PFS/EFS definition. No outcome mapping or frozen label changed.


Evaluation-cohort KM matching implemented (3 November)
----------------------------------------------------
Author authorised both equal-size and equal-observed-KM comparisons. This
supersedes the earlier note that an evaluation-KM frontier was not implemented.
New outputs from render_clinical_score_panels:
- evaluation_km_matched_groups.csv: both ends, achieved KM, counts and count
  differences, per-1,000 differences, paired intervals and valid-draw fractions.
- evaluation_km_frontiers.csv: every attainable tie-preserving boundary,
  empirical KM, selected size, support and target-match flags.
- <score_id>/evaluation_km_matching.png (and configured formats): two frontier
  panels with clinical and selected model points in the shared figure style.

Defaults in evaluation_km_matching: enabled, 200 paired draws, seed 2026,
minimum_group 20, minimum_at_risk 10. No tolerance, no spline, no monotonic
assumption. Low and High selected independently and can overlap. Evaluation
outcomes intentionally select these descriptive operating points; this does
not modify fitted models or frozen operating points. The procedure runs before
full-partition construction so a partition failure does not erase its outputs.
Frozen-anchor files and the existing four-panel clinical summary keep their
original meanings. New tables are included in the review-package allowlist.

New KM tables also feed paper-number generation: per-score/end macros carry
index/model counts and additional patients per 1,000 with paired intervals,
and clinical_evaluation_km_matching.csv preserves the full report in the paper
numbers directory. These macros have the clinical_evaluation_km_ prefix and do
not replace the frozen-anchor or equal-size numbers.

Validation: 35 targeted clinical-renderer, KM-matching, clinical-summary and
paper-number tests passed, including native train/tuning end-to-end exports.
A synthetic frontier figure was visually inspected. No held-out data were read.


Holm implemented after author agreement (3 November)
---------------------------------------------------
Fixed eight-member primary family, seed 42, pooled all-horizon equal-cell AUC:
P1/P2/P5/T2/O1/router_borrow_AB/E1/O2. headline_holm.csv contains raw/input/
adjusted p-values, explicit simulation floor, family completeness and decisions.
Missing members retain slots and make the table provisional. Bayesian tails
are refused as frequentist inputs. This is also runnable on saved summaries;
no GPU or prediction rerun is required.
