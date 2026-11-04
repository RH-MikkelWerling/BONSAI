# Test-release review — 3 November 2026

**Verdict: close, but not yet ready to sign and run the complete held-out analysis.**
The model decision is substantially settled. The remaining issues are agreement
between the written protocol, executable analysis and actual server configuration.
This review does not open the test set or change a scientific selection rule.

Reviewed the original lock and its amendments, the closing note, test scenarios,
prediction queue, inventory/analysis code, clinical renderer/configuration,
environment setter and overnight orchestration script.

## What is ready

- Local primary: J_OA_OWN, seed 42; learned routing R_AB remains a secondary
  model, regardless of which performs better on test.
- Best-checkpoint selection by tuning endpoint-group horizon BCE; last-checkpoint
  testing was explicitly withdrawn before test inspection.
- Encoder learning-rate rule resolves to 0.1x; the national choice has an explicit
  tuning-only rule and a fixed 0.0052 threshold.
- Cross-seed comparisons and seed-average contrasts are registered. These average
  fitted-model performance, not patient predictions into an ensemble. Their
  patient-bootstrap intervals do not estimate training-seed uncertainty.
- Clinical primary groups are rank/size matched on the same score-available
  patients without using their future outcomes. The complementary survival/
  specificity anchors are developed separately on training data. These are
  distinct analyses and the caption distinguishes them.
- Supported-cell comparisons, paired patient draws and both summary tables are
  implemented. XGBoost is compared on matched patient/label rows within each
  contrast, not by averaging its available cells independently of the comparator.
- AKI remains resolved; no outcome rebuild or refit follows from this review.

## 1. Holm headline family is not executable yet — resolve before sign-off

The 26 October amendment and closing-note section 6b promise Holm-adjusted
headline tests. The manifest still says
`multiplicity_specification: pending_analysis_lock`; the analysis outputs
`p_value` but contains no Holm adjustment for this family.

Freeze the exact hypothesis list, metric, all-horizon summary, weighting and
handling of pending national results. “OPERA against the clinical indices” is
not an exact hypothesis: specify which index/population, low/high comparisons,
or an explicitly defined aggregate. Alternatively explicitly amend those
clinical comparisons to descriptive status before test inspection. The choice
cannot be supplied by a plotting function or selected after seeing test results.

Then implement/output adjusted p-values and distinguish those headline tests
from the descriptive interval-based better/worse/equivalent labels. Do not
silently shrink the planned family because a national result arrives later.

## 2. Disease-balanced weighting does not match the lock — resolve before sign-off

The 30 October amendment defines equal disease weight, then equal **endpoint
family** weight, then equal outcome-variant weight, then equal horizon weight.
`effect_summaries(..., aggregation="disease_balanced")` currently uses equal
disease, **outcome**, and horizon weights; it never uses the loss-endpoint family
in `weights_for`. Its output labels correctly describe the implemented rule.

Either implement the written family-balanced weighting, with a meaningful check
using families with unequal numbers of variants, or explicitly amend the prose
to the implemented equal-outcome rule. Keep the 52 training/loss families distinct
from the broader clinical display groups. The equal-cell primary selection
numbers are not invalidated by this secondary-weighting discrepancy.

## 3. National-primary interpretation and handoff need one coherent definition

Section 4b's national tuning values and final encoder remain unfilled. This is
allowed to be staged by section 7, provided the rule and comparisons are fixed
now and national test results are not read before the tuning-only choice.

However, `O2` is always `gain(OPERA,J_L)`. If national OPERA wins, this contrast
changes **both encoder and readout**. It cannot support the scenario's claim that
outcome-specific reading alone improved on uniform borrowing. Preserve the local
mechanism contrast `OAOWN_J`, and use the matching national joint contrast for
national readout interpretation; explicitly distinguish those from the total
national-OPERA versus local-joint headline comparison.

`scripts/set_test_env.py` currently selects the local J_OA_OWN checkpoint and
alias. Its clinical retargeting replaces only `J_L`, so a configuration already
set to J_OA_OWN_L will not become R_E/R_EA through that substitution. The overnight
script also locates a local `/L/fix_...` run. Neither is a complete national-primary
handoff as written. When section 4b is filled, the primary alias, clinical model,
attribution checkpoint/config/vocabulary and patient-map exports must agree.
The local ladder/seed analyses remain labelled local.

## 4. Do not use the current overnight wrapper as an unattended final-test runner

`scripts/train_then_evaluate.sh` remains useful for rehearsals, but:

- It logs an evaluation error and can still mark a shard done when no training
  fits remain. A timeout also falls through to the done marker.
- Analysis actions continue after a failure. `OPERA_PAPER_FIGURES_AVAILABLE_ONLY=1`
  can then render a partial gallery. “Finished” is not proof of complete results.
- It sets inventory splits to the evaluation split only. For held-out analysis,
  the frozen tuning-selected S/D/O comparator and tuning-derived operating points
  require tuning predictions too, or an explicit separately frozen decision
  artifact. Include both split inventories while reporting held-out estimates.
- The wrapper uses `setdefault` for the estimand, retaining an old wrong value if
  present. The inventory action defaults to the 365-day horizon and known-status
  estimand unless the active environment explicitly overrides them.

Use a separate final run root, explicit six horizons and `cif_ipcw`, tuning plus
held-out artifacts, a frozen fit roster, and required-output reconciliation before
calling the run complete. This is scientific completeness checking, not a blanket
file-synchronisation or source-freshness scan.

The closing note now explicitly permits every registered design to be tested
descriptively. That scope is broader than evaluating just the primary models;
freeze the actual registered roster at signature so later bundles cannot silently
extend it. No design may be promoted or discarded on test performance.

## 5. Clinical endpoint configuration needs its final server-side review

The synthetic summary validates presentation and derivation, not actual endpoint
mapping. The example configuration still contains required nulls, including the
treatment-failure target, provenance/applicability fields and IPS target. Its
`native_endpoints` map omits some indices shown in the proof, such as ISS,
IPSSWM and MALT-IPI. `native_entries` silently leaves an unmapped score on its
template endpoint instead of guaranteeing a native-endpoint comparison.

The active server configuration may already contain these mappings; it was not
available locally for this review. Before sign-off inspect its expanded score
roster and tuning `score_inventory.csv`: correct index/population, model alias,
native or explicitly labelled proxy endpoint, allowed horizon/follow-up rule,
group order, and cutoff rule. A claim about every index's own endpoint requires
every included index to have a verified mapping. Keep the frozen training anchors
and report achieved test risk/specificity; training matching does not guarantee
equal achieved test risk.

## 6. Tighten the scenario interpretations before freezing them

- Inconclusive O2 is uncertainty, not evidence of no specialisation benefit.
  Equivalence and inconclusiveness should not share a negative-result conclusion.
- Positive P1/P5 must themselves replicate before scenario C can call borrowing
  the robust result.
- A/B's proposed mechanism conclusions require their decomposition results;
  three headline contrast labels alone do not establish where the signal sits.
- A near-zero/global-correction point estimate is not by itself evidence that
  the correction adds nothing. Likewise, similarity of routing weights is
  descriptive, not proof about the predictive contributions of those sources.
- The original native horizons are often longer than the available 1–2-year
  evaluation. Preserve the explicit shorter-horizon/proxy wording already
  supplied by the table and mappings.

These are interpretation limits, not requests for more architecture search.

## Mechanical defect fixed in this review

The prediction queue's evaluation intent used architecture-aware export flags,
but its actual command called `export_flags` again without the configuration.
For national attention models named R_E/R_EA this could still request nonexistent
routes despite recording a route-free intent. Both calls now use the architecture
configuration. Added two command-level regression cases, covering E and EA.
This changes requested diagnostic exports, not the fitted model or predictions.

## Verification and release sequence

The broad local suite passed **143 tests** across paired analysis, paper workflow,
stage-9/14 models, clinical grouping, clinical summary and rendering. After the
export fix, **three focused tests passed**, including both new national-command
cases and prediction reuse. `git diff --check` passed. No server artifact or
held-out patient data was opened; the filled tuning figures in the closing note
could not be independently reconciled without that server aggregate inventory.

1. Resolve the Holm family and secondary weighting; align the scenario wording.
2. Finalise the active clinical and analysis configurations and final runner;
   freeze bootstrap method explicitly (the analysis currently defaults to the
   frequentist multiplier method, conditional on fitted IPCW weights).
3. Rehearse those final configurations on tuning, including completion/coverage,
   required headline contrasts, both weightings, clinical tables and relevant
   interpretation exports. Existing valid fits/predictions can be reused.
4. Sign the note, record the code/config/fit-roster identity and commit. Do not
   introduce an extra source-freshness approval gate.
5. Run the held-out evaluation/analysis once under that specification. Complete
   the predeclared national tuning choice before looking at national held-out
   results. Paper/layout polishing can continue without changing the analysis.
