# Final experiment gates and the three study questions

This is the current pre-launch checklist, supplementing
`supervision_factorial_20261014.md` and `hidden256_paper_run_plan.yaml`.
It records intended comparisons, not completed results or permission to open
held-out labels. The collaborator-matched local256 pretraining is running.
The complete contrast inventory and prioritized additional questions are in
`contrast_plan_20261014.md`. External local DAPT is an explicit required stage
with before/after contrasts, not folded into the external-pretraining label.

## What each experiment identifies

| Study question | Primary comparison | Interpretation and limit |
| --- | --- | --- |
| Does pooling diseases help? | D-S and J-O on the local256 shared architecture | Adding disease supervision without/with outcome pooling |
| Does pooling outcomes help? | O-S and J-D on the same architecture | Adding outcome supervision without/with disease pooling |
| When does sharing help or hurt? | Per-cell paired effects versus train event/patient support, disease, endpoint and horizon | Natural-rarity heterogeneity; associations, not randomized scarcity effects |
| Does the router add value? | Locked router versus J, plus matched source-bank arms | Additional conditional adaptation above pooled supervision |
| Is the neural approach better than standard tabular prediction? | Locked neural model and J versus tuned EHR XGBoost on identical patients | Practical predictive comparison; sequence/feature differences are part of each pipeline |
| Does broader external pretraining help over local pretraining? | Change encoder origin under fixed shared/locked-router downstream protocols | External-pretraining package effect; not an isolated effect of pretraining size |

S = one disease/one outcome; D = all diseases/one outcome; O = one
disease/all outcomes; J = all diseases/all outcomes. D/O are training scopes,
not router source-bank masks. All four share the architecture, local256 encoder,
endpoint definitions and prospective split. Their contrasts isolate the
specified supervised-pooling procedures while holding pretraining fixed.
Data quantity, optimization steps, task weighting and available supervised
signals change with scope; do not call the effects pure biological mechanisms
or equal-compute information effects. Baseline pretrained representations may
already contain disease/outcome-relevant information.

Report paired AUC and Brier at the main 365-day horizon and all six horizons
in supplements, including negative transfer and support. Prespecify summaries
by disease, endpoint group, horizon and training-event/patient support. Keep
training support separate from held-out minority-class support (precision).
Use patient-resampled paired uncertainty; preserve patient dependence when
aggregating correlated endpoints. Across-seed uncertainty is separate from
patient-bootstrap uncertainty. Natural rarity describes where effects vary;
an artificial label-fraction experiment is needed only for a controlled
label-scarcity claim. Do not add it implicitly to the core scope.

## Gate 1: one auditable data/encoder contract

- Freeze the included-patient manifest, endpoint release, eligibility, splits,
  cohort map and vocabulary hashes. Record supervisor sign-off separately.
- Verify the disease identity supplied to every router was available at index.
  Retrospective +14/+90-day diagnosis rescue is not automatically an available
  feature; see `locked_prediction_analysis_20261014.md` and its identity audit.
- Audit the actual local256 saved checkpoint and resolved specification:
  width/layers/heads, causal objective, time/value encoding, sequence length,
  training cutoff, normalization/vocabulary fitting provenance and token budget.
  The user confirms matching collaborator specifications; verify saved artifacts.
- Strict checkpoint reconstruction and forward pass; token IDs/value/time units
  agree with processed inputs. Confirm exclusive index truncation, shared index
  across outcomes, physical-file pooling and fine-disease filtering.
- Freeze tuning selection inventory/known-status counts and cohort/outcome
  support; classify unsupported, missing-tuning and failed cells explicitly.
- Resolve architecture-appropriate physical/logical batch settings; smoke-test
  optimizer steps, finite loss, validation, checkpoint save/reload and resume.
  Check short datasets produce batches. Benchmark memory/throughput at width256.
- Run tests/lint and secure-server preflights, including the new full-tuning BCE
  selector. Local generator checks are not a GPU/data-readiness certification.

## Gate 2: choose router policy before the full grid

Source mode, query mode, adapter rank and query width are different knobs.
"Interacting versus disease-only" is not a clean contrast: interacting describes
the query; disease-only describes enabled sources. A disease-only router can
also use an interacting patient/disease/target query.

1. **Optimization policy:** archive the matched64 anchored/borrowed-update,
   matrix-only epsilon and curriculum evidence. Record one gradient-ownership
   rule, adapter epsilon and curriculum policy, provisionally if needed. Verify
   live gradients, optimizer groups/moments and FP32/mixed-precision projected
   corrections on256. If the transferred policy fails this diagnostic, make one
   bounded matched correction; do not repeat the old optimization grids by default.
   Factorial S/D/O/J uses curriculum off. For a router-versus-J architecture
   claim the router must also use curriculum off, or a matched shared control
   with that policy must be added and reported.
2. **Source choice:** at rank8/query128 and interacting query compare neither,
   disease sources only, outcome sources only and both. Keep all other settings
   identical. Disease-only versus both is the incremental outcome-bank test.
   Outcome-only versus neither alone cannot rule out an interaction with disease
   sources. Disease-only is an acceptable final model. If benefit is inconclusive,
   prefer the simpler source inventory under the recorded tuning rule.
3. **Correction capacity:** on the provisionally selected source mode compare
   rank8 versus rank16, keeping interacting/query128 fixed. Rank is the adapter
   bottleneck, not patient-embedding width. Check actual corrections, damping,
   parameter count, tuning AUC/Brier/BCE and cost. Prefer rank8 if inconclusive.
4. **Query structure:** compare additive versus interacting at the selected
   source mode/rank and identical residual scale, initialization and budget.
   Compare query128 versus query512 once at the selected rank in interacting
   mode if retaining the planned query-capacity check. Larger width is not
   automatically needed because encoder width increased.
5. **Lock and match:** save selected policy, source mode, query mode, rank,
   query width, checkpoint/hash, seed, residual scale, home floor and shrinkage.
   Repeat required source controls at the final settings if they changed after
   step2; reuse a fit only if its entire final protocol matches. If query/rank
   selection changes the provisional source conclusion, make one matched final
   disease-versus-both confirmation, not a Cartesian architecture sweep.

Keep scale0.7/home-floor0.05/event-tau10/shared-logit0 as the planned reference;
do not add a new scale sweep. Gate weights and nonzero gradients establish use,
not predictive usefulness. Use identical target patients and paired tuning
predictions. Architecture decisions remain tuning-only; rank/query/source
preferences are empirical and can interact, so the bounded search does not
claim a global optimum. Final reported fits use the common full-tuning
`val/cr_horizon_bce_endpoint_group_macro` selector, 30-epoch budget and matched
seed replication. Prefer the simpler candidate when inconclusive; record the
practical tolerance and aggregation rule before adjudication. No numerical
tolerance is silently invented here.

The source factorial measures incremental source-bank architecture effects.
For a stronger claim that *learned off-home routing* causes gains, add one
trained matched fixed/home-only-routing comparator. Post-hoc source removal is
a sensitivity analysis and does not replace that trained comparator. This is
optional for the three predictive study questions.

## Gate 3: tabular comparison is ready

- One frozen pre-index feature matrix with feature-generation provenance;
  preprocessing/imputation fitted on train, hyperparameters on tuning only.
- At every target cell, neural/tabular prediction IDs, labels, eligibility,
  horizon and competing-event definition match exactly. Preserve known-status
  masks and unsupported cells; no model-specific silent population intersection.
- Tune XGBoost fairly and retain logistic regression as a useful reference.
  Choose tabular hyperparameters using horizon BCE/log loss consistent with the
  probability task; do not insist tree iterations equal neural epochs. Record
  budgets. Standard single-cell XGBoost is the practical usual-care comparator;
  it need not replicate every neural pooling regime.
- Label EHR-only and registry-enriched feature sets separately. Registry/IPI
  complete-case analyses require the same restricted patients for every model.
- Freeze any probability calibration on tuning and assess it only on held-out.
  Report discrimination, calibration/Brier, uncertainty and cell-level failures.
  A neural win would compare whole predictive pipelines, not prove routing was
  the cause; the shared-J/router contrasts provide that additional separation.

## External pretraining: reduced matched experiment, not a second full grid

The basic question is whether externally pretrained initialization improves
the same downstream predictors. Freeze the downstream model on local tuning,
then change encoder origin. Do not choose a fresh rank/query/source winner for
every encoder: that changes the question and creates unequal search budgets.

| Encoder initialization | Shared J | Locked router |
| --- | --- | --- |
| Local256 pretraining | Required; reuse final matched fit | Required; reuse final matched fit |
| External pretrained, remapped vocabulary, no local SSL adaptation | Required for the direct-pretraining claim | Required for the direct-pretraining claim |
| External pretrained plus local DAPT | Required for the adapted-external claim | Required for the adapted-external claim |

This is six downstream fits per seed, with only four new fits if the two local
ones already match. Each evaluates the full supported disease/outcome/horizon
panel. A direct external arm still gets supervised downstream training: "no
local DAPT" does not mean zero-shot prediction. External-plus-DAPT versus direct
external estimates the contribution of local SSL adaptation to that external
initialization. Compare encoder origins within each readout; compare router-J
within each origin. Their difference estimates pretraining-by-router interaction.

If only external-plus-DAPT is practical, two new fits per seed suffice to ask
whether that adapted external pipeline beats local pretraining. They do not
separate direct external initialization from local DAPT. State the narrower
claim instead of inventing a direct arm with a zero-epoch training run.

Always repeat checkpoint/vocabulary/architecture audits and a bounded GPU
smoke test for each new origin. Verify code-system correspondence, exact
token-string remapping and unmatched-token frequencies, numeric/time units,
maximum context, causal objective and precision. Source heads/optimizer state
are not transferred. Audit external patient overlap, access to post-cutoff
events and outcome labels, and pretraining dates. Unverifiable temporal/patient
provenance limits a strictly prospective claim and must be stated. More data
alone is not established if tokenization, objective, training duration or source
population also differ; call this an external-pretraining comparison.

The current repository supports audited external-to-local DAPT. A direct,
local-vocabulary checkpoint export is still a separate implementation/audit
requirement before the direct rows are executable. The source bundle and audit
must support the exact time/value architecture, not merely matching width256.

No repeat is needed for feature matrices, tabular training, cohort/Table1,
clinical endpoint construction or the whole rank/query/epsilon/curriculum
search. Reuse saved tabular predictions if patients/labels/protocol match.
Reuse evaluation code/figure templates, but regenerate actual external
predictions, uncertainty, route/embedding outputs and provenance.

**Boundary of the reduced design:** it does not answer whether external
pretraining replaces cross-disease or cross-outcome supervised pooling. For
that stronger claim, repeat S/D/O/J on a prespecified balanced subset of cells
under the chosen external origin. Compare, for example,
`(J-S)_external - (J-S)_local`, and the separate D-S/O-S/J-O/J-D contrasts.
Select that subset before external results, using training support and clinical
coverage rather than observed gains. Results support the sampled-cell claim.
Full-landscape pooling-by-pretraining claims require the full factorial again.
Frozen probes across origins are optional if claiming improved frozen
representations; full finetuning measures practical initialization value.

## Before releasing the full run queue

Record one signed-off data contract, one audited local256 checkpoint, one
router configuration and common selection rule; review matched tuning evidence
for the bounded checks; freeze estimability/support definitions, practical
tolerance, aggregation and seed roster; generate separate config/hash/output
inventories for S/D/O/J, final router/source controls and tabular predictions.
Preflight and smoke-test each *regime*, then check each cell's actual support.
Queue expensive specialists only after these gates. External audits/fits can
follow the local study without blocking width-independent tabular preparation.
Keep held-out labels closed until the entire intended comparison roster and
selection rules are frozen. New exploration after release needs a separately
labelled study rather than retuning the locked result.
