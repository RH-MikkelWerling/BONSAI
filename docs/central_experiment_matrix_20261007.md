# Central OPERA experiments and run order

## Claim–comparator map

| Claim | Primary contrast | What it isolates |
|---|---|---|
| Joint labels overcome scarcity | Pooled shared BCE versus same-encoder disease × outcome specialists | Benefit of supervised training-data pooling |
| Which axis of pooled labels matters? | All-pooled versus one-disease/all-outcomes and all-diseases/one-outcome training | Disease and outcome training-pool contributions |
| Adaptive borrowing adds value | Selected matched-scale router versus pooled shared BCE | Added value beyond ordinary pooling |
| Source choice and interaction matter | Neither/disease/outcome/both source-bank factorial; additive versus interactive at equal scale | Router mechanism, conditional on pooled training |
| Broader pretraining adds value | Local Time2Vec versus compatible PHAIR direct versus PHAIR + local DAPT, each under matched readout | Encoder-transfer effect and interaction with routing |
| Does broader pretraining replace label pooling? | Pooled shared BCE versus disease x outcome specialists, repeated with the same PHAIR-derived encoder on both sides | Pretraining-by-pooling interaction, distinct from the router effect |
| Clinical benchmark | Locked EHR-only XGBoost versus neural models on identical patients and risk sets | Value beyond task-specific tabular learning |

Clinical outcome families are reporting and hierarchical-analysis strata. They
are **not** silently collapsed into one outcome, and a single-cell specialist
may not train on other outcomes. Each specialist can share its own six horizons
in one model. The disease × outcome grid has at most 24 × 69 cells before
support/eligibility screening, not 24 × 69 × six separate fits. The true run
count must come from the approved release's estimable-cell manifest.
Cells with no or too few training events cannot support a conventional
specialist but may still receive predictions from a pooled model. Record this
as a **coverage** result separately from the paired specialist-performance
contrast. A numerical AUC for such a cell still requires adequate held-out
events, non-events and follow-up; the existence of a pooled prediction alone
does not establish predictive validity. Freeze training- and evaluation-support
rules independently.

## Priority and execution gates

1. **Now, CPU/read-only:** Inventory final clean router checkpoints, configs,
   completion markers and code provenance. Export the aggregate validation JSON
   using `docs/manuscript_result_lock_20261007.md`. Audit the fresh equal-scale
   additive/interactive pair. Do not interpret the old interaction-absent run.
2. **Now, GPU as available:** Generate fixed-checkpoint paired tuning predictions
   for the clean five-arm ladder and no-source control; make the model overview,
   outcome atlas, source-borrowing and route-movement drafts. Replicate finalist
   seeds before selecting architecture and scale. Keep held-out closed.
3. **Specialist pilot, then grid:** Freeze the event-support threshold and
   specialist optimization policy. Run a support-stratified pilot to estimate
   runtime and failure rate. If feasible, schedule every estimable disease ×
   outcome specialist from the same pretrained encoder, one model per cell
   covering all horizons. Record skipped/failed cells, not just successful fits.
   The pilot is for compute planning, not a substitute for the full-panel
   contrast. A pooled-checkpoint-initialized specialist arm is secondary.
   The data module accepts an explicit secure
   `outcome_label_cache_dir`: post-eligibility train/tuning survival labels
   are keyed by fine cohort, outcome, split, label/eligibility rules and source
   file path/size/mtime. The first fit builds the cache; subsequent local/PHAIR
   fits and multi-outcome runs can reuse it. Held-out labels cannot enter this cache. Benchmark cold
   versus warm startup in the pilot; sequence-tensor loading and GPU epochs
   are not cached. The existing `finetune_all_horizons` entry point exposes
   `--label-cache-dir` but remains a *post-pooled adaptation* experiment, not
   the independent-encoder primary specialist grid.
4. **Factorial training-pool attribution:** One-disease/all-outcomes and
   all-diseases/one-outcome models distinguish the two pooling axes. These are
   worthwhile if compute permits after the direct specialist-vs-pooled contrast;
   they are not the router source-only models.
5. **Encoder transfer:** Audit PHAIR checkpoint/vocabulary/time compatibility,
   then run local-only, PHAIR direct and PHAIR + domain-adapted encoders under
   at least shared BCE and the selected router. This identifies whether broad
   pretraining substitutes for or complements supervised sharing.
   To claim that pooling still helps after PHAIR, repeat the matched
   pooled-BCE-versus-specialist comparison with a common PHAIR-derived encoder
   on both sides. Use the same estimable-cell roster and held-out risk sets as
   the local-encoder grid. A support-stratified PHAIR specialist pilot can
   budget compute, but a pilot alone cannot support a full-panel interaction
   claim. PHAIR direct and PHAIR plus local DAPT need not both receive a full
   specialist grid unless the paper claims the pooling effect separately for
   both; prioritize the selected PHAIR-derived encoder for that grid.
6. **Final lock:** Freeze model choices, estimands, support threshold, outcome
   families, patient bootstrap/hierarchical specification and the tabular
   comparator contract before held-out evaluation. Then make one paired
   held-out results table for the paper.

## Bayesian primary analysis

For each estimable disease × outcome × horizon, estimate the patient-paired
performance difference between pooled shared BCE and its same-encoder
specialist; use training primary-event count as the scarcity axis. Model
nonlinear rarity effects with disease and outcome partial pooling, and allow
clinical-family variation rather than forcing a common benefit. Report the
population-weighted mean, equal-cell mean, lower-tail harm and the proportion
of cells helped. Repeat as a nested contrast for router minus pooled BCE and
as a practical total contrast for router minus specialist. AUC, Brier and
calibration should be treated as distinct estimands. No tuning-selected
bootstrap interval is labelled as held-out confirmation.

## Parallel data and paper track

The user's tabular feature matrices can proceed in the separate repository
while GPU training runs. Table 1 history and contact context should be derived
from the same raw pre-index MEDS event stream after verifying the specialty and
region code prefixes and event multiplicity. The older `SDS_kontakter` R
extractor is not the primary route for these rows. The MEDS summarizer now
supports optional specialty and region prefix keys, but the server vocabulary
must be inspected before those keys are configured.
The patient demographics, EHR2MEDS token summaries and quality-registry
PS/risk summaries likewise require refreshed, ID-audited exports before Table 1
and the cohort flow can be finalized. Their current manuscript numbers remain
provisional wherever marked.
