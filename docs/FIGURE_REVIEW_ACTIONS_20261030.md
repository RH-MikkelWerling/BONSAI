# Actions from the author's figure review

Received 30 October 2026. Source: `opera_figure_review_notes.json`, exported at
13:49:53 UTC (14:49:53 Copenhagen). The original 98 entries are preserved
verbatim in [the received JSON](FIGURE_REVIEW_USER_NOTES_20261030.json).
All 98 IDs match the 99-entry catalogue. Only
`stress_headline_all_variants` has no annotation.

[Open the gallery with your notes preloaded](../scratch/paper_reading_20261028/figure_master_review_20261030/index.html).
[Open the per-figure revision tracker](FIGURE_REVIEW_REVISION_TRACKER_20261030.csv).

Implementation follow-up: all received comments now have recorded actions, revised
assets and remaining evidence in the tracker. Open the [revised gallery](../scratch/paper_reading_20261028/author_revision_20261030/index.html) and
[implementation report](FIGURE_REVISIONS_IMPLEMENTED_20261030.md). Main/supplement
choices specify placement, not scientific approval. Historical tuning and synthetic
layout evidence remain explicitly distinguished. Model selection and frozen
endpoint definitions are unchanged. Newly generated clinical rehearsal recipes
now use baseline-only cohort-size matching with independent development anchors.

## Decisions received

| Decision | Entries |
|---|---:|
| Move to main | 11 |
| Move to supplement | 23 |
| Revise differently | 40 |
| Hold for results | 2 |
| Accept proposed changes | 1 |
| Keep original | 1 |
| Omit / archive | 12 |
| Notes without a selected decision | 8 |

The main candidates are `modeling_paradigms`, `information_axes`, `fig_pooling`,
`fig_hierarchy`, `supp_horizons`, `fig_external`, `fig_headline`,
`clinical_nccn_ipi_risk_groups`, `proposed_clinical_operating_points`,
`fig_atlas` and `supp_calibration_maps`. These are candidates for plates or
components of plates, not an automatic decision to submit eleven full figures.
`router_mechanism` is an additional conditional main candidate, pending results.
The calibration-map and atlas promotions must remain visible in subsequent
storyboards rather than being lost to the earlier six-slot proposal.

The original pre-LDH system remains the baseline. Retain original gallery
embeddings, clinical plates and attribution structure wherever requested.
Archived layout probes remain available as validation assets but stay out of
the manuscript. The original router is removed from the model overview, while
its audit provenance is retained separately.

## Scientific questions raised by the notes

### Hierarchy summaries and the model contrasts

The current `figure_hierarchy` panel C does use **median and IQR across cells**.
This is a descriptive spread, not a confidence interval. Panel B fits a
quadratic in log10 training events with a cell-bootstrap ribbon. It is not
a spline or a patient-bootstrap estimate of a learning curve.

The separately trained contrasts are:

| Figure term | Actual comparison | What it can establish |
|---|---|---|
| Own correction | Own-task-only control minus global-only control | Difference between two trained architectures |
| Borrowing | Full routed model minus own-task-only control | Increment from allowing borrowing in a separately trained model |
| Both | Full routed model minus global-only control | Combined architectural difference |

These are not a within-fit additive decomposition of AUC. The frozen mechanism
audit answers a complementary question by intervening on components of one
fitted model. Figure labels and a short control key need to make that distinction
readable. The exact numbered crossover currently drawn by the renderer should
be removed rather than interpreted as an event-count threshold.

### Counting cells helped and harmed

For pooling and strategy panels, count positive, negative and exact-zero
paired point differences on a declared shared cell/horizon roster. Also give
the denominator and unavailable count. If displaying the number of intervals
wholly above/below zero, name it separately. A near-zero gain or an interval
crossing zero does not establish absence of negative transfer.

### Support and the cells not evaluated

The support requirement limits which **disease-specific performance estimates**
are reported. It does not mean the model cannot predict those patients or that
their supervision was unused. Distinguish unavailable fits, insufficient
evaluation events, unavailable inputs and clinically inapplicable cells.

The test population can support more estimates, but its event counts and
interval widths remain empirical questions. Include the full support display,
the existing prespecified support-cutoff sensitivity and whole-cohort outcome
analysis. Do not lower support cutoffs because the resulting gains look better.
Larger evaluable low-resource coverage helps assess the scope of the claim.
It does not guarantee either model benefit or adequate precision.

### Two-year DLBCL treatment failure

The repository contains a `treatment_failure` outcome and 730-day predictions
are part of the configured horizon grid. However,
`opera/R/build_composites.R` defines this composite from second-line treatment
and overall survival. The HemaSphere endpoint discussed in the review includes
relapse as well as second-line treatment and death. Registry relapse can inform
the second-line identification pipeline, but that does not establish inclusion
of every relapse in the frozen composite.

A two-year comparison using the existing composite is a useful candidate
supplementary analysis if the necessary frozen predictions and clinical score
records are available. Label it **second-line treatment or death**, with the
definition difference stated. Do not silently redefine the endpoint, claim
an exact replication, or apply the published probability cutoffs to a different
model/endpoint. The refreshed clinical recipe now adds this companion when the frozen predictions are available. It is excluded from the one-year overall-survival headline.

### IPA, Brier and calibration

For a cross-outcome mechanism summary, IPA supplies an outcome-normalised skill
measure already present in the evaluation. Raw Brier differences can remain a
companion table or sensitivity, avoiding duplicate panels. The metric and
aggregation roster should be declared before test evaluation. Direct clinical
operating-point counts and achieved survival answer a different question and
should not be replaced by either metric.

### Distinct risk landscapes and the global correction

Add neutropenia as the author's clinically motivated contrasting example,
subject to support and exact endpoint labelling. Use common patient coordinates
when the point is to compare where risk occurs. Explain which representations
are shared and which are outcome-specific.

Keep an arrow-only companion and arrows on the main maps as requested, with
legends that identify the projected correction. Similar projected directions
do not isolate the global adapter or demonstrate predictive benefit. That
interpretation remains with the raw-vector decomposition and frozen mechanism
audit. Do not select examples after inspecting test-set gains.

## Revision batches

| Batch | Concrete changes | Evidence needed before sign-off |
|---|---|---|
| Schematics | Match joint connection weights, replace guaranteed-performance wording, try one compact pretraining cue, align router drawing with selected implementation | Selected configuration and exact readout checked. Shared path unscaled, global correction outside mixture, own-source handling and gates correctly drawn |
| Clinical plates | Retain original layouts, standardise category colours and legends, reserve space below PR panel A, use new group construction, include all supported indices | Refreshed clinical counts, eligibility, cutoff diagnostics and paired tables. Keep equal-size and KM-matched questions separate |
| Pooling and hierarchy | Label individual outcomes on exhaustive maps, improve grid/legend spacing, add common-roster helped/harmed counts, supply explicit trained-control key | Paired results, support and extreme-gain audit. Descriptive IQR distinct from patient uncertainty |
| Router explanations | Lead with mechanism figure if justified, retain probability companion, simplify routing/equivalence presentation, rethink pale borrowing-map summaries | Actual mechanism, source-removal provenance and geometry reports. Do not amplify weak effects with arbitrary colour scaling |
| Headline and tabular | Keep original headline grammar, show relevant selected models, merge complementary benchmark/horizon/calibration material, share legend across A/B | Common patient and cell rosters, model identities, national results if available, X/XP coverage audit |
| National | Consider disease rows ordered by cohort size, add Overall row to companion, fix label outlines and crossing lines | Actual downstream fits and matched estimates. No claim of benefit from previews |
| Patient maps | Retain original composition, add labelled disease centroids or another auditable disease key, choose contrasting risk landscapes, fix IPI high-risk contrast | Correct projection provenance, current clinical strata and final outcome selection |
| Attribution and support | Retain original structure, increase panel B/top spacing, title legends, keep verified-code audit, use multi-horizon recent-record pages, unify missingness symbols | Fresh IG, code mappings and layout check. Synthetic patient panels remain explicitly labelled |

## Already implemented separately, awaiting refreshed server figures

- The recent-record comparison supports all six horizons and clinical families,
  with paired uncertainty and explicit estimable-outcome counts. See
  [the refresh instructions](CLINICAL_AND_RECORD_GAP_REFRESH_20261030.md).
- The explicit clinical refresh mode supplies cohort-size-matched strata,
  separate development-KM-matched Low groups and independent high-specificity
  classifiers. Future generated recipes now request this mode explicitly. Already running server processes and existing recipe files remain unchanged.
- Whole-cohort evaluation now exposes missing model/task rows and includes
  within/between-disease AUC diagnostics. See
  [the XGBoost audit](WHOLE_COHORT_X_AUDIT_20261030.md).
- Cross-family reconciliation now avoids repeated full-table joins and emits
  progress. A running old process retains its old code. See
  [the runtime note](CROSS_FAMILY_AUDIT_RUNTIME_20261030.md).

The completed local redraw and per-comment changes are documented in the
[implementation report](FIGURE_REVISIONS_IMPLEMENTED_20261030.md). The latest
server results still require an identifiable fresh redraw and empirical audit.
The [CPU refresh command](FIGURE_REVISION_SERVER_REFRESH_20261030.md) prepares
that review without repeating source-removal inference or training.
