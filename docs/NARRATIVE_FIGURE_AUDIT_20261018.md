# Whole-manuscript narrative and figure audit

## Central argument

Many clinically important disease–outcome problems in haematology have little labelled experience individually. Test whether borrowing across outcomes, across diseases and from population pretraining improves prediction, determine its limits, and compare the resulting clinical model against credible alternatives. Nation-scale reuse is the widest extension of this argument, with implications for shared Danish model-development infrastructure.

Keep three roles explicit: supervision scope (specialist/disease-pooled/outcome-pooled/joint), encoder origin (local/national/adapted national), and routing (an architectural modifier). The final clinical model is selected on tuning data; it is not necessarily the local joint model. Existing pipeline limitations documented in NATION_SCALE_PAPER_PLAN_20261018.md remain unresolved by prose edits.

## Reading order and figure roles

| Figure | What it establishes | Boundary / action |
| --- | --- | --- |
| Paradigms and information axes | Why broader records and shared supervision differ from isolated clinical scores and tabular tasks | Preserve both original schematics. Nation-scale encoder origin is described in caption/text; an optional transfer inset belongs in the national plate. |
| Predictability atlas | Absolute discrimination across clinical risks and horizons | Keep main: it establishes the clinical landscape before comparative gains. It does not establish a benefit from pooling. Read with support, missing cells and supplementary calibration. |
| Headline comparisons | Whether a shared model improves on credible alternatives | Keep main. Its A/B comparisons overlap with the tabular plate deliberately: headline breadth versus calibration/domain detail. The final model identity needs to propagate from a tuning lock; current labels are fixed arms. |
| Tabular comparison | Domain-specific gains and probability calibration | Keep main for calibration, not a second recital of the headline average AUC. D/E are examples, not evidence that all cells are calibrated. Example selection must be tuning-locked. |
| IPI risk groups | Clinical meaning on a score's actual endpoint and population | Keep main. Other indices remain supplementary. Explicitly distinguish tuning-matched specificity from achieved test specificity and survival from event-free survival. |
| Pooling plate | Independent and conditional contributions of disease/outcome sharing | Keep main. Separate observed gains from biological mechanisms; the support curve is associative. Interaction is presently in text/full tables, not directly drawn in this plate: a compact interaction estimate could eventually replace a redundant domain summary, but this is not implemented. |
| National transfer | Direct reuse, adaptation increment and downstream label requirements | Keep main, before routing. The existing A–E plate is sufficient. Median/IQR across cells are task spread, not confidence bounds; no statistical equivalence claim follows from overlapping curves. |
| Attribution | What the fitted model uses globally and for an individual | Keep main and preserve the user's improved design. Attributions do not establish why pooling helped. Restricted-input evaluations are interventions on inputs to a fixed model, not retrained source-value experiments. The synthetic individual is illustrative, not an observed clinical case. |
| Routing, strategy, subgroup and embedding plates | Architectural increments, practical alternatives and descriptive diagnostics | Supplement with main-text results. UMAPs and gate weights do not prove transfer; broad embedding galleries should not displace the central comparisons. |

The current sequence is defensible: show clinical performance, then isolate its sources. The new Results roadmap makes that choice explicit. If the national result becomes the dominant empirical finding, move its whole section and figure before pooling in the final edit; do not decide based on which synthetic result looks strongest.

## Edits made in this pass

- Added a Results roadmap and an explicit unresolved final-model identity marker.
- Removed the illustrative 25% national label-saving claim from Key findings.
- Distinguished coverage of many modelling tasks from demonstrated replacement of deployed tools.
- Reframed headings to remain valid for null or heterogeneous findings.
- Corrected the risk-group figure fallback text from event-free to overall survival.
- Removed the inference that outcome-specific attributions explain the mechanism of positive transfer.
- Distinguished temporal retrospective evaluation from actual deployment.
- Connected the conditional conclusion to both shared supervision and reusable national representations.
- Reconciled stale figure-placement instructions with the actual assembly.

## Remaining scientific/editorial gates

1. Central final-model selection and propagation must be implemented before final clinical outputs; do not relabel local contrast numbers as national results.
2. National training scale, provenance and data overlap need verified reporting, with PHAIR credit.
3. Replace placeholders with actual positive, null and harmful findings; do not leave the optimistic abstract, Key findings or conclusion as unqualified results.
4. Final Table 1 needs supervisor sign-off. Clinical relevance requires calibrated probabilities and matched endpoint/population comparisons, not ranking alone.
5. Table 2 currently contains many contrasts and a hierarchical label beside arithmetic estimates. Preserve the distinction between estimands; consider moving the full table to the supplement once a compact main table can be populated with the actual primary claims.
6. Outcome definitions and treatment-line estimation belong in the detailed definitions supplement; main Methods already states the decision time, register-derived proxies, ascertainment and competing risks. Maintain those caveats.
7. Final figure captions and legends must name the locked model and distinguish CI, credible band and across-task IQR. Prototype numbers and synthetic patient examples remain explicitly labelled.

This pass reviews internal logic and figure alignment, not a fresh external reference verification. No new literature-based factual claims or citations were added.

## Methods placement and Discussion follow-up

Keep intended use, cohort/split, pretraining overlap/cutoff, outcome eligibility, supervision scopes, transfer design, final-model selection, benchmark input differences, index matching, primary metrics and panel estimands in the main paper: these determine the interpretation.

Moved the full sample-size rationale and precision discussion to Supplementary Methods, leaving the estimability threshold and its limitation in main text. Shortened the main threshold-analysis and attribution paragraphs; full definitions and checks remain supplementary. The regional grouping rule is explicitly retained there. Architecture dimensions, schedules, code lists, treatment reconstruction and detailed sensitivity specifications remain supplementary.

Added Discussion coverage of versioning, validation of each downstream application, shared failure modes, total compute versus downstream label efficiency, and uneven benefits for rare or sparsely recorded populations. Result-dependent points still needing completion include informative null/negative transfer, direct versus adapted reuse, calibration versus ranking, performance where scores are unavailable, and a defined prospective clinical evaluation. These already have places in the text; fill them with actual findings rather than adding repetitive general paragraphs.

## Superseding revision after author feedback

The earlier modest Methods reduction was insufficient. Main Methods now has six subsections and approximately 900 whitespace-delimited words versus 2,500 before this revision. There is no sample-size subsection. Detailed clinical-index comparison rules were relocated to the evaluation supplement; national training budgets were retained in the model supplement. Main text retains the clinical design, crucial ascertainment/temporal caveats, controlled contrasts and interpretation of evaluation. The existing supplementary sample-size rationale remains supplementary.

Removed the added infrastructure-management Discussion paragraph. The scientific discussion now emphasises three questions: which previously unsupported predictions become feasible; when disease/outcome sharing provides complementary rather than redundant or harmful information; and whether a Danish national encoder can support specialised prediction. It explicitly distinguishes a trainable rare-task model from adequate independent validation for clinical use. This supersedes the earlier recommendation to expand the Discussion around versioning and maintenance.
