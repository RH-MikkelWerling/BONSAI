# Complete figure review — original pre-LDH style

Implementation follow-up: [the revised gallery](../scratch/paper_reading_20261028/author_revision_20261030/index.html) and [per-comment tracker](FIGURE_REVIEW_REVISION_TRACKER_20261030.csv) now record the changes requested in the author’s exported notes. This document preserves the original review baseline.

30 October 2026. The user prefers the original figure system. That is the baseline. The simplified LDH plates are not used in this review. Existing result figures and renderer code are unchanged.

[Open the complete review gallery](../scratch/paper_reading_20261028/figure_master_review_20261030/index.html). The gallery offers full-resolution figures, source PDFs, individual decisions and notes, plus JSON export/import of the user’s review.

The catalogue accounts for all 69 PNGs in the downloaded bundle, every one of the 39 IDs in the main/attribution recipes, the mean/zero sensitivity, all ten clinical-score series, source schematics, mechanism companions and proposed additions. Clinical aliases are not counted as independent findings. Dynamic pagination or failed clinical partitions can change the number of actual output files.

## Main story and placement

Sharing information is the organising argument. The story moves from what can be shared, to which supervision helps, to the conditions under which specialisation and borrowing help, to information outside haematology, then to practical and clinical performance. Conventional calibration and unfavourable XGBoost comparisons remain central evidence. Projections and gate weights are supplementary diagnostics, not proof of transfer.

| Proposed main slot | Question | Existing figure language |
|---|---|---|
| 1. One clinical question, three ways to share information | What is being shared, and what stays comparable? | modeling_paradigms, information_axes, supervision_grid |
| 2. What supervision helps a prediction? | What is gained from other diseases, other outcomes and their combination? | fig_pooling, supp_pooling_cells, supp_horizons |
| 3. When does the shared model benefit from corrections? | How do own-task and borrowed corrections relate to available training events? | fig_hierarchy, supp_routing, router_mechanism |
| 4. Does sharing beyond haematology help? | What do national pretraining and subsequent cohort adaptation add? | fig_external, supp_national_where |
| 5. How does the resulting model compare with practical alternatives? | Does a consolidated predictor remain competitive with specialists and strong tabular models? | fig_headline, fig_tabular, fig_benchmarks, supp_model_overview, fig_calibration |
| 6. What difference could this make for a clinician? | How many patients or events are identified, and at what achieved risk or specificity? | clinical_nccn_ipi_risk_groups, clinical_nccn_ipi_clinical_score, proposed_clinical_operating_points |

This six-slot plan is an editorial proposal. The headline/tabular comparison can remain two main figures if combining it makes either unreadable. It is not permission to omit negative results or force a venue-specific display limit.

## What improves quality while retaining the style

- Preserve the indigo/teal/ochre palette, the familiar heatmaps, interval forests, empirical curves and illustrated schematics. Avoid decorative redesign or converting all figures to generic bars.
- Make every panel answer one question, with an explicit comparator, metric, horizon, population and uncertainty definition. Move repeated methodological prose to the caption.
- Standardise panel headings using the shared style at final placement size. Aim for readable 8–9 pt axis/legend type and roughly 10–11 pt panel headings as a project target, not a claim about journal requirements. Never shrink a dense atlas until labels become tiny.
- Reserve real grid cells for legends, colourbars and clinical operating-point tables. Increasing the canvas without preserving placed type size does not solve layout.
- Reuse disease/domain order and colour semantics. Keep missing, zero, unsupported and inapplicable visually distinct. Share signed scales across directly comparable maps and mark any saturation.
- Use paired uncertainty for performance comparisons. Keep descriptive IQRs and row-bootstrap trends explicitly different from patient-bootstrap intervals. Do not fabricate precision for an absolute mean or clinical count.
- Export vector text/lines with appropriately rasterised dense points. Keep component panels editable. The supplied LDH author PDF requests combined and individual multipart figures and editable vector artwork. Its flow-diagram requirement is an editable Word/PowerPoint file. These are submission packaging tasks, not reasons to adopt the LDH redesign.

## Priority issues before any figure is called final

1. The current `source_router.tex` schematic omits the revised outside-mixture global correction and visually connects gating to the unscaled shared path. Correct the diagram after the selected-fit mechanism audit, preserving the illustration style.
2. The hierarchy’s event-count crossover is descriptive, not a clinical threshold. Name separately trained controls precisely and match the caption to the actual horizon scope. Its harm-selection panel does not by itself establish rescue.
3. Clinical group construction is unresolved. Layout fixes cannot validate crossed anchors. Report every index, keep native categories and separate low/high questions if that protocol is chosen. Do not advertise matching evaluation risk/specificity from a training match.
4. Mean/zero disease-source effects need audited provenance and interpretable fixed scales. A pale map may be the correct scientific finding. Gate mass or a dramatic projection cannot supply a missing source-effect result.
5. The attribution recipe uses synthetic patient-episode CSVs for its patient illustrations. Label these panels prominently as synthetic. Fresh IG summaries do not convert the illustrated patient into an observed explanation.
6. National figures and the local mechanism figures have only layout previews locally. They cannot receive empirical sign-off until actual server outputs arrive.
7. Several useful additions currently exist only as tables or proposals, including a whole-cohort result figure, a complete clinical operating-point summary and a support-cutoff sensitivity figure. They are not already produced by the current figure recipes.

## How to use the gallery

Open the main-story links first, then filter by question or placement. Original exports appear first whenever available. Pre-LDH 29 October template updates are expandable comparisons. Every synthetic proof has an explicit warning. Stress probes and unreferenced legacy sources are listed in the archive rather than silently promoted.

Choose a decision and add panel-specific notes. Export the JSON before closing if the review needs to be preserved or sent back. Local browser storage is a convenience and may be restricted for file URLs. No web service or upload is required.

## Per-figure decisions

### Disease and outcome sharing — `fig_pooling`

**Main · 2 · Refine.** The heatmap, aligned disease forests and support plot already tell the core story in our visual language.

Retain the four-panel layout. Give conditional J−O/J−D comparisons a visible place. Keep one disease/domain order across comparable maps. Identify means/CIs separately from descriptive medians/IQRs. Increase legend space beneath B without reducing data area.

Pending: Fresh paired summaries and extreme-specialist audit.

Sign-off: All plotted models and horizons named, B uses exported paired uncertainty, no unsupported equivalence claim, and unusually large gains are accompanied by absolute specialist AUC and support.

### Every supervision contrast — `supp_pooling_cells`

**Supplement · Refine.** Exhaustive disease × outcome maps reveal heterogeneity hidden by averages.

Keep the original grids. Paginate by clinical domain or use landscape pages at readable type size. Add a labelled OPERA−joint panel as an architectural contrast, keeping it separate from the supervision factorial. Avoid shrinking five maps onto one page.

Pending: Fresh paired cell tables.

Sign-off: Every intended outcome and disease is present, unsupported cells differ from zero, and signed scales are shared for the same metric.

### Sharing across prediction horizons — `supp_horizons`

**Supplement · Layout acceptable.** The original four forest panels are readable and make horizon effects directly comparable.

Preserve the design and full negative range. Reserve room for long domain labels and add supported-cell counts in the caption/table. Use a companion common-roster sensitivity rather than interpreting changing task composition as a time effect.

Pending: Fresh horizon-specific estimates and support.

Sign-off: All six horizons, interval definitions, model identities and denominators are correct. Layout is acceptable provisionally, not a final-data sign-off.

### Own correction and borrowing versus training support — `fig_hierarchy`

**Main · 3 · Revise interpretation.** The scatter, two coloured support trends and support-band forest communicate partial pooling well.

Keep this visual grammar. Remove the precise crossover line/number. Label own−global, full−own and full−global explicitly as separately fitted models. Move the harm-selection scatter to the rescue supplement or name its global-only baseline precisely. The quadratic log-event trend and row-bootstrap ribbon are descriptive.

Pending: Fresh trained controls, support summaries and mechanism audit.

Sign-off: No claim of an event-count threshold, no causal learning curve, no additive AUC decomposition, and caption matches the actual horizon scope.

### Routed versus joint and adapter controls — `supp_routing`

**Supplement · Refine.** Paired domain forests show both helpful and unhelpful effects without decorative emphasis.

Keep the forests. Separate primary revised-router controls from the original-router bank sensitivity. Replace aliases with a short descriptive key. Use original-versus-revised as sensitivity, not a second opportunity to pick a winner.

Pending: Primary AB control identities and matched rosters.

Sign-off: Each row identifies architecture, encoder and contrast; crossing zero is not called equivalence.

### How the gate allocates weight — `supp_router_usage`

**Supplement · Refine.** The disease/outcome bank matrices and allocation bars provide useful structural diagnostics.

Keep the matrices and all source columns, including source-only diseases. Make bank-size and flat-allocation references explicit. Label the last panel as entropy of mean weights, not patient-level stability. Do not equate gate mass with useful transfer or the null slot with the unscaled shared path.

Pending: Fresh selected-fit route export.

Sign-off: Rows and columns match the selected checkpoint, bank denominators are correct, own sources are identifiable, and per-patient stability claims use per-patient evidence.

### Which disease-source corrections matter? Mean replacement — `fig_borrowing_map`

**Supplement · Hold for evidence.** The original square source × target layout is the right idiom for an exhaustive dependence map.

Keep it, but use interpretable common units and a fixed shared scale across mean/zero. Display unavailable cells separately. Prefer raw estimates plus uncertainty and a clearly separate shrinkage diagnostic. Remove normalized positive-share or lineage summaries when the underlying effects are near zero.

Pending: Fresh source-removal provenance and mean/zero comparison.

Sign-off: Exact source direction and intervention are labelled. Effects are fixed-fit reliance, not donor training effects. Weak effects remain visually weak.

### Disease-source effects in detail — `supp_borrowing_detail`

**Supplement · Refine.** The original forest and domain-map combination permits a detailed audit beyond the overview.

Keep the composition but split it over pages if necessary. Give full-source removal, single-source removal and leave-one-in controls distinct names. Do not equate the ablation's none condition with a separately trained own-only fit. Use the same map scale as the overview.

Pending: Fresh ablation levels, pair estimates and controls.

Sign-off: Baselines, retained global/outcome paths, pair denominators and mean replacement references are explicit.

### Disease-source dependence: zero-replacement sensitivity — `fig_borrowing_map_zero`

**Supplement · Hold for evidence.** A matched map makes the sensitivity transparent.

Retain the same order, scale and units as mean replacement. State that zero removal changes the average offset as well as individual variation. Avoid presenting this as independent replication.

Pending: Fresh zero exports and same target roster as mean.

Sign-off: Matched mean/zero provenance, no automatic colour amplification and no claim that the two interventions are interchangeable.

### Zero-replacement details — `supp_borrowing_detail_zero`

**Supplement · Refine.** The original detailed view supports diagnosis of source effects.

Match the mean-detail layout and baseline key. Keep uncertainty readable. If duplicating the entire page adds little, retain the estimates as a table plus a compact mean-versus-zero comparison.

Pending: Fresh zero-export detail tables.

Sign-off: The sensitivity remains available even if its full duplicate plate is moved to the archive.

### Overall performance and consolidation — `fig_headline`

**Main · 5 · Refine.** The old reference ladder, per-cell comparison and horizon panels are more informative than the LDH simplification.

Keep the familiar arrangement. Make the primary model explicit in every panel. Include the relevant model families and encoder variants in a compact absolute-performance panel, with the full list in the companion overview. Avoid mixing joint and OPERA under one identity. Move duplicate clinical/subgroup panels out if they crowd the central comparison.

Pending: Fresh primary/national roster, clinical protocol and subgroup tables.

Sign-off: No stale clinical panel, no missing panels disguised as results, both tabular strategies retained, and no national winner selected on test performance.

### All declared models on a common roster — `supp_model_overview`

**Supplement · Layout acceptable.** The point distributions and model summaries show the portfolio honestly.

Keep the original horizontal structure. Group primary models, encoder variants and sensitivity controls with whitespace. Use identical cell rosters for directly compared summaries and explain when points describe tasks rather than patients.

Pending: Final model roster and common supported cells.

Sign-off: All declared variants are visible, descriptive means are not labelled bootstrap intervals, and changing availability is reported.

### Comparison with strong tabular predictors — `fig_tabular`

**Supplement · main candidate · Refine.** Domain forests, the event-support scatter and horizon profiles are useful and stylistically established.

Keep the four panels. Spell out the baseline for every gain, use the same clinical-domain taxonomy as pooling and leave a dedicated legend band below D. Show pooled XGBoost as well as per-task XGBoost. Retain negative neural differences.

Pending: Fresh X/XP/neural reconciliation.

Sign-off: Common risk sets/metrics verified, full x ranges visible, no unexplained groupings, and the support trend is not described as an experiment varying training size.

### Discrimination and conventional calibration — `fig_benchmarks`

**Supplement · main-panel source · Refine.** The old domain forests and reliability examples connect ranking with probability quality.

Retain this style. Use full reliability curves with counts and uncertainty rather than examples represented by one bin. Choose examples by a rule fixed before test. Avoid duplicating the headline and tabular plates in full.

Pending: Fresh calibration and prespecified example cells.

Sign-off: Raw calibration, readable examples, clear AUC versus slope interval definitions, and no example chosen for an attractive held-out result.

### Calibration across models and clinical domains — `fig_calibration`

**Supplement · Refine.** The original slope/bias forests are an important counterweight to AUC-only reporting.

Keep conventional plots. Preserve finite extremes, using a labelled expanded view or explicit compressed scale rather than clipping. Show estimation failures and cell counts. Replace any one-point reliability example with a supported prespecified example. Do not silently recalibrate the reported model.

Pending: Fresh raw slopes/intercepts and reliability curves.

Sign-off: Ideal slope 1 and zero bias clear, all finite intervals retained, and IQRs are not called CIs.

### Absolute predictability — `fig_atlas`

**Supplement · Layout acceptable.** The support scatter and domain summaries provide context for the size of gains.

Keep the original two panels. Explicitly label the fixed joint reference or add a separately named OPERA companion. This is absolute performance, not evidence that sharing caused improvement.

Pending: Declared model identity and fresh atlas.

Sign-off: Model and aggregation named, zero/support exclusions accounted for, and task variation distinguished from uncertainty.

### Every disease and outcome: absolute AUC — `supp_atlas_matrix`

**Supplement · Refine.** An exhaustive matrix is valuable and should remain available.

Preserve all cells and make grey unavailable distinct from below-chance AUC. Paginate into readable landscape domain blocks, with repeatable legends and support alongside. Do not reduce type to fit one page.

Pending: Fresh atlas and coverage mask.

Sign-off: All 69 endpoints and declared diseases covered, final-size labels legible and exclusions traceable to coverage.

### Which disease–outcome questions can be estimated? — `supp_coverage`

**Supplement · Layout acceptable.** The categorical grid reveals what a headline average leaves out.

Keep the full matrix and counts. Distinguish too few training events, missing exports, one-class evaluation and confirmed fitting failures. Do not label absent specialist predictions as failed fits without a receipt.

Pending: Fresh fit/support reasons.

Sign-off: Legend totals reconcile, states are mutually interpretable and this comparison's roster is not confused with the model's ability to predict.

### Training-event scarcity across the whole grid — `supp_support`

**Supplement · Layout acceptable.** The full grid and cumulative distribution directly motivate sharing.

Keep the original all-cell view rather than cropping to supported cells. Explicitly display zero, unknown and inapplicable counts. Annotate evaluation support separately from descriptive training-event bands.

Pending: Full training-support export including zeros.

Sign-off: Full denominators retained and the five-case/control criterion is not misrepresented as a training-event threshold or proof of clinical precision.

### Corrections in tasks where joint sharing was harmful — `supp_rescue`

**Supplement · Revise interpretation.** The original scatter and small forest communicate the heterogeneity better than a complex redesign.

Keep that layout. Name the x/y baselines and the tuning-defined strata. Mark tuning-on-tuning results descriptive. On test, retain the original tuning strata rather than redefine hurt cells using test outcomes.

Pending: Frozen tuning-defined strata evaluated on held-out data.

Sign-off: No circular claim of rescue, paired estimates retain the same tasks and supported denominators are shown.

### One shared model versus a selected portfolio — `supp_strategy`

**Supplement · Hold for evidence.** The old scatter/selection-bar layout makes consolidation concrete.

Keep it. Reserve more space for A and align the legends. Include OPERA in an expanded selection policy only as a separately named, development-frozen strategy. Do not compare selection frequencies for one roster with performance of another without labelling it.

Pending: BEST_TUNING predictions and a declared selection roster.

Sign-off: The previously missing performance panel is populated by selected-model predictions, selection uses tuning only, and all models share the reported evaluation population.

### National pretraining and cohort adaptation — `fig_external`

**Main · 4 · Await results.** The original three-origin comparison and label-efficiency panels fit the sharing argument.

Keep this design. Separate joint and OPERA within each encoder origin, show paired national−local and adaptation increments, and use observed training fractions. Avoid interpolated sample-saving claims without uncertainty and a declared rule.

Pending: National downstream models and actual fraction runs.

Sign-off: No national values filled from synthetic previews, all six planned arms accounted for, matched cells and actual training fractions identified.

### Where national information helps — `supp_national_where`

**Supplement · Await results.** A compact forest is appropriate for prespecified heterogeneity.

Keep the original forest style and support/coverage strata. Report all declared subgroups with uncertainty, not only beneficial ones. Do not confuse fewer vocabulary matches with fewer clinical events.

Pending: National subgroup contrasts and coverage.

Sign-off: No empty placeholder presented as a null result, no post-test subgroup search and no inference of national benefit before downstream evaluation.

### Patient-subgroup performance — `fig_who_benefits`

**Supplement · Refine.** Paired forests keep the subgroup story clinically readable.

Keep the old forest arrangement. Group rows by covariate with whitespace, include subgroup denominators and show uncertainty. Use one declared primary identity and do not conflate patient strata with task-level training-support bands.

Pending: Fresh patient-level subgroup aggregates.

Sign-off: Every subgroup uses documented matching/weighting and missing subgroup data remain visible.

### Geographic heterogeneity within Denmark — `supp_regions`

**Supplement · Refine.** The map plus quantitative intervals balances geographical orientation and effect size.

Keep it. Increase vertical space before the lower row and retain full x ranges in every forest. Use consistent model colours in all panels and separate colourbars for different quantities. The maps are within-system heterogeneity, not independent geographic validation.

Pending: Fresh region aggregates, geometry and missing-region counts.

Sign-off: Region denominators and missingness reconcile, lower labels do not clip, and legend meanings do not change unnoticed across panels.

### Patient representation and predicted risks — `fig_patient_map`

**Supplement · Refine.** The old seven-panel map is visually engaging, with the same patients recoloured for different questions.

Keep this style and shared bounds. For A, retain a faint pooled cloud with a small number of directly labelled centroids/contours, plus full disease facets as a companion. Do not use 24 competing contours as the only disease key. Separate predicted-risk points from estimated calibration surfaces.

Pending: Fresh selected-fit coordinates, risks and projection provenance.

Sign-off: Every disease is identifiable somewhere, map stage and projection fit are recorded, and geometry is not interpreted as proof of predictive benefit.

### Every disease on the same patient map — `supp_patient_disease_groups`

**Supplement · Layout acceptable.** Highlight-one-disease facets solve the crowded map's identification problem.

Keep all fine diseases on identical coordinates and bounds, with readable n labels. Paginate if necessary. Keep the background faint but visible and do not refit a separate projection in each facet.

Pending: Fresh coordinates and fine-disease membership.

Sign-off: Every observed disease appears, no point omission and no pooled category silently substitutes for fine diseases.

### Status, predicted risk and local calibration on the embedding — `supp_calibration_maps`

**Supplement · Refine.** The original repeated-outcome grid allows direct visual alignment.

Retain the grid but give each row a short, readable label. Split arrows into their dedicated plate. Use common residual limits that include all finite values or visibly flag saturation. Distinguish smoothed observed risk from patient predicted risk.

Pending: Fresh selected-fit predictions, status, vectors and residual estimates.

Sign-off: No clipped values without markers, masks/support explicit, unchanged projection across rows, and conventional calibration reported separately.

### Projected corrections across outcomes — `supp_router_arrows`

**Supplement · Hold for evidence.** A dedicated arrow row is much clearer than arrows buried in a five-row plate.

Keep the current pre-LDH arrow style, common magnification and status key. Add a short numerical similarity summary in the caption. Similar arrows cannot identify the global adapter or establish useful routing.

Pending: Raw-vector audit and fresh shared-coordinate vector exports.

Sign-off: Base/adapted vectors use one fixed projection, raw-vector agreement reviewed, and no claim that an arrow points towards risk.

### Representations across encoder/readout stages — `supp_embedding_comparison`

**Supplement · Refine.** A stage-by-covariate atlas is useful as a descriptive supplement.

Retain small multiples and common style. Show observed category membership rather than an averaged ordinal clinical score. State when stage projections are separately fitted. Separate encoder representations from outcome-specific routed readouts.

Pending: Matched-stage coordinates, patient identities and projection recipes.

Sign-off: No displacement claim between independent projections, stage identities complete, and final routed risk may vary without moving the shared encoder map.

### Recorded information associated with each predicted risk — `supp_attribution_atlas`

**Supplement · Paginate.** The original source/time bars beside the outcome–concept matrix contain substantial auditable information.

Keep the visual language. Split into clinical-domain pages with repeated legends and readable labels. Show expected-direction boxes only where an actual documented expectation exists. Unmapped concepts stay explicitly unresolved. Explain signed covariance shares and reference dependence without calling them causal variance explained.

Pending: Corrected IG, code labels, references and concept support.

Sign-off: All Danish labels auditable, counts and unsupported concepts visible, no unsupported biological sign claims, and no tiny one-page atlas masquerading as readable print artwork.

### Outcome similarity and information-removal effects — `supp_attribution_structure`

**Supplement · Refine.** The original similarity matrix and removal comparison are both strong diagnostic ideas.

Keep both, but split onto separate pages if labels remain small. Label what agreement statistic is used. Treat nested endpoint definitions as a source of expected similarity. Add uncertainty for restrictions or point to the complete paired-effect table.

Pending: Fresh IG agreement and restricted-input results.

Sign-off: Attribution agreement is not positive transfer; removal at evaluation is not retraining without that source; the pooled-outcome estimand is explicit.

### Overview of model-input dependence — `fig_attribution`

**Supplement · optional composite · Refine.** The original multi-panel overview links sources, concepts and a timeline.

Keep it as an optional summary rather than another main figure. Avoid repeating the full atlas and structure plots. The recipe currently points to synthetic_patient_episodes_risk.csv: the patient timeline is a synthetic illustration, not a newly explained clinical patient.

Pending: Fresh IG/restrictions plus explicit synthetic-episode labels.

Sign-off: Synthetic panel labelled individually, concept units explicit, all empirical panels use the same audited checkpoint and no unlabelled mixture of illustration and results.

### Illustrative patient history and multiple risks — `supp_attribution_patient`

**Methods illustration · Design only.** The old timeline-and-bubbles illustration is intuitive and visually coherent.

Retain only if it teaches something the atlas cannot. The recipe uses synthetic_patient_episodes_logodds.csv. Label it Synthetic patient prominently and name the logit/risk scale; do not imply an actual patient-level IG export or sensitivity result.

Pending: Correct synthetic labelling and code/scale verification.

Sign-off: Figure and caption both identify the illustration, all events respect the treatment landmark, and the bubble magnitude legend names the quantity.

### Prediction after hiding the last weeks of the record — `supp_recent_record_gaps`

**Supplement · Layout acceptable.** The forest is an effective, restrained display of recency dependence.

Keep the original forest style, common zero line and all four gaps. Show outcome-domain estimates with actual paired intervals. Preserve all-outcomes and clinical domain labels. This does not move prediction time earlier or simulate retraining.

Pending: Actual paired 7/14/21/28-day removal results.

Sign-off: No synthetic values, horizon and pooled-across-disease estimand explicit, same patients/checkpoint and last-week boundaries verified.

### Outcome-risk maps · page 1 of 6 — `supp_embedding_gallery_1`

**Supplement · Layout acceptable.** The original small multiples cover the full endpoint set without reducing it to selected examples.

Keep the original layout and shared risk scale. Repeat the model, horizon and predicted-risk legend. Shorten panel labels without losing grade/definition. The shared encoder coordinates do not need to move when an outcome-specific routed readout changes risk.

Pending: Fresh selected-fit predicted CIFs on shared coordinates.

Sign-off: All pages accounted for, all intended endpoints included once, correct output probabilities and no map-shape claim about routing benefit.

### Outcome-risk maps · page 2 of 6 — `supp_embedding_gallery_2`

**Supplement · Layout acceptable.** The original small multiples cover the full endpoint set without reducing it to selected examples.

Keep the original layout and shared risk scale. Repeat the model, horizon and predicted-risk legend. Shorten panel labels without losing grade/definition. The shared encoder coordinates do not need to move when an outcome-specific routed readout changes risk.

Pending: Fresh selected-fit predicted CIFs on shared coordinates.

Sign-off: All pages accounted for, all intended endpoints included once, correct output probabilities and no map-shape claim about routing benefit.

### Outcome-risk maps · page 3 of 6 — `supp_embedding_gallery_3`

**Supplement · Layout acceptable.** The original small multiples cover the full endpoint set without reducing it to selected examples.

Keep the original layout and shared risk scale. Repeat the model, horizon and predicted-risk legend. Shorten panel labels without losing grade/definition. The shared encoder coordinates do not need to move when an outcome-specific routed readout changes risk.

Pending: Fresh selected-fit predicted CIFs on shared coordinates.

Sign-off: All pages accounted for, all intended endpoints included once, correct output probabilities and no map-shape claim about routing benefit.

### Outcome-risk maps · page 4 of 6 — `supp_embedding_gallery_4`

**Supplement · Layout acceptable.** The original small multiples cover the full endpoint set without reducing it to selected examples.

Keep the original layout and shared risk scale. Repeat the model, horizon and predicted-risk legend. Shorten panel labels without losing grade/definition. The shared encoder coordinates do not need to move when an outcome-specific routed readout changes risk.

Pending: Fresh selected-fit predicted CIFs on shared coordinates.

Sign-off: All pages accounted for, all intended endpoints included once, correct output probabilities and no map-shape claim about routing benefit.

### Outcome-risk maps · page 5 of 6 — `supp_embedding_gallery_5`

**Supplement · Layout acceptable.** The original small multiples cover the full endpoint set without reducing it to selected examples.

Keep the original layout and shared risk scale. Repeat the model, horizon and predicted-risk legend. Shorten panel labels without losing grade/definition. The shared encoder coordinates do not need to move when an outcome-specific routed readout changes risk.

Pending: Fresh selected-fit predicted CIFs on shared coordinates.

Sign-off: All pages accounted for, all intended endpoints included once, correct output probabilities and no map-shape claim about routing benefit.

### Outcome-risk maps · page 6 of 6 — `supp_embedding_gallery_6`

**Supplement · Layout acceptable.** The original small multiples cover the full endpoint set without reducing it to selected examples.

Keep the original layout and shared risk scale. Repeat the model, horizon and predicted-risk legend. Shorten panel labels without losing grade/definition. The shared encoder coordinates do not need to move when an outcome-specific routed readout changes risk.

Pending: Fresh selected-fit predicted CIFs on shared coordinates.

Sign-off: All pages accounted for, all intended endpoints included once, correct output probabilities and no map-shape claim about routing benefit.

### Clinical scores, tabular models and sequences — `modeling_paradigms`

**Main · 1 · Refine.** The existing illustrated style is retained.

Keep the original illustrated comparisons. Make a few-variable score an example, not a definition of every index. Acknowledge the pooled tabular comparator. Check drawn lookback windows against the fitted recipe. Avoid suggesting that all clinical indices apply to only one disease.

Pending: Source wording, selected-fit identity and count checks.

Sign-off: Vector source and placed-size render agree with the final methods and population ledger.

### Sharing around one target task — `information_axes`

**Main · 1 · Refine.** The existing illustrated style is retained.

Keep the original disease–outcome grid and restrained palette. Distinguish supervision, pretraining and readout correction. Connections are conceptual, not measured source effects. Add national information beyond the grid without drawing dozens more arrows. Replace the blanket higher-attainable-performance implication with an opportunity to learn shared structure; the study tests when sharing actually helps.

Pending: Source wording, selected-fit identity and count checks.

Sign-off: Vector source and placed-size render agree with the final methods and population ledger.

### The four matched supervision scopes — `supervision_grid`

**Main · 1 panel / supplement detail · Layout acceptable.** The existing illustrated style is retained.

Preserve the same target outline and row/column orientation. All four share local pretraining. OPERA has the same supervised grid as joint and differs in readout. Filled cells mean supervision, not AUC or learned routes.

Pending: Source wording, selected-fit identity and count checks.

Sign-off: Vector source and placed-size render agree with the final methods and population ledger.

### Danish sources and pretreatment tokenisation — `model_inputs`

**Supplement · Refine.** The existing illustrated style is retained.

Keep the source cards and patient sequence. Verify token spellings against the actual adaptive vocabulary. Department/region/specialty belong with event-source information. Readability must survive final placement, and an example NPU code must not silently assert an unverified clinical identity.

Pending: Source wording, selected-fit identity and count checks.

Sign-off: Vector source and placed-size render agree with the final methods and population ledger.

### Cohort, exclusions and temporal splits — `cohort_flow`

**Supplement · Refine.** The existing illustrated style is retained.

Keep the original flow. Separate local pretraining from the supervised cohort and national pretraining. Reconcile every count with Table 1 and actual split receipts. The user-supplied LDH guide requests an editable Word/PowerPoint flow diagram at submission, in addition to this source render.

Pending: Source wording, selected-fit identity and count checks.

Sign-off: Vector source and placed-size render agree with the final methods and population ledger.

### The selected router's computation — `source_router`

**Supplement · Needs correction.** The existing illustrated style is retained.

Keep the four-column illustrated style, but update its content. The present diagram omits the revised global adapter outside the mixture and draws a gate connection to the unscaled shared path. Explicitly separate unscaled encoder/global terms, own sources within the mixture, borrowed disease/outcome sources and the null slot. Confirm the drawing against the selected fit.

Pending: Source wording, selected-fit identity and count checks.

Sign-off: Vector source and placed-size render agree with the final methods and population ledger.

### Alternative routing intuition — `source_router_intuition`

**Archive · legacy source · Legacy.** The existing illustrated style is retained.

Not referenced by the current main or supplementary manuscript. Keep as source material, not an extra publication figure or proof of the selected implementation.

Pending: Source wording, selected-fit identity and count checks.

Sign-off: Vector source and placed-size render agree with the final methods and population ledger.

### Historical adapter-bank factorial — `router_source_factorial`

**Archive · legacy source · Legacy.** The existing illustrated style is retained.

Not referenced by the current figure assembly. If retained as sensitivity, identify which original/revised architecture it actually describes and do not present it as the current primary factorial.

Pending: Source wording, selected-fit identity and count checks.

Sign-off: Vector source and placed-size render agree with the final methods and population ledger.

### Alternative readout architectures — `readout_architectures`

**Archive · legacy source · Legacy.** The existing illustrated style is retained.

The manuscript does not currently reference this catalogue. Do not overwhelm the borrowing story with architectures not needed for its controlled comparisons. Preserve the source for technical reference.

Pending: Source wording, selected-fit identity and count checks.

Sign-off: Vector source and placed-size render agree with the final methods and population ledger.

### Historical study-design sketch — `study_design_mock`

**Archive · legacy source · Legacy.** The existing illustrated style is retained.

A mock source is not an evaluated scientific result. Retire from the publication roster; keep only as historical design material.

Pending: Source wording, selected-fit identity and count checks.

Sign-off: Vector source and placed-size render agree with the final methods and population ledger.

### NCCN-IPI · DLBCL · Survival and reclassification — `clinical_nccn_ipi_risk_groups`

**Main · 6 candidate · Protocol hold.** The old paired survival curves, risk tables and horizon forest are worth retaining.

Keep the native index curves and original style. Any model curves or reclassification matrix requiring a full ordered partition must follow the revised protocol, not crossed anchors. Use separate low-risk/high-risk analyses if those are the chosen questions. Retain empty/unsupported groups explicitly rather than manufacturing categories.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### NCCN-IPI · DLBCL · Precision–recall, score distributions and age strata — `clinical_nccn_ipi_clinical_score`

**Supplement · Protocol hold.** The old PR curve and within-index model-risk distributions communicate continuous information well.

Preserve this visual arrangement. Give the PR curve a separate operating-point table below it, rather than text on the data. Reserve a dedicated legend area and four-category label space. Keep achieved specificity and risk visible. The age cross-classification panels depend on the unresolved model-group protocol.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### NCCN-IPI · DLBCL · Counts, predicted risk and observed risk by age — `clinical_nccn_ipi_clinical_score_age`

**Supplement · Protocol hold.** Showing counts separately from predicted and observed outcomes is useful.

Keep the original heatmap grammar, but split a six-panel page if text becomes too small. Use the same probability scale for comparable panels, explicit uncertainty and an unambiguous unsupported-cell symbol. Avoid duplicate age views when the full clinical-score plate already answers the question.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### IPI · score-available population · Survival and reclassification — `clinical_ipi_computed_risk_groups`

**Supplement · Protocol hold.** The old paired survival curves, risk tables and horizon forest are worth retaining.

Keep the native index curves and original style. Any model curves or reclassification matrix requiring a full ordered partition must follow the revised protocol, not crossed anchors. Use separate low-risk/high-risk analyses if those are the chosen questions. Retain empty/unsupported groups explicitly rather than manufacturing categories.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### IPI · score-available population · Precision–recall, score distributions and age strata — `clinical_ipi_computed_clinical_score`

**Supplement · Protocol hold.** The old PR curve and within-index model-risk distributions communicate continuous information well.

Preserve this visual arrangement. Give the PR curve a separate operating-point table below it, rather than text on the data. Reserve a dedicated legend area and four-category label space. Keep achieved specificity and risk visible. The age cross-classification panels depend on the unresolved model-group protocol.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### IPI · score-available population · Counts, predicted risk and observed risk by age — `clinical_ipi_computed_clinical_score_age`

**Supplement · Protocol hold.** Showing counts separately from predicted and observed outcomes is useful.

Keep the original heatmap grammar, but split a six-panel page if text becomes too small. Use the same probability scale for comparable panels, explicit uncertainty and an unambiguous unsupported-cell symbol. Avoid duplicate age views when the full clinical-score plate already answers the question.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### CLL-IPI · Survival and reclassification — `clinical_cll_ipi_all_subtypes_risk_groups`

**Supplement · Protocol hold.** The old paired survival curves, risk tables and horizon forest are worth retaining.

Keep the native index curves and original style. Any model curves or reclassification matrix requiring a full ordered partition must follow the revised protocol, not crossed anchors. Use separate low-risk/high-risk analyses if those are the chosen questions. Retain empty/unsupported groups explicitly rather than manufacturing categories.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### CLL-IPI · Precision–recall, score distributions and age strata — `clinical_cll_ipi_all_subtypes_clinical_score`

**Supplement · Protocol hold.** The old PR curve and within-index model-risk distributions communicate continuous information well.

Preserve this visual arrangement. Give the PR curve a separate operating-point table below it, rather than text on the data. Reserve a dedicated legend area and four-category label space. Keep achieved specificity and risk visible. The age cross-classification panels depend on the unresolved model-group protocol.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### CLL-IPI · Counts, predicted risk and observed risk by age — `clinical_cll_ipi_all_subtypes_clinical_score_age`

**Supplement · Protocol hold.** Showing counts separately from predicted and observed outcomes is useful.

Keep the original heatmap grammar, but split a six-panel page if text becomes too small. Use the same probability scale for comparable panels, explicit uncertainty and an unambiguous unsupported-cell symbol. Avoid duplicate age views when the full clinical-score plate already answers the question.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### FLIPI2 · Survival and reclassification — `clinical_flipi2_all_subtypes_risk_groups`

**Supplement · Protocol hold.** The old paired survival curves, risk tables and horizon forest are worth retaining.

Keep the native index curves and original style. Any model curves or reclassification matrix requiring a full ordered partition must follow the revised protocol, not crossed anchors. Use separate low-risk/high-risk analyses if those are the chosen questions. Retain empty/unsupported groups explicitly rather than manufacturing categories.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### FLIPI2 · Precision–recall, score distributions and age strata — `clinical_flipi2_all_subtypes_clinical_score`

**Supplement · Protocol hold.** The old PR curve and within-index model-risk distributions communicate continuous information well.

Preserve this visual arrangement. Give the PR curve a separate operating-point table below it, rather than text on the data. Reserve a dedicated legend area and four-category label space. Keep achieved specificity and risk visible. The age cross-classification panels depend on the unresolved model-group protocol.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### FLIPI2 · Counts, predicted risk and observed risk by age — `clinical_flipi2_all_subtypes_clinical_score_age`

**Supplement · Protocol hold.** Showing counts separately from predicted and observed outcomes is useful.

Keep the original heatmap grammar, but split a six-panel page if text becomes too small. Use the same probability scale for comparable panels, explicit uncertainty and an unambiguous unsupported-cell symbol. Avoid duplicate age views when the full clinical-score plate already answers the question.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### MIPI · Survival and reclassification — `clinical_mipi_all_subtypes_risk_groups`

**Supplement · Protocol hold.** The old paired survival curves, risk tables and horizon forest are worth retaining.

Keep the native index curves and original style. Any model curves or reclassification matrix requiring a full ordered partition must follow the revised protocol, not crossed anchors. Use separate low-risk/high-risk analyses if those are the chosen questions. Retain empty/unsupported groups explicitly rather than manufacturing categories.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### MIPI · Precision–recall, score distributions and age strata — `clinical_mipi_all_subtypes_clinical_score`

**Supplement · Protocol hold.** The old PR curve and within-index model-risk distributions communicate continuous information well.

Preserve this visual arrangement. Give the PR curve a separate operating-point table below it, rather than text on the data. Reserve a dedicated legend area and four-category label space. Keep achieved specificity and risk visible. The age cross-classification panels depend on the unresolved model-group protocol.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### MIPI · Counts, predicted risk and observed risk by age — `clinical_mipi_all_subtypes_clinical_score_age`

**Supplement · Protocol hold.** Showing counts separately from predicted and observed outcomes is useful.

Keep the original heatmap grammar, but split a six-panel page if text becomes too small. Use the same probability scale for comparable panels, explicit uncertainty and an unambiguous unsupported-cell symbol. Avoid duplicate age views when the full clinical-score plate already answers the question.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### R-ISS · Survival and reclassification — `clinical_r_iss_all_subtypes_risk_groups`

**Supplement · Protocol hold.** The old paired survival curves, risk tables and horizon forest are worth retaining.

Keep the native index curves and original style. Any model curves or reclassification matrix requiring a full ordered partition must follow the revised protocol, not crossed anchors. Use separate low-risk/high-risk analyses if those are the chosen questions. Retain empty/unsupported groups explicitly rather than manufacturing categories.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### R-ISS · Precision–recall, score distributions and age strata — `clinical_r_iss_all_subtypes_clinical_score`

**Supplement · Protocol hold.** The old PR curve and within-index model-risk distributions communicate continuous information well.

Preserve this visual arrangement. Give the PR curve a separate operating-point table below it, rather than text on the data. Reserve a dedicated legend area and four-category label space. Keep achieved specificity and risk visible. The age cross-classification panels depend on the unresolved model-group protocol.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### R-ISS · Counts, predicted risk and observed risk by age — `clinical_r_iss_all_subtypes_clinical_score_age`

**Supplement · Protocol hold.** Showing counts separately from predicted and observed outcomes is useful.

Keep the original heatmap grammar, but split a six-panel page if text becomes too small. Use the same probability scale for comparable panels, explicit uncertainty and an unambiguous unsupported-cell symbol. Avoid duplicate age views when the full clinical-score plate already answers the question.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### IPS · Survival and reclassification — `clinical_ips_all_subtypes_risk_groups`

**Supplement · Protocol hold.** The old paired survival curves, risk tables and horizon forest are worth retaining.

Keep the native index curves and original style. Any model curves or reclassification matrix requiring a full ordered partition must follow the revised protocol, not crossed anchors. Use separate low-risk/high-risk analyses if those are the chosen questions. Retain empty/unsupported groups explicitly rather than manufacturing categories.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### IPS · Precision–recall, score distributions and age strata — `clinical_ips_all_subtypes_clinical_score`

**Supplement · Protocol hold.** The old PR curve and within-index model-risk distributions communicate continuous information well.

Preserve this visual arrangement. Give the PR curve a separate operating-point table below it, rather than text on the data. Reserve a dedicated legend area and four-category label space. Keep achieved specificity and risk visible. The age cross-classification panels depend on the unresolved model-group protocol.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### IPS · Counts, predicted risk and observed risk by age — `clinical_ips_all_subtypes_clinical_score_age`

**Supplement · Protocol hold.** Showing counts separately from predicted and observed outcomes is useful.

Keep the original heatmap grammar, but split a six-panel page if text becomes too small. Use the same probability scale for comparable panels, explicit uncertainty and an unambiguous unsupported-cell symbol. Avoid duplicate age views when the full clinical-score plate already answers the question.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### IPSSWM · Survival and reclassification — `clinical_ipsswm_all_subtypes_risk_groups`

**Supplement · Protocol hold.** The old paired survival curves, risk tables and horizon forest are worth retaining.

Keep the native index curves and original style. Any model curves or reclassification matrix requiring a full ordered partition must follow the revised protocol, not crossed anchors. Use separate low-risk/high-risk analyses if those are the chosen questions. Retain empty/unsupported groups explicitly rather than manufacturing categories.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### IPSSWM · Precision–recall, score distributions and age strata — `clinical_ipsswm_all_subtypes_clinical_score`

**Supplement · Protocol hold.** The old PR curve and within-index model-risk distributions communicate continuous information well.

Preserve this visual arrangement. Give the PR curve a separate operating-point table below it, rather than text on the data. Reserve a dedicated legend area and four-category label space. Keep achieved specificity and risk visible. The age cross-classification panels depend on the unresolved model-group protocol.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### IPSSWM · Counts, predicted risk and observed risk by age — `clinical_ipsswm_all_subtypes_clinical_score_age`

**Supplement · Protocol hold.** Showing counts separately from predicted and observed outcomes is useful.

Keep the original heatmap grammar, but split a six-panel page if text becomes too small. Use the same probability scale for comparable panels, explicit uncertainty and an unambiguous unsupported-cell symbol. Avoid duplicate age views when the full clinical-score plate already answers the question.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### MALT-IPI · Survival and reclassification — `clinical_malt_ipi_all_subtypes_risk_groups`

**Supplement · Protocol hold.** The old paired survival curves, risk tables and horizon forest are worth retaining.

Keep the native index curves and original style. Any model curves or reclassification matrix requiring a full ordered partition must follow the revised protocol, not crossed anchors. Use separate low-risk/high-risk analyses if those are the chosen questions. Retain empty/unsupported groups explicitly rather than manufacturing categories.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### MALT-IPI · Precision–recall, score distributions and age strata — `clinical_malt_ipi_all_subtypes_clinical_score`

**Supplement · Protocol hold.** The old PR curve and within-index model-risk distributions communicate continuous information well.

Preserve this visual arrangement. Give the PR curve a separate operating-point table below it, rather than text on the data. Reserve a dedicated legend area and four-category label space. Keep achieved specificity and risk visible. The age cross-classification panels depend on the unresolved model-group protocol.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### MALT-IPI · Counts, predicted risk and observed risk by age — `clinical_malt_ipi_all_subtypes_clinical_score_age`

**Supplement · Protocol hold.** Showing counts separately from predicted and observed outcomes is useful.

Keep the original heatmap grammar, but split a six-panel page if text becomes too small. Use the same probability scale for comparable panels, explicit uncertainty and an unambiguous unsupported-cell symbol. Avoid duplicate age views when the full clinical-score plate already answers the question.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### R-IPI · Survival and reclassification — `clinical_r_ipi_all_subtypes_risk_groups`

**Supplement · Protocol hold.** The old paired survival curves, risk tables and horizon forest are worth retaining.

Keep the native index curves and original style. Any model curves or reclassification matrix requiring a full ordered partition must follow the revised protocol, not crossed anchors. Use separate low-risk/high-risk analyses if those are the chosen questions. Retain empty/unsupported groups explicitly rather than manufacturing categories.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### R-IPI · Precision–recall, score distributions and age strata — `clinical_r_ipi_all_subtypes_clinical_score`

**Supplement · Protocol hold.** The old PR curve and within-index model-risk distributions communicate continuous information well.

Preserve this visual arrangement. Give the PR curve a separate operating-point table below it, rather than text on the data. Reserve a dedicated legend area and four-category label space. Keep achieved specificity and risk visible. The age cross-classification panels depend on the unresolved model-group protocol.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### R-IPI · Counts, predicted risk and observed risk by age — `clinical_r_ipi_all_subtypes_clinical_score_age`

**Supplement · Protocol hold.** Showing counts separately from predicted and observed outcomes is useful.

Keep the original heatmap grammar, but split a six-panel page if text becomes too small. Use the same probability scale for comparable panels, explicit uncertainty and an unambiguous unsupported-cell symbol. Avoid duplicate age views when the full clinical-score plate already answers the question.

Pending: Threshold-development decision, fresh clinical report and native-category audit.

Sign-off: No crossed/forced model groups, native names/category counts preserved, clinical indication checked, all achieved risks and specificities visible, and same score-available population used.

### Clinical-score overview · page 1 — `supp_score_overview`

**Supplement · Protocol hold.** The original overview provides a repeatable series across indices.

Preserve the series but paginate to keep native labels and forests readable. Drop generic all-subtypes text. Include every configured index, with a diagnostic panel when model groups are unavailable. Do not show only the five successful partitions.

Pending: Complete score roster and coherent clinical comparison protocol.

Sign-off: No cut-off forest labels, no omitted failing index, and legends/colours consistent with the detailed plates.

### Clinical-score overview · page 2 — `supp_score_overview_2`

**Supplement · Protocol hold.** The original overview provides a repeatable series across indices.

Preserve the series but paginate to keep native labels and forests readable. Drop generic all-subtypes text. Include every configured index, with a diagnostic panel when model groups are unavailable. Do not show only the five successful partitions.

Pending: Complete score roster and coherent clinical comparison protocol.

Sign-off: No cut-off forest labels, no omitted failing index, and legends/colours consistent with the detailed plates.

### Clinical-score overview · page 3 — `supp_score_overview_3`

**Supplement · Protocol hold.** The original overview provides a repeatable series across indices.

Preserve the series but paginate to keep native labels and forests readable. Drop generic all-subtypes text. Include every configured index, with a diagnostic panel when model groups are unavailable. Do not show only the five successful partitions.

Pending: Complete score roster and coherent clinical comparison protocol.

Sign-off: No cut-off forest labels, no omitted failing index, and legends/colours consistent with the detailed plates.

### What explains the advantage over joint? — `router_mechanism`

**Supplement · main-panel source · Await results.** A companion can close a real evidence or reporting gap.

The three-question layout links benefit over joint, correction reliance and gate variation. Keep it in the established indigo/teal/ochre style. It is a synthetic design preview until the independent audit arrives. Never add intervention AUC differences or call ablation a causal division of training benefit.

Pending: Independent mechanism audit.

Sign-off: Actual inputs, units, uncertainty and scope recorded. A proposed plot is not promised as an output of the running job.

### Mechanism audit: probability accuracy — `router_mechanism_probability`

**Supplement · Await results.** A companion can close a real evidence or reporting gap.

Retain the matching IPA companion to avoid explaining only ranking. Use identical rows and intervention labels to the AUC plate, with metric-specific axes and the same contrast direction. Current local picture is synthetic.

Pending: Independent mechanism audit.

Sign-off: Actual inputs, units, uncertainty and scope recorded. A proposed plot is not promised as an output of the running job.

### All frozen-fit mechanism interventions — `router_mechanism_sensitivity`

**Supplement · Await results.** A companion can close a real evidence or reporting gap.

Keep the detailed AUC/Brier/IPA matrix of interval rows as a technical companion. Explicitly label zero/mean removal and fixed-gate variants. The current local version is synthetic and does not establish that selective gates matter.

Pending: Independent mechanism audit.

Sign-off: Actual inputs, units, uncertainty and scope recorded. A proposed plot is not promised as an output of the running job.

### Every clinical index: separate low/high comparisons — `proposed_clinical_operating_points`

**Main · 6 panel / supplement detail · Proposed · not implemented.** A companion can close a real evidence or reporting gap.

A compact paired forest across all ten indices should report extra low-risk patients with both achieved risks, plus events captured with both specificities. Keep native populations and denominators. This is a new graphical export from existing anchor tables, not something the current runner already produces. Paired intervals still require implementation.

Pending: Explicit implementation/analysis decision.

Sign-off: Actual inputs, units, uncertainty and scope recorded. A proposed plot is not promised as an output of the running job.

### Whole-cohort outcome performance — `proposed_whole_cohort`

**Supplement · Proposed · tables only.** A companion can close a real evidence or reporting gap.

The new evaluator currently produces aggregate CSVs, not a figure. A two-column forest of outcome-level AUC/IPA and paired differences on the shared model population would complement disease-specific cells. Whole-cohort discrimination includes between-disease differences, so it cannot replace within-disease results.

Pending: Explicit implementation/analysis decision.

Sign-off: Actual inputs, units, uncertainty and scope recorded. A proposed plot is not promised as an output of the running job.

### Sensitivity to evaluation support requirements — `proposed_support_sensitivity`

**Supplement · Proposed · tables only.** A companion can close a real evidence or reporting gap.

Show estimates and retained counts for declared case/control support rules, with changing rosters explicit. Existing local support sensitivities are descriptive. A stricter subset is not a new training-size experiment, and a narrow interval cannot be reconstructed from marginal interval endpoints.

Pending: Explicit implementation/analysis decision.

Sign-off: Actual inputs, units, uncertainty and scope recorded. A proposed plot is not promised as an output of the running job.

### clinical sparse tied groups — `clinical_sparse_tied_groups`

**Archive · layout test · Design only.** Useful for long labels, many diseases, ties and full-range calibration checks.

Keep outside the manuscript and scientific results. A passing synthetic stress check is not an empirical audit.

Pending: Actual result figures still require their own inspection.

Sign-off: Clearly watermarked, never selected or cited as a finding.

### clinical sparse tied score — `clinical_sparse_tied_score`

**Archive · layout test · Design only.** Useful for long labels, many diseases, ties and full-range calibration checks.

Keep outside the manuscript and scientific results. A passing synthetic stress check is not an empirical audit.

Pending: Actual result figures still require their own inspection.

Sign-off: Clearly watermarked, never selected or cited as a finding.

### stress calibration extremes — `stress_calibration_extremes`

**Archive · layout test · Design only.** Useful for long labels, many diseases, ties and full-range calibration checks.

Keep outside the manuscript and scientific results. A passing synthetic stress check is not an empirical audit.

Pending: Actual result figures still require their own inspection.

Sign-off: Clearly watermarked, never selected or cited as a finding.

### stress disease facets 10 diseases — `stress_disease_facets_10_diseases`

**Archive · layout test · Design only.** Useful for long labels, many diseases, ties and full-range calibration checks.

Keep outside the manuscript and scientific results. A passing synthetic stress check is not an empirical audit.

Pending: Actual result figures still require their own inspection.

Sign-off: Clearly watermarked, never selected or cited as a finding.

### stress disease facets 24 diseases — `stress_disease_facets_24_diseases`

**Archive · layout test · Design only.** Useful for long labels, many diseases, ties and full-range calibration checks.

Keep outside the manuscript and scientific results. A passing synthetic stress check is not an empirical audit.

Pending: Actual result figures still require their own inspection.

Sign-off: Clearly watermarked, never selected or cited as a finding.

### stress headline all variants — `stress_headline_all_variants`

**Archive · layout test · Design only.** Useful for long labels, many diseases, ties and full-range calibration checks.

Keep outside the manuscript and scientific results. A passing synthetic stress check is not an empirical audit.

Pending: Actual result figures still require their own inspection.

Sign-off: Clearly watermarked, never selected or cited as a finding.

### stress headline fallback font — `stress_headline_fallback_font`

**Archive · layout test · Design only.** Useful for long labels, many diseases, ties and full-range calibration checks.

Keep outside the manuscript and scientific results. A passing synthetic stress check is not an empirical audit.

Pending: Actual result figures still require their own inspection.

Sign-off: Clearly watermarked, never selected or cited as a finding.

### stress patient map 10 diseases — `stress_patient_map_10_diseases`

**Archive · layout test · Design only.** Useful for long labels, many diseases, ties and full-range calibration checks.

Keep outside the manuscript and scientific results. A passing synthetic stress check is not an empirical audit.

Pending: Actual result figures still require their own inspection.

Sign-off: Clearly watermarked, never selected or cited as a finding.

### stress patient map 24 diseases — `stress_patient_map_24_diseases`

**Archive · layout test · Design only.** Useful for long labels, many diseases, ties and full-range calibration checks.

Keep outside the manuscript and scientific results. A passing synthetic stress check is not an empirical audit.

Pending: Actual result figures still require their own inspection.

Sign-off: Clearly watermarked, never selected or cited as a finding.

## Audit limits

This pass visually inspected the 85 pre-LDH render/stress images in contact sheets and enlarged core/dense panels, alongside the available original exports and the recipe/source files. Contact sheets are coverage checks, not pixel-level approval of every label in every current empirical image. Native clinical-group counts, real rerun spacing and all final placed-size figures must be checked again on the returned server bundle.

No held-out data, new national predictions, patient arrays, fitting or new confidence intervals were used. Historical scientific problems are described rather than silently fixed by redrawing. Source-schematic renders reproduce the existing TeX, including issues called out above.
