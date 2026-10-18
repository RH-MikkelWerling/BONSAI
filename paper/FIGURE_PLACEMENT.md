# Figure placement after reviewing the synthetic assembly

This is a scientific storyboard, not final journal numbering. Placement follows
the questions agreed before held-out evaluation; it is not chosen by which
synthetic panels look most favourable. All prototype results are invented.

| Display | Placement now | Reason / final assembly |
| --- | --- | --- |
| Modelling paradigms and original information axes | Main, retained | Explain the clinical and learning framework; preserve both original figures. Consider one combined conceptual plate later. |
| S/D/O/J schematic (formerly Figure 3) | Supplement | Makes the pooling counterfactuals explicit; can become a design panel in the pooling figure. |
| Cohort flow and current Table 1 | Flow supplementary; Table 1 main | Current assembly puts detailed exclusions in the supplement and introduces counts in main Results. Table 1 requires sign-off. |
| Predictability atlas | Main | Gives clinical breadth and absolute performance. Detailed cell/horizon coverage stays supplementary. |
| Pooling effects and interactions | Main | Answers the first headline question. Pair maps with uncertainty and support. |
| Routing effects (`supp_routing` asset) | Supplement, described in main | An architectural modifier of shared learning. Retain null results; promotion is not required for the central borrowing argument. |
| Joint versus tuning-selected cell strategy (`supp_strategy` asset) | Supplement, described in main | Tests the practical case for one shared model; tuning-only selection and common support remain essential. |
| Neural versus tabular | Main | Answers the second headline question. Include a compact calibration panel in the final plate. |
| Detailed calibration | Supplement, referenced in main text | Important validation, but redundant as a separate full main figure. The final tabular plate still needs a calibration summary; that composite is not yet rendered. |
| Nation-scale direct / domain-adapted comparisons | Main | Answers the third headline question. Preserve the direct and adaptation increments separately. |
| Nation-scale benefit strata (`supp_national_where`) | Supplement; candidate main panel | Coverage/support strata could join the nation-scale pretraining plate if clinically interpretable and prespecified. No automatic promotion based on significance. |
| Model versus NCCN-IPI risk groups | Main clinical example | Makes continuous prediction clinically legible; use matched score-available DLBCL patients and locked baseline groups. |
| Detailed clinical-score precision/recall, scatter and age panels | Move to supplement | Supports the main risk-group example without repeating the previous paper's full figure sequence. Other scores are separate disease-specific extensions. |
| Patient-subgroup effects | Move to supplement | Prespecified secondary heterogeneity. Promote a compact panel only if central to the final interpretation, with complete supplementary reporting. |
| Support, coverage, full cell maps and all horizons | Supplement | Essential scope and robustness evidence; main text reports their key denominators. |
| Gate use, routing rescue, embedding calibration maps | Supplement | Descriptive mechanism/secondary evidence; route weights and projections do not establish why performance changes. |

The working preview retains full plates to make editorial choices reviewable.
For publication, aim to combine conceptual plates and pooling/routing/strategy
panels rather than leave the enlarged main section as a long figure sequence.
Do not remove calibration, coverage or negative findings to meet a figure limit.
Detailed architecture catalogues are Methods/supplement material, not results.

The standalone definitions supplement is now appended to the compiled manuscript
PDF after supplementary figures. It remains separately compilable. Spacing uses
ragged-bottom pages and compact float separation rather than forced population
page breaks. Figure numbers follow placement and must be rechecked after assembly.

## Predictability plate: proposed next composition

Keep absolute predictability distinct from comparative model gains. Use the
same clinical ordering and style across these panels:

- A: compact 365-day outcome atlas, with unavailable entries grey and event
  support alongside; keep the complete 69-outcome by six-horizon matrix in
  the supplement.
- B: disease by endpoint-group absolute performance map, to reveal variation
  hidden by an across-disease summary. Median group AUC is descriptive and
  must not be labelled an AUC pooled across patient populations.
- C: prespecified representative horizon profiles or domain summaries, with
  support and uncertainty clearly distinguished from task spread.
- D: probability accuracy/calibration for those same examples, linking good
  discrimination to interpretable absolute risk.

This is an editorial composition proposal, not a newly generated figure.
Avoid duplicating the pooling gain maps or selecting example outcomes by their
held-out performance. Final selection depends on the target venue's display
budget; the complete original atlas remains available in the synthetic proof.

## Embedding maps: optional supplementary evidence

A patient representation plate can reuse one locked projection of index-time
embeddings, colouring the same test patients by disease, age, record length,
and selected predicted risks. Fit the projection on training representations
and transform test patients; distinguish an exploratory test-fitted view if
used instead. Keep outcomes out of projection fitting, retain censoring in
outcome overlays, and do not use test maps to select models or hyperparameters.
UMAP geometry is descriptive, not evidence that shared learning helps: that
comes from paired S/D/O/J contrasts. Record the seed, metric and parameters;
check that broad patterns survive a second projection or PCA.

Vocabulary maps are a separate, lower-priority diagnostic: colour token
embeddings by source/type and training frequency, exclude padding/special
symbols from clinical interpretation, and inspect neighbours in the original
embedding space. Context-free input embeddings do not explain contextual
patient risk or establish transfer. Compare aligned vocabulary coverage and
neighbour changes across encoder origins only if the mapping is audited.
Neither projection is required for the three primary study questions.

### Registered five-stage patient plate

`supp_embedding_comparison` is included directly in the manuscript and both
prototype and real figure renderers. Panel order: local pretrained; local
fine-tuned (tuning-selected); PHAIR pretrained; PHAIR locally domain-adapted;
PHAIR fine-tuned (tuning-selected, direct/adapted origin recorded). Use a single
cohort available under all five artifacts and a consistent disease colour key.
Choose one overall fine-tuned model per origin using the locked tuning criterion.
If a winning readout has only task-specific representations, name the encoder
representation used before inspecting test plots.

Real renderer input: `matched_patient_embeddings.csv`, with long columns
`model`, `patient_id`, `x`, `y`, `disease`. It rejects duplicate or unmatched
patients, inconsistent disease labels and nonfinite coordinates and records
input hashes. Upstream export/projection is still needed: export index-time
training/test vectors for each checkpoint, documenting layer, pooling, temporal
cutoff, artifact hash, preprocessing and UMAP settings. Fit each projection on
training vectors and transform the same test patients. Keep a projection
manifest alongside the coordinates. Prototype coordinates are invented layout
examples; they cannot replace these five real checkpoint exports.

The patient plate now uses five model columns and metadata rows: disease,
IPI risk group, age and sequence length. Each column reuses its coordinates
across rows; colour scales are shared across models. Real coordinate tables
also require `ipi_group`, `age`, `sequence_length`. IPI categories must come
from the applicable clinical population; unavailable/nonapplicable patients
stay grey. Sequence length must consistently mean the tokens actually entering
the encoder after truncation, with pre-truncation history length available as
an optional separate row. Additional categorical/continuous metadata rows can
be supplied through `color_rows` options without changing the projections.

## Full-pass layout revision, 14 October 2026

Calibration is promoted to the main tabular-benchmark sequence. Routing and
portfolio plates move to the supplement, with main Results subsections and
references retained. This puts the proof at nine main figures plus Table 1
while retaining every prototype and both original conceptual diagrams. The
three headline questions govern main space; readout/operational alternatives
remain visible secondary evidence. This revision supersedes the earlier
routing/strategy promotion entries in the table above.
