# Nation-scale pretraining in OPERA

Nation-scale reuse is a central contribution alongside sharing supervision across diseases and outcomes. Give it a dedicated abstract sentence, an Introduction rationale, a main Results section and the existing encoder-transfer figure. Lead with joint-model transfer; routing is a subsequent modifier. Preserve the current information-axes and attribution figures.

## Main evidence

The existing figure covers the required questions: direct national versus local transfer; the incremental effect of local adaptation; variation across diseases; and performance with fewer labelled training patients. No complete repeat of every specialist experiment is needed for this argument. Report calibration and uncertainty alongside discrimination on matched eligible cells.

Matching the full-data local model requires a prespecified practical margin, not a nonsignificant difference. Report the smallest observed training fraction meeting that criterion, with uncertainty, or state that this was not established. Reduced downstream label requirements do not imply reduced total compute. If helpful, add a compact population → PHAIR encoder → direct/local adaptation → haematology inset to this figure rather than another full figure.

## Verified facts needed

Obtain these from the national training artifact and its owners:

- Unique individuals actually trained on; observation years and cutoff; included registers; events/tokens processed.
- PHAIR provenance, reference, contributors, encoder/checkpoint version and appropriate credit. Distinguish producing the encoder from evaluating its reuse in OPERA.
- Architecture, objective, vocabulary/preprocessing version, steps, epochs, checkpoint selection, resources and wall time where available.
- Pretraining eligibility and overlap with OPERA training, tuning and test patients. Document whether held-out patients' pre-cutoff histories entered unsupervised pretraining and the temporal safeguards; excluding outcome labels alone does not establish patient-independent evaluation.
- Differences between national and local data deliveries and vocabulary coverage; alignment of cohorts, labels, prediction dates and downstream selection.

Put a few verified scale facts in main Methods; put full provenance, architecture, exposure, overlap and preprocessing specifications in the supplement. Do not substitute the Danish population count for actual pretraining participation.

## Interpretation

Direct benefit supports immediate reuse. Benefit after adaptation supports a general-to-specialist transfer strategy. Benefit with fewer labels strengthens relevance to small populations and new outcomes. Heterogeneous or null results delimit reuse and should remain visible.

The comparison concerns usable encoders as delivered. It does not isolate population scale from architecture, source data or training budget, and it does not establish international portability. Testing a national representation on clinically defined specialist endpoints against established scores and tuned tabular pipelines is itself a substantial translational contribution.

## Implementation audit: final model selection is not yet centralised

The encoder-origin analysis includes J_L, R_L, J_E, R_E, J_EA and R_EA. This supports the transfer contrasts, but is not an automatic final-model selector.

Concrete remaining coupling found in the current code:

- `opera/run/build_paper_plot_tables.py`: local J_L/R_L mappings, coverage selections, performance-table models, and default outcome-support/routing models.
- `opera/run/analyse_locked_predictions.py`: J_L/R_L comparisons with tuning-selected strategies.
- `opera/run/paper_tuning_summary.py`: collects verified checkpoint scores but explicitly does not infer a winner or decision lock; its note warns against ranking scores from different target scopes.
- `opera/configs/paper_embedding_tables.example.yaml`: local fine-tuned and prediction examples still point at J_L.

Before final clinical evaluation, introduce one explicit, auditable model-selection artifact: candidates, common tuning cells and patients, metric/aggregation, calibration requirements if used, missing-cell policy, tie-break, selected model/checkpoint and source hashes. Compare candidates using the same readout/task protocol; do not rank incompatible training losses or independently optimised monitor scores. Lock this before held-out results.

Pass that selected ID through clinical/tabular comparisons, calibration, decision points, risk groups, attribution export/summaries and patient representations. Keep fixed-arm pooling and national-transfer analyses separate and retain the local fine-tuned embedding column for the encoder comparison. Add relevant checks proving that switching the locked ID changes all final-model products without relabelling controlled contrasts. Until that propagation is implemented and tested, the workflow must not be described as automatically following a national winner.
