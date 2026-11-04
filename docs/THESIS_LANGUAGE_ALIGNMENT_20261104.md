# Thesis-to-paper language alignment — 4 November 2026

Read the current thesis abstract and the kappa's synopsis, sharing framework, OPERA methods, discussion and conclusions. Updated `paper/opera_body_consolidated.tex` and the supplementary model explanation. Thesis files were not edited.

The synopsis makes the clinical question concrete: measurements and histories already exist; the task is to learn what they mean, including from patients usually studied separately, while preserving differences relevant to the prediction. Adopted that language without the thesis's house-price analogy or extended exposition.

Changes:

- Explain what recorded history can add to a clinical score and acknowledge why the scores remain useful.
- Use **sharing and specialisation** for the neural architecture. Distinguish this from statistical **partial pooling**, retained as a precedent rather than a claim about the fitted estimator.
- Remove the implied guarantee that specific corrections depend on estimation uncertainty. The selected design does not impose an uncertainty-based shrinkage rule.
- Explain that specialists already benefit from shared pretraining. Supervision-scope contrasts measure additional sharing during outcome-labelled training.
- Replace “borrowing uniformly” and “once everything is shared” with the actual joint-model comparison. Shared components need not impose identical predictions across tasks.
- Describe diffuse routing as broad sharing of corrections, without claiming statistical partial pooling or a causal explanation for performance.
- Distinguish learning from other tasks from selecting particular tasks. Similar readout performance does not locate the limiting mechanism.
- Describe national pretraining as a complete strategy changing population, vocabulary, laboratory representation and exposure together.
- Retain technical partial-pooling terminology for the actual hierarchical statistical analysis and the explicitly weighted R_PP variant.

No results, model choices, thresholds or evaluation definitions changed. Pending numerical placeholders remain pending. Rebuilt the consolidated review PDF.

Thesis points to reconcile later: its discussion describes size-matched clinical comparisons without the now-implemented descriptive evaluation-cohort KM matching; some modelling passages still use disease “lineage”. Biological lineage in the haematology background is a distinct, legitimate use.
