# Current Table 1: supervisor review and manuscript handoff

Generate Table 1 from the frozen **supervised population**, one row per patient,
with its existing train/tuning/held_out membership. Do not substitute the SSL
population, model prediction IDs, or one endpoint's eligible risk set. The
current expected total is 37,531 (29,105 / 1,980 / 6,446), subject to sign-off;
37,557 describes the first-line source before supervised membership. Explain
and reconcile any discrepancy instead of forcing these counts.

On the secure server, set these paths to the actual frozen inputs:

```bash
python -m opera.run.export_table1 \
  --population "$SUPERVISED_POPULATION" \
  --metadata "$CURRENT_CLINICAL_METADATA" \
  --output-dir "$TABLE1_REVIEW_DIR" \
  --inspect-inputs

python -m opera.run.export_table1 \
  --population "$SUPERVISED_POPULATION" \
  --metadata "$CURRENT_CLINICAL_METADATA" \
  --output-dir "$TABLE1_REVIEW_DIR" \
  --disclosure-policy targeted
```

The population must contain the frozen split labels; otherwise supply the
existing `--split-manifest`. If needed, supply `--column-map` for actual source
column names. Optional `--overall-survival` adds verified death/follow-up
summaries; do not use `--restrict-to-overall-survival` unless that explicitly
constitutes the approved population definition. For corrected pre-index history
summaries, add `--meds-dir` and a verified `--meds-code-prefixes` JSON. The
exporter excludes demographic/background tokens from clinical history; do not
reuse the earlier history span that inadvertently equalled age.

For supervisor review, use `release_review/table1_pooled_display.csv` as the
compact Overall/train/tuning/held_out table and `release_review/table1_values_long.csv`
for grouped/fine disease, score and treatment detail. A formatted spreadsheet or
PDF is sufficient for the discussion. The exporter writes CSVs, not a finished
LaTeX table. Keep `LOCAL_ONLY_internal` and patient-level metadata on the server.
The existing institutional disclosure review applies before moving an aggregate
bundle outside the secure environment; supervisor scientific sign-off and
release approval are separate steps.

Ask the supervisor to confirm:

- Population, index definition, exclusions and temporal split denominators.
- Grouped/fine disease assignments and treatment category curation.
- Demographics, missingness, score-system denominators and risk-category layout.
- Corrected clinical-history definitions and which summaries belong in the main
  table versus detailed supplementary tables.

Retain the input/export provenance with the exact reviewed version and record
any requested changes. Render that version to
`paper/tables/table1_current_review.tex` for manuscript discussion; the manuscript
labels it review-pending. After approval, use
`paper/tables/table1_current_approved.tex`, which takes precedence. Approval is
not inferred from a filename: record reviewer/date and the reviewed export hash.
Keep the historical `paper/tables/table1.tex` as a layout reference, not as the
current cohort description. Fine-cohort/treatment supplement layouts should be
regenerated from this same export; the formerly documented standalone Table 1
supplement is absent from this checkout.
