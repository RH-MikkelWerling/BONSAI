# Disease-group terminology audit — 4 November 2026

Use **disease group**, not **lineage**, for the paper taxonomy. Its categories combine clinical and diagnostic groupings and do not consistently identify cell lineage.

Checked all paper sources (including captions and tables), figure generators, design-search display labels, table/number display labels, and current gallery descriptions. Updated the main/supplementary text, model-details table, specialisation legend, task-organisation statistics, legacy borrowing labels, design-search rows and proof-gallery descriptions.

Compatibility names such as `disease_lineage`, the exported `lineage` column, route-level keys, `lineage_summary` and `borrowing_lineage.csv` remain unchanged. These are internal identifiers, not biological claims. Checkpoint-lineage documentation refers to model provenance and remains unchanged. Disease-group memberships and statistical calculations are unchanged.

The current specialisation and task-organisation previews were refreshed. Previously exported figures and tables must be regenerated from the updated code to inherit the wording; historical archives are not rewritten.

Validation: parsed modified Python display generators; checked paper sources for remaining lineage wording and checked whitespace with `git diff --check`.
