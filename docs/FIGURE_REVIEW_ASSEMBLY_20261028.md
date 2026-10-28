# Assemble the completed server figures

Select the finished local_figures workspace, not plot_rebuild or its old figures.
The commands below never modify its old PDFs or rerun borrowing inference.

```bash
figure_root=data/paper_256_v3/analysis/local_figures_20261028_143127
.venv-v100/bin/python -m opera.run.assemble_figure_review --analysis-root "$figure_root"
```

The printed fresh review directory contains main (including current source
figures), attributions, recent_records, nested clinical_scores, reports and a
review_manifest.json. Read the three renderer manifests for missing inputs and
input/code hashes. partial_review is expected if any required analysis is absent.
The absence of a plot is not treated as successful completion. This is a tuning
review, not final test evidence. Existing clinical figures are copied with their
original hashes, so clinical_score inventory/provenance must be inspected for
training-derived cutoffs before using them in the final paper.

If integrated-gradient summaries are absent or belong to an old fit, run on a GPU
allocation, using a fresh directory:

```bash
.venv-v100/bin/python -m opera.run.prepare_attribution_refresh \
  --analysis-root "$figure_root" \
  --output-dir data/paper_256_v3/analysis/ig_refresh_20261028 \
  --execute
```

This verifies selected checkpoint/configuration hashes from completion.json,
prepares tuning-only one-year IG for the selected fit, runs IG and then the CPU
summary. It can be substantial GPU work. Its refresh_manifest.json records exact
commands. If interrupted, use those commands to resume rather than reinitialise
the directory. The attribution runner has patient-level resume support.
Clinical concept mappings in the example recipe still require review; these
plots are a research review, not certification of those mappings.

Then assemble using that explicit attribution directory:

```bash
.venv-v100/bin/python -m opera.run.assemble_figure_review \
  --analysis-root "$figure_root" \
  --attribution-dir data/paper_256_v3/analysis/ig_refresh_20261028/attributions
```

For missing or outdated clinical scores, use the clinical-only workflow in
COMPLETE_LOCAL_FIGURES_20261028.md. Pass --clinical-dir /path/to/clinical_refresh/clinical_scores when assembling
to use a clinical-only refresh alongside the completed borrowing workspace.
A missing training-prediction export cannot
be fixed by rendering. Similarly, missing recent-record restriction summaries
require their inference/analysis stage, not a plot replacement.

Only the review directory is assembled for transfer: it includes figures and a
small allowlist of aggregate reports, not raw predictions, patient tables,
embeddings or INTERNAL arrays. Inspect the review manifests and index-specific
support limitations before choosing which outputs enter the manuscript.
