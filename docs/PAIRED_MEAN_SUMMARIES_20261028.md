# Two paired-mean summaries

The main analysis now writes both `effect_summaries.csv` (equal cell) and `effect_summaries_disease_balanced.csv` (equal disease, then outcome within disease, then horizon within outcome). Both use the same supported cells and existing paired draws. This is disease balancing, not endpoint-family balancing; grade variants still count as outcomes. No inverse-variance weights, changing bootstrap task inventories or refitting are introduced. Positive Brier gains mean reference minus candidate; AUC/IPA gains mean candidate minus reference.

Plot-table generation writes `panel_effects_disease_balanced.csv` and `paired_mean_comparison.csv`. The latter includes AUC, Brier and IPA, both weighting schemes, horizon-specific and pooled estimates, intervals and counts. The supplementary recipe prints it when present. Paper-number exports retain existing equal-cell keys and add `_disease_balanced` suffixes; this does not silently replace earlier results or change model selection.

## Existing run: no need to repeat paired scoring

Wait for the current analysis to complete and write its draw archive. After syncing, run:

```sh
.venv-v100/bin/python -m opera.run.resummarise_paired_analysis \
  --analysis-dir /absolute/path/to/completed/prediction_analysis \
  --output-dir /absolute/path/to/new/paired_mean_summaries
```

This reads `contrasts.csv`, `paired_bootstrap_index.csv`, `paired_bootstrap_draws_INTERNAL.npz` and the source analysis manifest. It performs CPU aggregation, not model inference or patient resampling, and leaves the source directory untouched. It writes both summary CSVs, the combined comparison CSV and a provenance manifest. Existing support eligibility is retained; this does not retroactively correct legacy analyses that used different support criteria.

Inspect those outputs before copying the two reviewed effect_summaries CSVs into the completed analysis's prediction_analysis directory for the next plot-table/number rebuild. Keep the summary_manifest.json beside the release as provenance. Large saved draw archives still need sufficient CPU memory to load. Do not export INTERNAL archives from the server.

## Writing dependency

Read and edit now: Introduction, clinical questions, study design, model/control identities, comparator definitions, Results ordering and Discussion outline. Methods can specify the two estimands, but verify implementation/settings against the eventual analysis manifest.

Wait for paired analysis to write final comparative Results, uncertainty statements, abstract Findings, table values, model-ranking claims and conclusions about meaningful differences. Assess coverage and point estimates before interpreting intervals, and check probability accuracy as well as AUC.

Optional galleries, exhaustive cell maps and exploratory mechanism panels need not block drafting the core paper. Borrowing statements specifically require fresh verified borrowing summaries. Clinical survival paragraphs require their score-specific population counts, follow-up and curves. External-pretraining statements require the separate transfer branch.
