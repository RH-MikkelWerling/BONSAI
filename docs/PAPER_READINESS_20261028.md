# Reading and analysis handoff — 28 October 2026

## Read and edit one manuscript

Read `scratch/paper_reading_20261028/opera_manuscript_reading.pdf`: a line-numbered editorial copy including the supplement. It is built from the current source; no results were invented or imported from the server. Missing results remain placeholders. This is a text/claims review, not a verified result release.

Edit `paper/opera_body.tex` for the abstract, introduction, methods, results and discussion. Edit `paper/opera_supplement_figures.tex` for supplementary figure captions, and `paper/supplement_evaluation_details.tex` / `paper/supplement_model_details.tex` for detailed methods. `paper/opera_manuscript_reading.tex` is only the build wrapper; do not fork the prose into it or edit generated macros. The standard tuning wrapper remains `paper/opera_manuscript_tuning_preview.tex`.

Build from `paper/`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../scratch/paper_reading_20261028 opera_manuscript_reading.tex
```

Reading order: scientific argument and scope; interpretation of main contrasts; clinical comparison and endpoint choice; sensitivity claims; final numerical writing after verified figures arrive. Mark unresolved claims as HOLD with a reason rather than adjusting the story to unverified rankings. The draft contains X placeholders and prospective test-language; the banner alone does not convert these into observed tuning findings. The full preview still includes an older definitions appendix; its counts/definitions are not certified by compilation.

## Borrowing: what is appropriate to run and show

Keep the main figure centred on fitted-model cross-disease dependence: full predictions versus replacing other-disease correction terms by their tuning means, plus unshrunk source-alone effects for inspection and a clearly labelled shrunken descriptive map. The outcome bank and shared encoder remain active. This does not estimate all-bank borrowing, a causal biological effect, or the benefit of retraining with extra diseases.

Keep router versus joint and the trained own-source reference as separate contrasts. Verify the reference identity: the selected R_AB fit's own-source reference may restrict both banks, whereas the disease-source ablation retains the outcome bank. They answer different questions. Leave-one-in and leave-one-out do not bound a Shapley value with arbitrary interactions.

The candidate-pair file is explicitly DRAFT pending clinician review. Current Methods already describe the pairs as descriptive. Preserve that position; no confirmatory "prespecified success" claim or outcome-driven promotion of attractive pairs. A reviewed, versioned plan may be locked before held-out evaluation; do not retroactively call tuning-derived choices prespecified. The set permutation statistic is exploratory and does not remove dependence between pairs or source-size confounding.

Fresh CPU borrowing analyses now use paired frequentist multiplier bootstrap weights and supported cells (>=5 cases and >=5 controls), and require matching cache provenance. Provenance includes selected fit/run, ablation, code/config hashes, prediction file size/mtime signatures, membership, reference versions, bootstrap settings and support thresholds. Large-file signatures are explicitly not content hashes. Legacy per-source caches cannot be silently resumed; use a fresh output root. Source-removal metadata now describes mean versus zero ablation accurately. Missing draw archives cause recomputation rather than an incorrect cache hit.

Next server work: inspect the selected model's source-removal evaluation_complete.json and design.json against the intended checkpoint, split and ablation implementation. If matching, rerun only CPU `borrowing-map` in a fresh analysis root (the paper_experiments action uses that root's source_removal directory). If missing/mismatched, regenerate source-removal inference first. Re-render from the fresh verified aggregates; the current overnight runner only copies old borrowing aggregates. No need to interrupt the current main rebuild.

## Yesterday's additions: implemented versus running

| Addition | Implemented/wired | Current plotting rebuild |
|---|---|---|
| IPW sampling sensitivity | IPW_J and IPW_R contrasts, sampling roster and Table 2/number export | Included when newly evaluated artifacts are present; completion must be checked in returned inventory/results |
| Supported-cell performance | >=5 cases and controls, coverage retained | Recomputed |
| Frequentist multiplier intervals | Main analysis; now fresh borrowing CPU summaries too | Main recomputed; cached borrowing not refreshed |
| Age, sex, history amount, calendar/region and IPI subgroups | Patient groups plus subgroup AUC/Brier reanalysis in the current template | Conditional on active template and available metadata; missing covariates remain explicit |
| Later-diagnosis/identity-time sensitivity | identity_date_sensitivity in current template | Conditional on evidence-date metadata; inspect status tables |
| 7/14/21/28-day gaps | Restriction recipes and inference/comparison code | NOT launched; separate input-restrictions job, active recipe must contain four gaps and run.split=tuning |
| Gap manuscript numbers | Added restrict_gap_7d/14d/21d/28d macro keys | Populated only when restriction summary exists |
| Five outcome embedding maps and score-vs-OPERA | Server recipe generator plus table/render stages | Attempted after recipes generated; vector arrows need their separate exports |
| Borrowing/source-removal | Inference and CPU summary code | Old aggregate tables copied; fresh validation/recompute still required |
| Attribution/IG and restrictions | Dedicated jobs, CPU summaries and renderers | Cached summaries may render; no new IG or restriction inference launched |
| External encoder comparisons | Separate branch | Not created by plotting |

Two manuscript wiring fixes: supplement tables now use resultnumbersdir rather than always tables/generated; attribution-number lookup now uses attributions/summary_365d, matching the regular summary producer. When importing the reviewed aggregate release, place supplement_tables.tex beside results_macros.tex in tables/tuning_preview/.

Restriction/gap summaries currently pool patients over diseases per outcome; they are not the disease-balanced cell estimand used in the borrowing/main summaries. Label that distinction when interpreting them. Do not silently compare differently weighted estimates as though they were the same headline metric.

## What to return from the server

Return aggregate overnight_run.json, results_validation.json, config_audit.json, inventory/analysis status summaries and error tails. For borrowing, return source_removal.json and the evaluation/design provenance (aggregate metadata only). These establish which implemented analyses actually completed. Then import reviewed PDFs and aggregate number/table files, not patient-level data.
