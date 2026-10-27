# Repository and Claude handoff — 27 October 2026

This overview is based on the checkout, current source/configuration, manuscript,
recent commits, and the supplied Claude transcript. Server queues, fitted models,
and secure empirical outputs have not been inspected. Statements about completed
server runs in historical notes are not fresh verification.

## What the study is doing

OPERA tests whether related predictions in haematology benefit from sharing
supervised experience across diseases and outcomes. The approved paper scope is
24 fine diseases and 69 outcomes. Table 1 records 37,531 supervised patients:
29,105 training, 1,980 tuning, 6,446 test. Those counts are distinct from the
pretraining population and outcome-specific eligible risk sets.

The current temporal contract is training through 2021, tuning in 2022 and test
from 2023 (`opera/configs/manifests/temporal_split.yaml`). Older guides that say
tuning 2022–2023 / test 2024+ are superseded. Physical SSL train/validation files
are separate from these supervised splits. Supervised histories stop strictly
before each patient's first-line treatment index, with a prediction token at
that index. Follow-up/censor dates define labels, not available input history.

The central factorial is S (one disease/outcome), D (diseases pooled for one
outcome), O (outcomes pooled within one disease), J (both pooled). The contrasts
separate these axes and their interaction. R changes the readout under joint
supervision. XGBoost/logistic, pooled XGBoost and registry-enriched XGBoost test
practical alternatives. Local, national/PHAIR, and national plus local DAPT
encoders separate delivered external transfer from domain adaptation. Clinical
indices, calibration, coverage, label savings, source removal and input
restrictions qualify the model comparisons. Embedding maps are descriptive.

## Code map and execution

| Location | Responsibility |
| --- | --- |
| `bonsai/` | Generic MEDS processing, vocabulary/tokenization, encoder, pretraining, reusable censoring/collation/training |
| `opera/functional/`, `opera/modules/` | Outcome-aware datasets/datamodules, supervised objectives, competing-risk heads, source-anchored readouts, transfer and tabular features |
| `opera/evaluation/` | Label/follow-up contracts, patient-matched metrics/bootstrap, secondary analyses, subgroup and input restrictions |
| `opera/analysis/` | Bayesian contrast/rarity models and diagnostics |
| `opera/run/` | Runnable experiment preparation, queues, evaluation, inventories, analysis, plot tables and rendering |
| `opera/configs/` | Protocol templates, contrast manifests, taxonomy, display choices and recipes |
| `scripts/` | Server launchers, workers, rehearsal and diagnostic orchestration |
| `opera/visualization/` | Shared style and empirical paper figure functions |
| `paper/` | Article, shared body, supplements, bibliography, TikZ schematics and reporting contracts |
| `talk/` | Presentation generation and animations; separate from empirical paper analysis |
| `tests/` | Data/architecture/metric/provenance/workflow/rendering regression checks |
| `docs/` | Dated decisions, audits and runbooks, including superseded plans |

`paper_experiments.py` and `scripts/run_paper_experiments.sh` orchestrate the
workflow. Immutable run bundles fix training inputs; workers claim fits and
prediction jobs. `paper_analysis_inventory.py` adapts completed neural/tabular
artifacts into a source-traceable analysis recipe. `analyse_locked_predictions.py`
produces estimates, contrasts, shared bootstrap draws, calibration, strata,
strategy and patient-gain outputs. `build_paper_plot_tables.py` turns those into
display tables without fitting new models. `render_paper_results.py` validates
recipe signatures and hashes the real inputs. Clinical-score and attribution
workflows have their own secure inputs and renderers. `build_paper_numbers.py`
and the supplement-table renderer populate manuscript artifacts.

Production defaults in `.env.paper.example` use all six horizons and `cif_ipcw`;
the standalone analysis example still defaults to known-status / one year.
The generated recipe, not an old docstring or generic example, establishes the
actual run. Old joint-model docstrings (768-wide, Kendall-only BCE) likewise do
not describe the current 256-wide paper procedure.

## Current paper and decisions

`paper/opera_manuscript.tex` is the article entry point; `paper/opera_body.tex`
holds the shared prose (also usable by the thesis). Supplementary figures are
in `paper/opera_supplement_figures.tex`. The manuscript includes explicit
`\result` and `\placeholder` fields: successful layout builds do not mean
empirical results or author metadata are finalized. Synthetic and tuning-preview
artwork must keep its labels and cannot fill held-out claims.

Use the latest amendments in `docs/ANALYSIS_LOCK_DRAFT_20261015.md`, not merely
its early sections. It documents optimizer/routing issues and reruns, selection
rules and the 27 October amendment making R_AB the primary routed model.
The inventory aliases it as R_L through `--routed-model R_AB_L`; the originally
locked router remains a sensitivity. OPERA is a separate tuning-selected model
alias. Do not assume that OPERA and a specific routed fit are interchangeable.
The current borrowing-pair displays are descriptive after broad borrowing was
seen on tuning; the earlier formal pair-test plan is superseded.

Headline reporting pools horizons, with per-horizon displays alongside it.
Patient-subset AUC reanalysis uses equal disease–outcome–horizon cell summaries,
retaining cells finite in every shared bootstrap replicate. Patient-level Brier
subgroup summaries use patient weighting; these are different estimands.
The dated lock document contains differing historical weighting proposals, so
the final frozen analysis configuration and implementation must be reconciled
before reporting a weighting rule as final. The lock file still calls itself
DRAFT, and figure choices still say `candidate_pending_lock`.

Remaining submission work includes locked real results/numbers/figures/tables,
direction-dependent prose, discussion/abstract conclusions, author/funding and
encoder provenance details, refreshed history summaries, clinical mapping/index
review, final analysis-lock reconciliation and disclosure/author review. See
`paper/README.md`, `paper/SUPPLEMENT_CONTENTS_PLAN.md` and
`docs/SUBMISSION_READINESS_20261018.md`; their older factual claims must be read
against later amendments and actual secure outputs.

## Work completed from the supplied Claude handoff

- Retained and checked the 7/14/21/28-day inference gaps. They count back from
  the index prediction token; special/background tokens remain, ordinary events
  strictly inside the gap are removed, and the exact boundary is retained.
  This perturbs an already-truncated model input; it does not backfill older
  events or retrain the model.
- Retained resume behavior for input restrictions: existing prediction files
  are reused. Reuse assumes the same checkpoint, split and restriction recipe;
  use a fresh analysis root if any of those inputs change.
- Completed subgroup AUC using the existing paired analysis and shared patient
  resamples. The stratified loop now shares the Brier subgroup bin/missing-label
  helper, avoiding one reanalysis per raw age/history value.
- Added stratified history, age, sex and IPI to the analysis template alongside
  region/year. Plot tables write `patient_subgroups_auc.csv` from those analyses.
- Added the two-panel AUC/Brier `figure_who_benefits`, wired it into the full
  recipe and synthetic prototype, and updated the manuscript caption. A common
  row inventory leaves unestimable AUC entries blank.
- Checked top-left panel title alignment at save time, resolving canvas layout
  before measuring and preserving title height during repeated saves.

## What should run on the server

No new training is required for these additions. They need four restricted GPU
prediction passes and CPU reanalysis/rendering. Existing IG exports can be reused.
Do long work inside separately scheduled jobs with enough walltime; `nohup`
does not extend an allocation. Use the site's existing scheduler resource values.

1. Sync changed source/configs and verify `code_manifest.sha256`. Preserve server
   data, private env files, immutable training bundles and completed results.
2. Update the *active* attribution recipe (including rehearsal copies) with the
   four gap entries from the example. Run `input-restrictions` using that env;
   existing restrictions are skipped, then the paired comparison is regenerated.
   Confirm the checkpoint and split refer to the intended model/population.
3. Update the active analysis recipe's `stratified_reanalysis` with the added
   patient groups from the template. Preserve its artifact roster, aliases,
   estimand, horizons and all other locked settings. An already-generated
   `analysis_inventory/analysis.yaml` does not pick up template changes itself.
   Run the analysis into a fresh output directory, then rebuild plot tables
   against that output. Do the tuning rehearsal before the approved test run.
4. Update an existing figure recipe's `fig_who_benefits`: function
   `figure_who_benefits`; keep `tables.table`; add `tables.auc` pointing to
   `patient_subgroups_auc.csv`; keep the styles/order JSON inputs; remove its
   old `options.xlabel`. New workspaces inherit this from the full template.
5. Render into a fresh directory; inspect AUC/Brier and title placement, run
   `figure-readiness`, then regenerate numbers/supplement tables and build the
   manuscript once the approved real results are available.

For a newly prepared analysis workspace, the launcher sequence is `inventory`,
`analyse`, `plot-tables`, `init-figures`, `clinical-scores`, `figures` (each with
`--execute` and the intended `--env-file`). Existing steps have different reuse
rules: analysis and figure rendering require fresh output directories; do not
blindly rerun the entire sequence over completed outputs. `figure-readiness`
reports remaining missing inputs but is not a server training-progress monitor.
Use `queue-status` / `status --split tuning` and the actual scheduler to establish
what training or inference remains. No server completion claim is made here.

## Local verification

84 tests passed across input restrictions, secondary analysis, plot tables,
paper figures/recipes, attribution orchestration, rehearsal regressions and
paper workflow. One tabular integration test was excluded after confirming
that XGBoost cannot load because this Mac lacks `libomp.dylib`. The initial
run also exposed a stale test expecting 28 rather than the recipe's 31 figures;
the expectation and path checks now reflect the existing borrowing inputs.

The two-panel synthetic layout was rendered to PNG/PDF and visually inspected
at `/private/tmp/opera_handoff_review/who_benefits.png` (synthetic only).
`git diff --check` passes. Ruff has existing findings in the touched files;
comparison with HEAD found no newly introduced diagnostic types/messages.
All listed source checksums were refreshed and verified. No server jobs were
launched and no held-out patient data were accessed.
