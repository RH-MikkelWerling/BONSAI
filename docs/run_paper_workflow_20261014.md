# Run the paper workflow

Entry point: `bash scripts/run_paper_experiments.sh ACTION [OPTIONS]`.
It parses `.env` with Python rather than shell-sourcing it. All mutating/training
stages preview commands unless `--execute` is supplied. `check` and `status` are
read-only. Activate the intended server Python environment first; optionally
set `OPERA_PYTHON` to its interpreter. These scripts run inside your existing
GPU/CPU allocation; they do not submit jobs or choose GPU IDs for you.

The manuscript was force-built with pdfLaTeX/BibTeX via latexmk. The preview is
`paper/render/study_direction_20261014/opera_manuscript.pdf`. Seven pages; no
undefined references/citations or overfull boxes. Empirical results remain
placeholders and current Table 1 still requires approval.

## Configure environment once

There was no `.env` in the local checkout; a mode-0600 template is now present
locally and ignored by Git. It deliberately leaves scientific server inputs
blank. The tracked `.env.paper.example` is the portable overlay. Do **not** copy
this local file over the secure server's existing `.env`.

Copy `.env.paper.example` to a private server settings file, fill paths, preview
then apply the selective update:

```bash
python -m opera.run.configure_paper_env --env-file "$PWD/.env" --settings-file /secure/paper.settings
python -m opera.run.configure_paper_env --env-file "$PWD/.env" --settings-file /secure/paper.settings --apply
```

The updater preserves other values/comments, rejects duplicate target keys and
backs up an existing file. It never prints unrelated values. Empty overlay
placeholders do not erase existing settings; explicit `--set KEY=` can unset a
paper key. Use `--env-file` on the launcher if this workflow lives in a separate
private env file. Exported scheduler `CUDA_VISIBLE_DEVICES` takes precedence.

Required before preparing neural configs:

| Key | Meaning |
|---|---|
| OPERA_RELEASE_DIR | Approved 69-outcome release containing experiment_registry_primary_v2.yaml |
| BONSAI_OUTCOMES_DIR | Exactly OPERA_RELEASE_DIR/outcomes |
| BONSAI_COHORT_MEMBERSHIP | Sealed metadata parquet with split and fine disease identity |
| OPERA_SEQUENCE_DATA_DIR | Exact joined-bin sequences/vocabulary used by the running encoder |
| OPERA_PAPER_LOCAL_CKPT | Actual future/current local256 best.ckpt, never an old64 checkpoint |
| OPERA_PAPER_BASE_CONFIG | Approved full-panel **shared linear**, endpoint-group-macro horizon-BCE config |
| OPERA_PAPER_ROUTER_REFERENCE | Explicit chosen anchored/borrowed adapter policy and epsilon |
| OPERA_PAPER_ROOT | New persistent paper output root, separate from earlier runs |

The base can be the approved `time2vec_grouped_shared_bce.yaml` candidate.
`time2vec_plain_shared_bce.yaml` uses ungrouped loss and is not this base.
The generator enforces a trainable encoder and no pooled supervised warm start.
It resolves and freezes environment-dependent input paths into each generated
config. Changing `.env` later cannot silently redirect an existing bundle.
A changed scientific/data setting requires a fresh bundle.

```bash
bash scripts/run_paper_experiments.sh check
bash scripts/run_paper_experiments.sh prepare --execute
```

Preparation can happen before pretraining finishes. `check` reports a pending
checkpoint; execution waits for completed-epoch metadata and strict encoder
loading. `check` is a path/roster check, not clinical approval or a GPU smoke test.

## When local pretraining finishes

```bash
bash scripts/run_paper_experiments.sh smoke --execute
bash scripts/run_paper_experiments.sh stage1 --shard-count 4 --shard-index 0
```

Smoke trains one epoch/two batches in a separate **non-paper** directory,
validates on complete tuning data, verifies an optimizer step and reloads the
checkpoint on CPU. It does not replace memory/throughput measurements at
representative long sequences or the resume check in the experiment gates.

The second command previews the full initial queue: J, D, O, S and the four
source-screen candidates. Add `--execute` on workers 0/1/2/3, using the same
shard count. Alternatively launch scopes/phases separately:

```bash
bash scripts/run_paper_experiments.sh train --regime J --execute
bash scripts/run_paper_experiments.sh train --regime D --execute
bash scripts/run_paper_experiments.sh train --regime O --execute
bash scripts/run_paper_experiments.sh train --regime S --shard-count 4 --shard-index 0 --execute
bash scripts/run_paper_experiments.sh train --phase source_screen --execute
```

Do not launch `stage1` and separate overlapping queues concurrently. Scientific
run identities deduplicate repeated candidates across phases; directory locks
reject concurrent ownership and training fingerprints govern completion/resume.
A full unexcluded supervision panel is up to 1,750 fits per seed: plan GPU time.
`--max-runs` limits a queue for initial checks; it is not a scientific subset lock.

## Shared GPU worker (recommended on qsub)

Instead of sharding queues by hand, start the same worker in every GPU job:

```bash
# inside each qsub job / terminal, after activating the environment
bash scripts/paper_gpu_worker.sh            # run until nothing is runnable
WAIT=1 bash scripts/paper_gpu_worker.sh     # keep polling for unblocked work
bash scripts/run_paper_experiments.sh queue-status   # overview from any shell
```

Each worker claims the highest-priority fit nobody else is running, trains it
and continues. Priority: J, D, O, source screen, rank screen, query screen,
final source controls, external arms, then the S specialists. Claims are
exclusive-create files with a one-minute heartbeat, so workers on different
nodes coordinate through the same claim files. After 30 minutes without a
heartbeat a claim is reported stale but remains reserved. Confirm the scheduler
allocation and its children have stopped, then archive that run's claim file.
The restarted fit resumes from its own `last.ckpt`; never clear a claim merely
because its heartbeat expired. Completed fits are recognised by the
verified checkpoint-selection record in `training_complete.json` and are never
retrained. A failing fit is retried once more, then left with
`worker_failed.json` for inspection; a worker stops after three consecutive
failures (usually an environment or GPU problem).

The worker re-reads `OPERA_PAPER_DECISIONS` and looks for the stage-2 bundle on
every pass. Writing the source decision therefore unblocks the rank screen
without restarting workers. Workers started before pretraining finishes wait
(with `WAIT=1`) rather than fail: the encoder audit gates every fit.

## Bounded tuning decisions, then stage 2

```bash
bash scripts/run_paper_experiments.sh tuning-summary --phase source_screen --execute
```

Inspect `tuning_summary_stage1_source_screen/matched_readout_scores.csv`.
Only full-panel/readout rows have matched target sets and comparable selection
scores. Scores from different S/D/O target sets must not be ranked against each
other. No held-out files are read by this command.

Set `OPERA_PAPER_DECISIONS` to a JSON containing the tuning-chosen `source_mode`.
These are syntax examples, not scientific selections:

```json
{"source_mode":"disease_only"}
```

```bash
bash scripts/run_paper_experiments.sh train --phase rank_screen --execute
bash scripts/run_paper_experiments.sh tuning-summary --phase rank_screen --execute
```

Add the chosen `rank` (8 or 16) to the decisions JSON, then run `query_screen`
and its tuning summary. If shared/no-source is retained, skip rank/query search;
record that decision rather than selecting a more complex method by default.

Set `OPERA_PAPER_ROUTER_LOCK` to the final lock JSON containing `source_mode`,
`rank`, `query_mode`, `query_width` and the evidence/tolerance used. With the same
base, seed roster and budgets, prepare the immutable final bundle:

```bash
bash scripts/run_paper_experiments.sh prepare --stage 2 --execute
bash scripts/run_paper_experiments.sh train --phase final_sources --execute
```

This replicates matched no-source/disease/outcome/both controls. Repeated
scientific configs reuse prior fits. Final R is assigned only from this lock;
unselected tuning candidates never acquire the final-model alias.

## External transfer and DAPT

Fill `OPERA_PAPER_EXTERNAL_SOURCE_CKPT` and `OPERA_PAPER_EXTERNAL_SOURCE_VOCAB`.

RoPE layout: current upstream FGA-DIKU BONSAI checkpoints (flash or SDPA) use
interleaved rotary embeddings and are read with upstream conventions
automatically (interleaved RoPE, Time2Vec over kilo-hours = local
`scaled_time2vec`). Only an upstream **flash** checkpoint trained before
2026-09-29 (commit a3fa6ac) needs
`OPERA_PAPER_EXTERNAL_SOURCE_ROPE_LAYOUT=non_interleaved`. Ask the collaborators
which commit trained the checkpoint. The export records the layout explicitly.
Upstream numbers the background segment 1; this pipeline renumbers it to 0 --
verify this convention with the collaborators before the E/EA fits.

```bash
bash scripts/run_paper_experiments.sh export-external --execute
bash scripts/run_paper_experiments.sh dapt --execute
```

E is a vocabulary-remapped encoder-only export without local SSL. EA initializes
from that **same export**, reuses the exact selected local sequence directory,
imports the source architecture, and starts a new optimizer. Default DAPT is
20 epochs, physical batch 1 × accumulation 128; review effective batch and GPU
memory for the actual external model. The DAPT run writes
`external_dapt/version_0/best.ckpt`. OPERA checks the source-export hash and
completion sidecar. An externally approved different width/budget must be
explicit in the paper env settings.

For the six-arm L/E/EA × J/R matrix, prepare stage 2 **with** external paths:

```bash
bash scripts/run_paper_experiments.sh prepare --stage 2 --include-external --execute
bash scripts/run_paper_experiments.sh train --phase final_sources --execute
bash scripts/run_paper_experiments.sh train --phase external --execute
```

Stage-2 output is immutable. If it was already prepared without external arms,
use a fresh bundle directory with the underlying `prepare_paper_run_bundle`
command; do not edit its files. Prefer preparing stage 2 with the future external
paths from the outset when the external ladder is planned. Existing local fits
are retained under the same neural run root.

## Tabular features and baselines

`OPERA_PAPER_FEATURE_DIR` is a folder with one raw matrix per cohort, named by `cohort_fine` (e.g. `DLBCL.pkl`), plus `all.<ext>` for the pooled XGBoost. Features are built per cohort, so their
columns differ; each specialist fit reads its own cohort's matrix and the pooled
XGBoost reads `all`. Every matrix must be at the first-line index, with
train-only feature fitting and no post-index predictors.

```bash
bash scripts/run_paper_experiments.sh features --execute   # <R>/features/<name>__{sequence_matched,registry_augmented}.parquet
bash scripts/run_paper_experiments.sh tabular --shard-count 4 --shard-index 0 --execute
```

`tabular` stops before fitting if any cohort's prepared matrix is missing, and
lists the expected file names (a run without `--execute` lists them too).

The queue derives fine-disease/outcome tasks and structural exclusions from the
S configs, covers the requested six horizons and seed roster, and fits logistic
regression plus XGBoost with known-status CIF labels and tuning log loss. It
uses the ascertainment eligibility view and retains competing deaths as known
negatives. Historical tabular defaults remain unchanged. Single-class tuning
can select on log loss; unfit single-class training cells remain explicit failed
fits. Completion markers verify source/feature/code hashes and prediction files.
Failures are recorded per cell, independent cells continue, and the queue exits
nonzero if any fit failed. Inspect logs/coverage rather than treating failures
as negative scientific results. Install the `.[tabular]` extra for XGBoost.

## Evaluation and analytical handoff

```bash
bash scripts/run_paper_experiments.sh status --stage 2
bash scripts/run_paper_experiments.sh evaluate --stage 2 --split tuning --execute
bash scripts/run_paper_experiments.sh evaluate --stage 2 --split held_out --execute
```

Evaluate only after the relevant method and analysis decisions are locked.
Tuning scores and evaluations use only the checkpoint each completed fit
selected (its recorded score, epoch and checkpoint hash); logs from abandoned
attempts are ignored. Held-out evaluation is limited to locked paper models;
screening candidates can only be evaluated on tuning.
Default split is tuning. Evaluation requires completed training and its best
checkpoint; it records missing fits and checks output/input hashes before reuse.
All candidate fits can be evaluated for diagnostics, but only the locked final
aliases enter the paper inventory. `--export-routes` adds route exports; large
route files remain on the secure server. Use evaluation shards like training.

Set `OPERA_PAPER_PATIENT_METADATA` (fine identity and declared preindex covariates),
`OPERA_PAPER_CELL_METADATA` (train support, approved endpoint groups/families),
and `OPERA_PAPER_DATA_CONTRACT` (frozen release/split/eligibility identity).
The analysis template defaults to the tracked example; customize its bins and
covariates before held-out inspection. Patient column mappings from that recipe
also need to leave a canonical `cohort_fine` for pooled exports.

```bash
bash scripts/run_paper_experiments.sh inventory --stage 2 --execute
bash scripts/run_paper_experiments.sh analyse --execute
bash scripts/run_paper_experiments.sh prepare-bayes --execute
```

Inventory joins tabular probabilities to exact locked labels (no patient-set
intersection), maps native grouped predictions to fine diseases, assigns final
model aliases and records planned/missing coverage. Main analysis defaults to
365 days under `OPERA_PAPER_ROOT/analysis/main365_<estimand>`. Native CIFs are restricted to
requested horizons **before melting**, avoiding loading six copies needlessly.
For supplements set `OPERA_PAPER_ANALYSIS_HORIZONS` and a fresh
`OPERA_PAPER_ANALYSIS_ROOT`; preserve the main analysis outputs.

Run the paired analysis twice, once per estimand (each gets its own default
root, `main365_ipcw` or `main365_known_status`):

- `OPERA_PAPER_ANALYSIS_ESTIMAND=cif_ipcw` — the paper's primary
  population-risk numbers: full-risk-set IPCW AUC and Brier, identical to the
  native evaluator, with the same paired bootstrap and contrasts.
- `OPERA_PAPER_ANALYSIS_ESTIMAND=binary_known_status_unweighted` — the
  secondary known-status analysis that matches the training objective.

Re-run `inventory`, `analyse` and `prepare-bayes` for each. Bayesian recipes
take their estimand from the analysis they read, so one template serves both.

Bayesian preparation writes, per question family and metric, an exchangeable
primary model, a spline support curve and a binned support check, with
explicit cell rosters and exclusion reasons. Priors/tolerances remain null
unless you supply a locked template through `OPERA_PAPER_BAYES_TEMPLATE`.
No posterior fitting is done during recipe generation.

```bash
bash scripts/run_paper_experiments.sh bayes --prior-only --execute
bash scripts/run_paper_experiments.sh bayes --execute
```

The CLI places prior-only outputs in a sibling `__prior_check` directory, so
the subsequent posterior run uses the original recipe output directory.
`OPERA_PAPER_BAYES_RECIPES` can contain a JSON list of selected recipe paths;
otherwise the six generated primary recipes are used. Do not launch all
sensitivity branches blindly. Fit each horizon/metric separately; inference
conditions on fitted seeds. See `bayesian_question_analysis_20261014.md` for
interpretation and diagnostic gates.

## Restart and limitations

Reuse the same neural root/bundle to continue interrupted fits under the
existing fingerprint/last-checkpoint contract. Tabular/evaluation intents allow
retrying the same inputs; changed inputs require fresh roots. Preparation,
analysis inventories and result directories are immutable. Never overwrite an
old64 run or change the architecture of the running pretraining job.

Local tests cover script plans, env preservation, exact artifact provenance,
known-status labels, tuning-only selection and native config path freezing.
Actual approved server files, GPU memory, clinical signoff and empirical result
checks remain server work. The scripts do not fabricate those approvals or data.

## Real clinical-score figures for every applicable score

After verified tuning AND held-out prediction exports, prepare an inventory
including 730 days (the default main365 inventory alone is insufficient). Use a
fresh analysis root for the clinical-score horizon. Copy
`opera/configs/clinical_score_panels.example.yaml` to secure configuration and
register every actual score/population pair. Audit IPI, NCCN-IPI, age-adjusted
IPI, CLL-IPI, FLIPI/FLIPI2, MIPI and ISS/R-ISS fields when present; names alone
do not establish clinical applicability, score dates or interchangeable groups.
Each job requires exact source columns/category mappings, native outcome,
fixed model/seed, pre-index dates, current tuning-locked cutoffs and review
provenance. Other scores with 2–4 categories use their original category order;
more categories require a reviewed plotting extension, not silent merging.

Set `OPERA_PAPER_CLINICAL_SCORE_CONFIG` in the server environment, then:

```bash
scripts/run_paper_experiments.sh clinical-scores
scripts/run_paper_experiments.sh clinical-scores --execute
```

The action uses the verified analysis inventory rather than finding arbitrary
checkpoints. It produces per-score risk-group and clinical-score PDF/PNG panels,
availability/status CSV, group point estimates (not CI tables), and input hashes.
The operating threshold is selected on the tuning split only. Groups use locked
baseline risk cutoffs and remain fixed through follow-up. Real patient records
remain secure. Output directories must be fresh; blocked/missing or sparse scores
are recorded rather than filled with synthetic data. Inspect all statuses before
calling the whole report complete. Cutoff-lock verification is a clinical review
requirement, not something the plotting script can infer from a provenance string.

The general prototype renderer remains synthetic. This new real-data runner
covers clinical-score panels; conversion of all other prototype tables/panels
into a unified real-results renderer is separate outstanding work. Never copy
synthetic plots into `figures/results` as if the analysis generated them.

## Real general figure renderer

Set `OPERA_PAPER_FIGURE_CONFIG` to a server-local recipe based on
`opera/configs/paper_results_figures.example.yaml`. Each plate explicitly maps
plot-ready tables to named `paper_figures.figure_*` arguments. The renderer
reads real CSV/parquet and writes PDF/PNG plus input/output hashes:

```bash
scripts/run_paper_experiments.sh figures
scripts/run_paper_experiments.sh figures --execute
```

Use a fresh output directory. This action never generates synthetic inputs.
Plot-ready adapters between raw analysis tables and figure schemas remain to
be verified for each plate: matching filenames alone is not sufficient. The
existing clinical-scores action renders every reviewed score/population record;
register actual score/category/date columns and tuning locks on the server.
Compare original score categories against fixed neural risk bands on identical
score-available patients. Performance and calibration are outcome/horizon matched;
do not compare raw clinical point scores with probabilities as if calibrated.

The support spline is now the primary prepared Bayesian variant; exchangeable
and support-bin variants are sensitivities. Priors and practical tolerances
still require metric-scale locks. Existing prepared immutable recipes are not
silently modified; prepare a new recipe directory after this protocol change.

## Complete server figure workspace

Initialise a dedicated directory under the selected estimand's analysis root:

```bash
scripts/run_paper_experiments.sh init-figures --execute
```

This creates `figure_inputs/plot_tables`, `figure_inputs/embedding_exports`,
`figure_inputs/figures.yaml` and `required_inputs.json`. The recipe has absolute
server-local paths for all twenty registered plates; it generates no data.
With `OPERA_PAPER_FIGURE_CONFIG` empty, the figure action uses this new recipe.
Existing reviewed recipes are not overwritten. No server path is inferred from
a workstation path. `OPERA_PAPER_ANALYSIS_ROOT` selects the parent directory;
otherwise the workflow uses its existing estimand-specific default under
`OPERA_PAPER_ROOT/analysis`.

### Full order from locked fits to figures

```bash
# Before the analysis (train/tuning labels and pre-index records only)
scripts/run_paper_experiments.sh cell-metadata --execute     # -> OPERA_PAPER_CELL_METADATA
scripts/run_paper_experiments.sh patient-metadata --execute  # -> OPERA_PAPER_PATIENT_METADATA
# Predictions: paper models export embeddings, the locked routers export routes
scripts/run_paper_experiments.sh evaluate --split tuning --execute
scripts/run_paper_experiments.sh evaluate --split held_out --execute
# Analysis
scripts/run_paper_experiments.sh inventory --execute         # picks up the router's routes
scripts/run_paper_experiments.sh analyse --execute
scripts/run_paper_experiments.sh bayes --execute
# Figure inputs and figures
scripts/run_paper_experiments.sh init-figures --execute
scripts/run_paper_experiments.sh plot-tables --execute
scripts/run_paper_experiments.sh embedding-tables --execute  # OPERA_PAPER_EMBEDDING_CONFIG
scripts/run_paper_experiments.sh figures --validate-render --execute
scripts/run_paper_experiments.sh figures --execute
scripts/run_paper_experiments.sh clinical-scores --execute   # risk-group figures
```

Patient metadata combines the cohort membership with the registry's `index`
(which prognostic index scored the patient) and `IPI` (low/medium/high)
columns, either from the membership file or `OPERA_PAPER_CLINICAL_METADATA`.
It counts each patient's pre-index history from the same tokenised records the
models read, truncated strictly before the index date, and, with the national
vocabulary, the share of that history the national encoder knows. The
embedding config (`opera/configs/paper_embedding_tables.example.yaml`) names
the npz exports per model stage; projections are fitted on tuning patients and
applied unchanged to held-out patients. The clinical-score config
(`opera/configs/clinical_score_panels.example.yaml`) selects patients by their
`index` value and uses the category rank as the score; the score marked
`manuscript_figures: true` is also saved under the manuscript figure names.

### From analysis outputs to plot tables

Two steps fill most of `figure_inputs/plot_tables` from the locked analysis.
The cell metadata is needed before `inventory`/`analyse`, because the paired
analysis stratifies effects by it; build it once from training and tuning labels
only (never held-out):

```bash
scripts/run_paper_experiments.sh cell-metadata --execute   # -> $OPERA_PAPER_CELL_METADATA
# ... inventory, analyse, bayes ...
scripts/run_paper_experiments.sh plot-tables --execute
```

`plot-tables` translates contrast codes (P1 -> disease-pooled vs specialist,
T2 -> joint vs XGBoost, E3 -> adapted national vs local, ...), model codes and
registry names into the tables the figures read, and writes
`plot_tables_manifest.json` listing each table as written, missing its analysis
input, or pending an upstream builder. Set `OPERA_PAPER_ATLAS_METRICS` to the
selected joint run's held-out `direct_cr_pooled_metrics.csv` for the atlas.
Calibration examples come from `opera/configs/paper_figure_choices.yaml`; lock
them before rendering held-out results. Still pending upstream builders:
clinical-score patients, panel-level patient subgroups, embedding/calibration
maps.

Review clinical endpoint/cutoffs/operating threshold and mapping JSONs, then run:

```bash
scripts/run_paper_experiments.sh figures --validate-render --execute
scripts/run_paper_experiments.sh figures --execute
scripts/run_paper_experiments.sh clinical-scores --execute
```

The first command launches read-only render validation; it saves no figures.
The second writes PDF/PNG and input/output hashes to a fresh figure directory.
The third renders each registered clinical score, now including the separate
age plate: patient counts, mean predictions, and observed probability. The
same matched population supplies all three rows. Probability estimates are
suppressed below ten patients; counts remain visible. These point estimates
require separate uncertainty for claims about age differences.

CSV/parquet inputs use named table arguments; dictionary/array mappings such as
calibration examples, router group maps and calibration-map outcome arrays use
hashed JSON inputs. The full recipe lists every file. Calibration-map arrays
must be in the coordinate table's patient order. A score-derived operating
threshold must be supplied in the locked clinical-score configuration; the
illustrative full-plate recipe does not choose it for real data.

Still required upstream: adapters from raw analysis outputs to the named plotting
schemas; five-checkpoint embedding extraction/projection; and the dedicated
intended-endpoint score/tabular/sequence evaluator. Initialising paths and a
full recipe does not complete these scientific data-generation steps. The
renderer will fail on missing or incompatible inputs rather than use synthetic
or earlier-run values. Until these exports exist, the entire figure pipeline
is not yet an unattended end-to-end job.

## Supplementary numeric tables

Copy `opera/configs/supplement_tables.example.yaml` into the figure workspace,
then map its inputs to reviewed aggregate summaries (complete cell-level CSVs
remain separate). Render with:

```bash
python -m opera.run.render_supplement_tables \
  --config /absolute/server/figure_inputs/supplement_tables.yaml \
  --output-dir /absolute/server/reviewed_supplement_tables --execute
```

The output contains LaTeX tables, a combined `supplement_tables.tex` and source
hashes. Copy the reviewed combined file to
`paper/tables/generated/supplement_tables.tex` before building the manuscript.
The narrative supplement includes it automatically. Empty header templates
are rejected. Input captions, aggregation and selected columns must describe
what was actually estimated; this renderer performs no scientific aggregation.

New figure-workspace initialisations also register absolute paths for the
supplementary summary tables in `supplement_tables.yaml` and add them to the
required-input manifest. After reviewed summaries are present:

```bash
scripts/run_paper_experiments.sh supplement-tables
scripts/run_paper_experiments.sh supplement-tables --execute
```

The default output is `supplement_tables` under the selected analysis root;
`OPERA_PAPER_SUPPLEMENT_TABLE_CONFIG` can override the recipe. Existing
workspaces retain their reviewed recipes; copy the new template into an older
workspace explicitly rather than reinitialising it over existing files.

## Bounded, reusable integrated gradients (18 October update)

Set `OPERA_PAPER_ATTRIBUTION_CHECKPOINT` to the tuning-selected downstream
checkpoint, `OPERA_PAPER_ATTRIBUTION_CONFIG_NAME` to its exact configuration
name, and `OPERA_PAPER_ATTRIBUTION_CONFIG_DIR` to that configuration's directory.
Set `OPERA_PAPER_ATTRIBUTION_CONFIG` to a reviewed copy of
`opera/configs/attribution_analysis.example.yaml`. The defaults explain three
outcomes at two horizons on a planned 300-patient stratified sample. Check the
exact outcome names against the selected release. One GPU process owns this
analysis; it is separate from the shared training pool.

```bash
# Show the planned command without launching it.
scripts/run_paper_experiments.sh attributions
# Process five patients of the SAME planned sample, not a new five-person sample.
scripts/run_paper_experiments.sh attributions --attribution-max-new-patients 5 --execute
# Continue, reusing hash-verified patient explanations and the baseline.
scripts/run_paper_experiments.sh attributions --execute
# CPU-only summaries and plots; reuse attributions at both horizons.
scripts/run_paper_experiments.sh attribution-summary --attribution-horizon 365 --execute
scripts/run_paper_experiments.sh attribution-figures --attribution-horizon 365 --execute
scripts/run_paper_experiments.sh attribution-summary --attribution-horizon 730 --execute
scripts/run_paper_experiments.sh attribution-figures --attribution-horizon 730 --execute
```

The pilot is marked incomplete and population summaries reject it until the
planned sample finishes. Summary directories are horizon-specific. For a revised
CPU summary (changed clinical labels, code review or suppression), use the lower-
level `summarise_attributions --input-dir ... --config ... --horizon-days ...
--output-dir <fresh-directory>`; this does not repeat GPU attribution. Point
`OPERA_PAPER_ATTRIBUTION_SUMMARY_DIR` to that directory when using the standalone
`attribution_figures.example.yaml` renderer. Existing summary directories are
not silently appended to or overwritten.

Initially, unreviewed concept mappings are omitted with a warning. Review code
identities AND lab-bin meaning before enabling named expectations. Figure
inputs are already summary tables: no raw-to-plot adapter is needed for these
three plates. Raw token, patient, perturbation and history-reveal exports stay
inside the secure environment. Summary/figure manifests carry hashes. See
`EXPLAINABILITY_PLAN_20261018.md` for interpretation, compute scope and checks.

### Concept-level and individual IG figures (18 October update)

The existing `attribution-summary` action now exports the concept ranking,
aggregate signed distributions and concept-by-lookback map. The existing
`attribution-figures` action automatically renders the three default outcome
overviews in addition to the source/time/reliability plates. No new environment
variables or extraction jobs are required. Summary and figure destinations must
be fresh; do not overwrite an earlier reviewed output tree.

The individual patient chart is a separate internal-only command:
`python -m opera.run.render_individual_attribution --help`.
It reads the same cached extraction and needs an explicit subject, outcome,
horizon and fresh output directory. It never runs the encoder. The global
recipe exports aggregate tables only; individual plots and provenance remain
INTERNAL. See `EXPLAINABILITY_PLAN_20261018.md` for interpretation and examples.
