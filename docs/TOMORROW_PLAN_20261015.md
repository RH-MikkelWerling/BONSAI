# Working plan — Thursday 15 October 2026

> **Superseded for launch steps by `RUNBOOK_20261015.md`** (single seed 42, local `best.ckpt` from the 20-epoch run, 20-epoch DAPT, new metadata/plot-table steps).


## What is ready and what is still missing

| Area | Current coverage | Tomorrow's action |
| --- | --- | --- |
| Local pretraining | Confirmed 256/4 layers/4 heads/scaled Time2Vec; completion/reconstruction gate implemented | Check actual 20-epoch completion, sidecar, selected checkpoint, vocabulary and cutoff. Validation progress alone does not satisfy completion. |
| S/D/O/J supervision | Immutable current-release configs and repeated-seed roster supported | Prepare once with final server paths, 69-outcome release, 24 fine diseases, reviewed batches and seeds. |
| Router controls | Source/rank/query screens and matched final source banks implemented | Start source screen; select branches on tuning only, then record decisions and final lock. |
| Shared GPU queue | Exclusive ownership, heartbeat, verified completion, resume and common fixed/shared claim mechanism tested | Use one worker per independently assigned GPU and one shared run root; run the GPU smoke first. |
| Nation-scale/DAPT | Remapping, rotary-layout reconstruction and origin controls supported | Obtain/verify PHAIR cutoff/corpus/layout/vocabulary; direct export and EA must share the mapped starting artifact. |
| Tabular controls | Real logistic/XGBoost queue and tuning tests pass | Generate/audit the real feature source, training-only preprocessing and matched information; then run baselines. The feature launcher consumes a pre-exported matrix, not raw clinical history. |
| Predictions and paired analysis | Completion/selection, inventory, native outputs, shared bootstrap and seed handling supported | Freeze test-analysis decisions before opening held-out results; export tuning first for router/score locks. |
| Bayesian summaries | Support spline primary, raw effects, algebra/covariance, prior/diagnostic gates tested | Lock metric-scale priors, tolerances, panel weighting, support and sensitivity choices; do not use test results to choose them. |
| Clinical-score risk plots | Score dates/mappings, matched populations, groups, age counts/predictions/observations supported | Inventory all actual score/version/population fields; resolve cutoffs and applicable endpoints. |
| Intended-endpoint score benchmark | Analysis contract and manuscript reporting present; dedicated evaluator unfinished | Implement score/tabular/sequence matched evaluation and justified probability mappings. |
| Patient embedding comparison | Five-model matrix renderer and provenance contract present | Implement orchestration from five checkpoints to audited index-time vectors and train-fitted projections. |
| Real result figures | Twenty registered plates, workspace initializer, schema/render validation and hashes present | Implement remaining raw-analysis-to-plot adapters; recipes require real tables/JSON, not prototypes. |
| Supplementary numbers | Thirteen header schemas, narrative methods and reviewed-summary renderer present | Export real complete-cell tables and readable summaries; include reviewed LaTeX in manuscript. |
| Cohort/Table 1 | Draft accounting and release/review tooling present | Reconcile fresh export and obtain supervisor approval; do not substitute historical Table 1. |
| Manuscript | Clinical/ML motivation, CORE-BEHRT attribution, main/supplement structure and synthetic proof built | First prose edit now; replace result placeholders only with locked empirical estimates. |

## First: establish a valid server launch

Preserve the server's existing `.env` when syncing the changed and new files.
The local `.env` still contains unresolved server paths. Use the runbook to set
release/sequence/membership/outcome paths, base/router reference, local checkpoint,
paper output root and the fitted seed. Every model is trained once with seed 42
(decision 14 October 2026); intervals describe patient sampling only.
No checkpoint or data directory should be overwritten during code transfer.

In an allocated GPU terminal, using the activated server Python environment:

```bash
scripts/check_paper_server.sh --gpu
scripts/run_paper_experiments.sh prepare --execute
scripts/run_paper_experiments.sh queue-status
```

Preparation can occur while pretraining finishes. If the immutable bundle already
exists, review it rather than overwrite it; changed scientific settings require
a fresh bundle/output identity. The worker now fails explicitly if no bundle
exists. The import/path check is not a real optimizer/GPU check.

After the local checkpoint and completion metadata are available:

```bash
scripts/run_paper_experiments.sh smoke --execute
```

Inspect finite training/validation, selected checkpoint/reload and memory in the
isolated smoke output. Then, in each separately allocated GPU terminal:

```bash
scripts/run_paper_experiments.sh worker --wait --execute
```

All workers use the same shared bundles, run root and decision file. Each uses
one assigned visible GPU. Monitor from any terminal with `queue-status`.
Stale claims are reserved, not stolen. Confirm that the entire scheduler
allocation and its children stopped before archiving a stale claim. This is
necessary to prevent another GPU from overwriting a still-live fit.

Do not expect thousands of specialist fits to finish overnight. Measure early
fit throughput, failure rate and GPU memory, then estimate the remaining budget.
Success for the first session is a verified current-256 fit, a healthy shared
queue and source-screen progress, not a promised completion time for the panel.

## In parallel: unblock the practical comparisons

Prepare the tabular feature source and audit pre-index availability, windows,
training-fitted preprocessing and score-component availability. Check score
fields and their original publications; record intended endpoint, index and
horizon differences. This can progress if local pretraining is still running.
Produce the fresh Table 1 review export and review request; actual supervisor
sign-off is an external dependency, not an automated script result.

Collect source-screen tuning outputs when available. Record source mode, rank
and query decisions in sequence, then prepare the locked stage-2 matrix once.
Do not relabel preliminary screen fits as final models. PHAIR provenance and
DAPT preparation can proceed independently, but their downstream matrix is
conditional on audited encoder artifacts and the fixed downstream protocol.

## Then: complete the analysis handoff

Before held-out analysis, lock estimand/primary metric, horizon, common-cell
weighting, comparison families, practical tolerances, calibration and seed
summary. The example environment defaults to known-status analysis; a claim
about full-risk-set IPCW requires the separate `cif_ipcw` analysis tree. Neither
default substitutes for a scientific lock. Keep test results out of selection.

Implementation priorities are the intended-endpoint clinical-score evaluator,
raw-output plotting adapters and five-stage embedding extraction/projection.
These are the three substantive missing pieces before an unattended figures
and numbers pipeline. The existing table/plot renderers do not create their
scientific inputs. Initialise server-local workspaces with `init-figures`,
validate all populated plates with `figures --validate-render --execute`, and
render reviewed summaries with `supplement-tables --execute`.

## Audit evidence carried forward

See SERVER_AUDIT_20261014.md for the focused passing suites, concurrency fix,
local OpenMP/PyTensor environment limitations, GPU-only skips and real-server
requirements. No unrelated full repository suite was run. The modified R source
parses, both manuscript builds pass, and the supplements retain explicit pending
empirical content. No secure-server job or deployment has been performed here.
