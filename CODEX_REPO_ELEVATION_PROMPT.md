# Codex Prompt: Elevate BONSAI / OPERA End to End

You are working in the BONSAI / OPERA repository. Treat this as a senior
engineering, research-software, and reproducibility hardening task. Do not stop
at a review or proposal. Inspect the current worktree, implement the highest
leverage fixes, add tests, run verification, and leave the repository in a
coherent state.

## Product and scientific intent

BONSAI is an end-to-end EHR foundation-model pipeline:

- ingest MEDS-like longitudinal records;
- tokenize codes and temporal features;
- pretrain a ModernBERT-based EHR encoder;
- construct prospective or post-hoc outcomes;
- finetune patient-level prediction models.

OPERA extends BONSAI for hematology registry adaptation:

- hematology-only pretraining and domain-adaptive pretraining;
- vocabulary expansion for specialized registry tokens;
- survival-informed, multi-outcome contrastive adaptation;
- per-task, survival, hybrid, and joint multi-cohort finetuning;
- tabular and clinical-score baselines;
- prospective evaluation, calibration, IPCW/survival metrics, subgroup and
  significance analyses;
- rarity, transfer, retrieval, interpretability, aggregation, and paper plots.

The central research question is whether outcome-guided adaptation and joint
cross-disease learning produce reusable patient representations that outperform
general pretraining, DAPT, non-contrastive adaptation, per-cohort models, and
tabular baselines, especially for small or rare cohort-outcome cells.

## Working rules

1. Read `README.md`, `DATA_FORMAT.md`, `OPERA_REPOSITORY_GUIDE.md`,
   `OPERA_EXPERIMENTS.md`, `paper/opera_analysis_plan.tex`, configs, tests, and
   all touched modules before editing.
2. Inspect `git status` first. Preserve user changes. In particular, do not
   overwrite or discard the untracked `bonsai/functional/checkpointing.py`.
3. Inspect `git stash list` and `stash@{0}` read-only. It contains a WIP version
   of expanded outcome helpers. Recover useful behavior deliberately through
   reviewed edits; do not blindly apply the stash.
4. Keep changes scoped and incremental. Prefer existing patterns, but replace
   patterns that make CI falsely green or scientific behavior ambiguous.
5. Do not require private clinical data for CI. Build small deterministic
   synthetic fixtures that exercise real contracts.

## Current verified baseline

Preserve these repairs and advance beyond them:

- the existing pytest suite passes and includes Hydra composition checks;
- BONSAI and OPERA outcome, cloning, truncation, follow-up, and checkpoint
  contracts have been reconciled;
- base BONSAI and OPERA finetuning truncate at `index_date`, not
  follow-up `censor_date`;
- pretrained BONSAI finetuning rejects missing or unexpected encoder keys;
- OPERA Hydra configs include the BONSAI config search path;
- Ruff check and format cover `bonsai`, `opera`, and `tests`;
- CI uses pytest plus a measured coverage floor;
- the checked-in synthetic BONSAI lifecycle runs on CPU;
- `.env` and generated LaTeX files are no longer tracked;
- optional heavy research dependencies are separated into named extras;
- Benjamini-Hochberg correction and configured event-grid failures are tested;
- readiness and sweep execution share a strict typed configuration contract;
- variants can explicitly include or exclude outcomes, with truthful planned
  cell counts;
- outcome-level registry dates preserve explicit null overrides instead of
  inheriting an unrelated cohort-wide date;
- outcome eligibility sidecars are validated and can be aggregated into a
  strict split-level cohort-flow artifact.

Read `REPOSITORY_AUDIT.md` before editing. Prioritize its remaining
high-severity findings, especially typed runtime cell/artifact records,
manifest convergence, automatic eligibility derivation, final paired
denominators, and genuinely tiny end-to-end OPERA smoke coverage.

## Phase 1: Restore one executable system

The checked-in BONSAI and OPERA layers have drifted. Resolve all import,
signature, configuration, and checkpoint contract mismatches.

Known failures to verify and fix include:

- OPERA and tests expect BONSAI outcome APIs that are absent from the current
  `bonsai/functional/outcomes.py`: pandas-compatible binarization, competing
  events, per-split minimum follow-up, prospective split assignment and
  validation, split summaries, and `find`.
- OPERA datasets import `clone_subject`, but the current subject-data module
  does not define it.
- OPERA pretraining passes mixed-window truncation arguments that the current
  `PretrainDataModule`, datasets, and truncation helper do not consistently
  accept.
- OPERA passes `checkpoint_metadata` to BONSAI Lightning modules whose current
  constructors do not accept or attach it.
- `opera/run/finetune.py` uses `OmegaConf` without importing it.
- Verify every advertised CLI can import and resolve Hydra config before any
  expensive training begins.
- Audit model reconstruction and state-dict loading. Fail on meaningful missing
  or unexpected encoder keys instead of printing counts and continuing.
- Fix device-indexing, shape, test-step, empty-dataset, zero-worker, and
  one-class edge cases found during integration.

Create contract tests for every BONSAI API consumed by OPERA. Prefer one
canonical implementation over duplicate pandas/polars logic, with explicit
conversion boundaries and immutable inputs.

## Phase 2: Make tests and CI truthful

Replace the current mixed and partially ineffective CI setup with a clear
quality gate:

- run `pytest`, not `unittest discover`, for the pytest-style suite;
- lint and format `bonsai`, `opera`, and `tests`;
- test on the supported Python version, initially Python 3.12;
- add package import and CLI `--help`/Hydra composition smoke tests;
- add a tiny CPU end-to-end pipeline:
  data fixture -> outcome creation -> pretrain -> finetune -> evaluate ->
  result aggregation;
- add OPERA smoke coverage for DAPT, contrastive/joint model construction,
  checkpoint round-trip, and one synthetic sweep cell;
- add coverage reporting with a realistic threshold focused on core functional,
  dataset, checkpoint, and evaluation modules;
- pin GitHub Actions to stable release tags or commit SHAs, not moving branches.

The suite must fail when an advertised workflow is broken.

## Phase 3: Fix packaging and environment reproducibility

Audit imports against `pyproject.toml`. Declare required runtime dependencies
such as Polars, TorchMetrics, Matplotlib, and any other unconditional imports.
Move heavy or optional features such as FAISS, UMAP, lifelines, TabPFN, and
special plotting extras into named optional dependency groups with clear error
messages.

Then:

- choose and document a tested Python/dependency matrix;
- avoid implausibly narrow future-version pins unless required;
- add a lock or reproducible environment strategy suitable for CPU CI and GPU
  research servers;
- correct project URLs, package metadata, version exposure, and install docs;
- make `.env` a local file, provide `.env.example`, and ensure no secrets or
  machine-specific paths are tracked;
- remove generated LaTeX build artifacts from version control and ignore them.

## Phase 4: Consolidate configuration and orchestration

There are overlapping orchestration paths (`sweep.py`, manifests, `run_all.py`,
standalone runners) and very large modules. Establish one typed experiment
contract for:

- cohort;
- outcome and prediction horizon;
- registry coverage;
- split policy;
- model/checkpoint lineage;
- training objective;
- evaluation subset;
- rarity metadata;
- artifact schema.

Use dataclasses, Pydantic, or structured Hydra configs with validation. Reject
unknown fields and incompatible combinations early.

Refactor oversized modules such as `opera/run/sweep.py`, evaluation comparison,
metrics, retrieval, and visualization into focused units without changing
behavior. Turn paper manifests into executable DAGs or explicitly reduce them
to validated documentation, but do not leave two competing sources of truth.

Add resumability, atomic artifact writes, deterministic cell IDs, environment
capture, structured logs, dry-run plans, and a machine-readable run manifest.
Missing configured outcomes or swallowed exceptions must be visible in status
artifacts; required paper cells should fail readiness rather than silently skip.

## Phase 5: Strengthen scientific validity

Treat leakage prevention and denominator consistency as first-class invariants.
Add tests and validations for:

- all model inputs ending at the prediction index date;
- no subject overlap across train, validation, and prospective test;
- explicit registry eligibility and follow-up eligibility;
- consistent subject IDs and evaluation subsets across paired comparisons;
- fixed test sets across seeds and label-efficiency fractions;
- training-only fitting of imputation, scaling, calibration, thresholds, and
  feature selection;
- competing-event handling and cause-specific interpretation;
- one-class, low-event, heavy-censoring, and tiny-cohort behavior;
- deterministic seeds across Python, NumPy, PyTorch, Lightning, workers, and
  samplers;
- confidence intervals and multiplicity correction that preserve pairing and
  stratification;
- calibration assessed without fitting and evaluating on the same held-out
  predictions unless explicitly labeled apparent calibration.

Separate exploratory analyses from confirmatory paper outputs. Encode primary
endpoints, model contrasts, seeds, event thresholds, and multiplicity families
in a locked analysis manifest. Produce a CONSORT-like cohort flow and exclusion
table for every paper cell.

## Phase 6: Improve architecture and maintainability

- Define stable public interfaces for data records, outcome records, checkpoint
  metadata, predictions, and result rows.
- Replace ad hoc dictionaries where schemas materially reduce ambiguity.
- Centralize constants for split names, event codes, special tokens, artifact
  names, and canonical training stages.
- Add logging instead of `print` in library code.
- Replace broad `except Exception: pass` blocks with narrow handling and
  actionable diagnostics.
- Add type checking for core modules.
- Add concise docstrings to public APIs and comments only for non-obvious
  scientific logic.
- Benchmark memory and runtime for long sequences, contrastive pair matrices,
  retrieval, bootstrap loops, and large result aggregation. Add chunking or
  vectorization where needed.
- Keep visualization modules downstream of validated tidy result tables rather
  than embedding analysis logic in plotting functions.

## Phase 7: Documentation and usability

Rewrite the docs so a new researcher can run a public synthetic example without
private data. Remove stale `corebehrt` and Azure references unless they are
actually supported. Keep commands, paths, dependency versions, and artifact
names synchronized with code.

Provide:

- a concise architecture diagram and lifecycle;
- a data contract with validation commands;
- a stage-by-stage quickstart;
- a reproducibility guide for offline/private-data servers;
- checkpoint compatibility and migration guidance;
- an experiment matrix mapping every scientific claim to code, config, input,
  output, and statistical test;
- a limitations section that accurately describes competing risks, external
  validation, calibration, and generalizability.

## Phase 8: Deliverables and acceptance criteria

Work in reviewable increments. At minimum deliver:

1. A written audit with findings ranked by severity.
2. Restored BONSAI/OPERA API compatibility.
3. Correct dependency metadata and environment setup.
4. Truthful CI covering both packages.
5. Deterministic synthetic end-to-end smoke workflows.
6. Validated typed experiment/result contracts.
7. Refactored orchestration with explicit failure/status reporting.
8. Updated documentation and repository hygiene.

Before finishing, run all feasible checks and report exact results:

```text
python -m compileall bonsai opera tests
python -m pytest
ruff format --check bonsai opera tests
ruff check bonsai opera tests
```

Also run the synthetic BONSAI pipeline and at least one synthetic OPERA
checkpoint/evaluation/aggregation path. If a check cannot run, state precisely
why and leave a reproducible command for it.

Do not claim completion while core CLIs fail to import, tests are skipped by
the runner, configs point only to placeholders, or documented artifacts cannot
be produced from a fresh install.
