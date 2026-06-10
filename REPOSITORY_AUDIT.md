# BONSAI / OPERA Repository Audit

Date: 2026-06-10

## Purpose

BONSAI is the reusable EHR pipeline: MEDS-style ingestion, temporal
tokenization, ModernBERT pretraining, outcome construction, and patient-level
finetuning. OPERA adds hematology adaptation, survival-informed contrastive
learning, joint cross-disease models, clinical/tabular baselines, retrieval,
evaluation, aggregation, and paper figures.

The scientific claim is stronger than ordinary prediction benchmarking:
outcome-guided and cross-disease adaptation should produce reusable patient
representations, with the largest benefit in small cohort-outcome cells.

## Critical Findings Repaired

1. **BONSAI/OPERA API drift made advertised workflows non-executable.**
   Outcome helpers, immutable subject cloning, mixed-window truncation,
   checkpoint metadata, and several constructor signatures had diverged.
   These contracts are restored and covered by tests.
2. **Base BONSAI finetuning leaked post-index information.**
   `bonsai.run.train` and `bonsai.run.finetune` truncated at follow-up
   `censor_date`. They now truncate at prediction `index_date`.
3. **Checkpoint loading could hide encoder incompatibility.**
   BONSAI pretrained finetuning now loads encoder-only weights and rejects
   missing or unexpected backbone keys while allowing a new task head.
4. **OPERA Hydra training configs could not compose.**
   OPERA configs referenced BONSAI `core` and `hardware` groups without adding
   the BONSAI config root to Hydra's search path. All advertised Hydra configs
   now compose in tests.
5. **CI did not run the actual pytest suite.**
   `unittest discover` was replaced by a single Python 3.12 quality gate for
   compile, Ruff, pytest, and coverage. The synthetic lifecycle workflow was
   repaired and no longer depends on a tracked `.env`.
6. **Scientific multiplicity correction was wrong.**
   The Benjamini-Hochberg implementation discarded its right-to-left monotone
   adjustment. The corrected implementation has a regression test.
7. **Configured outcomes could disappear silently.**
   Event-grid construction no longer converts malformed configured outcome
   files into empty tensors through broad exception handling.
8. **Dependency metadata did not match imports.**
   Polars, TorchMetrics, and Matplotlib are declared. Heavy optional tools are
   separated into `tabular`, `visualization`, `survival`, `retrieval`, and
   `tabpfn` extras. Transformers is constrained below the incompatible 5.x
   ModernBERT schema.

## High-Priority Remaining Work

1. **Typed orchestration is incomplete.**
   `sweep.py` remains a large dictionary-driven runner. Introduce validated
   cohort, outcome, variant, cell, and artifact dataclasses or Pydantic models,
   reject unknown fields, and use one contract in readiness and execution.
2. **The paper manifests are descriptive, not executable.**
   Either make them the canonical DAG consumed by the runner or explicitly
   generate them from the executable sweep contract. Two sources of truth
   remain.
3. **Coverage is uneven.**
   Whole-repository coverage is about 35%. Core aggregation, retrieval,
   outcomes, and contrastive logic are substantially better covered, but
   orchestration, plotting, extraction, vocabulary expansion, and several
   Lightning modules remain weak.
4. **The synthetic fixture is not truly tiny.**
   It runs only ten train/validation batches, but data creation processes about
   10,000 subjects. Add a deterministic sub-second fixture for pull requests
   and keep the current lifecycle as a scheduled or explicit integration job.
5. **Scientific analysis locking remains procedural.**
   Primary endpoints, model contrasts, multiplicity families, seeds, minimum
   event thresholds, and exploratory/confirmatory status should be validated
   from one immutable analysis manifest.
6. **Cohort-flow artifacts are incomplete.**
   Generate per-cell inclusion/exclusion tables covering registry eligibility,
   prediction-index eligibility, follow-up, missing features, competing events,
   and final paired-comparison denominators.
7. **Large modules need decomposition.**
   `opera/run/sweep.py`, comparison/evaluation, retrieval, and visualization
   modules should be split along stable contracts. Refactor behind tests and
   preserve artifact schemas.
8. **Packaging of external config files needs a wheel-level decision.**
   Editable installs work with `BONSAI_CONFIG_PATH`. A built wheel still needs
   a documented external config directory or packaged default configs.

## Verification Baseline

The repository should maintain these gates:

```text
python -m compileall -q bonsai opera tests
python -m pytest tests -q
ruff check bonsai opera tests
ruff format --check bonsai opera tests
coverage run -m pytest tests -q
coverage report
```

The checked-in synthetic workflow must continue to create data and outcomes,
pretrain one CPU epoch, finetune from the checkpoint, and train from scratch.
