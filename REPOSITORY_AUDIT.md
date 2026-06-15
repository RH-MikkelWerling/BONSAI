# BONSAI / OPERA Repository Audit

Date: 2026-06-11

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
9. **Sweep configuration could drift between readiness and execution.**
   A shared dataclass-backed contract now validates cohorts, outcomes, variants,
   seeds, rarity settings, unknown fields, and incompatible outcome/training
   combinations before either path runs. Variant-specific outcome filters are
   explicit and planned-cell counts now match executed work.
10. **Outcome coverage and exclusion denominators were not auditable.**
    Outcome-level registry-date overrides now preserve explicit null values.
    Optional eligibility sidecars have a validated patient-level schema, and a
    strict CLI produces tidy cohort-flow counts by split, criterion, final
    eligibility, and exclusion reason.
11. **Eligibility sidecars were audit-only rather than model inputs.**
    All OPERA training and evaluation paths now apply the patient-outcome
    eligibility table before binarization. Multi-outcome datasets retain
    partially observed patients and mask only unavailable outcome dimensions.
12. **Fixed-window censoring and competing-death semantics were inconsistent.**
    Event-free follow-up is capped at the task horizon, early administratively
    censored controls are excluded from BCE training, and death is encoded as
    competing `event=2` only inside the configured risk window.
13. **Standalone evaluation could not strictly reconstruct ordinary Lightning
    checkpoints.** Wrapper metric buffers are now excluded from model-state
    loading, saved joint-model initialization settings are restored, and
    evaluation derives sequence length from the checkpoint.
14. **The advertised full sweep still depended on placeholders and silent
    configured-cell skips.** Environment-rooted artifact paths replace
    `/ckpts` and `/results`; five cohorts and aligned outcome filenames are
    configured; strict multi-cohort runs reject missing outcome, eligibility,
    competing-event, or split inputs.
15. **CI stopped before OPERA evaluation.** The synthetic lifecycle now runs a
    strict OPERA checkpoint evaluation and result aggregation after BONSAI
    pretraining and finetuning.

## Repaired Since Initial Audit (2026-06-11 addition)

16. **Leukemia registry cohort architecture implemented.**
    The train-on-grouped / eval-on-fine paradigm is fully wired:
    - `opera/functional/cohort_groups.py` — canonical `FINE_TO_GROUPED` /
      `GROUPED_TO_FINE` mapping, validation helpers, `resolve_training_cohort`.
    - `opera/configs/leukemia_contrastive.yaml` — multi-cohort contrastive
      training over the 10 grouped disease cohorts.
    - `opera/configs/leukemia_sweep.yaml` — evaluation sweep over all 25 fine
      diagnoses, each pointing to its grouped training artifact via
      `training_cohort`, `cohort_fine_col`, and `cohort_fine_value`.
    - `CohortSpec` extended with `training_cohort`, `cohort_fine_col`,
      `cohort_fine_value`; `format_variant_path` now supports
      `{training_cohort}` placeholder; `check_readiness.py` propagates it.
    - `evaluate.py` filters test subjects to `cohort_fine_col == cohort_fine_value`
      when both are set, enabling per-fine-cohort metric reporting without
      model or data architecture changes.
    - 48 new tests covering mapping completeness, round-trips, CohortSpec
      validation, and config structure.

## Repaired Since Initial Audit (2026-06-11 deep-optimization pass)

17. **Typed sweep orchestration implemented.**
    - `opera/run/sweep_types.py` — `SweepCellRecord` (frozen dataclass, `from_kwargs`,
      `to_dict`), `StatusTracker` (append / n_failed / n_skipped / n_success / write),
      `VALID_STAGES` / `VALID_STATUSES` constants.
    - `opera/run/sweep_commands.py` — pure argv builders `build_finetune_cmd`,
      `build_evaluate_cmd`, `build_prediction_evaluate_cmd` (no I/O, fully unit-testable).
    - `sweep.py` refactored to import and use both; inline override construction
      replaced by command builders; `append_cell_status` replaced by `StatusTracker`.

18. **Aggregation hardened against silent NaN propagation.**
    - `compute_model_delta_table` drops NaN metric rows before deltas, logs warnings.
    - `summarize_*` functions call `.dropna()` before groupby to prevent NaN groups.
    - New `compute_paired_denominators()` function checks that `n_total` is
      consistent across variants for each (cohort, outcome, evaluation_subset)
      cell and returns a mismatch flag table.
    - New `validate_result_rows_denominators()` convenience wrapper returns
      human-readable warning strings.

19. **Scientific analysis manifest locked.**
    - `opera/analysis_manifest.yaml` — primary endpoints, exploratory outcomes,
      pre-specified model contrasts, multiplicity family, minimum event thresholds,
      random seeds, confirmatory/exploratory cohort labelling.
    - `validate_analysis_manifest()` added to `opera/config_contracts.py`.

20. **Pytest infrastructure improved.**
    - `[tool.pytest.ini_options]` added to `pyproject.toml` with `smoke`,
      `integration`, and `slow` markers registered.

21. **Coverage substantially expanded (+55 tests, 231 → 286).**
    - `tests/test_sweep_commands.py` — argv builders and SweepCellRecord/StatusTracker.
    - `tests/test_sweep_utilities.py` — flatten_metrics, build_results_table,
      to_latex_table, rank_normalize_scores.
    - `tests/test_opera_lightning_modules.py` — OperaContrastiveModule,
      JointFinetuneModule, MOLModule (forward pass, loss shape, multi-outcome masking).
    - `tests/test_aggregation.py` extended — paired denominators, NaN handling.
    - `tests/test_check_readiness.py` extended — leukemia sweep readiness CI test,
      analysis manifest validation test.

## Repaired Since Initial Audit (2026-06-12 five-task pass)

22. **run_sweep() decomposed into testable cell functions.**
    - `_run_ipi_baseline_cell()` (209 lines) handles the full IPI baseline
      path — dry-run, subset-prediction, low-coverage fallback, per-seed
      evaluation, and tracker updates.
    - `_run_variant_cell()` (532 lines) handles one (cohort, outcome, variant,
      seed) cell — all three paths (results_file, predictions_file,
      finetune+evaluate) with their sub-cases.
    - `run_sweep()` reduced from **857 → 163 lines**; it is now a thin
      orchestrator that calls the two cell functions.

23. **Analysis manifest wired into check_readiness.py.**
    - `check_manifest_consistency(manifest_path, sweep_config_path)` cross-checks
      primary endpoints, primary contrasts, and seeds between the manifest and
      the sweep config; returns human-readable issue strings.
    - `--manifest` flag added to `check_readiness.py main()`.
    - Four new tests cover: pass, missing endpoint, missing model, seed mismatch.

24. **Tiny deterministic fixture infrastructure.**
    - `tests/fixtures/tiny_data.py` — 40-subject deterministic generators
      (`make_subjects`, `make_vocabulary`, `make_outcome_frame`,
      `make_outcome_parquet`, `make_vocabulary_pt`, `make_subject_data_pt`).
    - `tests/conftest.py` — session-scoped `tiny_subjects`, `tiny_vocab`, and
      function-scoped `tiny_fixture_dir` pytest fixtures.
    - `tests/test_tiny_fixture_smoke.py` — 6 `@pytest.mark.smoke` tests
      verifying fixture shape, vocab, splits, file layout, and binarization.
    - `testpaths = ["tests"]` added to `[tool.pytest.ini_options]`.

25. **CONSORT-like aggregation table.**
    - `collect_label_split_summaries(sweep_result_dir)` recursively collects
      `label_split_summary.csv` files from a sweep output tree.
    - `build_consort_table(label_summaries)` produces a CONSORT-like row per
      (cohort, outcome, split) with n_subjects, n_labelled, n_positive,
      n_insufficient_followup, prevalence, n_model_variants.
    - `check_competing_event_denominator_consistency(result_rows)` flags cells
      where competing-event counts differ across variants.
    - 6 new tests covering empty input, nested CSV discovery, structure, and
      competing-event consistency checks.

26. **Coverage expanded to 46% (+47 tests, 286 → 333).**
    - `tests/test_stratified_sampling.py` — 12 tests covering all functions
      in `opera/functional/stratified_sampling.py`.
    - `tests/test_visualization_headless.py` — headless matplotlib smoke tests
      for classification_plots, survival_plots, embedding_plots.
    - `tests/test_tiny_fixture_smoke.py` — 6 smoke tests.
    - Additional aggregation and check_readiness tests.
    - `fail_under` raised: 34% → 40% → **44%**.

## Repaired Since Initial Audit (2026-06-12 gradient-conflict pass)

27. **Cross-outcome weighting is configurable and diagnostically testable.**
    - `opera/modules/networks/cross_outcome_weighters.py` adds uniform,
      legacy-exact Kendall, and an isolated FAMO task-weighting core.
    - Configured FAMO training fails closed until same-batch post-optimizer
      loss recomputation is implemented. The previous sequential-minibatch
      update was not the published FAMO lifecycle.
    - Contrastive configs explicitly select uniform weighting with macro
      aggregation. Historical
      behavior remains available with `weighter: kendall`,
      `aggregation: pooled`, and `class_balanced: false`.
    - Optional effective-number class balancing uses global outcome counts,
      a configurable cap, and no changes to patient-pair geometry.
    - Empty outcomes and outcomes with zero informative pairs contribute zero
      and cannot move weighting state.
    - Old checkpoints with `contrastive_loss.log_sigma` migrate strictly to
      the new Kendall state key.
    - `opera/diagnostics/representation_gradient_conflict.py` measures
      per-outcome representation-gradient cosines on jointly eligible patient
      rows and writes matrices, support counts, event rates, a heatmap, and
      ranked summaries for multiple checkpoints.
    - The survival CDF and `torch.searchsorted` pair-weighting code is
      unchanged.

28. **Gradient-conflict and rarity-invariance artifact harnesses are tested.**
    - The representation diagnostic now writes batch-level pair cosines and
      joint support in addition to its existing aggregate matrices.
    - `opera/diagnostics/conflict_verdict.py` bootstraps over batches,
      separates supported conflict from low-support noise, summarizes conflict
      components, and writes a one-sentence surgery verdict.
    - `opera/analysis/rarity_invariance_panel.py` reuses evaluated result rows,
      the existing baseline-delta function, and `rarity_plots.py`.
    - The rarity harness rejects task, denominator, baseline, patient-subset,
      and class-balance mismatches before calculating a trend.
    - Synthetic predictions validate the complete rarity panel and label all
      generated numbers as pipeline validation rather than scientific results.
    - Real three-weighter results remain blocked because configured FAMO
      training correctly fails closed.

## Remaining Work

1. **Leukemia outcome files not yet configured.**
   `leukemia_contrastive.yaml` and `leukemia_sweep.yaml` await real parquet
   paths and the full grade-2 outcome list. The checked-in leukemia contrastive
   config currently contains five example outcomes, so it does not yet confirm
   the expected approximately 43-outcome active panel. Once received, run
   `check_readiness.py --config
   opera/configs/leukemia_sweep.yaml --manifest opera/analysis_manifest.yaml
   --require_existing_paths`.
2. **Packaging of external config files needs a wheel-level decision.**
   Editable installs work with `BONSAI_CONFIG_PATH`. A built wheel needs either
   packaged default configs or a documented external config directory.
3. **A valid FAMO comparator is not yet trainable.**
   The analysis harness can consume evaluated FAMO result rows, but the
   training path must first implement the published same-batch post-step loss
   recomputation and verify it under gradient accumulation.

## Verification Baseline

The repository should maintain these gates:

```text
python -m compileall -q bonsai opera tests
python -m pytest tests/ -q
ruff check bonsai opera tests
ruff format --check bonsai opera tests
coverage run -m pytest tests/ -q
coverage report
```

As of 2026-06-12 (paper-artifact harness pass), the local baseline is **354
passing tests**, clean Ruff lint/format, and successful bytecode compilation.
`run_sweep()` is 163 lines (was 857). Three remaining items are blocked
(leukemia data, wheel packaging, and valid FAMO training support).
