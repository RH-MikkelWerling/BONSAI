# Pretraining finish → paper experiments

The executable end-to-end entry point is now
`bash scripts/run_paper_experiments.sh`; see
[the workflow runbook](run_paper_workflow_20261014.md) for environment setup,
GPU smoke testing, phased queues, tabular controls, prediction inventory and
Bayesian recipes. The lower-level commands below remain supported.

No secure-server jobs have been launched by this repository work. The tools
below materialize the approved server paths and release roster into immutable
configs and execute phased queues inside an existing GPU allocation. This makes
the experiment inventory ready without pretending the local checkout contains
the approved data or the running encoder checkpoint.

## Prepare while pretraining runs

Use the approved, fully specified shared linear horizon-BCE config, its matching
69-outcome registry and the explicitly chosen anchored/borrowed router reference.
Reference adapter scale/home floor/event shrinkage/shared-logit initialization
and optimizer epsilon are recorded; these are held fixed during source/rank/query
comparisons. Do not use the historical 87-outcome config as the release input.
All configs use natural sampling, no curriculum, a trainable encoder and complete
tuning-set endpoint-group-macro BCE checkpoint selection.

```bash
python -m opera.run.prepare_paper_run_bundle \
  --base-config /absolute/path/approved_shared_69.yaml \
  --release-registry /absolute/path/approved_registry_69.yaml \
  --router-reference-config /absolute/path/chosen_router_reference.yaml \
  --encoder-checkpoint /absolute/path/local256/version_0/best.ckpt \
  --seeds 42 --epochs 30 --batch-size 1 \
  --output-dir /absolute/path/paper_bundle_stage1
```

Checkpoint paths may refer to artifacts still being produced. Batch size 1 is
a conservative physical starting point, not a throughput recommendation; retain
and review the base logical batch/gradient-accumulation settings. Benchmark on
the actual GPU before launching the whole queue. The bundle records input hashes,
per-config hashes, phase conditions and science-based run identities. It refuses
to overwrite an existing directory. No data are read and no training is started.

The inventory includes every structurally available S cell, every D column, every
O row and J for each seed; the first seed additionally has bounded source, rank
and query candidates. Conditional branches do not imply running the whole sweep.
For a complete 24 × 69 panel, supervision is at most 1,750 fits per seed
(1,656 S + 69 D + 24 O + 1 J), before structural exclusions. Specialists dominate
cost: estimate their runtime and parallelize disjoint shards. Scientific labels
alone do not justify promising they will finish quickly.

## On completion: audit, smoke-test, then queue

Wait for the post-fit `checkpoint_metadata.json` beside the actual checkpoint;
`best.ckpt` alone can already exist while pretraining is running. The execution
queue checks the confirmed local architecture (256/4 layers/4 heads/scaled
Time2Vec), completion epoch count, vocabulary size, strict encoder reconstruction
and a finite CPU forward pass. Review resolved pretraining config and cutoff,
normalization, sequence length and token budget separately; tensor compatibility
does not establish the full scientific data contract.

```bash
python -m opera.run.run_paper_queue \
  --bundle-dir /absolute/path/paper_bundle_stage1 --phase supervision \
  --regime J --runs-dir /absolute/path/paper_runs --max-runs 1
```

This is a dry run. Add `--execute` only inside the intended GPU allocation. The
executor also runs direct-competing-risk path/config preflight, captures logs,
locks the run directory and requires `training_complete.json`. It does not
submit SLURM jobs. Existing trainer fingerprint/resume logic governs reuse.
Identical candidates occurring in different phases share a run directory.

Before the full queue, complete Gate 1 smoke checks in
`final_experiment_gates_20261014.md`: a real optimizer step, finite losses,
complete validation, checkpoint selection/save/reload, resume, and memory/throughput
at the actual sequence lengths. The CPU encoder audit is not this GPU smoke test.

For the shared GPU pool, use `worker --wait --execute` in each separate
allocation; it schedules supervision and source screening automatically.
The explicit shard commands below remain an alternative.

Then run `supervision` (optionally `--regime S/D/O/J`) and `source_screen`.
Across GPU allocations use identical `--shard-count N` and distinct
`--shard-index 0..N-1`; assignments within each phase are deterministic and
disjoint. Scheduler/resource settings belong to the server allocation.

After tuning-only source screening, record a JSON decision, for example
`{"source_mode":"disease_only"}`. Pass its path with `--decisions` for
`rank_screen`; record rank as well for `query_screen`. The examples are syntax,
not selected scientific winners. If no routing candidate adds enough value,
retain shared J and report the router controls as a null result.

After selection, record `source_mode`, `rank`, `query_mode`, `query_width` plus
selection evidence and tolerances in a lock JSON. Materialize a fresh stage-2
bundle with the same stage-1 arguments plus `--router-lock`, and the external
checkpoint arguments below. Run `final_sources` to replicate N/RD/RO/RDO at the
locked rank/query. Repeated configurations reuse the same science hash/run root.
Use identical epochs/seeds/physical-logical batch settings when rebuilding.

## External direct transfer and local DAPT

The direct external arm now has a separate encoder-only exporter:

```bash
python -m bonsai.run.export_external_encoder \
  --checkpoint /absolute/path/external/best.ckpt \
  --source-vocab /absolute/path/external/vocabulary.pt \
  --target-vocab /absolute/path/approved_local/vocabulary.pt \
  --seed 42 --minimum-token-coverage 0.9 \
  --output-dir /absolute/path/external_direct_export
```

It remaps token strings, checks the native architecture/coverage, excludes heads
and optimizer state, records random initialization of local-only rows, and does
strict transfer/finite-forward checks. This does **not** do local SSL training.
Review source corpus overlap, coding/time/value semantics and transfer coverage.

Initialize EA from this **same exported E artifact**, including its exported
vocabulary, to keep local-only token-row initialization fixed:

```bash
python -u -m bonsai.run.pretrain \
  --config-name pretrain_external_bonsai_dapt_v1 \
  paths.initialization_ckpt=/absolute/path/external_direct_export/encoder.ckpt \
  paths.initialization_vocab=/absolute/path/external_direct_export/vocabulary.pt \
  hydra.run.dir=/absolute/path/external_local_dapt
```

Review this config's local processed-data paths, numeric/token preprocessing,
objective, batch size and 20-epoch budget before launch. It imports source
architecture, creates a new optimizer and uses the approved pre-2022 local SSL
view. Preserve the direct-export source hash in the DAPT transfer report.

Rebuild the stage-2 bundle with
`--external-direct-checkpoint /absolute/path/external_direct_export/encoder.ckpt`
and `--external-dapt-checkpoint /absolute/path/external_local_dapt/version_0/best.ckpt`.
The `external` queue is E/EA × J/R, four fits per seed. Queue execution checks
transfer stage, local-vocabulary hash, exact direct-export initialization hash
and completed DAPT (default at least 20
epochs). A different approved budget must be explicit via
`--minimum-dapt-epochs`; an external width differing from 256 must be explicit
via `--external-hidden-size` for execution. Architecture/corpus differences
remain limitations of a pure pretraining-scale claim.

## Analysis handoff

Freeze model/checkpoint/release IDs, export canonical matched predictions and
cost metadata on the secure server, then use
`opera.run.analyse_locked_predictions` with the documented recipe in
`locked_prediction_analysis_20261014.md`. Lock support/tolerances/weights/primary
metric/multiplicity/subgroup bins before inspecting held-out comparisons.
The current no-new-training analysis runner and shared bootstrap outputs are
implemented. The question-aligned covariance-aware Bayesian model and its plots are now
implemented; real-data inference and sensitivity checks still require validation; they are not launch prerequisites for model
training. Approved Table 1 and clinically reviewed endpoints remain manuscript
release prerequisites.

## Resume the existing 256 pretraining run

Inside a new GPU allocation after the original job stopped:

```bash
bash scripts/run_pretrain_hidden256.sh resume
```

The launcher reuses its usual 256 run directory, saved `hydra/config.yaml`
(or `.hydra/config.yaml`) and `version_0/last.ckpt`. It restores training state
and targets 20 total epochs, with early stopping disabled for finishing that
budget. Architecture environment variables need not be repeated. Set
`BONSAI_256_RUN_DIR` only if the original run used a custom directory.
`resume-preflight` prints the resolved resume config without training.
Missing saved config/checkpoint is an error; it never falls back to fresh
training. An optional server `setup_v100.sh` is sourced if present; otherwise
use an already activated environment, optionally specifying `OPERA_PYTHON`.
