# Locked guarded rank-8 router: remaining runs

Current-backbone reference: `data/models/approved_neural_router_guarded_20261011_v1/time2vec_router_interacting_guarded_scale07_e2e`. Keep its scale 0.7, rank 8, home floor 0.05, event tau 10, natural sampling, seed, encoder, release, 30-epoch budget, and endpoint-group macro selection fixed for the source-bank factorial. The three controls remove only the disease sources, outcome sources, or both. They are separate GPU jobs; these scripts do not submit jobs.

```bash
bash scripts/run_guarded_router_source_arm.sh prepare
bash scripts/run_guarded_router_source_arm.sh disease_only preflight
bash scripts/run_guarded_router_source_arm.sh disease_only launch
# On other GPU allocations, after the same prepare:
bash scripts/run_guarded_router_source_arm.sh outcome_only launch
bash scripts/run_guarded_router_source_arm.sh none launch
```

Use `resume` in place of `launch` if a job stops with `last.ckpt`. Do not put two arms in the same GPU allocation. After training, compare the four arms on the same tuning cells and prespecified epoch rule; keep held-out sealed.

Plan independent fine-disease by outcome specialists, inspect the exact count, then run bounded batches. These are full 30-epoch, DAPT-initialized fits selected by single-cell validation loss, not pooled-model fine-tuning. The queue checks that the DAPT path matches the locked router. Existing CLL v2 fits can be kept as a separate, documented queue; use `OPERA_SPECIALIST_EXCLUDE_CLL=1` while planning to avoid duplicate CLL training. Choose the number of GPU shards **once at plan time**; the manifest assigns each approved cell to exactly one shard, with separate checkpoints, logs, and lock files.

```bash
export OPERA_DAPT_CKPT="$PWD/data/models/daly_care_t100_joined_bins_with_sor_final_v1/pretrain_harmonized_scaled_time2vec_len8192_20ep/final_b4_20261003/version_0/best.ckpt"
export OPERA_SPECIALIST_EXCLUDE_CLL=1  # only if the prior CLL v2 queue is retained
export OPERA_SPECIALIST_SHARD_COUNT=4   # set to the specialist GPU slots available
bash scripts/run_independent_specialist_grid.sh plan
# Review manifest_report.json and models_per_shard before launching.
export OPERA_SPECIALIST_MAX_NEW=25
export OPERA_SPECIALIST_SHARD_INDEX=0   # use 0, 1, 2, 3 on four allocations
export OPERA_LOCAL_GPU=0                # GPU index local to this allocation
bash scripts/run_independent_specialist_grid.sh run
# Repeat run with the same shard index on subsequent allocations; completed cells are skipped.
```

Run each specialist shard in a separate GPU allocation. The three router source arms can also run in separate allocations, concurrently with specialist shards if enough GPUs are available. Budget GPU slots across both groups; the scripts do not reserve GPUs or submit cluster jobs.

For the first single-GPU smoke/timing batch after the four-shard plan already exists, run `bash scripts/run_specialist_single_gpu_pilot.sh` on one GPU allocation. It attempts at most two new fits from shard 0, preserves the existing manifest, and can be run again to advance by another two cells. It is a queue smoke test, not a representative runtime sample across diseases; inspect the two cell logs and timings before increasing `OPERA_SPECIALIST_MAX_NEW`.

To let one GPU work continuously through the full planned grid instead, run `bash scripts/run_specialist_single_gpu_full.sh`. It processes shards 0–3 sequentially, skipping completed fits and recorded failures on restart. The existing grid can still be parallelized later by launching individual shard workers on other GPUs; each shard has its own lock. A full pass means all cells were attempted or previously classified as failed, **not** that every cell yielded an estimable specialist. Inspect `logs/*.failed` and `status.tsv` in every shard before evaluating.

Small-disease fix (2026-10-12): the shared loader formerly used `drop_last=True` even when fewer than one logical batch (256 patients) remained, producing zero specialist training steps. It now retains that partial batch only when no full batch exists; pooled runs and specialists with at least one full batch retain the original drop-last behavior. The full one-GPU launcher retries prior failures whose logs contain `No training batches.`, moving their old `.failed` markers to `.failed.before_partial_batch_fix` for audit. Other failures remain skipped and still require inspection. Sync the fix before restarting the full command; do not erase old logs or checkpoints.

The manifest counts approved fits, not evaluable disease/outcome/horizon cells. Audit support and classify inestimable held-out cells before making specialist-versus-router claims. After the grid starts, compute runtime from the first batch and estimate total GPU-hours. Freeze this roster and evaluation code before opening held-out. The future pretrained-backbone repetition must use separate candidate/run roots and a new explicitly matched DAPT checkpoint; do not mix results with this backbone.

Tuning-only route figures for the selected router can proceed independently:

```bash
export OPERA_ROUTER_RUN="$PWD/data/models/approved_neural_router_guarded_20261011_v1/time2vec_router_interacting_guarded_scale07_e2e"
export OPERA_ROUTER_ATLAS_DIR="$PWD/data/results/router_guarded_rank8_full_atlas_20261012_INTERNAL"
export OPERA_ROUTER_PUBLICATION_DIR="$PWD/data/results/router_guarded_rank8_publication_atlas_20261012_INTERNAL"
bash scripts/make_scale03_router_full_atlas.sh
bash scripts/make_router_publication_atlas.sh
```

XGBoost/tabular baselines remain a separate workstream. No results from that workstream are assumed here.
