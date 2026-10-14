# Full CLL specialist queue

This queue attempts every non-excluded outcome in the approved plain shared-BCE
panel for `cohort_fine=CLL`. Each cell is an independent, end-to-end 30-epoch
model initialized from the same local Time2Vec DAPT checkpoint—not from pooled
supervised weights or the five-epoch timing pilot. The model predicts all six
prespecified horizons. It uses the approved natural patient sampling and a
single-outcome validation-loss checkpoint rule; keep that rule fixed across
cells and report it in comparisons. The singleton endpoint-group AUC reporting
hook is disabled: unlike validation BCE, it is undefined when a sparse tuning
cell has fewer than 20 cases or controls. Individual AUCs are still logged
where estimable.

Sync `scripts/run_cll_independent_specialist_queue.sh` and the updated
`opera/run/prepare_independent_specialist_pilot.py` to the secure server. On a
GPU allocation with sufficient walltime:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
export OPERA_CANDIDATE_DIR="$PWD/data/results/approved_neural_candidates_router_clean_20261007_v1"
export OPERA_DAPT_CKPT="$PWD/data/models/daly_care_t100_joined_bins_with_sor_final_v1/pretrain_harmonized_scaled_time2vec_len8192_20ep/final_b4_20261003/version_0/best.ckpt"
export OPERA_SPECIALIST_ROOT="$PWD/data/models/cll_independent_specialists_20261010_v2"
export OPERA_RELEASE_DIR="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
export OPERA_METADATA_PATH="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet"
export OPERA_OUTCOME_LABEL_CACHE_DIR="$PWD/data/results/outcome_label_cache_approved_v2_INTERNAL"
export CUDA_VISIBLE_DEVICES=0

bash scripts/run_cll_independent_specialist_queue.sh
```

The launcher owns one GPU, runs cells sequentially, and locks its output root
against concurrent launches. It skips completed cells and automatically resumes
an interrupted cell from its `last.ckpt` on the next invocation. Reuse exactly
the same environment and output root after a scheduler walltime stop. Each
cell has a log under `logs/`, a full YAML under `configs/`, and checkpoints
under `runs/`. `status.tsv` is append-only. A cell that fails is marked
`logs/cll_<outcome>.failed` and skipped on future invocations until reviewed;
three failures in one invocation stop the queue for inspection.

Use the new `v2` output root if `v1` already contains specialist YAMLs or
checkpoints. Do not silently modify a YAML belonging to a completed fit. The
earlier `v1` results can remain on the server as a diagnostic archive, but do
not combine its checkpoint-selection configuration with the `v2` paper grid.

Do not treat every attempted cell as estimable. The queue records failures
instead of silently erasing them. Before the paper analysis, verify exact
train/tuning support at every horizon, classify unsupported cells, and evaluate
the supported CLL specialists on the same patient-outcome-horizon risk sets as
the pooled model. Keep held-out data closed until the model/estimand roster is
locked. A 30-epoch run does not by itself establish specialist validity.
