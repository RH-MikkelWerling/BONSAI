# Independent OPERA launches

## Bounded overnight pilot on one GPU (29 September 2026)

After syncing, this queue runs `routerinteraction` then `encoder020`
sequentially in separate directories. The existing additive router is the
query-mode comparator; the completed `weightinggroups` low-rank run is the
encoder-0.05 comparator. All use the harmonized v4 pilot data. The interacting
router starts from the completed shared-BCE source with frozen encoder; the
encoder-0.20 experiment starts from DAPT with trainable encoder and the
endpoint-group low-rank readout. These are two distinct ablations, not a direct
router-versus-low-rank causal comparison. Both select checkpoints on tuning only.

Use an **idle allocated GPU** and keep the allocation alive for both jobs:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
export OPERA_SHARED_BCE_CKPT="$PWD/data/models/outcome_readout_pilot_harmonized_v4/direct_cr_objective_horizon_bce/best.ckpt"
export RUN_ROOT="$PWD/data/models/opera_overnight_router_lr_v4"
mkdir -p "$RUN_ROOT"
nohup bash scripts/run_opera_overnight_router_and_lr.sh "$RUN_ROOT" > "$RUN_ROOT/overnight_queue.log" 2>&1 &
echo "Queue PID: $!"
tail -f "$RUN_ROOT/overnight_queue.log"
```

To append `encoder100` after both default jobs, set
`OPERA_OVERNIGHT_INCLUDE_ENCODER100=1` **before** launch. Leave it unset for a
bounded two-job night. The queue stops if a preflight or training task fails,
preserves completed earlier jobs and does not touch existing completed runs.
The child launcher checks its input files and per-run locks. Do not start it
while another process uses that GPU. This queue does not implement minority
oversampling: unique-patient support safeguards and censor-aware correction
must be designed/tested before such a run is scientifically interpretable.

The query-mode ablation alone cannot isolate the benefit of **learned routing**
because adapted readout parameters also train. Fixed-routing and continued-
shared-source controls remain pending. The final MEDS release requires a new,
explicitly versioned suite; do not silently point these v4 runs at that data.

## Endpoint-group router queued after the weighting reference

### Interacting query (separate run)

`TASK=routerinteraction` uses `direct_cr_bce_router_interacting_endpoint_groups_v1`.
The learned query is `p + d + o + MLP(concat(p, d, o))`, where `p` is the
projected patient representation, `d` the fine-disease embedding and `o` the
target-outcome embedding. The MLP is Linear -> GELU -> Linear, with hidden
width 128 and no dropout. Disease/outcome embeddings start with nonzero random
values rather than identical zeros. Source keys, home floors, anchored gradient
ownership, support shrinkage and adapter initialization are unchanged.

This enables patient-dependent disease/outcome effects on relative source
preferences. It does not guarantee selective or useful routing, and it does
not address the small adapter-gradient initialization bottleneck by itself.
The encoder remains frozen to isolate the query change. Use the same shared-BCE
source as the additive control; do not resume its checkpoints in this new model.
Missing `adapter_query_mode` defaults to additive for old config/checkpoint
compatibility. The new task uses a distinct output directory.

After syncing model code, launcher and generated config, launch only when the
GPU is free:

```bash
nohup bash scripts/run_opera_independent_experiment.sh routerinteraction "$RUN_ROOT" > "$RUN_ROOT/routerinteraction_console.log" 2>&1 &
tail -f "$RUN_ROOT/routerinteraction_console.log"
```

`routergroups` is the anchored adapter/router with endpoint-group loss
aggregation, no extra family weights, and endpoint-group-macro checkpoint
selection (minimum 20 cases/controls). Its source must be the original shared
horizon-BCE checkpoint, not either low-rank weighting checkpoint. The encoder
freezing and source anchoring remain as designed; this is not a matched-cost
architecture-only comparison to the DAPT-initialized low-rank pilot.

After syncing the launcher, queue script and generated router config, use:

```bash
source setup_v100.sh
export OPERA_SHARED_BCE_CKPT="$PWD/data/models/outcome_readout_pilot_harmonized_v4/direct_cr_objective_horizon_bce/best.ckpt"
export RUN_ROOT="$PWD/data/models/opera_independent_v4"
mkdir -p "$RUN_ROOT"
nohup bash scripts/run_router_after_weighting.sh "$RUN_ROOT" > "$RUN_ROOT/routergroups_queue.log" 2>&1 &
echo "Queue PID: $!"
tail -f "$RUN_ROOT/routergroups_queue.log"
```

The default wait target is the seed-42 `weightingreference` directory under
RUN_ROOT (or OPERA_SEED if set). Pass its actual directory as the second script
argument if different. The queue checks source checkpoint type up front, waits
for `training_complete.json` and launch-lock removal, and times out after 12h
without launching on failure. Normal launch preflight and strict source loading
still check paths and compatibility. It does not stop/modify the current run,
does not inspect held-out predictions, and requires the GPU allocation to stay
active. Do not queue it twice or put unrelated jobs on that same GPU meanwhile.

Sync code, generated configs and this launcher before starting. Load the server
environment on each allocated GPU node. Each invocation writes its own Hydra
directory, checkpoints and appended log, preflights inputs and resumes incomplete
training. It never generates configs or waits for another job. Shared inputs must
remain immutable while jobs run. Separate GPU nodes can use the same output root.
Do not run concurrent jobs on the same GPU unless memory has been tested.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
export OPERA_EXPERIMENT_ROOT="$PWD/data/models/opera_independent_v4"
export OPERA_SHARED_BCE_CKPT="/absolute/path/to/completed/shared_bce/best.ckpt"
# Select this using tuning only; lowrank is a candidate, not a declared winner.
export LR_READOUT_CONFIG=direct_cr_objective_horizon_bce_factorized_cohort_outcome_prototypes
```

Run ONE of these per allocated GPU/node (not all on the same node):

```bash
bash scripts/run_opera_independent_experiment.sh lowrank "$OPERA_EXPERIMENT_ROOT"
bash scripts/run_opera_independent_experiment.sh anchored "$OPERA_EXPERIMENT_ROOT"
bash scripts/run_opera_independent_experiment.sh unanchored "$OPERA_EXPERIMENT_ROOT"
bash scripts/run_opera_independent_experiment.sh encoder005 "$OPERA_EXPERIMENT_ROOT"
bash scripts/run_opera_independent_experiment.sh encoder020 "$OPERA_EXPERIMENT_ROOT"
bash scripts/run_opera_independent_experiment.sh encoder100 "$OPERA_EXPERIMENT_ROOT"
```

For detached execution, choose TASK on that node:

```bash
TASK=lowrank
mkdir -p "$OPERA_EXPERIMENT_ROOT"
nohup bash scripts/run_opera_independent_experiment.sh "$TASK" "$OPERA_EXPERIMENT_ROOT" > "$OPERA_EXPERIMENT_ROOT/${TASK}_console.log" 2>&1 &
echo "Launcher PID: $!"
tail -f "$OPERA_EXPERIMENT_ROOT/${TASK}_console.log"
```

The low-rank baseline and encoder005 have identical training settings if the
chosen LR readout is the low-rank config. They are not independent seed replicas;
run one and reuse its results. That gives five distinct conditions. A sixth GPU
can repeat a prespecified condition with `OPERA_SEED=43`, or remain free.

At head LR 0.0005, encoder multipliers 0.05 / 0.20 / 1.00 give encoder rates
0.000025 / 0.0001 / 0.0005 before scheduling. Keep the head rate, splits, curriculum,
loss weighting, initialization and training budget fixed in this sweep. Do not
change LR_READOUT_CONFIG between its three runs or select it on held-out data.

Anchored and unanchored adapters require the same *existing* shared-BCE source
checkpoint; they freeze the encoder initially. The low-rank and LR runs start
from DAPT, not the new adapter run. Consequently parallel launch is valid, but
adapter-vs-DAPT comparisons are staged, not matched-cost architecture ablations.
The prototype model uses grouped cohorts; adapters use fine cohorts. Current
configs retain hierarchical support aggregation, family weights and family-macro
checkpoint selection: these are not newly neutralized comparisons. Settle the
selection policy before final confirmatory runs; do not change it mid-sweep.

After a hard kill, an empty `.launch_lock` directory may remain. Verify no process
is using that exact run, then remove only that empty lock with `rmdir` before
relaunching. Never change an experiment's inputs/config to resume its checkpoint;
use a new output root for a revised protocol.
