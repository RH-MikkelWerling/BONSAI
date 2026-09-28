# Independent OPERA launches

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
