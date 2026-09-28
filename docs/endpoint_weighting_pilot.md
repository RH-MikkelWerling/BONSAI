# Endpoint weighting pilot v1

Two matched low-rank multi-horizon BCE runs start from the same DAPT checkpoint,
seed, data, curriculum, learning rates and 30-epoch budget. Only the training
weighting differs. Both select checkpoints by the SAME new tuning-only score.
The historical run's family-selected best checkpoint is not a matched substitute:
either reconstruct the new score at every saved epoch, or run this reference.

Version 1 groups suffix `_g2plus`, `_g3plus`, `_g4plus` variants of the same named
endpoint. All other endpoints remain separate, including composites and their
components. This reduces duplicate grade votes, not all endpoint correlations.
The complete mapping is materialized in both generated configs and their resolved
Hydra configs. Clinical families remain descriptive and unchanged in this pilot;
their clinical revision is a separate task, not an architectural change here.

Reference loss: existing hierarchical-support aggregation and priority weights.
Revised loss: average eligible-patient BCE separately at each horizon, average
available horizons, average available grades within each endpoint group, then
average active groups. No family priority or support-shrinkage factors. Existing
hazard smoothness penalty remains. Curriculum multipliers still apply to grade
losses (including their ramp); absent grade labels are excluded from denominators.
This intentionally changes horizon normalization and support weighting together
as one policy comparison; it cannot isolate their separate causal effects.
Patients within a cell are equally weighted, not cohorts equally weighted.

Selection: tuning cells with >=20 observed cases and >=20 controls, determined
only from tuning labels. Horizon means -> grade means -> equal endpoint-group
mean of IPCW CIF AUC. Brier is logged by the same hierarchy as a secondary metric.
No eligible groups or nonfinite supported-cell metrics fail the run. The inventory
is exported as `endpoint_selection_inventory.json`, verified unchanged each epoch,
and must match between the two runs. Coverage is logged with the score. The support
threshold is pragmatic, not a precision guarantee. No held-out selection.

Prepare configs (already checked in; regeneration optional):

```bash
python -m scripts.prepare_endpoint_weighting_pilot
```

On EACH of two allocated GPU nodes, load the synced environment, then run ONE:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
export OPERA_EXPERIMENT_ROOT="$PWD/data/models/endpoint_weighting_pilot_v1"
TASK=weightinggroups   # use weightingreference on the other node
mkdir -p "$OPERA_EXPERIMENT_ROOT"
nohup bash scripts/run_opera_independent_experiment.sh "$TASK" "$OPERA_EXPERIMENT_ROOT" > "$OPERA_EXPERIMENT_ROOT/${TASK}_console.log" 2>&1 &
echo "PID: $!"
tail -f "$OPERA_EXPERIMENT_ROOT/${TASK}_console.log"
```

Before broader launches: compare the selected validation group score, grouped
Brier, coverage and per-endpoint gains/losses at matched horizons. Examine rare
versus common endpoints and clinical families descriptively. Do not demand that
one seed prove a small difference. If results are effectively tied, favor the
simple policy, explicitly report uncertainty, and confirm consequential differences
with another seed. Do not tune new weights on the held-out split. Loss weighting
and model selection are separate decisions even if the loss policy loses.
