# Local SEP-target sensitivity pilot

This exploratory experiment trains two fresh, matched autoregressive encoders:
exclude SEP targets versus include SEP targets. SEP remains an input in both.
Both inherit the actual saved production pretraining configuration, initialization
seed, data, vocabulary, architecture and epoch budget. Early stopping is disabled
in both; downstream models use each encoder's final-epoch last.ckpt. Their ordinary
pretraining validation losses include different target sets and must not be used
to choose the winning arm. This pilot does not add a clinical-token-only loss.

Each arm then trains the actual all-disease/all-outcome shared horizon-BCE joint
recipe with fine-tuning seeds 42, 43 and 44, retaining its tuning-selection rule.
Only encoder checkpoint and paired fine-tuning seed settings change. This uses
the production contrastive_multicohort entrypoint, not the older joint example.
No routers, held-out evaluation, primary run inventory or manuscript numbers are
changed. Existing published-model checkpoints are not overwritten or resumed.
One pretraining seed is a pilot, not evidence across pretraining initializations.

## Prepare on the server

From the repository root, with the usual server environment configured:

```bash
sensitivity_root=data/paper_256_v3/sep_target_sensitivity_20261028
.venv-v100/bin/python -m opera.run.prepare_sep_sensitivity \
  --env-file .env.rehearsal \
  --joint-config data/paper_256_v3/bundles/stage5/configs/seed42/L/J_all_diseases_all_outcomes.yaml \
  --output-dir "$sensitivity_root"
```

The preparer follows that joint recipe's dapt_ckpt to find its original pretraining
hydra/config.yaml or .hydra/config.yaml. If unavailable, pass
`--pretrain-config /actual/pretraining/run/hydra/config.yaml` explicitly. Do not
substitute an architecture example containing unknown head or layer counts.
Inspect sensitivity.json and the generated configs before launching. Every config
is hashed; the runner refuses subsequent edits. Use a fresh root for changed plans.

## Launch within GPU allocations

One allocated GPU can execute the arms sequentially:

```bash
mkdir -p logs
nohup bash -c '
  set -e
  .venv-v100/bin/python -m opera.run.run_sep_sensitivity --root "$1" --arm exclude_sep
  .venv-v100/bin/python -m opera.run.run_sep_sensitivity --root "$1" --arm include_sep
  .venv-v100/bin/python -m opera.run.summarise_sep_sensitivity --root "$1"
' _ "$sensitivity_root" > logs/sep_sensitivity_20261028.log 2>&1 &
```

With two allocated GPUs, launch one arm on each instead (do not also launch the
sequential command):

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv-v100/bin/python -m opera.run.run_sep_sensitivity \
  --root "$sensitivity_root" --arm exclude_sep > logs/sep_exclude.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup .venv-v100/bin/python -m opera.run.run_sep_sensitivity \
  --root "$sensitivity_root" --arm include_sep > logs/sep_include.log 2>&1 &
```

Use the devices assigned by the scheduler. After both parallel arms finish:

```bash
.venv-v100/bin/python -m opera.run.summarise_sep_sensitivity --root "$sensitivity_root"
```

The per-arm lock prevents duplicate launchers. Fine-tuning is preflighted before
each fit and produces tuning predictions using best.ckpt. All six evaluation
exports must exist before summarisation. Training logs are under each run's
sensitivity.log. Entry points resume their own last.ckpt after interruption;
evaluation outputs are immutable, so inspect existing completed predictions before
restarting an already evaluated arm.

## Interpretation

paired_tuning_analysis/paired_means.csv contains equal-cell and disease-balanced
paired effects with frequentist multiplier patient-bootstrap intervals, separately
for each fine-tuning seed. Eligibility is at least five cases and five controls;
positive gains favour SEP-inclusive pretraining, including reversed Brier gains.
The comparison uses fine disease membership from patient metadata. By default its
path comes from OPERA_PAPER_PATIENT_METADATA during preparation; override with
--patient-metadata if needed. Main-paper aggregation code is reused.

Evaluate AUC, Brier and IPA together; do not select a favourable seed or endpoint.
Patient-bootstrap intervals condition on the fitted models and do not quantify
pretraining variability. A consistent substantial improvement warrants replication
with further pretraining seeds before contemplating broader reruns. These tuning
results do not constitute an untouched test confirmation.
