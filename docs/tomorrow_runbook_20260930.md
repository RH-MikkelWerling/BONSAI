# OPERA runbook for 2026-09-30

This runbook is validation-only. Do not open held-out predictions or change the
code/config/data underneath the currently running `opera_overnight_router_lr_v4`
queue. Its first task is the interacting router; its second is encoder LR 0.20.
The optional 1.00 run is **not** automatically queued. The existing
`weightinggroups` run is the 0.05 comparator, not a new task to repeat.

## 1. Finish and inventory tonight's runs before syncing

On the allocated server node:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
RUN_ROOT="$PWD/data/models/opera_overnight_router_lr_v4"
tail -n 80 "$RUN_ROOT/overnight_queue.log"
find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -name training_complete.json -print
find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -name .launch_lock -print
pgrep -af 'run_opera_overnight_router_and_lr|contrastive_multicohort' || true
```

Expected completed runs are
`routerinteraction__direct_cr_bce_router_interacting_endpoint_groups_v1__seed42`
and `encoder020__direct_cr_bce_lowrank_endpoint_groups_v1__seed42`.
A completion marker without a clean queue exit needs investigation. An active
`.launch_lock` is not by itself a failure while training is live. Do not start
another job on that GPU and do not delete a lock without checking its process.
Because tonight's first launch exposed a partial sync, verify the server code
before interpreting the run, without changing it while the process is live:

```bash
rg -n 'inclusive: bool = False' bonsai/functional/censoring.py
rg -n 'routerinteraction' scripts/run_opera_independent_experiment.sh
rg -n 'adapter_query_mode|interacting' opera/modules/networks/source_anchored_readout.py
git status --short
```

These checks establish the checkout *now*, not necessarily the imported code
at process start if files were replaced mid-run. If source identity cannot be
established, label the overnight result exploratory and repeat the selected
condition from a pinned, fresh checkout before making a publication claim.

Once both runs are finished, export **validation-only** metadata. The bundler
does not include predictions, logs, checkpoints or held-out evaluation files:

```bash
OUT="$PWD/data/results/recent_validation_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
python -m opera.run.bundle_recent_validation \
  --root "$PWD/data/models/opera_overnight_router_lr_v4" \
  --root "$PWD/data/models/endpoint_weighting_pilot_v1" \
  --root "$PWD/data/models/outcome_readout_pilot_harmonized_v4" \
  --output "$OUT"
echo "$OUT"
```

If an additional suite root exists, add another `--root` rather than assuming
that a missing archive member means the run never happened. Keep the already
exported XGBoost validation aggregate CSV. For a fresh server export, use
`opera.run.export_validation_evaluation_aggregates` on its `evaluation/`
directory; it rejects results without an explicit validation split. Do not
copy patient-level prediction files off the server.

## 2. Review validation before commissioning further runs

Compare the configured-monitor-selected checkpoints at **365 days** on the
same outcome panel: endpoint-group macro, family macro, pooled per-outcome
CIF-AUC mean, Brier, calibration and minimum support. Compare routerinteraction
to the additive router **with the same shared-BCE source and frozen encoder**.
Compare encoder020 to weightinggroups/0.05 with the same DAPT source and
readout. An improvement under a different source, monitor or denominator is
not an architecture-only improvement. The existing XGBoost mean of 0.7752
versus neural 0.7464 is descriptive, not a matched estimand; do not rank them
definitively until validation predictions are evaluated on identical patients,
eligibility, competing-event and censoring rules. Do not choose on held-out.

Decision: run encoder100 only if the 0.20 result moves a prespecified
validation metric enough to justify another full run. Otherwise prioritize the
matched baseline, repeat seed and final-data work. The interacting query alone
cannot prove that *learned routing* is beneficial: fixed-routing and
continued-source controls remain needed if it looks promising.

**Router initialization comparison:** tonight's `routerinteraction` loads the DAPT
encoder and then strictly loads *all* compatible shared horizon-BCE tensors;
only the new adapter/router tensors lack a source counterpart. The encoder is
frozen. Thus this is a warm-started readout-adaptation test, not a router
trained de novo from DAPT. A DAPT-only initialization of the same router with
a fresh shared BCE head (preferably trained end-to-end) remains a separate,
comparison. The opt-in `routerfresh` task now supplies this explicit mode and
its own generated config. It uses DAPT initialization only, a fresh BCE head
(bias initialized from training-only hazards), fresh interacting adapters and
end-to-end encoder training at the existing 0.05 multiplier. Run it in a new
output root after the active job finishes and after syncing the complete change
set. Do not interpret a
negative warm-started result as ruling out end-to-end routing, or a positive
one as proving that routing rather than source initialization helped.

`routerfresh` changes both shared-BCE warm start and encoder freezing. It is a
test of the full proposed end-to-end recipe, not an isolated causal test of
initialization. A DAPT-fresh frozen control would be required to isolate the
warm-start effect. Launch after server preflight with:

```bash
RUN_ROOT="$PWD/data/models/opera_router_dapt_fresh_v1"
python -m opera.run.preflight_direct_competing_risk \
  --config-name generated/direct_cr_bce_router_interacting_dapt_fresh_v1 \
  --require-existing-paths \
  --override "paths.sequence_data_dir=$SEQUENCE_DATA_DIR" \
  --override "dapt_ckpt=$DAPT_CKPT"
nohup bash scripts/run_opera_independent_experiment.sh routerfresh "$RUN_ROOT" \
  > "$RUN_ROOT.launch.log" 2>&1 < /dev/null &
echo "PID: $!"
```

## 3. Sync after the queue stops, then preflight the new checkout

Sync the **whole current code/config/test change set**, not just the two shell
scripts. In particular include:

- `scripts/run_opera_independent_experiment.sh`,
  `scripts/run_opera_overnight_router_and_lr.sh`;
- both generated router YAMLs under `opera/configs/generated/` (currently
  untracked locally), plus the low-rank endpoint-group YAML;
- `opera/modules/networks/{opera_nets.py,source_anchored_readout.py}`,
  `opera/functional/direct_cr_contract.py`,
  `opera/run/generate_sweep_configs.py`;
- the BONSAI censoring, tokenization, sampler, embedding, finetuning and
  metric fixes, and OPERA's `OutcomeFinetuneDataModule.py`/`run/finetune.py`;
- tests and analysis/bundling scripts added since the prior sync.

Do **not** sync local `data/`, model checkpoints, `.pytest*/`,
`.pytest_bayes/`, or generated test artifacts into the server code tree.
Two tracked `.pytest_bayes/.../posterior.nc` files were rewritten by an older
test run locally; those are test artifacts, not training inputs. The tests now
write under their pytest temporary directory, but the already-modified binary
files should be reviewed separately before any commit.

After syncing, verify the exact failure modes seen tonight:

```bash
bash -n scripts/run_opera_independent_experiment.sh scripts/run_opera_overnight_router_and_lr.sh
rg -n 'routerinteraction|encoder020' scripts/run_opera_independent_experiment.sh
for f in \
  opera/configs/generated/direct_cr_bce_router_interacting_endpoint_groups_v1.yaml \
  opera/configs/generated/direct_cr_bce_lowrank_endpoint_groups_v1.yaml; do
  test -s "$f" || { echo "MISSING: $f"; exit 1; }
done
python -m pytest -q tests/test_functional/test_censoring.py \
  tests/test_finetune_max_len.py tests/test_source_anchored_readout.py \
  tests/test_recent_validation_bundle.py
```

With `setup_v100.sh` loaded, run the read-only config/path preflights too:

```bash
export SEQUENCE_DATA_DIR="$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_harmonized_v4"
export DAPT_CKPT="$BONSAI_MODELS/daly_care_t100_joined_bins_harmonized_v4/pretrain_harmonized_fourier_len8192_allwindows_20ep/version_0/best.ckpt"
export OPERA_SHARED_BCE_CKPT="$PWD/data/models/outcome_readout_pilot_harmonized_v4/direct_cr_objective_horizon_bce/best.ckpt"
python -m opera.run.preflight_direct_competing_risk \
  --config-name generated/direct_cr_bce_router_interacting_endpoint_groups_v1 \
  --require-existing-paths \
  --override "paths.sequence_data_dir=$SEQUENCE_DATA_DIR" \
  --override "dapt_ckpt=$DAPT_CKPT" \
  --override "source_opera_ckpt=$OPERA_SHARED_BCE_CKPT"
python -m opera.run.preflight_direct_competing_risk \
  --config-name generated/direct_cr_bce_lowrank_endpoint_groups_v1 \
  --require-existing-paths \
  --override "paths.sequence_data_dir=$SEQUENCE_DATA_DIR" \
  --override "dapt_ckpt=$DAPT_CKPT"
```

Check that no overnight job is still active before syncing model code. New
experiments after the censoring fix need a **new versioned output root**;
never resume a historical checkpoint after changing its code or inputs.

## 4. Freeze the final MEDS release before final training

The final dataset is not yet available locally. Give it a new immutable path
and record file sizes/hashes, vocabulary size/hash, sequence tensor schema,
train/tuning/held-out subject IDs, first-line index-date counts and endpoint
event/censoring counts. Compare these against v4 and run the split-contract
validator. In particular, establish whether the final vocabulary and numeric
value representation can load the v4 DAPT checkpoint strictly; do not assume
compatibility because the outcome counts look similar. If not, retrain DAPT
and start a new experiment series. Keep the final MEDS data out of the active
v4 queue and use a fresh model root.

On the server, before requesting a final-data GPU run, compose and check each
selected config using `opera.run.preflight_direct_competing_risk` with
`--require-existing-paths` and explicit `paths.sequence_data_dir`, `dapt_ckpt`
and (for a router) `source_opera_ckpt` overrides. Then run a bounded training
smoke test and validation evaluation on the final-data root. The PHAIR model
also needs an early checkpoint/config/vocabulary load-and-forward-pass test in
this environment before it becomes a source model.

## 5. Prioritized experiments after final-data validation

1. Reproduce the selected simple shared BCE, endpoint-group low-rank BCE and
   pooled XGBoost on final MEDS with the same validation cohort and metric
   implementation. This is the publication comparison foundation.
2. If the interacting router genuinely improves the matched validation score,
   run fixed-routing and continued-source controls, then a second seed. If it
   does not, stop the router expansion.
3. Run a single prespecified outcome-balancing ablation against the selected
   BCE setup only after implementing/testing inverse-square-root emphasis,
   minimum **unique** event support, capped repeat exposure, equal optimizer
   steps and calibration reporting. The current multi-outcome event-aware
   sampler has support thresholds, but is **not** this balanced experiment.
4. Disease mixture/pan-disease versus disease-specific comparison and Bayesian
   disease-benefit forest plot need matched outcome-by-disease validation cells,
   uncertainty and the same final data. Do not launch an exhaustive sweep
   before the first three gates above.

This is a staged decision plan, not a command to run all candidates tomorrow.
