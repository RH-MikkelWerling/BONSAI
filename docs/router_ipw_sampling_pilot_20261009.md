# All-horizon event sampling pilot

This is a single controlled tuning ablation of the clean Time2Vec interactive
router at residual scale 0.3. The approved candidate YAML remains unchanged.
The launcher adds only `training.batch_sampling` overrides and uses a separate
run directory. Held-out data must remain closed.

The proposal samples patients with replacement from 50% natural-uniform draws
and 50% event-positive outcome-horizon draws. All six BCE horizons are included:
30, 60, 90, 180, 365, and 730 days. Within each horizon, supported outcomes
receive focus probability proportional to the inverse square root of their
training event count. A task needs at least 10 distinct events and 20 known
negative labels. Every draw carries its exact inverse proposal probability
`1 / (N q_i)` into the horizon BCE and smoothness penalty. The BCE uses
self-normalized weighted means, so correction is consistent but not exactly
unbiased at finite logical-batch size. Validation is never resampled or weighted.

The 50% uniform component ensures every training patient has positive
probability and caps their importance weight at 2. The sampler prints its
expected effective-sample fraction and expected unique-patient fraction per
logical batch. A horizon without a supported focus task aborts the run.

After syncing the changed Python files, launcher, and three test files to the server:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh

export OPERA_RELEASE_DIR="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
export OPERA_FAMILY_DIR="$PWD/data/results/approved_training_families_v1_20261006T052120Z"
export OPERA_CANDIDATE_DIR="$PWD/data/results/approved_neural_candidates_router_clean_20261007_v1"
export OPERA_NEURAL_RUN_ROOT="$PWD/data/models/approved_neural_router_sampling_20261009_v1"
export OPERA_SAMPLING_ABLATION=ipw_event_horizon_bce
export OPERA_RESUME=0

python -m pytest tests/test_minority_event_sampling.py \
  tests/test_competing_risk_loss.py tests/test_direct_cr_contract.py -q
OPERA_LAUNCH=0 bash scripts/run_approved_neural_candidate_v1.sh \
  time2vec_router_interacting_scale03_e2e
OPERA_LAUNCH=1 bash scripts/run_approved_neural_candidate_v1.sh \
  time2vec_router_interacting_scale03_e2e
```

The run directory is
`$OPERA_NEURAL_RUN_ROOT/time2vec_router_interacting_scale03_e2e_ipw_event_horizon_bce`.
For an interrupted run, set `OPERA_RESUME=1` and use the same launch command;
the launcher requires a nonempty `last.ckpt` and refuses a completed run.

Before interpreting a gain, compare the fixed-checkpoint tuning predictions
with the random-sampling scale-0.3 run on the same outcome-horizon cells. Check
endpoint-group-macro CIF-AUC, per-horizon rare-task discrimination, calibration,
and negative transfer. If the weighted run helps, repeat with another seed
before choosing a finalist. This is an ablation, not an automatic replacement
for the original run.
