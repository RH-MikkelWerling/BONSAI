# OPERA experiment factory: claims, queue, gates and stopping rules

**Design review, 7 October 2026.** This is a specification for a durable,
stage-gated queue on the secure GPU cluster, not permission to submit new jobs
or to inspect held-out outcomes. It extends
`opera_paper_next_actions_20261007.md`. The study team manually syncs code and
source files to the secure server; this repository cannot see live GPU/job
state. The factory must be deployed and dry-run on the server before unattended
submission. PHAIR is an external checkpoint still in production.

## 1. Scientific contract: what a completed paper must answer

| Question | Necessary comparison | What must be held fixed | What it does **not** prove |
| --- | --- | --- | --- |
| Does joint supervised learning help scarce tasks? | Same-encoder, same-loss pooled shared model versus one fine-disease × outcome specialist in **every trainable cell** | Encoder, outcome definition, horizon, index, eligible patients, optimization budget and selection policy | A pilot or handful of cells is not a whole-landscape result |
| How much do disease versus outcome labels contribute to pooling? | All pooled; one disease/all outcomes; all diseases/one outcome; single-cell specialist | Same encoder, loss, split and readout; training-task set is the intended variable | Source-bank masking in a jointly trained router is *not* this comparison |
| Does adaptive borrowing beat ordinary pooled learning? | Router versus loss-matched pooled shared model, with a plain multi-horizon BCE reference also shown | Same outcome weighting/checkpoint metric, encoder and budget for the mechanistic contrast | Router versus **plain** BCE alone changes both readout and loss aggregation |
| Do disease and outcome source banks each matter? | No-source, disease-only, outcome-only, both under the same interacting query | Source-bank eligibility only | Gate mass is not source importance; source removal without retraining is diagnostic |
| Does joint patient × disease × target querying matter? | Fresh additive 0.1 versus interacting 0.1; separately compare interacting 0.1/0.2/0.3 | Same source banks, residual scale in the query contrast, seed/initialization and all inputs | A scale-0.3 versus additive-0.1 comparison cannot isolate interaction |
| Does broad pretraining add value and modify supervised sharing? | Local encoder, PHAIR direct, PHAIR plus local adaptation, each crossed with shared readout and selected router | Compatible architecture/features; same supervised split, loss, protocol and budget | If checkpoint/token/time systems differ without reconciliation, this is not a clean pretraining contrast |
| Does the approach beat a strong conventional predictor? | EHR-only XGBoost on the same patient/task/risk-set contract, with registry-augmented results separate | Baseline information cutoff, index, eligibility, horizon, metric and reporting population | Old tabular results on a different release are not a benchmark |

There are 24 fine diseases × 69 approved outcomes at most, **not** 24 × 69 ×
six specialists: one cell model can predict all six horizons. Training-support
and evaluation-support rules differ. Cells with no trainable specialist still
matter as a *coverage* analysis, but no AUC is claimed without adequate
evaluation events, controls and follow-up. Prespecify the primary 365-day
contrast and the six-horizon supplement. Report Brier and calibration beside
discrimination; include lower-tail harm and clinically important families,
not just a pooled mean.

**Specialist implementation gap.** `opera/run/finetune_all_horizons.py`
fine-tunes an already pooled-supervised direct-risk checkpoint. It is a useful
secondary *post-pooling adaptation* experiment, not the independent specialist
counterfactual in the first row above. The primary cell launcher must be built
from the common pretrained encoder, with no pooled supervised weights, while
matching the pooled direct-risk architecture, loss, horizons, eligibility and
selection rule. The older fixed-horizon `finetune.py` is not that matched arm.
Do not dispatch the full specialist grid until a pilot verifies this lineage.

**Identification caveats.** The plain shared BCE run is the sensible practical
reference, but the candidate generator deliberately changes its
`cross_outcome.aggregation` to `macro`; the grouped shared and router arms
use `endpoint_group_macro`. Therefore the effect of routing requires the
loss-matched grouped shared arm. The router consumes `cohort_fine` directly;
its advantage over a model without this covariate is not solely adapter
borrowing. Include an index-available-disease sensitivity, and, if the paper
claims an isolated *borrowing* effect, a disease-identity-aware non-borrowing
control. Likewise, later registry diagnosis can legitimately repair coding
lag but can be unavailable for a real-time treatment-date prediction.

## 2. One-shot user experience, with deliberate stop points

The desired interface is **one sealed plan and one explicit launch per stage**:

1. Operator syncs a frozen code/config bundle to the server, runs a read-only
   `plan`/`doctor`, reviews expected jobs, GPUs, walltime/cost ceiling and
   protected-data boundary.
2. Operator authorizes a stage, not each variant. A controller submits at most
   the stage's concurrency limit, records scheduler IDs, and continues until
   every authorized job either passes validation or enters a visible failure
   state. Re-invoking the controller is idempotent.
3. When all stage prerequisites complete, CPU reducers create a small
   disclosure-safe **review packet**: run ledger, checksums, common-epoch
   metrics, paired tuning contrasts, subgroup harms, resource use and
   unresolved failures. They do not launch successor stages by default.
4. A **hard gate** pauses the DAG for scientific review. The operator records
   a versioned decision (`continue`, `repair`, `stop`, `run predeclared
   replication`), the reviewer/date, and the reason. No best-model selection
   or held-out access occurs as an accidental dependency of a queue run.
5. The next authorized stage may then be started with one command. PHAIR
   arrival triggers its own compatibility gate; it does not stall unrelated
   specialist or tabular work.

This is an experiment factory, **not** an adaptive search that tries variants
forever. The full menu, contrast families, maximum seeds, maximum specialist
cells, GPU-hour/currency cap and stop gates are frozen **before** reviewing
the relevant tuning results. A gate can reduce the next stage or stop it; a
new hypothesis requires a separately named plan revision, never a silent
queue expansion. No automated job may target `held_out` before the final
analysis-lock signature.

The proposed operator commands (design names, **not implemented commands**)
are `doctor`, `plan`, `start --stage`, `status`, `review-packet`, and
`approve-gate --decision`. `start` only submits the named approved stage;
`status` can safely be run any number of times. The plan file, not the
terminal user's memory, supplies paths and variants. A representative
machine-readable stage record would contain:

```yaml
stage_id: router_fixed_tuning_v1
depends_on: [router_inventory_v1]
authorized_split: tuning
variants: [plain_shared, grouped_shared, additive_01, interactive_01,
           disease_sources, outcome_sources, no_sources]
resources:
  gpu_per_worker: 1
  max_active_gpu_jobs: 3
  max_queued_jobs: 8
  max_gpu_hours: APPROVED_NUMBER
  max_attempts_per_run: 2
outputs: [checkpoint_or_prediction, paired_aggregate, review_packet]
stop_after: review_packet
review_gate: router_architecture_decision_v1
```

`APPROVED_NUMBER` is intentionally unresolved: the operator sets it after
the resource pilot and site billing review. A manifest with an unresolved
budget cannot submit. All jobs in one stage share an approved release and
code bundle; per-variant differences are explicit and machine-checked.

### Suggested stage DAG and review packets

| Stage | Runs/actions | May run alongside | Hard stop / deliverable |
| --- | --- | --- | --- |
| 0: input contract | Hash approved outcomes, splits, sequence data, fine-disease roster, code/config, local pretrain checkpoints; reconcile 78,668 pretraining identities and 37,531 supervised members | Table 1 and tabular repairs | Signed input ledger; unresolved mismatch stops all new training |
| 1: current router inventory | Verify seven clean completed runs and plain/grouped shared controls; audit fresh equal-scale query modes | Specialist support counting, phenotype timing audit | Common-epoch/checkpoint packet; no numerical winner chosen from rounded loss |
| 2: fixed tuning evaluation | One export per frozen checkpoint, paired metrics; routes, CIF change and source diagnostics | Table 1, consort, figure drafts | Scientific review of architecture/scale, family harms and calibration; choose prespecified finalist(s) |
| 3: confirmation seeds | Matched-seed finalists and loss-matched baselines, bounded seed budget | Specialist pilot | Stop if effect is unstable, contract differs, or calibration degrades under prespecified tolerances; otherwise lock architecture |
| 4: specialist pilot | Support-stratified cells to estimate memory, runtime, convergence and event threshold behavior | Replication and tabular work | Review feasible grid/cost; no extrapolation from pilot to all cells |
| 5: specialist grid and pooling-axis factorial | Every trainable fine-disease × outcome cell; all-pooled / one-disease / one-outcome regimes as scoped in the frozen plan | PHAIR compatibility if available | Cell coverage/failure packet and matched tuning rarity contrasts; stop if grid infeasible and narrow the claim explicitly |
| P: PHAIR arrival | Checkpoint strict-load/forward test, vocabulary/time/numeric mapping, then local/PHAIR/PHAIR+adaptation × shared/router | Stages 4--5 | Compatibility gate first; pretraining-transfer review and compute-accounting gate second |
| 6: final lock | Freeze estimands, finalists, support, priors/bootstrap, figures, benchmark contract and exclusions | Nothing that changes model choice | Human-approved analysis-lock manifest; one-way opening of held-out |
| 7: held-out evaluation | One paired evaluation for frozen models/cells; no tuning of architecture or thresholds | Table formatting only | Immutable results bundle, uncertainty, harms, calibrated conclusions and disclosure review |

The *minimum main-paper set* before PHAIR is pooled shared BCE, loss-matched
shared, fresh additive/interacting matched-scale pair, disease/outcome/both
source-bank arms, the selected router, and the supported-cell specialist grid.
The no-source router is an architecture sanity control. The 0.2/0.3 scale
arms and exploratory readouts support tuning/supplementary analysis; they do
not justify new broad sweeps without a gate. The one-disease/one-outcome
training-pool factorial can be support-stratified if the full grid is too
costly, but then the paper must state precisely what it can generalize to.

## 3. Durable manifest and state machine

Use a server-side append-only run ledger plus immutable per-run manifests.
Each job has an **identity fingerprint** computed from at least:

- stage, role, variant, seed and expected output type;
- Git commit **plus dirty-diff archive/hash**, imported router source hash,
  generated YAML hash, package/environment lock hash;
- outcome release, family map, sealed membership/splits, sequence snapshot,
  vocabulary, pretrained checkpoint hash, source checkpoint hash;
- outcome roster/order, fine-disease roster/order, horizon list, training
  eligibility, loss aggregation, selection metric, optimizer budget;
- CPU/GPU resource request, walltime, retry policy and authorized split.

The run directory is deterministic from the fingerprint. A changed input
creates a **new** run identity; it never resumes into an old directory. The
plan may reuse a completed artifact only after verifying the full fingerprint
and its validation report, not because a familiar folder name exists.

State transitions: `PLANNED → PREFLIGHTED → READY → SUBMITTED → RUNNING →
CHECKPOINTED → VERIFYING → COMPLETE`; terminal alternatives are
`NEEDS_REVIEW`, `FAILED_TRANSIENT`, `FAILED_CONTRACT`, `BUDGET_STOP` and
`SKIPPED_UNSUPPORTED`. A missing scheduler process plus a valid `last.ckpt`
is `CHECKPOINTED`, **not** `COMPLETE`. `training_complete.json` alone is also
insufficient: require best checkpoint, expected metric inventory, strict
checkpoint load, matching hashes and clean output/subject keys. Stage-level
reducers run only when all required jobs are `COMPLETE` or explicitly
adjudicated `SKIPPED_UNSUPPORTED`.

On shared storage, use atomic per-run lock creation and record owner,
scheduler job ID and heartbeat. Never start two workers on one run. A stale
lock is not automatically stolen merely because a walltime expired: confirm
the scheduler job is gone before a single recorded reclaim. The present
`run_approved_neural_candidate_v1.sh` has preflight, `OPERA_RESUME` and
existing-run checks, but no cross-server atomic lock; another launcher has
one. The new queue must add that lock or wrap the approved launcher. Its
underlying training config already has `training.auto_resume: true`; the
queue must verify that resolved setting and the `last.ckpt` identity on every
resume. It must not use the historical malformed interactive checkpoint.

Each attempt gets an attempt number, scheduler ID, start/end timestamps,
exit code, GPU model/hours, max memory, log path and reason. Resume only
known transient interruption with the **same** fingerprint, max attempts and
total walltime. A code/config/data/hash mismatch, CUDA OOM, NaN, missing
outcome, corrupt checkpoint or repeated identical failure goes to review;
never blindly resubmit. Checkpointing should be frequent enough for the
cluster walltime, and a job that reaches its cap without convergence is
reported as truncated, not silently accepted as successful.

### Queue resource policy

- Put training/evaluation tasks on actual GPU resource requests (one local
  visible GPU per worker unless measured need says otherwise). A `qsub` job
  without an explicit site-approved GPU resource may be counted as a CPU job;
  the exact Moab/PBS syntax and partition/account must be copied from an
  approved successful GPU job, not guessed locally.
- Keep small CPU reductions in a separate low-concurrency lane. Do not flood
  the scheduler with one CPU job per specialist. Bundle small reductions and
  use bounded GPU job arrays only if the site supports them safely.
- Set `max_active_gpu_jobs`, `max_queued_jobs`, per-stage GPU-hour and
  currency caps, maximum attempts and a deadline. These are manifest fields
  reviewed by the operator. Do not infer that a low hourly unit price makes
  an unbounded matrix economical; estimate from the pilot and charge all
  seeds, failures and evaluation exports.
- Submit only when the preceding artifacts are verified; use explicit
  dependencies where the scheduler supports them and a controller ledger
  otherwise. Fair scheduling matters: avoid a thousand-cell grid starving
  smaller PHAIR compatibility or final-replication jobs.
- A single status command must print: stage, queued/running/complete/failed
  counts, GPUs in use, hours/cost so far, projected remainder, oldest
  heartbeat, missing prerequisites, next gate and **the next safe action**.

## 4. Statistical/clinical gates (must be answered, not automated away)

**Gate A: release and leakage.** Are patient IDs, treatment index, outcome
release, 69-outcome roster, splits and observation rules identical? Is the
fine-disease label available at index or retrospectively assigned? Keep the
coding-lag cohort but audit 14-/90-day later-diagnosis use and comparison with
index-available routing. Check pretraining cutoff/identity overlap with the
held-out period; do not call pretraining patient-disjoint if it is not.

**Gate B: what the router actually changed.** Compare additive versus
interacting at scale 0.1 on matched initialization, complete epochs and
fixed-checkpoint tuning predictions. Inspect routes, full-vector shifts,
source-removal CIF changes and calibration. If mode/config provenance fails,
stop; if predictions/routes scarcely differ, do not claim an interaction
mechanism even if selected validation losses differ slightly.

**Gate C: practical versus mechanistic gain.** Plain BCE is the practical
reference. Grouped shared BCE is the loss-matched readout control. Compare
family-balanced and equal-outcome metrics, Brier/calibration, and rare-task
lower tail. Show the source-bank factorial as *conditional on pooled training*.
Prespecify a minimal useful change or cost-benefit tolerance before using
the packet to select the architecture; do not keep searching until an AUC
improvement appears.

**Gate D: specialist feasibility.** Freeze minimum training events and
follow-up, optimization schedule, per-cell convergence criteria and handling
of unsupported cells before the pilot. Once runtime is observed, decide
whether the entire estimable grid fits the cap. If not, choose a transparent
stratified claim or higher budget; do not quietly omit difficult cells.

**Gate E: PHAIR compatibility and transfer.** A strict load and forward pass
must establish feature/time/numeric compatibility. If fine-tuning budgets
or architectures differ, the analysis becomes a system comparison rather
than an isolated pretraining effect. Report DAPT compute in addition to
supervised compute. PHAIR may improve both pooled and router arms, neither,
or interact with the value of borrowing.

**Gate F: final analysis lock.** Clinical endpoint definitions, training and
evaluation support, primary 365-day metric, patient bootstrap, hierarchical
rarity model, multiple-comparison presentation, subgroup/harm reporting,
selected checkpoints/seeds and tabular comparator are signed. Only then can
the held-out adapter be granted read access. Any post-test model change is a
new exploratory analysis, not a second chance at a primary result.

The current `evaluate_direct_competing_risk.py` CLI defaults to `held_out`.
Every automated tuning evaluation must therefore pass `--split tuning`
explicitly, and the orchestrator must independently reject a held-out path or
split until Gate F is signed. For the six-horizon report, pass all six horizons
explicitly; a 365-day screen alone is not the final evaluation. Resample
patients jointly across compared models/cells so uncertainty respects shared
patients, outcomes and disease groups. A temporal held-out set is not an
external cohort; report it as temporal validation unless a genuinely external
site/cohort is evaluated.

## 5. What can be done now while GPUs run

1. Audit the 26 patients absent from supervised survival/split; distinguish
   recorded death on/before treatment, administrative follow-up on/before
   treatment, missing patient record/date and cutoff effects. The figure
   must not assign all 26 a cause without source-level evidence.
2. Finish Table 1 from the exact supervised population using demographics,
   pre-index MEDS counts/history, region/specialty, disease-specific registry
   scores and follow-up. Keep internal exact data on server; get institutional
   sign-off on the combined suppression scheme and CONSORT.
3. Verify the user's 78,668 pretraining population against the frozen MEDS
   manifest (distinct persons versus sequence shards) and pre-2022 cutoff.
   Keep cohort attrition and pretraining sample definition conceptually clear.
4. User-owned separate repo: repair tabular feature matrices and train the
   EHR-only benchmark with the same 69 outcomes and temporal/eligibility
   contract. Registry-augmented tabular models are a separate informational
   comparator, not a fair EHR-only baseline.
5. Complete neural predictability atlas, readable outcome forest, borrowing
   maps, full-space movement diagnostics and captions from **fixed tuning
   exports**. Keep the original source-anchor schematic; the exploratory icon
   variant remains outside the manuscript until a better editable design is
   approved. Figure aesthetics never choose a checkpoint.
6. Draft the paper in claim order: scarcity and estimability; benefit of
   supervised pooling; incremental adaptive borrowing and patient specificity;
   broad pretraining and domain adaptation; clinical/tabular benchmark;
   calibration, harms, limitations and deployability. Separate descriptive
   tuning screens from locked held-out results in every caption and table.

## 6. Implementation sequence for the factory (no live submission yet)

**A. Inventory-only controller.** Read existing run directories and
manifests; emit a deterministic JSON ledger and human status table without
touching checkpoints. Verify completed markers, best/last checkpoints,
Hydra configs, code/roster sidecars and output uniqueness. This immediately
answers "what do we already have?" and prevents duplicate expensive runs.

**B. Dry-run DAG planner.** Machine-readable approved stage plan with role,
dependencies, run identity, resource ceilings and review gates. `plan` emits
exact proposed shell/scheduler commands and rejects unknown variants,
unresolved inputs, held-out steps, duplicate output directories and budget
overflow. Test it against synthetic complete, partial, stale-lock and
contract-mismatch trees before server sync.

**C. Bounded submit/resume adapter.** Only after A/B pass on the cluster:
explicit `--submit --stage STAGE_ID` and a human-reviewed site GPU job
template; atomic lock; one run per GPU; scheduler ID/heartbeat; fail-closed
resume; bounded retries. It wraps the existing preflight and approved
launcher rather than bypassing their release/metadata/vocabulary checks.
No hidden `OPERA_LAUNCH=1` default.

**D. Reducers and gate packets.** Aggregate-only scripts compute the
predeclared common-epoch and paired metrics and write immutable review
packets. A signed gate decision unlocks the next stage. `status` always
states the next safe command. An operator can stop after any gate without
leaving orphaned dependent submissions.

**E. Cluster dry run.** Start with a single small/reproducible GPU job and
one preempt-and-resume simulation in a new output namespace. Confirm GPU
resource accounting, no duplicate run, same fingerprints after resume,
checkpoint epoch/optimizer continuity, successful marker validation and
clean failure handling before using the system for the specialist grid.

This sequence avoids writing an untested `qsub` command or running an
unbounded queue on a restricted server. It also distinguishes a **workable
factory design** from a claim that the present launcher already implements
one: today it launches one candidate per invocation, and a human still
coordinates candidate lists, GPUs, resume and review gates.

## 7. Completeness audit and immediate next action

Cross-check each paper claim against its comparator, release, support rule,
selection metric, uncertainty, limitation and figure before final lock. The
current plan explicitly includes: old interacting-checkpoint invalidity;
loss-aggregation confounding; disease-label availability; no-source and
source-factorial controls; matched 0.1 interaction and capacity contrasts;
same-encoder specialists and untrainable-cell coverage; both supervised
pooling axes; replicated seeds; PHAIR compatibility/domain adaptation;
EHR-only tabular benchmark; censoring/calibration; Table 1/CONSORT; secure
data handling; budget/concurrency/retry caps; and a hard held-out gate.

**Next server action remains the seven-run aggregate inventory** in
`opera_paper_next_actions_20261007.md`. It is CPU-only and does not submit
training. After seeing its JSON and completion failures, choose the next
stage using the decision rule there. In parallel, the first factory coding
step should be the read-only inventory/controller, not a bulk submitter.

### Local verification performed for this design

The local project virtual environment passed the existing focused tests for
approved candidate generation, aggregate validation comparison, additive
versus interacting audit, and paired tuning-prediction comparison (16 tests).
This checks the pieces the proposed controller would initially wrap; it does
**not** exercise a multi-GPU scheduler, specialist launcher, PHAIR checkpoint,
live data, or the final held-out gate. Git Bash could not start in the local
Windows sandbox (MSYS process/pipe permission error), so shell parsing and
preempt-and-resume behavior must be verified on the secure Linux server.
No GPU job was submitted or resumed during this review.
