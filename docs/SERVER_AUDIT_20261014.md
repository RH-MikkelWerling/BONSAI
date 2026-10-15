# Server handoff audit — 14 October 2026

> **Superseded for launch steps by `RUNBOOK_20261015.md`** (single seed 42, local `best.ckpt` from the 20-epoch run, 20-epoch DAPT, new metadata/plot-table steps).


The code was audited locally; no server jobs were launched. A successful local
suite cannot verify the cluster's CUDA environment, shared filesystem or private
data paths. Those require the allocated-GPU checks below.

## Corrections made

- Removed automatic stale-claim takeover. An expired heartbeat does not prove
  a remote training process is dead; duplicate training could overwrite its
  checkpoints. Stale/unreadable claims remain reserved until confirmed stopped.
- Added ownership tokens: another worker cannot release a claim it does not own.
  Heartbeats stop if the ownership token changes. A completion check after
  acquisition closes the scan-to-claim race for a newly finished fit.
- Fixed and shared neural queues now use the same exclusive claim mechanism.
  Attempt-reporting failures still release the claim through a nested finally.
- Newly generated factorial configurations request one device and one node.
  Each GPU allocation runs one worker, against the same bundles and run root.
- Aligned the example/fallback seed roster with the planned 42/43/44 fits.
  An existing explicit `OPERA_PAPER_SEEDS=42` still overrides the fallback;
  choose the final roster deliberately before bundle preparation.
- Corrected an outdated Bayesian recipe test that still expected an
  exchangeable primary after the agreed support-spline change.

Claiming uses exclusive creation on shared storage. Cross-process stress tests
verify one winner locally; multi-node filesystem semantics still need checking
on the actual shared mount. Do not delete claims while their allocations run.
After a killed allocation, confirm the scheduler and child processes are stopped,
archive that run's `.worker_claim.json`, then restart a worker. Resume uses the
run's own `last.ckpt`. This trades automatic timeout recovery for preventing
concurrent writes to a possibly live fit.

## Before tomorrow's training

1. Synchronise all changed **and new** code/configuration files. Preserve the
   server `.env`, datasets, existing checkpoints and run outputs. The local `.env`
   has unresolved server paths and must not replace the server environment.
2. In the server environment, set the selected release, sequence, membership,
   outcome, base/reference config and local checkpoint paths. Retain the 20-epoch
   completion gate. Review seeds, batches, effective batch, optimizer and data
   contract before creating an immutable current-256 bundle.
3. Run `scripts/check_paper_server.sh --gpu` inside an allocation. The import
   check catches runtime-library errors early; CUDA must expose one assigned GPU.
   Prepare the stage-1 bundle once with `scripts/run_paper_experiments.sh prepare
   --execute`. Multiple workers consume it and never prepare competing bundles.
4. After the local encoder completion gate is satisfied, run
   `scripts/run_paper_experiments.sh smoke --execute` on one allocated GPU.
   This requires the prepared bundle. Inspect completion before starting the pool.
5. In every independently allocated GPU terminal run
   `scripts/run_paper_experiments.sh worker --wait --execute`. Scheduler GPU masks
   take precedence over legacy dotenv masks. Use the same shared code, bundles,
   decisions and run roots. Inspect `scripts/run_paper_experiments.sh queue-status`.
6. Record tuning decisions and prepare stage 2 once when appropriate. Workers
   re-read the decision file and stage-2 bundle; a restart is unnecessary.

The worker blocks until encoder completion/provenance checks pass. Old 64-wide
fits cannot satisfy the current local encoder gate. Decisions and saved selection
records determine downstream analysis; running a training process to exit alone
is insufficient evidence of a valid completed fit.

## Analysis and manuscript readiness

Training queue readiness does not imply that every downstream analysis adapter
is complete. Full real figure recipes, dedicated input workspaces, score plotting,
age panels and supplementary table rendering exist. Raw-output plotting
adapters, the five-stage embedding export/projection orchestration and the
intended-endpoint clinical-score evaluator remain incomplete. Do not promise an
unattended full results manuscript from finished fits yet. Initialise figure
inputs with `init-figures --execute`, populate reviewed sources and use
`figures --validate-render --execute` before rendering real plates.

The manuscript contains honest placeholders and synthetic design proofs. The
current cohort counts still require supervisor sign-off. Source definitions and
model/evaluation supplements are present; numeric templates are not empirical
results. Both manuscript builds are checked for unresolved citations and overflow.

## Local verification and environment limits

Focused tests cover bundles, factorial scopes, queue gates/claims, resumed
checkpoint selection, tabular fitting/tuning, clinical-score timing, real-figure
and table rendering, Bayesian identities/rosters/prior checks, rotary-layout and
checkpoint transfer, secondary analyses and representation extraction.
An eight-process claim test checks mutual exclusion beyond threads.

Local XGBoost initially could not locate OpenMP; exposing the runtime bundled
with the local PyTorch installation allowed the real baseline test to pass.
Local PyTensor initially could not write its compilation cache; a task-local
cache fixes that environment restriction. Neither workaround is hard-coded into
server code. Linux imports must pass in the actual environment. Flash-attention
GPU-only comparisons cannot be verified on this CPU workstation. The changed R
outcome source was parsed successfully; secure-cohort regeneration was not run.


Verification records: the main operational batch passed 108 checks and exposed
one outdated expected primary-recipe name; after updating both AUC/Brier recipe
expectations, the focused recipe/bundle/factorial batch passed all nine checks.
The eight-process claiming regression passed. The separate reconstruction,
transfer, rotary-layout, secondary-analysis and extraction suite passed its
CPU checks with two GPU-only skips. A real Bayesian sampling/artifact smoke
plus the two dropout-gradient checks passed (three checks). No full unrelated
repository test suite was run.

Final dispatcher suite: 18 checks passed. Figure-function suite: four checks passed.

Follow-up coverage audit: the worker now explicitly rejects missing bundles,
and the older shell-worker recovery comment is corrected. The dedicated
15 October plan cross-references every experiment/analysis component, makes
remaining implementations explicit and separates pretraining-independent work
from checkpoint-dependent GPU launches.
