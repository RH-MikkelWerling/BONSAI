# Supervisor meeting: preliminary OPERA findings (7 October 2026)

This is a discussion brief, not a results section. All performance figures
below come from user-shared **single-seed tuning** summaries on the approved
panel; no held-out result or patient-level paired estimate has been reviewed
here. Four Time2Vec architecture runs were interrupted by a remote connection
abort and have resumed from `last.ckpt`; their final ranking is pending.

## What is established now

- The sealed release has **69 approved outcomes**, **52 endpoint groups** and
  a proposed **12-family** training taxonomy. This gives a consistent panel
  for matched neural experiments rather than a mixture of old endpoints.
- Both Fourier and scaled Time2Vec pretraining/checkpoints run on the server.
  In the earlier factorized-readout comparison, Fourier's best tuning
  endpoint-group-macro AUC was **0.744856** versus **0.741486** for Time2Vec
  (difference **+0.003370**). The difference was small and single-seed.
- In the later *plain shared-head BCE* comparison, the user-shared comparator
  reported Time2Vec minus Fourier **+0.011152** at each arm's best epoch,
  **+0.013602** at the last epoch, and **+0.011936** averaged over the final
  five epochs. The opposite ranking across setups means encoder choice cannot
  be declared independently of readout/loss. It also removes the concern that
  Time2Vec is broken or unsuitable for the planned PHAIR representation.
- Within the Fourier endpoint-group-BCE architecture screen, the
  user-shared best-epoch contrasts were **+0.004571** for family trunks versus
  grouped shared readout and **+0.016215** for the additive router versus
  grouped shared readout. These are descriptive, selected-epoch contrasts;
  they are not paired uncertainty estimates or proof that routing caused the
  gain. The interacting router and matched Time2Vec versions still need their
  resumed runs inventoried before a ranking is frozen.

The safe one-sentence preliminary finding is: **joint neural prediction on
the approved panel is feasible, and structured readouts—especially routing—
are promising on tuning data; the preferred encoder and architecture remain
conditional on a matched evaluation.**

## What the new mechanistic experiment asks

The 2×2 source-type ablation compares the same pooled, interacting router
with neither adapter bank, disease adapters only, outcome adapters only, or
both. It asks whether each *kind of learned source borrowing* helps and
whether they complement one another. It does **not** remove disease identity
or target identity from the query, and does **not** isolate the total effect
of pooling data across diseases/outcomes. The revised manuscript diagram
shows this as four cells and two ``add source'' contrasts. No ablation result
exists yet. The four-arm run should follow the resumed architecture inventory
and be replicated for promising contrasts.

## What cannot yet be claimed

- No final best OPERA architecture: the resumed Time2Vec shared, family-trunk,
  additive-router and interacting-router runs have not been compared here.
- No superiority over pooled or disease-specific XGBoost on this **same
  approved panel**: the tabular feature matrices and eligibility alignment
  still need repair. Older tabular comparisons answer a different question.
- No held-out generalization, calibration advantage, significant effect,
  or causal attribution to disease/outcome sharing. Best-of-run tuning AUC
  is selection-biased; family improvements may trade off against one another.

## Highest-value work before the meeting

1. When all four resumed runs finish, execute the existing aggregate-only
   `opera.run.compare_approved_neural_validation_v1` on the common run root.
   Verify identical selection inventories, checkpoint rule, epochs, and
   resolved source checkpoints. Show best, **common completed epoch**, last,
   final-five-epoch and family contrasts, not just a leader-board score.

   ```bash
   cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
   source setup_v100.sh
   RUN_ROOT="$PWD/data/models/approved_neural_v1_20261006T054836Z"
   REPORT="$PWD/data/results/approved_neural_validation_supervisor_20261007.json"
   python -m opera.run.compare_approved_neural_validation_v1 \
     --run "fourier_plain_shared_bce=$RUN_ROOT/fourier_plain_shared_bce" \
     --run "time2vec_plain_shared_bce=$RUN_ROOT/time2vec_plain_shared_bce" \
     --run "shared=$RUN_ROOT/shared" \
     --run "family_trunks=$RUN_ROOT/family_trunks" \
     --run "router_additive_e2e=$RUN_ROOT/router_additive_e2e" \
     --run "time2vec_grouped_shared_bce=$RUN_ROOT/time2vec_grouped_shared_bce" \
     --run "time2vec_family_trunks=$RUN_ROOT/time2vec_family_trunks" \
     --run "time2vec_router_additive_e2e=$RUN_ROOT/time2vec_router_additive_e2e" \
     --run "time2vec_router_interacting_e2e=$RUN_ROOT/time2vec_router_interacting_e2e" \
     --output "$REPORT"
   ```

   The comparator refuses an existing report path; choose a new name for a
   rerun. Run only after each selected directory has `training_complete.json`.
2. Prepare one compact tuning plot/table: model × encoder × readout, endpoint-
   group macro AUC trajectory, best epoch, final-five-epoch mean, and any
   noteworthy family trade-offs. Mark every cell as single-seed tuning.
3. Repair tabular matrices against the 69-outcome roster and the same sealed
   membership/eligibility. First audit patient IDs, index dates, split labels,
   observation windows, leakage, feature missingness and task support. Then
   fit a pooled XGBoost control with identical outcome/horizon definitions;
   postpone grand neural-versus-tabular claims until its validation output is
   aligned and available.
4. If compute permits, launch the three missing source-type arms against the
   existing full interacting-router arm, then replicate the most informative
   contrasts. Keep held-out data sealed until architecture/model selection.

The meeting can use the [router source-type diagram](../paper/figures/router_source_factorial.tex)
as a hypothesis figure and the [approved encoder inventory](approved_abspos_v2_tuning_inventory_20261006.md)
as the audited starting point. Avoid copying preliminary numbers into the
abstract or conclusion until matched evaluation and uncertainty are complete.
