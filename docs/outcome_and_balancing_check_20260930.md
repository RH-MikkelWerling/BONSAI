# Outcome and minority-event checks (30 September 2026)

The `weightinggroups` model already uses direct six-horizon BCE and
endpoint-group loss aggregation. The proposed `minorityevent` task is a
**blocked design prototype, not launch-ready**: the competing-risk trainer
rejects event-enriched sampling because it changes the fitted absolute risks
without inverse-probability correction. Do not run the command below until a
valid sampling correction or a natural-distribution loss-weighting alternative
is implemented and tested. The draft sampler focuses a 365-day outcome
only when at least 50 distinct training patients have its primary event by
day 365 and at least 50 distinct eligible patients are event-free through day
365. It uses inverse-square-root event-count probabilities, four event slots
per logical batch and at most two appearances per patient per epoch. These are
sampling safeguards, not an assertion that 50 events ensures a reliable AUC.
Run the support audit first and keep validation sampling untouched.

Reserved launch pattern after that statistical gate is cleared:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
python -m opera.run.preflight_direct_competing_risk \
  --config-name generated/direct_cr_bce_lowrank_minority_event_v1 \
  --require-existing-paths
RUN_ROOT="$PWD/data/models/opera_minority_event_v1"
nohup bash scripts/run_opera_independent_experiment.sh minorityevent "$RUN_ROOT" \
  > "$RUN_ROOT.launch.log" 2>&1 < /dev/null &
echo "PID: $!"
```

The outcome-source review has not fully cleared the clinical definitions.
Before regenerating final-data labels, run this aggregate-only audit on both
the current and final outcome directories:

```bash
python -m opera.run.audit_outcome_exports \
  --outcomes-dir "$BONSAI_OUTCOMES_DIR" \
  --output "$PWD/data/results/outcome_export_audit_v4.csv"
```

The audit flags duplicate/missing IDs, invalid event/censor/index ordering and
split-date disagreements. It deliberately emits no subject IDs. The coverage
gate is now audit-idempotent; a rerun with no pre-coverage patients leaves its
earlier exclusion audit intact. Remaining substantive checks need the R source
environment and aggregate server counts: the confirmed J01EE01/JO1EE01 typo,
fibrinogen assay/units, infection and co-occurrence observation eligibility,
the two administrative cutoffs, and the duplicated composite execution block.
Those items can alter labels or eligibility and should be resolved before the
final MEDS release becomes the analysis ground truth. See
`docs/ascertainment_review.md` for the evidence and outstanding decisions.
