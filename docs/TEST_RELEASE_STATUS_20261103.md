# Test-release status after clinical KM implementation — 3 November 2026

Not yet a signed release. This is a current-status update to the earlier
TEST_LOCK_REVIEW, whose findings describe code before today's corrections.
No held-out data were opened in this review.

## Resolved locally

- Secondary weighting now uses disease → approved loss_endpoint → outcome
  variant → horizon. The equal-cell primary remains unchanged.
- Total-system, same-encoder and retained local-design contrasts are explicit;
  national selection no longer reverts the local counterpart to the old router.
- Architecture-aware export flags are used in actual national evaluation
  commands, not only in their recorded intent.
- Clinical comparisons are descriptive, outside Holm. Equal-size comparisons,
  training-frozen anchors and evaluation-cohort KM matching remain distinct.
- KM matching searches both extremes with paired target re-estimation,
  tie-preserving boundary selection, support rules and frontier plots. New
  outputs and paper-number exports are implemented and tested.
- Exact endpoint versus proxy disclosure is exported. This does not itself
  resolve every active server-side endpoint/population mapping.

## Remaining release work

1. **Executable Holm table.** The agreed eight contrasts are P1, P2, P5, T2,
   O1, router_borrow_AB, E1 and O2. The inherited national contrast is E1
   (J_E versus J_L), not a test-selected national comparison. Freeze primary
   AUC, equal-cell, all-horizon, seed-42 summary explicitly. The current
   evaluator still emits unadjusted p-values and the symbolic manifest still
   says pending_analysis_lock. Add a dedicated family table, preserve missing
   planned members rather than shrinking the family, and distinguish Holm
   decisions from descriptive CI labels. New matched-encoder reporting
   contrasts do not automatically expand this family.

2. **Evaluation-only final orchestration.** Do not use train_then_evaluate.sh
   unchanged. It can mark failed/timed-out evaluation shards done, continue
   analysis after failures and render partial figures. It also has local-only
   checkpoint lookup, split-only inventories and inherited/default estimand
   and horizon settings. Final orchestration must use the frozen fit roster,
   six horizons, cif_ipcw, explicit multiplier bootstrap, tuning + held-out
   inventory and held-out reporting. Require the declared outputs to succeed.
   These are analysis-completeness requirements, not file-freshness scans.

3. **Selected-encoder handoff.** scripts/set_test_env.py currently hardcodes
   local OPERA and its clinical substitution only replaces J_L. Ensure the
   selected alias, clinical specs, attribution checkpoint/config/vocabulary,
   and patient-map exports all refer to the same fit. Local mechanism/seed
   analyses retain their own clearly local identities.

4. **Actual server clinical roster.** Review the expanded endpoint audit and
   tuning score inventory from the active config, not just the example. Resolve
   exact registry score names/versions, native populations, categories, timing,
   endpoint proxies and remaining nulls. IPS/FLIPI2/MALT-IPI mappings remain
   pending in the template. The new KM plots need one actual tuning rehearsal,
   not only synthetic validation.

5. **National tuning choice and signature.** Section 4b still lacks national
   tuning effects and the resulting encoder choice. Follow its fixed rule
   before examining national test results. The lock permits a staged release,
   but final-OPERA claims require this choice and consistent handoff. Freeze
   exact bundle/fit roster/configs and record the dated signature/commit.

## Practical order

Finish items 1–3 locally; obtain item 4's aggregate server reports and national
training/tuning results as they complete. Rehearse the exact final orchestration
and active clinical config on tuning using existing fits/predictions. Reconcile
required contrast coverage and outputs, then sign and run held-out evaluation.
No additional architecture search or outcome rebuild is required by today's
findings. Figure cosmetics and discussion polishing can continue after locking,
provided they do not change scientific definitions or select findings on test.


## Follow-up: avoid another full rehearsal cycle

Holm is now implemented in opera.evaluation.headline_holm and the main analysis
exports headline_holm.csv. It can be added directly from saved effect summaries:

```bash
.venv-v100/bin/python -m opera.evaluation.headline_holm \
  --effect-summaries /path/to/prediction_analysis/effect_summaries.csv \
  --bootstrap-method multiplier \
  --output /path/to/prediction_analysis/headline_holm.csv
```

Use the bootstrap method recorded in the source analysis manifest. No fitting,
prediction scoring or resampling is needed. Thus item 1 above is implemented;
its targeted tests and the analysis integration are checked locally.

A full rerun of every tuning analysis is not a new requirement. Validate only
the changed pieces: actual clinical mapping/KM outputs and the final invocation
settings/failure handling. Reuse the completed rehearsal evidence elsewhere.
Final selected OPERA is used for clinical and descriptive checks. Same-encoder
controls answer the matched scientific questions; the selected local design
remains reported independently.

National availability can be staged under section 7. Freeze its tuning-only
selection rule now, release already-fixed local comparisons if desired, then
apply that same rule and analyse the national arm later. Final-OPERA claims
remain pending until its encoder is selected; local test findings must not
change the national training/selection rules. This is a scheduling choice,
not a requirement to wait for another complete rehearsal of all models.


## Launch and handoff hardening implemented

train_then_evaluate.sh now stops on worker/evaluation/action failures, checks
recorded canonical-model evaluation statuses before marking a shard done,
propagates failed shards, and exits on timeout. EVALUATE_ONLY=1 skips training.
It explicitly sets six horizons, cif_ipcw, multiplier bootstrap/1,000 draws,
and includes tuning + held-out inputs only for a held-out invocation. It resolves
the explicitly selected seed-42 alias on any encoder for checkpoint/config
handoff. The selected alias also retargets clinical model keys. The completion
marker means the launched actions succeeded; figure-readiness still records
optional missing panels rather than certifying that every possible analysis ran.

set_test_env.py now accepts --model and --split (default tuning), resolves the
selected encoder rather than hardcoding L, and can retarget a clinical config
already using J_OA_OWN_L or a national alias. No source-freshness checks added.
National vocabulary/input paths still require the actual national fitted config
and exports; they cannot be certified from unavailable server artifacts.

This is incremental hardening of the existing launcher, not a new experiment.
The full explanatory/clinical rehearsal remains rerun_full_tuning_rehearsal.py
with a completed source workspace; the launcher alone does not execute every
explanatory analysis. Use the full rehearsal if it is the simplest server job,
but treat it as a single acceptance run of these changes, not another search.
