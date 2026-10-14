# OPERA manuscript result lock — 7 October 2026

The main question is whether scarce disease–outcome labels can be supplemented
by (1) broader self-supervised encoder pretraining and (2) patient-dependent
borrowing from other labelled disease and outcome tasks. These are distinct
transfer axes and should be compared under matched downstream protocols.

## Primary supervised comparison

All five arms train on the same pooled approved panel and use the same
Time2Vec initialization, endpoints, loss, split, and checkpoint rule:

1. Plain shared multi-horizon BCE (reference).
2. Disease-indexed adapter sources only.
3. Outcome-indexed adapter sources only.
4. Both source banks, additive query.
5. Both source banks, interactive query.

The source-only arms do **not** test training on one disease or one outcome.
The no-source router is an additional architecture-matched control. A clean
additive-versus-interactive contrast needs the same residual scale; comparing
interactive scale 0.3 with additive scale 0.1 would confound query mode and
capacity. The 0.1/0.2/0.3 sweep is a separate sensitivity analysis. The old
`approved_neural_v1_20261006T054836Z/time2vec_router_interacting_e2e`
checkpoint is excluded from this comparison because its interaction branch
was absent in the server audit.

## First aggregate export from the secure server

Run from the repository root after checking all seven runs have completed.
This is CPU-only; it reads training metrics and checkpoint metadata, does not
generate patient-level predictions, and leaves existing outputs untouched.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
ROOT="$PWD/data/models/approved_neural_router_clean_20261007_v1"
OUT="$PWD/data/results/router_clean_validation_20261007_v1.json"
for RUN in time2vec_router_additive_e2e time2vec_router_interacting_e2e time2vec_router_interacting_scale02_e2e time2vec_router_interacting_scale03_e2e time2vec_router_disease_only_e2e time2vec_router_outcome_only_e2e time2vec_router_no_sources_e2e; do
  test -s "$ROOT/$RUN/last.ckpt" || { echo "Missing checkpoint: $RUN" >&2; exit 1; }
done
test ! -e "$OUT" || { echo "Output already exists: $OUT" >&2; exit 1; }
python -m opera.run.compare_approved_neural_validation_v1 \
  --run "additive_01=$ROOT/time2vec_router_additive_e2e" \
  --run "interactive_01=$ROOT/time2vec_router_interacting_e2e" \
  --run "interactive_02=$ROOT/time2vec_router_interacting_scale02_e2e" \
  --run "interactive_03=$ROOT/time2vec_router_interacting_scale03_e2e" \
  --run "disease_sources=$ROOT/time2vec_router_disease_only_e2e" \
  --run "outcome_sources=$ROOT/time2vec_router_outcome_only_e2e" \
  --run "no_sources=$ROOT/time2vec_router_no_sources_e2e" \
  --output "$OUT"
```

If this reports a contract mismatch, inspect the mismatch and configs before
aggregating; do not force unlike runs into one comparison. Share the JSON
aggregate and the run-completion/provenance summary, not raw patient records.
The manuscript needs, for each arm, common-epoch and selected-checkpoint
endpoint-group macro AUC, validation loss, best epoch, and the exact residual
scale and query mode. These are descriptive tuning data, not final evidence.

Next, export fixed-checkpoint paired tuning predictions for all approved
outcomes and six horizons, then compute patient-paired AUC, Brier and
calibration contrasts with event support and uncertainty. Check the clean
additive/interactive 0.1 pair with `opera.run.audit_router_query_pair` before
claiming an interaction effect. Replicated seeds and a frozen selection rule
precede one held-out evaluation. Keep held-out labels closed until then.

## Figures and supplementary analyses

- Main model comparison: five-arm pooled outcome-macro and family-balanced
  discrimination, with paired contrasts and uncertainty; distinguish tuning
  screen from locked held-out result.
- Source-type factorial: no sources, disease only, outcome only, both under the
  same interactive query; report source removal and patient-dependent weights.
  Borrowing matrices are directional (target rows, source columns), not
  symmetric similarity matrices. Show disease and outcome banks on separate
  scales, and add target-conditioned slices rather than interpreting a global
  mean as universal borrowing.
- Neural predictability atlas: selected frozen model, all 69 approved outcomes
  and six horizons, event support, uncertainty and unsupported cells visible.
- Patient representation panels: fixed shared embedding projection plus
  outcome-specific routed displacement; the arrow is a projected change, not
  the full-space effect. Include full-space residual size and outcome-specific
  prediction changes when interpreting arrows.
- Exploratory architecture screen, specialist fine-tuning and scale sweep move
  to supplementary analyses unless they become essential to a specific claim.

## Joint-training counterfactual

To claim a benefit across the whole estimable disease–outcome landscape,
compare the joint predictor with one specialist per supported fine-disease ×
outcome cell, using the same pretrained encoder and held-out patients. Each
specialist predicts all six horizons, so the horizon axis does not multiply
the number of training runs. Freeze an event-support threshold and report
unsupported cells; a convenience sample of tasks supports only a sampled-task
claim. A second cell-specialist arm initialized from the pooled supervised
checkpoint tests whether joint learning helps mainly as initialization.
Coarser disease-only and outcome-only *training* regimes can be run as an
intermediate factorial, but are not substitutes for the cell-specialist
counterfactual and must not be confused with router source-bank masks.

## Table 1, cohort flow and EHR2MEDS

Current verified denominators are 37,557 sealed first-line patients and
37,531 with released survival/supervised split (29,105 training, 1,980 tuning,
6,446 held-out; 26 without outcome split). The checked-in Table 1 remains a
historical layout and must not be copied into the Results as current data.

1. Re-export sex and birth date from the patient table with the verified
   `patientid`–`subject_id` mapping; check uniqueness and unmatched IDs before
   joining. Derive age at the common index date. Death date is not a baseline
   predictor.
2. Re-export pre-index EHR2MEDS history summaries, excluding demographic and
   administrative DOB/DOD/SEX/BACKGROUND tokens. Audit actual code namespaces
   against `opera/configs/table1_meds_prefixes.json` before calculating unique
   medication, diagnosis and laboratory concept counts.
3. For contact context, define the modal pre-index region with a fixed tie
   rule and count distinct pre-index specialties. Record missingness and avoid
   post-index contacts.
4. Join risk category and performance status only from their disease quality
   registries. Show their complete-case denominator and missingness by cohort;
   do not let their low availability silently shrink the main Table 1 cohort.
5. Reconcile upstream cohort-flow attrition with the current first-line
   release, regenerate the suppressed disclosure-review Table 1, and only then
   update `paper/tables/table1.tex` and `paper/figures/cohort_flow.tex`.

The feature-matrix and tabular-comparator work is a separate repo/user stream.
It can proceed while neural aggregate export and PHAIR compatibility work run.
PHAIR is not equivalent to the local Time2Vec domain-adapted checkpoint; its
vocabulary, numerical/time representation and checkpoint loading need an
explicit audit before matched transfer experiments.
