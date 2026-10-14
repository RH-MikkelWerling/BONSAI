# Tuning-only router gain mechanism audit

This audit starts with paired outcome-level discrimination, **not** router
weights. It deliberately includes positive, near-zero, and negative contrasts.
All selection and diagnostic work stays on tuning data. The selected checkpoints
and their tuning comparisons are exploratory; held-out evaluation remains closed.

## Stage 1: rank paired gains and select probes

Use the same approved plain Time2Vec baseline and clean interacting scale-0.3
checkpoint as in the model comparison. Locate their completed tuning prediction
Parquets; do not substitute a validation-only metrics CSV or an older checkpoint.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
find data/results -name direct_cr_predictions.parquet -print
export OPERA_BASELINE_TUNING_PREDICTIONS="$PWD/data/results/REPLACE_BASELINE/direct_cr_predictions.parquet"
export OPERA_SCALE03_TUNING_PREDICTIONS="$PWD/data/results/REPLACE_SCALE03/direct_cr_predictions.parquet"
bash scripts/audit_scale03_router_gain_mechanism.sh
```

The program requires identical patient/outcome/cohort/follow-up/event rows
between models, then uses paired patient bootstraps for censoring-aware CIF-AUC
at 365 days. Unsupported outcomes are inventoried rather than assigned an AUC.
Review `all_outcome_gains.csv` and `selected_mechanism_outcomes.csv` first.
Selection is by largest positive gain, closest to zero, and largest loss;
overlapping choices appear once.

If a complete scale-0.3 route export already exists, include it before running:

```bash
export OPERA_SCALE03_ROUTE_EXPORTS="$PWD/data/results/REPLACE_SCALE03_ROUTES/source_route_weights.parquet"
export OPERA_GAIN_AUDIT_DIR="$PWD/data/results/router_scale03_gain_mechanism_with_routes_20261010_INTERNAL"
bash scripts/audit_scale03_router_gain_mechanism.sh
```

The route CSV reports within-cell patient gate variation as total variation
(half the L1 distance from the cell-average route). It does not claim that gate
variation improves predictions. Partial route exports are allowed; absent
selected outcomes are listed as missing in `audit_manifest_INTERNAL.json`.
Do not infer that a missing outcome has flat routing.

## Stage 2: checkpoint interventions (next GPU pass)

For the selected cells, use the same patient/target rows and checkpoint for
normal predictions and interventions. Compare (a) unchanged gates,
(b) patient-shuffled or within-cell-mean gates, and (c) named source removal.
The existing evaluator supports (c) with `--remove-source`, but it renormalizes
remaining gates; its contrast is *removal plus redistribution*, not a source's
causal contribution. A clean fixed-gate control requires retraining with
matched encoder, adapters, optimizer, data, epochs and seeds. Do not interpret
the existing removal option as that trained control.

Prioritize whether predicted CIF and outcome-level AUC/Brier change, not only
whether route vectors change. Predefine the intervention list using Stage 1
outcomes, and do not select the largest-looking source after seeing effects
without labeling the analysis exploratory.

## Stage 3: one controlled residual-scale pilot

Only after Stage 1–2, compare a fresh scale-0.7 run with a replicated 0.3 run
under identical configuration, seed, schedule and selection rule. Scale changes
the adapter residual magnitude, not gate temperature/selectivity. Predeclare
overall endpoint-group AUC, the selected outcome contrasts, calibration/Brier,
and negative-transfer count as the review gate before any wider sweep.
