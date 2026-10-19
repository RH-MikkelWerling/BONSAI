# Stage 1 launch and router decisions (19 October 2026)

Encoder: 256-wide, 10 epochs (`epoch_010.ckpt`); width probe +0.003 < 0.01, so
256 kept. Every fit: budget 30 epochs, patience 5, seed 42, selection on the
full tuning set (endpoint-group horizon BCE).

## What runs where

| Stage | Phase | Fits | Waits for |
|---|---|---|---|
| 1 | Joint J | 1 | — |
| 1 | Disease-pooled D | 69 | — |
| 1 | Outcome-pooled O | 24 | — |
| 1 | Pretraining controls (random init RI, frozen encoder FZ) | 2 | — |
| 1 | Source screen: router with source bank none / disease / outcome / both (rank 8, interacting query 128) | 4 | — |
| 1 | Rank screen: chosen bank at rank 8, 16 and 32 | 2 new | source bank decided (rule 2) |
| 1 | Query screen: chosen bank and rank; additive 128, interacting 128, interacting 512 | ≤2 new | rank decided (rule 3) |
| 1 | Specialists S | ~1,650 | — (run last) |
| 2 | Router controls (JDE, RDH, ROH), final source banks (4) | 7 | router lock |
| 3 | National arms: E and EA × joint and locked router | 4 | lock + PHAIR encoder + DAPT |
| 3 | Data efficiency: cohort and adapted national joint at 10/25/50% | 6 | lock + EA |

Screens are deduplicated by configuration, so a rank-screen or query-screen fit
that equals an earlier one is not trained twice.

## Launch (server, from the repo root)

```bash
# .env
OPERA_PAPER_LOCAL_WIDTH=256
OPERA_PAPER_LOCAL_CKPT=data/models/daly_care_t100_joined_bins_with_sor_final_v2/pretrain_scaled_time2vec_hidden256/version_0/epoch_010.ckpt
OPERA_PAPER_MIN_LOCAL_EPOCHS=10
OPERA_PAPER_MIN_DAPT_EPOCHS=10
OPERA_PAPER_EPOCHS=30
OPERA_PAPER_PATIENCE=5
OPERA_PAPER_NUM_WORKERS=2
OPERA_PAPER_DECISIONS=<paper root>/router_decisions.json   # written below

sha256sum -c code_manifest.sha256 --quiet                  # synced code
grep -rh "dapt_ckpt" "$OPERA_PAPER_ROOT"/bundles/stage1/configs | sort -u   # must be epoch_010.ckpt
#   otherwise: new OPERA_PAPER_ROOT (bundles are immutable), copy tabular_runs*/ across
bash scripts/run_paper_experiments.sh prepare --execute
bash scripts/run_paper_experiments.sh smoke --execute
mkdir -p logs
nohup env PACK=2 bash scripts/paper_gpu_workers_packed.sh > logs/packed_$(hostname)_$(date +%H%M).log 2>&1 &   # each of 4 GPUs
bash scripts/run_paper_experiments.sh queue-status          # after ~1 h: throughput and GPU-hours left
```

## Router decisions (tuning data only; prespecified)

After each screen finishes:

```bash
bash scripts/run_paper_experiments.sh tuning-summary --phase source_screen --execute
```

1. **Capacity gate.** The router with no source bank (`none`) must lie within
   ±0.005 tuning AUC of the joint model. Otherwise a gain from routing could be
   extra capacity, and the source banks are not interpreted.
2. **Source bank** (`source_mode`). Disease-only routing is an acceptable final
   model. Outcome sources (`both` or `outcome_only`) are kept only with an
   incremental, supported gain over `disease_only` in paired tuning AUC and
   Brier. If inconclusive, choose the simpler `disease_only`.
3. **Rank** (8, 16, 32): prefer the smallest rank unless a larger one is clearly better. Rank 32 keeps the rank-to-width ratio of the 64-wide model (8/64 = 32/256).
4. **Query** (additive 128, interacting 128, interacting 512): prefer
   interacting 128 unless another is clearly better; 512 is a capacity check.

Record each decision as it is made (workers pick up the gated screens on their
next pass):

```bash
echo '{"source_mode": "disease_only"}' > "$OPERA_PAPER_DECISIONS"              # after rule 2
echo '{"source_mode": "disease_only", "rank": 8}' > "$OPERA_PAPER_DECISIONS"   # after rule 3
```

Router lock, after rule 4 (all four keys):

```bash
echo '{"source_mode": "disease_only", "rank": 8, "query_mode": "interacting", "query_width": 128}' \
  > "$OPERA_PAPER_ROOT/router_lock.json"
# .env: OPERA_PAPER_ROUTER_LOCK=<that file>
# Router controls and final source banks: start at once, no PHAIR needed
bash scripts/run_paper_experiments.sh prepare --stage 2 --execute
# National arms and data efficiency: when the E export and EA (DAPT) exist
bash scripts/run_paper_experiments.sh prepare --stage 3 --include-external --execute
```

Workers read every `bundles/stage*` when started; restart them after a
`prepare`. Fits already trained in an earlier bundle are recognised by their
configuration and not repeated.

Values above are examples; write the ones the screens support. Keep each
tuning summary with the decision it supported.

## Gates after stage 1 (in order)

1. **Router lock** (rules 1–4 above) → stage 2 (controls, final source banks).
2. **Tabular baselines complete:** sequence-matched XGBoost and logistic,
   pooled XGBoost (XP), registry-augmented XGBoost (XR).
3. **Tuning evaluation** of every fit: `evaluate --stage 2 --split tuning`.
4. **Analysis lock**, before any test data are opened: fill the open values in
   `docs/ANALYSIS_LOCK_DRAFT_20261015.md`, the nulls in the clinical-score
   config (endpoint names, provenance), and lock the attribution config. That
   needs the code tables and `concept_mapping_reviewed: true`, otherwise Figure
   8C and S30C stay empty. Record whether the test set opens in two steps (cohort
   arms first, national arms later, their protocol fixed in advance).
5. **Held-out evaluation:** `evaluate --stage 2 --split held_out`, inventory,
   analysis, Bayesian summaries, figures, clinical scores.
6. **Explainability, on the locked best model and the held-out set.** These are
   inference only, with no new training: integrated gradients for every test
   patient (`attributions`, then `attribution-summary` and
   `attribution-figures`; roughly tens of GPU-hours), and the input
   restrictions, which re-evaluate the same model with parts of the record
   withheld (`input-restrictions`; six restricted passes over the test set).
7. **National arms (stage 3):** memorization check → transfer approval →
   `export-external` (coverage ≥ 0.9) → `dapt` (10 epochs) →
   `prepare --stage 3 --include-external` → national test results.
