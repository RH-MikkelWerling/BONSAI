# Additive versus interacting router: implementation audit and server check

This is a tuning-only diagnostic, not a held-out model comparison. Do not
claim an interaction benefit from rounded epoch AUCs or selected best epochs.

## Code audit (completed locally)

- Both approved variants are generated from the same router template; the
  candidate config changes `competing_risk.adapter_query_mode` from `additive`
  to `interacting`. Verify the **resolved run configs** with the script below;
  candidate config equality alone does not prove the executed runs matched.
- The additive source logit is a dot product of each source key with
  `patient_query + fine_disease_query + target_outcome_query`. Before the
  home-source floor, the target-outcome contrast in log odds between two
  sources is independent of patient and home disease. This is a real
  expressivity restriction, not merely a plotting artifact.
  Softmax normalization and the forced home-source floor can still make raw
  weights look non-additive; inspect source-pair log odds after accounting for
  the home floor when testing the architectural distinction.
- Interactive mode adds an MLP of the concatenated patient, disease, and
  outcome queries before source-key scoring. The same patient can therefore
  change source preference differently by target and home disease. The source
  adapters and the risk head are otherwise the same structures.
- The interaction parameters are included in the non-encoder AdamW group, and
  the direct-risk loss receives the routed hazards. Local tests verify nonzero
  interaction gradients, additive separability, interactive dependence,
  source masking, and checkpoint round-trips. These prove the branch can train,
  **not** that the completed server run used it meaningfully.
- The evaluation loader reconstructs the model from saved checkpoint init
  config and loads its state **strictly**. If the interaction tensors are in
  the checkpoint and the mode is saved as `interacting`, they cannot silently
  be ignored by a successful strict load.
- **Historical initialization confound and invalid interactive checkpoint:**
  the October 6 run labelled interacting saved no `query_interaction.*`
  tensors in either best or last checkpoint, although its metadata says
  `interacting`. It is not evidence for an interacting model. The older
  initializer also gave the two arms different query embeddings and RNG
  trajectories. The current source now constructs the interaction MLP under
  an RNG fork, leaves fine/target queries matched, and zero-initializes its
  final layer. Local tests verify identical shared tensors, next RNG draw,
  initial routes/representations, and later nonzero interaction gradients.
  These changes require **fresh additive and interacting runs**; do not
  reinterpret or overwrite the historical checkpoints. Replicate seeds for
  a confirmatory comparison.
- Both modes use a shared logit initialized to 3, low-rank adapters with small
  initial `up` weights, and a residual scale. It is plausible for the
  interaction branch to change routes yet barely move hazards initially.
  Only trained checkpoint and paired prediction diagnostics can establish
  whether that happened here.

## Read-only checkpoint and log audit (CPU server)

After syncing the repository and sourcing `setup_v100.sh`:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
RUN_ROOT="$PWD/data/models/approved_neural_v1_20261006T054836Z"
python -m opera.run.audit_router_query_pair \
  --additive-run "$RUN_ROOT/time2vec_router_additive_e2e" \
  --interacting-run "$RUN_ROOT/time2vec_router_interacting_e2e" \
  --output-dir "$PWD/data/results/router_query_audit_checkpoints_20261007_v1"
```

Inspect `router_query_audit_INTERNAL.json`, `common_epoch_metrics.csv`, and
`common_router_tensors.csv`. The script checks executed configs, best-checkpoint
mode and interaction tensors, checkpoint hash and epoch, common validation
epochs (including resumed logger versions), last-versus-best checkpoint epochs,
training-completion markers, and whether corresponding learned router tensors
are exactly or approximately equal. It raises on missing or
mislabelled query branches.

The JSON's `current_router_code` records the exact module file imported by the
server process, its SHA-256, the repository HEAD, whether that file is dirty,
and whether the loaded class contains the interacting branch. This identifies
the **current server implementation**. The resolved run configs and saved
checkpoint metadata identify which mode each run used. The audit also requires
`competing_risk.fine_column` to be `cohort_fine` and the checkpoint's saved
fine-disease roster to match the resolved config. The grouped cohort labels
organize data loading; `cohort_fine_index` is the tensor passed to the router.
Unless training-time
source hashes were archived separately, the current file hash cannot prove
that an older checkpoint was trained under precisely today's uncommitted code.

## Paired tuning prediction and route audit (one evaluation per checkpoint)

If existing evaluation directories already contain both
`direct_cr_predictions.parquet` and `source_route_weights.parquet`, reuse them.
Otherwise run the following on a free GPU (sequentially or on separate GPU
servers) with the same current code and environment:

```bash
RELEASE="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
export BONSAI_COHORT_MEMBERSHIP="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1/population_metadata.parquet"
export BONSAI_OUTCOMES_DIR="$RELEASE/outcomes"
EVAL_ROOT="$PWD/data/results/router_query_paired_tuning_20261007_v1"
for RUN in time2vec_router_additive_e2e time2vec_router_interacting_e2e; do
  CUDA_VISIBLE_DEVICES=0 python -m opera.run.evaluate_direct_competing_risk \
    --checkpoint "$RUN_ROOT/$RUN/best.ckpt" \
    --config-dir "$RUN_ROOT/$RUN/hydra" --config-name config \
    --split tuning --horizons 365 --device cuda --export-routes \
    --output-dir "$EVAL_ROOT/$RUN"
done
python -m opera.run.audit_router_query_pair \
  --additive-run "$RUN_ROOT/time2vec_router_additive_e2e" \
  --interacting-run "$RUN_ROOT/time2vec_router_interacting_e2e" \
  --additive-eval "$EVAL_ROOT/time2vec_router_additive_e2e" \
  --interacting-eval "$EVAL_ROOT/time2vec_router_interacting_e2e" \
  --output-dir "$PWD/data/results/router_query_audit_paired_20261007_v1"
```

The paired audit refuses mismatched patient/target keys, follow-up labels,
home-disease IDs, and non-tuning data. Its CSVs report full-precision CIF
differences and source-gate differences, not just rounded AUC differences.
`target_interaction_probe.csv` additionally checks the additive model's
constant-across-patients target contrast in log odds between two **non-home**
disease sources, using matched overall-survival and sepsis route rows. The
same pair is tested in the interacting model. A nonzero interacting spread
demonstrates learned conditional routing for that pair; it does not imply an
AUC gain.
Neither patient IDs nor raw route rows are copied into the audit output.

Interpret in this order: config and checkpoint identity; paired route
differences; paired CIF differences; common-epoch AUC/loss differences. If
routes differ but CIFs barely move, inspect the trained residual and head
scales; the interaction may be real but practically inactive. If routes and
CIFs are exactly identical, inspect checkpoint paths/loading. If they differ
but AUCs tie, the rounded metric concealed a real yet currently unhelpful
change. A positive effect still needs replicated seeds and an initialization-
matched run before thesis-level claims.

## Better router figures

`plot_router_borrowing.py` now emits `router_borrowing_anchor_targets.png`
for five prespecified clinical targets. Panel A shows the mutually exclusive
route budget; B and C show target-specific deviations from each source's
overall popularity, for outcome and disease sources respectively. This removes
the bright vertical stripes caused by globally popular sources. It retains
the raw mass overview and full CSV matrices. The red/blue specificity display
is descriptive gate preference, **not** an estimated prediction contribution.
