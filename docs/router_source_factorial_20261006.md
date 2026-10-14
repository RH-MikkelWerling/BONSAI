# Router source-type factorial ablation

The primary question is whether disease-indexed and outcome-indexed adapter
*sources* each add value, and whether their contributions interact. This is a
2×2 experiment conditional on pooled training and a shared Time2Vec encoder;
it is not a test of training on separate diseases or outcomes.

| Arm | Disease sources | Outcome sources | Candidate variant |
| --- | --- | --- | --- |
| A | off | off | `time2vec_router_no_sources_e2e` |
| B | on | off | `time2vec_router_disease_only_e2e` |
| C | off | on | `time2vec_router_outcome_only_e2e` |
| D | on | on | `time2vec_router_interacting_e2e` |

All arms retain the interacting query's patient, fine-disease and target-outcome
context. The source-mode gate masks disabled banks *before* softmax and
disables their home updates, while leaving the router parameter inventory
unchanged. Arm A is therefore a shared-path control with an inactive router;
it is not parameter-count matched in **active** parameters. The existing
`time2vec_grouped_shared_bce` model remains a useful conventional shared-head
control but is not identical to Arm A's initialization and model inventory.

For result labels, use **disease sources only**, **outcome sources only**, and
**neither source bank**. “Disease-only” does not remove outcome conditioning
from the query or multi-outcome supervision. The separate question of query
context is documented in [the end-to-end audit](router_end_to_end_audit_20261013.md).

Use the same 69-outcome release, 52 endpoint groups, 12 training families,
sealed metadata, sequence snapshot, Time2Vec DAPT checkpoint, BCE objective,
checkpoint metric, seed, optimizer/schedule and epoch budget in every arm.
The generated variants differ only in `competing_risk.adapter_source_mode`;
the D arm omits the key and uses the backwards-compatible `both` default.
Prepare a **new** candidate directory after syncing this code. Do not reuse a
previous candidate manifest for *new* source-type arms: older manifests do not
contain their YAMLs. The verifier still accepts the exact older 13-variant
inventory to resume existing runs without changing their configs.

On each one-GPU server, use `OPERA_LOCAL_GPU=0`; use the same immutable
`OPERA_CANDIDATE_DIR` and a shared run root across servers, but a separate
variant per server. The existing launcher supports all four names and checks
the candidate manifest before launch. Example after setting the same
`OPERA_RELEASE_DIR`, `OPERA_FAMILY_DIR`, `OPERA_CANDIDATE_DIR`, and
`OPERA_NEURAL_RUN_ROOT` on all servers:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
python -m opera.run.prepare_approved_neural_runs_v1 \
  --release-dir "$OPERA_RELEASE_DIR" --family-dir "$OPERA_FAMILY_DIR" \
  --output-dir "$OPERA_CANDIDATE_DIR"
export OPERA_LOCAL_GPU=0 OPERA_LAUNCH=1
bash scripts/run_approved_neural_candidate_v1.sh time2vec_router_no_sources_e2e
# On separate one-GPU servers, replace the variant with:
# time2vec_router_disease_only_e2e
# time2vec_router_outcome_only_e2e
# time2vec_router_interacting_e2e (reuse a completed matching run if exact contract agrees)
```

Before interpreting results, verify resolved Hydra configs and DAPT checkpoint
paths match except source mode and output path, and that all four finish the
same evaluation epochs. Primary tuning metric: endpoint-group-macro CIF AUC
at a **common prespecified epoch** or a checkpoint chosen by the same rule.
Report AUC and calibration/Brier as paired patient-level contrasts, plus
per-family outcomes and route mass by source type. Estimate disease effect
as `(B−A)` and `(D−C)`, outcome effect as `(C−A)` and `(D−B)`, and interaction
as `D−B−C+A`. For AUC this last quantity is descriptive, not a causal
interaction. Replicate promising arms across seeds and use paired tuning
uncertainty; best-of-run differences alone are selection-biased. Keep the
held-out test split sealed until model selection is frozen.

The diagram in `paper/figures/router_source_factorial.tex` is schematic. Once
runs finish, replace any interpretive claims with empirical route-weight
summaries and paired contrasts; never infer source use merely from architecture.
