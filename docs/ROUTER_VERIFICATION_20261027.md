# Verify tuning router inference

Sync the current source, then run from the server repo root inside an allocated GPU node, using the same Python environment as the rehearsal:

```bash
python -m opera.run.verify_router_predictions \
  --env-file .env.rehearsal \
  --device cuda \
  --output-dir "data/paper_256_v3/analysis/router_verification_$(date +%Y%m%d_%H%M%S)"
```

This command runs no training and does not overwrite existing predictions, analyses, or figures. It refuses an existing output directory. Keep the node allocation alive until it finishes; nohup does not extend scheduler walltime.

The command checks the selected J_L and R_L tuning artifacts against their frozen bundle configurations, checkpoint selection markers and file hashes. It recomputes native and paper metrics from the full saved tuning predictions, compares joint/router prediction keys, follow-up and absolute CIF differences, then replays two patients per fine disease through the production evaluator with batch size one. The replay must match all saved horizons within atol=1e-5 and rtol=1e-4. Selection uses IDs and disease only, without looking at labels or gains.

For the routed checkpoint, the replay also exports encoder-only and zero-borrowing predictions. Encoder-only means that checkpoint's encoder and head, not the separately trained joint model. Zero-borrowing removes borrowed contributions at unchanged routing weights while preserving own and global corrections. This is a diagnostic intervention, not the primary mean-replacement source-removal analysis or a performance estimate.

Return only `report.json`. Keep `selected_subjects_INTERNAL.json`, the replay parquets and detailed logs on the server. If a subprocess fails, inspect J_L.log or R_L.log there; report.json retains completed checks.

`replay_passed` means both small replays reproduced saved predictions. Metric reconciliation and provenance still need inspection: missing estimates, status mismatches and material numeric errors need explanation. A correct replay cannot establish the scientific value of borrowing, and a median cell gain near zero can coexist with a positive supported, disease-balanced mean. The known missing bootstrap intervals remain a separate analysis issue; these checks do not manufacture confidence intervals or establish their validity.

## Router map arrows and horizons

The horizons figure now defaults to four panels: D-S, O-S, J-S and R1
(routed minus joint). It still shows median/IQR across cells, not the
patient-pooled AUC. The single-panel option is also supported.

The calibration maps already had an arrow renderer, but the embedding-table
builder always set routed coordinates to null. `calibration_maps.routed_vectors`
now accepts an outcome-to-NPZ mapping of production `--export-routed-outcome`
exports. The companion JSON is required and checked for target outcome and
split. The map stage must use that router checkpoint's shared patient embeddings;
the builder checks vector equality and patient coverage. Both before and after
vectors pass through the same scaler/PCA/UMAP fitted on tuning data. It refuses
to superimpose another checkpoint's correction on the joint model's latent map.
Use router predictions with the router stage for the calibration panels.

The arrows are local means filtered by count/coherence. The simulation pass
added a single labelled magnification shared across the entire row; explicit
router_arrow_scale=1 restores projection units. There is no per-outcome normalization. Zero
movement no longer attempts an empty-median scaling calculation. Projection
movement is descriptive, not an estimate of performance benefit. If the export
files do not yet exist, the optional row remains absent; no synthetic arrows
are generated. The example embedding configuration documents this input.

Scope of shared issues: identifier joins can affect every embedding model;
stale derived tables can affect any contrast; strict fixed-roster bootstrap
failure affects sparse AUC panels across models. The wrong-horizon selector was
specific to the adapter-bank figure. The production replay does not reveal a
router inference bypass. Sparse ranking ties remain a hypothesis until the
cell-distribution audit is returned.

## Additional analysis integrity checks

Horizon pooling now requires bootstrap draws for every finite point-estimate
cell before pooling. Previously its inner join could silently omit a horizon
from the interval while retaining it in the pooled point estimate. This is a
confirmed code path, not evidence that existing server artifacts lost draws.

Plot-table training metadata rejects conflicting rows for a cell rather than
arbitrarily keeping the first. Calibration-map and gallery inputs similarly
reject conflicting patient/outcome prediction duplicates; calibration-map
outcomes must be nonempty/unique and have some common evaluated patients.
Embedding-table construction completes validation before replacing its outputs.
The local displacement grid rejects empty/nonfinite coordinate inputs.

The intended arrow layout is a separate row across the outcome columns. Each
arrow averages individual before-to-after displacement vectors within a spatial
grid bin, anchored at the mean starting coordinate of that bin. Count and
coherence thresholds suppress undersupported or cancelling directions. These
checks and layout changes do not require changing the ongoing CPU audit command.
