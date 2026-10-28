# Paper results pipeline audit — 28 October 2026

## Scope and limits

Reviewed the production paths for bundle/model selection, native prediction
exports, normalization and paired analysis, summary aggregation, plot-table
translation, clinical-score configuration, rendering, readiness, source-removal
caching, manuscript numbers and recovery orchestration. Local CPU fixtures and
workflow tests exercise these paths. This is not a fresh server-data audit,
GPU checkpoint replay or certification that every scientific result is correct.
No training objective, model prediction or primary evaluation roster was changed
in this audit. Do not rerun expensive paired analysis for these changes.

## Confirmed fixes in this pass

1. Summary joins now discard unregistered diagnostic contrasts before translating
   contrast labels. Previously they could collide at a missing label and break a
   many-to-one join. The aggregate-output-to-manuscript-number integration test
   exposed this issue and passes after the fix.
2. Execution now catches figure construction, canvas drawing AND file-saving
   failures. Other plates finish, incomplete output files are removed, a manifest
   is retained, and the command returns failure. Missing preview inputs remain
   reported skips; an actual render error is not reported as successful execution.
3. --report now writes a report during execution too. It records actual imported
   figure-module location, Python executable, warnings about insufficient draws,
   absent summary columns and missing borrowing provenance.
4. Recovery locks the selected analysis root using Linux flock. A duplicate
   invocation stops before archiving or writing results. The script rejects
   unknown flags. Render-only recovery refreshes plot tables automatically so an
   old table schema cannot silently preserve obsolete summaries.
5. Recovery always writes an aggregate figure_results_audit.json after the render
   attempt, including failed attempts, before updating manuscript numbers.
6. Freshness checks now compare recorded input content hashes and PDF hashes in
   addition to code hashes. Missing/replaced inputs and modified/missing outputs
   are listed separately. Empty CSVs are reported rather than crashing the audit.
7. Config generation records availability/missing columns and complete-record
   counts for computed IPI and NCCN-IPI. The aggregate run audit also checks their
   metadata columns in existing workspaces. This does not manufacture missing
   registry components or claim the scores were evaluated when they were not.

## Existing checks verified or retained

- Explicit selected-fit identities, checkpoint/config hashes and immutable output
  directories in prediction exports; no silent substitution of a different fit.
- Exact patient/label/data-contract parity and follow-up/IPCW parity for paired
  model comparisons. Patient sets are not silently intersected.
- Fixed supported main comparison roster, positive-IPCW case/control counts and
  shared patient bootstrap weights; coverage is distinct from model fit coverage.
- Equal-cell and disease-balanced paired summaries remain separate estimands.
  Descriptive median/IQR fallbacks are labelled and absent summary columns warn.
- Clinical categories are explicitly harmonized; unknown labels fail validation.
  Available-score populations are defined by observed score availability.
- Real plot recipes require real analysis inputs. Missing plates are not replaced
  by synthetic scientific results. Synthetic patient illustrations are a separate,
  explicitly labelled attribution input.
- Source-removal cache validation is separate from table/figure rebuilding. Copied
  old aggregate borrowing tables do not become newly verified by rerendering.
- Isolated overnight orchestration retains independent outputs when another stage
  fails, and records incomplete status rather than treating a preview as complete.

## Validation

The broad CPU workflow run passed 152 selected cases, covering paper workflow,
input builders, clinical scores, analysis, table translation, plots, rendering,
readiness, number exports, run bundles and recovery orchestration. Additional
focused audit/render/config checks passed after subsequent hardening.

One real XGBoost-training integration test could not execute in this Mac
Python environment: libxgboost cannot load libomp.dylib. It was run, failed at
library import, and excluded from the broad rerun. This is a local dependency
limitation, not a passing XGBoost training test or evidence of a server model bug.
GPU training and secure-server datasets were not run locally. Shell syntax was
checked; flock-based recovery execution is intended for the Linux server.

## Server continuation

After syncing, wait for any existing writer to this root to finish. Then use:

```bash
plot_root=data/paper_256_v3/analysis/plot_rebuild_20261028_080223
mkdir -p logs
nohup bash scripts/rerun_tuning_figure_outputs.sh "$plot_root" --render-only \
  > logs/figure_pipeline_checked_20261028.log 2>&1 &
```

The wrapper rebuilds plotting tables, renders, writes figure_results_audit.json,
and updates numbers/tables if rendering succeeds. It does not repeat model
inference, training or bootstrap analysis. Inspect the aggregate audit, render
manifest and recovery logs before interpreting a preview as complete. Missing
clinical score source fields, external origin estimates, router vector exports,
and cached borrowing remain explicit outstanding inputs, not repaired data.
