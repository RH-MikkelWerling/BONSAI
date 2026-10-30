# Second allocation: CPU audits of the running rehearsal

Use a separate allocation with four CPU cores and access to the shared repo
and data. GPU is not needed. This launcher explicitly hides GPUs from its
child processes. It leaves the active rehearsal, its queued IG work and its
receipts alone. Do not launch another full rehearsal or recovery copy.

## Files

Sync these three files if they are not already present on the server:

- `scripts/run_parallel_tuning_audits.py`
- `opera/diagnostics/router_mechanism.py`
- `opera/run/audit_frozen_router_mechanism.py`

The other audit modules already belong to the synced rehearsal. No source
manifest update or code-freshness flag is required.

## One command block

Run from the server repository root inside the other allocation. Change the
first path only if the active full workspace has a different name.

```bash
opera_full_root=data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123
opera_audit_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_audit_root="$opera_full_root/parallel_audits_$opera_audit_stamp"
opera_audit_log="logs/parallel_audits_$opera_audit_stamp.log"
mkdir -p logs

nohup .venv-v100/bin/python -u scripts/run_parallel_tuning_audits.py \
  --analysis-root "$opera_full_root/analysis" \
  --output-dir "$opera_audit_root" \
  --workers 2 --threads 4 --bootstrap 400 \
  > "$opera_audit_log" 2>&1 &

echo "Audit directory: $opera_audit_root"
echo "Progress log: $opera_audit_log"
tail -f "$opera_audit_log"
```

`nohup` does not extend scheduler walltime. Each launch needs a fresh output
directory. No test split, training job or fresh encoder inference is requested.

## What runs

The jobs run sequentially within this allocation, alongside the existing
rehearsal. A failed independent job does not suppress the others.

1. Snapshot the existing clinical aggregate reports and configuration audit.
   Missing reports are listed. No new clinical groups or thresholds are fitted.
2. Audit raw correction vectors across the five outcome exports.
3. Reconcile XGBoost, pooled XGBoost, joint and OPERA AUC/Brier implementations
   against saved estimates and report model/figure provenance.
4. Run the frozen-router mechanism audit. It reconstructs saved probabilities,
   compares the router's full/encoder/global paths with joint, removes correction
   banks and tests fixed task-average gates. Four hundred shared patient
   bootstrap draws and all declared horizons are used.
5. Evaluate whole-cohort outcomes for OPERA, joint and both XGBoost strategies
   at 30, 60, 90, 180, 365 and 730 days, with 1,000 paired multiplier draws.
   Fine-disease censoring strata are retained. Available and common model
   populations are distinguished.

These jobs do not supply the still-pending bounded GPU replay of the extreme
TCL/transformed-FL specialists. Nor do they move the main run's attribution
work. They obtain the independent reports that are ready now. The outer
rehearsal may later perform its own standard CPU reports too. Separate outputs
avoid races, and sidecar reports are not silently adopted as main-run outputs.

Once the mechanism stage starts, its detailed progress is in:

```bash
tail -f "$opera_audit_root/logs/mechanism.log"
```

Full completion produces:

```text
<audit directory>/parallel_audits_review.zip
```

Return that archive. The explicit allowlist includes aggregate reports,
clinical tables, whole-cohort metrics and the mechanism review ZIP. It excludes
logs, patient metadata, individual predictions, vectors and reference tensors.
The mechanism ZIP can be returned earlier as soon as it appears at:

```text
<audit directory>/mechanism/router_mechanism_review.zip
```

`parallel_audits.json` records completed/failed jobs. `complete` means execution
finished, not scientific approval of the results. Inspect numerical mismatches,
eligibility gaps and unavailable clinical anchors before release. A failed job
leaves its server-side log and does not turn the aggregate package into a
claim that all audits passed.

## Verification

The launcher has tests for read-only treatment of the active workspace,
continuation after an independent failure, tuning-only scope, missing clinical
reports and aggregate-only packaging. The existing mechanism tests verify
the numerical interventions and checkpoint reconstruction separately.
The combined launcher/mechanism suite passed **27 tests** locally. CLI help,
Python compilation and `git diff --check` also passed. Actual server inputs
and empirical mechanism results remain to be reviewed.
