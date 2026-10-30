# Finish plan and independent server audits — 30 October 2026

The local rehearsal should establish a trustworthy analysis and figure suite.
It need not demonstrate a gain from selective gates or outperform XGBoost to
be ready for held-out evaluation. The remaining questions are about valid
comparisons, a defensible interpretation, complete reporting and frozen choices.

## The currently slow stage

`analysis/logs/recent_record_gaps.log` belongs to
`opera.run.compare_input_restrictions`. At this stage the four GPU exports
for removing the last 7, 14, 21 and 28 days have already completed. The
comparison runs on CPU. By default it makes 1,000 paired patient-bootstrap
draws per restriction and outcome. With all 69 outcomes this is 276,000 paired
draws, each previously calculating two AUCs using case/control pair matrices,
re-estimating censoring and computing an unused Brier score. The old command
normally prints nothing until the calculation finishes. A quiet log for an
hour is consistent with computation and does not prove the process stopped.

From the server repository root, observe the process without interrupting it:

```bash
opera_full_root=data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123
ps -u "$USER" -o pid,ppid,stat,etime,time,pcpu,pmem,args | rg '[c]ompare_input_restrictions|[c]omplete_local_figures|[r]esume_full_tuning_rehearsal'
tail -n 40 "$opera_full_root/analysis/logs/recent_record_gaps.log"
```

Repeat the process observation after a short interval. Increasing CPU time
indicates work. Sustained process state `D` or disappearance requires checking
allocation status, I/O and the enclosing log. Low GPU utilisation alone says
nothing about whether this CPU stage is healthy. Do not launch another
recovery against an active workspace. `nohup` does not extend walltime.

The local comparison now uses the existing vectorised censoring/weighted-rank
CIF AUC implementation and prints per-outcome and quarter-bootstrap progress.
The multinomial draws, paired patients, censoring re-estimation in each draw,
competing-event treatment, horizons and output schema are unchanged. New
regression tests compare all outcome estimates, domain/macro estimates and
intervals against the original implementation at the same seed, including
censoring, risk ties and exact-horizon follow-up. Empty or single-class
resamples retain undefined AUC. Syncing this file cannot accelerate a Python
process that has already imported it. Leave the current calculation running
if it is making progress. Use the faster version for a later necessary retry
or the held-out run after local sign-off, without repeating completed exports.

## Start the two router audits now

For the full set of independent ready CPU jobs, use the newer
[single second-allocation launcher](PARALLEL_TUNING_ALLOCATION_20261030.md).
It includes both router audits below, metric reconciliation, whole-cohort
evaluation and a snapshot of the clinical diagnostics. Do not launch both
command sets, which would duplicate the mechanism analysis.

These are CPU jobs. A separate CPU allocation with access to the shared
repository is sufficient. Four spare cores also permit running alongside the
main job. They read completed inputs and write a fresh sibling directory.
They neither edit the main run's receipts nor mark its stages complete.

Sync the two new audit files listed in
[the mechanism instructions](ROUTER_MECHANISM_SERVER_AUDIT_20261030.md), if not
already present. No blanket source-manifest check is required. Set
`opera_full_root` to the actual running workspace if it differs below.

```bash
opera_full_root=data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123
opera_audit_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_audit_root="$opera_full_root/parallel_audits_$opera_audit_stamp"
opera_audit_log="logs/router_mechanism_$opera_audit_stamp.log"
mkdir -p "$opera_audit_root" logs

.venv-v100/bin/python -m opera.run.audit_router_vectors \
  --vectors-root "$opera_full_root/analysis/routed_vectors" \
  --output "$opera_audit_root/router_vectors.json"

nohup env OPERA_ANALYSIS_WORKERS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-v100/bin/python -u -m opera.run.audit_frozen_router_mechanism \
  --analysis-root "$opera_full_root/analysis" \
  --output-dir "$opera_audit_root/mechanism" \
  --bootstrap 400 --threads 4 \
  > "$opera_audit_log" 2>&1 &

echo "Audit directory: $opera_audit_root"
echo "Mechanism log: $opera_audit_log"
tail -f "$opera_audit_log"
```

Return `router_vectors.json` and
`mechanism/router_mechanism_review.zip`. The ZIP has an explicit aggregate
allowlist and excludes patient vectors, identifiers, predictions and reference
tensors. The vector report measures actual outcome-to-outcome differences,
without inferring the global adapter's contribution from a common direction.
The mechanism audit reconstructs saved probabilities before evaluating global,
own-task, borrowed-bank and gate interventions. Its full analysis is also CPU
work, so completion time is not assumed to be immediate.

A second GPU is useful for the national model's downstream fits or independent
attribution inference. It will not accelerate the current bootstrap or these
audits. Do not start another full rehearsal. The current outer runner already
owns the queued IG workspace and its later restriction exports. Moving that
work to another GPU requires an explicit hand-off so two processes do not
write the same files or recompute the same work. The commands above deliberately
avoid that collision and get the router evidence without waiting for IG.

## Account for every clinical index

The last downloaded clinical inventory, not a live observation of the current
server run, contains **ten comparisons**:

| Index | Historical full risk-group figure | Scored tuning patients |
|---|---|---:|
| Computed IPI | Rendered | 909 |
| NCCN-IPI, DLBCL | Rendered | 343 |
| CLL-IPI | Ordered cutoffs unavailable | 104 |
| FLIPI2 | Rendered | 107 |
| IPS | Ordered cutoffs unavailable | 124 |
| IPSSWM | Ordered cutoffs unavailable | 77 |
| MALT-IPI | Ordered cutoffs unavailable | 68 |
| MIPI | Ordered cutoffs unavailable | 80 |
| R-IPI | Rendered | 457 |
| R-ISS | Rendered | 362 |

This matches the five figures the user sees. The renderer discovers recorded
indices through the registry `index` and `IPI` metadata fields. IPI and
NCCN-IPI also have explicit computed entries. It does not reconstruct every
other index from raw variables. Discovery accepts recorded scores, while the
generated comparison template currently requires 50 scored evaluation patients.
Every score in the historical ten met that threshold. The five missing figures
were not excluded by that patient-count minimum.

Their independently selected low-risk and high-specificity anchors did not
define an ordered partition. The screenshots confirm crossed anchors for
CLL-IPI, IPS and IPSSWM. Smaller tuning samples do not fully explain that:
the cutoffs were selected using training predictions. For IPS, two recorded
categories also cannot generally satisfy two independently chosen anchors
using one partition boundary. Previous forced `1e-6` separation was removed
because it did not satisfy the requested matching constraints.

Current code retains fitted binary anchors and their recall, precision, MCC,
specificity and confusion counts in `classification_at_anchors.csv`, even
when the full partition is unavailable. An individual anchor may itself be
unestimable. `cutoff_diagnostics.csv` distinguishes those cases. Check the
fresh reports before assigning causes to the new run.

Print the current complete inventory now, without waiting or recomputing:

```bash
opera_clinical_dir="$opera_full_root/analysis/clinical_scores"
.venv-v100/bin/python - "$opera_clinical_dir" <<'PY'
import sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
table = pd.read_csv(root / 'score_inventory.csv')
columns = [c for c in ('score_id', 'status', 'score_available_at_index_n',
                      'cutoff_failure_code', 'reason') if c in table]
print(table[columns].to_string(index=False))
print('\nStatus counts:')
print(table.status.value_counts().to_string())
PY
```

Return `score_inventory.csv`, `cutoff_diagnostics.csv`,
`classification_at_anchors.csv`, `score_population_flow.csv`,
`headline_scores.csv` and `configs/config_audit.json` from this workspace.
No patient-level input is needed for that reconciliation. If the corrections
postdated the completed clinical stage, use a fresh CPU-only
`refresh_clinical_figures` directory as in the agenda. It reuses prediction
exports and does not retrain or re-encode patients.

Before sign-off, give every configured index a visible disposition. A failed
ordered partition must not remove the index from the comparison story. Retain
the native index and model discrimination results and any valid binary-anchor
comparisons, with an explicit unavailable risk-group panel. Refreshing reports
alone does not implement that diagnostic-panel redesign. Do not restore
artificial thresholds or silently switch to quantile groups. Train-derived
matching does not guarantee equal observed tuning or held-out event rates or
specificity. Those achieved values must be reported separately.

## Ordered route to a final paper

| Work | Completion evidence | Dependency |
|---|---|---|
| Router identity and mechanism | Selected revised fit reconstructs saved predictions, raw-vector audit, full/encoder/global versus joint, own/borrowed-bank and gate interventions interpreted | Can start now on CPU |
| Comparator validity | Cross-family metric reconciliation, matched coverage, bounded TCL/transformed-FL specialist replay, unresolved tabular-selection receipts checked | Existing tuning fits only, some replay may need GPU |
| Complete clinical reporting | Every index accounted for, corrected anchor diagnostics, native categories, achieved risks/specificities, paired uncertainty for clinical-count contrasts or explicitly descriptive counts | Existing train/tuning predictions |
| Attribution and input history | Corrected IG runtime receipt, quality checks, completed recent-week and broader input-removal reports | Current run, GPU for remaining inference |
| Figure and numbers audit | Fresh dated review, exact plotted contrasts, no missing families, readable labels/legends, numerical tables traced to inputs | Fresh aggregates above |
| National arm | Completed checkpoint plus vocabulary/configuration transfer checks, planned downstream fits, tuning selection and thresholds fixed | National pretraining, then downstream training |
| Analysis lock | Model/seed roster, primary contrasts, support rules, horizons, weighting, sensitivities, clinical threshold rules and explicit gaps recorded | Local audits plus national tuning decisions |
| Held-out evaluation | One fresh workspace using the locked recipe, all success/failure receipts retained | Analysis lock |
| Submission versions | Results populated, figures/tables/supplement reconciled, abstract/research-in-context/discussion aligned, author/ethics/data/code/funding statements and LDH format checked | Held-out outputs |

In parallel with computation, finish the lean LDH draft's prose, supplemental
definition qualifications, figure captions and table shells. Keep a clearly
marked tuning snapshot. Do not fill pending national or test results by
extrapolation. The story remains whether, what and how to share information.
The discussion must distinguish model consolidation, predictive gain and
evidence about the necessity of adaptive routing.

For graphical sign-off, inspect the complete fresh gallery, not only the
router figure. Priorities remain clinical diagnostic panels and complete
index coverage, source-map provenance, hierarchy contrasts and descriptive
ribbons, headline model roster, patient-map disease readability, calibration
clipping, support/coverage exclusions and consistent panel typography. A
successful script exit cannot substitute for this visual inspection.

The practical target is **local sign-off while the national arm finishes**.
A combined untouched test readout still depends on the national downstream
fits and tuning choices, not only the pretraining checkpoint. National
superiority, narrow routing maps or statistically significant borrowing are
not release requirements. Unresolved prediction identity, broken comparisons
and silently missing clinical analyses are.

## Local verification of the runtime improvement

Thirteen focused comparison/input-restriction tests passed. They include
direct equality with the original restriction bootstrap at fixed draws,
existing fast-CIF reference comparisons and the token-removal tests. This
checks the local implementation, not the current server process or its runtime.
In a synthetic 2,500-patient benchmark over ten calls, AUC computation fell
from about 15.9 ms to 1.74 ms per call, a roughly nine-fold improvement. This
is not an end-to-end speed estimate for the server bootstrap. The two AUCs
differed only at floating-point rounding precision (less than `1e-15`).
