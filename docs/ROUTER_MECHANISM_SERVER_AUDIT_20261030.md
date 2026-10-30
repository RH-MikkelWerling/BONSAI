# Frozen-router mechanism audit — server command

This is the separate follow-up to the running full tuning rehearsal. It uses
saved pooled encoder vectors on CPU. It does not retrain, re-encode medical
records, inspect the test set or require a source manifest. Leave the active
run running. This can start **while `recent_record_gaps` is running**. By that
stage, the selected-fit receipt, tuning inventory and five vector exports
already exist. It does not require `complete_figures` to finish. Use spare
CPU cores or a separate CPU allocation on the same shared filesystem.

## Files to sync

Only two new Python files are needed:

- `opera/diagnostics/router_mechanism.py`
- `opera/run/audit_frozen_router_mechanism.py`

Existing prediction/model implementations are unchanged. The audit checks the
selected checkpoint and directly reconstructs saved probabilities. It records
its own source hashes without comparing them with an old source manifest.

## Run

From the repository root, with eight CPU cores available. GPU is not required:

```bash
opera_full_root=data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123
opera_mechanism_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_mechanism_root="$opera_full_root/router_mechanism_$opera_mechanism_stamp"
opera_mechanism_log="logs/router_mechanism_$opera_mechanism_stamp.log"
mkdir -p logs

nohup env OPERA_ANALYSIS_WORKERS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-v100/bin/python -u -m opera.run.audit_frozen_router_mechanism \
  --analysis-root "$opera_full_root/analysis" \
  --output-dir "$opera_mechanism_root" \
  --bootstrap 400 --threads 8 \
  > "$opera_mechanism_log" 2>&1 &

tail -f "$opera_mechanism_log"
```

This uses all configured target outcomes and analysis horizons, retaining
the inventory's case/control cutoffs and disease-conditional IPCW weights.
One set of positive patient multiplier draws is reused across outcomes,
horizons and interventions. Comparisons use matched patients, labels and
weights. Reference means use eligible tuning patients without their outcome
labels. Intervals condition on the fitted checkpoint and reference means.

The default vector source is the selected prediction folder's
`patient_embeddings.npz`. If absent, it uses a routed-vector export from this
analysis workspace. IDs are aligned explicitly and vectors must cover every
evaluated patient. An alternative selected-fit tuning NPZ can be supplied
with `--vectors /actual/path/to/vectors.npz`. Scheduler walltime still applies.

## Refresh figures without rerunning the audit

After syncing the updated `opera/run/audit_frozen_router_mechanism.py`, use
the audit output directory printed by the earlier command:

```bash
.venv-v100/bin/python -m opera.run.audit_frozen_router_mechanism \
  --output-dir "$opera_mechanism_root" --render-only
```

This rewrites figures, HTML and the aggregate review ZIP from existing CSVs.
It does not reload model checkpoints, perform inference, recompute metrics or
alter the scientific report. No analysis-root argument is needed.

## Outputs

Before interpretation, the component sum must match the production readout/head,
and reconstructed cumulative incidence must match every selected saved
prediction at `atol=1e-5, rtol=1e-4`. A mismatch stops with a diagnostic report.
The 14 interventions comprise:

- Zero and tuning-mean replacement of the global term, own-task terms,
  borrowed-disease bank, borrowed-outcome bank and both borrowed banks.
  Other gates and terms stay fixed.
- The router checkpoint's encoder/head alone, and its encoder/head plus
  global term. These are not the separately trained J_L and N_A_L models.
- Within-task average gates, retaining patient-dependent adapter outputs.
- Average relative borrowed composition, retaining each patient's original
  borrowing budget and own/null weights.

It also compares full OPERA, its encoder/head without corrections and its
global-only path directly with the declared J_L tuning predictions. Those
comparisons use matched coverage and the same patient draws. They help
locate whether the advantage is already present in the router-trained shared
path or appears with corrections. Removing corrections is a within-fit
counterfactual, not a causal partition of training effects.

| File | Purpose |
|---|---|
| `report.json` | Fit, scope, reconstruction errors and intervention completion |
| `component_head_effects.csv` | Signed/absolute component effects before hazard clipping, by disease, outcome, cause and interval |
| `risk_changes.csv` | Risk changes overall and by observed horizon status |
| `estimates.csv`, `contrasts.csv`, `contrast_coverage.csv` | Metrics, paired effects and support gaps |
| `effect_summaries.csv` | Paired intervals by horizon and available strata, with equal-cell and disease-balanced summaries |
| `trained_control_ladder.csv` | Existing separately trained local controls on a common supported roster, when available |
| `joint_path_*.csv` | Paired comparisons of full/router-encoder/router-global paths with the separately trained joint model |
| `router_mechanism.pdf`, `.png` | Three questions: gain over joint, useful corrections and individual routing, shown with AUC |
| `router_mechanism_probability.pdf`, `.png` | Matching three-panel view of IPA |
| `router_mechanism_sensitivity.pdf`, `.png` | All 14 interventions in nine grouped rows, with AUC, Brier and IPA |
| `index.html`, `figure_render.json` | Browsable review and figure/input provenance |
| `router_mechanism_review.zip` | Allowlisted aggregate results and figures to return |
| `references_INTERNAL.pt` | Tuning-reference tensors, retained on the server |

Return **`router_mechanism_review.zip`**. It excludes reference tensors,
patient vectors, individual predictions, IDs and bootstrap patient weights.
It is produced only after successful completion. These diagnostic figures
are not silently inserted into the paper's figure suite.

## Interpretation

Metric contrasts are **full minus intervention**, oriented so positive
AUC/IPA/Brier-improvement values favour full OPERA. The risk-change table is
**intervention minus full**, explicitly recorded in its `direction` column.
Head-term changes are not risk or accuracy gains. Disabled competing causes
are marked inactive.

1. If component removal worsens accuracy, this fit relies on it usefully on
   this population. Large vectors or risk shifts alone do not establish benefit.
2. If average gates preserve performance, broad task-trained functions can
   explain benefit with little evidence for patient-specific allocation.
   A non-significant difference does not establish equivalence.
3. If freezing source composition hurts while preserving borrowing amount,
   source choice contributes within this fit. If freezing all gates hurts
   but freezing composition does not, variation in borrowing amount or own
   allocation is a candidate explanation.
4. Risk/AUC effects do not add. Separately trained controls and fixed-checkpoint
   interventions answer different questions.

These are reliance tests. A claim that adaptive routing is necessary still
requires a matched trained static-gate comparator. Weak or diffuse effects
remain valid findings. The audit distinguishes these possibilities without
another model search or opening held-out data.

## Local verification

The new audit has numerical and end-to-end tests for exact component
reconstruction, original component-CIF agreement, outcome ordering/subsets,
mean ablation, borrowed-budget/home/null preservation, zero borrowing,
eligibility-aware reference means, batch invariance, checkpoint reconstruction,
disabled competing causes, saved-vector ID alignment, shared IPCW/resamples,
joint-path comparisons, refusal of held-out artifacts, failed replay, and
aggregate-only ZIP contents. A synthetic three-panel plot was inspected and
its tick spacing corrected. Synthetic outputs are not paper results.

The focused router/whole-cohort suite passed **74 tests**, with four
CUDA-only tests skipped locally. Its receipt is
`scratch/code_audit_20261030/frozen_router_mechanism_release.xml`.
The complete repository regression suite then passed **1,739 tests**, with
12 skips for unavailable GPU/FlashAttention or optional Bayesian dependencies,
and no failures. Its receipt is
`scratch/code_audit_20261030/release_with_frozen_router_mechanism.xml`.

The empirical server audit is pending. Passing implementation tests does not
predetermine which components will improve accuracy.
