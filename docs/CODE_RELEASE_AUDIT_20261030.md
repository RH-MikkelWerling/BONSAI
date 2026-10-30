# Code and release audit — 30 October 2026

**Status: local verification passed; server release remains pending. The test
set remains unopened.**
Recovery now accepts synced source code by default. The blanket runtime
fingerprint comparison is opt-in, while scientific-input and completed-output
checks remain enabled. Original and resumed code fingerprints are recorded.
Completed stages are reused, so the clinical corrections below still require
fresh clinical reporting if that stage completed before those corrections.

## Confirmed faults and corrections

### Clinical risk-group thresholds

The high-risk anchor selected a weighted control-score quantile, but classified
patients at that score as positive (`risk >= cutoff`). Its stated specificity
counted controls below the boundary. Consequently, the selected control and
all controls tied with it could be placed on the wrong side of the target.
The corrected boundary uses the next observed risk above the selected control
score, or a value just above that score when nobody can be flagged. It reaches
at least the index's training specificity, with the next attainable level when
ties prevent an exact match. Achieved specificity is still reported separately
on evaluation patients.

The low-risk search examined 197 percentile candidates between 1% and 99%.
This could miss a group below the first percentile even if it contained the
required 20 patients. It could also stop short of the largest feasible group.
The corrected search evaluates every attainable boundary using cumulative
weighted event counts. The event-rate target and minimum group size remain
unchanged. Crossing anchors and unsupported native partitions still fail
explicitly. These corrections do not make incompatible anchors compatible.

Regression examples cover exact boundary classification, tied controls, and
a 20-patient low-risk group in a 5,000-patient training population. The latter
was invisible to the old percentile grid.

**Consequence:** refresh clinical thresholds, operating-point tables, and
figures from the existing training and tuning predictions. Retraining or
re-exporting those predictions is unnecessary. The clinical counts in the
editorial paper remain provisional until replaced by this refresh.

### Final-test figure assembly and packaging

Review assembly hardcoded tuning paths and watermarks. Clinical ZIP validation
also assumed tuning, and borrowing tables were selected from tuning folders.
Assembly now accepts `--split tuning` or `--split held_out` and records that
choice in its manifest. Packaging checks the matching clinical receipt and
includes borrowing outputs from that split only. Held-out assembly clears an
inherited tuning watermark. Regression tests cover held-out assembly and ZIPs.

The direct evaluator also unnecessarily restricted routed-vector exports to
tuning. It now accepts either evaluation split, preserving the split in the
vector receipt. Projection already fits on tuning patients and applies that
mapping to held-out patients. The tuning-only completion/rehearsal runners
remain tuning-only.

### Tabular tuning failures

An all-failed candidate search could return an empty parameter dictionary and
silently substitute defaults. It now raises with the last error. Deliberate
fallbacks for unsupported selection populations or models remain distinct.
The grid, selection metric, and successful fitting behavior are unchanged.
Existing fitted models and predictions have not been altered. Before release,
inspect existing tabular metadata for unexplained empty `tune_params` where
the tuning population supported selection.

### Other implementation and verification corrections

- Registered the recent-record gap figure in the shared attribution recipe,
  so figure-readiness checks can identify its producer.
- Corrected string-to-date parsing in the upstream first-line comparison audit
  for Polars 2 compatibility. Frozen outcome and treatment definitions are
  unchanged.
- Updated a removed private-name import in an IPCW reference test, incomplete
  validation-module fixtures, and old plot expectations. Kept router-weight
  normalization checks intact and repaired the synthetic fixture that had
  discarded sources without preserving their probability mass.
- Closed figures left open by several tests, which had contaminated later
  tests' checks for figure leaks. Missing optional Bayesian dependencies are
  reported as skips rather than blocking unrelated tests.

## Critical paths inspected

The review covered checkpoint reconstruction and selection receipts, router
source indexing and anchored gradients, evaluation-mode router gradients,
disease-term decomposition, tuning-only mean-ablation references, temporal
membership, sequence restrictions, tabular preprocessing and model selection,
canonical prediction adaptation, censoring weights, paired metrics, patient
bootstrap reuse, clinical populations and thresholds, figure provenance,
review assembly, and aggregate packaging.

The inspected primary metric path treats competing death as an observed
control and retains zero-weight early censoring in the full-risk-set Brier
denominator. It checks patient, label, release-contract, and follow-up parity.
Patient bootstrap draws are shared across outcomes, horizons, methods, and
training seeds. These properties are covered by numerical regression tests.
No new router forward-pass arithmetic or fitted parameters were introduced.

This is a broad local test and critical-path review, not a claim that each
line of the repository has been proved correct. It does not certify unseen
server files, registry records, GPU execution, or the pending national model.

## Verification record

The first full run exposed an outdated test import during collection. After
repairing that harness issue, it ran 1,712 tests with 17 failures. These were
investigated rather than suppressed. Local XGBoost initially lacked a visible
OpenMP library. Using the library already bundled with PyTorch allowed the
tabular tests to run. No package installation or model refitting was needed.

Final full-suite verification: **1,710 passed, 12 skipped, zero failures and
zero errors** (1,722 collected tests). The skips comprise five PyMC checks,
one xarray check, four CUDA checks, and two FlashAttention/SDPA GPU parity
checks. Those paths are not locally certified. The report is recorded in
`scratch/code_audit_20261030/release_pytest.xml` and its log.
Ruff checks for syntax and undefined
names cover `bonsai/`, `opera/`, and `scripts/`. Shell scripts pass `bash -n`.
All OPERA R files parse, and the base-R first-line instrumentation test passes.

## Server sequence

1. Resume the existing full tuning workspace using the
   [recovery instructions](RESUME_FULL_TUNING_REHEARSAL_20261030.md). Copy only
   the wrapper for this step. It leaves the existing scientific runtime
   fingerprint unchanged.
2. Once that run ends, sync the reviewed fixes into the release checkout.
   Generate a fresh CPU-only clinical report from its existing predictions:

   ```bash
   opera_full_root=data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123
   opera_clinical_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
   opera_clinical_root="$opera_full_root/clinical_corrected_$opera_clinical_stamp"

   .venv-v100/bin/python -m opera.run.refresh_clinical_figures \
     --workspace "$opera_full_root/analysis" \
     --output-dir "$opera_clinical_root"
   ```

   Keep the corrected clinical workspace beside the old report. Later review
   assembly should receive its `clinical_scores` directory explicitly.
3. Reconcile the completed server replay, raw vectors, mean/zero removals,
   cross-family metrics, refreshed clinical reports, and missing-analysis list.
   Review national checkpoint/configuration/vocabulary receipts and downstream
   tuning choices when they arrive.
4. Freeze the actual local/national model roster, cutoffs, estimand, weighting,
   and exclusions. Prepare the held-out commands from that record. The existing
   full rehearsal command is not a held-out launcher.

A clean local suite supports this release decision. It does not replace the
server receipts or the dated analysis lock. Weak borrowing or a national model
that does not improve performance is a valid result, not a reason to keep
changing the model after opening the test set.

Clinical classification tables currently provide point estimates. Survival
intervals in the plots do not supply paired intervals for claims about extra
low-risk patients or additional events captured. Specify and implement those
intervals before making such claims, or limit the clinical account to the
available descriptive estimates. Pending manuscript text must not be treated
as evidence that a missing uncertainty analysis has already been performed.
