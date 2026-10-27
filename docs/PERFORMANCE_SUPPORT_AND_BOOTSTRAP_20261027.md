# Supported-cell performance and bounded-cost uncertainty

The paper analysis runner now defaults to at least five observed/IPCW-positive
cases AND five observed/IPCW-positive controls in each evaluated disease/outcome/
horizon cell. The same eligibility is applied to all models after exact patient,
label and IPCW alignment. It applies to main, stratified and late-diagnosis
paired re-analyses. Cells remain in estimates/contrasts/coverage with counts and
NaN performance values; low-support cells skip expensive bootstrap metric work
and patient loss-gain contributions. Coverage distinguishes insufficient test
support from a failed model fit. General low-level metric APIs remain available
for raw diagnostics; this does not delete training tasks or predictions.

Eligibility is fixed using the original evaluated sample, not selected afresh
on each bootstrap replicate. Each subgroup re-analysis applies the same rule
to its own original subgroup sample. The point estimate and interval therefore
refer to the same eligible task inventory. This is a minimum stability rule,
not a guarantee that five events is enough for precise inference.

The main paper runner now defaults to `bootstrap_method: multiplier`: a
frequentist exchangeably weighted patient bootstrap using positive Exponential(1)
weights normalized to sum to N within split/disease. Exactly the same patient
weights are reused across models, outcomes, horizons and training seeds. The
fixed original eligible roster and all observed classes survive in every draw.
No rejection loops or per-replicate cell dropping are needed. Ordinary
`multinomial` resampling remains available as a sensitivity analysis; low-level
APIs keep their explicit/default ordinary-resampling behavior, so separate
analyses do not silently switch. Record the actual method for every output.

This is frequentist resampling uncertainty conditional on the fitted models,
fixed IPCW weights and the original eligible roster. Positive weighting does not
require a Bayesian interpretation: exchangeably weighted bootstraps have a
frequentist empirical-process justification (Praestgaard and Wellner, 1993):
https://sites.stat.washington.edu/jaw/JAW-papers/jaw-praest-93AP.pdf
The older opt-in `bayesian` name remains available but is not the paper default;
its manifest interpretation is explicitly different. No posterior probability
interpretation is applied to `multiplier` outputs. Existing bootstrap sign-tail
measures remain approximate; this change does not manufacture valid null tests
or address model-selection uncertainty.

Synthetic validation (`scratch/bootstrap_validation/`) compared percentile
intervals from ordinary and multiplier weights in 200 datasets at each N=50 and
N=200, with 399 draws, for paired AUC and Brier differences. Widths were similar.
At N=200 coverage was 93.5%/92.5% for ordinary/weighted AUC and 94.5%/94.0% for
Brier. At N=50 it was 94.0%/93.0% for AUC and 92.5%/91.0% for Brier. These
numbers have Monte Carlo error and show imperfect small-sample coverage; they
are a smoke check, not a coverage guarantee for rare-event IPCW panels. Test
cases verify positive weights, deterministic draws, eligibility and pairing.

Use a fresh analysis output directory. The same saved predictions and frozen
configuration suffice: no training, inference or patient-level export is
needed. Explicit existing `bootstrap_method: multinomial` settings override the
new runner default; update deliberately when preparing the new analysis config.
The analysis manifest records both method and support thresholds. Increasing
workers multiplies memory use; do not blindly run many analyses concurrently.

## Training eligibility is distinct

Training datamodules enforce patient/outcome ascertainment and registry
eligibility. Endpoint-group AUC/Brier validation summaries have a configurable
support rule, commonly 20 cases and 20 controls, applied to pooled outcome/
horizon validation data. The horizon-BCE checkpoint score uses its own
known-status loss calculation and is a separate path. IPW sampling additionally
has training-pool minimums (the bundle's IPW settings use 10 unique events and
20 controls). None of these guarantees at least five cases/controls in every
fine-disease/outcome/horizon evaluation cell, and none implies that unsupported
evaluation cells were excluded from training.

Both queue-status and prediction-status verify completed checkpoints with
SHA256, so they can scan large volumes of checkpoint data over shared storage.
A long status call is not evidence of an ongoing training job. Avoid repeatedly
launching identical status audits while earlier checks are still running.
