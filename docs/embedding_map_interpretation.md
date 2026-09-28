# Interpreting embedding diagnostics

Training curves must use logged epoch aggregates (`train/loss_epoch` takes
precedence over `train/loss_step`). Repeated unsuffixed observations within an
epoch are not averaged retrospectively: batch means may have different weights
and logging frequency. They are skipped with a warning.

At horizon t, primary events are positives; competing deaths are known negatives
for primary-event cumulative incidence. Patients followed beyond t are known
negatives; early censoring is unknown, not a negative. This helper currently
requires follow-up strictly beyond t for event-free patients; check agreement
with the evaluation pipeline's boundary convention before comparing exports.

The risk surface is a normalized Gaussian-kernel IPCW average. Without censoring
weights it is conceptually the same local event-fraction estimator as the KDE
facets, but binning, bandwidth, eligibility and support masks must also match
before expecting identical surfaces. A pooled marginal censoring KM assumes
independent censoring; heterogeneous cohorts may require cohort-stratified or
conditional censoring weights. Censor-survival clipping at 0.05 is a sensitivity
choice, not a guarantee of reliable correction.

The default calibration surface subtracts the identically smoothed predicted
CIF from observed IPCW risk, using the same evaluation patients and 2D kernel.
Positive values indicate underprediction. Full-representation neighborhoods are
an explicit alternative and do not yield the same observed surface as the 2D
risk panel. Never use training predictions to claim out-of-sample calibration.

Optional local z maps bootstrap paired residuals at patient level while holding
the model, censoring weights, bandwidth and projection fixed. They omit model
and censor-model estimation uncertainty. Cells overlap and are correlated; these
are exploratory diagnostics, not multiplicity-adjusted significance tests.

Effective support is Kish ESS: `(sum a)^2 / sum a^2`, where `a = kernel * IPCW`.
It is neither a literal patient count nor event count, and high ESS cannot make
a rare endpoint precise. Risk/residual panels mask low ESS and low relative
kernel mass; support plots mask low mass only so low ESS remains visible. The
default 1% density threshold and ESS 25 need sensitivity checks. Calibration
must also be reported with conventional outcome/cohort-level metrics.

Suggested paper layout: a small, prespecified representative set at one primary
horizon in the main paper; family-organized risk facets in supplements, with
shared projection and probability scale; full status/calibration/support panels
for representative cases and cases with enough events/support. Export all maps
for diagnostic use, but do not select paper examples for their apparent visual
separation. Cohort/outcome metric tables or heatmaps provide the exhaustive atlas;
embedding maps illustrate structure rather than demonstrate discrimination.

Rendering interpolates continuous fields and support criteria onto a finer
display grid, without changing the estimation bandwidth or calculating more
independent observations. A geometric contour clips the rendered surface at
the support threshold instead of showing a jagged boolean pixel mask. Risk,
observation time and support use plasma, and signed residuals a zero-centered
diverging scale. Matching palette
colors across these quantities would not imply matching numerical units.
