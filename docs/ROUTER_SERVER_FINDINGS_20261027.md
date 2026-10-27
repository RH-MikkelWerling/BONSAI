# Server router audit findings — 27 October 2026

Source: Mikkel's `router_audit_current.json`, supplied after the initial audit
and before the metric-recomputation extension. This records the audited server
analysis; it does not independently reproduce inference or prove every older
PNG was rendered from these current inputs.

## Model identity verified by the returned audit

- Analysis R_L and R_AB_L both resolve to `seed42/L/fix_R_AB`, prediction run
  `79d39ad1e7cb51472026`, with both-bank source-anchored adapters, global adapter
  enabled and two warm-up epochs. The configured override is R_AB_L.
- OPERA is configured as R_L, therefore refers to this remapped fixed router.
- R_ORIG_L is the earlier `query_both_r8_interacting_q512` fit.
- Every listed bundle-config, checkpoint and prediction checksum matches.
- The route export directory matches the primary R_L prediction directory.
- Source-removal provenance identifies R_AB_L / fix_R_AB, with J_L and
  RBH_AB_L as the intended references.

## Routing health

The original router puts its top gate on overall-survival's source in 99.82%
of rows, with mean top weight 0.8993: the documented collapse is reproduced.
The fixed R_AB passes all reported support-weighted health criteria. Its mean
maximum gate is 0.0722 and mean effective source count is 70.8026; the leading
support-weighted source dominates only 15.07% of rows and six distinct top
borrowed sources occur across diseases. This establishes absence of that
single-source collapse under the reported criterion, not optimality or strong
patient-specific specialization. Routing remains broad.

## Reported AUC gain is present; summaries disagree

All numbers below are AUC points (raw AUC multiplied by 100), point estimates
only; the current panel confidence intervals are unavailable.

| Contrast and horizons | Decision weighting, >=5 cases/controls | Equal-cell mean, all finite cells | Median of all finite cells |
| --- | ---: | ---: | ---: |
| Fixed router R_AB minus J, all six horizons | +0.8616 | +0.1737 | 0 |
| Fixed router R_AB minus J, 365 days | +1.6588 | +0.7206 | 0 |
| Original collapsed router minus J, all six horizons | +1.2474 | +0.1181 | 0 |
| Fixed router minus RBH_AB, all six horizons | +1.3790 | +1.2448 | 0 |

Thus Mikkel's recollection of roughly one AUC point is supported by the returned
fine-cell decision-weighted data. It need not arise from patient-pooled training
logs or confusing the no-borrowing comparator with J. Those were possibilities
before the audit; this result narrows the explanation to summary/support policy
for the figures in question.

All-horizon summaries include 6,608 finite disease–outcome–horizon entries,
versus 3,004 meeting the decision support rule. At 365 days these counts are
1,216 and 600. These are cell–horizon entries, not unique patients. Weighting and
support filtering change together here; the audit does not separate their
individual contributions. A zero median cannot be read as a zero panel mean.

## Remaining statistical issue

All 42 returned AUC panel rows have `n_valid_bootstrap: 0`, NaN confidence bounds
and `interval_status: insufficient_draws`. These are unusable panel intervals,
not evidence of no effect and not proof that bootstrap resampling was never
run. The full-panel fixed-inventory policy can invalidate every pooled draw
when tiny cells repeatedly resample to one class. The roster and interval policy
must be reconciled with the intended reporting/decision protocol before claims.
Do not fill missing intervals with zeros or treat heterogeneity IQRs as CIs.

The updated optional `--recompute-metrics` check remains pending. It will
recalculate metrics from saved CIFs against native and paper tables. Existing
checks already establish artifact hashes/identity and explain the large point-
summary discrepancy; no new training is motivated by this audit.

## Embedding diagnosis confirmed

CSV IDs are strings and Parquet IDs are int64. The returned join has zero rows
before normalization and 68,174 matched outcome rows after normalization. The
ID fix directly addresses the blank outcome panels; existing tables can be
rerendered with corrected code and the requested masked smooth colour fields.

## Confirmed figure-code defects found after reading this audit

The adapter-bank forest in `figure_routing` selected `gate[contrast].iloc[0]`
without filtering metric or horizon. Its real input contains all six horizons
and the pooled summary, for several metrics. Thus the row chosen could be
30-day AUC next to a pooled-horizon panel, regardless of the requested metric.
This is a genuine plotting bug, independent of weighting. Fixed with explicit
metric/horizon selection and rejection of duplicate/missing requested summaries.
Regression tests use distinct values for 30-day AUC, pooled AUC and pooled Brier.

Missing CIs in that panel were silently drawn as point-only diamonds while the
axis promised 95% CIs. They are now open diamonds with an explicit unavailable-CI
note. This exposes the inference failure; it does not manufacture intervals.
The original bank factorial is now titled as such rather than suggesting it is
a matched decomposition of the fixed primary router.

The route-usage plot called a ratio of gate weights "Share of correction by
bank" and called `1 - shared_weight` "Routing intensity". These quantities do
not include adapter output norms, the global correction or risk-head projection.
The patient base vector is always present in this architecture. The labels now
say "Gate mass by bank" and "Source-slot allocation". Gate summaries cannot
by themselves diagnose absence of predictive contribution.

Plot-table exports now retain case/control support and bootstrap status when
available, so downstream checks can distinguish low-support/invalid intervals
from precise null findings. Median/IQR heterogeneity panels remain labelled as
such; a primary supported, disease-balanced performance summary still needs to
be integrated coherently with the resolved panel inference policy.

## Production replay received and local follow-up

The returned verification report has status `replay_passed`. Both selected
checkpoints reproduce 46 sampled patients at six horizons, with maximum absolute
CIF errors around 1e-6. Native and paper metric recomputation agrees to floating
point precision, without missing estimates or finite-status mismatches. The
patient-pooled mean AUCs are 0.7710872006 (J) and 0.7836230421 (fixed R): a
1.253584-point difference. This is a different estimand from within-fine-disease
cell summaries. Zero-borrowing intervention changes sampled 365-day CIFs by a
median absolute 0.0189554; this is an intervention effect, not proof of improved
AUC. No training error or inference bypass is established by this report.

The next CPU-only audit includes exact positive/negative/zero cell counts,
separately for supported cells (at least five cases and controls) and sparse
cells, at each horizon and across horizons. Tiny nonzero gains are distinguished
from exact zero; this must be read before attributing zeros to sparse ranks.

Effect-summary output now records the strict roster bootstrap policy, number of
unstable cells, replicate count and minimum finite cells per draw. These fields,
plus interval status and valid draw count, survive panel plot-table construction.
The estimator itself is unchanged. A test demonstrates the pathological case:
each draw contains one invalid cell, leaving zero valid panel draws despite all
point estimates being finite. This behavior is not repaired by replacing missing
values with zero or choosing cells after looking at gains.

A confirmed stale-output bug was fixed in plot-table construction: a failed
missing-input rebuild previously left a same-named old derived table on disk.
That table is now removed and the manifest records the removal. Upstream patient
data and predictions are untouched. This fix covers missing inputs encountered
by that builder; file existence alone is still not a universal freshness audit.

After syncing, run the CPU-only audit (no GPU replay needed):

```bash
mkdir -p logs/rehearsal
python -m opera.run.audit_tuning_router --env-file .env.rehearsal \
  > logs/rehearsal/router_cell_distribution.json
```

Return that aggregate JSON to inspect zero-gain composition. Existing hashes,
metric recomputation and inference replay need not be repeated for this change.
Then rebuild plot tables and rerender with the already-fixed embedding and
routing code:

```bash
bash scripts/run_paper_experiments.sh plot-tables --env-file .env.rehearsal --execute
bash scripts/run_paper_experiments.sh figure-readiness --env-file .env.rehearsal --execute
bash scripts/run_paper_experiments.sh figures --env-file .env.rehearsal --execute
```

Stop if readiness reports missing inputs; its output identifies the producer.
These commands replace derived plot tables and figure files in the configured
analysis root. Preserve the current figure directory first if a visual
before/after comparison is wanted. Existing prediction-analysis tables will not
gain new bootstrap diagnostic columns until that analysis is rerun into a fresh
output directory; the next run should follow a resolved reporting/interval
policy, rather than incur a full bootstrap rerun solely to add diagnostics.
Unusable intervals remain unresolved and must not be interpreted as zero effect.

## Returned cell-distribution audit

The current analysis inventory still contains the original/fixed routers and
joint/control models; neither newly finishing IPW fit appears in this returned
artifact list. Training completion must be checked separately from prediction
completion and inventory inclusion.

For R1 (fixed router minus joint), across all horizons:

- All 6,608 finite cell-horizon entries: 3,047 improved, 2,639 worsened, 922
  exactly zero; median zero.
- Of those exact zeros, 814 (88.3%) are in cells with fewer than five cases or
  controls. This is consistent with coarse empirical ranking estimates in small
  cells; it is not evidence of identical probability predictions.
- The already-defined minimum-five support roster has 3,004 entries: 1,714
  improved, 1,182 worsened, 108 zero; median gain 0.6403 AUC points, equal-cell
  mean 0.8679 points and disease-balanced mean 0.8616 points.
- At one year, 600 supported cells have 351 improvements, 229 losses and 20
  zeros; median gain 0.7835 points and disease-balanced mean 1.6588 points.

The zero medians over the broader roster are therefore arithmetically explained
by the observed mixture of signs and exact zeros. This resolves the suspicion
that a positive supported mean must imply a positive all-cell median. It does
not validate the currently unusable fixed-roster bootstrap intervals.

Shrinkage is quantitatively substantial: among 460 off-diagonal all-outcome
pairs, median absolute raw gain is 0.0578056 AUC points versus 0.0000241152 after
shrinkage; maximum absolute values are 3.11111 and 0.000996763 respectively
(not necessarily the same pair). These are already AUC-point units. Raw and
shrunken views must be compared before interpreting the near-blank pair maps.

Atlas alignment is explicitly false: OPERA resolves to R_L, while the supplied
atlas metrics path identifies the J_L run 0bd42c2dff316708536b. This is a real
model-identity distinction, not a router inference bug. The clinical-model check
is absent from this report, so no clinical alignment conclusion follows.
