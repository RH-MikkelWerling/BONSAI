# TLS and outcome-date audit boundary (INTERNAL; 2026-10-03)

## What the dates mean

There is no TLS-specific calendar coverage start in
`opera/R/lpr3_coverage_gate.R`. Its 2019-02-03 LPR3 go-live plus 30-day buffer
and 2018-05-15 SMR start apply to different LPR/SMR outcomes, not TLS. The SMR
start is still marked as needing source-capture justification.

The separate pair 2026-03-19 and 2026-03-04 are administrative **follow-up
end** constants in `calculate_adverse_events_update.R` and
`identification_second_line_treatment_v5.R`, respectively. The supplied code
does not explain why they differ. Their difference must not be described as a
TLS coverage period or resolved by assuming one date is a typo. Record the
actual source snapshot/end-of-completeness date and quantify how many events
and censor dates fall in the 15-day difference before harmonizing them. A
common earlier cutoff is a conservative sensitivity, not automatically the
final policy.

The outcome script's overall-survival export previously converted a recorded
death *after* 2026-03-19 into an event *on* 2026-03-19 using `pmin()`. That is
incorrect administrative censoring. The exporter now retains a death event
only when the recorded date is on or before the cutoff and fails if survival
dates are inconsistent. Before rerunning, aggregate-check whether any frozen
first-line indices occur on/after either administrative cutoff; an index
beyond the chosen cutoff cannot enter a valid follow-up risk set.

## Current endpoint disposition

- `tumor_lysis_syndrome`: **retire from the primary panel**. The historical
  implementation accepts two abnormal axes within a seven-day co-occurrence
  window in the first seven post-index days, does not enforce the ordinary
  laboratory monitoring eligibility for negatives, and uses approximate
  thresholds. It is not a CTCAE-v6 TLS diagnosis.
- `tls_laboratory_early_v1`: **exploratory candidate only**. Absolute lab
  thresholds and a complete same-day four-axis panel are required. Its label
  omits baseline-relative alternatives and pre-treatment TLS and is therefore
  not a complete CTCAE-v6 LTLS definition.
- `tls_laboratory_with_kidney_early_v1`: **exploratory candidate only**. It
  requires the above lab signal and same-day creatinine above reported ULN,
  but does not adjudicate baseline renal impairment, arrhythmia or seizure.
  Do not label it clinically adjudicated TLS.

The NCI CTCAE-v6 TLS definition allows the biochemical abnormalities in the
three days before or seven days after chemotherapy start and includes
baseline-relative alternatives; clinical TLS additionally requires clinical
manifestations. The candidate deliberately targets a narrower early
post-index signal. See the official [CTCAE-v6 definition](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf).

## Bounded next check

After the definitive first-line mapped/expanded files are frozen, rebuild
`overall_survival.parquet` from that index. Run the isolated TLS lab extract
and candidate exporter in `docs/candidate_tls_v1_runbook.md`, both to new
protected directories. Review NPU/unit mapping, nonnumeric values, complete
same-day panel counts and 30-day event support by train/tuning split. Keep
patient-level data on the server. If support or units fail, stop and retain
both candidates as exploratory/held; do not substitute an older survival file
or relax negative-ascertainment rules merely to boost N.

Separately, establish the administrative end date from the underlying source
snapshot and compare the 2026-03-04 versus 2026-03-19 censor/event impact.
Then freeze one documented follow-up policy across the new outcome and
second-line release, with any endpoint-specific exceptions justified by its
source capture. This decision is pending server-side evidence.
