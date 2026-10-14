# Ascertainment methods and pipeline counts

Draft supplement: `paper/opera_ascertainment_supplement.tex`. It records the
implementation, not assumptions inferred from endpoint names. Pending choices
are visibly marked. No grading, treatment selection, or eligibility rule was
changed by this work.

The 30 September aggregate export audit covered all 87 registry outcomes and
found zero duplicate IDs, invalid event/index/censor ordering, missing dates,
invalid split labels, or split-date mismatches. This does not validate source
grading or observation eligibility. The cross-stage split checker has since
been changed to report overlap counts without patient-ID examples; sync that
change before exporting its JSON report from the server.

## Decisions needed before finalization

1. Establish the exact first-line artifact consumed by modelling: mapped CSV,
   expanded CSV, FROZEN RDS, or registry-fallback object. The mapped CSV enables
   CD20 monotherapy; the later FROZEN reconstruction disables it and replaces
   `firstline_tbl` without rebuilding the earlier mapped CSV.
2. Resolve whether arm-derived cohort labels and registry date fallback belong
   in the primary population. Arm rescue can assign grouped names to the fine
   column. Reconcile this with the canonical 24 fine labels used in Python.
3. Archive external DALY-CARE package/codebook versions and grading functions.
   The repository mixes v6-labelled local graders with v5 fibrinogen comments;
   it cannot certify the version or thresholds of externally sourced graders.
4. Reconcile administrative cutoff 2026-03-19 (outcome engine) versus 2026-03-04
   (second-line engine). Preserve the current exports until a protocol decision.
5. Confirm any-lab coverage windows and laboratory gating of IV/prescription
   infection calls. Do not describe them as analyte-specific or purely source
   coverage eligibility. The active lab engine does not rescue event cases
   around the coverage gate, despite an older comment stating that it does.
6. Review `leukocytosis_g2plus`: its call requests threshold 3. Also verify the
   prescription prophylaxis code `JO1EE01` (letter O) against the actual codebook.
   These are audit questions; neither was silently corrected.
7. Freeze production execution order: `build_composites.R` contains a duplicated
   block and exploratory sections elsewhere overwrite objects. Separate these
   from the final provenance record before publication.

## Second-pass findings (29 September 2026)

These findings supersede uncertainties in the list above where stated. They
describe the supplied source, not proof that existing server exports were made
with precisely this revision.

### Resolved from implementation and the reference standard

- Both `calculate_adverse_events_update.R` (line 275) and
  `identification_second_line_treatment_v5.R` (line 789) explicitly reload
  `firstline_tbl_mapped_updated.csv`; `evaluate_split_v5.R` does too. The CSV
  is written before the final registry-fallback object is constructed. That
  fallback is not subsequently exported by the supplied first-line script.
  Thus the supplied downstream pipeline uses the mapped/CD20-enabled branch,
  not the later raw FROZEN object or `firstline_date_final` fallback.
- CTCAE v6 defines laboratory leukocytosis at grade 3, with no grade 1 or 2
  category. Threshold 3 is therefore not intrinsically a wrong call; the
  `leukocytosis_g2plus` name is misleading. The external grader still needs
  checking, as does the relevance of the CTCAE exclusion of leukemia-related
  increases for this population.
- CTCAE v6 retains the fibrinogen LLN-fraction and baseline-relative grading
  structure. A v5 comment alone does not establish incompatible thresholds.
  The actual limitations here are omitted baseline-relative grading, boundary
  conventions, and unverified units for the fixed fallback/floor constants.
- The ordinary laboratory outcome calls use the non-grade-5 columns. Local
  proximity-to-death grade-5 variables are created but are not those calls'
  event criterion. Do not describe these outputs as attributable fatal toxicity.
- Prescription infection requires an episode spanning at least one day between
  dispensing dates; this is not prescribed days of supply. A single dispensing
  date cannot qualify. IV episodes likewise use observed administration span,
  not direct measurement of continuous antimicrobial treatment duration.
- The intended separate-script order is first-line export, main outcomes and
  second-line outcomes, source coverage gate, then composites. Composite
  intersections propagate component eligibility and the earliest censor.

Reference: [NCI CTCAE v6](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf).

### Additional issues established without a user decision

- `apply_coverage_gate()` is not audit-idempotent: rerunning an already-gated
  file writes an empty `__coverage_audit.parquet`, replacing the original
  exclusions. Canonical retained patients remain unchanged in that situation.
  Preserve original audit files; do not infer zero exclusions from a rerun.
- The co-occurrence constructor uses the full first-line/follow-up population,
  not the ordinary baseline/entry laboratory coverage mask or last-draw censor.
  TLS, Hy's-law-style and DIC-like names are absent from `COVERAGE_GATES`.
  The final instruction to gate these "with the rest" does not implement such
  a gate. Their negatives therefore have a different observation assumption.
- The TLS construction is a post-index, first-seven-day, two-axis laboratory
  proxy with approximated thresholds. It is not strict Cairo–Bishop diagnosis.
  Hy's-law-style output uses grade-2 bilirubin rather than an explicit >2x ULN
  rule and does not exclude cholestasis/alternative causes. DIC-like output is
  a two-axis derangement proxy, not an adjudicated or ISTH-scored DIC endpoint.
- `cut()` defaults to right-closed intervals. Exact cut-point values in local
  electrolyte/fibrinogen graders can be assigned to the more severe category
  even where the reference uses a strict inequality. This needs boundary tests,
  not a blanket claim of exact CTCAE implementation.
- Fibrinogen tests for existence of `NPU.FIB` but filters with `NPU.FIBR`.
  The external dictionary is needed to establish which symbols exist and the
  selected assays' units. This is a concrete inconsistency, not proof of a
  failed production run.
- The prescription exclusion contains `JO1EE01` (letter O), not `J01EE01`.
  IV prophylaxis exclusions are commented out. Neither was silently changed.
- Repeated composite construction and exploratory object replacement are
  reproducibility hazards, rather than questions the user needs to interpret.

### Remaining information/decisions needed from the study team

1. Supply the server production driver/execution order and provenance of the
   mapped CSV, population metadata and outcome exports. Also supply the
   external DALY-CARE grader/codebook sources or their pinned version. No
   patient-level data are required to answer this provenance request.
2. Confirm pretraining cutoff/population and whether downstream tuning/held-out
   patients' pre-cutoff records were included; state the intended evaluation
   claim (new treated episodes versus entirely unseen patients).
3. Are retrospectively rescued disease identities available at prediction
   time, or solely for cohort analysis? Should arm-derived grouped identities
   remain in the primary cohort, with fine subtype explicitly unresolved?
4. Should outcomes mean first post-index documentation, or incident/worsening
   toxicity? Current laboratory construction implements the former. Accept
   laboratory-only CTCAE-inspired proxies, or require fuller clinical criteria?
5. Confirm the laboratory units/reference conventions (especially fibrinogen)
   and whether source coverage dates were chosen from observed capture
   diagnostics. Provide those diagnostics if available.
6. Decide observation eligibility for antimicrobial and co-occurrence endpoints:
   retain any-lab gating for antimicrobial proxies, and require a monitoring
   mask for co-occurrence negatives? Should a single prescription qualify, and
   which prophylactic agents should be excluded from each antimicrobial source?
7. Confirm whether the two administrative cutoffs intentionally reflect source
   snapshots or should be harmonized. Composite minimum follow-up already
   propagates the earlier component cutoff.

No threshold, date, clinical exclusion or eligibility definition was changed
in this second pass. Those changes would alter the scientific target and need
an explicit decision and versioned regeneration.

## Study-team clarifications and upstream check

The study team clarified that pretraining used all patients' pre-2022 records,
that supervised splits depend on first-line date, and that retrospective disease
rescue is intended for cohort analysis only. The supplied SSL configuration
specifies an exclusive 2022-01-01 boundary; R outcome assignment uses tuning
2022-01-01 through 2022-12-31 and held-out from 2023-01-01. Server-resolved
configuration remains the final provenance record. Disease-conditioned routers
explicitly consume disease indices, so their label provenance must be checked
before claiming retrospective labels are analysis-only for every architecture.

Upstream package inspected at commit
`a0742b0de4d0984d7a8491bd103144236a1f27b9`:
https://github.com/RH-CLL-LAB/dalycare_package

- Existing CTCAE graders explicitly cite v5. The platelet grader uses a
  25 x 10^9/L grade-3/4 boundary, compared with 10 x 10^9/L in v6. Mixed
  implementations should be called CTCAE-based, with per-endpoint version and
  laboratory-only approximations documented, not uniformly v6-compliant.
- Upstream `ATC_AB.R` includes J01EE01 and maps it to Sulfotrim. Thus the active
  local exclusion JO1EE01 is a confirmed code typo; correctly coded J01EE01
  rows survive it. Its effect on exported event dates/counts needs server data.
  Correcting the typo is separate from the clinical decision to exclude this
  agent in all settings. Exact matching of `P02` likewise does not exclude
  detailed P02C codes selected by ATC_AB; use of prefix exclusion needs intent
  confirmation. No production definition was changed during this diagnosis.
- The inspected `clean_lab_values.R` has no fibrinogen-specific conversion.
  Local fibrinogen fixed constants assume g/L, but the grader does not assert
  the cleaned unit. Source-unit distributions and NPU.FIBR still need server
  verification. The published tree does not include load_npu_common or the
  server's load_dalycare_package.R entry point.
- Both administrative cutoff dates are literal constants in the supplied
  scripts, not calculated source maxima. They may reflect different extraction
  snapshots, as suggested by the study team; this cannot be established from
  these scripts alone. Do not extend either source's follow-up speculatively.

For unresolved monitoring eligibility, recommend source-specific observation
requirements for antimicrobial proxies rather than assuming laboratory draws
prove medication capture; co-occurrence negatives require an explicit adequate
monitoring rule. Compare eligibility/event counts before versioning any change.

## Count outputs (instructions)

For a counts-only run, sync `opera/R/identification_first_line_treatment_v2.R`,
`opera/R/run_firstline_counts_only.R`, and `tests/r/test_firstline_counts.R`.
In the server's original DALY-CARE R environment, from the repository root:

```bash
Rscript tests/r/test_firstline_counts.R
Rscript opera/R/run_firstline_counts_only.R \
  "$PWD/opera/R/identification_first_line_treatment_v2.R" \
  "$PWD/data/results/firstline_count_runs"
```

Do not proceed to the second command if the test fails. The launcher uses a
new isolated working directory and stops the first-line script after its final
mapped-cohort count, before its production CSV/RDS exports and exploratory
plots. It prints the completed count-file path. Its output layout is:

```
data/results/firstline_count_runs/firstline_counts_<unique>/counts/<timestamp>_<pid>/
  pipeline_counts.csv
  pipeline_breakdowns.csv
  README.txt
```

The script still requires the external DALY-CARE loader, clinical seed tables,
and treatment data; this instrumentation does not make the exploratory script
a standalone production pipeline. Do not source the original script directly
solely to collect counts: its ordinary mode writes production exports.

`pipeline_counts.csv` distinguishes raw rows, unique patients, repeated-patient
rows, counting unit, branch and stage. `pipeline_breakdowns.csv` contains marginal
source, provenance, cohort and primary-arm counts; no patient IDs are exported.
Record/episode stages naturally repeat patients. Counts are INTERNAL and exact,
not disclosure-cleared. Adjacent stages from different branches/populations are
not subtractable as CONSORT exclusions.

The helper records actual make_firstline search/gap/CD20 settings in stage notes
when called with an audit label. Only production/pass-two calls are instrumented
inside the function; repeated exploratory sensitivity calls remain unlabelled.
The export snapshot is captured separately from the later FROZEN raw object.
Downstream split and valid-follow-up counts must come from outcome/metadata
exports, not reconstructed from first-line ascertainment counts.

## First-line count reconciliation (2026-09-30 server export)

The supplied internal `pipeline_counts.csv` gives 78,679 patients in the
loaded patient table and 78,668 with a diagnosis index. These are two separately
counted tables; the 11-person difference should not yet be labelled an exclusion
without an ID-level comparison. Of the 78,668 diagnosis-indexed patients,
38,479 have a first-line episode and 40,189 do not. Among the 38,479,
37,291 are assigned a disease in pass 1 and 1,188 remain unclassified.
Pass 2 classifies 198 of those 1,188, leaving 990 unclassified. Thus
37,291 + 198 = 37,489 are classified before exclusion of seven
`EXCLUDE_SECONDARY` patients, leaving 37,482 in the mapped CSV export
population. No patients were added by arm rescue in this run.

This is not yet an end-to-end CONSORT: the earlier population metadata has
37,494 patients, 12 more than this first-line rerun, and its supervised outcome
export has 37,467. The earlier 27-person no-follow-up difference and
29,108 / 1,967 / 6,392 split counts must not be subtracted from the new
37,482. Determine whether the 12-person difference reflects changed inputs,
first-line script version, or output lineage, then regenerate/compare downstream
metadata and outcome eligibility using the same first-line artifact. The paper
figure now uses only this run's internally reconciling ascertainment counts;
downstream nodes remain placeholders until aligned.

The 48,172 patients with treatment records and the 44,764 with indication-
eligible records are useful process diagnostics, but neither is a mutually
exclusive flow node: they come from different row filters/branches and cannot
be used as a simple subtraction from the diagnosis-indexed population. The
`pipeline_breakdowns.csv` cohort, arm and source counts are marginal, not
attrition categories, and contain exact small cells requiring disclosure review.

## Outcome-source follow-up (2026-09-30 static pass)

The R scripts are not yet a verified, uniform CTCAE-v6 implementation. The
NCI v6 reference is from 2025, not January 2026. Locally implemented v6-labelled
lab graders coexist with external DALY-CARE graders that were documented as
v5-based. Fibrinogen's v6 LLN/baseline rule resembles v5, but the local grader
omits the abnormal-baseline axis and assumes g/L for its missing-LLN fallback
and grade-4 floor without asserting units. CTCAE-based/laboratory proxy is the
accurate description until each endpoint's code, units, baseline convention,
threshold boundaries and external grader revision are verified. Changing the
version label alone would conceal potential differences; regrading must be a
versioned final-label decision with before/after event-count audits.

Two local NPU existence checks used names different from their actual filter
variables (`NPU.NA` versus `NPU.NAT`, `NPU.FIB` versus `NPU.FIBR`). Their checks
were corrected to the variables used and now fail early if unresolved. The
entire composite-builder block was duplicated and executed twice; the second
copy was removed. Neither change intentionally alters thresholds or prior v4
exports. Native R execution remains to be tested in the server environment.

Two label-affecting questions remain open: the active prescription infection
filter uses the mistyped `JO1EE01`, so `J01EE01` is not excluded, and the
`leukocytosis_g2plus` file is produced using a grade-3 threshold. NCI v6 has
no grade-2 leukocytosis category and advises against using the term for rises
due to leukemias, growth factors or steroids. Do not silently rename/drop these
endpoints or change the prescription exclusion in an existing release; quantify
their impact and record the clinical decision for a new label version.

For provenance, the old mapped CSV reportedly resides under
`/ngc/people/mikwer_r/`; locate and hash the exact artifact before any rerun.
The first-line script also consumes external DALY-CARE `load_dataset` tables,
an external loader, a treatment Parquet and CLL/MM seed Parquets. Hashing today's
inputs documents the current state but cannot prove which historical versions
made the old CSV. Unless an older source manifest/snapshot exists, cohort ID,
index-date, disease and arm comparison against an isolated candidate export is
the strongest empirical lineage check. A 12-person difference in total counts
alone does not bound how many shared patients changed index or label.

## Verification limits

The repository environment has no Rscript executable or DALY-CARE data/package.
Native R execution and population count reconciliation require the server.
LaTeX build and static instrumentation review can be performed locally.
