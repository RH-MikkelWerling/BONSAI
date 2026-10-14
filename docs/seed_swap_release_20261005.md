# Clinician-directed CLL/MM seed reversal (internal, 5 October 2026)

The clinician-confirmed primary first-line seed pair is **CLL
`CLL_V4_100lines.parquet` / MM `MM_wide_V5_100lines.parquet`**. The final CLL
shell assignment supersedes the earlier `CLL_wide_V4_100lines_excl_cd20.parquet`
assignment. The prior V4/V4 pair is the historical comparator only. Hold the
loader cache, cohort-identification logic, and split policy fixed where possible
for the first comparison. The treatment input also changes to
`table_treatment_clean_2026_09_10.parquet`, so the combined candidate cannot
isolate a seed-only effect. Compare old/new treatment Parquets separately.

The first-line script now requires explicit seed and treatment paths and
defaults to `OPERA_FIRSTLINE_SEED_POLICY=primary_cll_v4_mm_v5`. A deliberate V4
sensitivity run can use `historical_v4` with explicit V4 paths. The second-line
script requires the same CLL V4 / MM V5 seed paths and selects each patient's
earliest dated line-2 row, accommodating a row-level `date_start` representation.
The outcome script requires `OPERA_FIRSTLINE_MAPPED_CSV` and a new
`OPERA_OUTCOMES_DIR`, so it cannot silently reuse the historical V4/V4 cohort
or overwrite its outcome directory. Its older error text calling the historical
cohort V4/V4 is contextual, not a new seed check.

## Gate before any downstream rebuild

1. Confirm the two exact seed Parquets exist, their line/date schemas, patient
   identifiers and hashes. Confirm the September treatment Parquet and verify
   the SQL loader cache. The treatment Parquet is not part of that cache;
   rebuild the cache only if its SQL/loader inputs changed. Do not set the old
   `OPERA_EXPECTED_FIRSTLINE_N=37540` for the new candidate.
2. Produce a versioned mapped/expanded CLL V4 / MM V5 first-line candidate with
   `opera/R/run_firstline_candidate_export.R`. Compare it with the reviewed
   V4/V4 mapped CSV using `opera.run.compare_firstline_releases`. Run
   `opera.run.audit_firstline_seed_alignment` against both selected seed Parquets.
   Review cohort membership, changed index dates (>30 days and >365 days),
   cohort/arm changes, missing seed dates, and CLL/MM source transitions.
3. Only after adopting that candidate, rerun second-line/TTNT, outcomes and
   index-derived feature matrices with the **same mapped CSV**. Old outcome
   comparisons that assume identical patient IDs/index dates cannot isolate
   a procedure-code effect across the seed change.
4. Re-run structural outcome audit, source/date/coverage audit, administrative
   cutoff review and candidate lab comparison on the new risk set. Inspect
   creatinine CTCAE-v6-candidate versus >1.5x-baseline support/overlap. Resolve
   remaining grade-name mismatches and missing composites, then apply the
   panel-decision rubric; no held-out-performance-based endpoint selection.

The previous 37,540 first-line and 37,512 outcome-candidate counts, as well as
the 81-file structural audit, are **not** new-release counts. Do not launch
finetuning or call the outcomes final until the above contracts and exact
cutoff/source policy have been signed off.
