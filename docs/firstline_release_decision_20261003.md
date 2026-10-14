# First-line primary-cohort decision (INTERNAL; 2026-10-03)

> **Superseded on 5 October 2026 for seed versions.** Following clinician
> review, the next primary release must use the final
> `CLL_V4_100lines.parquet` and `MM_wide_V5_100lines.parquet`, not the V4/V4
> pair described below. The 37,540-person count and all downstream outcomes
> from the V4/V4 release are historical comparators, not final-data counts.
> The updated release also selects `table_treatment_clean_2026_09_10.parquet`;
> consequently this is not a seed-only comparison. Keep cohort policy fixed
> while rebuilding and auditing the mapped/expanded first-line exports. Do not infer the new cohort
> size or index dates from this historical note.

## Decision

Accept the existing mapped-export branch as the primary first-line cohort
policy, using the frozen V4 MM line-start seed, frozen CLL V4 seed, frozen
September 2026 cleaned-treatment table, and the verified first-line SQL loader
cache. Do not add patients from a registry date alone. Preserve the V5 MM
candidate as a sensitivity comparison, not the primary release. Preserve the
existing episode, indication, diagnosis, seed-priority and secondary-exclusion
rules unchanged for this release. This is a decision to stop changing the
cohort policy, not a claim that every omitted registry patient was adjudicated.

The V4 mapped-export **counts-only** run reports 37,540 patients. Its flow is
78,668 diagnosis-indexed, 38,618 with a pass-1 first-line index, 37,337
classified in pass 1, 212 additionally classified in pass 2, 1,069 remaining
unclassified, 37,549 before nine secondary-cohort exclusions, and 37,540
after exclusion. The 37,547-person count belongs to the V5 candidate. The
matching mapped and expanded files still need to be exported and checksummed
before 37,540 can be called an immutable server-side data artifact.

## Decision evidence and limits

- V4 MM dates are substantially closer to DaMyDa first-line dates than V5 in
  the changed-date comparison. The two may share upstream provenance, so this
  is consistency evidence rather than independent clinical validation.
- The registry-date audit identified omitted LYFO patients, but most of the
  large `no_cleaned_treatment_row` category do not have an affirmative
  chemotherapy flag. Specific immunotherapy and chemotherapy-regimen entries
  explain smaller subsets, not a credible thousand-person rescue. A populated
  field and a registry date do not establish an administered, pipeline-eligible
  first-line regimen. Leave the residual affirmative-treatment subset as a
  documented QC/sensitivity item; it does not alter the primary rule.
- The one-day source/registry date representation difference remains a known
  provenance limitation. No blanket date correction is applied.
- The 37,558 versus 37,547 earlier seed-gate difference and the V4/V5 MM
  comparison remain historical diagnostics, not competing final releases.

## Release sequence on the secure server

1. Confirm `OPERA_FIRSTLINE_INPUT_CACHE`,
   `OPERA_FIRSTLINE_CLL_SEED`, `OPERA_FIRSTLINE_MM_SEED`, and
   `OPERA_FIRSTLINE_TREATMENT` resolve to the verified loader cache, final
   CLL V4 and MM V5 files, and September 2026 treatment file. Do not let the
   script fall back to live SQL or unspecified collaborator paths.
   Record their absolute paths and SHA-256 hashes, plus the first-line script,
   loader, wrapper, cache manifest, git commit/diff, R/package versions, UTC
   execution time and the chosen clinical-policy text. Keep source snapshots,
   hashes and audit CSVs on protected server storage.
2. Run `opera/R/run_firstline_candidate_export.R` into a new private output
   root, never an existing release path. This executes the mapped branch and
   exports both mapped and expanded files from the same run, with pipeline
   counts. Assert one nonmissing unique patient ID per file, identical patient
   sets, valid index dates and nonmissing cohort/arm fields as applicable.
   Record the exact mapped patient count. Recheck the candidate-to-prior
   comparison and expected V4/V5 difference; do not infer the count.
3. Only after those gates pass, copy the reviewed artifacts to a new
   versioned, access-controlled release directory. Hash the copies and compare
   them to the source hashes. Record a manifest with source paths, hashes,
   execution timestamp, policy, exact counts and audit report paths. Do not
   overwrite historical CSVs or use the later exploratory `FROZEN` branch.
4. Treat this mapped file as the frozen **cohort/index source of truth** for
   downstream products. Rebuild second-line/TTNT, outcomes, observation,
   temporal splits and fine-tuning inputs against it. Check unique IDs,
   index-time leakage, split isolation and labelled-patient coverage before
   supervised training.
5. Generate Table 1 from the resulting frozen included-patient manifest and
   frozen split labels using `opera.run.export_table1`. The Table 1 denominator
   should reconcile to the first-line release, or an explicitly documented
   analysis-population restriction. Freeze the exact internal aggregate,
   disclosure-reviewed export, provenance and manuscript table together.
   Do not call Table 1 frozen merely because the first-line cohort is fixed.

Pretraining on the separate frozen MEDS release can continue independently.
The RKKP registry extracts used for the diagnostic should have their SQL
definitions/extraction times recorded; they are not inputs to the primary
cohort unless the policy is changed in a later release.
