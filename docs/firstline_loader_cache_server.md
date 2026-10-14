# First-line loader cache (secure server only)

This freezes the `patient`, `t_dalycare_diagnoses`, `Codes_all` and `ICD10.*`
objects produced by the DALY-CARE loader. It is an INTERNAL patient-level
artifact: keep it on approved storage, outside Git and outside syncs to a
personal computer. A complete cache is reused; an incomplete or changed cache
fails closed. It does not freeze the CLL/MM seeds or treatment Parquet, which
must be copied/hashed separately.

From the repository root on the server, choose a **new** directory under an
approved parent. Do not point at a previous cache or broad existing directory.

```bash
umask 077
CACHE_ROOT=/ngc/people/mikwer_r/firstline_release_inputs_v1
mkdir -p "$CACHE_ROOT"
Rscript opera/R/freeze_firstline_loader_inputs.R "$CACHE_ROOT/sql_loader_cache"
```

If that cache already exists, the same command verifies the three files against
its manifest and does not call `load_dataset()` again. If the directory exists
without a manifest, it stops; investigate rather than rerunning into it.

For an isolated first-line **candidate** run using the cache:

```bash
export OPERA_FIRSTLINE_INPUT_CACHE="$CACHE_ROOT/sql_loader_cache"
export OPERA_FIRSTLINE_SEED_POLICY=primary_cll_v4_mm_v5
export OPERA_FIRSTLINE_CLL_SEED="$PWD/opera/R/medicine data/data/output tables/CLL_V4_100lines.parquet"
export OPERA_FIRSTLINE_MM_SEED="$PWD/opera/R/medicine data/data/output tables/MM_wide_V5_100lines.parquet"
export OPERA_FIRSTLINE_TREATMENT="$PWD/opera/R/medicine data/data/output tables/table_treatment_clean_2026_09_10.parquet"
mkdir -p data/results/firstline_candidate_runs
Rscript opera/R/run_firstline_candidate_export.R \
  "$PWD/opera/R/identification_first_line_treatment_v2.R" \
  "$PWD/data/results/firstline_candidate_runs" \
  /ngc/people/mikwer_r
```

The last argument retains the historical working directory for the direct
Parquet reads in the unfrozen mode; with the cache enabled, all three copied
Parquet paths above are mandatory, so live and frozen inputs cannot be mixed.
Check that the September treatment file actually exists at the path above; if
it lives elsewhere, set the absolute path to that file instead. The treatment
Parquet is **not inside** the loader cache; the cache only contains patient,
diagnosis, and ICD-10 codebook objects. Changing the treatment file does not
invalidate a verified cache. If the loader source or underlying SQL extraction
has changed, create a new cache directory and compare its manifest with the
old one; never overwrite the old cache.

The CLL path above is the clinician-confirmed final assignment, not the earlier
`CLL_wide_V4_100lines_excl_cd20.parquet` assignment. The current code requires
those exact seed basenames and has no silent fallback.
Use the collaborator copies only after confirming that they are the intended
effective CLL/MM seed and treatment files. The seed adapter accepts either a
line-level `date_start_line` or a row-level `date_start`: in the latter case it
uses the earliest dated `line == 1` row per patient, with no `row_acc` filter.
Record which rule was used; V5 can change seed presence and some dates by over
one year, so its adoption remains provisional pending the cohort-level check.
This is not a final release: the
first-line policy still needs review. The run additionally emits
`unclassified_diagnostic_flags_INTERNAL.csv`, an aggregate-only breakdown of
pass-1 failures by diagnosis support and pass-2 recovery. `specific_any` means
mapped to an includable disease code, *not* clinically confirmed. The cache
manifest records UTC creation time, row counts and SHA-256
hashes. Separately record the database name, schema/table/view, loader query,
extraction timestamp and transaction/backup snapshot ID if available. Without
a database snapshot, the two `load_dataset()` calls may observe slightly
different live states; this limitation must be stated. Do not assume that a
cache created once proves the old v4 inputs were identical.

The first-line script's cache switch requires an explicit
`OPERA_FIRSTLINE_INPUT_CACHE`; with the variable set, it will not silently
fall back to live SQL. The loader source is also checked against the cached
loader hash. For complete release locking, use the snapshot tool to copy/hash
the direct Parquets, upstream code and this cache into a separate versioned
release area after reviewing its dry run.
