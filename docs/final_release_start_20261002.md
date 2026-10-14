# Final MEDS / first-line release: first gate (INTERNAL)

The new MEDS build is complete, but its path, builder commit and vocabulary
contract have not yet been supplied. Do not launch pretraining or supervised
training against a guessed directory. The old `weightinggroups` config points
at `daly_care_t100_joined_bins_harmonized_v4` and a v4 DAPT checkpoint; a
different token-ID map or numeric-bin scheme requires fresh pretraining or a
documented adaptation, not silent checkpoint reuse.

## First-line branch issue

`identification_first_line_treatment_v2.R` is a research/exploratory script,
not a single-pass production driver. It reads the DALY-CARE loader and several
direct treatment/seed Parquets, constructs a mapped branch, then later runs a
different `FROZEN` first-line pass (`ACCEPT_CD20 <- FALSE`) before writing the
mapped CSV from the earlier branch. It subsequently builds an RKKP date-fallback
object in memory without exporting it. `seed_MM` is also assigned twice, so
the first MM seed Parquet is not the effective one. Downstream outcome and
second-line scripts read `firstline_tbl_mapped_updated.csv`, not the later
fallback object. Do not run the entire script in an old working directory.

The new `opera/R/run_firstline_candidate_export.R` stops at the mapped-count
marker and writes only **CANDIDATE** mapped/expanded CSVs into a fresh isolated
directory, with counts. This preserves existing exports. Compare it to the old
mapped CSV on the server by ID, index date, cohort and arm. Then decide whether
to keep the mapped/CD20 policy or explicitly adopt a different clinical policy
and implement one clean production driver. The old 37,494 versus 37,482 count
difference is not enough to identify changed patients or changed dates.

The detailed comparison now shows 280 shared first-line date changes (261 over
30 days), 431 source changes, 173 primary-arm changes, 29 fine-cohort changes
and 15 grouped-cohort changes. CLL is prominent but not exclusive. The source
script binds CLL and MM line-1 seeds, chooses one by patient and date, then
joins seed dates to episodes by patient ID without checking the eventual
disease cohort. A CLL seed may therefore affect a non-CLL-labelled patient.
`opera.run.audit_firstline_seed_alignment` is a **counts-only diagnostic** for
this cross-cohort mechanism; it does not change the clinical definition. Run
it on the server with the candidate mapped CSV and the *effective* CLL and MM
seed Parquets:

```bash
python -m opera.run.audit_firstline_seed_alignment \
  --mapped /absolute/path/to/firstline_tbl_mapped_CANDIDATE.csv \
  --old-mapped /ngc/people/mikwer_r/firstline_tbl_mapped_updated.csv \
  --cll-seed '/ngc/projects2/dalyca_r/andkat_r/medicine data/data/output tables/CLL_wide_V4_100lines.parquet' \
  --mm-seed /ngc/projects2/dalyca_r/mikwer_r/MM_V4_5lines.parquet \
  --output "$PWD/data/results/firstline_seed_alignment_$(date -u +%Y%m%dT%H%M%SZ)_INTERNAL.json"
```

The optional historical mapped CSV comparison reports changes by selected
current seed, old-to-new `firstline_source` transitions and old/new seed-date
presence/change. This is necessary because “no current selected seed” can
include people who had a seed in the historical export. The latest aggregate
comparison had 193/280 changed dates with a current CLL seed, 3 with MM and
84 with no current selected seed; 375/431 source changes were in that last
category. The follow-up showed all 375 transitions were `clinical_seed` to
`inferred`, and 376 patients with no current selected seed had an old seed date
that is now absent. This strongly implicates changed seed input/provenance,
although an older CLL seed snapshot/hash is still needed to prove the exact
file-level change. The audit code was corrected because `NaT != NaT` inflated
the earlier `n_seed_date_changed_shared` for no-seed patients; do not cite that
earlier value. Re-run the corrected audit before using its seed-change totals.

## Snapshot gate: code AND data

Use `opera/run/snapshot_release_inputs.py` on the secure server. It accepts an
explicit JSON list of absolute files/directories, dry-runs by default, checks
free space, refuses symlinks/overwrite/nested targets, copies every listed
file, and verifies SHA-256 hashes. It is not an automatic dependency tracer:
the upstream `load_dataset()` function resolves paths outside this repo, and
the original tabular feature generator has not yet been identified. Those
resolved paths must be added before calling a snapshot complete.

The spec must include, at minimum:

- Final MEDS sequence directory **and** builder config/commit, vocabulary,
  numeric bins, time encoding and source manifests.
- First-line R script, external `load_dalycare_package.R`, installed DALY-CARE
  function/codebook sources, `patient`, diagnosis, treatment, CLL/MM seeds and
  RKKP tables actually resolved by the loader.
- Outcome, second-line, LPR3 coverage and composite scripts, plus their raw
  source tables and code lists.
- Original tabular feature generator and source inputs, existing matrix and
  its preprocessing profile. The repo's `prepare_tabular_features.py` filters
  an **already generated** matrix; it does not regenerate its EHR history.

`opera.run.audit_firstline_input_files` provides a concrete first-line input
inventory before the large copy. It hashes the R script, loader and four
direct Parquet reads, including the MM seed that is read then overwritten.
It lists the loader-resolved `patient`, `t_dalycare_diagnoses` and ICD-10
codebook paths as unresolved until explicitly supplied. These are **effective
inputs before the mapped-export marker**; the later exploratory RKKP loads
do not affect that mapped CSV. Run from the repository root:

```bash
python -m opera.run.audit_firstline_input_files \
  --source-workdir /ngc/people/mikwer_r \
  --script "$PWD/opera/R/identification_first_line_treatment_v2.R" \
  --loader /ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R \
  --output "$PWD/data/results/firstline_inputs_$(date -u +%Y%m%dT%H%M%SZ)_INTERNAL.json"
```

After inspecting the loader's actual resolution, rerun with repeated
`--known-table patient=/absolute/path` for file-backed inputs or
`--sql-table patient=DATABASE.SCHEMA.TABLE` for SQL-backed inputs. The latter
records the declared SQL identifier but **not** its contents; add a genuine
`--sql-snapshot-id` if one exists. SQL table name, executed query/view version,
extraction time, row count and snapshot/backup ID must be recorded separately.
Otherwise the inventory reports `sql_live_unfrozen`. Hashes from **today** only establish
today's state; proving the CLL seed changed since v4 requires an older copy,
manifest or archived hash. The first-line comparison's old-only/new-only and
changed-by-cohort breakdowns can identify whether the observed differences
cluster in CLL even if old input hashes are unavailable. Those breakdowns have
small cells and are INTERNAL, not disclosure-cleared.

Use one new snapshot root, preferably on the same protected storage as the
original tables. Preserve ACLs and do not transfer patient data. If a large
raw dataset is already immutable and storage duplication is infeasible, record
the exception, exact path, immutable snapshot ID and checksums rather than
claiming it was copied. A manifest of only today's hashes cannot establish
the historical v4 source version.

Example spec (replace paths after discovering them on the server):

```json
{
  "version": 1,
  "items": [
    {"name": "final_meds", "path": "/absolute/final/meds/directory"},
    {"name": "firstline_script", "path": "/absolute/repo/opera/R/identification_first_line_treatment_v2.R"},
    {"name": "dalycare_loader", "path": "/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R"}
  ]
}
```

```bash
python -m opera.run.snapshot_release_inputs --spec /secure/path/release_inputs.json \
  --output-dir /secure/new/release_snapshot
# Inspect the file count, total bytes and resolved paths first. Only then:
python -m opera.run.snapshot_release_inputs --spec /secure/path/release_inputs.json \
  --output-dir /secure/new/release_snapshot --execute
```

## Immediate jobs, in order

1. Obtain the final MEDS path, builder command/config and original tabular
   feature generator path; fill the dependency spec. Inspect old first-line
   artifact and server loader resolution. Snapshot inputs before rerunning.
2. Export a candidate mapped first-line cohort, not to an old production path:

   ```bash
   mkdir -p data/results/firstline_candidate_runs
   Rscript opera/R/run_firstline_candidate_export.R \
     "$PWD/opera/R/identification_first_line_treatment_v2.R" \
     "$PWD/data/results/firstline_candidate_runs" \
     "/ngc/people/mikwer_r"
   ```

   The third argument is the **historical source working directory** used to
   resolve `../../projects2/...` Parquet paths. Verify it on the server before
   running; do not substitute the repository or output directory blindly.

   Then run `opera.run.compare_firstline_releases` on the historical mapped
   CSV and the candidate CSV; it writes aggregate-only overlap, date, cohort
   and arm changes (plus input hashes) to a new JSON file. The historical
   path must first be located on the server.

   ```bash
   python -m opera.run.compare_firstline_releases \
     --old /absolute/path/to/old/firstline_tbl_mapped_updated.csv \
     --candidate /absolute/path/to/firstline_tbl_mapped_CANDIDATE.csv \
     --output /absolute/new/internal/firstline_comparison.json
   ```

3. Compare old/candidate cohorts and resolve branch policy. Regenerate
   population metadata, second line, outcome parquets, coverage gates and
   composites in one new versioned directory from the chosen mapped CSV.
   Do **not** combine a new first-line CSV with old outcomes.
4. Rebuild tabular feature matrices from the same frozen source snapshot and
   first-line index dates; then run `prepare_tabular_features.py` as a separate
   profile filter. Check all feature timestamps are before prediction index,
   identical split/ID definitions and whether RKKP fields are deliberately
   included or excluded. The MEDS change alone does not logically require
   changing a stable pre-index EHR feature definition, but rebuilding from
   the same final source release is the cleanest comparison here.
5. Rerun the corrected isolated TLS lab extract and candidate exporter per
   `docs/candidate_tls_v1_runbook.md`. Review unit/NPU inventory, nonnumeric
   values, complete-panel eligibility and train/tuning counts before adding
   TLS to any training roster. Keep TLS exploratory meanwhile.
6. Freeze the final outcome panel and source-methods appendix. Audit the new
   release, then begin fresh pretraining with the final MEDS vocabulary and
   pre-2022 cutoff. Do not use held-out performance for panel selection.

## Methods record to write from frozen sources

For each upstream script/function record: exact source commit/hash; input
tables and date coverage; patient linkage and exclusions; index derivation;
cohort and treatment classification (including seed/RKKP rescue); lab cleaning,
NPU mapping and unit conversion; endpoint thresholds and observation/censoring;
split/cutoff logic; output schema; and known approximations. Distinguish the
all-patient pre-2022 MEDS pretraining population from the first-line-treated
supervised population. Clinical prose should say CTCAE-based/CTCAE-inspired
where symptoms, interventions or adjudication cannot be reconstructed.
