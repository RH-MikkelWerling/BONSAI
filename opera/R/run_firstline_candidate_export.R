# Isolated candidate export of the mapped branch used by downstream R scripts.
# It stops at the existing counts-only marker, before the later FROZEN branch.
# Usage: Rscript opera/R/run_firstline_candidate_export.R SCRIPT.R NEW_OUTPUT_ROOT SOURCE_WORKDIR

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Expected first-line script, output root and source working directory")
if (!nzchar(Sys.getenv("OPERA_FIRSTLINE_INPUT_CACHE", "")))
  stop("Set OPERA_FIRSTLINE_INPUT_CACHE to a verified loader cache; live SQL is not allowed for a release candidate")
script_path <- normalizePath(args[[1]], mustWork = TRUE)
root <- normalizePath(args[[2]], mustWork = TRUE)
source_workdir <- normalizePath(args[[3]], mustWork = TRUE)
run_dir <- tempfile(pattern = "firstline_candidate_", tmpdir = root)
if (!dir.create(run_dir, mode = "0700", showWarnings = FALSE))
  stop("Could not create isolated run directory")
Sys.chmod(run_dir, mode = "0700")
Sys.setenv(OPERA_FIRSTLINE_COUNTS_ONLY = "1",
           OPERA_FIRSTLINE_COUNT_DIR = file.path(run_dir, "counts"))
setwd(source_workdir)
completed <- FALSE
tryCatch(source(script_path, local = globalenv(), echo = FALSE),
         opera_firstline_counts_complete = function(condition) completed <<- TRUE)
if (!completed) stop("First-line script did not reach the mapped-export marker")
setwd(run_dir)
if (!exists("firstline_tbl_mapped", inherits = FALSE) ||
    !exists("df_expanded", inherits = FALSE)) stop("Mapped outputs missing")
mapped <- get("firstline_tbl_mapped", inherits = FALSE)
expanded <- get("df_expanded", inherits = FALSE)
if (!all(c("patientid", "firstline_date", "cohort_grouped", "cohort_fine") %in% names(mapped)))
  stop("Mapped output lacks required columns")
if (anyNA(mapped$patientid) || anyDuplicated(mapped$patientid))
  stop("Mapped output has missing/duplicate patient IDs")
expected_n <- Sys.getenv("OPERA_EXPECTED_FIRSTLINE_N", "")
if (nzchar(expected_n)) {
  expected_n_int <- suppressWarnings(as.integer(expected_n))
  if (is.na(expected_n_int) || nrow(mapped) != expected_n_int)
    stop("Mapped candidate count differs from OPERA_EXPECTED_FIRSTLINE_N")
}
if (!setequal(mapped$patientid, expanded$patientid) ||
    anyDuplicated(expanded$patientid)) stop("Expanded and mapped patient sets differ")
if (anyNA(mapped$firstline_date)) stop("Mapped output has missing first-line dates")
readr::write_csv(mapped, file.path(run_dir, "firstline_tbl_mapped_CANDIDATE.csv"))
readr::write_csv(expanded, file.path(run_dir, "df_expanded_CANDIDATE.csv"))
writeLines(c(
  "INTERNAL candidate only; not the production firstline_tbl_mapped_updated.csv.",
  "Exported at mapped_csv_export_population marker, before FROZEN branch.",
  "The mapped branch is the one consumed by supplied downstream scripts.",
  paste0("Seed policy: ", Sys.getenv("OPERA_FIRSTLINE_SEED_POLICY", "primary_cll_v4_mm_v5")),
  paste0("CLL seed: ", Sys.getenv("OPERA_FIRSTLINE_CLL_SEED")),
  paste0("MM seed: ", Sys.getenv("OPERA_FIRSTLINE_MM_SEED")),
  paste0("Treatment source: ", Sys.getenv("OPERA_FIRSTLINE_TREATMENT")),
  paste0("Original source working directory: ", source_workdir),
  "Compare patient IDs, dates, cohorts and arms with historical artifact before adoption."
), file.path(run_dir, "README_INTERNAL.txt"))
message("Candidate first-line export: ", normalizePath(run_dir))
