# Copy an ungated outcome panel, then apply the explicitly adjudicated SMR gate.
# Usage: Rscript opera/R/run_coverage_gate_versioned.R RAW_DIR NEW_DIR GATE_SCRIPT
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L)
  stop("Usage: Rscript run_coverage_gate_versioned.R RAW_DIR NEW_DIR GATE_SCRIPT")
raw_dir <- normalizePath(args[[1]], mustWork = TRUE)
new_dir <- args[[2]]
gate_script <- normalizePath(args[[3]], mustWork = TRUE)
if (!dir.exists(raw_dir) || !file.exists(file.path(raw_dir, "infections_IV_based.parquet")))
  stop("RAW_DIR lacks infections_IV_based.parquet")
if (file.exists(file.path(raw_dir, "coverage_gated_outcomes.csv")))
  stop("RAW_DIR is already gated; use the sealed ungated panel")
if (file.exists(new_dir)) stop("NEW_DIR must not already exist")
files <- list.files(raw_dir, all.files = TRUE, no.. = TRUE, full.names = TRUE)
if (!length(files) || any(dir.exists(files))) stop("RAW_DIR must contain files only")
if (!dir.create(new_dir, recursive = TRUE, showWarnings = FALSE, mode = "0700"))
  stop("Could not create NEW_DIR")
new_dir <- normalizePath(new_dir, mustWork = TRUE)
if (!all(file.copy(files, new_dir, overwrite = FALSE)))
  stop("Incomplete copy into NEW_DIR; inspect it before retrying")
Sys.chmod(new_dir, mode = "0700")
OUTPUT_DIR <- new_dir
Sys.setenv(OPERA_VERSIONED_COVERAGE_GATE = "1")
source(gate_script, local = TRUE)
Sys.unsetenv("OPERA_VERSIONED_COVERAGE_GATE")
writeLines(c(paste0("ungated_source=", raw_dir),
             paste0("smr_coverage_start=", format(SMR_COVERAGE_START)),
             "smr_date_basis=national_reporting_start_not_completeness"),
           file.path(new_dir, "coverage_gate_provenance.txt"))
message("Versioned gated panel: ", new_dir)
