# Run the instrumented first-line reconstruction in an isolated directory.
# Usage: Rscript opera/R/run_firstline_counts_only.R /absolute/path/to/identification_first_line_treatment_v2.R /absolute/output/root
# This executes the original data-loading and ascertainment logic, but exits
# after the final mapped-cohort count and before its production CSV/RDS exports.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Expected first-line R script path and output root")
}
script_path <- normalizePath(args[[1]], mustWork = TRUE)
output_root <- args[[2]]
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
output_root <- normalizePath(output_root, mustWork = TRUE)
run_dir <- tempfile(pattern = "firstline_counts_", tmpdir = output_root)
if (!dir.create(run_dir, showWarnings = FALSE)) {
  stop("Could not create a unique isolated run directory")
}
run_dir <- normalizePath(run_dir, mustWork = TRUE)
setwd(run_dir)
Sys.setenv(
  OPERA_FIRSTLINE_COUNT_DIR = file.path(run_dir, "counts"),
  OPERA_FIRSTLINE_COUNTS_ONLY = "1"
)
message("[1L count] Isolated working directory: ", run_dir)

completed <- FALSE
tryCatch(
  source(script_path, local = globalenv(), echo = FALSE),
  opera_firstline_counts_complete = function(condition) {
    message(condition$message)
    completed <<- TRUE
  }
)
if (!completed) {
  stop("First-line script ended without the counts-only completion marker")
}
count_files <- list.files(
  file.path(run_dir, "counts"), pattern = "^pipeline_counts[.]csv$",
  recursive = TRUE, full.names = TRUE
)
if (length(count_files) != 1L) {
  stop("Expected exactly one completed pipeline_counts.csv")
}
counts <- utils::read.csv(count_files[[1]], stringsAsFactors = FALSE)
required_stages <- c(
  "loaded_patient_population", "pass1_assigned_cohort",
  "after_secondary_cohort_exclusion", "mapped_csv_export_population"
)
if (!all(required_stages %in% counts$stage)) {
  stop("Counts file lacks required first-line stages")
}
message("[1L count] Complete aggregate counts: ", count_files[[1]])
message("[1L count] The first-line script's production CSV/RDS exports were skipped")
