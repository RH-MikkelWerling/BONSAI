# Extract baseline demographics for the released supervised Table 1 population.
# Run on the secure server only:
# Rscript opera/R/export_table1_patient_demographics.R SURVIVAL.parquet OUTPUT.parquet
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected released overall-survival parquet and a new output parquet")
suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
})
survival_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_path <- args[[2L]]
if (file.exists(output_path)) stop("Refusing to overwrite: ", output_path)
source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset("patient")
if (!exists("patient")) stop("load_dataset('patient') did not create patient")
required <- c("patientid", "sex", "date_birth")
if (!all(required %in% names(patient)))
  stop("Missing patient-table columns: ", paste(setdiff(required, names(patient)), collapse = ", "))

survival <- as.data.frame(read_parquet(survival_path, col_select = "subject_id"))
if (anyNA(survival$subject_id) || anyDuplicated(survival$subject_id))
  stop("Survival file must contain unique, nonmissing subject IDs")
canonical_ids <- function(values, label) {
  if (is.numeric(values) && !inherits(values, "integer64")) {
    if (any(!is.finite(values)) || any(values != trunc(values)) ||
        any(abs(values) > 2^53))
      stop(label, " has non-integer or non-exact numeric IDs")
    return(format(values, scientific = FALSE, trim = TRUE))
  }
  ids <- as.character(values)
  if (anyNA(ids) || !all(grepl("^[0-9]+$", ids)))
    stop(label, " must have nonmissing decimal-integer IDs")
  ids
}
ids <- canonical_ids(survival$subject_id, "Survival")
source_ids <- canonical_ids(patient$patientid, "Patient table")
if (anyNA(source_ids) || anyDuplicated(source_ids))
  stop("Patient table must contain unique, nonmissing patient IDs")
index <- match(ids, source_ids)
if (anyNA(index)) stop("Patient table does not cover every supervised subject")
selected <- as.data.frame(patient[index, required, drop = FALSE])
if (!identical(canonical_ids(selected$patientid, "Selected patients"), ids))
  stop("Patient join changed subject order")

# Preserve the survival file's ID type so the downstream Python join matches.
# Double-precision IDs are accepted only when exact whole numbers <= 2^53.
selected$subject_id <- survival$subject_id
selected$sex <- as.character(selected$sex)
if (!all(is.na(selected$sex) | selected$sex %in% c("M", "F")))
  stop("Unexpected sex code in patient table; inspect before recoding")
selected$birth_date <- as.Date(selected$date_birth)
if (any(!is.na(selected$date_birth) & is.na(selected$birth_date)))
  stop("Unparseable date_birth values")
output <- selected[c("subject_id", "sex", "birth_date")]
if (nrow(output) != nrow(survival) || anyDuplicated(output$subject_id))
  stop("Demographic output does not match supervised population")
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_parquet(output, output_path)
cat("Supervised patients:", nrow(output), "\n")
cat("Missing sex:", sum(is.na(output$sex)), "\n")
cat("Missing birth date:", sum(is.na(output$birth_date)), "\n")
cat("Output:", normalizePath(output_path), "\n")
