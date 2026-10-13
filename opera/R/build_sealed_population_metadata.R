# Versioned, one-row-per-patient metadata from the sealed first-line export and
# the server's IPI dataset. Multiple IPI rows are preserved, never selected by
# row order or averaged across diseases.
# Rscript opera/R/build_sealed_population_metadata.R FIRSTLINE.csv SURVIVAL.parquet OUTPUT_DIR
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Expected first-line CSV, released survival parquet, new output directory")
suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(jsonlite)
  library(readr)
})
firstline_path <- normalizePath(args[[1]], mustWork = TRUE)
survival_path <- normalizePath(args[[2]], mustWork = TRUE)
output_dir <- args[[3]]
if (file.exists(output_dir)) stop("Output directory already exists: ", output_dir)

loader <- "/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R"
sha256 <- function(path) {
  line <- suppressWarnings(system2("sha256sum", shQuote(path), stdout = TRUE))
  if (length(line) != 1L || !is.null(attr(line, "status")))
    stop("Could not hash input/output: ", path)
  substring(line, 1L, 64L)
}
source(loader)
load_dataset("IPI")
if (!exists("IPI")) stop("load_dataset('IPI') did not create IPI")
ipi <- as.data.frame(IPI)
id_candidates <- intersect(c("patientid", "patient_id", "subject_id"), names(ipi))
if (length(id_candidates) != 1L)
  stop("Expected exactly one IPI patient-ID column; observed columns: ", paste(names(ipi), collapse = ", "))
names(ipi)[names(ipi) == id_candidates] <- "subject_id"
if (anyNA(ipi$subject_id)) stop("IPI contains missing patient IDs")
ipi$subject_id <- as.character(ipi$subject_id)

first <- read_csv(firstline_path, col_types = cols(patientid = col_character()), show_col_types = FALSE)
required <- c("patientid", "firstline_date", "cohort_grouped", "cohort_fine")
if (!all(required %in% names(first))) stop("Sealed first-line export lacks required columns")
if (anyNA(first$patientid) || anyDuplicated(first$patientid)) stop("Invalid sealed first-line patient IDs")
if (anyNA(first$firstline_date) || anyNA(first$cohort_fine) || anyNA(first$cohort_grouped))
  stop("Sealed first-line export has missing date/cohort labels")
metadata <- first %>% rename(subject_id = patientid)
metadata$subject_id <- as.character(metadata$subject_id)

survival <- as.data.frame(read_parquet(survival_path))
if (!all(c("subject_id", "index_date", "split") %in% names(survival)))
  stop("Released survival outcome lacks subject_id, index_date or split")
survival$subject_id <- as.character(survival$subject_id)
if (anyNA(survival$subject_id) || anyDuplicated(survival$subject_id) ||
    !all(survival$subject_id %in% metadata$subject_id))
  stop("Released survival patients are not a unique subset of the sealed first-line cohort")
pos <- match(survival$subject_id, metadata$subject_id)
if (anyNA(survival$index_date) ||
    any(as.Date(survival$index_date) != as.Date(metadata$firstline_date[pos])))
  stop("Released survival index dates disagree with sealed first-line dates")
if (anyNA(survival$split)) stop("Released survival has missing splits")
if ("split" %in% names(metadata)) {
  if (anyNA(metadata$split[pos]) ||
      any(as.character(metadata$split[pos]) != as.character(survival$split)))
    stop("First-line and released survival splits disagree")
} else {
  metadata$split <- NA_character_
  metadata$split[pos] <- as.character(survival$split)
}

# Keep every distinct IPI row. A patient can legitimately have multiple
# disease-specific assessments; JSON preserves all source fields and order.
# Identical duplicate source rows are dropped only after counting them.
ipi_in_cohort <- ipi[ipi$subject_id %in% metadata$subject_id, , drop = FALSE]
identical_duplicates <- sum(duplicated(ipi_in_cohort))
ipi_in_cohort <- unique(ipi_in_cohort)
row_json <- vapply(seq_len(nrow(ipi_in_cohort)), function(i)
  as.character(toJSON(as.list(ipi_in_cohort[i, , drop = FALSE]), auto_unbox = TRUE, na = "null")),
  character(1))
records <- split(row_json, ipi_in_cohort$subject_id)
metadata$ipi_record_count <- lengths(records[metadata$subject_id])
metadata$ipi_record_count[is.na(metadata$ipi_record_count)] <- 0L
metadata$ipi_records_json <- vapply(metadata$subject_id, function(id) {
  values <- records[[id]]
  if (is.null(values)) return("[]")
  paste0("[", paste(values, collapse = ","), "]")
}, character(1))
if (nrow(metadata) != nrow(first) || anyDuplicated(metadata$subject_id))
  stop("Metadata build altered the sealed cohort")

if (!dir.create(output_dir, recursive = TRUE, showWarnings = FALSE))
  stop("Could not create output directory")
Sys.chmod(output_dir, mode = "0700")
write_parquet(metadata, file.path(output_dir, "population_metadata.parquet"))
write_parquet(data.frame(subject_id = ipi_in_cohort$subject_id,
                         ipi_record_json = row_json),
              file.path(output_dir, "ipi_records_long_INTERNAL.parquet"))
metadata_path <- file.path(output_dir, "population_metadata.parquet")
ipi_long_path <- file.path(output_dir, "ipi_records_long_INTERNAL.parquet")
manifest <- list(status = "internal_metadata_candidate", n_firstline = nrow(first),
                 n_survival = nrow(survival), n_ipi_source_rows = nrow(ipi),
                 n_ipi_rows_in_cohort = nrow(ipi_in_cohort),
                 n_identical_ipi_rows_removed = identical_duplicates,
                 n_patients_with_ipi = sum(metadata$ipi_record_count > 0L),
                 n_patients_with_multiple_ipi_rows = sum(metadata$ipi_record_count > 1L),
                 n_missing_outcome_split = sum(is.na(metadata$split)),
                 firstline = firstline_path, survival = survival_path,
                 loader = loader, ipi_columns = names(ipi),
                 sha256 = list(firstline = sha256(firstline_path),
                               survival = sha256(survival_path),
                               loader = sha256(loader),
                               metadata = sha256(metadata_path),
                               ipi_records_long = sha256(ipi_long_path)))
write_json(manifest, file.path(output_dir, "manifest_INTERNAL.json"), pretty = TRUE, auto_unbox = TRUE)
message("Metadata candidate: ", normalizePath(output_dir))
message("First-line patients: ", nrow(metadata), "; with IPI: ", manifest$n_patients_with_ipi,
        "; multiple IPI rows: ", manifest$n_patients_with_multiple_ipi_rows)
