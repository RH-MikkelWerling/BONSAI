# Secure-server-only aggregate diagnostic for the bounded MM V4/V5 decision.
# Usage: Rscript opera/R/compare_mm_seed_rkkp_dates.R V4_MAPPED.csv V5_MAPPED.csv OUTPUT.csv
# No patient identifiers or patient-level dates are exported.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Expected V4 mapped CSV, V5 mapped CSV, output CSV")
paths <- normalizePath(args[1:2], mustWork = TRUE)
output <- normalizePath(args[[3]], mustWork = FALSE)
if (file.exists(output)) stop("Refusing to overwrite: ", output)
if (!dir.exists(dirname(output))) stop("Create the private output directory first")

read_candidate <- function(path, prefix) {
  x <- readr::read_csv(path, show_col_types = FALSE)
  required <- c("patientid", "firstline_date", "cohort_fine", "firstline_source")
  if (!all(required %in% names(x)))
    stop("Candidate lacks required columns: ", path)
  x <- x |>
    transmute(patientid = as.character(patientid),
              firstline_date = as.Date(firstline_date),
              cohort_fine = as.character(cohort_fine),
              firstline_source = as.character(firstline_source))
  if (anyNA(x$patientid) || anyDuplicated(x$patientid))
    stop("Missing or duplicate patient IDs: ", path)
  names(x)[-1] <- paste0(prefix, names(x)[-1])
  x
}
old <- read_candidate(paths[[1]], "v4_")
new <- read_candidate(paths[[2]], "v5_")

source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset("RKKP_DaMyDa_CLEAN")
if (!all(c("patientid", "date_treatment_1st_line_start") %in%
         names(RKKP_DaMyDa_CLEAN)))
  stop("DaMyDa schema lacks patientid or first-line start date")
registry <- RKKP_DaMyDa_CLEAN |>
  transmute(patientid = as.character(patientid),
            registry_date = as.Date(date_treatment_1st_line_start)) |>
  filter(!is.na(patientid), !is.na(registry_date)) |>
  group_by(patientid) |>
  summarise(registry_date = min(registry_date), .groups = "drop")

paired <- inner_join(old, new, by = "patientid") |>
  inner_join(registry, by = "patientid") |>
  filter(!is.na(v4_firstline_date), !is.na(v5_firstline_date)) |>
  mutate(delta_v5_minus_v4 = as.integer(v5_firstline_date - v4_firstline_date),
         v4_abs = abs(as.integer(v4_firstline_date - registry_date)),
         v5_abs = abs(as.integer(v5_firstline_date - registry_date)),
         # Sensitivity only; does not prescribe a blanket one-day correction.
         v4_abs_plus1 = abs(as.integer(v4_firstline_date + 1L - registry_date)),
         v5_abs_plus1 = abs(as.integer(v5_firstline_date + 1L - registry_date)),
         mm_mapped = v4_cohort_fine %in% c("MM", "PCD", "PCL", "SolM") |
           v5_cohort_fine %in% c("MM", "PCD", "PCL", "SolM"))

summarise_scope <- function(x, label) {
  tibble::tibble(
    scope = label,
    n_with_both_candidates_and_damyda = nrow(x),
    n_v5_date_changed = sum(x$delta_v5_minus_v4 != 0L),
    n_v5_date_over_30d_changed = sum(abs(x$delta_v5_minus_v4) > 30L),
    n_v5_later_than_v4 = sum(x$delta_v5_minus_v4 > 0L),
    n_v4_closer_to_registry = sum(x$v4_abs < x$v5_abs),
    n_v5_closer_to_registry = sum(x$v5_abs < x$v4_abs),
    n_tied_registry_distance = sum(x$v4_abs == x$v5_abs),
    n_v4_within_7d_registry = sum(x$v4_abs <= 7L),
    n_v5_within_7d_registry = sum(x$v5_abs <= 7L),
    n_v4_within_30d_registry = sum(x$v4_abs <= 30L),
    n_v5_within_30d_registry = sum(x$v5_abs <= 30L),
    n_v4_closer_with_plus1_sensitivity = sum(x$v4_abs_plus1 < x$v5_abs_plus1),
    n_v5_closer_with_plus1_sensitivity = sum(x$v5_abs_plus1 < x$v4_abs_plus1)
  )
}
report <- dplyr::bind_rows(
  summarise_scope(paired, "all_damyda_shared"),
  summarise_scope(filter(paired, mm_mapped), "mm_mapped_either_release"),
  summarise_scope(filter(paired, mm_mapped, delta_v5_minus_v4 != 0L),
                  "mm_mapped_changed_date"),
  summarise_scope(filter(paired, mm_mapped, abs(delta_v5_minus_v4) > 30L),
                  "mm_mapped_changed_over_30d")
)
readr::write_csv(report, output)
message("INTERNAL aggregate MM seed/DaMyDa comparison: ", output)
