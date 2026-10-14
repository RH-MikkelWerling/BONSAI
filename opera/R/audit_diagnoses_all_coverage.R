# Secure-server-only source coverage check. Prints no patient-level rows or
# small exact counts. Usage: Rscript opera/R/audit_diagnoses_all_coverage.R
suppressPackageStartupMessages({
  library(dplyr)
})
source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset("diagnoses_all")
required <- c("patientid", "diagnosis", "date_diagnosis")
if (!all(required %in% names(diagnoses_all)))
  stop("diagnoses_all missing required columns: ",
       paste(setdiff(required, names(diagnoses_all)), collapse = ", "))
dx <- diagnoses_all %>%
  transmute(patientid, diagnosis = as.character(diagnosis),
            event_date = as.Date(date_diagnosis)) %>%
  filter(!is.na(event_date))
cat("Latest nonmissing diagnosis date:",
    if (nrow(dx)) format(max(dx$event_date)) else "NONE", "\n")
for (year in 2024:2026) {
  in_year <- dx$event_date >= as.Date(sprintf("%d-01-01", year)) &
    dx$event_date < as.Date(sprintf("%d-01-01", year + 1L))
  n <- sum(in_year)
  band <- if (n == 0L) "none" else if (n < 11L) "1-10" else
    if (n < 100L) "11-99" else "100+"
  cat(year, "all diagnosis rows:", band, "\n")
}
cat("This checks the loaded diagnoses_all table, not completeness of each ICD family.\n")

# Match the outcome generator's SKS D-prefix rule, then inspect the same
# January 1–March 19 window in three years. Bands avoid disclosing small cells.
raw_code <- toupper(gsub("\\.", "", trimws(dx$diagnosis)))
is_sks_dx <- !is.na(raw_code) & startsWith(raw_code, "D")
icd3 <- substr(ifelse(is_sks_dx, substr(raw_code, 2L, nchar(raw_code)), ""), 1L, 3L)
families <- list(
  all_SKS_diagnoses = NULL,
  sepsis = c("A40", "A41"),
  serious_infection = c("A40", "A41", "B44", "B45", "B46", "B59"),
  herpes_zoster = "B02",
  invasive_fungal = c("B44", "B45", "B46"),
  myocardial_infarction_comparator = c("I21", "I22")
)
band_count <- function(n) if (n == 0L) "none" else if (n < 11L) "1-10" else
  if (n < 100L) "11-99" else "100+"
for (family in names(families)) {
  family_rows <- is_sks_dx &
    (is.null(families[[family]]) | icd3 %in% families[[family]])
  for (year in 2024:2026) {
    in_window <- dx$event_date >= as.Date(sprintf("%d-01-01", year)) &
      dx$event_date <= as.Date(sprintf("%d-03-19", year))
    cat(family, year, "Jan 1-Mar 19:",
        band_count(sum(family_rows & in_window, na.rm = TRUE)), "\n")
  }
}
