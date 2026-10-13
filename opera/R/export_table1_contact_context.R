# Baseline care-setting summaries from pre-index hospital contacts.
# Rscript opera/R/export_table1_contact_context.R POPULATION.parquet SURVIVAL.parquet OUTPUT.parquet [LOOKBACK_DAYS]
args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(3L, 4L)))
  stop("Expected sealed population, released survival, new output parquet, optional lookback days")
suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
})
population_path <- normalizePath(args[[1L]], mustWork = TRUE)
survival_path <- normalizePath(args[[2L]], mustWork = TRUE)
output_path <- args[[3L]]
lookback_days <- if (length(args) == 4L) as.integer(args[[4L]]) else 1095L
if (is.na(lookback_days) || lookback_days < 1L) stop("Lookback must be positive days")
if (file.exists(output_path)) stop("Refusing to overwrite: ", output_path)

population <- as.data.frame(read_parquet(population_path,
                                          col_select = c("subject_id", "firstline_date")))
survival <- as.data.frame(read_parquet(survival_path, col_select = "subject_id"))
if (anyNA(population$subject_id) || anyDuplicated(population$subject_id) ||
    anyNA(survival$subject_id) || anyDuplicated(survival$subject_id))
  stop("Population and survival require unique, nonmissing subject IDs")
position <- match(survival$subject_id, population$subject_id)
if (anyNA(position)) stop("Survival subjects not covered by sealed population")
baseline <- population[position, , drop = FALSE]
baseline$index_date <- as.Date(baseline$firstline_date)
if (anyNA(baseline$index_date)) stop("Unparseable first-line dates")

source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset("SDS_kontakter", selected_columns = c(
  "patientid", "dato_start", "region_ans", "hovedspeciale_ans"))
if (!exists("SDS_kontakter")) stop("Contact loader did not create SDS_kontakter")
contacts <- as.data.frame(SDS_kontakter)
required <- c("patientid", "dato_start", "region_ans", "hovedspeciale_ans")
if (!all(required %in% names(contacts)))
  stop("Contact table lacks: ", paste(setdiff(required, names(contacts)), collapse = ", "))

id_text <- function(x) {
  if (is.numeric(x) && !inherits(x, "integer64")) {
    if (any(!is.finite(x[!is.na(x)])) ||
        any(x[!is.na(x)] != trunc(x[!is.na(x)])) ||
        any(abs(x[!is.na(x)]) > 2^53)) stop("Non-exact numeric patient IDs")
    out <- format(x, scientific = FALSE, trim = TRUE)
    out[is.na(x)] <- NA_character_
    return(out)
  }
  as.character(x)
}
baseline$key <- id_text(baseline$subject_id)
contacts$key <- id_text(contacts$patientid)
contacts$contact_date <- as.Date(contacts$dato_start)
if (any(!is.na(contacts$dato_start) & is.na(contacts$contact_date)))
  stop("Unparseable contact dates")
contacts <- contacts[!is.na(contacts$key) & !is.na(contacts$contact_date), , drop = FALSE]
contacts <- contacts[contacts$key %in% baseline$key, , drop = FALSE]
position <- match(contacts$key, baseline$key)
index_date <- baseline$index_date[position]
contacts <- contacts[
  contacts$contact_date < index_date &
    contacts$contact_date >= index_date - lookback_days, , drop = FALSE]

clean_label <- function(x) {
  x <- trimws(as.character(x))
  x[is.na(x) | x == ""] <- NA_character_
  x
}
contacts$region <- clean_label(contacts$region_ans)
contacts$specialty <- clean_label(contacts$hovedspeciale_ans)
by_patient <- split(contacts, contacts$key)
summarize_one <- function(x) {
  if (is.null(x) || nrow(x) == 0L)
    return(list(region = NA_character_, region_status = "no_preindex_contact",
                n_unique_specialties = NA_integer_))
  regions <- x$region[!is.na(x$region)]
  specialties <- x$specialty[!is.na(x$specialty)]
  if (!length(regions)) {
    region <- NA_character_
    status <- "region_missing"
  } else {
    frequency <- table(regions)
    modes <- names(frequency)[frequency == max(frequency)]
    region <- if (length(modes) == 1L) modes[[1L]] else NA_character_
    status <- if (length(modes) == 1L) "unique_mode" else "tied_mode"
  }
  list(region = region, region_status = status,
       n_unique_specialties = if (length(specialties))
         length(unique(specialties)) else NA_integer_)
}
values <- lapply(baseline$key, function(key) summarize_one(by_patient[[key]]))
output <- data.frame(
  subject_id = baseline$subject_id,
  predominant_region = vapply(values, `[[`, character(1), "region"),
  predominant_region_status = vapply(values, `[[`, character(1), "region_status"),
  n_unique_specialties = vapply(values, `[[`, integer(1), "n_unique_specialties")
)
if (nrow(output) != nrow(survival) || anyDuplicated(output$subject_id))
  stop("Contact summary altered supervised population")
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_parquet(output, output_path)
cat("Supervised patients:", nrow(output), "\n")
cat("Lookback days:", lookback_days, "(strictly before index)\n")
cat("Region status:\n")
print(table(output$predominant_region_status, useNA = "ifany"))
cat("Missing specialty count:", sum(is.na(output$n_unique_specialties)), "\n")
cat("Output:", normalizePath(output_path), "\n")
