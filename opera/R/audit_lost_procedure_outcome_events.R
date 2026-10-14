# Read-only aggregate impact of LPR2 procedure keys lost by as.integer(v_recnum).
# Run from the repository root after the outcome export; no patient rows are printed.
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(arrow)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript opera/R/audit_lost_procedure_outcome_events.R OUTCOMES_DIR SEALED_FIRSTLINE_CSV")
}
outcomes_dir <- args[[1L]]
if (!dir.exists(outcomes_dir)) stop("Outcome directory not found: ", outcomes_dir)
firstline_csv <- args[[2L]]
if (!file.exists(firstline_csv)) {
  stop("First-line CSV not found: ", firstline_csv,
       ". Pass its path as the second argument.")
}

key_text <- function(x) {
  result <- if (is.numeric(x)) trimws(format(x, scientific = FALSE, trim = TRUE))
            else trimws(as.character(x))
  result[is.na(x)] <- NA_character_
  result
}

source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset("patient", selected_columns = c("patientid", "date_death_fu"))
firstline <- read_csv(firstline_csv, show_col_types = FALSE) %>%
  transmute(patientid, index_date = as.Date(firstline_date)) %>%
  inner_join(patient %>% select(patientid, date_death_fu), by = "patientid") %>%
  filter(index_date < as.Date(date_death_fu),
         index_date < as.Date("2026-03-19")) %>%
  mutate(censor_date = pmin(as.Date(date_death_fu), as.Date("2026-03-19"))) %>%
  select(patientid, index_date, censor_date)
if (anyNA(firstline$patientid) || anyDuplicated(firstline$patientid))
  stop("First-line analysis population has missing or duplicate patient IDs")
message("Reconstructed procedure-audit risk set: ", nrow(firstline), " patients")

load_dataset("SDS_t_adm", selected_columns = c("k_recnum", "patientid", "d_inddto"))
admissions <- SDS_t_adm %>%
  transmute(key = key_text(k_recnum), patientid,
            event_date = as.Date(d_inddto)) %>%
  filter(!is.na(key), nzchar(key), !is.na(event_date)) %>%
  inner_join(firstline, by = "patientid") %>%
  filter(event_date > index_date, event_date <= censor_date) %>%
  distinct(key, patientid, event_date)
rm(SDS_t_adm)
invisible(gc())

classify_code <- function(code) {
  case_when(
    code %in% c("BOQA0", "BOQA00", "BOQA01", "BOQA02", "BOQA03", "BOQA04", "BOQA05") ~ "RBC_transfusion",
    code %in% c("BOQC", "BOQC3") ~ "platelet_transfusion",
    grepl("^BPH", code) ~ "antibiotics",
    code %in% c("BGDA0", "BGDA7", "BGDA1", "BGDA2") ~ "respiratory_support",
    grepl("^BJFD", code) ~ "dialysis",
    grepl("^NAB", code) & !grepl("NAB1", code) ~ "emergency_stay",
    TRUE ~ NA_character_
  )
}

lost_events <- function(source, label) {
  candidate <- source %>%
    transmute(key = key_text(v_recnum),
              cast_key = suppressWarnings(as.integer(v_recnum)),
              outcome = classify_code(as.character(c_opr))) %>%
    filter(!is.na(key), nzchar(key), is.na(cast_key), !is.na(outcome)) %>%
    select(key, outcome) %>%
    distinct() %>%
    inner_join(admissions, by = "key") %>%
    select(outcome, patientid, event_date) %>%
    distinct()
  cat(label, ": lost-key eligible patient-outcome-date rows=", nrow(candidate),
      "\n", sep = "")
  candidate
}

load_dataset("SDS_t_sksopr", selected_columns = c("v_recnum", "c_opr"))
surgery <- lost_events(SDS_t_sksopr, "SDS_t_sksopr")
rm(SDS_t_sksopr)
invisible(gc())
load_dataset("SDS_t_sksube", selected_columns = c("v_recnum", "c_opr"))
other <- lost_events(SDS_t_sksube, "SDS_t_sksube")
rm(SDS_t_sksube)
invisible(gc())

candidate <- bind_rows(surgery, other)
transfusion <- candidate %>%
  filter(outcome %in% c("RBC_transfusion", "platelet_transfusion")) %>%
  mutate(outcome = "any_transfusion")
candidate <- bind_rows(candidate, transfusion) %>%
  group_by(outcome, patientid) %>%
  summarise(candidate_date = min(event_date), .groups = "drop")

for (name in c("RBC_transfusion", "platelet_transfusion", "any_transfusion",
               "antibiotics", "respiratory_support", "dialysis", "emergency_stay")) {
  path <- file.path(outcomes_dir, paste0(name, ".parquet"))
  if (!file.exists(path)) stop("Existing outcome missing: ", path)
  old <- read_parquet(path, col_select = c("subject_id", "outcome_date")) %>%
    transmute(patientid = subject_id, old_date = as.Date(outcome_date))
  if (nrow(old) != nrow(firstline) || anyDuplicated(old$patientid)) {
    stop("Existing outcome population mismatch: ", name)
  }
  matched <- candidate %>%
    filter(outcome == name) %>%
    left_join(old, by = "patientid")
  cat(name,
      ": lost_key_patients=", nrow(matched),
      " currently_no_event=", sum(is.na(matched$old_date)),
      " currently_later_event=", sum(!is.na(matched$old_date) &
                                      matched$candidate_date < matched$old_date),
      " currently_same_or_earlier_event=", sum(!is.na(matched$old_date) &
                                                matched$candidate_date >= matched$old_date),
      "\n", sep = "")
}
