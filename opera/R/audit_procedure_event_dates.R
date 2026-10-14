# Read-only aggregate audit: procedure date versus admission/contact start date.
# No patient-level rows are printed or written.
suppressPackageStartupMessages({ library(dplyr); library(readr) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L)
  stop("Usage: Rscript opera/R/audit_procedure_event_dates.R FIRSTLINE_CSV OUTPUT_CSV")
firstline_path <- args[[1L]]
output_path <- args[[2L]]
if (!file.exists(firstline_path)) stop("Missing first-line CSV: ", firstline_path)
if (!dir.exists(dirname(output_path))) stop("Missing output parent directory")

source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset("patient", selected_columns = c("patientid", "date_death_fu"))
cohort <- read_csv(firstline_path, show_col_types = FALSE) %>%
  transmute(patientid, index_date = as.Date(firstline_date)) %>%
  inner_join(patient %>% select(patientid, date_death_fu), by = "patientid") %>%
  filter(index_date < as.Date(date_death_fu), index_date < as.Date("2026-03-19")) %>%
  mutate(censor_date = pmin(as.Date(date_death_fu), as.Date("2026-03-19"))) %>%
  select(patientid, index_date, censor_date)
expected_n <- Sys.getenv("OPERA_EXPECTED_OUTCOME_N", "")
if (!nrow(cohort) || anyDuplicated(cohort$patientid) ||
    (nzchar(expected_n) &&
     (is.na(suppressWarnings(as.integer(expected_n))) ||
      nrow(cohort) != as.integer(expected_n))))
  stop("Unexpected outcome population; refusing date comparison")

key_text <- function(x) {
  result <- if (is.numeric(x)) trimws(format(x, scientific = FALSE, trim = TRUE))
            else trimws(as.character(x))
  result[is.na(x) | !nzchar(result)] <- NA_character_
  result
}
classify_code <- function(x) {
  x <- as.character(x)
  case_when(
    x %in% c("BOQA0", "BOQA00", "BOQA01", "BOQA02", "BOQA03", "BOQA04", "BOQA05") ~ "RBC_transfusion",
    x %in% c("BOQC", "BOQC3") ~ "platelet_transfusion",
    grepl("^BPH", x) ~ "antibiotics",
    x %in% c("BGDA0", "BGDA7", "BGDA1", "BGDA2") ~ "respiratory_support",
    grepl("^BJFD", x) ~ "dialysis",
    grepl("^NAB", x) & !grepl("NAB1", x) ~ "emergency_stay",
    TRUE ~ NA_character_
  )
}

load_dataset("SDS_t_adm", selected_columns = c("k_recnum", "patientid", "d_inddto", "d_uddto"))
admissions <- SDS_t_adm %>%
  transmute(key = key_text(k_recnum), patientid,
            encounter_date = as.Date(d_inddto),
            encounter_end = as.Date(d_uddto)) %>%
  filter(!is.na(key)) %>% inner_join(cohort, by = "patientid")
rm(SDS_t_adm)
invisible(gc())

load_dataset("SDS_kontakter", selected_columns = c("dw_ek_kontakt", "patientid", "dato_start", "dato_slut"))
contacts <- SDS_kontakter %>%
  transmute(key = key_text(dw_ek_kontakt), patientid,
            encounter_date = as.Date(dato_start),
            encounter_end = as.Date(dato_slut)) %>%
  filter(!is.na(key)) %>% inner_join(cohort, by = "patientid")
rm(SDS_kontakter)
invisible(gc())

summarize_source <- function(source, key_col, code_col, date_col, encounters, label) {
  event_rows <- source %>%
    transmute(key = key_text(.data[[key_col]]),
              outcome = classify_code(.data[[code_col]]),
              procedure_date = as.Date(.data[[date_col]])) %>%
    filter(!is.na(key), !is.na(outcome)) %>%
    inner_join(encounters, by = "key", relationship = "many-to-many") %>%
    select(patientid, index_date, censor_date, outcome,
           encounter_date, encounter_end, procedure_date) %>% distinct()
  event_rows %>%
    mutate(encounter_post = !is.na(encounter_date) & encounter_date > index_date &
             encounter_date <= censor_date,
           procedure_post = !is.na(procedure_date) & procedure_date > index_date &
             procedure_date <= censor_date,
           lag_days = as.integer(procedure_date - encounter_date)) %>%
    group_by(outcome) %>%
    summarise(source = label, n_patient_date_pairs = n(),
              n_patients = n_distinct(patientid),
              missing_procedure_date = sum(is.na(procedure_date)),
              missing_encounter_end = sum(is.na(encounter_end)),
              same_date = sum(lag_days == 0, na.rm = TRUE),
              procedure_after_encounter = sum(lag_days > 0, na.rm = TRUE),
              procedure_before_encounter = sum(lag_days < 0, na.rm = TRUE),
              procedure_after_encounter_end = sum(
                !is.na(procedure_date) & !is.na(encounter_end) &
                  procedure_date > encounter_end),
              procedure_more_than_30d_after_start = sum(lag_days > 30, na.rm = TRUE),
              n_postindex_by_encounter_date = sum(encounter_post),
              n_postindex_by_procedure_date = sum(procedure_post),
              n_procedure_only_postindex = sum(procedure_post & !encounter_post),
              n_encounter_only_postindex = sum(encounter_post & !procedure_post),
              n_patients_procedure_only_postindex =
                n_distinct(patientid[procedure_post & !encounter_post]),
              .groups = "drop")
}

load_dataset("SDS_t_sksube", selected_columns = c("v_recnum", "c_opr", "d_odto"))
lpr2_other <- summarize_source(SDS_t_sksube, "v_recnum", "c_opr", "d_odto",
                               admissions, "LPR2_other")
rm(SDS_t_sksube)
invisible(gc())
load_dataset("SDS_t_sksopr", selected_columns = c("v_recnum", "c_opr", "d_odto"))
lpr2_surgery <- summarize_source(SDS_t_sksopr, "v_recnum", "c_opr", "d_odto",
                                 admissions, "LPR2_surgery")
rm(SDS_t_sksopr, admissions)
invisible(gc())

load_dataset("SDS_procedurer_andre", selected_columns = c("dw_ek_kontakt", "procedurekode", "dato_start"))
lpr3_other <- summarize_source(SDS_procedurer_andre, "dw_ek_kontakt", "procedurekode",
                               "dato_start", contacts, "LPR3_other")
rm(SDS_procedurer_andre)
invisible(gc())
load_dataset("SDS_procedurer_kirurgi", selected_columns = c("dw_ek_kontakt", "procedurekode", "dato_start"))
lpr3_surgery <- summarize_source(SDS_procedurer_kirurgi, "dw_ek_kontakt", "procedurekode",
                                 "dato_start", contacts, "LPR3_surgery")

result <- bind_rows(lpr2_other, lpr2_surgery, lpr3_other, lpr3_surgery) %>%
  arrange(outcome, source)
write_csv(result, output_path)
print(result, n = nrow(result), width = Inf)
cat("Aggregate date audit: ", output_path, "\n", sep = "")
