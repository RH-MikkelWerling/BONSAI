# Read-only aggregate audit of LPR2 and LPR3 procedure join keys.
# No patient-level rows are printed or written.
suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
})
source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset("SDS_t_adm", selected_columns = c("k_recnum", "patientid"))
load_dataset("SDS_t_sksopr", selected_columns = c("v_recnum", "c_opr"))
load_dataset("SDS_t_sksube", selected_columns = c("v_recnum", "c_opr"))
load_dataset("SDS_kontakter", selected_columns = c("dw_ek_kontakt", "patientid"))
load_dataset("SDS_procedurer_kirurgi", selected_columns = c("dw_ek_kontakt", "procedurekode"))
load_dataset("SDS_procedurer_andre", selected_columns = c("dw_ek_kontakt", "procedurekode"))

key_text <- function(x) {
  result <- if (is.numeric(x)) trimws(format(x, scientific = FALSE, trim = TRUE))
            else trimws(as.character(x))
  result[is.na(x)] <- NA_character_
  result
}
adm <- SDS_t_adm %>%
  transmute(key = key_text(k_recnum), patientid) %>%
  filter(!is.na(key), nzchar(key))
adm_keys <- adm %>% distinct(key, patientid) %>%
  count(key, name = "n_patients")
cat("Admissions: rows=", nrow(adm),
    " distinct_keys=", n_distinct(adm$key),
    " keys_with_multiple_patients=", sum(adm_keys$n_patients > 1), "\n", sep = "")

audit_source <- function(source, label) {
  raw <- source %>%
    transmute(key = key_text(v_recnum), code = as.character(c_opr),
              integer_key = suppressWarnings(as.integer(v_recnum)))
  raw <- raw %>% filter(!is.na(key), nzchar(key))
  by_key <- raw %>% group_by(key) %>%
    summarise(n_procedure_rows = n(), lost_by_integer_cast = all(is.na(integer_key)),
              .groups = "drop") %>%
    left_join(adm_keys, by = "key")
  cat(label, ": rows=", nrow(raw),
      " distinct_keys=", nrow(by_key),
      " integer_conversion_NA_rows=", sum(is.na(raw$integer_key)),
      " distinct_keys_lost_by_integer_conversion=", n_distinct(raw$key[is.na(raw$integer_key)]),
      " lost_keys_matching_admissions=", sum(by_key$lost_by_integer_cast & !is.na(by_key$n_patients)),
      " procedure_rows_on_lost_matching_keys=", sum(by_key$n_procedure_rows[by_key$lost_by_integer_cast & !is.na(by_key$n_patients)]),
      " unmatched_keys_as_text=", sum(is.na(by_key$n_patients)),
      " keys_with_multiple_procedures=", sum(by_key$n_procedure_rows > 1),
      " keys_mapping_multiple_patients=", sum(by_key$n_patients > 1, na.rm = TRUE),
      "\n", sep = "")
}
audit_source(SDS_t_sksopr, "SDS_t_sksopr")
audit_source(SDS_t_sksube, "SDS_t_sksube")

# LPR3 uses the same 32-bit cast in the outcome builder; audit it separately.
kontakt <- SDS_kontakter %>%
  transmute(key = key_text(dw_ek_kontakt), patientid) %>%
  filter(!is.na(key), nzchar(key))
kontakt_keys <- kontakt %>% distinct(key, patientid) %>%
  count(key, name = "n_patients")
cat("SDS_kontakter: rows=", nrow(kontakt),
    " distinct_keys=", nrow(kontakt_keys),
    " keys_with_multiple_patients=", sum(kontakt_keys$n_patients > 1), "\n", sep = "")

audit_lpr3 <- function(source, label) {
  raw <- source %>%
    transmute(key = key_text(dw_ek_kontakt),
              integer_key = suppressWarnings(as.integer(dw_ek_kontakt))) %>%
    filter(!is.na(key), nzchar(key))
  by_key <- raw %>% group_by(key) %>%
    summarise(n_procedure_rows = n(), lost_by_integer_cast = all(is.na(integer_key)),
              .groups = "drop") %>%
    left_join(kontakt_keys, by = "key")
  cat(label, ": rows=", nrow(raw),
      " distinct_keys=", nrow(by_key),
      " integer_conversion_NA_rows=", sum(is.na(raw$integer_key)),
      " lost_keys_matching_contacts=", sum(by_key$lost_by_integer_cast & !is.na(by_key$n_patients)),
      " procedure_rows_on_lost_matching_keys=", sum(by_key$n_procedure_rows[by_key$lost_by_integer_cast & !is.na(by_key$n_patients)]),
      " unmatched_keys_as_text=", sum(is.na(by_key$n_patients)),
      " keys_mapping_multiple_patients=", sum(by_key$n_patients > 1, na.rm = TRUE),
      "\n", sep = "")
}
audit_lpr3(SDS_procedurer_kirurgi, "SDS_procedurer_kirurgi")
audit_lpr3(SDS_procedurer_andre, "SDS_procedurer_andre")
