# Aggregate, read-only reconciliation of the first-line and outcome populations.
# Usage: Rscript opera/R/audit_outcome_population_gate.R FIRSTLINE_MAPPED_CSV
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L || !file.exists(args[[1L]]))
  stop("Pass the sealed first-line mapped CSV as the only argument")
source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset("patient")

input <- read_csv(args[[1L]], show_col_types = FALSE)
stopifnot(!anyNA(input$patientid), !anyDuplicated(input$patientid))
patient_gate <- patient %>%
  select(patientid, date_death_fu, status) %>%
  mutate(in_patient_table = TRUE)
stopifnot(!anyNA(patient_gate$patientid), !anyDuplicated(patient_gate$patientid))

gate <- input %>%
  mutate(firstline_date = as.Date(firstline_date)) %>%
  left_join(patient_gate, by = "patientid", relationship = "many-to-one") %>%
  mutate(
    date_death_fu = as.Date(date_death_fu),
    reason = case_when(
      is.na(in_patient_table) ~ "missing_patient_row",
      is.na(firstline_date) ~ "missing_firstline_date",
      is.na(date_death_fu) ~ "missing_followup_date",
      firstline_date > date_death_fu ~ "followup_before_firstline",
      firstline_date == date_death_fu ~ "followup_equal_firstline",
      firstline_date >= as.Date("2026-03-19") ~ "index_on_or_after_2026-03-19",
      TRUE ~ "retained"
    )
  )

cat("Input rows:", nrow(input), "\n")
print(gate %>% count(reason, name = "n_patients"), n = Inf)
cat("Reconstructed outcome rows:", sum(gate$reason == "retained"), "\n")
cat("Death status among follow-up before/equal first line:\n")
print(gate %>%
  filter(reason %in% c("followup_before_firstline", "followup_equal_firstline")) %>%
  count(reason, status, name = "n_patients"), n = Inf)
cat("Cutoff sensitivity among rows passing the follow-up gate:\n")
print(gate %>%
  filter(reason %in% c("index_on_or_after_2026-03-19", "retained")) %>%
  summarise(
    index_on_or_after_2026_03_04 = sum(firstline_date >= as.Date("2026-03-04")),
    index_on_or_after_2026_03_19 = sum(firstline_date >= as.Date("2026-03-19")),
    index_after_2026_03_19 = sum(firstline_date > as.Date("2026-03-19"))
  ))

cat("Outcome risk-set difference from sealed first line:",
    nrow(input) - sum(gate$reason == "retained"), "\n")
