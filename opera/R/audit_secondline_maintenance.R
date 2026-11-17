# Audit of early second-line events: are maintenance or split induction counted as a new line?
#
# Rebuilds the second-line table exactly as the production run does (same five arguments and
# environment as run_secondline_from_sealed_firstline.R, but a NEW, empty output directory), then
# writes AGGREGATE tables only to <NEW_OUTPUT_DIR>/audit/:
#   events_by_cohort.csv        eligible, masked, events within 6/12/24 months, any-retreatment
#   time_to_secondline.csv      months from end of first line to second line, among 12-month events
#   early_secondline_regimens.csv  the 15 most common drug sets of second lines within 12 months
#   early_overlap_with_firstline.csv  share of those drug sets repeating first-line drugs
#   early_confidence.csv        evidence level and first-line end source among 12-month events
# No patient identifiers or dates leave the server.
#
# Usage: Rscript opera/R/audit_secondline_maintenance.R \
#   FIRSTLINE_SCRIPT SECONDLINE_SCRIPT SEALED_MAPPED_CSV NEW_OUTPUT_DIR SOURCE_WORKDIR
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(stringr); library(readr) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) stop("Same five arguments as run_secondline_from_sealed_firstline.R")
here <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
runner <- file.path(here, "run_secondline_from_sealed_firstline.R")
source(runner, local = globalenv(), echo = FALSE)   # builds ttnt_tbl in globalenv
stopifnot(exists("ttnt_tbl"))
out <- file.path(normalizePath(args[[4L]]), "audit")
dir.create(out, showWarnings = FALSE, mode = "0700")

t <- ttnt_tbl %>%
  distinct(patient_id, .keep_all = TRUE) %>%
  mutate(months_to_2L = as.numeric(secondline_date - as.Date(firstline_date)) / 30.44,
         months_end_to_2L = as.numeric(secondline_date - as.Date(firstline_end)) / 30.44,
         fine = coalesce(as.character(cohort_fine), as.character(cohort_grouped)))
within <- function(m) ttnt_event == 1L & !is.na(months_to_2L) & months_to_2L <= m

events <- t %>%
  group_by(cohort_grouped, fine) %>%
  summarise(patients = n(), eligible = sum(ttnt_eligible, na.rm = TRUE), masked = sum(!ttnt_eligible, na.rm = TRUE),
            events_6m = sum(ttnt_event == 1L & months_to_2L <= 6, na.rm = TRUE),
            events_12m = sum(ttnt_event == 1L & months_to_2L <= 12, na.rm = TRUE),
            events_24m = sum(ttnt_event == 1L & months_to_2L <= 24, na.rm = TRUE),
            any_retreatment = sum(retreatment_event == 1L, na.rm = TRUE), .groups = "drop") %>%
  mutate(share_12m_of_eligible = round(events_12m / eligible, 3))
write_csv(events, file.path(out, "events_by_cohort.csv"))

early <- t %>% filter(ttnt_event == 1L, !is.na(months_to_2L), months_to_2L <= 12)
write_csv(early %>%
            mutate(bin = cut(months_end_to_2L, c(-Inf, 1, 2, 3, 4, 6, 9, 12, Inf),
                             labels = c("<1", "1-2", "2-3", "3-4", "4-6", "6-9", "9-12", ">12"))) %>%
            count(cohort_grouped, bin, name = "events"),
          file.path(out, "time_to_secondline.csv"))

norm <- function(x) str_squish(str_to_lower(coalesce(x, "")))
write_csv(early %>% mutate(regimen = norm(secondline_codes_text)) %>%
            count(cohort_grouped, regimen, name = "events") %>%
            group_by(cohort_grouped) %>% mutate(share = round(events / sum(events), 3)) %>%
            slice_max(events, n = 15, with_ties = FALSE) %>% ungroup(),
          file.path(out, "early_secondline_regimens.csv"))

first_col <- intersect(c("firstline_codes_text", "firstline_codes_text_1L", "firstline_codes_text_norm"), names(t))[1]
if (!is.na(first_col)) {
  split_set <- function(x) str_split(norm(x), "\\s*\\|\\s*")
  ov <- early %>%
    mutate(s1 = split_set(.data[[first_col]]), s2 = split_set(secondline_codes_text),
           overlap = purrr::map2_dbl(s1, s2, ~ if (length(.y) && any(nzchar(.y)))
             mean(.y[nzchar(.y)] %in% .x[nzchar(.x)]) else NA_real_),
           only_repeat = overlap == 1)
  write_csv(ov %>% group_by(cohort_grouped) %>%
              summarise(events = n(), median_overlap = median(overlap, na.rm = TRUE),
                        all_drugs_repeat_first_line = sum(only_repeat, na.rm = TRUE), .groups = "drop"),
            file.path(out, "early_overlap_with_firstline.csv"))
}
write_csv(early %>% count(cohort_grouped, secondline_confidence,
                          firstline_end_source = if ("firstline_end_source" %in% names(early))
                            firstline_end_source else NA, name = "events"),
          file.path(out, "early_confidence.csv"))
message("Audit tables written to ", out)
