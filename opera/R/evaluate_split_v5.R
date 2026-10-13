# ==============================================================================
# evaluate_split.R
#
# Decide where the train / tuning / test cutoffs should fall. For a candidate
# pair of index-year cutoffs it reports, per outcome (and optionally cohort) and
# per split:
#   n eligible, n events, event rate
#   follow-up distribution (censor_date - index_date)
#   time-to-event distribution among events (outcome_date - index_date)
#   fraction with follow-up >= a fixed horizon, and events seen before it
#
# Why all three: a late test cutoff leaves recent patients little time before
# STUDY_CUTOFF, so an outcome with long time-to-event will show few test events
# purely from truncated follow-up, not rarity. Seeing follow-up next to event
# timing is what tells you whether a split is usable for that outcome.
#
# The split column inside the parquets is ignored; the split is recomputed from
# index_date for whatever cutoff you pass, so you can explore freely.
#
# Needs: arrow, dplyr, tidyr, stringr, ggplot2, lubridate, and a `first_line`
# carrying patientid, index_date, and a cohort column.
# ==============================================================================

suppressPackageStartupMessages({
  library(arrow); library(dplyr); library(tidyr)
  library(stringr); library(ggplot2); library(lubridate)
})

suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(dplyr)
  library(lubridate)
  library(purrr)
  library(readr)
  library(stringr)
  library(survival)
  library(tibble)
  library(tidyr)
})

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR       <- Sys.getenv("OPERA_OUTCOMES_DIR", "")
if (!nzchar(OUTPUT_DIR) || !dir.exists(OUTPUT_DIR))
  stop("Set OPERA_OUTCOMES_DIR to the completed outcome directory")
COHORT_COL       <- "cohort_fine"
HORIZON_DAYS     <- 365L      # fixed-horizon reference (e.g. 1-year risk)
VIABILITY_EVENTS <- 50L       # test events below this flagged as thin
OS_OUTCOME       <- "overall_survival"  # parquet whose event date IS death,
                                        # used to recover death as a competing risk
MIN_AT_RISK      <- 20L       # truncate each split's incidence curve once the
                              # at-risk set falls below this (stops the recent
                              # test stratum's unstable right tail being read)

# ── BCE-at-horizon viability ──────────────────────────────────────────────────
# A binary "event within H days" label is only scoreable for patients whose
# label is determined by H: positives (event <= H), negatives (event-free
# through H, incl. event-after-H and, under the cause-specific convention,
# death-before-H-without-event), and indeterminate (admin-censored before H,
# still event-free and alive -> dropped). Viability is about POSITIVES and
# NEGATIVES at H in the TEST split, not events ever.
BCE_HORIZONS            <- c(30L, 90L, 180L, 365L, 730L, 1095L)  # horizons to test
                              # short windows (30/90/180) suit acute outcomes
                              # (hospitalisation, sepsis, AKI, early cytopenias)
                              # and have little truncation; long windows suit the
                              # slow clinical endpoints. Each outcome is gated and
                              # reported at its own smallest viable horizon.
MIN_POS                 <- 50L    # min positives (and negatives) per class at H
MAX_INDETERMINATE       <- 0.50   # max fraction of the cohort dropped as unlabelled
DRIFT_TOL               <- 0.05   # |CIF_test - CIF_train| at H above this = drift flag
# Label convention for a competing death before H without the outcome:
#   TRUE  = cause-specific: it is a NEGATIVE (outcome did not occur by H).
#           Consistent with the Aalen-Johansen panel. The eval code MUST match.
#   FALSE = death treated as censoring: it is INDETERMINATE (dropped).
COMPETING_DEATH_AS_NEGATIVE <- TRUE
FOCAL_HORIZON           <- 365L   # fallback horizon for the per-cohort BCE plot
                                  # (outcomes with no viable horizon); viable
                                  # outcomes are shown at their own recommended H

# Cohort granularity hierarchy. Viability is reported at each level present in
# first_line (finest first), plus an "overall" pooled estimate (cohort = "ALL").
# The per-cohort cumulative-incidence plot uses cohort_grouped.
COHORT_LEVELS    <- c("cohort_fine", "cohort_grouped")  # missing ones are skipped
FOCAL_OUTCOME    <- "second_line_treatment"  # outcome for the per-cohort_grouped CIF
SUMMARY_DIR      <- file.path(OUTPUT_DIR, "_split_eval")
dir.create(SUMMARY_DIR, showWarnings = FALSE, recursive = TRUE)

# Candidate cutoffs to inspect in detail (index YEARS). train < tuning_start;
# tuning_start <= tuning < test_start; test >= test_start.
TUNING_START_YEAR <- 2022L
TEST_START_YEAR   <- 2023L
# Candidate test-start years to sweep:
SWEEP_TEST_YEARS  <- 2022:2023

firstline_mapped_path <- Sys.getenv("OPERA_FIRSTLINE_MAPPED_CSV", "")
if (!nzchar(firstline_mapped_path) || !file.exists(firstline_mapped_path))
  stop("Set OPERA_FIRSTLINE_MAPPED_CSV to the sealed mapped first-line CSV")
firstline_tbl_mapped <- read_csv(firstline_mapped_path)

first_line <- firstline_tbl_mapped

first_line <- first_line %>% 
  filter(cohort_grouped != "EXCLUDE_SECONDARY")

# ── Cohort lookup ─────────────────────────────────────────────────────────────
if (!exists("first_line"))
  stop("`first_line` not found. Define it (patientid, index_date, cohort) first.")
if (!COHORT_COL %in% names(first_line))
  stop(sprintf("`first_line` has no '%s' column. Add the disease subtype to first_line.", COHORT_COL))
cohort_lookup <- first_line %>% transmute(subject_id = patientid, cohort = .data[[COHORT_COL]])

# Granularity levels actually available in first_line (finest first).
PRESENT_LEVELS <- intersect(COHORT_LEVELS, names(first_line))
if (length(PRESENT_LEVELS) == 0)
  stop("None of COHORT_LEVELS present in first_line: ",
       paste(COHORT_LEVELS, collapse = ", "))
level_lookup <- first_line %>%
  select(subject_id = patientid, all_of(PRESENT_LEVELS)) %>%
  distinct(subject_id, .keep_all = TRUE)


# ── Read outcome parquets into one per-patient table ──────────────────────────
files <- list.files(OUTPUT_DIR, pattern = "\\.parquet$", full.names = TRUE)
files <- files[!str_detect(basename(files), "__")]
if (length(files) == 0) stop("No outcome parquets found in ", OUTPUT_DIR)

read_one <- function(path) {
  read_parquet(path) %>%
    transmute(
      subject_id,
      index_date   = as.Date(index_date),
      index_year   = year(as.Date(index_date)),
      outcome_date = as.Date(outcome_date),
      event        = !is.na(outcome_date),
      followup_days = as.numeric(as.Date(censor_date) - as.Date(index_date)),
      tte_days     = as.numeric(as.Date(outcome_date) - as.Date(index_date)),
      outcome      = str_remove(basename(path), "\\.parquet$"))
}

patients <- bind_rows(lapply(files, read_one)) %>%
  left_join(cohort_lookup, by = "subject_id") %>%
  left_join(level_lookup, by = "subject_id") %>%
  mutate(across(any_of(c("cohort", PRESENT_LEVELS)), ~ coalesce(.x, "unknown")))


# ── Split assignment from index year ──────────────────────────────────────────
assign_split <- function(yr, tuning_start, test_start)
  factor(case_when(yr <  tuning_start ~ "train",
                   yr <  test_start   ~ "tuning",
                   TRUE               ~ "test"),
         levels = c("train", "tuning", "test"))


# ── Evaluate one candidate split ──────────────────────────────────────────────
evaluate_split <- function(tuning_start, test_start,
                           by_cohort = FALSE, horizon = HORIZON_DAYS, write = TRUE) {

  grp <- if (by_cohort) c("outcome", "cohort", "split") else c("outcome", "split")

  tab <- patients %>%
    mutate(split = assign_split(index_year, tuning_start, test_start)) %>%
    group_by(across(all_of(grp))) %>%
    summarise(
      n_eligible   = n(),
      n_events     = sum(event),
      event_rate   = mean(event),
      events_pre_h = sum(event & tte_days <= horizon),         # events seen within horizon
      followup_median = median(followup_days, na.rm = TRUE),
      followup_p25 = quantile(followup_days, 0.25, na.rm = TRUE),
      followup_p75 = quantile(followup_days, 0.75, na.rm = TRUE),
      pct_followup_ge_h = mean(followup_days >= horizon, na.rm = TRUE),
      tte_median   = median(tte_days[event], na.rm = TRUE),
      tte_p25      = quantile(tte_days[event], 0.25, na.rm = TRUE),
      tte_p75      = quantile(tte_days[event], 0.75, na.rm = TRUE),
      .groups = "drop") %>%
    arrange(outcome, if (by_cohort) cohort, split)

  if (write) {
    f <- file.path(SUMMARY_DIR, sprintf("split_eval_t%d_T%d%s.csv",
                                        tuning_start, test_start,
                                        if (by_cohort) "_bycohort" else ""))
    write_csv(tab, f)
  }
  tab
}


# ── Sweep test-start years: test events + follow-up per outcome ───────────────
sweep_test_start <- function(test_years = SWEEP_TEST_YEARS, horizon = HORIZON_DAYS) {
  res <- lapply(test_years, function(ts) {
    patients %>%
      filter(index_year >= ts) %>%
      group_by(outcome) %>%
      summarise(test_start = ts,
                test_n = n(),
                test_events = sum(event),
                test_events_pre_h = sum(event & tte_days <= horizon),
                test_followup_median = median(followup_days, na.rm = TRUE),
                .groups = "drop")
  })
  tab <- bind_rows(res) %>% arrange(outcome, test_start)
  write_csv(tab, file.path(SUMMARY_DIR, "sweep_test_start.csv"))
  tab
}


# ── Run for the chosen candidate ──────────────────────────────────────────────
eval_tab  <- evaluate_split(TUNING_START_YEAR, TEST_START_YEAR)
sweep_tab <- sweep_test_start()

# console: outcomes thin or short-followed in the test split for this cutoff
test_check <- eval_tab %>%
  filter(split == "test") %>%
  mutate(flag = case_when(n_events < VIABILITY_EVENTS ~ "few_test_events",
                          pct_followup_ge_h < 0.5     ~ "short_test_followup",
                          TRUE                        ~ "ok")) %>%
  arrange(n_events)
message(sprintf("\nTest split at TEST_START = %d (horizon %d days, threshold %d events):",
                TEST_START_YEAR, HORIZON_DAYS, VIABILITY_EVENTS))
print(test_check %>% select(outcome, n_eligible, n_events, pct_followup_ge_h,
                            followup_median, tte_median, flag), n = 100)


# ── Plots for the chosen candidate ────────────────────────────────────────────
n_out <- n_distinct(eval_tab$outcome)

# events per split
p_events <- ggplot(eval_tab, aes(reorder(outcome, n_events), n_events, fill = split)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75, alpha= 0.2) +
  geom_hline(yintercept = VIABILITY_EVENTS, linetype = 2, color = "red") +
  coord_flip() +
  labs(title = sprintf("Events per split (tuning>=%d, test>=%d)",
                       TUNING_START_YEAR, TEST_START_YEAR),
       subtitle = sprintf("dashed line = %d events", VIABILITY_EVENTS),
       x = NULL, y = "events", fill = NULL) +
  theme_minimal(base_size = 9)
ggsave(file.path(SUMMARY_DIR, "events_per_split.pdf"), p_events,
       width = 9, height = max(5, 0.22 * n_out + 2), limitsize = FALSE)

# follow-up per split (median with IQR)
p_fu <- ggplot(eval_tab, aes(reorder(outcome, followup_median), followup_median, color = split)) +
  geom_point(position = position_dodge(width = 0.6), size = 1.6) +
  geom_errorbar(aes(ymin = followup_p25, ymax = followup_p75),
                position = position_dodge(width = 0.6), width = 0) +
  geom_hline(yintercept = HORIZON_DAYS, linetype = 2, color = "grey40") +
  coord_flip() +
  labs(title = "Observed follow-up per split (median, IQR)",
       subtitle = sprintf("dashed line = %d-day horizon", HORIZON_DAYS),
       x = NULL, y = "censor_date - index_date (days)", color = NULL) +
  theme_minimal(base_size = 9)
ggsave(file.path(SUMMARY_DIR, "followup_per_split.pdf"), p_fu,
       width = 9, height = max(5, 0.22 * n_out + 2), limitsize = FALSE)

# Observed time-to-event AMONG CASES (exploratory only). This is the empirical
# distribution conditioned on having been observed to event, so it is length-
# biased toward early events: a recently indexed patient can only contribute an
# event if it happened fast, while their slow events are still pending and
# invisible. The pull is negligible for short-latency outcomes (almost everyone
# who will event already has) and strong for long-latency ones (overall_survival,
# second_line_treatment, next_malignancy, MACE), making them look more catchable
# in a short window than they are. Keep as a quick descriptive look; use the
# cumulative-incidence panel below for any timing claim.
p_tte <- patients %>% filter(event) %>%
  ggplot(aes(tte_days)) +
  stat_ecdf(geom = "step") +
  geom_vline(xintercept = HORIZON_DAYS, linetype = 2, color = "red") +
  facet_wrap(~ outcome, scales = "free_x") +
  labs(title = "Observed time to event among cases",
       subtitle = sprintf("uncensored, length-biased toward early events at recent index years; red line = %d-day horizon", HORIZON_DAYS),
       x = "days from index to event", y = "cumulative fraction of OBSERVED events") +
  theme_minimal(base_size = 7)
ggsave(file.path(SUMMARY_DIR, "observed_tte_among_cases.pdf"), p_tte,
       width = 12, height = max(6, 0.6 * sqrt(n_out) + 4), limitsize = FALSE)

p_tte <- patients %>% filter(event) %>%
  ggplot(aes(tte_days)) +
  stat_ecdf(geom = "step") +
  geom_vline(xintercept = HORIZON_DAYS, linetype = 2, color = "red") +
  facet_wrap(~ outcome, scales = "free_x") +
  labs(title = "Observed time to event among cases",
       subtitle = sprintf("uncensored, length-biased toward early events at recent index years; red line = %d-day horizon", HORIZON_DAYS),
       x = "days from index to event", y = "cumulative fraction of OBSERVED events") +
  coord_cartesian(ylim = c(0, 1), xlim = c(0, 365 * 3))  +
  theme_minimal(base_size = 7)
ggsave(file.path(SUMMARY_DIR, "observed_tte_among_cases_3_years.pdf"), p_tte,
       width = 12, height = max(6, 0.6 * sqrt(n_out) + 4), limitsize = FALSE)


# ── Formal timing: cumulative incidence by split (competing-risks aware) ──────
# Replaces the among-cases ECDF for any actual claim. Death is recovered from the
# OS parquet (its event date IS death) and treated as a competing risk via the
# Aalen-Johansen estimator, so a patient who dies without the outcome is removed
# from the at-risk set rather than censored. For overall_survival there is no
# competing event (death IS the event), so AJ reduces exactly to 1 - KM. Curves
# are stratified by temporal split, which doubles as a validity check on the
# split itself: overlapping curves mean a temporally stable event process; curves
# that separate mean calendar-time drift in the outcome, i.e. the split is testing
# on a different process than it trained on. Each stratum is truncated where its
# at-risk set drops below MIN_AT_RISK and bands are 95% CI, so the short-follow-up
# test tail cannot be overread.

# Recover death date (absolute) from the OS parquet.
death_lookup <- patients %>%
  filter(outcome == OS_OUTCOME, event) %>%
  transmute(subject_id, death_date = outcome_date) %>%
  distinct(subject_id, .keep_all = TRUE)
has_death <- nrow(death_lookup) > 0
if (!has_death)
  message("NOTE: no '", OS_OUTCOME, "' parquet with events found; cumulative ",
          "incidence will treat death as censoring (no competing risk).")

# Per (patient, outcome) competing-risks status: 1 = the outcome, 2 = death
# without the outcome, 0 = administrative censoring. time is days from that
# outcome's own index_date to whichever happened.
patients_cr <- patients %>%
  mutate(split = assign_split(index_year, TUNING_START_YEAR, TEST_START_YEAR)) %>%
  left_join(death_lookup, by = "subject_id") %>%
  mutate(
    death_days = as.numeric(death_date - index_date),
    status = case_when(
      event                                              ~ 1L,
      !is.na(death_days) & death_days <= followup_days + 0.5 ~ 2L,
      TRUE                                               ~ 0L),
    time = dplyr::case_when(
      status == 1L ~ tte_days,
      status == 2L ~ death_days,
      TRUE         ~ followup_days)) %>%
  filter(!is.na(time), time >= 0) %>%
  mutate(time = pmax(time, 0.5))       # avoid t = 0 in the estimator

mk_status_factor <- function(s) {
  labs <- c(`0` = "censored", `1` = "event", `2` = "death")
  present <- labs[as.character(sort(unique(s)))]
  factor(labs[as.character(s)],
         levels = c("censored", setdiff(present, "censored")))
}

# Extract per-stratum incidence curve for the "event" state from a survfit object.
extract_curves <- function(fit) {
  strata <- fit$strata
  if (is.null(strata)) strata <- c(all = length(fit$time))
  idx_end   <- cumsum(strata)
  idx_start <- c(1L, head(idx_end, -1) + 1L)
  has_ps <- !is.null(fit$pstate)
  if (has_ps) {
    ev_col <- match("event", fit$states)
    s0_col <- match("(s0)",  fit$states); if (is.na(s0_col)) s0_col <- 1L
  }
  purrr::map_dfr(seq_along(strata), function(i) {
    r <- idx_start[i]:idx_end[i]
    sname <- sub("split=", "", names(strata)[i])
    if (has_ps) {
      nr <- if (is.matrix(fit$n.risk)) fit$n.risk[r, s0_col] else fit$n.risk[r]
      tibble(split = sname, time = fit$time[r],
             cif   = fit$pstate[r, ev_col],
             lower = if (!is.null(fit$lower)) fit$lower[r, ev_col] else NA_real_,
             upper = if (!is.null(fit$upper)) fit$upper[r, ev_col] else NA_real_,
             n_risk = nr)
    } else {
      tibble(split = sname, time = fit$time[r],
             cif   = 1 - fit$surv[r],
             lower = if (!is.null(fit$upper)) 1 - fit$upper[r] else NA_real_,
             upper = if (!is.null(fit$lower)) 1 - fit$lower[r] else NA_real_,
             n_risk = fit$n.risk[r])
    }
  })
}

cif_one <- function(oc) {
  d <- patients_cr %>% filter(outcome == oc)
  if (sum(d$status == 1L) < 1L || n_distinct(d$split) < 1L) return(NULL)
  d$status_f <- mk_status_factor(d$status)
  fit <- tryCatch(
    survfit(Surv(time, status_f) ~ split, data = d, conf.int = 0.95),
    error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  out <- extract_curves(fit)
  out$outcome <- oc
  out
}

cif_df <- purrr::map_dfr(sort(unique(patients_cr$outcome)), cif_one) %>%
  mutate(split = factor(split, levels = c("train", "tuning", "test"))) %>%
  filter(n_risk >= MIN_AT_RISK)          # horizon-cap each stratum
write_csv(cif_df, file.path(SUMMARY_DIR, "cumulative_incidence_by_split.csv"))

p_cif <- ggplot(cif_df, aes(time, cif, color = split, fill = split)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.12, color = NA) +
  geom_step() +
  geom_vline(xintercept = HORIZON_DAYS, linetype = 2, color = "grey40") +
  facet_wrap(~ outcome, scales = "free_x") +
  coord_cartesian(ylim = c(0, 1), xlim = c(0, 365 * 3)) +
  labs(title = sprintf("Cumulative incidence by split%s",
                       if (has_death) " (Aalen-Johansen, death as competing risk)"
                       else " (KM; death treated as censoring -- no OS parquet)"),
       subtitle = sprintf("overall_survival reduces to 1 - KM; curves truncated where at-risk < %d; bands = 95%% CI; dashed = %d-day horizon",
                          MIN_AT_RISK, HORIZON_DAYS),
       x = "days from index", y = "cumulative incidence", color = NULL, fill = NULL) +
  theme_minimal(base_size = 7)
ggsave(file.path(SUMMARY_DIR, "cumulative_incidence_by_split.pdf"), p_cif,
       width = 12, height = max(6, 0.6 * sqrt(n_out) + 4), limitsize = FALSE)

# sweep: test events vs candidate test-start, heatmap with threshold flag
p_sweep <- ggplot(sweep_tab, aes(factor(test_start), reorder(outcome, test_events),
                                 fill = test_events)) +
  geom_tile(color = "grey92") +
  geom_text(aes(label = test_events, color = test_events < VIABILITY_EVENTS), size = 2.3) +
  scale_color_manual(values = c(`TRUE` = "red", `FALSE` = "grey20"), guide = "none") +
  scale_fill_viridis_c(trans = "log1p", breaks = c(0, 50, 200, 1000)) +
  labs(title = "Test events vs candidate TEST_START",
       subtitle = sprintf("red = fewer than %d test events", VIABILITY_EVENTS),
       x = "candidate TEST_START (test = this index year onward)", y = NULL,
       fill = "test events") +
  theme_minimal(base_size = 8)
ggsave(file.path(SUMMARY_DIR, "sweep_test_events.pdf"), p_sweep,
       width = 8, height = max(6, 0.2 * n_out + 2), limitsize = FALSE)

# ── Cohort breakdown for the chosen candidate ─────────────────────────────────
# Eligibility is the shared coverage gate, so within each split every lab outcome
# has the same per-cohort denominator; what varies is the event count. The test
# cells are what gate per-cohort claims (training is pooled, so thin per-cohort
# train cells are tolerable). COHORT_COL should point at the coarse grouping;
# rerun with it set to the fine diagnosis to see the finer cells.
eval_cohort <- evaluate_split(TUNING_START_YEAR, TEST_START_YEAR, by_cohort = TRUE)
n_coh <- n_distinct(eval_cohort$cohort)

# per-cohort denominator per split (shared across lab outcomes)
cohort_sizes <- eval_cohort %>%
  group_by(cohort, split) %>%
  summarise(n_eligible = max(n_eligible), .groups = "drop") %>%
  pivot_wider(names_from = split, values_from = n_eligible, values_fill = 0)
write_csv(cohort_sizes, file.path(SUMMARY_DIR, "cohort_sizes_by_split.csv"))
message("\nPer-cohort eligible counts by split:")
print(cohort_sizes, n = 100)

# test cells too thin for a per-cohort claim at this cutoff
cohort_test_thin <- eval_cohort %>%
  filter(split == "test", n_events < VIABILITY_EVENTS) %>%
  arrange(outcome, n_events) %>%
  select(outcome, cohort, n_eligible, n_events, pct_followup_ge_h, tte_median)
message(sprintf("\nTest outcome x cohort cells below %d events at TEST_START = %d:",
                VIABILITY_EVENTS, TEST_START_YEAR))
print(cohort_test_thin, n = 200)

# events per outcome x cohort, faceted by split
p_cohort <- ggplot(eval_cohort, aes(cohort, reorder(outcome, n_events), fill = n_events)) +
  geom_tile(color = "grey92") +
  geom_text(aes(label = n_events, color = n_events < VIABILITY_EVENTS), size = 1.9) +
  scale_color_manual(values = c(`TRUE` = "red", `FALSE` = "grey20"), guide = "none") +
  scale_fill_viridis_c(trans = "log1p", breaks = c(0, 50, 200, 1000)) +
  facet_wrap(~ split) +
  labs(title = sprintf("Events per outcome x cohort by split (tuning>=%d, test>=%d)",
                       TUNING_START_YEAR, TEST_START_YEAR),
       subtitle = sprintf("red = fewer than %d events", VIABILITY_EVENTS),
       x = "cohort", y = NULL, fill = "events") +
  theme_minimal(base_size = 7) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
ggsave(file.path(SUMMARY_DIR, "events_by_outcome_cohort_split.pdf"), p_cohort,
       width = max(9, 1.1 * n_coh * 3 + 2), height = max(6, 0.18 * n_out + 2),
       limitsize = FALSE)

message("\nWrote split-evaluation tables and plots to ", SUMMARY_DIR)


# ==============================================================================
# BCE-at-horizon viability + final outcome manifest (the send-off artifact)
# ==============================================================================
# Uses patients_cr (status 1/2/0, time, split) built in the cumulative-incidence
# section. For each outcome x split x horizon it counts the patients whose binary
# "event within H" label is determined, so we can answer "are there enough events
# AT THE HORIZON for a BCE test", not just "enough events ever".

label_counts_note <- "n_pos: event<=H | n_neg: event-free through H | n_indet: censored<H"

bce_tab <- purrr::map_dfr(BCE_HORIZONS, function(H) {
  patients_cr %>%
    group_by(outcome, split) %>%
    summarise(
      horizon   = H,
      n_pos     = sum(status == 1L & time <= H),
      ev_after  = sum(status == 1L & time >  H),   # event after H -> negative
      free_to_H = sum(status == 0L & time >= H),    # followed event-free to H
      death_lt  = sum(status == 2L & time <  H),    # died before H, no outcome
      death_ge  = sum(status == 2L & time >= H),    # alive event-free at H, died later
      cens_lt   = sum(status == 0L & time <  H),    # admin-censored before H
      .groups = "drop")
}) %>%
  mutate(
    n_neg   = if (COMPETING_DEATH_AS_NEGATIVE)
                ev_after + free_to_H + death_lt + death_ge
              else
                ev_after + free_to_H + death_ge,
    n_indet = if (COMPETING_DEATH_AS_NEGATIVE) cens_lt else cens_lt + death_lt,
    evaluable  = n_pos + n_neg,
    pos_rate   = ifelse(evaluable > 0, n_pos / evaluable, NA_real_),
    indet_rate = ifelse(evaluable + n_indet > 0,
                        n_indet / (evaluable + n_indet), NA_real_),
    viable_bce = n_pos >= MIN_POS & n_neg >= MIN_POS &
                 indet_rate <= MAX_INDETERMINATE) %>%
  select(outcome, split, horizon, n_pos, n_neg, n_indet,
         evaluable, pos_rate, indet_rate, viable_bce) %>%
  arrange(outcome, split, horizon)
write_csv(bce_tab, file.path(SUMMARY_DIR, "bce_viability_by_horizon.csv"))

# Recommended horizon = smallest horizon viable in the TEST split.
rec_horizon <- bce_tab %>%
  filter(split == "test", viable_bce) %>%
  group_by(outcome) %>%
  summarise(rec_horizon = min(horizon), .groups = "drop")

# Temporal drift: |CIF_test - CIF_train| evaluated at each outcome's horizon
# (default 365 if no viable horizon). Uses the truncated cif_df from above.
drift_wide <- cif_df %>%
  left_join(rec_horizon, by = "outcome") %>%
  mutate(H = coalesce(rec_horizon, 365L)) %>%
  filter(time <= H) %>%
  group_by(outcome, split) %>%
  slice_max(time, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(outcome, split, cif) %>%
  pivot_wider(names_from = split, values_from = cif, names_prefix = "cif_")
for (cc in c("cif_train", "cif_tuning", "cif_test"))
  if (!cc %in% names(drift_wide)) drift_wide[[cc]] <- NA_real_
drift_tab <- drift_wide %>%
  mutate(drift = ifelse(!is.na(cif_train) & !is.na(cif_test),
                        abs(cif_test - cif_train), NA_real_),
         drift_flag = !is.na(drift) & drift > DRIFT_TOL)

# Per-cohort: how many cohorts clear the test floor for this outcome.
cohort_claim <- eval_cohort %>%
  filter(split == "test") %>%
  group_by(outcome) %>%
  summarise(
    n_cohorts_test_viable = sum(n_events >= VIABILITY_EVENTS),
    cohorts_test_viable   = paste(sort(cohort[n_events >= VIABILITY_EVENTS]),
                                  collapse = ";"),
    .groups = "drop")

# Test totals at the chosen single cutoff (for context alongside the at-horizon
# numbers).
test_totals <- eval_tab %>%
  filter(split == "test") %>%
  transmute(outcome, test_n_eligible = n_eligible, test_n_events_ever = n_events,
            test_pct_followup_ge_h = pct_followup_ge_h, tte_median)

# Per-outcome at-horizon test counts at the recommended (or default 365) horizon.
bce_at_rec <- bce_tab %>%
  filter(split == "test") %>%
  left_join(rec_horizon, by = "outcome") %>%
  mutate(use_H = coalesce(rec_horizon, 365L)) %>%
  filter(horizon == use_H) %>%
  transmute(outcome, horizon, test_n_pos = n_pos, test_n_neg = n_neg,
            test_indet_rate = round(indet_rate, 3))

manifest <- test_totals %>%
  left_join(rec_horizon, by = "outcome") %>%
  left_join(bce_at_rec %>% select(-horizon), by = "outcome") %>%
  left_join(drift_tab %>% select(outcome, drift, drift_flag), by = "outcome") %>%
  left_join(cohort_claim, by = "outcome") %>%
  mutate(
    horizon_days = rec_horizon,
    verdict = case_when(
      is.na(rec_horizon)                 ~ "pooled_only",
      drift_flag %in% TRUE               ~ "viable_check_drift",
      TRUE                               ~ "viable_per_outcome"),
    per_cohort_claim = ifelse(coalesce(n_cohorts_test_viable, 0L) >= 1L,
                              "supported_for_listed_cohorts", "pooled_only")) %>%
  select(outcome, verdict, horizon_days, test_n_pos, test_n_neg, test_indet_rate,
         test_n_events_ever, test_pct_followup_ge_h, tte_median,
         drift, drift_flag, per_cohort_claim,
         n_cohorts_test_viable, cohorts_test_viable) %>%
  arrange(factor(verdict, levels = c("viable_per_outcome", "viable_check_drift",
                                     "pooled_only")),
          desc(test_n_pos))
write_csv(manifest, file.path(SUMMARY_DIR, "outcome_manifest.csv"))

message("\nOutcome manifest verdict counts:")
print(manifest %>% count(verdict))
message("\nOutcomes pooled-only (no horizon with >= ", MIN_POS,
        " test positives & negatives):")
print(manifest %>% filter(verdict == "pooled_only") %>%
        select(outcome, test_n_events_ever, tte_median), n = 100)

# BCE positives at each horizon, TEST split: the visual answer to "enough events
# at the right horizon". Red = below MIN_POS positives.
p_bce <- bce_tab %>% filter(split == "test") %>%
  ggplot(aes(factor(horizon), reorder(outcome, n_pos), fill = n_pos)) +
  geom_tile(color = "grey92") +
  geom_text(aes(label = n_pos, color = n_pos < MIN_POS), size = 2.2) +
  scale_color_manual(values = c(`TRUE` = "red", `FALSE` = "grey20"), guide = "none") +
  scale_fill_viridis_c(trans = "log1p", breaks = c(0, 50, 200, 1000)) +
  labs(title = "Test-split POSITIVES at each BCE horizon",
       subtitle = sprintf("positives = event within H; red = fewer than %d (thin for a BCE test at that horizon)", MIN_POS),
       x = "horizon H (days)", y = NULL, fill = "test positives") +
  theme_minimal(base_size = 8)
ggsave(file.path(SUMMARY_DIR, "bce_positives_by_horizon.pdf"), p_bce,
       width = 8, height = max(6, 0.2 * n_out + 2), limitsize = FALSE)

message("\nWrote BCE-viability tables, manifest, and plots to ", SUMMARY_DIR)


# ==============================================================================
# BCE viability sweep: horizon x candidate test-start x cohort
# ==============================================================================
# The full feasibility map. For every binary endpoint it asks, at each candidate
# test-start year (test = index_year >= ts) and each horizon, whether the test
# set has enough determined labels. This surfaces the core tension: a later ts
# means recent, short-follow-up patients, so a longer horizon buys positives only
# until truncation pushes them into the indeterminate set. cohort = "ALL" is the
# pooled view; the per-cohort rows show where per-cohort test claims survive.
#
# Built once from patient x outcome x index_year blocks, then summed over
# index_year >= ts, so the 6 x 4 x (cohorts+1) cross is cheap.

block_cols <- c("n_pos", "ev_after", "free_to_H", "death_lt", "death_ge", "cens_lt")

# Blocks at one granularity. level_col = NULL -> overall pooled (cohort "ALL").
make_blocks <- function(level_col) {
  base <- patients_cr
  if (is.null(level_col)) {
    base <- base %>% mutate(cohort_level = "overall", cohort = "ALL")
  } else {
    base <- base %>% mutate(cohort_level = level_col,
                            cohort = as.character(.data[[level_col]]))
  }
  purrr::map_dfr(BCE_HORIZONS, function(H) {
    base %>%
      group_by(outcome, cohort_level, cohort, index_year) %>%
      summarise(
        horizon   = H,
        n_pos     = sum(status == 1L & time <= H),
        ev_after  = sum(status == 1L & time >  H),
        free_to_H = sum(status == 0L & time >= H),
        death_lt  = sum(status == 2L & time <  H),
        death_ge  = sum(status == 2L & time >= H),
        cens_lt   = sum(status == 0L & time <  H),
        .groups = "drop")
  })
}

# overall + every granularity present in patients_cr (cohort_grouped, cohort_fine)
levels_to_run <- c(list(NULL), as.list(intersect(PRESENT_LEVELS, names(patients_cr))))
blocks <- purrr::map_dfr(levels_to_run, make_blocks)

finalize_bce <- function(df) {
  df %>% mutate(
    n_neg = if (COMPETING_DEATH_AS_NEGATIVE)
              ev_after + free_to_H + death_lt + death_ge
            else
              ev_after + free_to_H + death_ge,
    n_indet = if (COMPETING_DEATH_AS_NEGATIVE) cens_lt else cens_lt + death_lt,
    evaluable  = n_pos + n_neg,
    pos_rate   = ifelse(evaluable > 0, n_pos / evaluable, NA_real_),
    indet_rate = ifelse(evaluable + n_indet > 0,
                        n_indet / (evaluable + n_indet), NA_real_),
    viable_bce = n_pos >= MIN_POS & n_neg >= MIN_POS &
                 indet_rate <= MAX_INDETERMINATE)
}

bce_sweep <- purrr::map_dfr(SWEEP_TEST_YEARS, function(ts) {
  blocks %>%
    filter(index_year >= ts) %>%
    group_by(outcome, cohort_level, cohort, horizon) %>%
    summarise(test_start = ts, across(all_of(block_cols), sum), .groups = "drop")
}) %>%
  finalize_bce() %>%
  select(outcome, cohort_level, cohort, test_start, horizon, n_pos, n_neg, n_indet,
         evaluable, pos_rate, indet_rate, viable_bce) %>%
  arrange(outcome, cohort_level, cohort, test_start, horizon)
write_csv(bce_sweep, file.path(SUMMARY_DIR, "bce_viability_sweep.csv"))

# ── Plot A: pooled positives, horizon x test-start ────────────────────────────
p_sweep_h <- bce_sweep %>% filter(cohort_level == "overall") %>%
  ggplot(aes(factor(test_start), reorder(outcome, n_pos), fill = n_pos)) +
  geom_tile(color = "grey92") +
  geom_text(aes(label = n_pos, color = n_pos < MIN_POS), size = 1.7) +
  scale_color_manual(values = c(`TRUE` = "red", `FALSE` = "grey20"), guide = "none") +
  scale_fill_viridis_c(trans = "log1p", breaks = c(0, 50, 200, 1000)) +
  facet_wrap(~ horizon, nrow = 1, labeller = labeller(horizon = function(x) paste0("H=", x, "d"))) +
  labs(title = "BCE positives in the test set: horizon x test-start (all cohorts)",
       subtitle = sprintf("positives = event within H among index_year >= test-start; red = fewer than %d", MIN_POS),
       x = "candidate test-start year", y = NULL, fill = "test positives") +
  theme_minimal(base_size = 7) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
ggsave(file.path(SUMMARY_DIR, "bce_positives_horizon_x_teststart.pdf"), p_sweep_h,
       width = max(10, 2.4 * length(BCE_HORIZONS) + 2),
       height = max(6, 0.2 * n_out + 2), limitsize = FALSE)

# ── Plot B: smallest viable horizon per outcome x test-start (feasibility map) ─
min_viable <- bce_sweep %>%
  filter(cohort_level == "overall", viable_bce) %>%
  group_by(outcome, test_start) %>%
  summarise(min_h = min(horizon), .groups = "drop")
feasibility <- tidyr::expand_grid(outcome = sort(unique(bce_sweep$outcome)),
                                  test_start = SWEEP_TEST_YEARS) %>%
  left_join(min_viable, by = c("outcome", "test_start")) %>%
  mutate(min_h_lab = ifelse(is.na(min_h), "none", paste0(min_h, "d")),
         min_h_lab = factor(min_h_lab,
                            levels = c(paste0(sort(BCE_HORIZONS), "d"), "none")))
# order outcomes by how achievable they are (count of viable test-starts)
out_order <- feasibility %>% group_by(outcome) %>%
  summarise(n_ok = sum(!is.na(min_h)), best = min(min_h, na.rm = TRUE), .groups = "drop") %>%
  mutate(best = ifelse(is.infinite(best), 99999, best)) %>%
  arrange(desc(n_ok), best) %>% pull(outcome)
feasibility <- feasibility %>% mutate(outcome = factor(outcome, levels = rev(out_order)))
p_feasible <- ggplot(feasibility, aes(factor(test_start), outcome, fill = min_h_lab)) +
  geom_tile(color = "grey92") +
  scale_fill_viridis_d(option = "D", direction = -1, na.value = "grey85",
                       drop = FALSE) +
  labs(title = "Smallest viable BCE horizon per outcome x test-start (all cohorts)",
       subtitle = sprintf("'none' = no horizon up to %dd clears %d pos/neg with indet <= %.0f%%",
                          max(BCE_HORIZONS), MIN_POS, 100 * MAX_INDETERMINATE),
       x = "candidate test-start year", y = NULL, fill = "min viable H") +
  theme_minimal(base_size = 7)
ggsave(file.path(SUMMARY_DIR, "bce_min_viable_horizon.pdf"), p_feasible,
       width = 8, height = max(6, 0.2 * n_out + 2), limitsize = FALSE)

# ── Plot C: per-cohort positives at EACH OUTCOME'S horizon, across test-start ──
# Uses the per-outcome recommended horizon (smallest viable in the pooled test),
# falling back to FOCAL_HORIZON for outcomes with no viable horizon, so a slow
# outcome's cohorts are shown at the horizon they would actually be tested at
# rather than a flat 365.
p_cohort_bce <- bce_sweep %>%
  filter(cohort_level == "cohort_grouped") %>%
  left_join(rec_horizon, by = "outcome") %>%
  mutate(use_H = coalesce(rec_horizon, FOCAL_HORIZON)) %>%
  filter(horizon == use_H) %>%
  mutate(outcome_lab = paste0(outcome, " (H=", use_H, "d)")) %>%
  ggplot(aes(cohort, reorder(outcome_lab, n_pos), fill = n_pos)) +
  geom_tile(color = "grey92") +
  geom_text(aes(label = n_pos, color = n_pos < MIN_POS), size = 1.5) +
  scale_color_manual(values = c(`TRUE` = "red", `FALSE` = "grey20"), guide = "none") +
  scale_fill_viridis_c(trans = "log1p", breaks = c(0, 50, 200, 1000)) +
  facet_wrap(~ test_start, nrow = 1) +
  labs(title = "Per-cohort BCE positives at each outcome's recommended horizon, across test-start",
       subtitle = sprintf("horizon shown per row; red = fewer than %d positives; cohort test claims survive only where green", MIN_POS),
       x = "cohort", y = NULL, fill = "test positives") +
  theme_minimal(base_size = 6) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
ggsave(file.path(SUMMARY_DIR, "bce_cohort_positives_x_teststart.pdf"), p_cohort_bce,
       width = max(12, 1.0 * n_coh * length(SWEEP_TEST_YEARS) + 2),
       height = max(6, 0.18 * n_out + 2), limitsize = FALSE)

# console: pooled viability count per (test_start, horizon)
sweep_summary <- bce_sweep %>%
  filter(cohort_level == "overall") %>%
  group_by(test_start, horizon) %>%
  summarise(n_viable_outcomes = sum(viable_bce, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = horizon, values_from = n_viable_outcomes,
              names_prefix = "H")
message("\nViable outcomes (all cohorts) by test-start x horizon:")
print(sweep_summary, n = 100)

message("\nWrote BCE viability sweep (horizon x test-start x cohort) to ", SUMMARY_DIR)


# ── Per-cohort_grouped cumulative incidence for the focal outcome ─────────────
# The coarse-grained timing/drift view: same Aalen-Johansen estimator as the
# overall panel, but within each cohort_grouped, so cohort-specific drift that
# the pooled curve hides becomes visible. Bounded to one outcome to stay legible.
if (FOCAL_OUTCOME %in% patients_cr$outcome) {
  focal_cif <- purrr::map_dfr(sort(unique(patients_cr$cohort)), function(co) {
    d <- patients_cr %>% filter(outcome == FOCAL_OUTCOME, cohort == co)
    if (sum(d$status == 1L) < 1L) return(NULL)
    d$status_f <- mk_status_factor(d$status)
    fit <- tryCatch(survfit(Surv(time, status_f) ~ split, data = d, conf.int = 0.95),
                    error = function(e) NULL)
    if (is.null(fit)) return(NULL)
    out <- extract_curves(fit); out$cohort <- co; out
  })
  if (nrow(focal_cif) > 0) {
    focal_cif <- focal_cif %>%
      mutate(split = factor(split, levels = c("train", "tuning", "test"))) %>%
      filter(n_risk >= MIN_AT_RISK)
    p_focal <- ggplot(focal_cif, aes(time, cif, color = split, fill = split)) +
      geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.12, color = NA) +
      geom_step() +
      geom_vline(xintercept = HORIZON_DAYS, linetype = 2, color = "grey40") +
      facet_wrap(~ cohort, scales = "free_x") +
      coord_cartesian(ylim = c(0, 1)) +
      labs(title = sprintf("Cumulative incidence by split, per cohort_grouped: %s", FOCAL_OUTCOME),
           subtitle = sprintf("Aalen-Johansen, death competing; truncated where at-risk < %d; bands 95%% CI", MIN_AT_RISK),
           x = "days from index", y = "cumulative incidence", color = NULL, fill = NULL) +
      theme_minimal(base_size = 8)
    ggsave(file.path(SUMMARY_DIR, sprintf("cif_by_cohort_%s.pdf", FOCAL_OUTCOME)),
           p_focal, width = 11, height = 8, limitsize = FALSE)
  }
} else {
  message("FOCAL_OUTCOME '", FOCAL_OUTCOME, "' not found; skipping per-cohort CIF.")
}

# ── Which cohort_grouped cohorts are viable, per outcome, at the chosen cutoff ─
# The proper BCE-based per-cohort claim (positives/negatives at the recommended
# horizon, test = index_year >= TEST_START_YEAR), to sit beside the overall
# manifest verdict. rec_horizon comes from the overall analysis; default 365.
cohort_viability_rec <- bce_sweep %>%
  filter(cohort_level == "cohort_grouped", test_start == TEST_START_YEAR) %>%
  left_join(rec_horizon, by = "outcome") %>%
  mutate(use_H = coalesce(rec_horizon, 365L)) %>%
  filter(horizon == use_H) %>%
  group_by(outcome) %>%
  summarise(
    rec_horizon      = first(use_H),
    n_cohorts_viable = sum(viable_bce, na.rm = TRUE),
    cohorts_viable   = paste(sort(cohort[viable_bce %in% TRUE]), collapse = ";"),
    .groups = "drop") %>%
  arrange(desc(n_cohorts_viable))
write_csv(cohort_viability_rec,
          file.path(SUMMARY_DIR, "cohort_viability_at_recommended_horizon.csv"))

message("\nPer-outcome count of cohort_grouped cohorts viable at the recommended horizon:")
print(cohort_viability_rec, n = 100)
message("\nWrote per-cohort cumulative incidence and cohort-viability summary to ", SUMMARY_DIR)


# ==============================================================================
# BCE viability ANATOMY: negatives, indeterminate, and the binding gate
# ==============================================================================
# The positives plots above render only one of the three gates. Viability is
#   n_pos >= MIN_POS  AND  n_neg >= MIN_POS  AND  indet_rate <= MAX_INDETERMINATE
# (the rule at the bce_tab / bce_sweep mutate). n_neg and indet_rate are already
# computed there; the plots below draw them, so a cell that clears positives but
# fails on the negative floor or (the usual culprit) on truncation-driven
# indeterminate is visible instead of hidden behind a positives count. These are
# the binary-classification companion to the cumulative-incidence panels, which
# already serve the time-to-event view.

bce_test <- bce_tab %>% filter(split == "test")

# ── A. Test-split NEGATIVES per outcome x horizon (mirror of the positives map) ─
# Negatives mostly grow with H (more patients become event-free-through-H), so
# this rarely binds; it bites for high-incidence endpoints at long H, where
# almost everyone has evented and the 0-class thins out.
p_bce_neg <- bce_test %>%
  ggplot(aes(factor(horizon), reorder(outcome, n_neg), fill = n_neg)) +
  geom_tile(color = "grey92") +
  geom_text(aes(label = n_neg, color = n_neg < MIN_POS), size = 2.2) +
  scale_color_manual(values = c(`TRUE` = "red", `FALSE` = "grey20"), guide = "none") +
  scale_fill_viridis_c(trans = "log1p", breaks = c(0, 50, 200, 1000)) +
  labs(title = "Test-split NEGATIVES at each BCE horizon",
       subtitle = sprintf("negatives = event-free through H (under the cause-specific convention a death-before-H without the outcome is a negative); red = fewer than %d", MIN_POS),
       x = "horizon H (days)", y = NULL, fill = "test negatives") +
  theme_minimal(base_size = 8)
ggsave(file.path(SUMMARY_DIR, "bce_negatives_by_horizon.pdf"), p_bce_neg,
       width = 8, height = max(6, 0.2 * n_out + 2), limitsize = FALSE)

# ── B. Test-split INDETERMINATE fraction per outcome x horizon ────────────────
# The constraint the positives plot actively hides: admin-censored event-free
# before H, dropped from a plain BCE eval. Moves OPPOSITE to positives as H grows
# at a late cutoff, so an endpoint can look positive-rich at H=1095 yet be
# unscoreable because most of the cohort never reaches the horizon.
p_bce_indet <- bce_test %>%
  ggplot(aes(factor(horizon), reorder(outcome, coalesce(indet_rate, 0)), fill = indet_rate)) +
  geom_tile(color = "grey92") +
  geom_text(aes(label = sprintf("%.0f%%", 100 * indet_rate),
                color = indet_rate > MAX_INDETERMINATE), size = 2.0) +
  scale_color_manual(values = c(`TRUE` = "red", `FALSE` = "grey20"), guide = "none") +
  scale_fill_viridis_c(option = "B", direction = -1, limits = c(0, 1),
                       labels = function(x) paste0(100 * x, "%")) +
  labs(title = "Test-split INDETERMINATE fraction at each BCE horizon",
       subtitle = sprintf("indeterminate = censored event-free before H, so unlabelled and dropped; red = above %.0f%%", 100 * MAX_INDETERMINATE),
       x = "horizon H (days)", y = NULL, fill = "indeterminate") +
  theme_minimal(base_size = 8)
ggsave(file.path(SUMMARY_DIR, "bce_indeterminate_by_horizon.pdf"), p_bce_indet,
       width = 8, height = max(6, 0.2 * n_out + 2), limitsize = FALSE)

# ── C. The binding-gate map: viable, or which gate fails, per outcome x horizon ─
# The single decision plot. For the chosen cutoff (test = index_year >=
# TEST_START_YEAR) it classifies every outcome x horizon as viable or, if not,
# which of the three gates is responsible. This is the "why" that the pooled
# min-viable-horizon feasibility map compresses away.
gate_levels <- c("viable", "fails: positives", "fails: negatives",
                 "fails: indeterminate", "fails: multiple")
bce_gate <- bce_test %>%
  mutate(
    f_pos  = n_pos < MIN_POS,
    f_neg  = n_neg < MIN_POS,
    f_ind  = indet_rate > MAX_INDETERMINATE,
    n_fail = as.integer(f_pos) + as.integer(f_neg) + as.integer(f_ind %in% TRUE),
    gate = case_when(
      n_fail == 0 ~ "viable",
      n_fail > 1  ~ "fails: multiple",
      f_pos       ~ "fails: positives",
      f_neg       ~ "fails: negatives",
      TRUE        ~ "fails: indeterminate"),
    gate = factor(gate, levels = gate_levels))
gate_order <- bce_gate %>% group_by(outcome) %>%
  summarise(n_ok = sum(gate == "viable"), .groups = "drop") %>%
  arrange(n_ok) %>% pull(outcome)
bce_gate <- bce_gate %>% mutate(outcome = factor(outcome, levels = gate_order))
gate_pal <- c("viable"               = "#2c7fb8",
              "fails: positives"     = "#fdae61",
              "fails: negatives"     = "#d7191c",
              "fails: indeterminate" = "#7b3294",
              "fails: multiple"      = "grey55")
p_gate <- ggplot(bce_gate, aes(factor(horizon), outcome, fill = gate)) +
  geom_tile(color = "grey92") +
  scale_fill_manual(values = gate_pal, drop = FALSE) +
  labs(title = sprintf("BCE viability and the binding gate, test split (test >= %d)", TEST_START_YEAR),
       subtitle = sprintf("viable = %d+ positives AND %d+ negatives AND indeterminate <= %.0f%%; colour = which gate fails",
                          MIN_POS, MIN_POS, 100 * MAX_INDETERMINATE),
       x = "horizon H (days)", y = NULL, fill = NULL) +
  theme_minimal(base_size = 7)
ggsave(file.path(SUMMARY_DIR, "bce_binding_constraint.pdf"), p_gate,
       width = 8, height = max(6, 0.2 * n_out + 2), limitsize = FALSE)

# ── D. Per-cohort label composition for the focal outcome ─────────────────────
# The across-cohorts view, the binary-classification companion to the per-cohort
# cumulative-incidence panel. Positives, negatives and indeterminate are dodged
# (not stacked) so the MIN_POS floor reads directly on positives AND negatives,
# and the indeterminate bar shows how much of each cohort is unscoreable at H.
if (FOCAL_OUTCOME %in% bce_sweep$outcome &&
    "cohort_grouped" %in% bce_sweep$cohort_level) {
  focal_comp <- bce_sweep %>%
    filter(cohort_level == "cohort_grouped", outcome == FOCAL_OUTCOME,
           test_start == TEST_START_YEAR) %>%
    select(cohort, horizon, n_pos, n_neg, n_indet) %>%
    pivot_longer(c(n_pos, n_neg, n_indet), names_to = "class", values_to = "n") %>%
    mutate(class = factor(dplyr::recode(class,
                            n_pos = "positive", n_neg = "negative",
                            n_indet = "indeterminate"),
                          levels = c("positive", "negative", "indeterminate")))
  if (nrow(focal_comp) > 0) {
    p_focal_comp <- ggplot(focal_comp, aes(factor(horizon), n, fill = class)) +
      geom_col(position = position_dodge(width = 0.8), width = 0.75) +
      geom_hline(yintercept = MIN_POS, linetype = 2, color = "red") +
      facet_wrap(~ cohort, scales = "free_y") +
      scale_fill_manual(values = c("positive" = "#1a9850", "negative" = "#4575b4",
                                   "indeterminate" = "#bdbdbd")) +
      labs(title = sprintf("BCE label composition by cohort: %s (test >= %d)",
                           FOCAL_OUTCOME, TEST_START_YEAR),
           subtitle = sprintf("positives and negatives each need >= %d (red line); indeterminate are dropped; point FOCAL_OUTCOME elsewhere to inspect another endpoint", MIN_POS),
           x = "horizon H (days)", y = "patients", fill = NULL) +
      theme_minimal(base_size = 8)
    ggsave(file.path(SUMMARY_DIR, sprintf("bce_composition_cohort_%s.pdf", FOCAL_OUTCOME)),
           p_focal_comp, width = 11, height = 8, limitsize = FALSE)
  }
} else {
  message("FOCAL_OUTCOME '", FOCAL_OUTCOME,
          "' or cohort_grouped level absent from bce_sweep; skipping composition plot.")
}

# ── E. Indeterminate fraction across test-start (the truncation tension) ──────
# Parallel to bce_positives_horizon_x_teststart.pdf, on the gate the positives
# sweep cannot show. Reading the two side by side is the honest feasibility map:
# positives say how much signal a late cutoff keeps, this says how much of the
# cohort that same cutoff renders unlabelled.
p_indet_sweep <- bce_sweep %>% filter(cohort_level == "overall") %>%
  ggplot(aes(factor(test_start), reorder(outcome, coalesce(indet_rate, 0)), fill = indet_rate)) +
  geom_tile(color = "grey92") +
  geom_text(aes(label = sprintf("%.0f", 100 * indet_rate),
                color = indet_rate > MAX_INDETERMINATE), size = 1.7) +
  scale_color_manual(values = c(`TRUE` = "red", `FALSE` = "grey20"), guide = "none") +
  scale_fill_viridis_c(option = "B", direction = -1, limits = c(0, 1),
                       labels = function(x) paste0(100 * x, "%")) +
  facet_wrap(~ horizon, nrow = 1,
             labeller = labeller(horizon = function(x) paste0("H=", x, "d"))) +
  labs(title = "BCE indeterminate fraction: horizon x test-start (all cohorts)",
       subtitle = sprintf("later test-start + longer H -> more truncation -> more dropped; red = above %.0f%% (cell value is the %%)", 100 * MAX_INDETERMINATE),
       x = "candidate test-start year", y = NULL, fill = "indeterminate") +
  theme_minimal(base_size = 7) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
ggsave(file.path(SUMMARY_DIR, "bce_indeterminate_horizon_x_teststart.pdf"), p_indet_sweep,
       width = max(10, 2.4 * length(BCE_HORIZONS) + 2),
       height = max(6, 0.2 * n_out + 2), limitsize = FALSE)

message("\nWrote BCE anatomy plots (negatives, indeterminate, binding gate, ",
        "per-cohort composition, indeterminate sweep) to ", SUMMARY_DIR)

# ==============================================================================
# Ascertainment-drift diagnostic: cumulative incidence by index-year band
# ==============================================================================
# Re-expresses incidence by finer index-year bands instead of the 3-way split.
# Real clinical change trends smoothly across bands; a SOURCE-COVERAGE change
# shows as a monotone fan that plateaus once the source stabilises, with older
# bands depressed because their early follow-up sits in the under-capture era.
# Procedure / medication / microbiology outcomes (antibiotics, transfusions,
# emergency_stay, sepsis, serious_infection_composite, hospitalisation) are the
# ones to scrutinise; labs and overall_survival should overlap across bands.
# The band where the fan plateaus is the coverage-gate cutoff for that outcome.

INDEX_BANDS <- c(-Inf, 2015, 2018, 2020, 2021, Inf)
INDEX_BAND_LABS <- c("<=2015", "2016-2018", "2019-2020", "2021", "2022+")
band_of <- function(yr) {
  yr <- suppressWarnings(as.numeric(yr))
  interior <- INDEX_BANDS[is.finite(INDEX_BANDS)]      # e.g. 2015, 2018, 2020, 2021
  idx <- findInterval(yr, interior + 1L)               # 0..length(interior)
  factor(INDEX_BAND_LABS[idx + 1L], levels = INDEX_BAND_LABS)
}

patients_band <- patients_cr %>% mutate(band = band_of(index_year))
stopifnot("index-year banding produced all-NA (check index_year / INDEX_BANDS)" =
            any(!is.na(patients_band$band)))

band_cif <- purrr::map_dfr(sort(unique(patients_band$outcome)), function(oc) {
  purrr::map_dfr(INDEX_BAND_LABS, function(bd) {
    d <- patients_band %>% filter(outcome == oc, band == bd)
    if (nrow(d) < 20L || sum(d$status == 1L) < 1L) return(NULL)
    d$status_f <- mk_status_factor(d$status)
    fit <- tryCatch(survfit(Surv(time, status_f) ~ 1, data = d, conf.int = 0.95),
                    error = function(e) NULL)
    if (is.null(fit)) return(NULL)
    out <- extract_curves(fit)        # single curve; we tag band/outcome ourselves
    out$band <- bd
    out$outcome <- oc
    out
  })
}) %>%
  mutate(band = factor(band, levels = INDEX_BAND_LABS)) %>%
  filter(n_risk >= MIN_AT_RISK)

p_band <- ggplot(band_cif, aes(time, cif, color = band, fill = band)) +
  geom_step() +
  geom_vline(xintercept = HORIZON_DAYS, linetype = 2, color = "grey40") +
  facet_wrap(~ outcome, scales = "free_x") +
  coord_cartesian(ylim = c(0, 1), xlim = c(0, 365 * 3)) +
  scale_color_viridis_d(option = "C", end = 0.9) +
  scale_fill_viridis_d(option = "C", end = 0.9) +
  labs(title = "Cumulative incidence by INDEX-YEAR band (ascertainment-drift check)",
       subtitle = "Monotone fan that plateaus = calendar-time source-coverage drift, not clinical change. Bands ribbon omitted for legibility.",
       x = "days from index", y = "cumulative incidence", color = "index band", fill = "index band") +
  theme_minimal(base_size = 7)
ggsave(file.path(SUMMARY_DIR, "cif_by_index_year_band.pdf"), p_band,
       width = 12, height = max(6, 0.6 * sqrt(n_out) + 4), limitsize = FALSE)

# Numeric companion: incidence at the 365-day reference per outcome x band, so
# the fan can be read as a table (last CIF value at or before the horizon).
band_at_h <- band_cif %>%
  filter(time <= HORIZON_DAYS) %>%
  group_by(outcome, band) %>%
  slice_max(time, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(outcome, band, cif_at_horizon = cif) %>%
  pivot_wider(names_from = band, values_from = cif_at_horizon)
write_csv(band_at_h, file.path(SUMMARY_DIR, "cif_by_index_year_band_at_horizon.csv"))

message("\nWrote ascertainment-drift diagnostic (CIF by index-year band) to ", SUMMARY_DIR)

