# ==============================================================================
# identification_time_to_next_treatment.R  (v4)
#
# Time to next line of treatment (TTNT) -- second-line identification.
#
# DESIGN PHILOSOPHY (v4)
# ----------------------
# The first-line identification script works because it has a biological anchor
# (diagnosis date), builds treatment episodes from the same registry sources,
# and applies quality filters in layers. For second-line, we have the same
# infrastructure: firstline_end is our anchor, the same treatment records are
# available, and the same filter logic applies.
#
# v4 reuses the 1L episode-building machinery and filter hierarchy, but
# calibrates it for the specific challenges of the post-1L window:
#
#   (1) Anti-CD20/rituximab alone is systematically excluded.
#       A single anti-CD20 record constitutes maintenance, not 2L.
#       This was the dominant source of false positives in v1/v2.
#
#   (2) Corroboration requirement is stricter than 1L.
#       For 2L, heme ICD requires BOTH multi-date AND multi-code
#       (n_dates >= 2 AND n_codes >= 2), not OR.
#       The post-1L window is noisier and single-record evidence is
#       insufficient to establish a new treatment line.
#
#   (3) Two outcome columns are produced:
#       retreatment_event -- low bar: any therapy-like non-maintenance episode.
#                           Answers: "did this patient receive treatment again?"
#                           Use for sensitivity analysis.
#       ttnt_event        -- high bar: retreatment with adequate evidence quality.
#                           Answers: "did this patient start a new line of therapy?"
#                           Use as primary OPERA outcome.
#       The gap between these two rates tells you how much noise is in the data.
#
#   (4) RKKP 2L dates integrated as primary evidence where available.
#       When RKKP records a relapse/2L date it overrides the treatment-derived
#       date. Uncomment section 9 to activate per RKKP module.
#
#   (5) Confidence scoring distinguishes:
#       high:     anchor code (non-maintenance) + heme ICD, or RKKP
#       moderate: anchor code alone, or heme ICD + multi-date + multi-code
#       low:      minimal evidence (Tier 4 fallback)
#
#   (6) Per-outcome eligibility (the OPERA mask). Every patient with a
#       firstline_date is assigned one of three states for the PRIMARY outcome:
#         eligible_event    -- confident 2L; trustworthy positive.
#         eligible_censored -- no event AND no weak-evidence episode AND has
#                              follow-up; trustworthy negative.
#         masked_*          -- ambiguous; NOT written to the outcome parquet.
#       A weak-evidence-only patient (low_single retreatment) is MASKED, never
#       silently demoted to a censored negative -- demoting it would manufacture
#       a false negative. Masking = absence from the parquet = the per-outcome
#       eligibility mask OPERA already consumes.
#
#   (7) Export adapter (section 12) writes the fixed OPERA outcome schema
#       (subject_id, split, outcome_date, index_date, censor_date) for eligible
#       patients only, plus a __audit parquet carrying every reason code and all
#       confidence/timing metadata. The downstream Python loader binarises this
#       into "2L within window" exactly as for every other outcome.
#
# COHORT RELIABILITY / SOURCING
#   High:     DLBCL_like, HL         (clear salvage episodes, KM in lit. band)
#   Moderate: MCL, T_NHL, Indolent_B_NHL  (episode-derived; gate on KM band)
#   Low:      BL_LBL, HCL            (rapid cycling/sparse coding; episode-derived
#                                     NEGATIVES masked, RKKP-confirmed events kept)
#   Curated:  CLL_SLL, MM           (continuous therapy: episode algorithm NEVER
#                                     run; a clinician-curated 2L dataset supplies
#                                     the label, source membership = eligibility,
#                                     2L date = event; RKKP optional fallback)
#
# PREREQUISITES
#   identification_first_line_treatment_v2.R must be sourced first.
#   Requires: tx_flagged, firstline_tbl_mapped, patient
#             ANCHOR_CODES, PROPHY_SUSPECT_CODES, TRUST_RKKP
#
# Author: Mikkel Werling  2026-03-28
# ==============================================================================

suppressPackageStartupMessages({
  library(arrow); library(data.table); library(dplyr); library(lubridate)
  library(purrr); library(stringr); library(tibble); library(tidyr)
  library(ggplot2)
})


# ==============================================================================
# 1.  Disease configuration
# ==============================================================================

TTNT_CONFIG <- tribble(
  ~cohort_grouped,  ~expected_1L_duration, ~min_gap_days, ~consolidation_window, ~maintenance_window, ~overlap_threshold, ~reliability, ~note,
  #
  # expected_1L_duration: imputes firstline_end for single-day episodes only.
  #
  # min_gap_days: minimum days after firstline_end before any episode qualifies.
  #   Conservative values: absorbs last-drug-to-restaging plus restaging window.
  #   Too tight = consolidation and supportive care leak through.
  #   Too wide = real early relapses missed (primary refractory cases).
  #
  # consolidation_window: episodes within this window with high code overlap
  #   with 1L are classified as consolidation (first-line completion), not 2L.
  #
  # maintenance_window: episodes within this window consisting of only
  #   maintenance-compatible codes are classified as maintenance, not 2L.
  #
  # overlap_threshold: fraction of 1L codes that must reappear in a new episode
  #   for it to be classified as consolidation.
  #
  "DLBCL_like",     126,  60,  90,   NA,  0.45, "high",     "R-CHOP 6x21d=18wk. Restaging ~6wk post-treatment.",
  "HL",             120,  60,  90,   NA,  0.45, "high",     "ABVD 4-6x. PET restaging ~6wk post-treatment.",
  "BL_LBL",          28,  45,  90,   NA,  0.50, "low",      "Rapid drug cycling. Use OS. TTNT exploratory only.",
  "MCL",            140,  60, 180,  548,  0.40, "moderate", "R-CHOP/R-HAD induction. ASCT guard. Rituximab maint ~18m.",
  "T_NHL",          126,  60,  90,   NA,  0.40, "moderate", "CHOP-like 6x or GemOx-based.",
  "HCL",              7,  90,  NA,   NA,  0.30, "low",      "Sparse coding. Use OS. TTNT exploratory only.",
  "Indolent_B_NHL", 168,  90, 120,  730,  0.35, "moderate", "R-chemo x6. 2yr rituximab maintenance excluded."
)

TTNT_EXCLUDED_COHORTS  <- c("CLL_SLL", "MM")
TTNT_UNRELIABLE_COHORTS <- c("BL_LBL", "HCL")

# Anti-CD20 / maintenance codes -- episodes consisting solely of these are
# maintenance, not second-line treatment. Expanded with text aliases from v2.
# A rituximab-CONTAINING combination (R-ICE, R-DHAP etc.) still qualifies
# because the non-rituximab components will be present as additional codes.
CD20_ONLY_CODES <- c(
  "rituximab", "L01FA01", "L01XC02", "BOHJ11",
  "cd20 antibody",                                       # dominant text alias
  "monoclonal antibodies and antibody drug conjugates",  # generic coding bucket
  "obinutuzumab", "L01FA03", "L01XC15", "BOHJ19H5",     # obinutuzumab
  "ofatumumab",   "L01FA02"                              # ofatumumab
)

# Broader maintenance codes: include lenalidomide maintenance (FL/MCL)
MAINTENANCE_ONLY_CODES <- c(
  CD20_ONLY_CODES,
  "lenalidomide", "L04AX04", "BWHB82"
)

# ASCT conditioning: episodes of these within consolidation_window are
# first-line ASCT consolidation, not second-line.
ASCT_CODES <- c(
  "asct", "asct (conditioning)", "high-dose melphalan",
  "L01AA03", "BWHA154",    # melphalan iv
  "BWHA451",               # thiotepa
  "BWHA452",               # carmustine (BEAM)
  "BWHA162", "BWHA301"     # high-dose cytarabine (BEAM conditioning)
)


# ==============================================================================
# 2.  Prepare firstline_end
#     Single-day episodes (firstline_end == firstline_date) imputed.
# ==============================================================================

prepare_firstline_end <- function(firstline_tbl, config) {
  duration_map <- config %>% select(cohort_grouped, expected_1L_duration)
  
  firstline_tbl %>%
    left_join(duration_map, by = "cohort_grouped") %>%
    mutate(
      firstline_end  = as.Date(firstline_end),
      firstline_date = as.Date(firstline_date),
      
      firstline_single_day = !is.na(firstline_end) &
        (firstline_end == firstline_date),
      
      firstline_end_source = case_when(
        is.na(firstline_end) ~ "missing",
        firstline_single_day ~ "imputed_single_day",
        TRUE                 ~ "observed"
      ),
      
      firstline_end = case_when(
        firstline_end_source == "observed" ~ firstline_end,
        !is.na(firstline_date) & !is.na(expected_1L_duration) ~
          firstline_date + as.integer(expected_1L_duration),
        TRUE ~ as.Date(NA)
      )
    ) %>%
    select(-expected_1L_duration)
}


# ==============================================================================
# 3.  Helpers
# ==============================================================================

norm_codes <- function(x) tolower(trimws(x))

codes_to_vec <- function(codes_str) {
  if (is.na(codes_str) || !nzchar(codes_str)) return(character(0))
  str_split(codes_str, " \\| ")[[1]]
}

overlap_fraction <- function(new_codes, codes_1L) {
  if (length(codes_1L) == 0 || all(codes_1L == "")) return(0)
  mean(norm_codes(codes_1L) %in% norm_codes(new_codes))
}

# Check each code individually to avoid recycling-length warning.
codes_match_any <- function(codes_vec, reference_codes) {
  if (length(codes_vec) == 0) return(FALSE)
  ref_lc  <- norm_codes(reference_codes)
  test_lc <- norm_codes(codes_vec)
  any(test_lc %in% ref_lc) ||
    any(sapply(ref_lc, function(p) any(str_detect(test_lc, fixed(p)))))
}

# Is this episode made up entirely of maintenance-compatible codes?
# Each code checked individually to avoid length-mismatch warning.
is_maintenance_only <- function(codes_vec) {
  if (length(codes_vec) == 0) return(FALSE)
  maint_lc <- norm_codes(MAINTENANCE_ONLY_CODES)
  codes_lc <- norm_codes(codes_vec)
  all(sapply(codes_lc, function(co) {
    co %in% maint_lc || any(str_detect(co, fixed(maint_lc)))
  }))
}

# Is this episode anti-CD20 alone (most specific maintenance check)?
is_cd20_only <- function(codes_vec) {
  if (length(codes_vec) == 0) return(FALSE)
  cd20_lc  <- norm_codes(CD20_ONLY_CODES)
  codes_lc <- norm_codes(codes_vec)
  all(sapply(codes_lc, function(co) {
    co %in% cd20_lc || any(str_detect(co, fixed(cd20_lc)))
  }))
}

has_asct_codes <- function(codes_vec, text_vec = character(0)) {
  all_lc  <- norm_codes(c(codes_vec, text_vec))
  if (length(all_lc) == 0) return(FALSE)
  asct_lc <- norm_codes(ASCT_CODES)
  any(all_lc %in% asct_lc) ||
    any(sapply(asct_lc, function(p) any(str_detect(all_lc, fixed(p)))))
}

# Confidence level for a qualifying 2L episode.
# Key distinction: anchor code + heme ICD = high; anchor alone = moderate;
# heme ICD with corroboration = moderate; minimal evidence = low.
# end_was_imputed downgrades moderate -> low_imputed.
episode_confidence_2L <- function(has_anchor, has_heme_icd, has_rkkp,
                                  has_prophy_suspect, n_dates, n_codes,
                                  end_was_imputed) {
  base <- case_when(
    has_rkkp                                       ~ "high_rkkp",
    has_anchor & has_heme_icd                      ~ "high_anchor_icd",
    has_anchor & !has_prophy_suspect               ~ "moderate_anchor",
    # Requires both multi-date AND multi-code: stricter than 1L (which used OR)
    has_heme_icd & n_dates >= 2 & n_codes >= 2    ~ "moderate_icd_corroborated",
    n_dates >= 3 & n_codes >= 3                    ~ "low_multievent",
    TRUE                                           ~ "low_single"
  )
  if_else(
    end_was_imputed & base %in% c("high_anchor_icd", "moderate_anchor",
                                  "moderate_icd_corroborated"),
    str_c(base, "_imputed"),
    base
  )
}


# ==============================================================================
# 4.  Episode builder (identical to 1L machinery)
#     Operates on treatment records after firstline_end + min_gap_days.
# ==============================================================================

build_post_firstline_episodes <- function(tx_ok, firstline_index,
                                          min_gap_days, episode_gap_days = 21) {
  tx_post <- tx_ok %>%
    inner_join(
      firstline_index %>%
        select(patient_id, firstline_date, firstline_end,
               firstline_end_source, firstline_codes, firstline_codes_text) %>%
        mutate(across(c(firstline_date, firstline_end), as.Date)),
      by = "patient_id"
    ) %>%
    filter(!is.na(firstline_end),
           tx_date > firstline_end + min_gap_days) %>%
    arrange(patient_id, tx_date, code)
  
  if (nrow(tx_post) == 0) return(tibble())
  
  t_ep <- tx_post %>%
    group_by(patient_id) %>%
    mutate(
      gap_days    = as.integer(tx_date - lag(tx_date)),
      new_episode = if_else(is.na(gap_days) | gap_days > episode_gap_days, 1L, 0L),
      episode_id  = cumsum(new_episode)
    ) %>%
    ungroup()
  
  t_ep %>%
    group_by(patient_id, episode_id) %>%
    summarise(
      firstline_date          = first(firstline_date),
      firstline_end           = first(firstline_end),
      firstline_end_source    = first(firstline_end_source),
      firstline_codes_1L      = first(firstline_codes),
      firstline_codes_text_1L = first(firstline_codes_text),
      indication_text         = paste(sort(unique(dx_text[!is.na(dx_text)])),    collapse = " | "),
      indication_codes        = paste(sort(unique(dx_code[!is.na(dx_code)])),    collapse = " | "),
      episode_start           = min(tx_date),
      episode_end             = max(tx_date),
      n_tx_rows               = n(),
      n_tx_dates              = n_distinct(tx_date),
      n_codes                 = n_distinct(code),
      has_anchor              = any(code %in% ANCHOR_CODES),
      has_prophy_suspect      = any(code %in% PROPHY_SUSPECT_CODES),
      has_rkkp_source         = any(is_rkkp),
      has_heme_indication     = any(ind_heme_icd),
      pct_heme_indication     = mean(ind_heme_icd),
      codes_sorted            = paste(sort(unique(code)),                          collapse = " | "),
      codes_text_sorted       = paste(sort(unique(code_text[!is.na(code_text)])), collapse = " | "),
      dates_sorted            = paste(sort(unique(as.character(tx_date))),          collapse = ", "),
      .groups = "drop"
    ) %>%
    mutate(
      episode_duration_days = as.integer(episode_end - episode_start),
      days_after_1L_end     = as.integer(episode_start - firstline_end)
    )
}


# ==============================================================================
# 5.  Episode classifier
#
# Two-pass classification:
#
# Pass 1 -- Exclusion guards (same as 1L, applied first):
#   asct_consolidation: ASCT/conditioning codes within consolidation window
#   consolidation:      high code overlap with 1L within consolidation window
#   maintenance:        maintenance-only codes within maintenance window
#
# Pass 2 -- Classify remaining episodes by evidence quality:
#   retreatment:   any therapy-like non-maintenance episode (low bar).
#                  "Did this patient receive treatment again?"
#   secondline:    retreatment with adequate confidence (high bar).
#                  "Did this patient start a new line of therapy?"
#   weak:          some evidence but below the threshold for either outcome
#
# The split between retreatment and secondline is:
#   retreatment requires: not cd20_only, and (anchor | heme_ICD+multi | rkkp)
#   secondline  requires: retreatment AND confidence >= moderate
#                         (i.e., heme ICD with BOTH multi-date AND multi-code,
#                          or anchor code, or RKKP)
# ==============================================================================

classify_episodes <- function(eps, consolidation_window, maintenance_window,
                              overlap_threshold) {
  eps %>%
    rowwise() %>%
    mutate(
      codes_vec_new = list(codes_to_vec(codes_sorted)),
      text_vec_new  = list(codes_to_vec(codes_text_sorted)),
      codes_vec_1L  = list(codes_to_vec(coalesce(firstline_codes_1L, ""))),
      
      # ── Pass 1: exclusion guards ───────────────────────────────────────────
      is_asct = (
        !is.na(consolidation_window) &
          days_after_1L_end <= consolidation_window &
          has_asct_codes(codes_vec_new, text_vec_new)
      ),
      
      is_consolidation = (
        !is.na(consolidation_window) &
          days_after_1L_end <= consolidation_window &
          overlap_fraction(codes_vec_new, codes_vec_1L) >= overlap_threshold
      ),
      
      is_maintenance = (
        !is.na(maintenance_window) &
          days_after_1L_end <= maintenance_window &
          is_maintenance_only(c(codes_vec_new, text_vec_new))
      ),
      is_cd20_alone = is_cd20_only(c(codes_vec_new, text_vec_new)),
      
      # Evidence bar identical to the frozen 1L therapy_like gate.
      #   - anchor drug: counts
      #   - RKKP: counts only with real content (matches 1L, no longer source-alone)
      #   - CD20 mono: counts as a real course when repeated (>=2 dates), as in 1L
      #   - prophy-suspect alone: 3+ dates AND 2+ codes
      #   - heme ICD is NOT a standalone pass (removed, same as 1L); such episodes
      #     must clear the generic fallback on their own content
      # Maintenance/continuation is handled in episode_class (is_maintenance),
      # not by a hard CD20 exclusion here.
      retreatment_like = case_when(
        has_anchor                                                      ~ TRUE,
        TRUST_RKKP & has_rkkp_source & (n_codes >= 2 | n_tx_dates >= 2) ~ TRUE,
        is_cd20_alone & n_tx_dates >= 2                                 ~ TRUE,
        has_prophy_suspect                                             ~ (n_tx_dates >= 3 & n_codes >= 2),
        TRUE                                                           ~ (n_tx_dates >= 2 | n_codes >= 2)
      ),
      
      # ── Confidence for qualifying episodes ────────────────────────────────
      confidence = episode_confidence_2L(
        has_anchor, has_heme_indication, has_rkkp_source,
        has_prophy_suspect, n_tx_dates, n_codes,
        end_was_imputed = (firstline_end_source == "imputed_single_day")
      ),
      
      # Secondline = retreatment with at minimum moderate confidence
      secondline_quality = retreatment_like &
        confidence %in% c(
          "high_rkkp",
          "high_anchor_icd",   "high_anchor_icd_imputed",
          "moderate_anchor",   "moderate_anchor_imputed",
          "moderate_icd_corroborated", "moderate_icd_corroborated_imputed"
        ),
      
      # ── Final episode label ────────────────────────────────────────────────
      episode_class = case_when(
        is_asct           ~ "asct_consolidation",
        is_consolidation  ~ "consolidation",
        is_maintenance    ~ "maintenance",
        is_cd20_alone & n_tx_dates < 2     ~ "cd20_only",       # explicit label for audit
        secondline_quality ~ "secondline",
        retreatment_like  ~ "retreatment_weak", # counts for retreatment but not 2L
        TRUE              ~ "no_signal"
      )
    ) %>%
    ungroup() %>%
    select(-codes_vec_new, -text_vec_new, -codes_vec_1L)
}


# ==============================================================================
# 6.  Main: make_secondline()
# ==============================================================================

make_secondline <- function(tx, firstline_tbl, config = TTNT_CONFIG,
                            episode_gap_days = 21) {
  stopifnot(all(c("ind_ok","patient_id","tx_date","code","is_rkkp",
                  "ind_heme_icd","dx_text","dx_code","code_text") %in% names(tx)))
  stopifnot(all(c("patientid","firstline_date","firstline_end",
                  "firstline_end_source","cohort_grouped",
                  "firstline_codes","firstline_codes_text_norm") %in% names(firstline_tbl)))
  
  fl <- firstline_tbl %>%
    filter(cohort_grouped %in% config$cohort_grouped,
           !is.na(firstline_date), !is.na(firstline_end)) %>%
    transmute(
      patient_id = patientid, cohort_grouped, cohort_fine,
      firstline_date       = as.Date(firstline_date),
      firstline_end        = as.Date(firstline_end),
      firstline_end_source,
      firstline_codes,
      firstline_codes_text = firstline_codes_text_norm
    )
  
  tx_ok <- tx %>%
    filter(ind_ok, !is.na(patient_id), !is.na(tx_date), !is.na(code)) %>%
    mutate(tx_date = as.Date(tx_date), code = as.character(code))
  
  all_episodes <- list()
  
  for (i in seq_len(nrow(config))) {
    grp <- config$cohort_grouped[i]
    cfg <- config[i, ]
    fl_grp <- fl %>% filter(cohort_grouped == grp)
    if (nrow(fl_grp) == 0) next
    tx_grp <- tx_ok %>% filter(patient_id %in% fl_grp$patient_id)
    
    eps <- build_post_firstline_episodes(
      tx_ok            = tx_grp,
      firstline_index  = fl_grp,
      min_gap_days     = cfg$min_gap_days,
      episode_gap_days = episode_gap_days
    )
    if (nrow(eps) == 0) next
    
    all_episodes[[grp]] <- classify_episodes(
      eps                  = eps,
      consolidation_window = cfg$consolidation_window,
      maintenance_window   = cfg$maintenance_window,
      overlap_threshold    = cfg$overlap_threshold
    ) %>%
      mutate(cohort_grouped = grp)
  }
  
  bind_rows(all_episodes)
}


# ==============================================================================
# 7.  Build TTNT outcome table
#     Two outcome columns:
#       retreatment_event -- any non-maintenance post-gap treatment (low bar)
#       ttnt_event        -- confident new line of therapy (high bar, primary)
# ==============================================================================

build_ttnt_outcome <- function(all_episodes, firstline_tbl, patient_tbl,
                               followup_end_date) {
  fl_index <- firstline_tbl %>%
    filter(cohort_grouped %in% TTNT_CONFIG$cohort_grouped, !is.na(firstline_date)) %>%
    transmute(
      patient_id           = patientid,
      cohort_grouped, cohort_fine,
      firstline_date       = as.Date(firstline_date),
      firstline_end        = as.Date(firstline_end),
      firstline_end_source = coalesce(firstline_end_source, "observed"),
      firstline_single_day = coalesce(firstline_single_day, FALSE)
    )
  
  death_info <- patient_tbl %>%
    select(patient_id = patientid, death_date = date_death) %>%
    mutate(death_date = as.Date(death_date))
  
  follow_end <- as.Date(followup_end_date)
  
  # ── Earliest secondline episode (high-bar outcome) ─────────────────────────
  secondline_first <- all_episodes %>%
    filter(episode_class == "secondline") %>%
    arrange(patient_id, episode_start) %>%
    group_by(patient_id) %>%
    slice(1) %>%
    ungroup() %>%
    transmute(
      patient_id,
      secondline_date         = episode_start,
      secondline_end          = episode_end,
      secondline_codes        = codes_sorted,
      secondline_codes_text   = codes_text_sorted,
      secondline_indication   = indication_text,
      secondline_n_dates      = n_tx_dates,
      secondline_n_codes      = n_codes,
      secondline_has_anchor   = has_anchor,
      secondline_has_rkkp     = has_rkkp_source,
      secondline_has_heme_icd = has_heme_indication,
      secondline_confidence   = confidence,
      secondline_single_day   = (episode_end == episode_start),
      secondline_end_imputed  = (firstline_end_source == "imputed_single_day"),
      days_1L_end_to_2L       = days_after_1L_end
    )
  
  # ── Earliest retreatment episode (low-bar outcome) ─────────────────────────
  retreatment_first <- all_episodes %>%
    filter(episode_class %in% c("secondline", "retreatment_weak")) %>%
    arrange(patient_id, episode_start) %>%
    group_by(patient_id) %>%
    slice(1) %>%
    ungroup() %>%
    transmute(
      patient_id,
      retreatment_date      = episode_start,
      retreatment_class     = episode_class,
      retreatment_confidence = confidence
    )
  
  # ── Episode audit ──────────────────────────────────────────────────────────
  episode_audit <- all_episodes %>%
    filter(!is.na(episode_class)) %>%
    group_by(patient_id) %>%
    summarise(
      n_episodes_total      = n(),
      n_secondline_eps      = sum(episode_class == "secondline"),
      n_retreatment_weak_eps = sum(episode_class == "retreatment_weak"),
      n_consolidation_eps   = sum(episode_class == "consolidation"),
      n_asct_eps            = sum(episode_class == "asct_consolidation"),
      n_maintenance_eps     = sum(episode_class == "maintenance"),
      n_cd20_only_eps       = sum(episode_class == "cd20_only"),
      n_no_signal_eps       = sum(episode_class == "no_signal"),
      .groups = "drop"
    )
  
  fl_index %>%
    left_join(secondline_first,  by = "patient_id") %>%
    left_join(retreatment_first, by = "patient_id") %>%
    left_join(episode_audit,     by = "patient_id") %>%
    left_join(death_info,        by = "patient_id") %>%
    mutate(
      censor_date = pmin(coalesce(death_date, follow_end), follow_end),
      
      # ── Primary outcome: confident 2L ──────────────────────────────────────
      ttnt_event    = as.integer(!is.na(secondline_date) & secondline_date <= censor_date),
      ttnt_end_date = if_else(ttnt_event == 1L, secondline_date, censor_date),
      ttnt_days     = as.integer(ttnt_end_date - firstline_date),
      tfi_days      = as.integer(ttnt_end_date - firstline_end),
      
      # ── Secondary outcome: any retreatment ────────────────────────────────
      retreatment_event    = as.integer(!is.na(retreatment_date) & retreatment_date <= censor_date),
      retreatment_end_date = if_else(retreatment_event == 1L, retreatment_date, censor_date),
      retreatment_days     = as.integer(retreatment_end_date - firstline_date),
      
      # Gap between outcomes: high rate = much weak evidence in the data
      only_retreatment_weak = (retreatment_event == 1L & ttnt_event == 0L),
      
      followup_days       = as.integer(censor_date - firstline_date),
      short_followup_flag = followup_days < 365,
      
      # ── Reliability ────────────────────────────────────────────────────────
      cohort_reliability = case_when(
        cohort_grouped %in% TTNT_UNRELIABLE_COHORTS ~ "low",
        cohort_grouped %in% c("DLBCL_like","HL")    ~ "high",
        TRUE                                         ~ "moderate"
      ),
      
      ttnt_reliability = case_when(
        ttnt_event == 0                                     ~ "censored",
        cohort_reliability == "low"                         ~ "low_cohort",
        secondline_confidence %in% c("high_rkkp",
                                     "high_anchor_icd")     ~ "high",
        str_starts(secondline_confidence, "moderate")       ~ "moderate",
        TRUE                                                ~ "low"
      ),

      # ── Per-outcome eligibility (the OPERA mask) ───────────────────────────
      # Three honest states. We never convert a weak-evidence (low_single)
      # patient into a censored negative -- that would manufacture a false
      # negative -- so they are masked instead. ttnt_event == 1 already implies
      # confidence >= moderate (a low_single episode is classified
      # retreatment_weak and never sets ttnt_event), so the masking below only
      # ever removes ambiguous NEGATIVES. BL_LBL / HCL: episode-derived
      # negatives are untrustworthy, so the whole cohort is masked here; RKKP
      # integration (section 9) re-promotes any RKKP-confirmed events.
      ttnt_eligibility_reason = case_when(
        is.na(firstline_date)                       ~ "masked_no_firstline",
        censor_date <= firstline_date               ~ "masked_no_followup",
        cohort_grouped %in% TTNT_UNRELIABLE_COHORTS ~ "masked_low_reliability_cohort",
        ttnt_event == 1L                            ~ "eligible_event",
        only_retreatment_weak                       ~ "masked_weak_evidence_only",
        TRUE                                        ~ "eligible_censored"
      ),
      ttnt_eligible = startsWith(ttnt_eligibility_reason, "eligible"),

      across(starts_with("n_") & ends_with("_eps"), ~replace_na(.x, 0L))
    )
}


# ==============================================================================
# 8.  Optional: integrate RKKP 2L dates (primary evidence where available)
#     Uncomment per RKKP module and run after build_ttnt_outcome().
# ==============================================================================

integrate_rkkp_2L <- function(ttnt_tbl, rkkp_2L_tbl, followup_end_date) {
  # rkkp_2L_tbl must have: patientid, rkkp_2L_date (Date)
  # Applies only to rows already in ttnt_tbl (the episode cohorts). For the
  # continuous-therapy cohorts use seed_continuous_cohorts() instead.
  follow_end <- as.Date(followup_end_date)
  
  ttnt_tbl %>%
    left_join(
      rkkp_2L_tbl %>%
        transmute(patient_id = patientid, rkkp_2L_date = as.Date(rkkp_2L_date)),
      by = "patient_id"
    ) %>%
    mutate(
      # Use RKKP date if it is a valid post-index date AND (we found nothing, or
      # RKKP is earlier than our derived date). The > firstline_date guard stops
      # a malformed RKKP record from creating an event on/before the index.
      use_rkkp = !is.na(rkkp_2L_date) &
        rkkp_2L_date > firstline_date & (
          is.na(secondline_date) | rkkp_2L_date < secondline_date
        ),
      secondline_date       = if_else(use_rkkp, rkkp_2L_date, secondline_date),
      secondline_confidence = if_else(use_rkkp, "high_rkkp", secondline_confidence),
      # Recompute primary outcomes
      ttnt_event    = as.integer(!is.na(secondline_date) & secondline_date <= censor_date),
      ttnt_end_date = if_else(ttnt_event == 1L, secondline_date, censor_date),
      ttnt_days     = as.integer(ttnt_end_date - firstline_date),
      tfi_days      = as.integer(ttnt_end_date - firstline_end),
      ttnt_reliability = if_else(use_rkkp & ttnt_event == 1L, "high", ttnt_reliability),
      # A confirmed RKKP event re-promotes an otherwise-masked patient (e.g. a
      # weak-evidence-only or low-reliability-cohort patient) to eligible_event.
      ttnt_eligibility_reason = if_else(
        use_rkkp & ttnt_event == 1L, "eligible_event", ttnt_eligibility_reason
      ),
      ttnt_eligible = startsWith(ttnt_eligibility_reason, "eligible")
    ) %>%
    select(-use_rkkp, -rkkp_2L_date)
}


# ==============================================================================
# 8b. Seed continuous-therapy cohorts (CLL_SLL, MM) from a curated 2L source.
#
# The episode algorithm is NEVER run for these cohorts: continuous therapy
# (BTK/venetoclax in CLL, lenalidomide maintenance in MM) fabricates phantom
# 2L episodes. Instead a curated clinical source supplies the 2L EVENTS, and the
# CLL/MM first-line cohort is the denominator: a patient in the cohort but
# absent from the events source is labelled a censored negative (no 2L).
#
# This "everyone else had no event" behaviour is correct ONLY if the events
# source is COMPLETE over the cohort it claims to cover -- an omitted true 2L
# becomes a false negative. If the source's catchment is narrower than the full
# cohort, pass eligible_ids to restrict the denominator to that catchment.
# Either way, confirm completeness with the KM-in-band plot (section 11): a
# cumulative-2L curve far below the literature band means missed events.
#
# secondline_events: tibble(patientid, cohort_grouped, secondline_date)
#   events only (one row per patient who had a 2L). source_label flows into the
#   audit (secondline_confidence / cohort_reliability) so provenance is honest.
#
# Returns rows shaped to bind_rows() onto ttnt_tbl; columns absent here are
# filled with NA on bind, which is harmless for the export adapter.
# ==============================================================================

seed_continuous_cohorts <- function(firstline_tbl, patient_tbl,
                                    secondline_events, followup_end_date,
                                    source_label = "curated",
                                    eligible_ids = NULL) {
  # secondline_events: tibble(patientid, cohort_grouped, secondline_date)
  #   EVENTS ONLY -- one row per CLL/MM patient who HAD a 2L. The CLL/MM
  #   first-line cohort is the denominator: a patient absent from this table is
  #   labelled a censored negative (no 2L). This is valid ONLY if the events
  #   table is COMPLETE over the eligible denominator; an omitted true 2L
  #   becomes a false negative. Verify completeness with the KM-in-band plot.
  # eligible_ids: optional character vector of patientids whose negative status
  #   is trustworthy (i.e. the catchment the events table actually covers). Use
  #   it when the curated set is not population-wide. NULL = treat every CLL/MM
  #   first-line patient as eligible.
  follow_end <- as.Date(followup_end_date)

  fl <- firstline_tbl %>%
    filter(cohort_grouped %in% TTNT_EXCLUDED_COHORTS, !is.na(firstline_date)) %>%
    transmute(
      patient_id     = patientid, cohort_grouped, cohort_fine,
      firstline_date = as.Date(firstline_date),
      firstline_end  = as.Date(firstline_end)
    )

  if (!is.null(eligible_ids)) {
    fl <- fl %>% filter(patient_id %in% eligible_ids)
  }

  death_info <- patient_tbl %>%
    transmute(patient_id = patientid, death_date = as.Date(date_death))

  ev <- secondline_events %>%
    transmute(patient_id = patientid, cohort_grouped,
              src_2L_date = as.Date(secondline_date)) %>%
    filter(!is.na(src_2L_date)) %>%
    distinct(patient_id, cohort_grouped, .keep_all = TRUE)

  # QA: curated 2L patients with no matching first-line row signal an ID-space
  # mismatch or cohort under-capture (their 2L cannot be placed without an index).
  n_unmatched <- ev %>%
    anti_join(fl, by = c("patient_id", "cohort_grouped")) %>%
    nrow()
  if (n_unmatched > 0L) {
    message(sprintf(
      "seed_continuous_cohorts[%s]: %d curated 2L patients had no matching first-line row (ID-space mismatch or cohort under-capture?)",
      source_label, n_unmatched))
  }

  fl %>%
    left_join(ev, by = c("patient_id", "cohort_grouped")) %>%   # denominator = cohort
    left_join(death_info, by = "patient_id") %>%
    mutate(
      censor_date = pmin(coalesce(death_date, follow_end), follow_end),
      # Guard against a 2L date on/before index (data error): treat as no event.
      secondline_date       = if_else(!is.na(src_2L_date) &
                                        src_2L_date > firstline_date,
                                      src_2L_date, as.Date(NA)),
      secondline_confidence  = if_else(!is.na(secondline_date),
                                       paste0("high_", source_label),
                                       NA_character_),
      ttnt_event    = as.integer(!is.na(secondline_date) &
                                   secondline_date <= censor_date),
      ttnt_end_date = if_else(ttnt_event == 1L, secondline_date, censor_date),
      ttnt_days     = as.integer(ttnt_end_date - firstline_date),
      tfi_days      = as.integer(ttnt_end_date - firstline_end),
      retreatment_event     = ttnt_event,        # no separate weak tier here
      retreatment_days      = ttnt_days,
      only_retreatment_weak = FALSE,
      followup_days         = as.integer(censor_date - firstline_date),
      short_followup_flag   = followup_days < 365,
      firstline_end_source  = "observed",
      cohort_reliability    = paste0(source_label, "_only"),
      ttnt_reliability      = if_else(ttnt_event == 1L, "high", "censored"),
      # Patients with no curated 2L fall to eligible_censored (the intended
      # "everyone else had no event" behaviour); only no-follow-up is masked.
      ttnt_eligibility_reason = case_when(
        censor_date <= firstline_date ~ "masked_no_followup",
        ttnt_event == 1L              ~ "eligible_event",
        TRUE                          ~ "eligible_censored"
      ),
      ttnt_eligible = startsWith(ttnt_eligibility_reason, "eligible")
    )
}


# ==============================================================================
# 9.  Run
# ==============================================================================

firstline_mapped_path <- Sys.getenv("OPERA_FIRSTLINE_MAPPED_CSV", "")
if (!nzchar(firstline_mapped_path) || !file.exists(firstline_mapped_path))
  stop("Set OPERA_FIRSTLINE_MAPPED_CSV to the reviewed CLL V4 / MM V5 mapped first-line CSV")
firstline_tbl_mapped <- read_csv(firstline_mapped_path)

cat("\n=== Preparing firstline_end (imputing single-day episodes) ===\n")
firstline_tbl_mapped <- prepare_firstline_end(firstline_tbl_mapped, TTNT_CONFIG)

firstline_tbl_mapped <- firstline_tbl_mapped %>%
  mutate(
    cohort_grouped = case_when(
      cohort_fine == "MCL" ~ "MCL",
      TRUE ~ cohort_grouped
    )
  )

firstline_tbl_mapped %>%
  filter(cohort_grouped %in% TTNT_CONFIG$cohort_grouped) %>%
  count(cohort_grouped, firstline_end_source) %>%
  group_by(cohort_grouped) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  pivot_wider(names_from  = firstline_end_source,
              values_from = c(n, pct),
              values_fill = list(n = 0L, pct = 0)) %>%
  arrange(desc(coalesce(n_observed, 0L))) %>%
  print()

cat("\n=== Building post-first-line episodes ===\n")
#weirdo fix 
firstline_tbl = firstline_tbl_mapped

all_episodes <- make_secondline(tx = tx_flagged, firstline_tbl = firstline_tbl_mapped)

cat("\n=== Episode class breakdown by cohort ===\n")
all_episodes %>%
  count(cohort_grouped, episode_class) %>%
  group_by(cohort_grouped) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  arrange(cohort_grouped, desc(n)) %>%
  print(n = 60)

# build death date for patient table

patient <- patient %>%
  mutate(date_death = if_else(status == 1, as.Date(date_death_fu), as.Date(NA)))

FOLLOWUP_END_DATE <- as.Date(Sys.getenv("OPERA_SECONDLINE_CUTOFF", "2026-03-19"))
if (is.na(FOLLOWUP_END_DATE) ||
    !FOLLOWUP_END_DATE %in% as.Date(c("2026-03-04", "2026-03-19")))
  stop("OPERA_SECONDLINE_CUTOFF must be 2026-03-04 or 2026-03-19")
message("Second-line administrative cutoff: ", FOLLOWUP_END_DATE)

ttnt_tbl <- build_ttnt_outcome(
  all_episodes      = all_episodes,
  firstline_tbl     = firstline_tbl_mapped,
  patient_tbl       = patient,
  followup_end_date = FOLLOWUP_END_DATE
)

load_dataset("RKKP_LYFO_CLEAN")
load_dataset("RKKP_CLL_CLEAN")
load_dataset("RKKP_DaMyDa_CLEAN")

# ── Episode cohorts: let an RKKP 2L date override / rescue the derived date ──
# LYFO only, because only the episode cohorts (DLBCL_like, HL, MCL, T_NHL,
# Indolent_B_NHL, BL_LBL, HCL) are LYFO-registered and already rows in ttnt_tbl.
rkkp_lyfo_2L <- RKKP_LYFO_CLEAN %>%
  transmute(patientid, rkkp_2L_date = as.Date(date_treatment_2nd_line)) %>%
  filter(!is.na(rkkp_2L_date))

ttnt_tbl <- integrate_rkkp_2L(ttnt_tbl, rkkp_lyfo_2L, FOLLOWUP_END_DATE)

# ── Continuous-therapy cohorts (CLL, MM): seeded from a CURATED clinical 2L set
# More reliable than RKKP. This file is EVENTS ONLY (only patients who had a 2L).
# The CLL/MM first-line cohort is the denominator, so every CLL/MM patient NOT in
# this file is labelled a censored negative (no 2L). That is valid only if the
# file captures essentially all 2L events in the cohort; if its catchment is
# narrower, pass eligible_ids = <covered patientids> to restrict the denominator.
#
# TODO(once you have the file): set the path, the patientid column (must be the
# same ID space as firstline_tbl_mapped$patientid -- map first if it is not),
# the cohort label, and the 2L date column.


cll_seed_path <- Sys.getenv("OPERA_FIRSTLINE_CLL_SEED", "")
mm_seed_path <- Sys.getenv("OPERA_FIRSTLINE_MM_SEED", "")
if (!nzchar(cll_seed_path) || !nzchar(mm_seed_path) ||
    !file.exists(cll_seed_path) || !file.exists(mm_seed_path))
  stop("Second-line requires the same explicit CLL V4 and MM V5 seed files as first-line")
if (basename(cll_seed_path) != "CLL_V4_100lines.parquet" ||
    basename(mm_seed_path) != "MM_wide_V5_100lines.parquet")
  stop("Second-line requires final CLL_V4_100lines.parquet and MM_wide_V5_100lines.parquet")
CLL_AND <- read_parquet(cll_seed_path)
MM_AND <- read_parquet(mm_seed_path)

seed_line_two <- function(df, disease) {
  start_field <- if ("date_start_line" %in% names(df)) "date_start_line" else
    if ("date_start" %in% names(df)) "date_start" else
      stop(disease, " seed lacks date_start_line and date_start")
  if (!all(c("patientid", "line") %in% names(df)))
    stop(disease, " seed lacks patientid or line")
  message("[2L seed] ", disease, " date field: ", start_field)
  df %>%
    filter(line == 2) %>%
    transmute(patientid, secondline_date = as.Date(.data[[start_field]])) %>%
    filter(!is.na(patientid), !is.na(secondline_date)) %>%
    arrange(patientid, secondline_date) %>%
    distinct(patientid, .keep_all = TRUE)
}

secondline_curated <- bind_rows(
  seed_line_two(CLL_AND, "CLL") %>% mutate(cohort_grouped = "CLL_SLL"),
  seed_line_two(MM_AND, "MM") %>% mutate(cohort_grouped = "MM")
)

ttnt_cll_mm <- seed_continuous_cohorts(
  firstline_tbl     = firstline_tbl_mapped,
  patient_tbl       = patient,
  secondline_events = secondline_curated,
  followup_end_date = FOLLOWUP_END_DATE,
  source_label      = "curated"
)

# Optional RKKP cross-check: concordance of curated-vs-RKKP 2L on the overlap
# (supports the "we can estimate 2L well" claim). Uncomment to compute it:
# rkkp_registries <- bind_rows(
#   RKKP_CLL_CLEAN    %>% transmute(patientid, cohort_grouped = "CLL_SLL",
#                                   secondline_date = as.Date(date_treatment_2nd_line)),
#   RKKP_DaMyDa_CLEAN %>% transmute(patientid, cohort_grouped = "MM",
#                                   secondline_date = as.Date(date_treatment_2nd_line_start))
# ) %>% filter(!is.na(secondline_date))
# curated_vs_rkkp <- secondline_curated %>% filter(!is.na(secondline_date)) %>%
#   inner_join(rkkp_registries, by = c("patientid", "cohort_grouped"),
#              suffix = c("_curated", "_rkkp")) %>%
#   mutate(agree_30d = abs(as.integer(secondline_date_curated -
#                                     secondline_date_rkkp)) <= 30)
# message(sprintf("curated-vs-RKKP 2L: %d overlap, %.1f%% within 30 days",
#                 nrow(curated_vs_rkkp), 100 * mean(curated_vs_rkkp$agree_30d)))

ttnt_tbl <- bind_rows(ttnt_tbl, ttnt_cll_mm)

cat("\n=== Eligibility mask breakdown by cohort ===\n")
cat("Only 'eligible_*' rows are written to the OPERA outcome parquet.\n")
ttnt_tbl %>%
  count(cohort_grouped, ttnt_eligibility_reason) %>%
  group_by(cohort_grouped) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  arrange(cohort_grouped, desc(n)) %>%
  print(n = 60)

# ==============================================================================
# 10. Summary tables
# ==============================================================================

cat("\n=== Primary vs secondary outcome comparison ===\n")
cat("Large gap between ttnt and retreatment = high weak-evidence rate\n")
ttnt_tbl %>%
  group_by(cohort_grouped, cohort_reliability) %>%
  summarise(
    n                      = n(),
    pct_ttnt               = round(100 * mean(ttnt_event), 1),
    pct_retreatment        = round(100 * mean(retreatment_event), 1),
    pct_only_weak          = round(100 * mean(only_retreatment_weak), 1),
    median_ttnt_days       = median(ttnt_days[ttnt_event == 1], na.rm = TRUE),
    median_tfi_days        = median(tfi_days[ttnt_event == 1],  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n)) %>%
  print()

cat("\n=== Confidence breakdown for primary (ttnt) events ===\n")
ttnt_tbl %>%
  filter(ttnt_event == 1) %>%
  count(cohort_grouped, secondline_confidence) %>%
  group_by(cohort_grouped) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  arrange(cohort_grouped, desc(n)) %>%
  print(n = 40)

cat("\n=== Sensitivity: high+moderate reliability only ===\n")
ttnt_tbl %>%
  filter(!is.na(cohort_grouped)) %>%
  mutate(ttnt_event_reliable = as.integer(
    ttnt_event == 1 & ttnt_reliability %in% c("high","moderate")
  )) %>%
  group_by(cohort_grouped) %>%
  summarise(
    n                = n(),
    pct_all_events   = round(100 * mean(ttnt_event), 1),
    pct_reliable     = round(100 * mean(ttnt_event_reliable), 1),
    delta            = round(pct_all_events - pct_reliable, 1),
    .groups = "drop"
  ) %>%
  arrange(desc(delta)) %>%
  print()

### RESOLVED: this is not pct_reliable-vs-all. The primary outcome uses the
### confident events (high+moderate confidence; that is ttnt_event, which can
### never be low_single). Ambiguous patients are MASKED (absent from the
### parquet), not demoted to censored negatives. Imputed-timing events are kept
### (the windowed binarisation downstream is forgiving of small timing error).
### pct_reliable here is therefore a diagnostic, not a switch: a large delta
### just flags cohorts dominated by low-confidence events for inspection.

cat("\n=== EXCLUDED / unreliable cohorts ===\n")
cat("BL_LBL, HCL: TTNT exploratory. Use OS as primary outcome.\n")
cat("CLL_SLL, MM: use OS or RKKP PFS.\n")
firstline_tbl_mapped %>%
  filter(cohort_grouped %in% c(TTNT_EXCLUDED_COHORTS, TTNT_UNRELIABLE_COHORTS),
         !is.na(cohort_grouped)) %>%
  count(cohort_grouped, sort = TRUE) %>%
  print()


# ==============================================================================
# 11. Validation plots
# ==============================================================================

library(survival)

# A: KM curves -- both ttnt and retreatment on same panel
# Primary question: are the ttnt (solid) curves close to literature benchmarks?
# Secondary: how far above are the retreatment (dashed) curves?
benchmarks <- tribble(
  ~cohort_grouped,  ~lit_2yr,
  "DLBCL_like",     0.25, "HL", 0.10,
  "MCL",            0.55, "Indolent_B_NHL", 0.25,
  "T_NHL",          0.45, "BL_LBL", 0.30,
  "HCL",            0.08
)

km_ttnt <- ttnt_tbl %>%
  filter(!is.na(ttnt_days), ttnt_days >= 0) %>%
  group_by(cohort_grouped) %>%
  group_map(~ {
    fit <- survfit(Surv(ttnt_days / 365.25, ttnt_event) ~ 1, data = .x)
    tibble(time = fit$time, surv = fit$surv,
           lower = fit$lower, upper = fit$upper,
           outcome = "2L (confident)", cohort_grouped = .y$cohort_grouped)
  }) %>%
  bind_rows()

km_retx <- ttnt_tbl %>%
  filter(!is.na(retreatment_days), retreatment_days >= 0) %>%
  group_by(cohort_grouped) %>%
  group_map(~ {
    fit <- survfit(Surv(retreatment_days / 365.25, retreatment_event) ~ 1, data = .x)
    tibble(time = fit$time, surv = fit$surv,
           lower = fit$lower, upper = fit$upper,
           outcome = "Any retreatment", cohort_grouped = .y$cohort_grouped)
  }) %>%
  bind_rows()

bind_rows(km_ttnt, km_retx) %>%
  ggplot(aes(x = time, y = 1 - surv,
             colour = outcome, linetype = outcome)) +
  geom_step(linewidth = 0.9) +
  geom_hline(data = benchmarks, aes(yintercept = lit_2yr),
             linetype = "dashed", colour = "#DC2626") +
  geom_text(data = benchmarks,
            aes(x = 0.1, y = lit_2yr + 0.04,
                label = sprintf("Lit. ~%.0f%%", lit_2yr * 100)),
            colour = "#DC2626", size = 3, hjust = 0, inherit.aes = FALSE) +
  facet_wrap(~cohort_grouped, scales = "free_y", ncol = 3) +
  scale_x_continuous(limits = c(0, 3), breaks = 0:3,
                     labels = c("0","1yr","2yr","3yr")) +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_colour_manual(values = c("2L (confident)" = "#2563EB",
                                 "Any retreatment" = "#9CA3AF")) +
  scale_linetype_manual(values = c("2L (confident)" = "solid",
                                   "Any retreatment" = "dashed")) +
  labs(title    = "TTNT v4: confident 2L vs any retreatment",
       subtitle = "Red dashed = literature 2yr relapse rate. Gap between curves = weak-evidence events.",
       x = "Years from first-line start",
       y = "Cumulative event rate",
       colour = NULL, linetype = NULL) +
  theme_bw(base_size = 11) +
  theme(strip.text = element_text(face = "bold"),
        legend.position = "bottom")

# B: Episode breakdown -- all classes including cd20_only and retreatment_weak
all_episodes %>%
  filter(!is.na(cohort_grouped), !is.na(episode_class)) %>%
  count(cohort_grouped, episode_class) %>%
  group_by(cohort_grouped) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ggplot(aes(x = reorder(cohort_grouped, -n), y = pct, fill = episode_class)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = c(
    "secondline"         = "#1D4ED8",
    "retreatment_weak"   = "#93C5FD",
    "cd20_only"          = "#F59E0B",
    "consolidation"      = "#D97706",
    "asct_consolidation" = "#7C3AED",
    "maintenance"        = "#059669",
    "no_signal"          = "#E5E7EB"
  )) +
  labs(title    = "Post-1L episode classification (v4)",
       subtitle = "Blue = counts toward primary outcome. Light blue = retreatment only. Orange = CD20 alone (excluded).",
       x = NULL, y = "% of post-1L episodes", fill = "Episode class") +
  theme_bw(base_size = 11) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))

# C: TFI distribution stacked by confidence group
ttnt_tbl %>%
  filter(ttnt_event == 1, tfi_days >= 0) %>%
  mutate(
    tfi_capped  = pmin(tfi_days, 730),
    conf_grp    = case_when(
      ttnt_reliability == "high"     ~ "High confidence",
      ttnt_reliability == "moderate" ~ "Moderate confidence",
      TRUE                           ~ "Low confidence"
    )
  ) %>%
  ggplot(aes(x = tfi_capped,
             fill = factor(conf_grp, c("High confidence",
                                       "Moderate confidence",
                                       "Low confidence")))) +
  geom_histogram(binwidth = 30, colour = "white", alpha = 0.85, position = "stack") +
  geom_vline(xintercept = c(60, 180, 365),
             linetype = "dashed", colour = "red", alpha = 0.7) +
  facet_wrap(~cohort_grouped, scales = "free_y", ncol = 3) +
  scale_x_continuous(breaks = c(0, 60, 180, 365, 548, 730),
                     labels = c("0","2m","6m","1yr","18m","2yr")) +
  scale_fill_manual(values = c(
    "High confidence"     = "#1D4ED8",
    "Moderate confidence" = "#60A5FA",
    "Low confidence"      = "#DC2626"
  )) +
  labs(title    = "TFI: 1L end to 2L start, stacked by confidence (v4)",
       x = "Days (1L end to 2L start)", y = "N patients", fill = NULL) +
  theme_bw(base_size = 11) +
  theme(strip.text = element_text(face = "bold"), legend.position = "bottom")

# D: QA -- most common drugs initiating 2L
ttnt_tbl %>%
  filter(ttnt_event == 1, !is.na(secondline_codes_text)) %>%
  mutate(first_code = map_chr(str_split(secondline_codes_text, " \\| "), ~.x[1])) %>%
  count(cohort_grouped, first_code) %>%
  group_by(cohort_grouped) %>%
  slice_max(n, n = 8, with_ties = FALSE) %>%
  ggplot(aes(x = reorder(first_code, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  facet_wrap(~cohort_grouped, scales = "free", ncol = 3) +
  labs(title    = "Most common drugs initiating 2L (v4)",
       subtitle = "CD20 alone should no longer dominate any cohort.",
       x = NULL, y = "N patients") +
  theme_bw(base_size = 10)

# ==============================================================================
# ==============================================================================
# km_with_literature_bounds.R
#
# KM plot with literature-derived shaded bands for TTNT validation.
#
# Sources for bounds (all population-based real-world studies):
#
# DLBCL:
#   Lower 16%: Belgian Cancer Registry (Daneels 2022, n=1888, incl. >80yr)
#   Upper 25%: US COTA real-world (PMC11004909, n=1347, mean age 67)
#   Swedish national registry (Harrysson 2021, n=4243): 18.9% at 2yr, 23.1% at 5yr
#   NOTE: these are proportions of ALL 1L patients receiving 2L, not relapse rates.
#   Relapse rates (~34-40%) are higher but not all relapsed patients receive 2L.
#
# HL:
#   Lower 10%: population-based estimate for unselected national cohort
#   Upper 20%: upper range from Hodgkin 2025 review (AJH, 10-30% relapse;
#              ~10-20% actually proceed to 2L in population-based settings)
#
# BL_LBL:
#   Lower 25%, upper 35%: narrow range from intensive regimen literature;
#   population-based data scarce, trial data used with downward adjustment
#
# MCL:
#   Lower 30%: lower bound of population-based real-world at 3yr
#   Upper 45%: Swedish national MCL registry (Nordstrom HemaSphere 2023,
#              n=1367): 40% received 2L over median 6.8yr follow-up;
#              at 3yr ~30-45% depending on age/regimen mix
#   NOTE: MCL events accumulate slowly (median PFS-1 = 29.4 months);
#         these bounds are for 3yr, not 2yr.
#
# T_NHL:
#   Lower 35%, upper 50%: T-NHL is heterogeneous with poor outcomes;
#   population-based data limited, range derived from subtype data
#
# Indolent_B_NHL (FL-dominated):
#   Lower 20%: POD24 rate (~20%) as floor for 2yr TTNT
#   Upper 30%: Swedish national FL study (Wasterlid eJHaem 2024, n=FL 2007-2014):
#              26% cumulative probability of starting 2L within 2yr;
#              Indolent_B_NHL includes LPL/EMZL which have higher relapse,
#              pushing upper bound slightly higher
#
# HCL:
#   Lower 5%, upper 15%: HCL has durable responses with cladribine;
#   real-world retreatment rare within 3yr
#
# ==============================================================================

library(survival); library(dplyr); library(ggplot2); library(tibble)
library(purrr); library(stringr)

# Literature-derived bounds -- population-based real-world studies only.
# Expressed as cumulative TTNT event rate (proportion of ALL 1L patients),
# not relapse rate. Lower bound = most conservative population-based estimate.
# Upper bound = highest credible population-based estimate at the reference time.
# Literature bounds -- updated with Nordic/Danish-specific evidence
#
# HL revised:
#   Lower 8%: Nordic Lymphoma Epidemiology Group (n=2582, DK/SE/NO/FI, ages 18-49):
#             5yr relapse 13.4%; lower TTNT because elderly relapsed patients
#             often receive BSC not active 2L in Danish practice.
#   Upper 18%: Danish LYFO full age spectrum; 20-30% relapse but not all treated.
#
# T_NHL revised:
#   Lower 25%: Netherlands Cancer Registry population-based (n=821, 2014-2019):
#              2yr PFS 57% -> 43% progressed; 78% of R/R received 2L -> ~34%
#              of all T_NHL got 2L. BUT Danish cohort contains ALK+ ALCL (~half
#              of all ALCL) which has 60-70% 5yr PFS; this dilutes the aggregate
#              rate downward. Lower bound adjusted to 25%.
#   Upper 45%: upper range for high-risk T_NHL subtypes (AITL, PTCL-NOS, ALK-ALCL).
LITERATURE_BOUNDS <- tribble(
  ~cohort_grouped,  ~lower, ~upper, ~reference_years, ~notes,
  "DLBCL_like",     0.16,   0.25,   2,   "Belgian n=1888 (16%) to US n=1347 (25%); Swedish 18.9% at 2yr",
  "HL",             0.08,   0.18,   4,   "Nordic LEG n=2582: 13.4% 5yr relapse; not all relapsed get 2L in DK elderly",
  "BL_LBL",        0.25,   0.35,   2,   "Limited population-based data; trial-derived with downward adjustment",
  "MCL",            0.30,   0.45,   3,   "Swedish national MCL registry: 40% 2L over 6.8yr; 30-45% at 3yr",
  "T_NHL",          0.25,   0.45,   2,   "NL Cancer Registry: ~34% TTNT at 2yr; DK cohort includes ALK+ ALCL (better outcomes)",
  "Indolent_B_NHL", 0.20,   0.30,   2,   "Swedish FL registry: 26% at 2yr; Indolent_B_NHL includes higher-risk subtypes",
  "HCL",            0.05,   0.15,   2,   "Durable cladribine responses; retreatment rare within 3yr",
  # CLL_SLL and MM are RKKP-seeded (not episode-derived). These bands are coarse:
  # their purpose is to detect gross RKKP UNDER-recording of 2L (curve far below
  # band = missing events = contaminated negatives), not fine calibration.
  "CLL_SLL",        0.20,   0.40,   3,   "Treated CLL: ~77% reach 2L+ lifetime (long horizon); cumulative ~20-40% by 3yr",
  "MM",             0.30,   0.50,   2,   "Treated MM: ~60-65% reach 2L lifetime; median TTNT 1L->2L ~20-24mo, so ~30-50% by 2yr"
)


plot_km_with_bounds <- function(ttnt_tbl,
                                bounds = LITERATURE_BOUNDS,
                                max_years = 4,
                                save_path = NULL) {
  
  # Build KM fits
  km_ttnt <- ttnt_tbl %>%
    filter(!is.na(ttnt_days), ttnt_days >= 0, !is.na(cohort_grouped)) %>%
    group_by(cohort_grouped) %>%
    group_map(~ {
      fit <- survfit(Surv(ttnt_days / 365.25, ttnt_event) ~ 1, data = .x)
      tibble(
        time   = fit$time,
        surv   = fit$surv,
        lower  = fit$lower,
        upper  = fit$upper,
        outcome = "2L (confident)",
        cohort_grouped = .y$cohort_grouped
      )
    }) %>%
    bind_rows()
  
  km_retx <- ttnt_tbl %>%
    filter(!is.na(retreatment_days), retreatment_days >= 0, !is.na(cohort_grouped)) %>%
    group_by(cohort_grouped) %>%
    group_map(~ {
      fit <- survfit(Surv(retreatment_days / 365.25, retreatment_event) ~ 1, data = .x)
      tibble(
        time   = fit$time,
        surv   = fit$surv,
        lower  = fit$lower,
        upper  = fit$upper,
        outcome = "Any retreatment",
        cohort_grouped = .y$cohort_grouped
      )
    }) %>%
    bind_rows()
  
  km_all <- bind_rows(km_ttnt, km_retx)
  
  # Build shaded band data.
  # The band spans x = 0 to max_years, y = lower to upper.
  # This represents the range of population-based real-world estimates;
  # if the blue curve falls within the band, the algorithm is well-calibrated.
  band_data <- bounds %>%
    mutate(
      xmin = 0,
      xmax = as.numeric(reference_years) + 0.5,  # extend slightly past reference year
      ymin = lower,
      ymax = upper
    )
  
  # Reference year tick marks
  ref_lines <- bounds %>%
    transmute(cohort_grouped, xref = as.numeric(reference_years))
  
  cohorts <- intersect(unique(km_all$cohort_grouped),
                       unique(bounds$cohort_grouped))
  
  p <- ggplot() +
    # Shaded literature band
    geom_rect(
      data = band_data %>% filter(cohort_grouped %in% cohorts),
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      fill = "#FCA5A5", alpha = 0.20, inherit.aes = FALSE
    ) +
    # Vertical reference line at the literature reference year
    geom_vline(
      data = ref_lines %>% filter(cohort_grouped %in% cohorts),
      aes(xintercept = xref),
      colour = "#DC2626", linetype = "dotted", alpha = 0.5
    ) +
    # KM curves
    geom_step(
      data = km_all %>% filter(cohort_grouped %in% cohorts),
      aes(x = time, y = 1 - surv,
          colour = outcome, linewidth = outcome, linetype = outcome)
    ) +
    # 95% CI ribbon for confident 2L only
    geom_ribbon(
      data = km_ttnt %>% filter(cohort_grouped %in% cohorts),
      aes(x = time, ymin = 1 - upper, ymax = 1 - lower),
      fill = "#2563EB", alpha = 0.08
    ) +
    # Band label
    geom_text(
      data = band_data %>% filter(cohort_grouped %in% cohorts),
      aes(x = 0.05,
          y = (ymin + ymax) / 2,
          label = sprintf("Lit. range\n%.0f\u2013%.0f%%",
                          ymin * 100, ymax * 100)),
      colour = "#B91C1C", size = 2.6, hjust = 0, vjust = 0.5,
      fontface = "italic"
    ) +
    facet_wrap(~cohort_grouped, scales = "free_y", ncol = 3) +
    scale_x_continuous(
      limits = c(0, max_years),
      breaks = 0:max_years,
      labels = c("0", paste0(1:max_years, "yr"))
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    scale_colour_manual(
      values = c("2L (confident)"  = "#2563EB",
                 "Any retreatment" = "#9CA3AF")
    ) +
    scale_linewidth_manual(
      values = c("2L (confident)"  = 1.0,
                 "Any retreatment" = 0.7)
    ) +
    scale_linetype_manual(
      values = c("2L (confident)"  = "solid",
                 "Any retreatment" = "dashed")
    ) +
    labs(
      title    = "TTNT: confident 2L and any retreatment with literature bounds",
      subtitle = paste0(
        "Red band = range of population-based real-world TTNT estimates across studies.\n",
        "Blue curve within band = well-calibrated. Dotted line = reference year for band."
      ),
      x        = "Years from first-line start",
      y        = "Cumulative event rate",
      colour   = NULL, linewidth = NULL, linetype = NULL
    ) +
    theme_bw(base_size = 11) +
    theme(
      strip.text       = element_text(face = "bold"),
      legend.position  = "bottom",
      legend.key.width = unit(1.2, "cm")
    )
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 12, height = 9, dpi = 300)
  }
  
  p
}


# Print the bounds table for reference / supplementary material
print_bounds_table <- function(bounds = LITERATURE_BOUNDS) {
  cat("\n=== Literature bounds for TTNT validation ===\n")
  cat("All figures are population-based real-world studies (not trial populations).\n")
  cat("Bounds represent proportion of ALL first-line patients proceeding to 2L,\n")
  cat("not relapse rates (which are systematically higher).\n\n")
  bounds %>%
    mutate(
      range = sprintf("%.0f\u2013%.0f%%", lower * 100, upper * 100),
      ref   = sprintf("at %dyr", reference_years)
    ) %>%
    select(Disease = cohort_grouped, Range = range,
           `Ref. point` = ref, Notes = notes) %>%
    print(n = 20)
}

# Run
print_bounds_table()
plot_km_with_bounds(ttnt_tbl, max_years = 4, save_path = "check_2nd_line.pdf")

# ==============================================================================
# 12. Export adapter -> OPERA make_opera_outcome schema
#
# TTNT is an ADMINISTRATIVE outcome (registry covers the cohort; no lab-coverage
# gate), so it does not go through make_opera_outcome_lab.R. But it must emit the
# same fixed schema every OPERA outcome shares so the Python loader binarises it
# ("2L within window") with zero changes:
#
#     subject_id, split, outcome_date, index_date, censor_date
#
# index_date   = firstline_date   (time from first-line start; comparable to the
#                                   IPI-era TTNT literature and to the KM bounds).
# outcome_date = secondline_date if eligible_event, else NA.
# censor_date  = death/cutoff horizon; rows with censor_date <= index_date are
#                already masked upstream (masked_no_followup), guaranteeing the
#                censor_date >= index_date invariant.
#
# Only ttnt_eligible patients are written to the outcome parquet. THIS IS the
# per-outcome eligibility mask: a patient absent from the parquet is masked for
# the TTNT objective and contributes to no TTNT pair in the contrastive loss.
# Every patient with a firstline_date is written to <outcome>__audit.parquet
# with the reason code plus confidence/timing metadata for sensitivity analyses.
#
# Depends on assign_splits() (sourced by the main outcome pipeline) and OUTPUT_DIR.
# ==============================================================================

validate_admin_outcome <- function(result, audit) {
  stopifnot(
    "one row per patient"         = !any(duplicated(result$subject_id)),
    "censor_date not NA"          = !any(is.na(result$censor_date)),
    "index_date not NA"           = !any(is.na(result$index_date)),
    "censor_date >= index_date"   = all(result$censor_date >= result$index_date),
    "outcome_date > index_date"   =
      all(is.na(result$outcome_date) | result$outcome_date > result$index_date),
    "outcome_date <= censor_date" =
      all(is.na(result$outcome_date) | result$outcome_date <= result$censor_date),
    "no excluded patient written" =
      length(intersect(result$subject_id,
                       audit$subject_id[!audit$ttnt_eligible])) == 0L
  )
  invisible(TRUE)
}

# event_col / date_col let the same adapter emit the sensitivity outcome:
#   primary     : export_opera_ttnt(ttnt_tbl)                       # confident 2L
#   sensitivity : export_opera_ttnt(ttnt_tbl,
#                   outcome_name = "treatment_failure_retreatment",
#                   event_col = "retreatment_event",
#                   date_col  = "retreatment_date",
#                   eligible_col = "retreatment_eligible")
# For the sensitivity variant a weak-evidence episode IS the event, so its
# eligibility differs from the primary mask: define retreatment_eligible as
# "in an included/seeded cohort with follow-up" before calling (a
# weak-evidence-only patient is eligible_event there, not masked).
export_opera_ttnt <- function(ttnt_tbl,
                              outcome_name = "treatment_failure_ttnt",
                              event_col    = "ttnt_event",
                              date_col     = "secondline_date",
                              eligible_col = "ttnt_eligible",
                              output_dir   = OUTPUT_DIR,
                              write_audit  = TRUE) {

  stopifnot("assign_splits() must be sourced" = exists("assign_splits"))

  cand <- ttnt_tbl %>%
    filter(!is.na(patient_id), !is.na(firstline_date)) %>%
    distinct(patient_id, .keep_all = TRUE) %>%
    mutate(
      .event    = .data[[event_col]],
      .evt_date = as.Date(.data[[date_col]]),
      .eligible = .data[[eligible_col]],
      index_date   = as.Date(firstline_date),
      censor_date  = as.Date(censor_date),
      outcome_date = if_else(.eligible & .event == 1L,
                             .evt_date, as.Date(NA))
    )

  result <- cand %>%
    filter(.eligible) %>%
    transmute(subject_id = patient_id, index_date, outcome_date, censor_date) %>%
    assign_splits(index_date_col = index_date) %>%
    select(subject_id, split, outcome_date, index_date, censor_date)

  audit <- cand %>%
    transmute(
      subject_id = patient_id, cohort_grouped, cohort_fine,
      ttnt_eligible = .eligible, ttnt_eligibility_reason,
      index_date, outcome_date, censor_date,
      ttnt_event, retreatment_event, only_retreatment_weak,
      tfi_days, secondline_confidence, ttnt_reliability,
      cohort_reliability, firstline_end_source, short_followup_flag
    )

  validate_admin_outcome(result, audit)

  write_parquet(result, file.path(output_dir, paste0(outcome_name, ".parquet")))
  if (write_audit)
    write_parquet(audit, file.path(output_dir,
                                   paste0(outcome_name, "__audit.parquet")))

  n_elig <- nrow(result)
  n_ev   <- sum(!is.na(result$outcome_date))
  sp     <- table(result$split)
  message(sprintf(
    "%-30s  elig=%d  events=%d (%.1f%% of elig)  masked=%d  train=%d  tuning=%d  held_out=%d",
    outcome_name, n_elig, n_ev,
    if (n_elig > 0) 100 * n_ev / n_elig else NA_real_,
    sum(!audit$ttnt_eligible),
    sp["train"]    %||% 0L,
    sp["tuning"]   %||% 0L,
    sp["held_out"] %||% 0L))

  invisible(result)
}

# ── Primary OPERA TTNT outcome ────────────────────────────────────────────────
ttnt_outcome <- export_opera_ttnt(ttnt_tbl, outcome_name = "second_line_treatment")
