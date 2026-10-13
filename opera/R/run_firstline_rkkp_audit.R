# Secure-server aggregate audit of registry first-line dates against the mapped
# candidate branch. Does not alter the cohort or export patient identifiers.
# Usage: Rscript opera/R/run_firstline_rkkp_audit.R SCRIPT.R OUTPUT_ROOT SOURCE_WORKDIR

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Expected first-line script, output root, source workdir")
script_path <- normalizePath(args[[1]], mustWork = TRUE)
output_root <- normalizePath(args[[2]], mustWork = TRUE)
source_workdir <- normalizePath(args[[3]], mustWork = TRUE)
run_dir <- tempfile(pattern = "firstline_rkkp_", tmpdir = output_root)
if (!dir.create(run_dir, mode = "0700", showWarnings = FALSE))
  stop("Could not create private audit directory")
Sys.chmod(run_dir, mode = "0700")

Sys.setenv(OPERA_FIRSTLINE_COUNTS_ONLY = "1",
           OPERA_FIRSTLINE_COUNT_DIR = file.path(run_dir, "pipeline_counts"))
setwd(source_workdir)
completed <- FALSE
tryCatch(source(script_path, local = globalenv(), echo = FALSE),
         opera_firstline_counts_complete = function(condition) completed <<- TRUE)
if (!completed) stop("First-line script did not reach mapped export marker")
required <- c("load_dataset", "RKKP_SOURCES", "treatments", "tx_flagged",
              "dx_index", "firstline_tbl_all", "pass1_unclassified",
              "firstline_tbl", "firstline_tbl_mapped", "t_dalycare_diagnoses",
              "DX_groups", "build_cohorts_from_dx", "treatment_path")
missing <- required[!vapply(required, exists, logical(1), inherits = TRUE)]
if (length(missing)) stop("Missing candidate audit objects: ", paste(missing, collapse = ", "))
expected_n <- Sys.getenv("OPERA_EXPECTED_FIRSTLINE_N", "")
if (nzchar(expected_n) &&
    (is.na(suppressWarnings(as.integer(expected_n))) ||
     nrow(firstline_tbl_mapped) != as.integer(expected_n)))
  stop("Mapped candidate count differs from OPERA_EXPECTED_FIRSTLINE_N")

# These three registry tables are diagnostic comparators, not cohort inputs.
# Their SQL source/version must be recorded separately before final release.
load_dataset("RKKP_LYFO_CLEAN")
load_dataset("RKKP_CLL_CLEAN")
load_dataset("RKKP_DaMyDa_CLEAN")
registry_specs <- list(
  LYFO = list(data = RKKP_LYFO_CLEAN, date = "date_treatment_1st_line"),
  CLL = list(data = RKKP_CLL_CLEAN, date = "date_treatment_1st_line"),
  DaMyDa = list(data = RKKP_DaMyDa_CLEAN, date = "date_treatment_1st_line_start")
)
registry_rows <- lapply(names(registry_specs), function(registry_name) {
  spec <- registry_specs[[registry_name]]
  if (!all(c("patientid", spec$date) %in% names(spec$data)))
    stop("Registry schema mismatch: ", registry_name)
  tibble::tibble(registry = registry_name,
                 patientid = spec$data$patientid,
                 registry_date = as.Date(spec$data[[spec$date]]))
})
registry <- dplyr::bind_rows(registry_rows) |>
  dplyr::filter(!is.na(patientid), !is.na(registry_date)) |>
  dplyr::group_by(registry, patientid) |>
  dplyr::summarise(registry_date = min(registry_date), .groups = "drop")
registry <- dplyr::bind_rows(registry,
  registry |> dplyr::group_by(patientid) |>
    dplyr::summarise(registry = "ANY_RKKP", registry_date = min(registry_date),
                     .groups = "drop"))

ids <- function(x) unique(x[!is.na(x)])
cleaned_ids <- ids(treatments$patientid)
rkkp_treatment_ids <- ids(treatments$patientid[treatments$source %in% RKKP_SOURCES])
valid_tx <- tx_flagged |>
  dplyr::filter(!is.na(patient_id), !is.na(tx_date), !is.na(code), ind_ok)
eligible_ids <- ids(valid_tx$patient_id)
window_ids <- valid_tx |>
  dplyr::inner_join(dx_index |> dplyr::select(patient_id, dx_date), by = "patient_id") |>
  dplyr::filter(tx_date >= as.Date(dx_date) - 30L,
                tx_date <= as.Date(dx_date) + 3650L) |>
  dplyr::pull(patient_id) |> ids()
pass1_ids <- ids(firstline_tbl_all$patient_id[!is.na(firstline_tbl_all$firstline_date)])
pass1_classified_ids <- setdiff(pass1_ids, ids(pass1_unclassified$patientid))
mapped_ids <- ids(firstline_tbl_mapped$patientid)
pre_secondary_ids <- ids(firstline_tbl$patientid)

audit <- registry |>
  dplyr::mutate(
    has_diagnosis = patientid %in% ids(dx_index$patient_id),
    has_cleaned_treatment = patientid %in% cleaned_ids,
    has_rkkp_sourced_treatment = patientid %in% rkkp_treatment_ids,
    has_eligible_treatment = patientid %in% eligible_ids,
    has_pass1_window_treatment = patientid %in% window_ids,
    has_pass1_index = patientid %in% pass1_ids,
    has_pass1_classification = patientid %in% pass1_classified_ids,
    has_pre_secondary_index = patientid %in% pre_secondary_ids,
    included = patientid %in% mapped_ids,
    disposition = dplyr::case_when(
      included ~ "included",
      !has_diagnosis ~ "no_diagnosis_record",
      !has_cleaned_treatment ~ "no_cleaned_treatment_row",
      !has_eligible_treatment ~ "no_eligible_treatment_row",
      !has_pass1_window_treatment ~ "no_pass1_window_treatment",
      !has_pass1_index ~ "no_pass1_qualifying_episode",
      !has_pre_secondary_index ~ "indexed_but_no_final_cohort",
      TRUE ~ "excluded_after_cohort_assignment"
    )
  )
stage_names <- c("has_diagnosis", "has_cleaned_treatment",
                 "has_rkkp_sourced_treatment", "has_eligible_treatment",
                 "has_pass1_window_treatment", "has_pass1_index",
                 "has_pass1_classification", "has_pre_secondary_index", "included")
stages <- do.call(rbind, lapply(stage_names, function(stage) {
  audit |>
    dplyr::group_by(registry) |>
    dplyr::summarise(stage = stage, n_patients = sum(.data[[stage]]),
                     .groups = "drop")
}))
attrition <- audit |>
  dplyr::count(registry, disposition, name = "n_patients")

# For omitted registry-dated patients, inspect *actual treatment rows* around
# the registry date. The raw Parquet is needed to distinguish excluded SP beh
# plans from missing treatment capture. The flagged table carries the exact
# indication decisions made by the first-line pipeline. This is diagnostic:
# proximity alone does not prove the row is the registry's first-line therapy.
missing_registry <- audit |>
  dplyr::filter(!included) |>
  dplyr::select(registry, patientid, registry_date, disposition)
raw_treatment_near <- arrow::open_dataset(treatment_path) |>
  dplyr::select(patientid, date_start, source, tx_code, tx_text) |>
  dplyr::filter(patientid %in% missing_registry$patientid) |>
  dplyr::collect() |>
  dplyr::inner_join(missing_registry, by = "patientid") |>
  dplyr::filter(!is.na(date_start),
                abs(as.integer(as.Date(date_start) - registry_date)) <= 30L) |>
  dplyr::group_by(registry, patientid) |>
  dplyr::summarise(
    n_raw_near = dplyr::n(),
    n_plan_near = sum(!is.na(source) & source == "SP beh"),
    n_missing_source_near = sum(is.na(source)),
    n_kept_source_near = sum(!is.na(source) & source != "SP beh"),
    .groups = "drop")
flagged_treatment_near <- tx_flagged |>
  dplyr::filter(patient_id %in% missing_registry$patientid) |>
  dplyr::mutate(patientid = patient_id) |>
  dplyr::inner_join(missing_registry, by = "patientid") |>
  dplyr::filter(!is.na(tx_date),
                abs(as.integer(as.Date(tx_date) - registry_date)) <= 30L) |>
  dplyr::group_by(registry, patientid) |>
  dplyr::summarise(
    n_flagged_near = dplyr::n(),
    n_code_present_near = sum(!is.na(code) & nzchar(trimws(code))),
    n_eligible_near = sum(!is.na(code) & nzchar(trimws(code)) & ind_ok,
                          na.rm = TRUE),
    n_excluded_icd_near = sum(ind_excl_icd, na.rm = TRUE),
    n_excluded_text_near = sum(ind_excl_text, na.rm = TRUE),
    n_excluded_code_near = sum(ind_excl_code, na.rm = TRUE),
    n_rkkp_source_near = sum(is_rkkp, na.rm = TRUE),
    n_rkkp_override_conflict_near = sum(
      is_rkkp & ind_ok & (ind_excl_icd | ind_excl_text | ind_excl_code),
      na.rm = TRUE),
    .groups = "drop")
near_date_flags <- missing_registry |>
  dplyr::left_join(raw_treatment_near, by = c("registry", "patientid")) |>
  dplyr::left_join(flagged_treatment_near, by = c("registry", "patientid")) |>
  dplyr::mutate(dplyr::across(dplyr::starts_with("n_"),
                              ~ tidyr::replace_na(.x, 0L)),
                near_date_gate = dplyr::case_when(
                  n_raw_near == 0L ~ "no_raw_row_within_30d",
                  n_kept_source_near == 0L & n_plan_near > 0L ~ "plan_only_or_plan_plus_missing_source",
                  n_kept_source_near == 0L ~ "source_missing_only",
                  n_code_present_near == 0L ~ "nonplan_row_without_usable_code",
                  n_eligible_near == 0L ~ "row_excluded_by_indication",
                  TRUE ~ "eligible_row_within_30d_but_no_final_export"
                ))
near_date_gate <- near_date_flags |>
  dplyr::count(registry, disposition, near_date_gate, name = "n_patients")
near_date_conflicts <- near_date_flags |>
  dplyr::group_by(registry) |>
  dplyr::summarise(
    n_missing_registry_patients = dplyr::n(),
    n_with_rkkp_sourced_row_near = sum(n_rkkp_source_near > 0L),
    n_with_rkkp_override_conflict_near =
      sum(n_rkkp_override_conflict_near > 0L),
    n_with_excluded_icd_near = sum(n_excluded_icd_near > 0L),
    n_with_excluded_text_near = sum(n_excluded_text_near > 0L),
    n_with_excluded_code_near = sum(n_excluded_code_near > 0L),
    .groups = "drop")

# Date-only upper bound: classify omitted patients at the earliest RKKP date
# using the same diagnosis mapper. This does NOT establish treatment evidence.
missing_any <- audit |>
  dplyr::filter(registry == "ANY_RKKP", !included, has_diagnosis) |>
  dplyr::select(patientid, registry_date, disposition)
if (nrow(missing_any)) {
  hypothetical <- build_cohorts_from_dx(
    tx = missing_any |>
      dplyr::transmute(patientid, tx_date = registry_date),
    dx = t_dalycare_diagnoses |>
      dplyr::filter(patientid %in% missing_any$patientid),
    DX_groups = DX_groups,
    lookback_days = 2L * 3650L,
    confirm_priority_threshold = 3L,
    allow_single_hq_hit = TRUE
  )$cohorts |>
    tibble::as_tibble() |>
    dplyr::select(patientid, cohort_fine)
  if (anyDuplicated(hypothetical$patientid))
    stop("Hypothetical RKKP-date cohort assignment duplicated patient IDs")
  date_only_upper_bound <- missing_any |>
    dplyr::left_join(hypothetical, by = "patientid") |>
    dplyr::mutate(has_classifiable_disease = !is.na(cohort_fine)) |>
    dplyr::count(disposition, has_classifiable_disease, name = "n_patients")
} else {
  date_only_upper_bound <- tibble::tibble(
    disposition = character(), has_classifiable_disease = logical(),
    n_patients = integer())
}

# Inventory treatment-related registry fields among omitted patients. Export
# field names and coverage only; free-text values stay inside the SQL/R session.
treatment_field_inventory <- dplyr::bind_rows(lapply(names(registry_specs),
  function(registry_name) {
    source <- registry_specs[[registry_name]]$data
    missing_ids <- audit$patientid[
      audit$registry == registry_name & !audit$included]
    subset <- dplyr::filter(source, patientid %in% missing_ids)
    field_names <- names(subset)[grepl(
      "treat|regim|behand|therapy|terapi|drug|atc|kur|protocol",
      names(subset), ignore.case = TRUE)]
    field_names <- field_names[!grepl("date|dato", field_names,
                                       ignore.case = TRUE)]
    dplyr::bind_rows(lapply(field_names, function(field) {
      present <- !is.na(subset[[field]])
      if (is.character(subset[[field]]))
        present <- present & nzchar(trimws(subset[[field]]))
      tibble::tibble(registry = registry_name, field = field,
                     n_missing_registry_patients = length(unique(missing_ids)),
                     n_patients_with_value = length(unique(subset$patientid[present])))
    }))
  }))
if (!ncol(treatment_field_inventory))
  treatment_field_inventory <- tibble::tibble(
    registry = character(), field = character(),
    n_missing_registry_patients = integer(), n_patients_with_value = integer())

# LYFO values, not just field completeness. Restrict this to named structured
# fields; the result remains INTERNAL because rare combinations may be small.
# Registry rows are deduplicated per patient/field/value before counting.
lyfo_value_fields <- intersect(c(
  "chemo_treatment_1st_line", "chemo_regime_1_type_1st_line",
  "chemo_regime_1_n_cycles_1st_line", "chemo_regime_2_type_1st_line",
  "steroid_monotherapy_1st_line", "immunotherapy_type_1st_line",
  "maintenance_treatment_1st_line", "high_dosis_treatment_1st_line",
  "registered_despite_no_planned_treatment_1st_line",
  "treatment_planned_or_initiated_diagnosis", "treatment"
), names(RKKP_LYFO_CLEAN))
lyfo_missing <- missing_registry |>
  dplyr::filter(registry == "LYFO") |>
  dplyr::select(patientid, disposition)
lyfo_treatment_values <- dplyr::bind_rows(lapply(lyfo_value_fields, function(field) {
  RKKP_LYFO_CLEAN |>
    dplyr::select(patientid, dplyr::all_of(field)) |>
    dplyr::inner_join(lyfo_missing, by = "patientid") |>
    dplyr::transmute(patientid, disposition, field = field,
                     value = trimws(as.character(.data[[field]]))) |>
    dplyr::filter(!is.na(value), nzchar(value)) |>
    dplyr::distinct(patientid, disposition, field, value)
}))
if (nrow(lyfo_treatment_values)) {
  lyfo_treatment_values <- lyfo_treatment_values |>
    dplyr::count(disposition, field, value, name = "n_patients") |>
    dplyr::arrange(field, dplyr::desc(n_patients), value, disposition)
} else {
  lyfo_treatment_values <- tibble::tibble(
    disposition = character(), field = character(), value = character(),
    n_patients = integer())
}
date_agreement <- audit |>
  dplyr::filter(included) |>
  dplyr::left_join(firstline_tbl_mapped |>
                     dplyr::select(patientid, firstline_date), by = "patientid") |>
  dplyr::mutate(abs_days = abs(as.integer(as.Date(firstline_date) - registry_date)),
                agreement = dplyr::case_when(
                  is.na(abs_days) ~ "missing_export_date",
                  abs_days == 0L ~ "exact",
                  abs_days <= 7L ~ "1_to_7_days",
                  abs_days <= 30L ~ "8_to_30_days",
                  TRUE ~ "over_30_days")) |>
  dplyr::count(registry, agreement, name = "n_patients")
signed_date_agreement <- audit |>
  dplyr::filter(included) |>
  dplyr::left_join(firstline_tbl_mapped |>
                     dplyr::select(patientid, firstline_date, firstline_source),
                   by = "patientid") |>
  dplyr::mutate(signed_days = as.integer(as.Date(firstline_date) - registry_date),
                signed_bin = dplyr::case_when(
                  is.na(signed_days) ~ "missing_export_date",
                  signed_days < -30L ~ "earlier_over_30_days",
                  signed_days < -7L ~ "earlier_8_to_30_days",
                  signed_days < -1L ~ "earlier_2_to_7_days",
                  signed_days == -1L ~ "earlier_1_day",
                  signed_days == 0L ~ "exact",
                  signed_days == 1L ~ "later_1_day",
                  signed_days <= 7L ~ "later_2_to_7_days",
                  signed_days <= 30L ~ "later_8_to_30_days",
                  TRUE ~ "later_over_30_days")) |>
  dplyr::count(registry, firstline_source, signed_bin, name = "n_patients")

utils::write.csv(stages, file.path(run_dir, "rkkp_stage_counts_INTERNAL.csv"),
                 row.names = FALSE)
utils::write.csv(attrition, file.path(run_dir, "rkkp_attrition_INTERNAL.csv"),
                 row.names = FALSE)
utils::write.csv(near_date_gate,
                 file.path(run_dir, "rkkp_near_date_treatment_gate_INTERNAL.csv"),
                 row.names = FALSE)
utils::write.csv(near_date_conflicts,
                 file.path(run_dir, "rkkp_near_date_indication_conflicts_INTERNAL.csv"),
                 row.names = FALSE)
utils::write.csv(date_only_upper_bound,
                 file.path(run_dir, "rkkp_date_only_upper_bound_INTERNAL.csv"),
                 row.names = FALSE)
utils::write.csv(treatment_field_inventory,
                 file.path(run_dir, "rkkp_treatment_field_inventory_INTERNAL.csv"),
                 row.names = FALSE)
utils::write.csv(lyfo_treatment_values,
                 file.path(run_dir, "rkkp_lyfo_treatment_values_INTERNAL.csv"),
                 row.names = FALSE)
utils::write.csv(date_agreement,
                 file.path(run_dir, "rkkp_date_agreement_INTERNAL.csv"),
                 row.names = FALSE)
utils::write.csv(signed_date_agreement,
                 file.path(run_dir, "rkkp_signed_date_agreement_INTERNAL.csv"),
                 row.names = FALSE)
writeLines(c(
  "INTERNAL exact aggregate counts; disclosure review before sharing.",
  "No patient-level data are exported by this wrapper.",
  "RKKP registry dates are diagnostic comparators, not automatic index rescue.",
  "Date-only disease classification is an upper bound, not a rescue cohort.",
  "Treatment-field inventory gives coverage only; no text values are written.",
  "LYFO treatment values are structured-field categories for omitted patients",
  "by first failing gate; exact small cells remain INTERNAL on this server.",
  "A nonmissing registry field is not necessarily a positive treatment flag.",
  "Near-date treatment gate uses raw rows within +/-30 days of RKKP date.",
  "SP beh plans and missing-source rows are not treated as administered.",
  "A row surviving near-date gates is not proof of first-line eligibility.",
  "RKKP-source indication overrides with contrary exclusion flags are counted.",
  "Signed date difference is mapped firstline_date minus RKKP registry date.",
  "ANY_RKKP deduplicates patients across the three registries; do not sum rows.",
  "Stages are marginal flags and need not form a monotonically decreasing funnel.",
  "Disposition is mutually exclusive, in priority order; pass1 stages use the",
  "30-day pre-diagnosis grace and 3650-day forward window of the mapped branch.",
  "Indexed_but_no_final_cohort includes pass2 failure; excluded_after_cohort_",
  "assignment can include secondary-cohort exclusion or other mapping steps.",
  "Registry SQL snapshots and extraction time are not frozen by this audit.",
  paste0("First-line script: ", script_path),
  paste0("Source workdir: ", source_workdir)
), file.path(run_dir, "README_INTERNAL.txt"))
message("RKKP aggregate audit: ", normalizePath(run_dir))
