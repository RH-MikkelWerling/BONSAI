# ==============================================================================
# build_composites.R
#
# Composite outcomes built post-hoc from existing outcome parquets. Slots in
# AFTER calculate_adverse_events_update.R and lpr3_coverage_gate.R, so it
# consumes the gated components. No new ascertainment: each composite is a
# per-patient union over component event dates. Touches no engine.
#
# Construction rules (see notes per rule):
#   eligibility = INTERSECTION. A patient must be present in EVERY component
#     parquet; absent from any -> masked from the composite (audited). The mask
#     propagates automatically: restrict or gate a component and the composite
#     restricts with it.
#   censor = INTERSECTION observation = earliest component censor, because to
#     declare composite-event-free you must have been observing every component.
#   event = UNION = earliest component event, kept only if it occurred at or
#     before the intersection censor (a later event in a long-observed component,
#     after another component dropped out, is not credited). For same-panel
#     components (CBC, chemistry) the censors coincide so this almost never bites;
#     for administrative components they are identical (death-or-cutoff).
#   death convention is set by MEMBERSHIP, not a separate code path:
#     include overall_survival as a component -> death is an EVENT (failure /
#       survival composites; death_is_event = TRUE).
#     omit it -> death stays as each component's censoring and is recovered as a
#       competing risk at evaluation (toxicity composites; death_is_event = FALSE).
#
# Output schema UNCHANGED: subject_id, split, outcome_date, index_date, censor_date.
# Reads the sealed base and second-line directories; writes to a new directory.
# ==============================================================================

suppressPackageStartupMessages({
  library(arrow); library(dplyr); library(purrr); library(rlang); library(tibble)
})
BASE_OUTCOMES_DIR <- Sys.getenv("OPERA_BASE_OUTCOMES_DIR", "")
SECONDLINE_DIR <- Sys.getenv("OPERA_SECONDLINE_DIR", "")
OUTPUT_DIR <- Sys.getenv("OPERA_COMPOSITES_DIR", "")
if (!nzchar(BASE_OUTCOMES_DIR) || !dir.exists(BASE_OUTCOMES_DIR) ||
    !nzchar(SECONDLINE_DIR) || !dir.exists(SECONDLINE_DIR))
  stop("Set OPERA_BASE_OUTCOMES_DIR and OPERA_SECONDLINE_DIR to existing directories")
if (!file.exists(file.path(BASE_OUTCOMES_DIR, "coverage_gated_outcomes.csv")))
  stop("Composite base outcomes are not coverage-gated; run and review the versioned coverage gate first")
if (!nzchar(OUTPUT_DIR) || file.exists(OUTPUT_DIR))
  stop("Set OPERA_COMPOSITES_DIR to a new, nonexistent directory")
component_path <- function(name) {
  root <- if (name == "second_line_treatment") SECONDLINE_DIR else BASE_OUTCOMES_DIR
  file.path(root, paste0(name, ".parquet"))
}
required_components <- c("second_line_treatment", "overall_survival",
                         "next_hematological_malignancy", "neutropenia_g3plus",
                         "thrombocytopenia_g3plus", "anemia_g3plus",
                         "ALT_increased_g3plus", "AST_increased_g3plus",
                         "bilirubin_increased_g3plus", "serious_infection_composite",
                         "infections_prescription_based", "infections_IV_based")
missing_components <- required_components[!file.exists(vapply(required_components,
                                                               component_path, character(1)))]
if (length(missing_components))
  stop("Missing composite components: ", paste(missing_components, collapse = ", "))
max_censor <- function(name) max(as.Date(arrow::read_parquet(component_path(name))$censor_date),
                                 na.rm = TRUE)
if (max_censor("second_line_treatment") != max_censor("overall_survival"))
  stop("Second-line and overall-survival administrative censor dates differ; resolve cutoff before composites")
if (!dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE, mode = "0700"))
  stop("Could not create composite output directory")
Sys.chmod(OUTPUT_DIR, mode = "0700")
SCHEMA <- c("subject_id", "split", "outcome_date", "index_date", "censor_date")

# NOTE: set these to the actual parquet names in your output dir. The second-line
# parquet shows up as "second_line_treatment" in the eval plots; change if yours
# is named differently (e.g. "treatment_failure_ttnt" or "TTNT").
SECOND_LINE <- "second_line_treatment"

build_composite <- function(outcome_name, components, death_is_event,
                            output_dir = OUTPUT_DIR) {
  
  paths <- vapply(components, component_path, character(1))
  missing <- components[!file.exists(paths)]
  if (length(missing)) {
    stop(outcome_name, " missing components: ", paste(missing, collapse = ", "))
  }
  
  has_os <- "overall_survival" %in% components
  if (death_is_event && !has_os)
    stop(outcome_name, ": death_is_event=TRUE needs overall_survival as a component")
  if (!death_is_event && has_os)
    stop(outcome_name, ": overall_survival present but death_is_event=FALSE")
  
  comps <- imap(set_names(paths, components), function(p, nm) {
    read_parquet(p) %>%
      transmute(subject_id,
                !!paste0("evt_", nm) := as.Date(outcome_date),
                !!paste0("cen_", nm) := as.Date(censor_date),
                !!paste0("idx_", nm) := as.Date(index_date),
                !!paste0("spl_", nm) := split)
  })
  
  all_ids   <- map(comps, "subject_id")
  union_ids <- reduce(all_ids, union)
  joined    <- reduce(comps, inner_join, by = "subject_id")   # intersection
  
  evt_cols <- paste0("evt_", components)
  cen_cols <- paste0("cen_", components)
  idx_cols <- paste0("idx_", components)
  
  composite_censor <- as.Date(do.call(pmin, c(joined[cen_cols], list(na.rm = TRUE))),
                              origin = "1970-01-01")
  earliest_event   <- suppressWarnings(
    as.Date(do.call(pmin, c(joined[evt_cols], list(na.rm = TRUE))), origin = "1970-01-01"))
  index_date <- joined[[idx_cols[1]]]
  
  stopifnot("component index_dates disagree" =
              all(vapply(idx_cols[-1], function(cc) all(joined[[cc]] == index_date), logical(1))))
  stopifnot("component splits disagree" =
              all(vapply(paste0("spl_", components[-1]),
                         function(cc) all(joined[[cc]] == joined[[paste0("spl_", components[1])]]),
                         logical(1))))
  
  is_event <- !is.na(earliest_event) & earliest_event <= composite_censor
  
  result <- tibble(
    subject_id   = joined$subject_id,
    split        = joined[[paste0("spl_", components[1])]],
    outcome_date = if_else(is_event, earliest_event, as.Date(NA)),
    index_date   = index_date,
    censor_date  = composite_censor
  )
  
  # audit: everyone in the union; who was dropped for missing a component
  present_count <- rowSums(vapply(all_ids, function(ids) union_ids %in% ids, logical(length(union_ids))))
  audit <- tibble(
    subject_id           = union_ids,
    n_components_present  = present_count,
    n_components          = length(components),
    eligible              = present_count == length(components),
    reason                = if_else(present_count == length(components),
                                    "eligible", "masked_missing_component")
  )
  
  stopifnot(
    "schema" = identical(names(result), SCHEMA),
    "one row per patient" = !any(duplicated(result$subject_id)),
    "censor not NA" = !any(is.na(result$censor_date)),
    "censor >= index" = all(result$censor_date >= result$index_date),
    "outcome > index" = all(is.na(result$outcome_date) | result$outcome_date > result$index_date),
    "outcome <= censor" = all(is.na(result$outcome_date) | result$outcome_date <= result$censor_date)
  )
  
  write_parquet(result, file.path(output_dir, paste0(outcome_name, ".parquet")))
  write_parquet(audit %>% filter(!eligible),
                file.path(output_dir, paste0(outcome_name, "__audit.parquet")))
  
  n_el <- nrow(result); n_ev <- sum(!is.na(result$outcome_date))
  message(sprintf("  %-34s elig=%d (of %d union)  events=%d (%.1f%%)  death_is_event=%s",
                  outcome_name, n_el, length(union_ids), n_ev,
                  if (n_el > 0) 100 * n_ev / n_el else 0, death_is_event))
  invisible(result)
}

message("Building composites:")

# Failure / survival composites (death is an EVENT, via overall_survival).
build_composite("treatment_failure",
                c(SECOND_LINE, "overall_survival"),
                death_is_event = TRUE)

build_composite("treatment_failure_transformation",
                c(SECOND_LINE, "overall_survival", "next_hematological_malignancy"),
                death_is_event = TRUE)

# Toxicity composites (death is a COMPETING risk; recovered at evaluation).
build_composite("severe_cytopenia",
                c("neutropenia_g3plus", "thrombocytopenia_g3plus", "anemia_g3plus"),
                death_is_event = FALSE)

build_composite("hepatotoxicity",
                c("ALT_increased_g3plus", "AST_increased_g3plus", "bilirubin_increased_g3plus"),
                death_is_event = FALSE)

build_composite("infections_all_definitions",
                c("serious_infection_composite", "infections_prescription_based", "infections_IV_based"),
                death_is_event = FALSE)

message("Done. Composites written to ", OUTPUT_DIR,
        " (with __audit parquets for masked patients).")
