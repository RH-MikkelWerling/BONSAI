# Run on the secure server after filling the marked source blocks below.
# Produces INTERNAL one-row-per-patient PS metadata and aggregate audit files.
# No cohort, split, outcome or model changes. Never infer raw ECOG from grouped PS.
suppressPackageStartupMessages({library(arrow); library(jsonlite)})

prepared_dir <- "data/clinical_registry_refresh/20261111/prepared_clinical"
output_dir <- "data/clinical_registry_refresh/20261111/quality_table1_20261116"
# Set TRUE only after you have identified and loaded all intended source tables.
all_required_sources_loaded <- FALSE
# Leave empty to reject cross-registry disagreements. If needed, explicitly order
# ALL loaded source names here after reviewing which registry should take priority.
source_priority <- character()

cohort <- as.data.frame(read_parquet(file.path(prepared_dir, "patient_metadata.parquet")))
cohort <- cohort[cohort$split %in% c("train", "tuning", "held_out"), , drop=FALSE]
canonical_ids <- function(x) {
  if (is.numeric(x) && !inherits(x, "integer64") &&
      any(!is.finite(x) | x != trunc(x) | abs(x) > 2^53))
    stop("Non-exact numeric patient IDs; preserve integer64/character IDs")
  result <- if (is.numeric(x) && !inherits(x, "integer64"))
    format(x, scientific=FALSE, trim=TRUE) else as.character(x)
  if (anyNA(result) || any(!nzchar(trimws(result)))) stop("Missing patient IDs")
  result
}
cohort$subject_id <- canonical_ids(cohort$subject_id)
if (anyDuplicated(cohort$subject_id)) stop("Frozen cohort has duplicate patient IDs")

parts <- list()
source_notes <- list()
add_ps_source <- function(data, source, id_column, ps_column, baseline_confirmed,
                          source_description) {
  if (source %in% names(parts)) stop("Duplicate source name: ", source)
  if (!isTRUE(baseline_confirmed))
    stop(source, ": first restrict to the appropriate diagnosis/baseline record, then confirm timing")
  if (!all(c(id_column, ps_column) %in% names(data)))
    stop(source, ": mapped ID/PS columns are absent")
  ids <- canonical_ids(data[[id_column]])
  raw <- trimws(as.character(data[[ps_column]]))
  missing <- is.na(raw) | raw == ""
  ps <- suppressWarnings(as.numeric(raw))
  bad <- !missing & (is.na(ps) | !is.finite(ps) | ps != trunc(ps) | !ps %in% 0:4)
  if (any(bad)) stop(source, ": unmapped PS values; inspect coding and explicitly recode missing/other values before calling add_ps_source")
  keep <- !missing & ids %in% cohort$subject_id
  selected <- unique(data.frame(subject_id=ids[keep], PS=ps[keep],
                                 quality_registry_ps_source=source))
  if (anyDuplicated(selected$subject_id))
    stop(source, ": conflicting PS records per patient; select the intended baseline episode explicitly")
  parts[[source]] <<- selected
  source_notes[[source]] <<- list(id_column=id_column, ps_column=ps_column,
    baseline_confirmed=TRUE, source_description=source_description,
    patients_with_ps_in_frozen_cohort=nrow(selected))
}

# ---------- SOURCE 1: refreshed LYFO (known existing extract) ----------
lyfo_path <- "data/clinical_registry_refresh/20261111/lyfo.parquet"
lyfo <- as.data.frame(read_parquet(lyfo_path))
add_ps_source(lyfo, "LYFO", "patientid", "PS_diagnosis", TRUE, lyfo_path)

# ---------- SOURCE 2: CLL quality registry -- FILL HERE ----------
# Load the actual table using your usual loader, read_parquet(), readRDS(), etc.
# source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
# load_dataset("REPLACE_WITH_ACTUAL_TABLE_NAME")
# cll_quality <- YOUR_LOADED_OBJECT
# names(cll_quality)  # Inspect locally on the server; identify IDs and baseline PS.
# cll_quality <- ... # Select the correct diagnosis/first-line episode if repeated.
# Recode any documented non-ECOG/missing codes explicitly here, not by guessing.
# add_ps_source(cll_quality, "CLL_registry", "REPLACE_ID_COLUMN",
#               "REPLACE_PS_COLUMN", FALSE, "REPLACE_SOURCE_TABLE_OR_PATH")
# Change FALSE to TRUE only after baseline timing/episode selection is checked.

# ---------- SOURCE 3: myeloma quality registry -- FILL HERE ----------
# myeloma_quality <- YOUR_LOADING_COMMAND
# names(myeloma_quality)
# myeloma_quality <- ... # Baseline/first-line record selection and explicit recoding.
# add_ps_source(myeloma_quality, "Myeloma_registry", "REPLACE_ID_COLUMN",
#               "REPLACE_PS_COLUMN", FALSE, "REPLACE_SOURCE_TABLE_OR_PATH")

# ---------- SOURCE 4+: other quality registries -- COPY/FILL AS NEEDED ----------
# other_quality <- YOUR_LOADING_COMMAND
# add_ps_source(other_quality, "DESCRIPTIVE_SOURCE_NAME", "REPLACE_ID_COLUMN",
#               "REPLACE_PS_COLUMN", FALSE, "REPLACE_SOURCE_TABLE_OR_PATH")
# For other Table 1 variables (stage, treatment, etc.), identify their source,
# timing and coding here too. Keep different staging/risk systems separate.
# The downstream merge below currently updates PS only; add other fields with
# explicit definitions rather than passing an unfiltered registry table through.

if (!length(parts)) stop("No PS sources configured")
if (dir.exists(output_dir)) stop("Use a fresh output directory: ", output_dir)
all_ps <- do.call(rbind, parts)
if (!nrow(all_ps)) stop("No supplied PS records overlap the frozen cohort")
value_counts <- tapply(all_ps$PS, all_ps$subject_id, function(x) length(unique(x)))
conflict_ids <- names(value_counts)[value_counts > 1]
if (length(conflict_ids) && !length(source_priority)) {
  dir.create(dirname(output_dir), recursive=TRUE, showWarnings=FALSE)
  conflict_path <- paste0(output_dir, "_ps_conflicts_", format(Sys.time(), "%Y%m%d_%H%M%S"), "_INTERNAL.parquet")
  write_parquet(all_ps[all_ps$subject_id %in% conflict_ids, ],
                conflict_path)
  stop("Cross-registry PS disagreement for ", length(conflict_ids),
       " patients; inspect ", conflict_path, " and specify a justified priority or correct the source")
}
if (length(source_priority)) {
  if (anyDuplicated(source_priority) || !setequal(source_priority, names(parts)))
    stop("source_priority must list every loaded source exactly once")
  all_ps <- all_ps[order(match(all_ps$quality_registry_ps_source, source_priority)), ]
}
selected <- all_ps[!duplicated(all_ps$subject_id), ]
dir.create(output_dir, recursive=TRUE)
write_parquet(selected, file.path(output_dir, "quality_ps_INTERNAL.parquet"))
coverage <- merge(cohort[c("subject_id", "split", "cohort_fine")], selected,
                  by="subject_id", all.x=TRUE, sort=FALSE)
coverage$has_refreshed_ps <- !is.na(coverage$PS)
coverage$quality_registry_ps_source[is.na(coverage$quality_registry_ps_source)] <- "No refreshed PS"
counts <- aggregate(list(n_patients=rep(1L, nrow(coverage))),
                    coverage[c("split", "cohort_fine", "quality_registry_ps_source", "has_refreshed_ps")], sum)
write.csv(counts, file.path(output_dir, "quality_ps_coverage.csv"), row.names=FALSE)
write_json(list(all_required_sources_loaded=all_required_sources_loaded,
  sources=source_notes, source_priority=source_priority,
  cross_registry_conflicting_patients=length(conflict_ids),
  frozen_patients=nrow(cohort), patients_with_refreshed_ps=nrow(selected),
  scope="PS only; other Table 1 quality-registry fields still require explicit mappings"),
  file.path(output_dir, "quality_registry_manifest.json"), pretty=TRUE, auto_unbox=TRUE)
cat("Prepared PS metadata:", file.path(output_dir, "quality_ps_INTERNAL.parquet"), "\n")
cat("All intended sources loaded:", all_required_sources_loaded, "\n")
cat("Keep INTERNAL files on the server; download only aggregate coverage/review outputs.\n")
