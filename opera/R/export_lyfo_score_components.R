# IPI / NCCN-IPI components at diagnosis from the lymphoma quality registry
# (LYFO), for the supervised patients only, one row per patient.
# Run on the secure server:
# Rscript opera/R/export_lyfo_score_components.R SURVIVAL.parquet OUTPUT.parquet [DATASET]
# DATASET defaults to RKKP_LYFO_CLEAN. The columns match the defaults of
# opera.run.compute_prognostic_scores; if the registry names differ, the script
# stops and lists what it found (pass a --column-map to the scorer instead).
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(2L, 3L))
  stop("Expected released overall-survival parquet, a new output parquet, optional dataset name")
suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
})
survival_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_path <- args[[2L]]
dataset <- if (length(args) == 3L) args[[3L]] else "RKKP_LYFO_CLEAN"
if (file.exists(output_path)) stop("Refusing to overwrite: ", output_path)
source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset(dataset)
if (!exists(dataset)) stop("load_dataset('", dataset, "') did not create it")
registry <- as.data.frame(get(dataset))

sites <- c("bone_marrow_diagnosis", "CNS_diagnosis", "liver_diagnosis",
           "ventricle_diagnosis", "lung_diagnosis", "small_intestine_diagnosis",
           "colon_diagnosis")
required <- c("patientid", "age_diagnosis", "LDH_diagnosis", "AA_stage_diagnosis",
              "PS_diagnosis", sites)
missing <- setdiff(required, names(registry))
if (length(missing))
  stop("Missing in ", dataset, ": ", paste(missing, collapse = ", "),
       "\nAvailable: ", paste(names(registry), collapse = ", "))

ids <- as.character(read_parquet(survival_path, col_select = "subject_id")$subject_id)
selected <- registry[as.character(registry$patientid) %in% ids, required, drop = FALSE]
selected <- unique(selected)
repeated <- sum(duplicated(selected$patientid))
if (repeated)
  stop(repeated, " supervised patients have more than one distinct LYFO registration; ",
       "decide which one is at the first-line diagnosis before scoring")
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_parquet(selected, output_path)
cat("Supervised patients in LYFO:", nrow(selected), "of", length(ids), "\n")
cat("Complete NCCN-IPI components:",
    sum(complete.cases(selected[c("age_diagnosis", "LDH_diagnosis",
                                  "AA_stage_diagnosis", "PS_diagnosis")])), "\n")
cat("Output:", normalizePath(output_path), "\n")
