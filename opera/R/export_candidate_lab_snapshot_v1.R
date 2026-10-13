# INTERNAL, read-only source extract for export_candidate_outcomes_v1.py.
# Run in the DALY-CARE R environment. Never copy the patient-level output off
# the secure server. The destination must be a new directory.

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(purrr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(tibble)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript opera/R/export_candidate_lab_snapshot_v1.R NEW_OUTPUT_DIR")
out_dir <- normalizePath(args[[1]], winslash = "/", mustWork = FALSE)
if (file.exists(out_dir)) stop("Refusing to reuse candidate snapshot directory: ", out_dir)
if (!dir.exists(dirname(out_dir))) stop("Parent directory must exist: ", dirname(out_dir))

source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset("SDS_laboratorieproevesvar", selected_columns = c(
  "patientid", "analysiscode", "samplingdate", "samplingtime", "samplevalue",
  "unit", "referenceinterval_lowerlimit", "referenceinterval_upperlimit"))
load_npu_common()

npu_names <- c(
  neutrophils = "NPU.NEU", ALT = "NPU.ALAT", AST = "NPU.ASAT",
  bilirubin = "NPU.BIL", creatinine = "NPU.KREA", eosinophils = "NPU.EOS")
missing <- npu_names[!vapply(npu_names, exists, logical(1), inherits = TRUE)]
if (length(missing)) stop("Missing NPU mappings: ", paste(missing, collapse = ", "))
mapping <- imap_dfr(as.list(npu_names), function(var_name, analyte) {
  tibble(NPU = as.character(get(var_name, inherits = TRUE)), analyte = analyte)
})
if (anyDuplicated(mapping$NPU)) stop("NPU code maps to more than one candidate analyte")

cleaned <- SDS_laboratorieproevesvar %>%
  clean_lab_values(value = samplevalue, unit = unit)
if (!"NPU" %in% names(cleaned) && "analysiscode" %in% names(cleaned)) {
  cleaned <- cleaned %>% mutate(NPU = as.character(analysiscode))
}
needed <- c("patientid", "NPU", "samplingdate", "value2",
            "referenceinterval_lowerlimit", "referenceinterval_upperlimit")
if (!all(needed %in% names(cleaned)) || !any(c("unit2", "unit") %in% names(cleaned)))
  stop("clean_lab_values output lacks: ",
       paste(c(setdiff(needed, names(cleaned)),
               if (!any(c("unit2", "unit") %in% names(cleaned))) "unit2/unit"),
             collapse = ", "),
       "; available: ", paste(names(cleaned), collapse = ", "))
unit_col <- if ("unit2" %in% names(cleaned)) "unit2" else "unit"
snapshot <- cleaned %>%
  mutate(NPU = as.character(NPU)) %>%
  inner_join(mapping, by = "NPU") %>%
  transmute(subject_id = patientid, analyte, npu_code = NPU,
            sample_date = as.Date(samplingdate), value = as.numeric(value2),
            unit = as.character(.data[[unit_col]]),
            ref_lower = suppressWarnings(as.numeric(referenceinterval_lowerlimit)),
            ref_upper = suppressWarnings(as.numeric(referenceinterval_upperlimit)))
if (!nrow(snapshot)) stop("No candidate analyte measurements; inspect NPU mapping")

dir.create(out_dir, recursive = FALSE)
write_parquet(snapshot, file.path(out_dir, "candidate_labs_v1_INTERNAL.parquet"))
unit_inventory <- snapshot %>%
  count(analyte, npu_code, unit, name = "n_measurements") %>%
  arrange(analyte, npu_code, unit)
write_csv(unit_inventory, file.path(out_dir, "unit_inventory_v1_INTERNAL.csv"))
write_csv(mapping, file.path(out_dir, "npu_mapping_v1_INTERNAL.csv"))
message("Wrote INTERNAL candidate lab snapshot to ", out_dir)
