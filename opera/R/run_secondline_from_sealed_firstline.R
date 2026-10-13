# Rebuild first-line treatment objects, verify the sealed mapped CSV byte-for-byte,
# then export second-line outcomes into a fresh directory.
# Usage: Rscript opera/R/run_secondline_from_sealed_firstline.R \
#   FIRSTLINE_SCRIPT SECONDLINE_SCRIPT SEALED_MAPPED_CSV NEW_OUTPUT_DIR SOURCE_WORKDIR
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L)
  stop("Expected first-line script, second-line script, sealed mapped CSV, new output directory, source working directory")
firstline_script <- normalizePath(args[[1L]], mustWork = TRUE)
secondline_script <- normalizePath(args[[2L]], mustWork = TRUE)
sealed_csv <- normalizePath(args[[3L]], mustWork = TRUE)
output_dir <- normalizePath(args[[4L]], mustWork = FALSE)
source_workdir <- normalizePath(args[[5L]], mustWork = TRUE)
if (dir.exists(output_dir) || file.exists(output_dir))
  stop("Second-line output path already exists: ", output_dir)
if (!dir.exists(dirname(output_dir)))
  stop("Create the second-line output parent first: ", dirname(output_dir))
if (!nzchar(Sys.getenv("OPERA_FIRSTLINE_INPUT_CACHE", "")))
  stop("Set OPERA_FIRSTLINE_INPUT_CACHE; live SQL is not allowed for the sealed-cohort check")
secondline_cutoff <- Sys.getenv("OPERA_SECONDLINE_CUTOFF", "2026-03-19")
if (!identical(secondline_cutoff, "2026-03-19"))
  stop("The sealed production second-line run requires OPERA_SECONDLINE_CUTOFF=2026-03-19")
Sys.setenv(OPERA_SECONDLINE_CUTOFF = secondline_cutoff)
Sys.setenv(OPERA_FIRSTLINE_COUNTS_ONLY = "1",
           OPERA_FIRSTLINE_COUNT_DIR = file.path(dirname(output_dir),
                                                  "secondline_firstline_check_counts"),
           OPERA_FIRSTLINE_MAPPED_CSV = sealed_csv)
setwd(source_workdir)
completed <- FALSE
tryCatch(source(firstline_script, local = globalenv(), echo = FALSE),
         opera_firstline_counts_complete = function(condition) completed <<- TRUE)
if (!completed || !exists("firstline_tbl_mapped", inherits = FALSE))
  stop("First-line rebuild did not reach the mapped export")
comparison_csv <- tempfile(pattern = "secondline_mapped_check_", fileext = ".csv")
readr::write_csv(firstline_tbl_mapped, comparison_csv)
same_mapped <- identical(unname(tools::md5sum(comparison_csv)),
                         unname(tools::md5sum(sealed_csv)))
unlink(comparison_csv)
if (!same_mapped)
  stop("Rebuilt mapped CSV differs from the sealed first-line CSV; no second-line output written")
message("Sealed mapped CSV reproduced byte-for-byte: ", sealed_csv)
if (!dir.create(output_dir, mode = "0700", showWarnings = FALSE))
  stop("Could not create second-line output directory")
Sys.chmod(output_dir, mode = "0700")
OUTPUT_DIR <- output_dir
TUNING_START <- as.Date("2022-01-01")
TEST_START <- as.Date("2023-01-01")
`%||%` <- function(x, y) if (is.null(x)) y else x
assign_splits <- function(df, index_date_col = index_date) {
  df %>% dplyr::mutate(split = dplyr::case_when(
    {{ index_date_col }} >= TEST_START ~ "held_out",
    {{ index_date_col }} >= TUNING_START ~ "tuning",
    TRUE ~ "train"
  ))
}
setwd(output_dir)
source(secondline_script, local = globalenv(), echo = FALSE)
writeLines(paste0("administrative_cutoff=", secondline_cutoff),
           file.path(output_dir, "secondline_provenance.txt"))
message("Second-line outputs written to ", output_dir)
