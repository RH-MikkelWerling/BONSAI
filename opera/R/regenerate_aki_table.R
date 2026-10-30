#!/usr/bin/env Rscript
# Regenerate the AKI episode table read by OPERA's outcomes (OPERA_AKI_TABLE) and
# by the tabular features (data source "AKI"): AKI_table_all_dalycare_simplified.csv.
#
# Uses the DALY-CARE core writer (core_scripts/AE_AKI.R, from AKI_function_toNGC_v2.R)
# either as published ("original") or with only the annual-baseline window corrected
# ("corrected"): 365 to more than 7 days before the sample, instead of 7 days before
# onward (lock amendment 2026-10-30). Nothing else in the definition changes.
#
# Run "original" first with --reference: it must reproduce the existing table exactly
# before the "corrected" table is trusted. Only aggregate counts are printed.
#
#   Rscript opera/R/regenerate_aki_table.R --writer PATH/core_scripts/AE_AKI.R \
#     --version original --output DIR/original/AKI_table_all_dalycare_simplified.csv \
#     --reference PATH/AKI_table_all_dalycare_simplified.csv [--setup FILE.R]
#   (the same with --version corrected and its own output directory)
#
# --setup sources an R file that loads the dalycare package (load_npu_common,
# load_dataset); without it, library(dalycare) is used.

suppressPackageStartupMessages({
  library(dplyr); library(lubridate); library(data.table); library(tidyr)
})

ORIGINAL_WINDOW <- "datetime >= datetime[i] - days(365) & datetime[i] - days(7) <= datetime"
CORRECTED_WINDOW <- "datetime >= datetime[i] - days(365) & datetime < datetime[i] - days(7)"
DEFAULT_COLUMNS <- c("patientid", "aki_start_time", "aki_episode")

parse_args <- function(argv) {
  out <- list()
  i <- 1
  while (i <= length(argv)) {
    key <- sub("^--", "", argv[[i]])
    if (i == length(argv) || startsWith(argv[[i + 1]], "--")) stop("Missing value for --", key)
    out[[key]] <- argv[[i + 1]]
    i <- i + 2
  }
  for (key in c("writer", "version", "output"))
    if (is.null(out[[key]])) stop("Required: --", key)
  if (!out$version %in% c("original", "corrected")) stop("--version must be original or corrected")
  out
}

# The writer's functions, as published or with only the window corrected.
writer_functions <- function(path, version) {
  source_text <- readLines(path, warn = FALSE)
  hits <- grepl(ORIGINAL_WINDOW, source_text, fixed = TRUE)
  if (sum(hits) != 1L)
    stop("Expected the published annual-baseline window exactly once in ", path,
         "; found ", sum(hits), ". Not the audited writer: stop and inspect.")
  if (version == "corrected")
    source_text[hits] <- sub(ORIGINAL_WINDOW, CORRECTED_WINDOW, source_text[hits], fixed = TRUE)
  functions <- new.env(parent = globalenv())
  eval(parse(text = source_text), envir = functions)
  functions
}

# One row per episode onset, in the columns of the reference table.
episode_starts <- function(all_aki, columns) {
  missing <- setdiff(columns, names(all_aki))
  if (length(missing)) stop("Writer output lacks reference columns: ", paste(missing, collapse = ", "))
  all_aki %>%
    filter(!is.na(aki_start_time)) %>%
    select(all_of(columns)) %>%
    arrange(patientid, aki_start_time)
}

as_utc <- function(x) {
  if (inherits(x, "POSIXct")) return(with_tz(x, "UTC"))
  parsed <- suppressWarnings(ymd_hms(x, tz = "UTC", quiet = TRUE))
  fallback <- is.na(parsed) & !is.na(x)
  parsed[fallback] <- suppressWarnings(ymd(x[fallback], tz = "UTC", quiet = TRUE))
  parsed
}

episode_keys <- function(table) {
  paste(table$patientid, format(as_utc(table$aki_start_time), "%Y-%m-%d %H:%M:%S"), sep = "|")
}

# Aggregate agreement between two episode tables (no identifiers printed).
compare_tables <- function(new, reference) {
  new_keys <- episode_keys(new)
  ref_keys <- episode_keys(reference)
  changed_patients <- union(
    new$patientid[!new_keys %in% ref_keys], reference$patientid[!ref_keys %in% new_keys])
  list(reference_onsets = length(ref_keys), regenerated_onsets = length(new_keys),
       onsets_in_both = sum(new_keys %in% ref_keys),
       only_reference = sum(!ref_keys %in% new_keys), only_regenerated = sum(!new_keys %in% ref_keys),
       reference_patients = length(unique(reference$patientid)),
       regenerated_patients = length(unique(new$patientid)),
       patients_with_any_difference = length(unique(changed_patients)),
       identical = setequal(new_keys, ref_keys) && length(new_keys) == length(ref_keys))
}

main <- function(argv) {
  args <- parse_args(argv)
  if (file.exists(args$output)) stop("Refusing to overwrite ", args$output)
  functions <- writer_functions(args$writer, args$version)
  reference <- if (!is.null(args$reference)) fread(args$reference, sep = "\t") else NULL
  columns <- if (is.null(reference)) DEFAULT_COLUMNS else names(reference)
  if (!is.null(args$setup)) source(args$setup) else suppressPackageStartupMessages(library(dalycare))

  started <- Sys.time()
  all_aki <- functions$run_AE_AKI_all()$all_AKI
  starts <- episode_starts(all_aki, columns)
  starts$aki_start_time <- format(as_utc(starts$aki_start_time), "%Y-%m-%d %H:%M:%S")
  dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
  fwrite(starts, args$output, sep = "\t")
  cat(sprintf("%s: %d onsets in %d patients written to %s (%.1f min)\n", args$version,
              nrow(starts), length(unique(starts$patientid)), args$output,
              as.numeric(difftime(Sys.time(), started, units = "mins"))))

  if (!is.null(reference)) {
    summary <- compare_tables(starts, reference)
    print(as.data.frame(summary), row.names = FALSE)
    if (args$version == "original" && !summary$identical)
      stop("The published writer does not reproduce the reference table. ",
           "Do not use the corrected table until this is explained ",
           "(different laboratory extract, patient set or writer version).")
  }
  invisible(NULL)
}

if (sys.nframe() == 0L) main(commandArgs(trailingOnly = TRUE))
