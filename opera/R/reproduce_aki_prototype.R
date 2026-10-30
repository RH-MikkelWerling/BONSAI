#!/usr/bin/env Rscript
# Rerun the AKI prototype that wrote AKI_table_all_dalycare_simplified.csv
# (creatinine_increase_tereza.R) in a fresh directory, changing at most the
# annual-baseline block, and compare its output with the table OPERA used.
#
# The saved prototype computes the annual baseline with slider over
# [t - 365 d, t - 7 d], but the tables (4 June) predate its last edit (15 June),
# and its commented-out sapply version looks ahead ([t - 7 d, unbounded)).
# Which one produced the table decides whether OPERA's AKI is affected.
#
#   --variant as_saved   the script exactly as saved
#   --variant lookahead  annual baseline from 7 days before onward, unbounded
#                        (the commented-out sapply window; .after = Inf)
#   --variant corrected  365 to more than 7 days before (strict, as in the
#                        reported fix; lock amendment 2026-10-30)
#
# Only aggregate counts are printed and written (comparison.json).
#
#   Rscript opera/R/reproduce_aki_prototype.R --script /ngc/people/mikwer_r/creatinine_increase_tereza.R \
#     --variant as_saved --output-dir DIR/as_saved --reference /ngc/people/mikwer_r/AKI_table_all_dalycare_simplified.csv
#
# --setup-override FILE replaces the script's load_dalycare_package.R source line
# (local tests only).

suppressPackageStartupMessages({ library(lubridate); library(data.table) })

BLOCK_START <- "^\\s*AKI\\[, base_mean_7d365 := slide_index_dbl\\("
BLOCK_END <- "^\\s*\\), by = patientid\\]"
WINDOWS <- list(
  lookahead = c(".before = days(7)", ".after  = Inf"),
  corrected = c(".before = days(365)", ".after  = -(days(7) + seconds(1))"))

parse_args <- function(argv) {
  out <- list()
  i <- 1
  while (i <= length(argv)) {
    key <- sub("^--", "", argv[[i]])
    if (i == length(argv) || startsWith(argv[[i + 1]], "--")) stop("Missing value for --", key)
    out[[key]] <- argv[[i + 1]]
    i <- i + 2
  }
  for (key in c("script", "variant", "output-dir"))
    if (is.null(out[[key]])) stop("Required: --", key)
  if (!out$variant %in% c("as_saved", names(WINDOWS)))
    stop("--variant must be as_saved, lookahead or corrected")
  out
}

# The prototype's lines with the annual-baseline block replaced (or not).
variant_script <- function(lines, variant, setup_override = NULL) {
  starts <- grep(BLOCK_START, lines)
  if (length(starts) != 1L)
    stop("Expected one active slider base_mean_7d365 block; found ", length(starts))
  ends <- grep(BLOCK_END, lines)
  end <- ends[ends > starts][1]
  if (is.na(end) || end - starts > 8L) stop("Could not find the end of the base_mean_7d365 block")
  if (variant != "as_saved") {
    window <- WINDOWS[[variant]]
    block <- c(
      sprintf("# reproduce_aki_prototype.R: annual baseline replaced (variant %s)", variant),
      "AKI[, base_mean_7d365 := slide_index_dbl(",
      "  c_value, datetime,",
      "  ~if (length(.x) > 0) mean(.x, na.rm = TRUE) else NA_real_,",
      paste0("  ", window[1], ","),
      paste0("  ", window[2]),
      "), by = patientid]")
    lines <- c(lines[seq_len(starts - 1L)], block, lines[-seq_len(end)])
  }
  if (!is.null(setup_override)) {
    loader <- grep("^\\s*source\\(.*load_dalycare_package\\.R", lines)
    if (length(loader) != 1L) stop("Expected one load_dalycare_package.R source line")
    lines[loader] <- sprintf("source('%s')", normalizePath(setup_override))
  }
  lines
}

as_utc <- function(x) {
  if (inherits(x, "POSIXct")) return(with_tz(x, "UTC"))
  parsed <- suppressWarnings(ymd_hms(x, tz = "UTC", quiet = TRUE))
  fallback <- is.na(parsed) & !is.na(x)
  parsed[fallback] <- suppressWarnings(ymd(x[fallback], tz = "UTC", quiet = TRUE))
  parsed
}

onset_keys <- function(table) {
  paste(table$patientid, format(as_utc(table$aki_start_time), "%Y-%m-%d %H:%M:%S"), sep = "|")
}

# Aggregate agreement of episode onsets (no identifiers).
compare_onsets <- function(new, reference) {
  new_keys <- onset_keys(new)
  ref_keys <- onset_keys(reference)
  changed <- union(new$patientid[!new_keys %in% ref_keys], reference$patientid[!ref_keys %in% new_keys])
  list(reference_onsets = length(ref_keys), regenerated_onsets = length(new_keys),
       onsets_in_both = sum(new_keys %in% ref_keys),
       only_reference = sum(!ref_keys %in% new_keys), only_regenerated = sum(!new_keys %in% ref_keys),
       reference_patients = length(unique(reference$patientid)),
       regenerated_patients = length(unique(new$patientid)),
       patients_with_any_difference = length(unique(changed)),
       identical = length(new_keys) == length(ref_keys) && setequal(new_keys, ref_keys))
}

main <- function(argv) {
  args <- parse_args(argv)
  output <- args[["output-dir"]]
  if (dir.exists(output) && length(list.files(output))) stop("Use a fresh --output-dir: ", output)
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  output <- normalizePath(output)
  lines <- variant_script(readLines(args$script, warn = FALSE), args$variant, args[["setup-override"]])
  used <- file.path(output, paste0("prototype_", args$variant, ".R"))
  writeLines(lines, used)

  started <- Sys.time()
  previous <- setwd(output)  # the prototype writes its tables to the working directory
  on.exit(setwd(previous), add = TRUE)
  source(used, local = new.env(parent = globalenv()))
  setwd(previous)
  table_path <- file.path(output, "AKI_table_all_dalycare_simplified.csv")
  if (!file.exists(table_path)) stop("The prototype did not write ", table_path)
  regenerated <- fread(table_path, sep = "\t")
  cat(sprintf("%s: %d onsets in %d patients (%.1f min)\n", args$variant, nrow(regenerated),
              length(unique(regenerated$patientid)),
              as.numeric(difftime(Sys.time(), started, units = "mins"))))

  summary <- list(variant = args$variant, script = normalizePath(args$script))
  if (!is.null(args$reference)) {
    summary <- c(summary, compare_onsets(regenerated, fread(args$reference, sep = "\t")))
    print(as.data.frame(summary[-(1:2)]), row.names = FALSE)
  }
  writeLines(jsonlite::toJSON(summary, auto_unbox = TRUE, pretty = TRUE),
             file.path(output, "comparison.json"))
  invisible(summary)
}

if (sys.nframe() == 0L) main(commandArgs(trailingOnly = TRUE))
