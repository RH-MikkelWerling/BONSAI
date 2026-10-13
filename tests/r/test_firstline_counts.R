# Run from the repository root: Rscript tests/r/test_firstline_counts.R
# Exercises only the base-R instrumentation; no registry data/package loaded.
test_firstline_counts <- function() {
  lines <- readLines("opera/R/identification_first_line_treatment_v2.R", warn = FALSE)
  marker_line <- grep('fl_count\\("mapped_csv_export_population"', lines)[1]
  guard_line <- grep('OPERA_FIRSTLINE_COUNTS_ONLY', lines)[1]
  export_line <- grep('saveRDS\\(firstline_tbl', lines)[1]
  stopifnot(!is.na(marker_line), !is.na(guard_line), !is.na(export_line),
            marker_line < guard_line, guard_line < export_line)
  start <- grep("^# Aggregate-only instrumentation", lines)[1]
  end <- grep('^source\\("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R"', lines)[1] - 1L
  stopifnot(!is.na(start), !is.na(end), end > start)
  scope <- new.env(parent = baseenv())
  eval(parse(text = lines[start:end]), envir = scope)
  scope$TRUST_RKKP <- TRUE
  input <- data.frame(patientid = c(1L, 1L, 2L, NA_integer_),
                      source = c("a", "b", "a", "a"))
  before <- input
  scope$fl_count("synthetic_records", input)
  stopifnot(identical(input, before))
  counts <- scope$.fl_audit$counts[[1]]
  stopifnot(counts$n_rows == 4L, counts$n_patients == 2L,
            counts$n_missing_id_rows == 1L, counts$n_repeated_patient_rows == 1L)
  scope$fl_count("empty", input[FALSE, ])
  stopifnot(scope$.fl_audit$counts[[2]]$n_patients == 0L)
  output_root <- tempfile("opera_firstline_count_test_")
  old <- Sys.getenv("OPERA_FIRSTLINE_COUNT_DIR", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("OPERA_FIRSTLINE_COUNT_DIR") else
      Sys.setenv(OPERA_FIRSTLINE_COUNT_DIR = old)
  }, add = TRUE)
  Sys.setenv(OPERA_FIRSTLINE_COUNT_DIR = output_root)
  path <- scope$fl_write_counts()
  # Test artifacts retained in R's temporary directory, never a production path.
  output <- read.csv(file.path(path, "pipeline_counts.csv"))
  breakdown <- read.csv(file.path(path, "pipeline_breakdowns.csv"))
  stopifnot(nrow(output) == 2L,
            !any(c("patientid", "patient_id", "subject_id") %in% names(output)),
            !any(c("patientid", "patient_id", "subject_id") %in% names(breakdown)))
  message("First-line aggregate instrumentation tests passed; artifacts: ", path)
}
test_firstline_counts()
