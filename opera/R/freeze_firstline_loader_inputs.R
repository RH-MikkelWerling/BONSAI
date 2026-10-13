# Secure-server only. Usage: Rscript opera/R/freeze_firstline_loader_inputs.R NEW_CACHE_DIR
# Reuses a verified cache, or loads the three first-line objects once and freezes them.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Expected exactly one cache directory")
cache_dir <- normalizePath(args[[1]], mustWork = FALSE)
if (file.exists(cache_dir) && !dir.exists(cache_dir)) stop("Cache path is not a directory")
if (dir.exists(cache_dir) && isTRUE(nzchar(Sys.readlink(cache_dir))))
  stop("Cache directory must not be a symlink")
sha256 <- function(path) {
  executable <- Sys.which("sha256sum")
  if (!nzchar(executable)) stop("sha256sum is required on the secure server")
  result <- suppressWarnings(system2(executable, shQuote(path), stdout = TRUE, stderr = TRUE))
  if (!is.null(attr(result, "status")) || length(result) != 1L)
    stop("Could not hash: ", path)
  substring(result, 1L, 64L)
}
names_expected <- c("patient.rds", "t_dalycare_diagnoses.rds", "icd10_codebook.rds")
manifest_path <- file.path(cache_dir, "manifest.rds")
if (dir.exists(cache_dir)) {
  if (!file.exists(manifest_path)) stop("Partial cache: directory exists without manifest")
  manifest <- readRDS(manifest_path)
  if (!identical(manifest$version, 1L) ||
      !identical(names(manifest$sha256), names_expected)) stop("Invalid cache manifest")
  for (name in names_expected) {
    path <- file.path(cache_dir, name)
    if (!file.exists(path) || isTRUE(nzchar(Sys.readlink(path))) ||
        !identical(sha256(path), unname(manifest$sha256[[name]])))
      stop("Missing or changed cache input: ", name)
  }
  message("Verified existing first-line loader cache: ", cache_dir)
  quit(save = "no", status = 0L)
}
parent <- dirname(cache_dir)
if (!dir.exists(parent)) stop("Create the approved parent directory first: ", parent)
if (!dir.create(cache_dir, mode = "0700", showWarnings = FALSE)) stop("Cannot create cache")
Sys.chmod(cache_dir, mode = "0700")
source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
load_dataset("patient")
load_dataset("t_dalycare_diagnoses")
load_dalycare_icd10()
if (!exists("patient") || !exists("t_dalycare_diagnoses") || !exists("Codes_all"))
  stop("Loader did not create the required first-line objects; cache is incomplete")
icd_names <- ls(envir = .GlobalEnv, pattern = "^ICD10[.]")
if (!length(icd_names)) stop("No ICD10.* definitions found; cache is incomplete")
codebook <- mget(c("Codes_all", icd_names), envir = .GlobalEnv, inherits = FALSE)
saveRDS(patient, file.path(cache_dir, names_expected[[1]]))
saveRDS(t_dalycare_diagnoses, file.path(cache_dir, names_expected[[2]]))
saveRDS(codebook, file.path(cache_dir, names_expected[[3]]))
paths <- file.path(cache_dir, names_expected)
hashes <- vapply(paths, sha256, character(1)); names(hashes) <- names_expected
manifest <- list(version = 1L, created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
                 loader_path = "/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R",
                 loader_sha256 = sha256("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R"),
                 sha256 = hashes,
                 rows = c(patient = nrow(patient), t_dalycare_diagnoses = nrow(t_dalycare_diagnoses),
                          Codes_all = nrow(Codes_all)),
                 icd10_objects = icd_names,
                 note = "INTERNAL. Record SQL database/table/view, extraction transaction or backup ID separately.")
saveRDS(manifest, manifest_path)
message("Created first-line loader cache: ", cache_dir)
