# ==============================================================================
# cohort_firstline.R
# First-line treatment date + cohort + treatment arm identification
#
# Outputs (written to environment):
#   firstline_tbl_mapped  — one row per patient, with cohort + treatment arm
#   df_expanded           — same, with one-hot treatment features
#
# Author: Mikkel Werling  2026-03-19
# ==============================================================================


# ------------------------------------------------------------------------------
# 0. Libraries
# ------------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(dplyr)
  library(lubridate)
  library(purrr)
  library(readr)
  library(stringr)
  library(tibble)
  library(tidyr)
})

# Fail before expensive loader calls if the direct inputs were not configured.
.firstline_direct_names <- c("OPERA_FIRSTLINE_CLL_SEED",
                             "OPERA_FIRSTLINE_MM_SEED",
                             "OPERA_FIRSTLINE_TREATMENT")
.firstline_missing_names <- .firstline_direct_names[
  !nzchar(Sys.getenv(.firstline_direct_names, ""))]
if (length(.firstline_missing_names))
  stop("Missing first-line input environment variable(s): ",
       paste(.firstline_missing_names, collapse = ", "))


# ------------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------------
# Aggregate-only instrumentation. No patient identifiers are written. Counts
# are INTERNAL audit outputs, not disclosure-cleared publication tables.
.fl_audit <- new.env(parent = emptyenv())
.fl_audit$counts <- list()
.fl_audit$breakdowns <- list()
.fl_audit$sequence <- 0L
.fl_audit$run_id <- paste0(format(Sys.time(), "%Y%m%dT%H%M%S"), "_", Sys.getpid())

fl_count <- function(stage, data, id_col = "patientid", branch = "production",
                     unit = "records", note = "") {
  if (!id_col %in% names(data)) {
    warning(sprintf("Count skipped: %s lacks %s", stage, id_col))
    return(invisible(NULL))
  }
  ids <- data[[id_col]]
  valid <- !is.na(ids)
  .fl_audit$sequence <- .fl_audit$sequence + 1L
  row <- data.frame(sequence = .fl_audit$sequence, branch = branch, stage = stage,
                    unit = unit, n_rows = nrow(data),
                    n_patients = length(unique(ids[valid])),
                    n_missing_id_rows = sum(!valid),
                    n_repeated_patient_rows = sum(valid) - length(unique(ids[valid])),
                    note = note, stringsAsFactors = FALSE)
  .fl_audit$counts[[length(.fl_audit$counts) + 1L]] <- row
  message(sprintf("[1L count] %s / %s: rows=%d patients=%d",
                  branch, stage, row$n_rows, row$n_patients))
  # Marginal summaries, not cross-products: categories may overlap and these
  # tables must NOT be interpreted as mutually exclusive attrition counts.
  for (column in intersect(c("source", "firstline_source", "cohort_source",
                             "cohort_grouped", "cohort_fine", "arm_primary",
                             "arm_exclude_primary", "reason", "split"), names(data))) {
    values <- as.character(data[[column]])
    values[is.na(values)] <- "[missing]"
    for (value in unique(values)) {
      sel <- values == value
      .fl_audit$breakdowns[[length(.fl_audit$breakdowns) + 1L]] <- data.frame(
        sequence = row$sequence, branch = branch, stage = stage,
        variable = column, category = value, n_rows = sum(sel),
        n_patients = length(unique(ids[sel & valid])), stringsAsFactors = FALSE)
    }
  }
  invisible(NULL)
}

fl_write_counts <- function() {
  output <- file.path(Sys.getenv("OPERA_FIRSTLINE_COUNT_DIR",
                                "firstline_pipeline_counts_LOCAL_ONLY"),
                      .fl_audit$run_id)
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(do.call(rbind, .fl_audit$counts),
                   file.path(output, "pipeline_counts.csv"), row.names = FALSE)
  if (length(.fl_audit$breakdowns)) utils::write.csv(
    do.call(rbind, .fl_audit$breakdowns),
    file.path(output, "pipeline_breakdowns.csv"), row.names = FALSE)
  writeLines(c("INTERNAL: exact aggregate counts; disclosure review required.",
               "Branches/units differ; do not subtract arbitrary adjacent rows.",
               "Repeated patient rows are expected for records/episodes, not patient tables.",
               paste0("TRUST_RKKP=", TRUST_RKKP),
               "Actual make_firstline settings are recorded in stage notes when instrumented.",
               "The later FROZEN objects are a separate branch, not this CSV export population.",
               "Split/follow-up/outcome eligibility are downstream, not invented here."),
             file.path(output, "README.txt"))
  message("[1L count] Wrote internal audit to ", output)
  invisible(output)
}

source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")

firstline_cache <- Sys.getenv("OPERA_FIRSTLINE_INPUT_CACHE", "")
if (nzchar(firstline_cache)) {
  cache_files <- c("patient.rds", "t_dalycare_diagnoses.rds", "icd10_codebook.rds")
  cache_manifest <- file.path(firstline_cache, "manifest.rds")
  if (!file.exists(cache_manifest)) stop("First-line cache missing manifest; refusing live SQL fallback")
  manifest <- readRDS(cache_manifest)
  hash_executable <- Sys.which("sha256sum")
  if (!nzchar(hash_executable) || !identical(manifest$version, 1L) ||
      !identical(names(manifest$sha256), cache_files)) stop("Invalid first-line cache manifest")
  loader_hash_line <- suppressWarnings(system2(
    hash_executable,
    shQuote("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R"),
    stdout = TRUE
  ))
  if (length(loader_hash_line) != 1L || !is.null(attr(loader_hash_line, "status")) ||
      !identical(substring(loader_hash_line, 1L, 64L), manifest$loader_sha256))
    stop("DALY-CARE loader changed since cache creation")
  for (cache_file in cache_files) {
    path <- file.path(firstline_cache, cache_file)
    if (!file.exists(path) || isTRUE(nzchar(Sys.readlink(path))))
      stop("Missing/linked cache file: ", path)
    hash_line <- suppressWarnings(system2(hash_executable, shQuote(path), stdout = TRUE))
    if (length(hash_line) != 1L || !is.null(attr(hash_line, "status")) ||
        !identical(substring(hash_line, 1L, 64L), unname(manifest$sha256[[cache_file]])))
      stop("First-line cache checksum mismatch: ", path)
  }
  patient <- readRDS(file.path(firstline_cache, "patient.rds"))
  t_dalycare_diagnoses <- readRDS(file.path(firstline_cache, "t_dalycare_diagnoses.rds"))
  codebook <- readRDS(file.path(firstline_cache, "icd10_codebook.rds"))
  list2env(codebook, envir = .GlobalEnv)
  message("Using frozen first-line loader inputs from ", firstline_cache)
} else {
  load_dataset("patient")
  load_dataset("t_dalycare_diagnoses")
  load_dalycare_icd10()
}
fl_count("loaded_patient_population", patient)
fl_count("loaded_diagnosis_records", t_dalycare_diagnoses)


## NOTE: We sometimes miss RKKP sources as the identification algorithm is conservative.
# We therefore include RKKP as a rescue option if so desired

TRUST_RKKP <- TRUE

RKKP_SOURCES <- c("LYFO", "DaMyDa", "DCLLR")

frozen_direct_names <- c("OPERA_FIRSTLINE_CLL_SEED", "OPERA_FIRSTLINE_MM_SEED",
                         "OPERA_FIRSTLINE_TREATMENT")
frozen_direct_paths <- Sys.getenv(frozen_direct_names, "")
if (!all(nzchar(frozen_direct_paths)))
  stop("Set OPERA_FIRSTLINE_CLL_SEED, OPERA_FIRSTLINE_MM_SEED and ",
       "OPERA_FIRSTLINE_TREATMENT explicitly; the V4 seed fallback is retired")
if (!all(file.exists(frozen_direct_paths))) stop("A direct first-line input is missing")
cll_seed_path <- frozen_direct_paths[[1]]
mm_seed_path <- frozen_direct_paths[[2]]
treatment_path <- frozen_direct_paths[[3]]
seed_policy <- Sys.getenv("OPERA_FIRSTLINE_SEED_POLICY", "primary_cll_v4_mm_v5")
if (!seed_policy %in% c("primary_cll_v4_mm_v5", "historical_v4"))
  stop("OPERA_FIRSTLINE_SEED_POLICY must be primary_cll_v4_mm_v5 or historical_v4")
if (seed_policy == "primary_cll_v4_mm_v5") {
  if (basename(cll_seed_path) != "CLL_V4_100lines.parquet" ||
      basename(mm_seed_path) != "MM_wide_V5_100lines.parquet")
    stop("Primary seeds must be final CLL_V4_100lines.parquet and MM_wide_V5_100lines.parquet")
  if (basename(treatment_path) != "table_treatment_clean_2026_09_10.parquet")
    stop("Primary first-line treatment must be table_treatment_clean_2026_09_10.parquet")
} else if (!grepl("V4", basename(cll_seed_path), fixed = TRUE) ||
           !grepl("V4", basename(mm_seed_path), fixed = TRUE)) {
  stop("Historical comparison requires explicitly selected CLL V4 and MM V4 files")
}
message("[1L seed] policy=", seed_policy, "; CLL: ", normalizePath(cll_seed_path),
        "; MM: ", normalizePath(mm_seed_path))
seed_CLL <- read_parquet(cll_seed_path)
seed_MM <- read_parquet(mm_seed_path)
treatments <- read_parquet(treatment_path) %>%
  collect() %>%
  { fl_count("treatment_records_before_plan_exclusion", .); . } %>%
  filter(source != "SP beh") # %>% 
  # filter(source %in% RKKP_SOURCES)# remove treatment plans — keep only administered

# --- Clinical first-line seed (CLL + MM), line 1 only ---
fl_count("treatment_records_after_plan_exclusion", treatments)
prep_seed <- function(df, disease) {
  # V4 line tables store date_start_line once per line. V5 stores treatment
  # rows with date_start; for line 1, use the earliest row per patient. The
  # row_acc flag is deliberately not used without an agreed definition.
  start_field <- if ("date_start_line" %in% names(df)) "date_start_line" else
    if ("date_start" %in% names(df)) "date_start" else
      stop("Seed table lacks date_start_line and date_start: ", disease)
  if (!all(c("patientid", "line") %in% names(df)))
    stop("Seed table lacks patientid or line: ", disease)
  message("[1L seed] ", disease, " date field: ", start_field)
  df %>%
    filter(line == 1) %>%
    transmute(
      patient_id   = patientid,
      seed_date    = as.Date(.data[[start_field]]),
      seed_regimen = if ("regimen" %in% names(df)) as.character(regimen) else NA_character_,
      seed_disease = disease
    ) %>%
    filter(!is.na(patient_id), !is.na(seed_date)) %>%
    arrange(patient_id, seed_date) %>%
    distinct(patient_id, .keep_all = TRUE)
}

seed_firstline <- bind_rows(
  prep_seed(seed_CLL, "CLL"),
  prep_seed(seed_MM,  "MM")
) %>%
  arrange(patient_id, seed_date) %>%
  distinct(patient_id, .keep_all = TRUE)   # if in both tables, keep earliest line-1 date

# Upgrade RKKP priority to 1 (best source)
t_dalycare_diagnoses <- t_dalycare_diagnoses %>%
  mutate(priority = if_else(datasource == "RKKP", 1L, as.integer(priority)))


# ==============================================================================
# 2. Code definitions
# ==============================================================================

# ------------------------------------------------------------------------------
# 2a. Radiotherapy SKS codes
# ------------------------------------------------------------------------------
RT_excluded <- c("BWGA", "D", "AFB", "ZZ0153A4")
RT_disease  <- c(
  "AFB04F3",   # MM with RT
  "AFB03F3"    # Lymphoma and CLL with RT
)

SKS.radiotherapy <- Codes_all %>%
  filter(type == "sks") %>%
  filter_str_detect(text, "strålebehandling") %>%
  filter_str_detect(code, paste0("^", RT_excluded), negate = TRUE) %>%
  bind_rows(Codes_all %>% filter(code %in% RT_disease)) %>%
  pull(code) %>%
  sort()

# ------------------------------------------------------------------------------
# 2b. CLL-specific codes
# ------------------------------------------------------------------------------
ATC.CLL <- c(
  "L01BB05",          # fludarabine
  "L01AA01",          # cyclophosphamide
  "L01AA09",          # bendamustine
  "L01AA02",          # chlorambucil
  "L01FA03", "L01XC15", # obinutuzumab
  "L01EL01", "L01XE27", # ibrutinib
  "L01EL03",          # zanubrutinib
  "L01EL02",          # acalabrutinib
  "L01EM01", "L01XX47", # idelalisib
  "L01XX52",          # venetoclax
  "L01XC04", "L04AA34"  # alemtuzumab
)

SKS.CLL <- c(
  "BWHA105",          # cyclophosphamide
  "BWHA171",          # chlorambucil
  "BWHA172",          # fludarabine iv
  "BWHA173",          # fludarabine po
  "BWHA174",          # fludarabine + cyclo iv
  "BWHA175",          # fludarabine + cyclo po
  "BWHA177",          # bendamustine
  "BWHA427",          # ibrutinib
  "BWHA428",          # idelalisib
  "BWHA438",          # venetoclax
  "BWHP114",          # zanubrutinib
  "BOHJ19H5",         # obinutuzumab
  "BOHJ16A",          # alemtuzumab
  "BWHA119"           # chop (only when dx code matches)
)

CLL_CODES <- c(ATC.CLL, SKS.CLL)

# ------------------------------------------------------------------------------
# 2c. Aggressive B-cell / DLBCL-relevant codes
# NOTE: deliberately excludes rituximab (L01FA01/L01XC02), doxorubicin (L01DB01),
# etoposide (L01CB01), cisplatin (L01XA01), oxaliplatin (L01XA03) — these are
# either prophylaxis-suspect or too broadly used; handled via PROPHY_SUSPECT_CODES
# ------------------------------------------------------------------------------
ATC.DLBCL <- c(
  "L01FA03", "L01XC15", # obinutuzumab
  "L01DB07",            # mitoxantrone
  "L01AA01",            # cyclophosphamide
  "L01AA06",            # ifosfamide
  "L01XA02",            # carboplatin
  "L01BC05",            # gemcitabine
  "L01CA02",            # vincristine
  "L01AA03",            # melphalan
  "L01AD02",            # lomustine
  "L01CA01",            # vinblastine
  "L01AA02",            # chlorambucil
  "L01BB05",            # fludarabine
  "L01AA09",            # bendamustine
  "L01DB11",            # pixantrone
  "L01AX03",            # temozolomide
  "L01BA01", "L04AX03", # methotrexate
  "L01BC01",            # cytarabine
  "L01XG01", "L01XX32", # bortezomib
  "L01AC01",            # thiotepa
  "L01AD01",            # carmustine
  "L04AX04"             # lenalidomide
)

SKS.DLBCL <- c(
  "BWHA104",            # mitoxantrone
  "BWHA105",            # cyclophosphamide
  "BWHA106",            # ifosfamide
  "BWHA109",            # carboplatin
  "BWHA162",            # cytarabine
  "BWHA161",            # triple CNS (cytarabine + mtx + prednisolon)
  "BWHA112",            # carboplatin + etoposide
  "BWHA114",            # gemcitabine
  "BWHA119",            # chop
  "BWHA120",            # mime
  "BWHA128",            # cisplatin + gemcitabine
  "BWHA134",            # cop
  "BWHA153",            # vincristine
  "BWHA154",            # melphalan
  "BWHA155",            # lomustine
  "BWHA164",            # cope
  "BWHA165",            # choep
  "BWHA169",            # vinblastine
  "BWHA171",            # chlorambucil
  "BWHA172", "BWHA173", # fludarabine
  "BWHA174", "BWHA175", # fludarabine + cyclophosphamide
  "BWHA177",            # bendamustine
  "BWHA189",            # pixantrone
  "BWHA215",            # temozolomide
  "BWHA219",            # high-dose methotrexate
  "BWHA258",            # maxi-chop
  "BWHA301",            # high-dose cytarabine
  "BWHA307",            # codox m
  "BWHA402",            # bortezomib
  "BWHA451",            # thiotepa
  "BWHA452",            # carmustine
  "BWHB82"              # lenalidomide
)


tx_MM <- c(
  'asct', #
  'alnuctamab', #0
  'basic cytostatic treatment', #
  'bendamustine', #
  'belantamab-mafodotin', #0
  'bi-speficic antibody (t-cell engaging)', #
  'bortezomib', #
  'carfilzomib', #
  'cevostamab',  #0
  'ciltacabtagene autoleucel', #0
  'cisplatin', #
  'complex cytostatic treatment', #
  'asct (conditioning)', #
  'cyclophosphamide', #
  'cyclophosphamide|dexametason (cy|dex)',
  'cytostatic_mm',
  'daratumumab', #
  'doxorubicin', #
  'elotuzumab', #
  'elranatamab', #
  'etoposide', #
  'forimtamig', #0
  'idecabtagene vicleucel', #0 
  'isatuximab',
  'ixazomib', #
  'lenalidomide', #
  'linvoseltamab', #0
  'lomustine', #
  'melphalan', #
  'high-dose melphalan',
  'mezigdomide', #0
  'monoclonal antibodies and antibody drug conjugates', #
  'other antineoplastic agents', #
  'other cytostatic treatment', #
  'other_mm', #
  'panobinostat', #
  'pomalidomide', #
  'selinexor',
  'talquetamab', #
  'teclistamab', #
  'thalidomide', #
  'venetoxclax', #
  "venetoclax"
)

additional_codes <- c("BWHA135", "vcr")


DLBCL_CODES <- c(ATC.DLBCL, SKS.DLBCL, tx_MM, additional_codes)

# Anchor codes: definitive hematological treatment codes used to confirm that an
# episode is a real treatment (vs prophylaxis or supportive care)
# Prophylaxis-suspect codes are tracked separately to require stronger evidence
ANCHOR_CODES <- unique(c(DLBCL_CODES, CLL_CODES, SKS.radiotherapy))

PROPHY_SUSPECT_CODES <- c(
  "etoposide", "doxorubicin", "cisplatin", "selinexor",
  "BWHA102",        # doxorubicin
  "BWHA107",        # cisplatin
  "BWHA108",        # oxaliplatin
  "BWHA111",        # etoposide
  "L01FA01", "L01XC02",  # rituximab (very broad — also prophylaxis/maintenance)
  "cd20 antibody", # cd 20 antibody
  "BOHJ11",         # cd20 antibody
  "BWHA1", "BWHA2", "BWHA4",  # generic cytostatic codes
  "basic cytostatic treatment",
  "methotrexate", "L04AX03",
  "gemcitabine",
  "bevacizumab",
  "L01BA01"
)

# Remove prophylaxis-suspect codes from anchor set
ANCHOR_CODES <- ANCHOR_CODES[!ANCHOR_CODES %in% PROPHY_SUSPECT_CODES]


# ==============================================================================
# 3. Treatment exclusion definitions
# ==============================================================================

# ------------------------------------------------------------------------------
# 3a. Hematological ICD-10 allowlist (Danish coding: C → DC, e.g. C83.3 → DC833)
# Treatments with a dx_code that MATCHES this list are kept unconditionally.
# Treatments with a dx_code that does NOT match this list (and is not heme) are
# excluded even if text looks clean — this is the main strictness upgrade.
# ------------------------------------------------------------------------------
HEME_ICD10_PREFIXES <- c(
  paste0("DC8", 1:9),         # DC81–DC89: lymphomas, plasma cell neoplasms
  paste0("DC9", c(0:6)),      # DC90–DC96: leukaemias, myeloma, unspecified
  "DD47"                      # borderline haematological (MGUS, MBL, etc.)
)

# ------------------------------------------------------------------------------
# 3b. Non-hematological ICD-10 exclusion prefixes
# Treatments with a 4-character dx_code prefix in this list are excluded.
# NOTE: must use str_sub(dx_code, 1, 4) — 4-char Danish codes like "DC34"
# ------------------------------------------------------------------------------
EXCL_ICD10_PREFIX <- c(
  "DC18", "DC19", "DC20",  # colorectal
  "DC25",                  # pancreas
  "DC34",                  # lung
  "DC50",                  # breast
  "DC56",                  # ovary
  "DC61",                  # prostate
  "DC71",                  # brain
  "DC78",                   # secondary / metastases
  "DC43"                   # melanoma 
)

# ------------------------------------------------------------------------------
# 3c. ATC code–level text exclusion (drugs never used for lymphoid malignancies)
# ------------------------------------------------------------------------------
EXCLUDE_ATC_TEXT <- c(
  "fluorouracil", "5-fluorouracil", "oxaliplatin", "capecitabine", "docetaxel",
  "streptozocin", "irinotecan", "mitomycin", "abemaciclib", "palbociclib",
  "axitinib", "lenvatinib", "sorafenib", "bevacizumab", "paclitaxel",
  "durvalumab", "olaparib", "binimetinib", "encorafenib", "brigatinib",
  "cabazitaxel", "vinorelbine", "pemetrexed", "cemiplimab", "epirubicin",
  "imatinib", "dasatinib", "nilotinib", "osimertinib", "ribociclib",
  "dabrafenib | trametinib", "atezolizumab", "nintedanib", "ruxolitinib",
  "azacitidine", "anagrelide", "hydroxurea", "hydroxycarbamide",
  "hydroxurea | hydroxycarbamide", "cisplatin", "carboplatin", "dacerbazine"
)

# ------------------------------------------------------------------------------
# 3d. Danish diagnosis text patterns for non-hematological indications
# Applied only when dx_code is absent (fallback tier)
# ------------------------------------------------------------------------------
EXCL_TEXT_PATTERNS <- c(
  "kræft i lunge", "lungekræft",
  "kræft i tyktarm", "tyktarmskræft", "colon", "kolon", "rektum",
  "brystkræft", "pancreas", "bugspytkirtel",
  "æggestok", "ovarie", "prostata", "hjerne",
  "fjernmetastase", "metastase", "levermetastase", "fjernspredning",
  "acne", "alkohol", "endetarmen", "nyrekræft", "livmoderkræft",
  "kræft i skjoldbruskkirtlen", "kræft i spiserøret",
  "kræft i tolvfingertarmen", "psoriasis vulgaris", "ledsmerter",
  "psoriasisartritis uns", "kræft i caecum", "dermatitis",
  "kræft i bindevæv", "kræft i caput pancreatis",
  "kræft i galdeveje uns", "kræft i nyrebækken",
  "kræft i retroperitoneum", "kræft i strubehovedet uns",
  "kræft i urinblæren uns", "kræft i æggeleder",
  "livmoderhalskræft", "psoriasis", "sygdom i hud eller underhud",
  "malignt melanom i huden uns", "reumatoid artritis",
  "kræft i mavesækken uns",
  "kræft i bronkier og lunge overgribende flere lokalisationer",
  "kræft i cardia", "anden hudkræft uns",
  "kræft i endetarmsåbningen uns", "kræft i galdeblæren",
  "kræft i flexura coli sinistra", "kræft i intrahepatiske galdegange",
  "kræft i næsehulen", "kræft i thymus", "kræft i urinleder",
  "anden hudkræft",
  "melanoma", "hudkræft"
)

# Pre-compile the text exclusion pattern once (cheaper repeated matching)
EXCL_TEXT_PATTERN_RE <- str_c(EXCL_TEXT_PATTERNS, collapse = "|")
EXCL_ATC_TEXT_RE     <- str_c(EXCLUDE_ATC_TEXT,      collapse = "|")

norm_da <- function(x) {
  x %>%
    tolower() %>%
    str_replace_all("[[:punct:]]", " ") %>%
    str_squish()
}


# ==============================================================================
# 4. Treatment flagging — 3-tier positive-first filter
#
# Tier 1 (strictest, highest priority):
#   If dx_code is present AND starts with a hematological ICD prefix → KEEP
#   If dx_code is present AND starts with a non-heme exclusion prefix  → EXCLUDE
#
# Tier 2 (intermediate, applied when dx_code is present but neither heme nor
#   explicitly non-heme):
#   Free-text indication contains a known non-heme Danish term → EXCLUDE
#
# Tier 3 (fallback, dx_code absent):
#   Free-text exclusion OR ATC/code-text exclusion → EXCLUDE; else KEEP (uncertain)
#
# This replaces the previous denylist-only approach, which also had a silent bug:
# str_sub(dx_code, 1, 3) was used but EXCL_ICD10_PREFIX contains 4-char codes,
# so the ICD exclusion never fired.
# ==============================================================================

tx_raw <- treatments %>%
  mutate(code = if_else(is.na(tx_code), tx_text, tx_code)) %>%
  select(
    patient_id = patientid,
    tx_date    = date_start,
    code,
    code_text  = tx_text,
    dx_text,
    dx_code,
    source
  )

tx_flagged <- tx_raw %>%
  mutate(
    tx_date       = as.Date(tx_date),
    dx_code       = as.character(dx_code),
    dx_text_norm  = norm_da(dx_text),
    # Extract 4-character prefix for ICD matching (e.g. "DC833" → "DC83")
    dx_prefix4    = str_sub(dx_code, 1, 4),
    
    # Was this from RKKP?
    is_rkkp = source %in% RKKP_SOURCES
  ) %>%
  mutate(
    # Tier 1a: positive heme confirmation from ICD
    ind_heme_icd  = !is.na(dx_code) & dx_prefix4 %in% HEME_ICD10_PREFIXES,
    
    # Tier 1b: confirmed non-heme from ICD (was silently broken before — fixed)
    ind_excl_icd  = !is.na(dx_code) & dx_prefix4 %in% EXCL_ICD10_PREFIX,
    
    # Tier 2: non-heme text signal on indication
    ind_excl_text = !is.na(dx_text_norm) &
      str_detect(dx_text_norm, EXCL_TEXT_PATTERN_RE),
    
    # Tier 3: ATC / code-text signal (drug is clearly non-heme)
    ind_excl_code = (!is.na(code_text) & str_detect(tolower(code_text), EXCL_ATC_TEXT_RE)) |
      tolower(code) %in% tolower(EXCLUDE_ATC_TEXT),
    
    # 3-tier decision: heme-confirmed > ICD-excluded > text/code signal > keep
    ind_ok = case_when(
      is_rkkp & TRUST_RKKP            ~ TRUE,
      ind_heme_icd                    ~ TRUE,   # confirmed heme
      ind_excl_icd                    ~ FALSE,  # confirmed non-heme by ICD
      ind_excl_text | ind_excl_code   ~ FALSE,  # excluded by signal
      TRUE                            ~ TRUE    # uncertain → keep (flagged below)
    ),
    
    ind_missing = is.na(dx_code) & is.na(dx_text)  # no indication info at all
  )

cat(sprintf(
  "\nTreatment flag summary:\n  Total rows   : %d\n  ind_ok=TRUE  : %d (%.1f%%)\n  ind_ok=FALSE : %d (%.1f%%)\n  heme_icd     : %d  excl_icd: %d  excl_text: %d  excl_code: %d\n",
  nrow(tx_flagged),
  sum(tx_flagged$ind_ok),  mean(tx_flagged$ind_ok) * 100,
  sum(!tx_flagged$ind_ok), mean(!tx_flagged$ind_ok) * 100,
  sum(tx_flagged$ind_heme_icd), sum(tx_flagged$ind_excl_icd),
  sum(tx_flagged$ind_excl_text), sum(tx_flagged$ind_excl_code)
))


### CHECK SOME SAMPLED ROWS:

samples_for_exclusion = tx_flagged %>% 
  filter(!ind_ok) %>% 
  sample_n(100)

check <- samples_for_exclusion %>%
  filter(ind_excl_icd) %>%
  group_by(dx_text) %>%
  summarize(n = n())


check <- samples_for_exclusion %>%
  filter(ind_excl_code) %>%
  group_by(code_text) %>%
  summarize(n = n())


check <- samples_for_exclusion %>%
  filter(ind_excl_text) %>%
  group_by(dx_code) %>%
  summarize(n = n())
  

### get numbers for consort diagram

tx_check <- tx_flagged %>%
  mutate(
    tx_date   = as.Date(tx_date),
    code      = as.character(code),
    code_text = if ("code_text" %in% names(.)) as.character(code_text) else NA_character_
  ) %>%
  filter(!is.na(patient_id), !is.na(tx_date), !is.na(code), ind_ok)

n_unique(tx_check$patient_id)

# ==============================================================================
# 5. First-line treatment identification
#
# Logic:
#   1. Restrict treatments to those occurring after first diagnosis (per patient)
#   2. Build treatment episodes by collapsing rows within `episode_gap_days`
#   3. Classify each episode as "therapy-like" using anchor codes or multi-date
#      evidence (requires stronger evidence if only prophylaxis-suspect codes)
#   4. Return the earliest qualifying episode per patient as first-line
#
# NOTE: `tx` must contain an `ind_ok` column (produced by tx_flagged above).
# ==============================================================================

make_firstline <- function(tx,
                           dx,
                           seed_tbl = NULL,
                           seed_max_gap = 30,
                           pre_dx_grace_days = 14, 
                           accept_cd20_mono = FALSE,
                           max_days_from_dx  = 365 * 10,
                           episode_gap_days  = 21,
                           min_episode_dates = 1,
                           require_therapy_like = TRUE,
                           audit_label = NULL) {
  
  stopifnot("ind_ok" %in% names(tx),
            "patient_id" %in% names(tx),
            "tx_date"    %in% names(tx),
            "code"       %in% names(tx),
            "is_rkkp"    %in% names(tx))
  
  tx2 <- tx %>%
    mutate(
      tx_date   = as.Date(tx_date),
      code      = as.character(code),
      code_text = if ("code_text" %in% names(.)) as.character(code_text) else NA_character_
    ) %>%
    filter(!is.na(patient_id), !is.na(tx_date), !is.na(code), ind_ok)
  if (!is.null(audit_label)) fl_count("indication_eligible_treatment_rows", tx2,
                                     "patient_id", audit_label,
                                     note = sprintf("grace=%s; max_days=%s; episode_gap=%s; seed_gap=%s; CD20_mono=%s; therapy_gate=%s",
                                                    pre_dx_grace_days, max_days_from_dx,
                                                    episode_gap_days, seed_max_gap,
                                                    accept_cd20_mono, require_therapy_like))
  
  dx2 <- dx %>%
    mutate(dx_date = as.Date(dx_date)) %>%
    select(patient_id, dx_date, diagnosis) %>%
    distinct()
  
  # Join and restrict to window after diagnosis
  t <- tx2 %>%
    inner_join(dx2, by = "patient_id") %>%
    filter(
      tx_date >= dx_date - pre_dx_grace_days,
      tx_date <= dx_date + max_days_from_dx
    ) %>%
    arrange(patient_id, tx_date, code)
  if (!is.null(audit_label)) fl_count("treatment_rows_in_diagnosis_window", t,
                                     "patient_id", audit_label)
  
  # Build episodes by grouping consecutive treatment dates within gap threshold
  t_ep <- t %>%
    group_by(patient_id) %>%
    mutate(
      gap_days    = as.integer(tx_date - lag(tx_date)),
      new_episode = if_else(is.na(gap_days) | gap_days > episode_gap_days, 1L, 0L),
      episode_id  = cumsum(new_episode)
    ) %>%
    ungroup()
  
  # Episode-level summaries
  ep <- t_ep %>%
    group_by(patient_id, episode_id) %>%
    summarise(
      dx_date               = first(dx_date),
      # indication text from the treatment records (not the registry diagnosis)
      indication_text       = paste(sort(unique(dx_text[!is.na(dx_text)])),   collapse = " | "),
      indication_codes      = paste(sort(unique(dx_code[!is.na(dx_code)])),   collapse = " | "),
      episode_start         = min(tx_date),
      episode_end           = max(tx_date),
      n_tx_rows             = n(),
      n_tx_dates            = n_distinct(tx_date),
      n_codes               = n_distinct(code),
      has_anchor            = any(code %in% ANCHOR_CODES),
      has_prophy_suspect    = any(code %in% PROPHY_SUSPECT_CODES),
      has_cd20_suspect = any(code %in% c("L01FA01", "L01XC02", "BOHJ11")) | any(str_detect(tolower(code_text), "rituximab|ofatumumab|cd20 antibody")),
      ## RKKP trigger - we say that RKKP is as good as an anchor
      has_rkkp_source = any(is_rkkp),
      # Use ICD10 codes as confirmation 
      has_heme_indication = any(ind_heme_icd),
      pct_heme_indication = mean(ind_heme_icd),
      codes_sorted          = paste(sort(unique(code)),                        collapse = " | "),
      codes_text_sorted     = paste(sort(unique(code_text[!is.na(code_text)])), collapse = " | "),
      dates_sorted          = paste(sort(unique(tx_date)),                     collapse = ", "),
      .groups = "drop"
    ) %>%
    mutate(
      episode_duration_days = as.integer(episode_end - episode_start),
      therapy_like =
        has_anchor |
        has_rkkp_source | 
        (n_tx_dates >= min_episode_dates & n_codes >= 2) | 
        (n_tx_dates >= 2)
    )
  
  # Strengthen evidence requirement when only prophylaxis-suspect codes are present
  ep <- ep %>%
    mutate(
      therapy_like = case_when(
        has_anchor ~ TRUE,
        TRUST_RKKP & has_rkkp_source & !(has_prophy_suspect & n_codes < 2) ~ TRUE,
        accept_cd20_mono & has_cd20_suspect & n_tx_dates >= 2 ~ TRUE,
        #has_heme_indication ~ TRUE,
        has_prophy_suspect ~ (n_tx_dates >= 3 & n_codes >= 2), # should probably be OR instead 
        TRUE ~ (n_tx_dates >= 2 | n_codes >= 2)
      )
    )
  
  # --- SEED: attach clinical first-line anchor (CLL/MM) if provided ---
  if (!is.null(seed_tbl)) {
    ep <- ep %>%
      left_join(seed_tbl %>% select(patient_id, seed_date), by = "patient_id") %>%
      mutate(
        # 0 if seed falls inside [start,end]; else distance to nearest edge
        seed_gap = if_else(
          is.na(seed_date), NA_real_,
          pmax(0,
               as.numeric(episode_start - seed_date),
               as.numeric(seed_date     - episode_end))
        ),
        is_seed_episode = !is.na(seed_gap) & seed_gap <= seed_max_gap
      )
  } else {
    ep <- ep %>% mutate(seed_date = as.Date(NA), seed_gap = NA_real_, is_seed_episode = FALSE)
  }
  if (!is.null(audit_label)) fl_count("episodes_before_therapy_gate", ep,
                                     "patient_id", audit_label, "episodes")
  
  # Keep therapy-like episodes, but never drop a clinically-seeded episode
  if (require_therapy_like) ep <- ep %>% filter(therapy_like | is_seed_episode)
  if (!is.null(audit_label)) fl_count("episodes_after_therapy_or_seed_gate", ep,
                                     "patient_id", audit_label, "episodes")
  
  # Selection: seeded patients -> nearest-to-seed episode; others -> earliest
  firstline <- ep %>%
    mutate(.seed_first = if_else(is_seed_episode, 0L, 1L)) %>%
    group_by(patient_id) %>%
    arrange(patient_id, .seed_first, seed_gap, episode_start, .by_group = TRUE) %>%
    slice(1) %>%
    ungroup() %>%
    transmute(
      patient_id,
      firstline_date               = episode_start,   # pipeline-native anchor
      seed_date,                                       # clinical date, QC only
      firstline_source             = if_else(is_seed_episode, "clinical_seed", "inferred"),
      firstline_seed_gap           = seed_gap,
      firstline_end                = episode_end,
      firstline_episode_id         = episode_id,
      firstline_codes              = codes_sorted,
      firstline_codes_text         = codes_text_sorted,
      firstline_n_dates            = n_tx_dates,
      firstline_n_codes            = n_codes,
      firstline_has_anchor         = has_anchor,
      firstline_has_rkkp_source    = has_rkkp_source,
      firstline_has_prophy_suspect = has_prophy_suspect,
      firstline_therapy_like       = therapy_like,
      firstline_has_heme_indication = has_heme_indication,
      firstline_indication         = indication_text,
      firstline_indication_codes   = indication_codes
    )
  
  # Left-join so patients with no qualifying episode are still in output
  dx2 %>%
    left_join(firstline, by = "patient_id")
}

# First diagnosis per patient (used as the index date for episode windowing)
dx_index <- t_dalycare_diagnoses %>%
  arrange(date_diagnosis) %>%
  group_by(patientid) %>%
  slice(1) %>%
  ungroup() %>%
  select(
    patient_id = patientid,
    dx_date    = date_diagnosis,
    diagnosis
  )

firstline_tbl <- make_firstline(tx_flagged, dx_index, seed_firstline, pre_dx_grace_days = 30,
                                seed_max_gap = 30, accept_cd20_mono = TRUE,
                                audit_label = "pass1")
fl_count("diagnosis_index_population", dx_index, "patient_id")
fl_count("clinical_firstline_seeds", seed_firstline, "patient_id")
fl_count("pass1_patients_with_index", firstline_tbl %>% filter(!is.na(firstline_date)),
         "patient_id", unit = "patients")
fl_count("pass1_patients_without_index", firstline_tbl %>% filter(is.na(firstline_date)),
         "patient_id", unit = "patients")

# Quick sanity check
cat("\n=== First-line coverage ===\n")
firstline_tbl %>%
  summarise(
    n_patients          = n_distinct(patient_id),
    n_with_firstline    = sum(!is.na(firstline_date)),
    n_without_firstline = sum( is.na(firstline_date)),
    pct_coverage        = mean(!is.na(firstline_date))
  ) %>%
  print()


# ==============================================================================
# 6. Cohort assignment
# ==============================================================================

# Disease group ICD-10 patterns (from load_dalycare_icd10() + supplementary)
DX_groups <- list(
  "AITL"          = ICD10.AITL,
  "ALCL"          = ICD10.ALCL,
  "AMYLOIDOSIS"   = ICD10.AMYLOIDOSIS,
  "BCL"           = ICD10.BCL,
  "BCL_UNS"       = ICD10.BCL_UNS,
  "BL"            = ICD10.BL,
  "CLL"           = ICD10.CLL,
  "DALYCARE"      = ICD10.DALYCARE,
  "DLBCL"         = ICD10.DLBCL,
  "EMZL"          = ICD10.EMZL,
  "FL"            = ICD10.FL,
  "HCL"           = ICD10.HCL,
  "HL"            = ICD10.HL,
  "LBL"           = ICD10.LBL,
  "Leukemia_UNS"  = ICD10.Leukemia_UNS,
  "LPL"           = ICD10.LPL,
  "Lymphoma_UNS"  = ICD10.Lymphoma_UNS,
  "MBL"           = ICD10.MBL,
  "MCL"           = ICD10.MCL,
  "MGUS"          = ICD10.MGUS,
  "MM"            = ICD10.MM,
  "MZL"           = ICD10.MZL,
  "NMZL"          = ICD10.NMZL,
  "PCD"           = ICD10.PCD,
  "PCL"           = ICD10.PCL,
  "PTCL"          = ICD10.PTCL,
  "RT"            = ICD10.RT,
  "SLL"           = ICD10.SLL,
  "SMZL"          = ICD10.SMZL,
  "SolM"          = ICD10.SolM,
  "TCL"           = ICD10.TCL,
  "UNS"           = ICD10.UNS
)

# Supplementary ICD-10 patterns (fills gaps not covered by load_dalycare_icd10)
ICD10_SUPPLEMENTARY <- list(
  "HL"          = c("DC810","DC811","DC812","DC813","DC814","DC819"),
  "FL"          = c("DC820","DC821","DC822","DC823","DC824","DC828","DC829"),
  "SLL"         = c("DC830"),
  "MCL"         = c("DC831"),
  "DLBCL"       = c("DC833","DC836","DC852"),
  "LBL"         = c("DC835","DC857"), # CONSIDER THROWING INTO UNS GROUP FOR SECOND
  "BL"          = c("DC837"),
  "BCL_UNS"     = c("DC838","DC839","DC851","DC882","DC883","DC888","DC889"),
  "TCL"         = c("DC840","DC841","DC842","DC847","DC848","DC849",
                    "DC860","DC861","DC862","DC863","DC864","DC915"),
  "PTCL"        = c("DC844","DC844H"),
  "AITL"        = c("DC845","DC865"),
  "ALCL"        = c("DC846","DC866","DC866A","DC866B"),
  "Lymphoma_UNS"= c("DC859"),
  "LPL"         = c("DC880"),
  "MGUS"        = c("DD472"),
  "LBL"         = c("DC910"),
  "CLL"         = c("DC911"),
  "HCL"         = c("DC914","DC916"),
  "Leukemia_UNS"= c("DC912","DC913","DC917","DC917B","DC919","DC91",
                    "DC950","DC951","DC959"),
  "EXCLUDE_SECONDARY" = c("DD472A")
)
ICD10_SUPPLEMENTARY <- lapply(ICD10_SUPPLEMENTARY, unique)

merge_icd10_supplementary <- function(DX_groups, supplementary) {
  for (grp in names(supplementary)) {
    if (grp %in% names(DX_groups)) {
      DX_groups[[grp]] <- unique(c(DX_groups[[grp]], supplementary[[grp]]))
    } else {
      DX_groups[[grp]] <- supplementary[[grp]]
    }
  }
  ### NOTE: Fix some non-exact matches that were distorting counts 
  
  # SLL: exact match only — prevents DC830B/C/D absorption
  DX_groups[["SLL"]] <- "^DC830$"
  
  # LPL: exact match for DC830B, keep DC880
  DX_groups[["LPL"]] <- c("^DC830B$", "DC880")
  
  # NMZL: exact match
  DX_groups[["NMZL"]] <- "^DC830C$"
  
  # SMZL: exact match
  DX_groups[["SMZL"]] <- "^DC830D$"
  
  # MZL: remove DC884x (those are EMZL) — keep only the generic unspecified codes
  # After removing DC830C/D (now exclusively in NMZL/SMZL), MZL becomes a
  # true catch-all for unsubtyped marginal zone lymphoma
  DX_groups[["MZL"]] <- character(0) # we want to have the distinct entities isolated - we can always figure it out afterwards
  # NOTE: if you want MZL to be a pure fallback that only fires when NMZL/SMZL
  # don't match, you can leave MZL empty here and rely on NMZL + SMZL directly.
  # The aggr_rank system would resolve ties but with identical ranks it's safer
  # to simply not duplicate. Consider removing MZL from DX_groups entirely and
  # mapping cohort_grouped from NMZL + SMZL directly.
  
  DX_groups
}

DX_groups <- merge_icd10_supplementary(DX_groups, ICD10_SUPPLEMENTARY)

# --- Group metadata ---
make_group_meta <- function(DX_groups) {
  groups      <- names(DX_groups)
  supergroups <- c("BCL","BCL_UNS","DALYCARE","HL","Leukemia_UNS","Lymphoma_UNS","PCD","TCL","UNS")
  exclude_primary <- c("DALYCARE","UNS","BCL_UNS","Lymphoma_UNS","Leukemia_UNS", "MGUS", "MBL") # NOTE: MGUS and MBL are by definition not getting treatment
  aggressive_groups <- c("RT","DLBCL","BL","PTCL","ALCL","AITL","LBL","PCL")
  
  specificity <- rep(5L, length(groups)); names(specificity) <- groups
  specificity[intersect(groups, supergroups)] <- 2L
  specificity[intersect(groups, c("UNS","DALYCARE","BCL_UNS","Lymphoma_UNS","Leukemia_UNS"))] <- 1L
  
  aggr_rank <- rep(99L, length(groups)); names(aggr_rank) <- groups
  aggr_rank[intersect(groups, c("RT"))]                            <- 1L
  aggr_rank[intersect(groups, c("DLBCL","BL","PTCL","ALCL","AITL","LBL","PCL", "MCL"))] <- 2L
  aggr_rank[intersect(groups, c("MM","SolM"))]               <- 3L
  aggr_rank[intersect(groups, c("CLL","SLL","LPL"))]               <- 4L
  aggr_rank[intersect(groups, c("FL","MZL","NMZL","SMZL","EMZL","HCL"))] <- 5L
  aggr_rank[intersect(groups, c("MBL","MGUS"))]                    <- 9L
  aggr_rank[intersect(groups, c("BCL","TCL","HL","PCD"))]           <- 10L
  aggr_rank[intersect(groups, c("BCL_UNS","Lymphoma_UNS","Leukemia_UNS","UNS","DALYCARE"))] <- 50L
  
  meta <- data.table(
    dx_group        = groups,
    is_super        = groups %chin% supergroups,
    include_primary = !(groups %chin% exclude_primary),
    aggressive      = groups %chin% aggressive_groups,
    specificity     = as.integer(specificity[groups]),
    aggr_rank       = as.integer(aggr_rank[groups])
  )
  # Derived transformation groups (Richter, transformed FL)
  meta <- rbind(meta, data.table(
    dx_group        = c("RT_DERIVED","TRANSFORMED_FL"),
    is_super        = FALSE,
    include_primary = TRUE,
    aggressive      = TRUE,
    specificity     = 6L,
    aggr_rank       = c(1L, 2L)
  ), fill = TRUE)
  
  coarse_map <- data.table(
    dx_group = c(
      "DLBCL","BL","FL","MCL","CLL","SLL","LPL","EMZL","MZL","NMZL","SMZL","HCL","MBL",
      "AITL","ALCL","PTCL","LBL","TCL","RT_DERIVED","TRANSFORMED_FL",
      "MM","SolM","MGUS","PCL","PCD",
      "HL"
    ),
    cohort_coarse = c(
      rep("B_CELL", 13),
      rep("T_CELL", 7),
      rep("PLASMA_CELL", 5),
      "HL"
    )
  )
  list(meta = meta, coarse_map = coarse_map)
}

# --- Pattern builder ---
build_dx_rules <- function(DX_groups) {
  rules <- rbindlist(lapply(names(DX_groups), function(g) {
    data.table(dx_group = g, raw_pat = DX_groups[[g]])
  }), fill = TRUE)
  rules <- rules[!is.na(raw_pat) & nzchar(raw_pat)]
  
  pre_anchored <- rules[startsWith(raw_pat, "^"), raw_pat]
  if (length(pre_anchored) > 0)
    warning(sprintf("build_dx_rules: %d pre-anchored pattern(s) used as-is.", length(pre_anchored)))
  
  rules[, stripped := sub("^\\^", "", raw_pat)]
  rules[, is_regex  := grepl("[\\$\\.|\\[\\]\\(\\)\\+\\*\\?\\\\]", stripped)]
  rules[, pattern   := fcase(
    startsWith(raw_pat, "^"), raw_pat,
    is_regex,                 raw_pat,
    default = paste0("^", raw_pat)
  )]
  rules[, c("stripped","is_regex") := NULL]
  rules[]
}

# --- Deduplication ---
dedup_dx_mapped <- function(dx_mapped) {
  dx_mapped <- as.data.table(copy(dx_mapped))
  core_cols    <- c("patientid","date_diagnosis","diagnosis","dx_group","priority")
  optional_cols <- c("tablename","datasource")
  by_cols      <- c(core_cols, intersect(optional_cols, names(dx_mapped)))
  dx_mapped    <- unique(dx_mapped, by = by_cols)
  setorder(dx_mapped, patientid, dx_group, date_diagnosis, priority)
  dx_mapped    <- dx_mapped[, .SD[1L], by = .(patientid, dx_group, date_diagnosis)]
  dx_mapped[]
}

# --- Code-to-group mapping ---
map_dx_to_groups <- function(dx, rules, code_col = "diagnosis", date_col = "date_diagnosis") {
  dx    <- as.data.table(copy(dx))
  rules <- as.data.table(copy(rules))
  stopifnot(code_col %in% names(dx), all(c("dx_group","pattern") %in% names(rules)))
  
  dx[, (code_col) := as.character(get(code_col))]
  dx[, (date_col) := as.IDate(get(date_col))]
  
  key_cols <- intersect(c("patientid", date_col, code_col, "priority"), names(dx))
  dx <- unique(dx, by = key_cols)
  dx[, row_id := .I]
  
  matches <- rbindlist(lapply(seq_len(nrow(rules)), function(i) {
    idx <- which(grepl(rules$pattern[i], dx[[code_col]], perl = TRUE))
    if (length(idx) == 0L) return(NULL)
    data.table(row_id = idx, dx_group = rules$dx_group[i])
  }), fill = TRUE)
  
  if (is.null(matches) || nrow(matches) == 0L) {
    dx[, row_id := NULL]
    return(dx[0L][, dx_group := character(0)])
  }
  matches <- unique(matches, by = c("row_id","dx_group"))
  out <- merge(dx, matches, by = "row_id", allow.cartesian = TRUE)
  out[, row_id := NULL]
  out[]
}

# --- Primary cohort assignment ---
assign_primary_cohort <- function(
    tx, dx_mapped, group_meta, coarse_map = NULL,
    lookback_days              = 2*3650L,
    grace_days                 = 14L,
    min_sep_days               = 14L,
    confirm_priority_threshold = 2L,
    allow_single_hq_hit        = TRUE
) {
  tx  <- as.data.table(copy(tx));  dxm <- as.data.table(copy(dx_mapped))
  gm  <- as.data.table(copy(group_meta))
  stopifnot(all(c("patientid","tx_date") %in% names(tx)))
  stopifnot(all(c("patientid","date_diagnosis","dx_group","priority") %in% names(dxm)))
  
  tx[,  tx_date        := as.IDate(tx_date)]
  dxm[, date_diagnosis := as.IDate(date_diagnosis)]
  setkey(tx, patientid); setkey(dxm, patientid)
  
  dxw <- dxm[tx, nomatch = 0L]
  dxw <- dxw[date_diagnosis >= (tx_date - lookback_days) &
               date_diagnosis <= (tx_date + grace_days)]
  
  patients_no_dx <- tx[!patientid %chin% dxw$patientid, patientid]
  
  if (nrow(dxw) == 0) {
    out <- tx[, .(patientid, tx_date,
                  cohort_fine = NA_character_, cohort_coarse = "UNKNOWN",
                  aggressive = FALSE, reason = "no_dx_in_window")]
    return(out)
  }
  
  dxw <- merge(dxw, gm, by = "dx_group", all.x = TRUE, allow.cartesian = TRUE)
  dxw[is.na(is_super),        is_super        := TRUE]
  dxw[is.na(include_primary), include_primary := FALSE]
  dxw[is.na(aggressive),      aggressive      := FALSE]
  dxw[is.na(specificity),     specificity     := 1L]
  dxw[is.na(aggr_rank),       aggr_rank       := 99L]
  
  setorder(dxw, patientid, dx_group, date_diagnosis)
  dxw[, n_hits    := .N,            by = .(patientid, dx_group)]
  dxw[, span_days := as.integer(max(date_diagnosis) - min(date_diagnosis)),
      by = .(patientid, dx_group)]
  dxw[, hi_pri_hit := as.integer(any(priority <= confirm_priority_threshold)),
      by = .(patientid, dx_group)]
  dxw[, confirmed := (hi_pri_hit == 1L) | (n_hits >= 2L & span_days >= min_sep_days)]
  if (allow_single_hq_hit)
    dxw[, confirmed := confirmed | (n_hits == 1L & priority == 1L)]
  
  dxc <- dxw[confirmed == TRUE]
  if (nrow(dxc) == 0) {
    out <- tx[, .(patientid, tx_date,
                  cohort_fine = NA_character_, cohort_coarse = "UNKNOWN",
                  aggressive = FALSE, reason = "no_confirmed_dx")]
    return(out)
  }
  
  dxc_best <- dxc[, .SD[order(priority, -date_diagnosis)][1], by = .(patientid, dx_group)]
  pres      <- unique(dxc_best[, .(patientid, dx_group)])
  
  # Derived: Richter transformation (CLL/SLL + DLBCL)
  rt_pats <- pres[dx_group %chin% c("CLL","SLL","DLBCL"),
                  .(has_cll = any(dx_group %chin% c("CLL","SLL")),
                    has_dlbcl = any(dx_group == "DLBCL")), by = patientid
  ][has_cll & has_dlbcl, patientid]
  
  # Derived: transformed FL (FL + DLBCL)
  tfl_pats <- pres[dx_group %chin% c("FL","DLBCL"),
                   .(has_fl = any(dx_group == "FL"),
                     has_dlbcl = any(dx_group == "DLBCL")), by = patientid
  ][has_fl & has_dlbcl, patientid]
  
  add_derived <- function(dxc_best, pats, label, agg_rank) {
    if (length(pats) == 0) return(dxc_best)
    rows <- dxc_best[patientid %chin% pats & dx_group == "DLBCL"]
    if (nrow(rows) == 0) rows <- dxc_best[patientid %chin% pats][, .SD[1], by = patientid]
    rows[, dx_group := label]
    rows[, `:=`(aggressive = TRUE, aggr_rank = agg_rank,
                is_super = FALSE, include_primary = TRUE, specificity = 6L)]
    rbind(dxc_best, rows, fill = TRUE)
  }
  #dxc_best <- add_derived(dxc_best, rt_pats,  "RT_DERIVED",    1L)
  #dxc_best <- add_derived(dxc_best, tfl_pats, "TRANSFORMED_FL", 2L)
  
  # ------------------------------------------------------------------------------
  # Derived group assignment with temporal plausibility constraints
  #
  # RT_DERIVED (Richter transformation): CLL/SLL must be diagnosed BEFORE DLBCL,
  # with a minimum gap of 90 days. A CLL patient who develops DLBCL on the same
  # admission is likely a coding artefact, not a true Richter transformation.
  #
  # TRANSFORMED_FL: FL must be diagnosed BEFORE DLBCL, with the same 90-day
  # minimum. De novo DLBCL in a patient who incidentally had FL coded years ago
  # at a different site would still qualify — but a simultaneous dual code does
  # not.
  #
  # min_days_prior: minimum days the indolent dx must predate the aggressive dx.
  # Clinically, true transformation virtually never happens at diagnosis — there
  # is always a prior indolent phase. 90 days is conservative; you could argue
  # for 180 days or even 365 days for FL transformation specifically.
  # ------------------------------------------------------------------------------
  
  add_derived_temporal <- function(dxc_best, dx_mapped,
                                   orig_groups, aggressive_group,
                                   label, agg_rank,
                                   min_days_prior = 90L) {
    
    dxc_best  <- as.data.table(copy(dxc_best))
    dx_mapped <- as.data.table(copy(dx_mapped))
    
    # Earliest confirmed date of the ORIGINAL (indolent) disease per patient
    orig_dates <- dx_mapped[
      dx_group %chin% orig_groups,
      .(orig_first_date = min(date_diagnosis)),
      by = patientid
    ]
    
    # Earliest confirmed date of the AGGRESSIVE disease per patient
    aggr_dates <- dx_mapped[
      dx_group == aggressive_group,
      .(aggr_first_date = min(date_diagnosis)),
      by = patientid
    ]
    
    # Patients who have both — apply temporal filter
    candidates <- merge(orig_dates, aggr_dates, by = "patientid") %>%
      filter(
        aggr_first_date > orig_first_date,                          # aggressive AFTER original
        as.integer(aggr_first_date - orig_first_date) >= min_days_prior  # by at least min_days_prior
      ) %>%
      pull(patientid)
    
    if (length(candidates) == 0) return(dxc_best)
    
    rows <- dxc_best[patientid %chin% candidates & dx_group == aggressive_group]
    if (nrow(rows) == 0)
      rows <- dxc_best[patientid %chin% candidates][, .SD[1], by = patientid]
    
    rows[, dx_group := label]
    rows[, `:=`(aggressive    = TRUE,
                aggr_rank     = agg_rank,
                is_super      = FALSE,
                include_primary = TRUE,
                specificity   = 6L)]
    
    rbind(dxc_best, rows, fill = TRUE)
  }
  
  
  dxc_best <- add_derived_temporal(
    dxc_best, dx_mapped,
    orig_groups      = c("CLL","SLL"),
    aggressive_group = "DLBCL",
    label            = "RT_DERIVED",
    agg_rank         = 1L,
    min_days_prior   = 90L
  )
  
  dxc_best <- add_derived_temporal(
    dxc_best, dx_mapped,
    orig_groups      = c("FL"),
    aggressive_group = "DLBCL",
    label            = "TRANSFORMED_FL",
    agg_rank         = 2L,
    min_days_prior   = 90L          # could argue for 180L or 365L for FL specifically
  )
  
  dxc_primary <- dxc_best[include_primary == TRUE]
  if (nrow(dxc_primary) == 0) {
    out <- tx[, .(patientid, tx_date,
                  cohort_fine = NA_character_, cohort_coarse = "UNCLASSIFIED",
                  aggressive = FALSE, reason = "confirmed_dx_only_in_excluded_groups")]
    return(out)
  }
  
  # Prefer fine over supergroup; prefer aggressive over non-aggressive
  dxc_primary[, has_fine := any(!is_super), by = patientid]
  dxc_primary <- dxc_primary[!(has_fine & is_super)][, has_fine := NULL]
  dxc_primary[, has_aggr := any(aggressive), by = patientid]
  dxc_primary <- dxc_primary[!(has_aggr & !aggressive)][, has_aggr := NULL]
  
  setorder(dxc_primary, patientid, aggr_rank, priority, -date_diagnosis, -specificity)
  chosen <- dxc_primary[, .SD[1], by = patientid]
  
  cohorts <- chosen[, .(
    patientid,
    cohort_fine = dx_group,
    aggressive  = aggressive,
    reason      = sprintf("aggr_rank=%s pri=%s dx_date=%s spec=%s n_hits=%s",
                          aggr_rank, priority, as.character(date_diagnosis),
                          specificity, n_hits)
  )]
  
  if (!is.null(coarse_map)) {
    cohorts <- merge(cohorts, as.data.table(coarse_map),
                     by.x = "cohort_fine", by.y = "dx_group", all.x = TRUE)
  } else {
    cohorts[, cohort_coarse := NA_character_]
  }
  cohorts[is.na(cohort_coarse),
          cohort_coarse := fifelse(
            grepl("T", cohort_fine), "T_CELL",
            fifelse(grepl("MM|MGUS|SolM|PCL|PCD", cohort_fine), "PLASMA_CELL",
                    fifelse(cohort_fine == "HL", "HL", "B_CELL"))
          )]
  
  out <- merge(tx[, .(patientid, tx_date)], cohorts, by = "patientid", all.x = TRUE)
  out[is.na(cohort_fine), `:=`(
    cohort_fine = NA_character_, aggressive = FALSE,
    cohort_coarse = "UNKNOWN", reason = "no_primary_label"
  )]
  out[patientid %chin% patients_no_dx, reason := "no_dx_in_window"]
  out[]
}

# --- Top-level wrapper ---
build_cohorts_from_dx <- function(
    tx, dx, DX_groups,
    lookback_days              = 3650L,
    grace_days                 = 14L,
    min_sep_days               = 14L,
    confirm_priority_threshold = 2L,
    allow_single_hq_hit        = TRUE
) {
  tx <- as.data.table(copy(tx)); dx <- as.data.table(copy(dx))
  stopifnot(all(c("patientid","tx_date") %in% names(tx)))
  stopifnot(all(c("patientid","date_diagnosis","diagnosis","tablename","datasource","priority") %in% names(dx)))
  
  tx[, tx_date        := as.IDate(tx_date)]
  dx[, date_diagnosis := as.IDate(date_diagnosis)]
  dx[, diagnosis      := as.character(diagnosis)]
  dx[, priority       := as.integer(priority)]
  
  # Reduce to minimal columns to prevent Cartesian join explosion
  dx_minimal <- dx[, .(patientid, date_diagnosis, diagnosis, priority)]
  dx_minimal <- unique(dx_minimal, by = c("patientid","date_diagnosis","diagnosis","priority"))
  setorder(dx_minimal, patientid, date_diagnosis, diagnosis, priority)
  dx_minimal <- dx_minimal[, .SD[1L], by = .(patientid, date_diagnosis, diagnosis)]
  
  cat(sprintf("dx: %d raw rows → %d after dedup (%.1f%% reduction)\n",
              nrow(dx), nrow(dx_minimal), 100 * (1 - nrow(dx_minimal) / nrow(dx))))
  
  rules     <- build_dx_rules(DX_groups)
  dx_mapped <- map_dx_to_groups(dx_minimal, rules)
  dx_mapped <- dedup_dx_mapped(dx_mapped)
  
  gm_out     <- make_group_meta(DX_groups)
  cohorts    <- assign_primary_cohort(
    tx = tx, dx_mapped = dx_mapped,
    group_meta = gm_out$meta, coarse_map = gm_out$coarse_map,
    lookback_days = lookback_days, grace_days = grace_days,
    min_sep_days = min_sep_days,
    confirm_priority_threshold = confirm_priority_threshold,
    allow_single_hq_hit = allow_single_hq_hit
  )
  
  # Audit table for unclassified patients
  failed_ids <- cohorts[is.na(cohort_fine), patientid]
  audit <- dx_minimal[patientid %chin% failed_ids] |>
    merge(tx[, .(patientid, tx_date)], by = "patientid", all.x = TRUE) |>
    as.data.table()
  audit[, days_from_tx := as.integer(date_diagnosis - tx_date)]
  audit <- merge(audit,
                 dx_mapped[patientid %chin% failed_ids, .(patientid, date_diagnosis, diagnosis, dx_group)],
                 by = c("patientid","date_diagnosis","diagnosis"), all.x = TRUE)
  setorder(audit, patientid, date_diagnosis)
  
  list(cohorts = cohorts, dx_mapped = dx_mapped, rules = rules,
       group_meta = gm_out$meta, audit = audit)
}

# --- Run cohort assignment ---
# tx needs patientid column; tx_flagged uses patient_id — align here
tx_for_cohorts <- firstline_tbl %>% 
  filter(!is.na(firstline_date)) %>%
  select(patientid = patient_id, tx_date = firstline_date) %>%
  distinct()

res <- build_cohorts_from_dx(
  tx        = tx_for_cohorts,
  dx        = t_dalycare_diagnoses,
  DX_groups = DX_groups,
  lookback_days              = 2*3650L,
  confirm_priority_threshold = 3L,
  allow_single_hq_hit        = TRUE
)
cohorts <- res$cohorts

cat("\n=== Cohort assignment summary ===\n")
table(cohorts$cohort_fine,   useNA = "ifany") |> sort(decreasing = TRUE) |> print()
table(cohorts$cohort_coarse, useNA = "ifany") |> print()
cat("\n=== Failure reasons ===\n")
cohorts[is.na(cohort_fine), .N, by = reason] |> print()
cat("\n=== Most common dx codes among unclassified patients ===\n")
res$audit[is.na(dx_group), .N, by = diagnosis][order(-N)][1:30] |> print()
audit <- res$audit

audit_with_treatments <- audit %>% left_join(firstline_tbl %>% mutate(patientid = patient_id))

# ==============================================================================
# 7. Merge cohort into firstline table + grouped cohort labels
# ==============================================================================

firstline_tbl_all <- firstline_tbl ### preserve all outputs for later arm rescue

firstline_tbl <- firstline_tbl %>%
  filter(!is.na(firstline_date)) %>%
  rename(patientid = patient_id) %>%
  left_join(cohorts, by = "patientid") %>%
  mutate(
    cohort_fine = if_else(cohort_fine == "RT_DERIVED", "RT", cohort_fine),
    cohort_grouped = case_when(
      cohort_fine %in% c("CLL","SLL")                                      ~ "CLL_SLL",
      cohort_fine %in% c("FL","EMZL", "MZL", "NMZL","SMZL", "LPL", "BCL")                    ~ "Indolent_B_NHL",
      cohort_fine %in% c("DLBCL","RT","TRANSFORMED_FL")       ~ "DLBCL_like",
      cohort_fine %in% c("MCL")                                            ~ "MCL",
      cohort_fine %in% c("HL")                                              ~ "HL",
      cohort_fine %in% c("MM","PCD","PCL", "SolM")                                  ~ "MM",
      cohort_fine %in% c("AITL","ALCL","PTCL","TCL")                       ~ "T_NHL",
      cohort_fine %in% c("HCL")                                             ~ "HCL",
      cohort_fine %in% c("BL", "LBL")                                      ~ "BL_LBL",
      TRUE ~ cohort_fine
    )
  )


# ==============================================================================
# 7b. PASS 2: Forward-looking rescue for cohort-unclassified patients
#
# Context:
#   Pass 1 (sections 5-7) finds first treatment from earliest heme diagnosis,
#   then assigns cohort contemporaneously. It fails when specific ICD codes
#   were registered AFTER the treatment that Pass 1 found — the classic case
#   from the audit: basic cytostatic treatment / methotrexate with no specific
#   dx in the ±14 day window.
#
# What Pass 2 does:
#   For patients with cohort_fine = NA after Pass 1, search FORWARD in their
#   full diagnosis history for the first date any specific, confirmable heme
#   group appears. Use that date as a new anchor and re-run treatment search
#   from [anchor - 90d, end of study]. Cohort is then assigned contemporaneously
#   around the new firstline_date, with a 90-day forward grace (vs 14d in Pass 1)
#   to handle remaining coding lag.
#
# Bias accepted:
#   Treatment that occurred before the specific diagnosis date (within 90d) may
#   be captured as first-line. This is intentional — 90 days covers plausible
#   coding lag. Treatment more than 90 days before the specific dx is excluded,
#   avoiding backdating to genuinely unrelated earlier treatments.
#
# What this does NOT do:
#   - Override any Pass 1 result where cohort_fine is not NA
#   - Change outcomes or follow-up — only the index date and cohort may differ
#
# cohort_source values produced:
#   "pass1"         — cohort assigned in Pass 1 (contemporaneous, ±14d window)
#   "pass2_fwd_dx"  — specific dx found forward; treatment re-identified from there
#   "arm_rescue"    — added in section 8b after arm functions are defined
# ==============================================================================

# Tag Pass 1 successes
firstline_tbl <- firstline_tbl %>%
  mutate(cohort_source = if_else(!is.na(cohort_fine), "pass1", NA_character_))

# Patients who need Pass 2: have a treatment date but no cohort
fl_count("pass1_assigned_cohort", firstline_tbl %>% filter(!is.na(cohort_fine)),
         unit = "patients")
fl_count("pass1_unclassified_cohort", firstline_tbl %>% filter(is.na(cohort_fine)),
         unit = "patients")
pass1_failed_ids <- firstline_tbl %>%
  filter(is.na(cohort_fine)) %>%
  pull(patientid)
pass1_unclassified <- firstline_tbl %>%
  filter(is.na(cohort_fine)) %>%
  select(patientid, firstline_date, reason, firstline_source,
         firstline_has_heme_indication, firstline_has_rkkp_source,
         firstline_n_codes)

cat(sprintf(
  "\n=== PASS 2: %d patients unclassified in Pass 1 ===\n",
  length(pass1_failed_ids)
))

if (length(pass1_failed_ids) > 0) {
  
  # ── Step 1: Find first SPECIFIC confirmable heme diagnosis ──────────────────
  #
  # "Specific" = include_primary = TRUE in make_group_meta().
  # This excludes UNS, BCL_UNS, Lymphoma_UNS, Leukemia_UNS, MGUS, MBL, DALYCARE.
  # We use the same confirmation criteria as assign_primary_cohort():
  # confirmed = high-priority hit (priority ≤ 3) OR 2+ hits ≥ 14 days apart,
  # OR single priority-1 hit.
  
  gm_p2             <- make_group_meta(DX_groups)
  includable_groups <- gm_p2$meta[include_primary == TRUE, dx_group]
  rules     <- build_dx_rules(DX_groups)
  
  # Diagnoses for failed patients only
  dx_p2 <- t_dalycare_diagnoses %>%
    filter(patientid %in% pass1_failed_ids) %>%
    as.data.table()
  
  # Map to groups (reuse existing rules object from section 6)
  dx_mapped_p2 <- map_dx_to_groups(
    dx_p2[, .(patientid, date_diagnosis, diagnosis, priority)],
    rules   # built in build_cohorts_from_dx() section 6 — must already be in env
    # If not available: rules <- build_dx_rules(DX_groups)
  )
  dx_mapped_p2 <- dedup_dx_mapped(dx_mapped_p2)
  dx_mapped_p2 <- as.data.table(dx_mapped_p2)
  
  # Apply confirmation logic
  dx_mapped_p2[, n_hits    := .N,   by = .(patientid, dx_group)]
  dx_mapped_p2[, span_days := as.integer(max(date_diagnosis) - min(date_diagnosis)),
               by = .(patientid, dx_group)]
  dx_mapped_p2[, hi_pri_hit := as.integer(any(priority <= 3L)),
               by = .(patientid, dx_group)]
  dx_mapped_p2[, confirmed  :=
                 (hi_pri_hit == 1L) |
                 (n_hits >= 2L & span_days >= 14L) |
                 (n_hits == 1L & priority == 1L)]   # allow_single_hq_hit
  
  # Keep confirmed, includable, specific groups
  dx_specific_p2 <- dx_mapped_p2[
    confirmed == TRUE & dx_group %chin% includable_groups
  ]
  
  cat(sprintf("  Patients with a confirmable specific dx anywhere in history: %d\n",
              dx_specific_p2[, uniqueN(patientid)]))
  
  # Earliest confirmed specific dx per patient = Pass 2 anchor date
  p2_anchors <- dx_specific_p2[,
                               .(dx_date   = min(date_diagnosis),
                                 diagnosis = first(diagnosis[date_diagnosis == min(date_diagnosis)])),
                               by = patientid
  ] %>%
    rename(patient_id = patientid)
  
  # ── Step 2: Re-run treatment search from Pass 2 anchor ──────────────────────
  #
  # 90-day pre-dx grace: covers coding lag (treatment before specific code registered)
  # 365*20 upper bound: open-ended, handles long watch-and-wait
  # Uses the SAME make_firstline() function and tx_flagged table as Pass 1.
  
  firstline_p2_raw <- make_firstline(
    tx                = tx_flagged %>% filter(patient_id %in% p2_anchors$patient_id),
    dx                = p2_anchors,
    pre_dx_grace_days = 90,
    max_days_from_dx  = 365 * 20,
    seed_tbl = seed_firstline,
    seed_max_gap = 30, accept_cd20_mono = TRUE, audit_label = "pass2"
  ) %>%
    filter(!is.na(firstline_date))
  
  cat(sprintf("  Treatment found for %d patients in Pass 2\n", nrow(firstline_p2_raw)))
  
  if (nrow(firstline_p2_raw) > 0) {
    
    # ── Step 3: Cohort assignment contemporaneous with Pass 2 firstline_date ──
    #
    # grace_days = 90L (vs 14L in Pass 1): the core coding-lag fix.
    # A patient treated on 2018-03-01 with specific DLBCL code arriving 2018-03-20
    # will now be correctly classified as DLBCL.
    
    tx_for_cohorts_p2 <- firstline_p2_raw %>%
      select(patientid = patient_id, tx_date = firstline_date) %>%
      distinct()
    
    res_p2 <- build_cohorts_from_dx(
      tx        = tx_for_cohorts_p2,
      dx        = t_dalycare_diagnoses,
      DX_groups = DX_groups,
      lookback_days              = 3650L,
      grace_days                 = 90L,   # wider than Pass 1 — coding lag fix
      confirm_priority_threshold = 3L,
      allow_single_hq_hit        = TRUE
    )
    cohorts_p2 <- res_p2$cohorts
    
    # ── Step 4: Build Pass 2 firstline table in same shape as Pass 1 ──────────
    
    firstline_tbl_p2 <- firstline_p2_raw %>%
      rename(patientid = patient_id) %>%
      left_join(as_tibble(cohorts_p2), by = "patientid") %>%
      mutate(
        cohort_source  = "pass2_fwd_dx",
        cohort_grouped = case_when(
          cohort_fine %in% c("CLL","SLL")                                      ~ "CLL_SLL",
          cohort_fine %in% c("FL","EMZL","MZL","NMZL","SMZL","LPL","BCL")    ~ "Indolent_B_NHL",
          cohort_fine %in% c("DLBCL","RT","RT_DERIVED","TRANSFORMED_FL")      ~ "DLBCL_like",
          cohort_fine %in% c("MCL")                                            ~ "MCL",
          cohort_fine %in% c("HL")                                             ~ "HL",
          cohort_fine %in% c("MM","PCD","PCL", "SolM")                                 ~ "MM",
          cohort_fine %in% c("AITL","ALCL","PTCL","TCL")                      ~ "T_NHL",
          cohort_fine %in% c("HCL")                                            ~ "HCL",
          cohort_fine %in% c("BL","LBL")                                       ~ "BL_LBL",
          TRUE ~ cohort_fine
        )
      )
    
    # Report Pass 2 outcomes
    n_p2_classified   <- sum(!is.na(firstline_tbl_p2$cohort_fine))
    n_p2_unclassified <- sum( is.na(firstline_tbl_p2$cohort_fine))
    fl_count("pass2_assigned_cohort", firstline_tbl_p2 %>% filter(!is.na(cohort_fine)),
             branch = "pass2", unit = "patients")
    cat(sprintf(
      "  Pass 2 cohort assigned: %d  |  still unclassified: %d\n",
      n_p2_classified, n_p2_unclassified
    ))
    cat("  Pass 2 cohort_grouped distribution:\n")
    firstline_tbl_p2 %>%
      filter(!is.na(cohort_fine)) %>%
      count(cohort_grouped, sort = TRUE) %>%
      print()
    
    # IDs still unclassified after Pass 2 — queued for arm rescue in section 8b
    p2_classified_ids <- firstline_tbl_p2 %>%
      filter(!is.na(cohort_fine)) %>%
      pull(patientid)
    
    arm_rescue_ids <- setdiff(
      pass1_failed_ids,
      p2_classified_ids
    )
    
    # ── Step 5: Stack Pass 1 + Pass 2 ─────────────────────────────────────────
    #
    # Pass 2 rows REPLACE Pass 1 rows for the same patients (Pass 1 had no cohort
    # for these patients — their rows are discarded, not kept as alternatives).
    # Patients with no treatment found in Pass 2 and no arm rescue remain absent.
    
    firstline_tbl <- bind_rows(
      firstline_tbl %>% filter(!is.na(cohort_fine)),          # Pass 1 successes
      firstline_tbl_p2 %>% filter(!is.na(cohort_fine))        # Pass 2 successes
    )
    
  } else {
    # Pass 2 found no treatments — all failed patients go to arm rescue
    arm_rescue_ids <- pass1_failed_ids
    firstline_tbl  <- firstline_tbl %>% filter(!is.na(cohort_fine))
    cat("  No treatments found in Pass 2 — all failed patients queued for arm rescue\n")
  }
  
  cat(sprintf("  Arm rescue queue: %d patients\n", length(arm_rescue_ids)))
  
} else {
  
  arm_rescue_ids <- character(0)
  cat("All patients classified in Pass 1\n")
  
}

cat("\n=== Post-Pass 2 cohort distribution ===\n")
count(firstline_tbl, cohort_grouped, cohort_source, sort = TRUE) %>%
  print(n = 40)

# Aggregate-only review of patients still lacking a fine disease label.
# These flags describe diagnostic support, not a proposed automatic rescue.
fl_count("unclassified_after_pass2", pass1_unclassified %>%
           filter(patientid %in% arm_rescue_ids), unit = "patients")
if (nrow(pass1_unclassified) > 0L) {
  specific_groups <- make_group_meta(DX_groups)$meta[include_primary == TRUE, dx_group]
  specific_dx <- as_tibble(res$dx_mapped) %>%
    filter(patientid %in% pass1_unclassified$patientid,
           dx_group %in% specific_groups) %>%
    inner_join(pass1_unclassified %>% select(patientid, firstline_date),
               by = "patientid") %>%
    mutate(days_from_index = as.integer(as.Date(date_diagnosis) -
                                        as.Date(firstline_date))) %>%
    group_by(patientid) %>%
    summarise(specific_any = TRUE,
              specific_in_pass1_window = any(days_from_index >= -7300L &
                                             days_from_index <= 14L),
              specific_in_pass2_window = any(days_from_index >= -3650L &
                                             days_from_index <= 90L),
              .groups = "drop")
  unclassified_flags <- pass1_unclassified %>%
    left_join(specific_dx, by = "patientid") %>%
    mutate(across(c(specific_any, specific_in_pass1_window,
                    specific_in_pass2_window), ~ replace_na(.x, FALSE)),
           pass2_assigned = !patientid %in% arm_rescue_ids) %>%
    count(reason, firstline_source, specific_any,
          specific_in_pass1_window, specific_in_pass2_window,
          pass2_assigned, name = "n_patients")
  .fl_audit$unclassified_flags <- unclassified_flags
}


# ==============================================================================
# 8. Treatment arm assignment
# ==============================================================================

norm_text <- function(x) {
  x %>%
    str_to_lower() %>%
    str_replace_all("\\s+", " ") %>%
    str_replace_all("\\s*\\|\\s*", " | ") %>%
    str_trim()
}

has_token <- function(txt, token) str_detect(txt, fixed(token, ignore_case = FALSE))
has_any   <- function(txt, tokens) any(vapply(tokens, function(t) has_token(txt, t), logical(1)))

# Token dictionaries for treatment name matching

TOK <- list(
  # --- Anti-CD20 / monoclonal antibodies ---
  anti_cd20     = c("rituximab","obinutuzumab","ofatumumab","cd20 antibody"),
  cd52          = c("cd52 antibody","alemtuzumab"),
  
  # --- Named regimen shortcuts (used when individual components not coded) ---
  chop_name     = c("chop"),
  choep_name    = c("choep"),
  cop_name      = c("cop"),
  mime_name     = c("mime"),
  
  # --- Conventional cytotoxics ---
  cyclophosphamide = c("cyclophosphamide","high-dose cyclophosphamide"),
  doxorubicin   = c("doxorubicin"),
  vincristine   = c("vincristine"),
  vinblastine   = c("vinblastine"),
  etoposide     = c("etoposide"),
  dacarbazine   = c("dacarbazine"),
  bleomycin     = c("bleomycin"),
  procarbazine  = c("procarbazine"),
  bendamustine  = c("bendamustine"),
  chlorambucil  = c("chlorambucil"),
  fludarabine   = c("fludarabine"),
  cladribine    = c("cladribine"),
  carboplatin   = c("carboplatin"),
  cisplatin     = c("cisplatin"),
  ifosfamide    = c("ifosfamide"),
  methotrexate  = c("methotrexate","high-dose methotrexate"),
  cytarabine    = c("cytarabine","high-dose cytarabine"),
  temozolomide  = c("temozolomide"),
  gemcitabine   = c("gemcitabine"),              # NEW: GemOx-R, GDP, PVAG, T-NHL
  intrathecal   = c("prednisolon intrathecal (triple cns)","triple cns","intrathecal"),
  
  # --- Targeted / novel agents ---
  btk           = c("ibrutinib","acalabrutinib","zanubrutinib"),
  venetoclax    = c("venetoclax"),
  pi3k          = c("idelalisib","duvelisib"),
  polatuzumab   = c("polatuzumab","polatuzumab vedotin"),  # NEW: pola-R-CHP
  
  # --- Pentostatins ----
  pentostatin = c("pentostatin", "deoxycoformycin"),
  
  # --- MM-specific agents ---
  bortezomib    = c("bortezomib"),
  carfilzomib   = c("carfilzomib"),
  ixazomib      = c("ixazomib"),
  lenalidomide  = c("lenalidomide"),
  thalidomide   = c("thalidomide"),
  pomalidomide  = c("pomalidomide"),
  daratumumab   = c("daratumumab"),
  isatuximab    = c("isatuximab"),               # NEW: isa-Kd, isa-Pd
  elotuzumab    = c("elotuzumab"),               # NEW: elo-Rd, elo-Pd
  melphalan     = c("melphalan"),
  selinexor     = c("selinexor"),                # NEW: XPO1 inhibitor
  panobinostat  = c("panobinostat"),             # NEW: HDAC inhibitor
  dexa_vad      = c("vincristine | doxorubicin | dexametason (vad)","vad"),
  cytostatic_mm = c("cytostatic_mm","other_mm"),
  
  # --- MM: bispecific antibodies (T-cell engaging) ---
  # Note: "bi-speficic" is a source-system misspelling — both forms caught
  bispec_mm     = c(
    "bi-speficic antibody", "bi-specific antibody",
    "bi-speficic antibody (t-cell engaging)",
    "teclistamab","elranatamab","talquetamab",
    "linvoseltamab","alnuctamab","forimtamig",
    "cevostamab"
  ),
  
  # --- MM: CAR-T cell therapies ---
  cart_mm       = c(
    "ciltacabtagene autoleucel","idecabtagene vicleucel",
    "cilta-cel","ide-cel",
    "belantamab-mafodotin"      # ADC — grouped here as advanced cell/antibody therapy
  ),
  
  # --- HL / T-NHL ---
  brentuximab   = c("brentuximab","brentuximab vedotin"),
  pd1           = c("pembrolizumab","nivolumab","ipilimumab | nivolumab"),
  
  # --- Procedures ---
  radiotherapy  = c("radiotherapy"),
  asct          = c("asct","asct (conditioning)"),
  allo          = c("allo","allo (conditioning)"),
  
  # --- Generic / unresolved cytostatic codes ---
  # These fire when the source system recorded a bucket code instead of individual drugs.
  # They should NOT automatically trigger exclusion — see assign_arms_one for handling.
  generic_basic          = c("cytostatic treatment","basic cytostatic treatment"),
  generic_complex        = c("complex cytostatic treatment"),
  generic_antineoplastic = c(
    "antineoplastic agents","other_chemo",
    "other immunotherapy","other_immunotherapy",
    "monoclonal antibodies and antibody drug conjugates",
    "other antineoplastic agents"
  ),
  
  # --- Clearly non-hematological drugs (trigger exclusion regardless of cohort) ---
  clearly_non_hematology = c(
    "imatinib","dasatinib","nilotinib","osimertinib","ribociclib",
    "dabrafenib | trametinib","atezolizumab","nintedanib","ruxolitinib",
    "azacitidine","anagrelide","hydroxurea","hydroxycarbamide",
    "hydroxurea | hydroxycarbamide"
  )
)

# Cohort predicate helpers
is_cll_like    <- function(cf, cc) cf %in% c("CLL","SLL")
is_hcl         <- function(cf, cc) cf == "HCL"
is_hl          <- function(cf, cc) cf == "HL"
is_mm_like     <- function(cf, cc) cf %in% c("MM","PCD","PCL", "SolM") | cc %in% "PLASMA_CELL"
is_aggressive_b <- function(cf, cc) cf %in% c("DLBCL","RT","RT_DERIVED","TRANSFORMED_FL", "MCL")
is_indolent_b  <- function(cf, cc) cf %in% c("FL","EMZL","MZL", "NMZL","SMZL", "LPL","BCL")
is_t_cell      <- function(cf, cc) cc %in% "T_CELL" | cf %in% c("AITL","ALCL","PTCL","TCL")
is_bl_lbl <- function(cf, cc) cf %in% c("BL", "LBL")
is_mcl <- function(cf, cc) cf == "MCL"

assign_arms_one <- function(txt_raw,
                            cohort_fine   = NA_character_,
                            cohort_coarse = NA_character_) {
  txt <- norm_text(txt_raw)
  cf  <- if (is.na(cohort_fine))   "" else as.character(cohort_fine)
  cc  <- if (is.na(cohort_coarse)) "" else as.character(cohort_coarse)
  
  # --------------------------------------------------------------------------
  # Token flags
  # --------------------------------------------------------------------------
  anti_cd20  <- has_any(txt, TOK$anti_cd20)
  cd52       <- has_any(txt, TOK$cd52)
  chop_name  <- has_any(txt, TOK$chop_name)
  choep_name <- has_any(txt, TOK$choep_name)
  cop_name   <- has_any(txt, TOK$cop_name)
  mime_name  <- has_any(txt, TOK$mime_name)
  cyclo      <- has_any(txt, TOK$cyclophosphamide)
  doxo       <- has_any(txt, TOK$doxorubicin)
  vincr      <- has_any(txt, TOK$vincristine)
  vinbl      <- has_any(txt, TOK$vinblastine)
  etop       <- has_any(txt, TOK$etoposide)
  daca       <- has_any(txt, TOK$dacarbazine)
  bleo       <- has_any(txt, TOK$bleomycin)
  proca      <- has_any(txt, TOK$procarbazine)
  benda      <- has_any(txt, TOK$bendamustine)
  chlor      <- has_any(txt, TOK$chlorambucil)
  fluda      <- has_any(txt, TOK$fludarabine)
  clad       <- has_any(txt, TOK$cladribine)
  carbo      <- has_any(txt, TOK$carboplatin)
  cispl      <- has_any(txt, TOK$cisplatin)
  ifos       <- has_any(txt, TOK$ifosfamide)
  mtx        <- has_any(txt, TOK$methotrexate)
  ara_c      <- has_any(txt, TOK$cytarabine)
  tmz        <- has_any(txt, TOK$temozolomide)
  gemci      <- has_any(txt, TOK$gemcitabine)
  intrathecal <- has_any(txt, TOK$intrathecal)
  btk        <- has_any(txt, TOK$btk)
  ven        <- has_any(txt, TOK$venetoclax)
  pola       <- has_any(txt, TOK$polatuzumab)
  bort       <- has_any(txt, TOK$bortezomib)
  carf       <- has_any(txt, TOK$carfilzomib)
  lena       <- has_any(txt, TOK$lenalidomide)
  thal       <- has_any(txt, TOK$thalidomide)
  poma       <- has_any(txt, TOK$pomalidomide)
  dara       <- has_any(txt, TOK$daratumumab)
  isatux     <- has_any(txt, TOK$isatuximab)
  elotuz     <- has_any(txt, TOK$elotuzumab)
  melph      <- has_any(txt, TOK$melphalan)
  selinex    <- has_any(txt, TOK$selinexor)
  pano       <- has_any(txt, TOK$panobinostat)
  vad        <- has_any(txt, TOK$dexa_vad)
  mm_bucket  <- has_any(txt, TOK$cytostatic_mm)
  bispec     <- has_any(txt, TOK$bispec_mm)
  cart       <- has_any(txt, TOK$cart_mm)
  bv         <- has_any(txt, TOK$brentuximab)
  pd1        <- has_any(txt, TOK$pd1)
  rt         <- has_any(txt, TOK$radiotherapy)
  asct       <- has_any(txt, TOK$asct)
  allo       <- has_any(txt, TOK$allo)
  pento      <- has_any(txt, TOK$pentostatin)
  generic_basic          <- has_any(txt, TOK$generic_basic)
  generic_complex        <- has_any(txt, TOK$generic_complex)
  generic_antineoplastic <- has_any(txt, TOK$generic_antineoplastic)
  non_heme               <- has_any(txt, TOK$clearly_non_hematology)
  
  # --------------------------------------------------------------------------
  # Explicit combination flags
  # --------------------------------------------------------------------------
  explicit_rchop         <- anti_cd20 & cyclo & doxo & vincr
  explicit_chop          <- !anti_cd20 & cyclo & doxo & vincr
  explicit_rchoep        <- anti_cd20 & cyclo & doxo & vincr & etop
  explicit_choep         <- !anti_cd20 & cyclo & doxo & vincr & etop
  explicit_abvd          <- bleo & daca & doxo & vinbl
  explicit_beacopp_like  <- bleo & cyclo & daca & doxo & etop & proca
  explicit_fcr           <- fluda & cyclo & anti_cd20
  explicit_fc            <- fluda & cyclo & !anti_cd20
  explicit_fr            <- fluda & anti_cd20 & !cyclo
  explicit_br            <- benda & anti_cd20
  explicit_chlor_anti_cd20 <- chlor & anti_cd20
  explicit_vrd           <- bort & lena
  explicit_vtd           <- bort & thal
  explicit_vcd           <- bort & cyclo
  explicit_krd           <- carf & lena
  explicit_dara_vrd      <- dara & bort & lena
  explicit_dara_vcd      <- dara & bort & cyclo
  explicit_vmp           <- bort & melph
  explicit_hcl_clad      <- clad
  explicit_platinum_etop <- carbo & etop
  explicit_cns_like      <- mtx | ara_c | intrathecal | tmz
  explicit_cop           <- cyclo & vincr & !doxo
  # NEW combos
  explicit_pola_rchp     <- pola & anti_cd20 & cyclo & doxo   # pola-R-CHP
  explicit_r_gemox       <- gemci & anti_cd20 & (carbo | cispl) # R-GemOx
  explicit_isa_kd        <- isatux & carf                       # isa-Kd
  explicit_isa_pd        <- isatux & poma                       # isa-Pd
  explicit_elo_rd        <- elotuz & lena                       # elo-Rd
  explicit_elo_pd        <- elotuz & poma                       # elo-Pd
  explicit_dara_isa      <- dara & isatux                       # unusual — flag
  
  # --------------------------------------------------------------------------
  # Defaults
  # --------------------------------------------------------------------------
  arm_primary         <- "Other/Unmapped"
  arm_secondary       <- "Other/Unmapped"
  arm_exclude_primary <- FALSE
  
  # --------------------------------------------------------------------------
  # Assignment — non-heme exclusion is unconditional
  # --------------------------------------------------------------------------
  if (non_heme) {
    arm_primary <- arm_secondary <- "Exclude: likely non-heme/other indication"
    arm_exclude_primary <- TRUE
    
    # --------------------------------------------------------------------------
    # CLL / SLL
    # --------------------------------------------------------------------------
  } else if (is_cll_like(cf, cc)) {
    if      (btk)                      { arm_primary <- "CLL: BTKi-based"
    arm_secondary <- if (anti_cd20) "CLL: BTKi + anti-CD20" else "CLL: BTKi monotherapy"
    } else if (ven & anti_cd20)        { arm_primary <- arm_secondary <- "CLL: Venetoclax + anti-CD20"
    } else if (ven)                    { arm_primary <- arm_secondary <- "CLL: Venetoclax-based"
    } else if (explicit_fcr)           { arm_primary <- arm_secondary <- "CLL: FCR-like"
    } else if (explicit_fc)            { arm_primary <- arm_secondary <- "CLL: FC-like"
    } else if (explicit_fr)            { arm_primary <- arm_secondary <- "CLL: Fludarabine + anti-CD20"
    } else if (explicit_chlor_anti_cd20) { arm_primary <- arm_secondary <- "CLL: Chlorambucil + anti-CD20"
    } else if (chlor)                  { arm_primary <- "CLL: Chlorambucil-based"
    arm_secondary <- "CLL: Chlorambucil monotherapy"
    } else if (explicit_br)            { arm_primary <- arm_secondary <- "CLL: BR-like"
    } else if (benda)                  { arm_primary <- "CLL: Bendamustine-based"
    arm_secondary <- "CLL: Bendamustine monotherapy"
    } else if (anti_cd20)              { arm_primary <- "CLL: anti-CD20-based"
    arm_secondary <- "CLL: anti-CD20 without clear backbone"
    } else if (cd52)                   { arm_primary <- arm_secondary <- "CLL: CD52-based"
    } else if (fluda)                  { arm_primary <- "CLL: Fludarabine-based"
    arm_secondary <- "CLL: Fludarabine monotherapy"
    } else if (cyclo)                  { arm_primary <- arm_secondary <- "CLL: Cyclophosphamide-based"
    } else if (cop_name | explicit_cop) { arm_primary <- arm_secondary <- "CLL: COP-like"
    # Generic cytostatic in CLL: most likely chlorambucil or CVP without drug-level coding.
    # CD20 co-occurrence is the most informative split. Do NOT exclude.
    } else if (generic_basic & anti_cd20)    { arm_primary <- arm_secondary <- "CLL: Chemo-immunotherapy (generic code)"
    } else if (generic_basic | generic_complex) { arm_primary <- arm_secondary <- "CLL: Chemotherapy (generic code)"
    }
    
    # --------------------------------------------------------------------------
    # HCL
    # --------------------------------------------------------------------------
  } else if (is_hcl(cf, cc)) {
    if      (explicit_hcl_clad & anti_cd20) { arm_primary <- arm_secondary <- "HCL: Cladribine + anti-CD20"
    } else if (explicit_hcl_clad)           { arm_primary <- arm_secondary <- "HCL: Cladribine-based"
    } else if (anti_cd20)                   { arm_primary <- arm_secondary <- "HCL: anti-CD20-based"
    } else if (pento)                       { arm_primary <- arm_secondary <- "HCL: Pentostatin-based"
    } 
    
    # --------------------------------------------------------------------------
    # Hodgkin Lymphoma
    # --------------------------------------------------------------------------
  } else if (is_hl(cf, cc)) {
    if      (bv & (explicit_abvd | explicit_beacopp_like)) { arm_primary <- arm_secondary <- "HL: Brentuximab + chemo"
    } else if (explicit_abvd | has_any(txt, c("abvd")))    { arm_primary <- arm_secondary <- "HL: ABVD-like"
    } else if (explicit_beacopp_like | has_any(txt, c("beacopp"))) { arm_primary <- arm_secondary <- "HL: BEACOPP-like"
    } else if (bv)    { arm_primary <- arm_secondary <- "HL: Brentuximab-based"
    } else if (pd1)   { arm_primary <- arm_secondary <- "HL: PD1-based"
    } else if (cop_name | explicit_cop | explicit_chop | chop_name) { arm_primary <- arm_secondary <- "HL: COP/CHOP-like other"
    } else if (rt & !generic_antineoplastic) {
      arm_primary <- arm_secondary <- "HL: Radiotherapy-only/procedure"
      arm_exclude_primary <- TRUE
    }
    
    # --------------------------------------------------------------------------
    # Multiple Myeloma / Plasma cell disorders
    # --------------------------------------------------------------------------
  } else if (is_mm_like(cf, cc)) {
    # Priority: most specific combos first, then single-agent novel, then legacy,
    # then generic fallbacks (which are NOT excluded in MM context).
    if      (explicit_dara_vrd)   { arm_primary <- arm_secondary <- "MM: Dara-VRd-like"
    } else if (explicit_dara_vcd) { arm_primary <- arm_secondary <- "MM: Dara-VCd-like"
    } else if (explicit_dara_isa) { arm_primary <- arm_secondary <- "MM: Dara + Isatuximab (unusual combo — review)"
    } else if (explicit_isa_kd)   { arm_primary <- arm_secondary <- "MM: Isa-Kd-like"
    } else if (explicit_isa_pd)   { arm_primary <- arm_secondary <- "MM: Isa-Pd-like"
    } else if (explicit_elo_rd)   { arm_primary <- arm_secondary <- "MM: Elo-Rd-like"
    } else if (explicit_elo_pd)   { arm_primary <- arm_secondary <- "MM: Elo-Pd-like"
    } else if (explicit_vrd)      { arm_primary <- arm_secondary <- "MM: VRd-like"
    } else if (explicit_vtd)      { arm_primary <- arm_secondary <- "MM: VTd-like"
    } else if (explicit_vcd)      { arm_primary <- arm_secondary <- "MM: VCd-like"
    } else if (explicit_krd)      { arm_primary <- arm_secondary <- "MM: KRd-like"
    } else if (explicit_vmp)      { arm_primary <- arm_secondary <- "MM: VMP-like"
    } else if (vad)               { arm_primary <- arm_secondary <- "MM: VAD-like"
    } else if (dara)              { arm_primary <- arm_secondary <- "MM: Daratumumab-based"
    } else if (isatux)            { arm_primary <- arm_secondary <- "MM: Isatuximab-based"
    } else if (elotuz)            { arm_primary <- arm_secondary <- "MM: Elotuzumab-based"
    } else if (bort)              { arm_primary <- arm_secondary <- "MM: Bortezomib-based"
    } else if (carf)              { arm_primary <- arm_secondary <- "MM: Carfilzomib-based"
    } else if (lena | poma | thal) {
      arm_primary   <- "MM: IMiD-based"
      arm_secondary <- if (lena) "MM: Lenalidomide-based" else if (poma) "MM: Pomalidomide-based" else "MM: Thalidomide-based"
    } else if (selinex)           { arm_primary <- arm_secondary <- "MM: Selinexor-based"
    } else if (pano)              { arm_primary <- arm_secondary <- "MM: Panobinostat-based"
    } else if (melph)             { arm_primary <- arm_secondary <- "MM: Melphalan-based"
    } else if (cyclo)             { arm_primary <- arm_secondary <- "MM: Cyclophosphamide-based"
    } else if (bispec)            { arm_primary <- arm_secondary <- "MM: Bispecific antibody"
    } else if (cart)              {
      arm_primary <- arm_secondary <- "MM: CAR-T cell therapy"
      arm_exclude_primary <- TRUE  # CAR-T as true 1L is rare — flag for review
    } else if (mm_bucket)         { arm_primary <- arm_secondary <- "MM: Cytostatic-MM bucket"
    } else if (asct)              {
      arm_primary <- arm_secondary <- "MM: Transplant/procedure-related"
      arm_exclude_primary <- TRUE
      # Generic cytostatic in MM without any drug token: historically MP, CTD, or similar
      # older/unresolved regimens. These are real treatments — do NOT exclude.
    } else if (generic_complex)   { arm_primary <- arm_secondary <- "MM: Complex cytostatic (unresolved backbone)"
    } else if (generic_basic)     { arm_primary <- arm_secondary <- "MM: Cytostatic (unresolved generic)"
    }
    
    ### BL and LBL
    
    } else if (is_bl_lbl(cf, cc)) {
      codox_m_name <- has_any(txt, c("codox m","codox-m","codox_m"))
      hypercvad_name <- has_any(txt, c("hypercvad","hyper-cvad","hyper cvad"))
      if      (codox_m_name | (cyclo & vincr & doxo & mtx & ara_c)) {
        arm_primary <- arm_secondary <- "BL/LBL: CODOX-M/IVAC-like"
      } else if (hypercvad_name | (cyclo & doxo & vincr & mtx & ara_c)) {
        arm_primary <- arm_secondary <- "BL/LBL: HyperCVAD-like"
      } else if (explicit_rchoep | explicit_choep) {
        # DA-EPOCH-R is component-identical to R-CHOEP in coded form
        arm_primary <- arm_secondary <- if (anti_cd20) "BL/LBL: DA-EPOCH-R-like" else "BL/LBL: DA-EPOCH-like"
      } else if (explicit_rchop | (chop_name & anti_cd20)) {
        arm_primary <- arm_secondary <- "BL/LBL: R-CHOP-like (atypical — review)"
      } else if (generic_complex & anti_cd20) {
        arm_primary <- arm_secondary <- "BL/LBL: R-chemo intensive (unresolved backbone)"
      } else if (generic_complex) {
        arm_primary <- arm_secondary <- "BL/LBL: Intensive chemo (unresolved backbone)"
      }
    
    # --------------------------------------------------------------------------
    # Aggressive B-NHL (DLBCL, Richter, transformed FL)
    # --------------------------------------------------------------------------
  } else if (is_aggressive_b(cf, cc)) {
    if      (explicit_rchoep | (choep_name & anti_cd20))   { arm_primary <- arm_secondary <- "Aggressive B-NHL: R-CHOEP-like"
    } else if (explicit_choep | (choep_name & !anti_cd20)) { arm_primary <- arm_secondary <- "Aggressive B-NHL: CHOEP-like"
    } else if (explicit_rchop  | (chop_name & anti_cd20))  { arm_primary <- arm_secondary <- "Aggressive B-NHL: R-CHOP-like"
    } else if (explicit_chop   | (chop_name & !anti_cd20)) { arm_primary <- arm_secondary <- "Aggressive B-NHL: CHOP-like"
    } else if (explicit_pola_rchp)   { arm_primary <- arm_secondary <- "Aggressive B-NHL: Pola-R-CHP-like"
    } else if (pola & anti_cd20)     { arm_primary <- arm_secondary <- "Aggressive B-NHL: Polatuzumab + anti-CD20"
    } else if (pola)                 { arm_primary <- arm_secondary <- "Aggressive B-NHL: Polatuzumab-based"
    } else if (explicit_platinum_etop | mime_name | (cispl & etop) | (ifos & etop)) {
      arm_primary <- arm_secondary <- "Aggressive B-NHL: Platinum/etoposide-salvage-like"
      arm_exclude_primary <- TRUE   # salvage as 1L is suspicious — exclude from primary analysis
    } else if (explicit_cns_like)    { arm_primary <- arm_secondary <- "Aggressive B-NHL: CNS/MTX/cytarabine-intensive"
    # R-GemOx: flagged for review — very rarely used as true 1L DLBCL
    } else if (explicit_r_gemox)     {
      arm_primary <- arm_secondary <- "Aggressive B-NHL: R-GemOx-like"
      arm_exclude_primary <- TRUE
    } else if (gemci & anti_cd20)    { arm_primary <- arm_secondary <- "Aggressive B-NHL: Gemcitabine + anti-CD20"
    } else if (anti_cd20 & cyclo)    { arm_primary <- arm_secondary <- "Aggressive B-NHL: anti-CD20 + alkylator"
    } else if (anti_cd20)            { arm_primary <- arm_secondary <- "Aggressive B-NHL: anti-CD20-based other"
    } else if (rt & !generic_antineoplastic) {
      arm_primary <- arm_secondary <- "Aggressive B-NHL: Radiotherapy/procedure-only"
      arm_exclude_primary <- TRUE
      # Generic cytostatic in aggressive B-NHL: backbone is almost certainly CHOP-like.
      # Anti-CD20 co-occurrence distinguishes R-CHOP-like from CHOP-like.
      # Do NOT exclude — these are real treatments with poor source resolution.
    } else if (generic_complex & anti_cd20) { arm_primary <- arm_secondary <- "Aggressive B-NHL: R-chemo (complex, unresolved backbone)"
    } else if (generic_complex)             { arm_primary <- arm_secondary <- "Aggressive B-NHL: Chemo-complex (unresolved backbone)"
    } else if (generic_basic & anti_cd20)   { arm_primary <- arm_secondary <- "Aggressive B-NHL: R-chemo (basic, unresolved backbone)"
    } else if (generic_basic)               { arm_primary <- arm_secondary <- "Aggressive B-NHL: Cytostatic (unresolved generic)"
    }
  
    
    ### MCL
    
  } else if (is_mcl(cf, cc)) {
    if      (explicit_br)                  { arm_primary <- arm_secondary <- "MCL: BR-like"
    } else if (explicit_rchop | (chop_name & anti_cd20)) { arm_primary <- arm_secondary <- "MCL: R-CHOP-like"
    } else if (explicit_choep | (choep_name & anti_cd20)) { arm_primary <- arm_secondary <- "MCL: R-CHOEP-like"
    } else if (btk)                        { arm_primary <- arm_secondary <- "MCL: BTKi-based"
    } else if (ven & bort)                 { arm_primary <- arm_secondary <- "MCL: Venetoclax + Bortezomib-based"
    } else if (bort & anti_cd20)           { arm_primary <- arm_secondary <- "MCL: VR-CAP-like"
    } else if (bort)                       { arm_primary <- arm_secondary <- "MCL: Bortezomib-based"
    } else if (anti_cd20 & cyclo)          { arm_primary <- arm_secondary <- "MCL: anti-CD20 + alkylator"
    } else if (anti_cd20)                  { arm_primary <- arm_secondary <- "MCL: anti-CD20-based other"
    } else if (generic_complex & anti_cd20) { arm_primary <- arm_secondary <- "MCL: R-chemo (complex, unresolved backbone)"
    } else if (generic_basic | generic_complex) { arm_primary <- arm_secondary <- "MCL: Cytostatic (unresolved generic)"
    }
    
    # --------------------------------------------------------------------------
    # Indolent B-NHL (FL, EMZL, LPL, MCL, BCL)
    # --------------------------------------------------------------------------
  } else if (is_indolent_b(cf, cc)) {
    if      (explicit_br)                  { arm_primary <- arm_secondary <- "Indolent B-NHL: BR-like"
    } else if (explicit_chlor_anti_cd20)   { arm_primary <- arm_secondary <- "Indolent B-NHL: Chlorambucil + anti-CD20"
    } else if (anti_cd20 & !cyclo & !benda & !chlor) { arm_primary <- arm_secondary <- "Indolent B-NHL: anti-CD20 monotherapy"
    } else if (explicit_rchoep | (choep_name & anti_cd20)) { arm_primary <- arm_secondary <- "Indolent B-NHL: R-CHOEP-like"
    } else if (explicit_choep  | (choep_name & !anti_cd20)) { arm_primary <- arm_secondary <- "Indolent B-NHL: CHOEP-like"
    } else if (explicit_rchop  | (chop_name & anti_cd20))   { arm_primary <- arm_secondary <- "Indolent B-NHL: R-CHOP-like"
    } else if (explicit_chop   | (chop_name & !anti_cd20))  { arm_primary <- arm_secondary <- "Indolent B-NHL: CHOP-like"
    } else if (benda)   {
      arm_primary   <- "Indolent B-NHL: Bendamustine-based"
      arm_secondary <- if (anti_cd20) "Indolent B-NHL: BR-like" else "Indolent B-NHL: Bendamustine monotherapy"
    } else if (chlor)   {
      arm_primary   <- "Indolent B-NHL: Chlorambucil-based"
      arm_secondary <- if (anti_cd20) "Indolent B-NHL: Chlorambucil + anti-CD20" else "Indolent B-NHL: Chlorambucil monotherapy"
    } else if (btk)     { arm_primary <- arm_secondary <- "Indolent B-NHL: BTKi-based"
    } else if (anti_cd20) { arm_primary <- arm_secondary <- "Indolent B-NHL: anti-CD20-based other"
    } else if (cyclo)   { arm_primary <- arm_secondary <- "Indolent B-NHL: Cyclophosphamide-based"
    # Generic cytostatic in indolent B-NHL: likely BR, CVP, or COP with poor code resolution.
    # Anti-CD20 co-occurrence is meaningful even when backbone is unresolved.
    } else if (generic_complex & anti_cd20) { arm_primary <- arm_secondary <- "Indolent B-NHL: R-chemo (complex, unresolved)"
    } else if (generic_basic  & anti_cd20)  { arm_primary <- arm_secondary <- "Indolent B-NHL: R-alkylator (unresolved)"
    } else if (generic_basic | generic_complex) { arm_primary <- arm_secondary <- "Indolent B-NHL: Cytostatic (unresolved generic)"
    }
    
    # --------------------------------------------------------------------------
    # T-cell NHL (PTCL, AITL, ALCL, TCL)
    # --------------------------------------------------------------------------
  } else if (is_t_cell(cf, cc)) {
    if      (explicit_choep | choep_name)  { arm_primary <- arm_secondary <- "T-NHL: CHOEP-like"
    } else if (explicit_chop | chop_name)  { arm_primary <- arm_secondary <- "T-NHL: CHOP-like"
    } else if (bv)     { arm_primary <- arm_secondary <- "T-NHL: Brentuximab-based"
    } else if (pd1)    { arm_primary <- arm_secondary <- "T-NHL: PD1-based"
    } else if (gemci)  { arm_primary <- arm_secondary <- "T-NHL: Gemcitabine-based"
    } else if (asct | allo) {
      arm_primary   <- "T-NHL: Transplant/procedure-related"
      arm_secondary <- if (allo) "T-NHL: ALLO-related" else "T-NHL: ASCT-related"
      arm_exclude_primary <- TRUE
      # Generic cytostatic in T-NHL: almost certainly CHOP or CHOEP with unresolved components.
    } else if (generic_complex) { arm_primary <- arm_secondary <- "T-NHL: Complex chemo (unresolved, likely CHOEP-like)"
    } else if (generic_basic)   { arm_primary <- arm_secondary <- "T-NHL: Cytostatic (unresolved generic)"
    }
  }
  
  # --------------------------------------------------------------------------
  # Final fallback — only fires if no cohort branch matched OR cohort is unknown.
  # Split carefully: generic cytostatic ≠ generic antineoplastic.
  # --------------------------------------------------------------------------
  if (arm_primary == "Other/Unmapped") {
    if      (asct | allo) {
      arm_primary   <- "Procedure/Transplant-related"
      arm_secondary <- if (allo) "ALLO-related" else "ASCT-related"
      arm_exclude_primary <- TRUE
    } else if (bispec) {
      # Bispecific without cohort context — still clinically meaningful, keep
      arm_primary <- arm_secondary <- "Bispecific antibody (cohort unresolved)"
    } else if (cart) {
      arm_primary <- arm_secondary <- "CAR-T cell therapy (cohort unresolved)"
      arm_exclude_primary <- TRUE
    } else if (generic_basic | generic_complex) {
      # A real cytostatic was given; source didn't resolve it; cohort unresolved.
      # Keep — do NOT exclude. Flag clearly for downstream review.
      arm_primary <- arm_secondary <- "Cytostatic (unresolved: cohort unknown)"
    } else if (generic_antineoplastic) {
      # "other_chemo", "other immunotherapy" etc. with no specific drug token
      # AND no cohort. Truly unclassifiable — exclude.
      arm_primary <- arm_secondary <- "Generic antineoplastic bucket"
      arm_exclude_primary <- TRUE
    } else if (rt) {
      arm_primary <- arm_secondary <- "Radiotherapy-only/procedure"
      arm_exclude_primary <- TRUE
    }
  }
  
  tibble(
    firstline_codes_text      = txt_raw,
    firstline_codes_text_norm = txt,
    arm_primary               = arm_primary,
    arm_secondary             = arm_secondary,
    arm_exclude_primary       = arm_exclude_primary
  )
}


apply_treatment_arms <- function(df,
                                 treatment_col     = "firstline_codes_text",
                                 cohort_fine_col   = "cohort_fine",
                                 cohort_coarse_col = "cohort_coarse") {
  stopifnot(
    is.data.frame(df),
    treatment_col     %in% names(df),
    cohort_fine_col   %in% names(df),
    cohort_coarse_col %in% names(df)
  )
  pmap_dfr(
    list(df[[treatment_col]], df[[cohort_fine_col]], df[[cohort_coarse_col]]),
    assign_arms_one
  )
}

firstline_tbl_mapped <- firstline_tbl %>%
  bind_cols(apply_treatment_arms(firstline_tbl))
fl_count("mapped_before_arm_rescue", firstline_tbl_mapped, unit = "patients")



# ==============================================================================
# 8b. ARM RESCUE — insert at end of section 8, after apply_treatment_arms()
#     and after firstline_tbl_mapped is produced from Pass 1 + Pass 2 patients.
#
# For patients in arm_rescue_ids, we have a treatment but no confirmed heme dx.
# We infer cohort from the arm label that assign_arms_one() already assigned.
# These are included in firstline_tbl_mapped but clearly flagged.
# ==============================================================================

if (length(arm_rescue_ids) > 0) {
  
  cat(sprintf("\n=== ARM RESCUE: %d patients ===\n", length(arm_rescue_ids)))
  
  # These patients are in the original firstline_tbl (before filtering to !is.na(cohort_fine))
  # Re-fetch their treatment records from firstline_tbl (pre-cohort-filter) if you
  # have kept a copy, or from the raw make_firstline() output. Here we assume
  # firstline_tbl_prefilter was saved; if not, re-run make_firstline() for these IDs.
  #
  # Simplest approach: keep a pre-filter copy of firstline_tbl before step 7.
  # Add this line at the very start of section 7 (before the filter):
  #   firstline_tbl_all <- firstline_tbl   # full output including NA-cohort rows
  #
  # Then here:
  rescue_base <- firstline_tbl_all %>%           # <── requires firstline_tbl_all
    mutate(patientid = patient_id) %>% 
    filter(
      patient_id %in% arm_rescue_ids |
        patientid  %in% arm_rescue_ids             # handles either column name
    ) %>%
    # Ensure we have the cohort columns (will be NA)
    mutate(
      cohort_fine   = NA_character_,
      cohort_coarse = NA_character_
    )
  
  # Apply arm mapping with no cohort context
  rescue_arms <- rescue_base %>%
    bind_cols(apply_treatment_arms(.))
  
  # Infer cohort_grouped from arm label prefix
  firstline_tbl_rescue <- rescue_arms %>%
    mutate(
      cohort_grouped = case_when(
        str_detect(arm_primary, "^MM:")                          ~ "MM",
        str_detect(arm_primary, "^CLL:")                         ~ "CLL_SLL",
        str_detect(arm_primary, "^HCL:")                         ~ "HCL",
        str_detect(arm_primary, "^HL:")                          ~ "HL",
        str_detect(arm_primary, "^MCL:")                         ~ "MCL",
        str_detect(arm_primary, "^Aggressive B-NHL:|^BL/LBL:")  ~ "DLBCL_like",
        str_detect(arm_primary, "^Indolent B-NHL:")              ~ "Indolent_B_NHL",
        str_detect(arm_primary, "^T-NHL:")                       ~ "T_NHL",
        TRUE                                                     ~ NA_character_
      ),
      cohort_fine   = cohort_grouped,
      cohort_coarse = case_when(
        cohort_grouped == "MM"                                    ~ "PLASMA_CELL",
        cohort_grouped == "T_NHL"                                 ~ "T_CELL",
        cohort_grouped == "HL"                                    ~ "HL",
        !is.na(cohort_grouped)                                    ~ "B_CELL",
        TRUE                                                      ~ "UNKNOWN"
      ),
      cohort_source = "arm_rescue"
    ) %>%
    filter(!is.na(cohort_grouped))   # truly unmappable patients are excluded
  
  n_rescued     <- nrow(firstline_tbl_rescue)
  fl_count("arm_rescued_cohort", firstline_tbl_rescue,
           branch = "arm_rescue", unit = "patients",
           note = "Fine identity inferred from arm; may be only grouped resolution")
  n_unrescuable <- length(arm_rescue_ids) - n_rescued
  
  cat(sprintf(
    "  Rescued via arm inference: %d  |  unrescuable (Other/Unmapped): %d\n",
    n_rescued, n_unrescuable
  ))
  
  if (n_rescued > 0) {
    cat("  Arm rescue cohort distribution:\n")
    count(firstline_tbl_rescue, cohort_grouped, sort = TRUE) %>% print()
    
    # Append to the mapped table
    firstline_tbl_mapped <- bind_rows(firstline_tbl_mapped, firstline_tbl_rescue)
  }
  
} else {
  cat("\nNo arm rescue needed\n")
}

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

cat("\n=== Final cohort_source breakdown ===\n")
fl_count("mapped_after_arm_rescue", firstline_tbl_mapped, unit = "patients")
firstline_tbl_mapped %>%
  count(cohort_source, cohort_grouped) %>%
  tidyr::pivot_wider(names_from = cohort_source, values_from = n, values_fill = 0L) %>%
  arrange(cohort_grouped) %>%
  print(n = 40)

cat("\n=== Overall coverage by pass ===\n")
firstline_tbl_mapped %>%
  count(cohort_source) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  print()



# ==============================================================================
# 9. One-hot treatment encoding (full episode)
# ==============================================================================

clean_token  <- function(x) str_to_lower(str_trim(x))
clean_colname <- function(x) {
  x %>% str_to_lower() %>%
    str_replace_all("[^a-z0-9]+", "_") %>%
    str_replace_all("^_|_$", "")
}

regimen_map <- list(
  abvd      = c("doxorubicin","bleomycin","vinblastine","dacarbazine"),
  choep     = c("cyclophosphamide","doxorubicin","vincristine","etoposide","prednisone_prednisolone"),
  chop      = c("cyclophosphamide","doxorubicin","vincristine","prednisone_prednisolone"),
  cop       = c("cyclophosphamide","vincristine","prednisone_prednisolone"),
  beacopp_like = c("bleomycin","cyclophosphamide","dacarbazine","doxorubicin","etoposide","procarbazine"),
  beam      = c("carmustine","etoposide","cytarabine","melphalan"),
  codox_m   = c("cyclophosphamide","vincristine","doxorubicin","methotrexate","cytarabine"),
  `codox m` = c("cyclophosphamide","vincristine","doxorubicin","methotrexate","cytarabine"),
  dhap      = c("dexamethasone","cytarabine","cisplatin"),
  epoch     = c("etoposide","prednisone_prednisolone","vincristine","cyclophosphamide","doxorubicin"),
  eshap     = c("etoposide","methylprednisolone","cytarabine","cisplatin"),
  gdp       = c("gemcitabine","cisplatin","dexamethasone"),
  hypercvad = c("cyclophosphamide","doxorubicin","vincristine","dexamethasone","methotrexate","cytarabine"),
  ice       = c("ifosfamide","carboplatin","etoposide"),
  mime      = c("ifosfamide","etoposide","methotrexate"),
  pvag      = c("prednisone_prednisolone","vinblastine","doxorubicin","gemcitabine")
)

expand_codes <- function(code_vec, regimen_map, aliases = NULL) {
  code_vec <- clean_token(code_vec)
  code_vec <- code_vec[!is.na(code_vec) & code_vec != ""]
  if (!is.null(aliases))
    code_vec <- ifelse(code_vec %in% names(aliases), aliases[code_vec], code_vec)
  expanded <- code_vec
  for (cd in code_vec)
    if (cd %in% names(regimen_map)) expanded <- c(expanded, regimen_map[[cd]])
  unique(expanded)
}

build_onehot <- function(df, text_col, prefix) {
  split_codes   <- str_split(df[[text_col]] %>% replace_na(""), fixed("|"))
  expanded_list <- map(split_codes, expand_codes, regimen_map = regimen_map)
  all_features  <- expanded_list %>% flatten_chr() %>% unique() %>% sort()
  mat <- sapply(all_features, function(f) as.integer(map_lgl(expanded_list, ~ f %in% .x)))
  colnames(mat) <- paste0(prefix, clean_colname(colnames(mat)))
  as.data.frame(mat)
}

onehot_episode <- build_onehot(firstline_tbl_mapped, "firstline_codes_text_norm", "tx_")
df_expanded    <- bind_cols(firstline_tbl_mapped, onehot_episode)


# ==============================================================================
# 10. Day-1 treatment covariate
#     (avoids look-ahead leakage: only codes dispensed on the exact index date)
# ==============================================================================

day1_codes <- treatments %>%
  transmute(
    patientid  = patientid,
    date_start = as.Date(date_start),
    code       = if_else(is.na(tx_code), tx_text, tx_code),
    code_text  = as.character(tx_text),
    dx_text    = as.character(dx_text),
    dx_code    = as.character(dx_code)
  ) %>%
  mutate(
    dx_text_norm  = norm_da(dx_text),
    dx_prefix4    = str_sub(dx_code, 1, 4),
    ind_heme_icd  = !is.na(dx_code) & dx_prefix4 %in% HEME_ICD10_PREFIXES,
    ind_excl_icd  = !is.na(dx_code) & dx_prefix4 %in% EXCL_ICD10_PREFIX,
    ind_excl_text = !is.na(dx_text_norm) & str_detect(dx_text_norm, EXCL_TEXT_PATTERN_RE),
    ind_ok = case_when(
      ind_heme_icd                  ~ TRUE,
      ind_excl_icd                  ~ FALSE,
      ind_excl_text                 ~ FALSE,
      TRUE                          ~ TRUE
    )
  ) %>%
  filter(ind_ok) %>%
  inner_join(
    firstline_tbl_mapped %>%
      select(patientid, firstline_date) %>%
      mutate(firstline_date = as.Date(firstline_date)),
    by = "patientid"
  ) %>%
  filter(date_start == firstline_date) %>%
  group_by(patientid, firstline_date) %>%
  summarise(
    day1_codes_text = paste(sort(unique(code_text[!is.na(code_text)])), collapse = " | "),
    day1_n_codes    = n_distinct(code),
    .groups = "drop"
  ) %>%
  mutate(day1_codes_text = if_else(day1_codes_text == "", NA_character_, day1_codes_text))

firstline_tbl_mapped <- firstline_tbl_mapped %>%
  mutate(firstline_date = as.Date(firstline_date)) %>%
  left_join(day1_codes, by = c("patientid","firstline_date"))

# Apply arm mapping to day-1 codes
has_day1  <- !is.na(firstline_tbl_mapped$day1_codes_text)
day1_arms <- firstline_tbl_mapped %>%
  filter(has_day1) %>%
  { pmap_dfr(list(.$day1_codes_text, .$cohort_fine, .$cohort_coarse), assign_arms_one) } %>%
  rename(
    day1_codes_text_norm = firstline_codes_text_norm,
    arm_day1_primary     = arm_primary,
    arm_day1_secondary   = arm_secondary,
    arm_day1_exclude     = arm_exclude_primary
  ) %>%
  select(-firstline_codes_text)

no_day1_stub <- tibble(
  .row                 = which(!has_day1),
  day1_codes_text_norm = NA_character_,
  arm_day1_primary     = NA_character_,
  arm_day1_secondary   = NA_character_,
  arm_day1_exclude     = NA
)
firstline_tbl_mapped <- firstline_tbl_mapped %>%
  mutate(.row = row_number()) %>%
  { bind_rows(bind_cols(filter(., has_day1), day1_arms),
              bind_cols(filter(., !has_day1), no_day1_stub)) } %>%
  arrange(.row) %>%
  select(-.row)

cat("\n=== Day-1 arm coverage ===\n")
firstline_tbl_mapped %>%
  summarise(
    n_total          = n(),
    n_has_day1_codes = sum(!is.na(day1_codes_text)),
    n_arm_mapped     = sum(!is.na(arm_day1_primary) & arm_day1_primary != "Other/Unmapped"),
    n_arm_unmapped   = sum(arm_day1_primary == "Other/Unmapped", na.rm = TRUE),
    n_arm_exclude    = sum(arm_day1_exclude == TRUE,             na.rm = TRUE)
  ) %>%
  print()


# ==============================================================================
# 11. Final outputs
# ==============================================================================

# One-hot for day-1 codes
onehot_day1 <- build_onehot(firstline_tbl_mapped, "day1_codes_text_norm", "day1_tx_")
df_expanded <- bind_cols(firstline_tbl_mapped, onehot_day1)

# get ready for export as covariates

column_overview <- colnames(df_expanded)

list_of_irrelevant_columns <- c(                                             
                                "firstline_date",                                            
                                "firstline_end"  ,                                            "firstline_episode_id",                                      
                                "firstline_codes" ,                                           "firstline_codes_text...8",                                  
                                "firstline_n_dates",                                          "firstline_n_codes"        ,                                 
                                "firstline_has_anchor",                                       "firstline_has_rkkp_source" ,                                
                                "firstline_has_prophy_suspect",                               "firstline_therapy_like"    ,                                
                                "firstline_indication",                                       "firstline_indication_codes" ,                               
                                "reason",                                                     ".row...35",
                                "firstline_codes_text...23",                                  "firstline_codes_text_norm"   ,                              
                                "arm_primary",                                                "arm_secondary"                ,                             
                                "arm_exclude_primary",                                        "day1_codes_text"               ,                            
                                "day1_n_codes",                                               "day1_codes_text_norm"           ,                           
                                "arm_day1_exclude",                                           ".row...34" )

relevant_columns <- c(column_overview[!column_overview %in% list_of_irrelevant_columns])

df_export <- df_expanded %>% select(all_of(relevant_columns))

cat("\n=== Treatment arm distribution ===\n")
firstline_tbl_mapped %>%
  filter(!arm_exclude_primary) %>%
  count(cohort_grouped, arm_primary, sort = TRUE) %>%
  print(n = 40)

check <- firstline_tbl_mapped %>%
  filter(!arm_exclude_primary) %>% 
  group_by(cohort_fine) %>%
  summarize(n = n())

check <- firstline_tbl_mapped %>%
  filter(!arm_exclude_primary) %>% 
  group_by(arm_primary) %>%
  summarize(n = n())

# firstline_tbl_mapped and df_expanded are now in the environment
# for use by downstream scripts (e.g. hypertension_check.R, OPERA feature tables)


### CHECK DATES
# 1. Distribution of days from first (cohort-specific) diagnosis to firstline_date
# Red flag: large numbers with negative or very large positive gaps
checking <- firstline_tbl %>%
  #left_join(cohorts, by = "patientid") %>%
  mutate(days_dx_to_tx = as.numeric(firstline_date - dx_date)) %>%
  group_by(cohort_fine) %>%
  summarise(
    n = n(),
    median_days = median(days_dx_to_tx, na.rm = TRUE),
    pct_before_dx  = mean(days_dx_to_tx < 0,   na.rm = TRUE),
    pct_over_1yr   = mean(days_dx_to_tx > 365,  na.rm = TRUE),
    pct_over_3yr   = mean(days_dx_to_tx > 1095, na.rm = TRUE)
  )

# 2. For CLL specifically: what fraction have RKKP-confirmed firstline_date
# vs only inferred from LPR/epikur? RKKP agreement is your gold standard.
firstline_tbl %>%
  #mutate(patientid = patient_id) %>%
  #left_join(cohorts, by = "patientid") %>%
  filter(cohort_fine %in% c("CLL","SLL")) %>%
  mutate(has_anchor = firstline_has_anchor) %>%
  count(has_anchor)

# 3. Cases where firstline is a single drug with no anchor code
# These are the most suspicious
firstline_tbl %>%
  filter(!firstline_has_anchor, firstline_n_codes == 1) %>%
  count(firstline_codes_text, sort = TRUE) %>%
  head(20)

fl_count("before_secondary_cohort_exclusion", firstline_tbl_mapped, unit = "patients")
firstline_tbl_mapped <- firstline_tbl_mapped %>% filter(cohort_grouped != "EXCLUDE_SECONDARY")
fl_count("after_secondary_cohort_exclusion", firstline_tbl_mapped, unit = "patients")
df_expanded <- df_expanded %>% filter(cohort_grouped != "EXCLUDE_SECONDARY")
fl_count("mapped_csv_export_population", firstline_tbl_mapped, unit = "patients",
         note = "Would be mapped CSV export; counts-only mode writes no production CSV")
fl_write_counts()
if (!is.null(.fl_audit$unclassified_flags)) {
  count_root <- Sys.getenv("OPERA_FIRSTLINE_COUNT_DIR", "firstline_pipeline_counts_LOCAL_ONLY")
  utils::write.csv(.fl_audit$unclassified_flags,
                   file.path(count_root, .fl_audit$run_id,
                             "unclassified_diagnostic_flags_INTERNAL.csv"),
                   row.names = FALSE)
}
if (identical(Sys.getenv("OPERA_FIRSTLINE_COUNTS_ONLY", "0"), "1")) {
  stop(structure(
    list(message = "First-line counts complete; skipped later exploratory and production exports",
         call = NULL),
    class = c("opera_firstline_counts_complete", "error", "condition")
  ))
}

check_cohorts <- firstline_tbl_mapped %>%
  group_by(cohort_fine, cohort_grouped) %>%
  summarize(n = n())

### AGREEMENT WITH ANDREAS


# Independent inference: NO seed prioritization
firstline_noseed <- make_firstline(tx_flagged, dx_index, seed_tbl = NULL, pre_dx_grace_days = 30,
                                   seed_max_gap = 30, accept_cd20_mono = TRUE)

# True agreement: inferred-only date vs clinical seed, on seeded patients
seed_agreement <- firstline_noseed %>%
  select(patient_id, inferred_date = firstline_date) %>%
  inner_join(seed_firstline %>% select(patient_id, seed_date), by = "patient_id") %>%
  filter(!is.na(inferred_date), !is.na(seed_date)) %>%
  mutate(day_diff = as.numeric(inferred_date - seed_date))


seed_agreement <- firstline_tbl %>%
  filter(!is.na(seed_date), !is.na(firstline_date)) %>%
  mutate(day_diff = as.numeric(firstline_date - seed_date))

seed_agreement %>%
  summarise(
    n          = n(),
    exact      = mean(day_diff == 0),
    within_7   = mean(abs(day_diff) <= 7),
    within_30  = mean(abs(day_diff) <= 30),
    median_gap = median(day_diff),
    p05        = quantile(day_diff, 0.05),
    p95        = quantile(day_diff, 0.95)
  )


firstline_noseed %>%
  select(-seed_date) %>% 
  inner_join(seed_firstline, by = "patient_id") %>%
  mutate(day_diff = as.numeric(firstline_date - seed_date)) %>%
  filter(day_diff < -7) %>%
  mutate(kept_by = case_when(
    firstline_has_rkkp_source     ~ "rkkp",
    firstline_has_anchor          ~ "anchor",
    firstline_has_heme_indication ~ "heme_icd",
    firstline_has_prophy_suspect  ~ "prophy_branch",
    TRUE                          ~ "fallback"
  )) %>%
  count(seed_disease, kept_by, sort = TRUE)

ggplot(seed_agreement, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-7, 0, 7), linetype = "dashed", colour = "red") +
  labs(
    title    = "Inferred episode_start minus clinical seed (days)",
    subtitle = sprintf("n = %d seeded patients (CLL + MM)", nrow(seed_agreement)),
    x = "firstline_date - seed_date (days)", y = "N patients"
  )


ggplot(seed_agreement, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(
    aes(y = after_stat(count / sum(count)) * 100), 
    binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-7, 0, 7), linetype = "dashed", colour = "red") +
  scale_y_continuous(breaks = seq(0, 100, by = 10)) + 
  labs(
    title    = "Inferred episode_start minus clinical seed (days)",
    subtitle = sprintf("n = %d seeded patients (CLL + MM)", nrow(seed_agreement)),
    x = "firstline_date - seed_date (days)", y = "Percentage of patients"
  )

### DIAGNOSE PROBLEMS 

disagree <- firstline_noseed %>%
  transmute(
    patient_id, dx_date, diagnosis,
    inferred_date  = firstline_date,
    inf_codes_text = firstline_codes_text,
    inf_ind_codes  = firstline_indication_codes,
    inf_has_prophy = firstline_has_prophy_suspect,
    inf_n_codes    = firstline_n_codes
  ) %>%
  inner_join(seed_firstline, by = "patient_id") %>%
  mutate(day_diff = as.numeric(inferred_date - seed_date)) %>%
  filter(day_diff < -7)

# (1) How early is early? weeks vs months vs years separates the mechanisms
disagree %>%
  mutate(bucket = cut(day_diff,
                      c(-Inf, -1825, -730, -365, -180, -90, -30, -7),
                      labels = c("5y+","2-5y","1-2y","6-12m","3-6m","1-3m","7-30d"))) %>%
  count(seed_disease, bucket) %>%
  tidyr::pivot_wider(names_from = bucket, values_from = n, values_fill = 0)

# (2) Different-disease check: does the inferred episode's indication match the seed disease?
disagree %>%
  mutate(
    inf_matches_seed = case_when(
      seed_disease == "CLL" & str_detect(inf_ind_codes, "DC911") ~ TRUE,
      seed_disease == "MM"  & str_detect(inf_ind_codes, "DC900") ~ TRUE,   # adjust to your ICD10.MM
      TRUE ~ FALSE
    )
  ) %>%
  count(seed_disease, inf_matches_seed)

# (3) Prophylaxis / pre-phase share, and what the early episode actually is
disagree %>% summarise(pct_prophy = mean(inf_has_prophy), median_ncodes = median(inf_n_codes))
disagree %>% count(inf_codes_text, sort = TRUE) %>% head(20)

firstline_noseed %>%
  select(-seed_date) %>% 
  right_join(seed_firstline %>% select(patient_id, seed_disease, seed_date), by = "patient_id") %>%
  mutate(day_diff = as.numeric(firstline_date - seed_date)) %>%
  summarise(
    n_seeded  = n(),
    n_exact   = sum(day_diff == 0, na.rm = TRUE),
    n_within7 = sum(abs(day_diff) <= 7, na.rm = TRUE),
    n_early   = sum(day_diff < -7, na.rm = TRUE),
    n_late    = sum(day_diff >  7, na.rm = TRUE),
    n_missing = sum(is.na(firstline_date)),
    .by = seed_disease
  )

firstline_noseed %>%
  select(-seed_date) %>%
  inner_join(seed_firstline, by = "patient_id") %>%
  mutate(day_diff = as.numeric(firstline_date - seed_date)) %>%
  filter(is.na(firstline_date) | day_diff > 7) %>%
  count(seed_disease, firstline_therapy_like, sort = TRUE)

firstline_noseed %>%
  inner_join(seed_firstline, by = "patient_id") %>%
  filter(is.na(firstline_date)) %>%
  left_join(tx_any, by = c("patient_id" = "patient_id")) %>%   # tx_any already in your script §5
  summarise(
    n              = n(),
    n_no_tx_at_all = sum(is.na(n_tx_total) | n_tx_total == 0),
    n_tx_filtered  = sum(n_tx_total > 0 & (is.na(n_tx_ok) | n_tx_ok == 0), na.rm = TRUE)
  )

firstline_noseed %>%
  select(-seed_date) %>% 
  inner_join(seed_firstline, by = "patient_id") %>%
  mutate(day_diff = as.numeric(firstline_date - seed_date)) %>%
  filter(day_diff > 7, firstline_therapy_like) %>%
  mutate(bucket = cut(day_diff, c(7, 30, 90, 365, Inf),
                      labels = c("8-30d","1-3m","3-12m","1y+"))) %>%
  count(bucket)

missing_ids <- firstline_noseed %>%
  select(-seed_date) %>% 
  inner_join(seed_firstline, by = "patient_id") %>%
  filter(is.na(firstline_date)) %>% pull(patient_id)

# do they survive ind_ok, and do they exist in the dx index?
tx_flagged %>%
  filter(patient_id %in% missing_ids) %>%
  summarise(n_pat = n_distinct(patient_id),
            n_pat_ind_ok = n_distinct(patient_id[ind_ok]))

dx_index %>% filter(patient_id %in% missing_ids) %>% summarise(n = n_distinct(patient_id))

# for those in BOTH tx and dx, is their treatment outside the window?
tx_flagged %>%
  filter(patient_id %in% missing_ids, ind_ok) %>%
  inner_join(dx_index, by = "patient_id") %>%
  inner_join(seed_firstline, by = "patient_id") %>%
  mutate(tx_vs_dx   = as.numeric(as.Date(tx_date) - as.Date(dx_date)),
         tx_vs_seed = as.numeric(as.Date(tx_date) - seed_date)) %>%
  group_by(patient_id) %>%
  summarise(has_tx_in_window = any(tx_vs_dx >= -14 & tx_vs_dx <= 3650),
            min_tx_vs_dx = min(tx_vs_dx), .groups = "drop") %>%
  count(has_tx_in_window)


firstline_noseed %>%
  select(-seed_date) %>%
  inner_join(seed_firstline, by = "patient_id") %>%
  mutate(day_diff = as.numeric(firstline_date - seed_date)) %>%
  filter(day_diff > 365, firstline_therapy_like) %>%
  select(patient_id, seed_date, firstline_date, firstline_codes_text, firstline_indication_codes) %>%
  head(20)

tx_flagged %>%
  filter(patient_id %in% missing_ids, ind_ok) %>%
  inner_join(dx_index, by = "patient_id") %>%
  filter(as.numeric(as.Date(tx_date) - as.Date(dx_date)) >= -14,
         as.numeric(as.Date(tx_date) - as.Date(dx_date)) <= 3650) %>%
  count(code_text, sort = TRUE) %>% head(20)

tx_flagged %>%
  filter(patient_id %in% missing_ids, ind_ok) %>%
  inner_join(dx_index, by = "patient_id") %>%
  group_by(patient_id) %>%
  summarise(min_tx_vs_dx = min(as.numeric(as.Date(tx_date) - as.Date(dx_date))), .groups = "drop") %>%
  filter(min_tx_vs_dx < 0) %>%
  summarise(
    within_30  = mean(min_tx_vs_dx >= -30),
    within_90  = mean(min_tx_vs_dx >= -90),
    within_180 = mean(min_tx_vs_dx >= -180),
    p10 = quantile(min_tx_vs_dx, 0.10),
    p05 = quantile(min_tx_vs_dx, 0.05)
  )


audit_seed_agreement <- function(fl, seeds = seed_firstline, label = "") {
  fl %>%
    select(patient_id, firstline_date) %>%
    right_join(seeds %>% select(patient_id, seed_disease, seed_date), by = "patient_id") %>%
    mutate(day_diff = as.numeric(firstline_date - seed_date)) %>%
    summarise(
      run         = label,
      n_seeded    = n(),
      pct_exact   = mean(day_diff == 0, na.rm = TRUE),
      pct_within7 = mean(abs(day_diff) <= 7, na.rm = TRUE),
      n_early     = sum(day_diff < -7, na.rm = TRUE),
      n_late      = sum(day_diff >  7, na.rm = TRUE),
      n_missing   = sum(is.na(firstline_date)),
      .by = seed_disease
    )
}


firstline_noseed <- make_firstline(tx_flagged, dx_index, seed_tbl = NULL,
                                   pre_dx_grace_days = 14, accept_cd20_mono = FALSE)

missing_ids <- firstline_noseed %>%
  inner_join(seed_firstline, by = "patient_id") %>%
  filter(is.na(firstline_date)) %>% pull(patient_id)

# Composition of the deep tail (>180d before diagnosis): what disease was it really?
firstline_noseed %>%   # dummy to keep pipe chain readable; real query below
  { tx_flagged } %>%
  filter(patient_id %in% missing_ids, ind_ok) %>%
  inner_join(dx_index, by = "patient_id") %>%
  mutate(tx_vs_dx = as.numeric(as.Date(tx_date) - as.Date(dx_date))) %>%
  filter(tx_vs_dx < -180) %>%
  count(code_text, dx_text, sort = TRUE) %>%
  head(30)


firstline_tbl %>%
  filter(!is.na(seed_date)) %>%
  summarise(n = n(),
            n_dateonly_lost = sum(is.na(firstline_date)),
            n_full          = sum(!is.na(firstline_date)))

grace_sweep <- purrr::map_dfr(
  c(14, 30, 60, 90),
  ~ make_firstline(tx_flagged, dx_index, seed_tbl = NULL,
                   pre_dx_grace_days = .x, accept_cd20_mono = FALSE) %>%
    audit_seed_agreement(label = paste0("grace_", .x))
) %>% arrange(seed_disease, run)

print(grace_sweep)

GRACE = 30

# --- Benefit: seed-off agreement, CD20 off vs on (CLL/MM ground truth) ---
fl_off <- make_firstline(tx_flagged, dx_index, seed_tbl = NULL,
                         pre_dx_grace_days = GRACE, accept_cd20_mono = FALSE)
fl_on  <- make_firstline(tx_flagged, dx_index, seed_tbl = NULL,
                         pre_dx_grace_days = GRACE, accept_cd20_mono = TRUE)

cd20_benefit <- bind_rows(
  audit_seed_agreement(fl_off, label = "cd20_off"),
  audit_seed_agreement(fl_on,  label = "cd20_on")
) %>% arrange(seed_disease, run)

print(cd20_benefit)

# --- Risk: production run (seeded), CD20 off vs on, UNSEEDED diseases only ---
# CLL/MM are seed-overridden in production, so CD20 only changes the rest.
fl_prod_off <- make_firstline(tx_flagged, dx_index, seed_tbl = seed_firstline,
                              pre_dx_grace_days = GRACE, accept_cd20_mono = FALSE)
fl_prod_on  <- make_firstline(tx_flagged, dx_index, seed_tbl = seed_firstline,
                              pre_dx_grace_days = GRACE, accept_cd20_mono = TRUE)

# NOTE: point cohort_lookup at your real cohort_fine table (patient_id, cohort_fine)
coverage_by_cohort <- function(fl, lbl) {
  fl %>%
    select(patient_id, firstline_date) %>%
    left_join(firstline_tbl_mapped %>% mutate(patient_id = patientid) %>% select(patient_id, cohort_fine), by = "patient_id") %>%
    group_by(cohort_fine) %>%
    summarise(!!lbl := mean(!is.na(firstline_date)),
              n = n_distinct(patient_id), .groups = "drop")
}

cd20_blast_radius <- coverage_by_cohort(fl_prod_off, "cov_off") %>%
  left_join(coverage_by_cohort(fl_prod_on, "cov_on") %>% select(-n), by = "cohort_fine") %>%
  mutate(delta_cov = cov_on - cov_off) %>%
  arrange(desc(delta_cov))

print(cd20_blast_radius, n =25)


cohort_lookup <- firstline_tbl_mapped %>%          # or wherever cohort_fine actually lives
  select(patient_id = patientid, cohort_fine) %>%
  distinct(patient_id, .keep_all = TRUE)

coverage_by_cohort <- function(fl, lbl) {
  fl %>%
    distinct(patient_id, firstline_date) %>%
    left_join(cohort_lookup, by = "patient_id") %>%
    group_by(cohort_fine) %>%
    summarise(!!lbl := mean(!is.na(firstline_date)),
              n = n_distinct(patient_id), .groups = "drop")
}

cd20_blast_radius <- coverage_by_cohort(fl_prod_off, "cov_off") %>%
  left_join(coverage_by_cohort(fl_prod_on, "cov_on") %>% select(-n), by = "cohort_fine") %>%
  mutate(delta_cov = cov_on - cov_off) %>%
  filter(!is.na(cohort_fine)) %>%          # drop the no-cohort bucket from the view
  arrange(desc(delta_cov))
print(cd20_blast_radius, n = 40)

# --- Risk detail: what do the newly-caught unseeded episodes look like? ---
new_ids <- setdiff(
  fl_prod_on  %>% filter(!is.na(firstline_date)) %>% pull(patient_id),
  fl_prod_off %>% filter(!is.na(firstline_date)) %>% pull(patient_id)
)

# 0. Did you actually build two different objects? (this has bitten us before)
identical(fl_prod_on, fl_prod_off)   # expect FALSE; if TRUE, one wasn't rebuilt with the toggle

# 1. Per-patient date diff — catches date SHIFTS, which coverage misses
prod_cmp <- fl_prod_off %>%
  distinct(patient_id, date_off = firstline_date) %>%
  full_join(fl_prod_on %>% distinct(patient_id, date_on = firstline_date),
            by = "patient_id") %>%
  mutate(changed = xor(is.na(date_off), is.na(date_on)) |
           (!is.na(date_off) & !is.na(date_on) & date_off != date_on))

prod_cmp %>% summarise(n = n(), n_changed = sum(changed))

# 2. Split the changed patients: seeded (should be 0) vs unseeded (the real answer)
prod_cmp %>%
  filter(changed) %>%
  mutate(seeded = patient_id %in% seed_firstline$patient_id) %>%
  left_join(cohort_lookup, by = "patient_id") %>%
  count(seeded, cohort_fine, sort = TRUE)

changed_unseeded <- prod_cmp %>% filter(changed,
                                        !patient_id %in% seed_firstline$patient_id) %>% pull(patient_id)

fl_prod_on %>%
  filter(patient_id %in% changed_unseeded) %>%
  left_join(cohort_lookup, by = "patient_id") %>%
  mutate(days_dx_to_fl = as.numeric(firstline_date - dx_date)) %>%
  summarise(n = n(), median_days = median(days_dx_to_fl, na.rm = TRUE),
            pct_single_code = mean(firstline_n_codes == 1, na.rm = TRUE),
            pct_very_early  = mean(days_dx_to_fl < 14, na.rm = TRUE),
            .by = cohort_fine) %>% arrange(desc(n))

fl_prod_on %>%
  filter(patient_id %in% changed_unseeded) %>%
  left_join(cohort_lookup, by = "patient_id") %>%
  filter(cohort_fine %in% c("HCL", NA) | is.na(cohort_fine)) %>%
  count(cohort_fine, firstline_has_heme_indication, firstline_n_codes == 1)

fl_prod_on %>%
  filter(patient_id %in% new_ids) %>%
  left_join(firstline_tbl_mapped %>% mutate(patient_id = patientid) %>% select(patient_id, cohort_fine), by = "patient_id") %>%
  filter(!cohort_fine %in% c("CLL","MM")) %>%
  mutate(days_dx_to_fl = as.numeric(firstline_date - dx_date)) %>%
  summarise(
    n_new            = n(),
    median_days      = median(days_dx_to_fl, na.rm = TRUE),
    pct_single_code  = mean(firstline_n_codes == 1, na.rm = TRUE),
    pct_prophy       = mean(firstline_has_prophy_suspect, na.rm = TRUE),
    pct_very_early   = mean(days_dx_to_fl < 14, na.rm = TRUE),   # suspicious: at/just after dx
    .by = cohort_fine
  ) %>% arrange(desc(n_new))

# ============ FROZEN CONFIG ============
GRACE        <- 30      # from Check 2
ACCEPT_CD20  <- FALSE   # from Check 3
SEED_MAX_GAP <- 30
TRUST_RKKP   <- TRUE
# =======================================

firstline_tbl     <- make_firstline(tx_flagged, dx_index, seed_tbl = seed_firstline,
                                    pre_dx_grace_days = GRACE, seed_max_gap = SEED_MAX_GAP,
                                    accept_cd20_mono = ACCEPT_CD20)
firstline_noseed  <- make_firstline(tx_flagged, dx_index, seed_tbl = NULL,
                                    pre_dx_grace_days = GRACE, accept_cd20_mono = ACCEPT_CD20)
final_audit       <- audit_seed_agreement(firstline_noseed, label = "FROZEN")

cfg <- list(GRACE = GRACE, ACCEPT_CD20 = ACCEPT_CD20, SEED_MAX_GAP = SEED_MAX_GAP,
            TRUST_RKKP = TRUST_RKKP, frozen_at = Sys.time())

saveRDS(firstline_tbl,    "firstline_tbl_FROZEN.rds")
saveRDS(firstline_noseed, "firstline_noseed_FROZEN.rds")
saveRDS(final_audit,      "firstline_audit_FROZEN.rds")
saveRDS(cfg,              "firstline_config_FROZEN.rds")
print(final_audit)

### 

write_csv(df_expanded, "df_expanded_updated.csv")
write_csv(firstline_tbl_mapped, "firstline_tbl_mapped_updated.csv")
fl_write_counts()

### check with SP


### CHECK WITH RKKP

load_dataset("RKKP_LYFO_CLEAN")

checking <- RKKP_LYFO_CLEAN %>% select(patientid, date_treatment_1st_line, hospital_id) %>% left_join(firstline_tbl)

checking_LYFO <- checking

checking <- checking %>%
  mutate(
    firstline_date          = as.Date(firstline_date),
    date_treatment_1st_line = as.Date(date_treatment_1st_line),
    has_ours   = !is.na(firstline_date),
    has_rkkp   = !is.na(date_treatment_1st_line),
    day_diff   = as.numeric(firstline_date - date_treatment_1st_line)
  )

# ==============================================================================
# 1. Coverage comparison
# ==============================================================================
checking %>%
  count(has_ours, has_rkkp) %>%
  mutate(label = case_when(
    has_ours &  has_rkkp ~ "Both found",
    has_ours & !has_rkkp ~ "We find, RKKP missing",
    !has_ours &  has_rkkp ~ "RKKP finds, we miss",
    TRUE                  ~ "Neither found"
  )) %>%
  select(label, n) %>%
  mutate(pct = round(100 * n / sum(n), 1))

# ==============================================================================
# 2. Date mismatch distribution (patients where both have a date)
# ==============================================================================
both <- checking %>% filter(has_ours, has_rkkp)

cat(sprintf("N with both dates: %d\n", nrow(both)))

both %>%
  summarise(
    median_diff_days = median(day_diff),
    mean_diff_days   = mean(day_diff),
    sd_diff_days     = sd(day_diff),
    pct_exact        = mean(day_diff == 0),
    pct_within_7d    = mean(abs(day_diff) <= 7),
    pct_within_30d   = mean(abs(day_diff) <= 30),
    pct_within_90d   = mean(abs(day_diff) <= 90),
    pct_over_90d     = mean(abs(day_diff) > 90),
    pct_we_earlier   = mean(day_diff < -7),   # we find earlier than RKKP
    pct_rkkp_earlier = mean(day_diff >  7)    # RKKP finds earlier than us
  )

#### SPECIFIC AUDIT

# ==============================================================================
# Systematic audit of cases where we are later than RKKP
# (excluding RKKP-sourced treatments to test LPR/epikur alone)
# ==============================================================================

# Patients where we are materially later than RKKP
late_patients <- both %>%
  filter(day_diff > 7) %>%   # we find date >7 days after RKKP
  select(patientid, firstline_date, date_treatment_1st_line, day_diff)

cat(sprintf("Patients where we are >7 days late: %d\n", nrow(late_patients)))

# For each late patient, look at ALL treatment records in the gap
# i.e. between RKKP date and our date — what exists there?
gap_treatments <- tx_flagged %>%
  mutate(patientid = patient_id) %>%
  inner_join(late_patients, by = "patientid") %>%
  filter(
    tx_date >= date_treatment_1st_line,   # from RKKP date
    tx_date <  firstline_date             # up to our (later) date
  )

# ==============================================================================
# Step 1: Are the treatments simply absent from LPR/epikur?
# i.e. RKKP knows about them but they never appear in the treatment table at all
# ==============================================================================
late_patients %>%
  left_join(
    gap_treatments %>%
      group_by(patientid) %>%
      summarise(n_gap_records = n(), .groups = "drop"),
    by = "patientid"
  ) %>%
  mutate(n_gap_records = replace_na(n_gap_records, 0L)) %>%
  count(
    has_gap_records = n_gap_records > 0
  ) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  mutate(label = if_else(has_gap_records,
                         "Treatments exist in gap but rejected/missed",
                         "No treatment records at all in gap"))

# ==============================================================================
# Step 2: Of those WITH records in the gap — why were they rejected?
# ==============================================================================
gap_treatments %>%
  mutate(patientid = patient_id) %>%
  group_by(patientid) %>%
  summarise(
    n_records       = n(),
    n_excl_icd      = sum(ind_excl_icd,  na.rm = TRUE),
    n_excl_text     = sum(ind_excl_text, na.rm = TRUE),
    n_excl_code     = sum(ind_excl_code, na.rm = TRUE),
    n_heme_icd      = sum(ind_heme_icd,  na.rm = TRUE),
    n_ok            = sum(ind_ok,         na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    rejection_reason = case_when(
      n_ok > 0                          ~ "passed_filter_but_no_episode",
      n_excl_icd  > 0                   ~ "excluded_by_icd",
      n_excl_text > 0                   ~ "excluded_by_text",
      n_excl_code > 0                   ~ "excluded_by_atc_code",
      TRUE                              ~ "unknown"
    )
  ) %>%
  count(rejection_reason, sort = TRUE) %>%
  mutate(pct = round(100 * n / sum(n), 1))

# ==============================================================================
# Step 3: For those that passed filtering but still didn't form an episode —
# what are the actual treatment codes in the gap?
# Most informative: tells you what drugs are being given that we're not counting
# ==============================================================================
passed_but_late <- gap_treatments %>%
  filter(ind_ok) %>%
  count(code_text, sort = TRUE)

cat("\n=== Treatment codes present in gap that passed filtering ===\n")
print(passed_but_late, n = 30)

# ==============================================================================
# Step 4: For those with NO records in the gap —
# check which data source the RKKP treatment came from vs what's in LPR
# This tells you if it's a source coverage problem
# ==============================================================================
no_gap_ids <- late_patients %>%
  left_join(
    gap_treatments %>%
      group_by(patientid) %>%
      summarise(n_gap_records = n(), .groups = "drop"),
    by = "patientid"
  ) %>%
  filter(is.na(n_gap_records)) %>%
  pull(patientid)

# What does the full treatment table look like for these patients
# around the RKKP date? Maybe source is SP beh (treatment plans) which we excluded
treatments_raw_gap <- treatments %>%   # raw, before SP beh filter
  filter(patientid %in% no_gap_ids) %>%
  inner_join(late_patients %>% select(patientid, date_treatment_1st_line),
             by = "patientid") %>%
  filter(
    as.Date(date_start) >= date_treatment_1st_line,
    as.Date(date_start) <  date_treatment_1st_line + 30
  ) %>%
  count(source, sort = TRUE)

cat("\n=== Sources of treatments in gap for patients with no LPR/epikur records ===\n")
print(treatments_raw_gap)

### DISTRIBUTION SPECIFICALLY FOR THE SP OVERLAP SENSITIVITY CHECK
unique(both$hospital_id)
sp = both %>% filter(hospital_id %in% c("HER", "RH", "ROS", "HILLEROED", "NAESTVED"), date_treatment_1st_line >= date("2018-01-01"), date_treatment_1st_line < date("2022-01-01"))
                     
ggplot(sp, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(sp)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "N patients"
  ) 


ggplot(sp, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(
    aes(y = after_stat(count / sum(count)) * 100), binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(sp)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "% of patients"
  )


# ==============================================================================
# 3. Distribution plot
# ==============================================================================
ggplot(both, aes(x = pmin(pmax(day_diff, -365), 365))) +
  geom_histogram(binwidth = 7, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-365, -180, -90, -30, 0, 30, 90, 180, 365),
    labels = c("-1yr","-6m","-90d","-30d","0","30d","90d","6m","1yr")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at ±1 year",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "N patients"
  )

ggplot(both, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "N patients"
  )
ggsave("LYFO_check_counts.pdf")

ggplot(both, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(
    aes(y = after_stat(count / sum(count)) * 100), binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "% of patients"
  )
ggsave("LYFO_check_percent.pdf")

#### FOR CLL

load_dataset("RKKP_CLL_CLEAN")

checking <- RKKP_CLL_CLEAN %>% select(patientid, date_treatment_1st_line, hospital_id) %>% left_join(firstline_tbl)

checking_CLL <- checking

checking <- checking %>%
  mutate(
    firstline_date          = as.Date(firstline_date),
    date_treatment_1st_line = as.Date(date_treatment_1st_line),
    has_ours   = !is.na(firstline_date),
    has_rkkp   = !is.na(date_treatment_1st_line),
    day_diff   = as.numeric(firstline_date - date_treatment_1st_line)
  )

# ==============================================================================
# 1. Coverage comparison
# ==============================================================================
checking %>%
  count(has_ours, has_rkkp) %>%
  mutate(label = case_when(
    has_ours &  has_rkkp ~ "Both found",
    has_ours & !has_rkkp ~ "We find, RKKP missing",
    !has_ours &  has_rkkp ~ "RKKP finds, we miss",
    TRUE                  ~ "Neither found"
  )) %>%
  select(label, n) %>%
  mutate(pct = round(100 * n / sum(n), 1))

# ==============================================================================
# 2. Date mismatch distribution (patients where both have a date)
# ==============================================================================
both <- checking %>% filter(has_ours, has_rkkp)

cat(sprintf("N with both dates: %d\n", nrow(both)))

both %>%
  summarise(
    median_diff_days = median(day_diff),
    mean_diff_days   = mean(day_diff),
    sd_diff_days     = sd(day_diff),
    pct_exact        = mean(day_diff == 0),
    pct_within_7d    = mean(abs(day_diff) <= 7),
    pct_within_30d   = mean(abs(day_diff) <= 30),
    pct_within_90d   = mean(abs(day_diff) <= 90),
    pct_over_90d     = mean(abs(day_diff) > 90),
    pct_we_earlier   = mean(day_diff < -7),   # we find earlier than RKKP
    pct_rkkp_earlier = mean(day_diff >  7)    # RKKP finds earlier than us
  )

sp = both %>% filter(hospital_id %in% c("HER", "RH", "ROS", "HILLEROED", "NAESTVED"), date_treatment_1st_line >= date("2018-01-01"), date_treatment_1st_line < date("2022-01-01"))


### SP CHECK


ggplot(sp, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(sp)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "N patients"
  )
#ggsave("CLL_check_counts.pdf")


ggplot(sp, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(
    aes(y = after_stat(count / sum(count)) * 100), binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(sp)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "% of patients"
  )
#ggsave("CLL_check_percent.pdf")

# ==============================================================================
# 3. Distribution plot
# ==============================================================================
ggplot(both, aes(x = pmin(pmax(day_diff, -365), 365))) +
  geom_histogram(binwidth = 7, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-365, -180, -90, -30, 0, 30, 90, 180, 365),
    labels = c("-1yr","-6m","-90d","-30d","0","30d","90d","6m","1yr")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at ±1 year",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "N patients"
  )

ggplot(both, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "N patients"
  )
ggsave("CLL_check_counts.pdf")


ggplot(both, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(
    aes(y = after_stat(count / sum(count)) * 100), binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "% of patients"
  )
ggsave("CLL_check_percent.pdf")

### FOR MM


load_dataset("RKKP_DaMyDa_CLEAN")

RKKP_DaMyDa_CLEAN <- RKKP_DaMyDa_CLEAN %>% mutate(date_treatment_1st_line = date_treatment_1st_line_start)

checking <- RKKP_DaMyDa_CLEAN %>% select(patientid, date_treatment_1st_line, hospital_id) %>% left_join(firstline_tbl)

checking_MM <- checking

checking <- checking %>%
  mutate(
    firstline_date          = as.Date(firstline_date),
    date_treatment_1st_line = as.Date(date_treatment_1st_line),
    has_ours   = !is.na(firstline_date),
    has_rkkp   = !is.na(date_treatment_1st_line),
    day_diff   = as.numeric(firstline_date - date_treatment_1st_line)
  )

# ==============================================================================
# 1. Coverage comparison
# ==============================================================================
checking %>%
  count(has_ours, has_rkkp) %>%
  mutate(label = case_when(
    has_ours &  has_rkkp ~ "Both found",
    has_ours & !has_rkkp ~ "We find, RKKP missing",
    !has_ours &  has_rkkp ~ "RKKP finds, we miss",
    TRUE                  ~ "Neither found"
  )) %>%
  select(label, n) %>%
  mutate(pct = round(100 * n / sum(n), 1))

# ==============================================================================
# 2. Date mismatch distribution (patients where both have a date)
# ==============================================================================
both <- checking %>% filter(has_ours, has_rkkp)

cat(sprintf("N with both dates: %d\n", nrow(both)))

both %>%
  summarise(
    median_diff_days = median(day_diff),
    mean_diff_days   = mean(day_diff),
    sd_diff_days     = sd(day_diff),
    pct_exact        = mean(day_diff == 0),
    pct_within_7d    = mean(abs(day_diff) <= 7),
    pct_within_30d   = mean(abs(day_diff) <= 30),
    pct_within_90d   = mean(abs(day_diff) <= 90),
    pct_over_90d     = mean(abs(day_diff) > 90),
    pct_we_earlier   = mean(day_diff < -7),   # we find earlier than RKKP
    pct_rkkp_earlier = mean(day_diff >  7)    # RKKP finds earlier than us
  )

sp = both %>% filter(hospital_id %in% c("HER", "RH", "ROS", "HILLEROED", "NAESTVED"), date_treatment_1st_line >= date("2018-01-01"), date_treatment_1st_line < date("2022-01-01"))

### SP 

ggplot(sp, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(sp)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "N patients"
  )
#ggsave("MM_check_counts.pdf")

ggplot(sp, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(
    aes(y = after_stat(count / sum(count)) * 100), binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(sp)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "% of patients"
  )
#ggsave("MM_check_percentage.pdf")


# ==============================================================================
# 3. Distribution plot
# ==============================================================================
ggplot(both, aes(x = pmin(pmax(day_diff, -365), 365))) +
  geom_histogram(binwidth = 7, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-365, -180, -90, -30, 0, 30, 90, 180, 365),
    labels = c("-1yr","-6m","-90d","-30d","0","30d","90d","6m","1yr")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at ±1 year",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "N patients"
  )

ggplot(both, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "N patients"
  )
ggsave("MM_check_counts.pdf")

ggplot(both, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(
    aes(y = after_stat(count / sum(count)) * 100), binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "% of patients"
  )
ggsave("MM_check_percentage.pdf")

# ==============================================================================
# 4. Where we find a date but RKKP doesn't — what are these patients?
#    Likely: pre-RKKP era, or diagnoses LYFO doesn't cover (MM, HCL, etc.)
# ==============================================================================

# merge all RKKP checks

checking <- checking_MM %>% rbind(checking_LYFO) %>% rbind(checking_CLL)

checking <- checking %>%
  mutate(
    firstline_date          = as.Date(firstline_date),
    date_treatment_1st_line = as.Date(date_treatment_1st_line),
    has_ours   = !is.na(firstline_date),
    has_rkkp   = !is.na(date_treatment_1st_line),
    day_diff   = as.numeric(firstline_date - date_treatment_1st_line)
  )

# ==============================================================================
# 1. Coverage comparison
# ==============================================================================
checking %>%
  count(has_ours, has_rkkp) %>%
  mutate(label = case_when(
    has_ours &  has_rkkp ~ "Both found",
    has_ours & !has_rkkp ~ "We find, RKKP missing",
    !has_ours &  has_rkkp ~ "RKKP finds, we miss",
    TRUE                  ~ "Neither found"
  )) %>%
  select(label, n) %>%
  mutate(pct = round(100 * n / sum(n), 1))

both <- checking %>% filter(has_ours, has_rkkp)

ggplot(both, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "N patients"
  )
ggsave("OVERALL_check_counts.pdf")

ggplot(both, aes(x = pmin(pmax(day_diff, -30), 30))) +
  geom_histogram(
    aes(y = after_stat(count / sum(count)) * 100), binwidth = 1, fill = "steelblue", colour = "white") +
  geom_vline(xintercept = c(-30, 0, 30), linetype = "dashed", colour = "red") +
  scale_x_continuous(
    breaks = c(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
    labels = c("-30d","-25d","-20d","-15d","-10d", "-5d", "0", "5d", "10d", "15d", "20d", "25d", "30d")
  ) +
  labs(
    title = "Ours minus RKKP (days); capped at 30 days",
    subtitle = sprintf("n = %d patients with both dates", nrow(both)),
    x = "firstline_date - date_treatment_1st_line (days)",
    y = "% of patients"
  )
ggsave("OVERALL_check_percentage.pdf")

checking %>%
  filter(has_ours, !has_rkkp) %>%
  count(cohort_fine, sort = TRUE)

# And their firstline_date distribution — are they old cases?
checking %>%
  filter(has_ours, !has_rkkp) %>%
  mutate(year = year(firstline_date)) %>%
  count(year) %>%
  print()

# ==============================================================================
# 5. Where RKKP finds a date but we don't — what are we missing?
#    These are your false negatives — inspect the treatment codes
# ==============================================================================
checking %>%
  filter(!has_ours, has_rkkp) %>%
  count(cohort_fine, sort = TRUE)

# How far into our window do these patients have ANY treatment record at all?
# If they have treatments but we rejected them, the filter is too aggressive.
# If they have no treatments at all, it's a data linkage gap.
tx_any <- tx_flagged %>%
  group_by(patient_id) %>%
  summarise(n_tx_total = n(), n_tx_ok = sum(ind_ok), .groups = "drop")

checking %>%
  filter(!has_ours, has_rkkp) %>%
  left_join(tx_any, by = c("patientid" = "patient_id")) %>%
  summarise(
    n                 = n(),
    n_no_tx_at_all    = sum(is.na(n_tx_total) | n_tx_total == 0),
    n_tx_but_filtered = sum(!is.na(n_tx_total) & n_tx_total > 0 & (is.na(n_tx_ok) | n_tx_ok == 0)),
    n_tx_ok_but_no_ep = sum(!is.na(n_tx_ok) & n_tx_ok > 0)
  )

# ==============================================================================
# 6. Large mismatches — most actionable QA cases
#    Pull these for manual inspection
# ==============================================================================
big_mismatches <- both %>%
  filter(abs(day_diff) > 90) %>%
  arrange(desc(abs(day_diff))) %>%
  select(
    patientid,
    firstline_date, date_treatment_1st_line, day_diff,
    firstline_codes_text, firstline_has_anchor, firstline_n_codes
  )

cat(sprintf("\nLarge mismatches (>90 days): %d (%.1f%%)\n",
            nrow(big_mismatches), 100 * nrow(big_mismatches) / nrow(both)))

# Look at treatment codes for large-mismatch cases where WE are earlier
# This is the most suspicious scenario — we may be picking up prophylaxis
# or an unrelated treatment before the actual lymphoma treatment
big_mismatches %>%
  filter(day_diff < -90) %>%   # we are >90d earlier than RKKP
  count(firstline_codes_text, sort = TRUE) %>%
  head(20)

big_mismatches %>%
  filter(day_diff > 90) %>%   # we are >90d earlier than RKKP
  count(firstline_codes_text, sort = TRUE) %>%
  head(20)

firstline_tbl_mapped <- firstline_tbl %>%
  mutate(patientid = patient_id) %>%
  left_join(
    RKKP_LYFO_CLEAN %>%
      select(patientid, rkkp_firstline_date = date_treatment_1st_line) %>%
      mutate(rkkp_firstline_date = as.Date(rkkp_firstline_date)) %>%
      distinct(patientid, .keep_all = TRUE),
    by = "patientid"
  ) %>%
  mutate(
    firstline_date_final = coalesce(as.Date(firstline_date), rkkp_firstline_date),
    firstline_date_source = case_when(
      !is.na(firstline_date)      ~ "inferred",
      !is.na(rkkp_firstline_date) ~ "RKKP_fallback",
      TRUE                        ~ NA_character_
    )
  )

firstline_tbl_mapped %>%
  count(firstline_date_source) %>%
  mutate(pct = round(100 * n / sum(n), 1))
