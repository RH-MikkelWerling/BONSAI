# ==============================================================================
# calculate the adverse event outcomes based on Christian's scripts
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

aki_table_path <- Sys.getenv("OPERA_AKI_TABLE", "")
if (!nzchar(aki_table_path) || !file.exists(aki_table_path))
  stop("Set OPERA_AKI_TABLE to the existing AKI_table_all_dalycare_simplified.csv before running outcomes")
aki_header <- names(readr::read_delim(aki_table_path, delim = "\t", n_max = 0L,
                                      show_col_types = FALSE))
aki_required <- c("patientid", "aki_start_time", "aki_episode")
if (!all(aki_required %in% aki_header))
  stop("AKI table missing required columns: ",
       paste(setdiff(aki_required, aki_header), collapse = ", "))


# ------------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------------
source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")


load_dataset("SDS_laboratorieproevesvar", selected_columns =c("patientid", "analysiscode", "samplingdate", "samplingtime", "samplevalue", "unit", "referenceinterval_lowerlimit", "referenceinterval_upperlimit"))
load_npu_common()

laboratory_measurements = SDS_laboratorieproevesvar %>% clean_lab_values(value = samplevalue, unit = unit)

##
load_dataset("diagnoses_all")
load_dataset("t_dalycare_diagnoses")

# ============================================================
# CTCAE v6 Adverse Event functions — missing lab outcomes
# Author: [your name]
# Reference: NCI CTCAE v6.0 (released 2025). Local lab-only adaptations below
# are not automatically equivalent to the full clinical grading criteria.
# ============================================================
# Functions included:
#   AE_hyponatremia
#   AE_hypokalemia
#   AE_hypomagnesemia
#   AE_LDH_increased
#   AE_hyperuricemia
#
# NOTE on NPU codes: all functions call load_npu_common().
# Verify that NPU.NAT, NPU.MAG, NPU.LDH, NPU.URAT exist in
# your npu_common definitions and match your lab system.
# NPU.KAL is reused from the existing hyperkalemia function.
# ============================================================


# ------------------------------------------------------------
# Hyponatremia
# CTCAE v6: G1 <LLN-130; G2 125-<130; G3 120-<125; G4 <120
# Unit: mmol/L. Default LLN = 136.
#
# NOTE: CTCAE v6 G2/G3 boundary involves symptomatic status
# (G2 = asymptomatic 125-130; G3 = symptomatic 125-130 OR
# any 120-125). From labs alone we cannot assess symptoms, so
# we use the conservative lab-only approximation:
#   G2 = 125-130, G3 = 120-<125, G4 = <120.
# This underestimates G3 (misses symptomatic 125-130 cases).
# ------------------------------------------------------------

AE_hyponatremia = function(data,
                           c_value = c_value,
                           analysiscode = analysiscode,
                           date_sample = samplingdate,
                           referenceinterval_lowerlimit = referenceinterval_lowerlimit,
                           days_grade_5 = 30) {
  
  if (!exists('patient'))      { load_dataset('patient') }
  if (!exists('NPU.NAT'))      { load_npu_common() }
  if (!exists('NPU.NAT')) stop('NPU.NAT is required for hyponatremia grading')
  
  data %>%
    filter({{ analysiscode }} %in% NPU.NAT,
           !is.na({{ c_value }})) %>%
    left_join(patient %>% select(patientid, status, date_death_fu), 'patientid') %>%
    mutate(
      time_to_death  = diff_days({{ date_sample }}, date_death_fu),
      value_numeric  = as.numeric({{ c_value }}),
      LLN            = as.numeric({{ referenceinterval_lowerlimit }}),
      LLN            = ifelse(is.na(LLN), 136, LLN),
      # Assign grades 1-4 from lowest to highest severity
      ae_hyponatremia = as.character(cut(value_numeric,
                                         c(-Inf, 120, 125, 130, Inf),
                                         c('4', '3', '2', '1'), right = FALSE)),
      # Zero out if >= LLN (normal)
      ae_hyponatremia = ifelse(value_numeric >= LLN, '0', ae_hyponatremia),
      ae_hyponatremia_g5 = factor(
        ifelse(time_to_death < {{ days_grade_5 }} & status == 1, '5', ae_hyponatremia),
        c('0', '1', '2', '3', '4', '5')),
      ae_hyponatremia = factor(ae_hyponatremia, c('0', '1', '2', '3', '4'))
    )
}


# ------------------------------------------------------------
# Hypokalemia
# LEGACY LAB-ONLY BINS, NOT CTCAE v6: G1 LLN-3.0; G2 3.0-2.5;
# G3 2.5-2.0; G4 <2.0. CTCAE v6 instead uses symptomatic status for
# G2, <3.0-2.5 mmol/L for G3 and <2.5 mmol/L for G4. Do not present
# hypokalemia_g2plus/g3plus from this function as v6-compliant.
# Unit: mmol/L. Default LLN = 3.5.
# ------------------------------------------------------------

AE_hypokalemia = function(data,
                          c_value = c_value,
                          analysiscode = analysiscode,
                          date_sample = samplingdate,
                          referenceinterval_lowerlimit = referenceinterval_lowerlimit,
                          days_grade_5 = 30) {
  
  if (!exists('patient'))   { load_dataset('patient') }
  if (!exists('NPU.KAL'))   { load_npu_common() }
  
  data %>%
    filter({{ analysiscode }} %in% NPU.KAL,
           !is.na({{ c_value }})) %>%
    left_join(patient %>% select(patientid, status, date_death_fu), 'patientid') %>%
    mutate(
      time_to_death = diff_days({{ date_sample }}, date_death_fu),
      value_numeric = as.numeric({{ c_value }}),
      LLN           = as.numeric({{ referenceinterval_lowerlimit }}),
      LLN           = ifelse(is.na(LLN), 3.5, LLN),
      ae_hypokalemia = as.character(cut(value_numeric,
                                        c(-Inf, 2.0, 2.5, 3.0, Inf),
                                        c('4', '3', '2', '1'))),
      ae_hypokalemia = ifelse(value_numeric >= LLN, '0', ae_hypokalemia),
      ae_hypokalemia_g5 = factor(
        ifelse(time_to_death < {{ days_grade_5 }} & status == 1, '5', ae_hypokalemia),
        c('0', '1', '2', '3', '4', '5')),
      ae_hypokalemia = factor(ae_hypokalemia, c('0', '1', '2', '3', '4'))
    )
}


# ------------------------------------------------------------
# Hypomagnesemia
# CTCAE v6: G1 LLN-0.5; G2 0.5-0.4; G3 0.4-0.3; G4 <0.3
# Unit: mmol/L. Default LLN = 0.7.
# ------------------------------------------------------------

NPU.MAG = c("NPU02647")

AE_hypomagnesemia = function(data,
                             c_value = c_value,
                             analysiscode = analysiscode,
                             date_sample = samplingdate,
                             referenceinterval_lowerlimit = referenceinterval_lowerlimit,
                             days_grade_5 = 30) {
  
  if (!exists('patient'))   { load_dataset('patient') }
  if (!exists('NPU.MAG'))   { load_npu_common() }
  
  data %>%
    filter({{ analysiscode }} %in% NPU.MAG,
           !is.na({{ c_value }})) %>%
    left_join(patient %>% select(patientid, status, date_death_fu), 'patientid') %>%
    mutate(
      time_to_death    = diff_days({{ date_sample }}, date_death_fu),
      value_numeric    = as.numeric({{ c_value }}),
      LLN              = as.numeric({{ referenceinterval_lowerlimit }}),
      LLN              = ifelse(is.na(LLN), 0.7, LLN),
      ae_hypomagnesemia = as.character(cut(value_numeric,
                                           c(-Inf, 0.3, 0.4, 0.5, Inf),
                                           c('4', '3', '2', '1'), right = FALSE)),
      ae_hypomagnesemia = ifelse(value_numeric >= LLN, '0', ae_hypomagnesemia),
      ae_hypomagnesemia_g5 = factor(
        ifelse(time_to_death < {{ days_grade_5 }} & status == 1, '5', ae_hypomagnesemia),
        c('0', '1', '2', '3', '4', '5')),
      ae_hypomagnesemia = factor(ae_hypomagnesemia, c('0', '1', '2', '3', '4'))
    )
}


# ------------------------------------------------------------
# LDH increased (Blood lactate dehydrogenase)
# CTCAE v6: G1 >ULN only (no further grading defined)
# Unit: U/L. No fixed ULN — must come from reference interval.
#
# NOTE: CTCAE provides only a single grade (G1 = >ULN).
# LDH is therefore stored as binary (0/1) plus grade 5.
# For OPERA, LDH is most valuable as a continuous lab feature
# rather than a graded AE — consider using both representations.
# ------------------------------------------------------------



AE_LDH_increased = function(data,
                            c_value = c_value,
                            analysiscode = analysiscode,
                            date_sample = samplingdate,
                            referenceinterval_upperlimit = referenceinterval_upperlimit,
                            days_grade_5 = 30) {
  
  if (!exists('patient'))   { load_dataset('patient') }
  if (!exists('NPU.LDH'))   { load_npu_common() }
  
  data %>%
    filter({{ analysiscode }} %in% NPU.LDH,
           !is.na({{ c_value }})) %>%
    left_join(patient %>% select(patientid, status, date_death_fu), 'patientid') %>%
    mutate(
      time_to_death   = diff_days({{ date_sample }}, date_death_fu),
      value_numeric   = as.numeric({{ c_value }}),
      ULN             = as.numeric({{ referenceinterval_upperlimit }}),
      # Only G1 defined in CTCAE v6 — binary outcome
      ae_LDH_increased = ifelse(is.na(ULN) | value_numeric <= ULN, '0', '1'),
      ae_LDH_increased_g5 = factor(
        ifelse(time_to_death < {{ days_grade_5 }} & status == 1, '5', ae_LDH_increased),
        c('0', '1', '5')),
      ae_LDH_increased = factor(ae_LDH_increased, c('0', '1'))
    )
}


# ------------------------------------------------------------
# Hyperuricemia
# CTCAE v6: G1 >ULN without consequences; G3 >ULN with
# physiologic consequences; G4 life-threatening.
# Unit: mmol/L (divide µmol/L by 1000 if needed).
#
# NOTE: G3 requires clinical evidence of physiologic
# consequences (renal, cardiac, neurological) which cannot
# be determined from the lab value alone. From lab data only,
# this function grades as:
#   G1 = >ULN (any elevation)
# G3 should be ascertained separately via LPR ICD-10 codes
# (e.g. M10 gout, N17 AKI in temporal proximity).
# Consider combining with TLS composite for clinical grading.
# ------------------------------------------------------------

AE_hyperuricemia = function(data,
                            c_value = c_value,
                            analysiscode = analysiscode,
                            date_sample = samplingdate,
                            referenceinterval_upperlimit = referenceinterval_upperlimit,
                            days_grade_5 = 30) {
  
  if (!exists('patient'))    { load_dataset('patient') }
  if (!exists('NPU.URAT'))   { load_npu_common() }
  
  data %>%
    filter({{ analysiscode }} %in% NPU.URAT,
           !is.na({{ c_value }})) %>%
    left_join(patient %>% select(patientid, status, date_death_fu), 'patientid') %>%
    mutate(
      time_to_death      = diff_days({{ date_sample }}, date_death_fu),
      value_numeric      = as.numeric({{ c_value }}),
      # Handle both mmol/L and µmol/L: if median > 100, assume µmol/L
      # and convert. Adjust if your lab system is consistent.
      ULN                = as.numeric({{ referenceinterval_upperlimit }}),
      # Lab-ascertainable grade only (G1 = >ULN; G3 requires LPR)
      ae_hyperuricemia   = ifelse(is.na(ULN) | value_numeric <= ULN, '0', '1'),
      ae_hyperuricemia_g5 = factor(
        ifelse(time_to_death < {{ days_grade_5 }} & status == 1, '5', ae_hyperuricemia),
        c('0', '1', '5')),
      ae_hyperuricemia   = factor(ae_hyperuricemia, c('0', '1'))
    )
}


### NOTE; THINGS DO ACTUALLY RELY ON PATIENT TABLE; NEED TO FIX THIS FIRST, THEN WE CAN DO STUFF.
getwd()

load_dataset("patient")

# patient_new = read_parquet("/ngc/people/mikwer_r/projects/ehr2meds-the-purge/ehr2meds-the-purge/data/raw/sp/CPMI_PatientInfo.parquet") %>%
#   collect()

firstline_mapped_path <- Sys.getenv("OPERA_FIRSTLINE_MAPPED_CSV", "")
if (!nzchar(firstline_mapped_path) || !file.exists(firstline_mapped_path))
  stop("Set OPERA_FIRSTLINE_MAPPED_CSV to the reviewed CLL V4 / MM V5 mapped first-line CSV")
firstline_tbl_mapped <- read_csv(firstline_mapped_path)
firstline_input_n <- nrow(firstline_tbl_mapped)
expected_firstline_n <- Sys.getenv("OPERA_EXPECTED_FIRSTLINE_N", "")
if (nzchar(expected_firstline_n) &&
    (is.na(suppressWarnings(as.integer(expected_firstline_n))) ||
     firstline_input_n != as.integer(expected_firstline_n)))
  stop("Outcome input differs from OPERA_EXPECTED_FIRSTLINE_N")
if (anyNA(firstline_tbl_mapped$patientid) ||
    anyDuplicated(firstline_tbl_mapped$patientid))
  stop("Outcome input has missing or duplicate patient IDs")

# need to not be any conflict in the death / index dates

firstline_tbl_mapped
firstline_tbl_mapped_joined <- firstline_tbl_mapped %>% left_join(patient)

firstline_tbl_mapped <- firstline_tbl_mapped_joined %>%
  filter(firstline_date < date_death_fu) #%>%
  #select(patientid, firstline_date, date_death_fu)
message("First-line rows without post-index patient follow-up: ",
        firstline_input_n - nrow(firstline_tbl_mapped))

### when the hell was the followup date

FOLLOW_UP_DATE = "2026-03-19"
# 
# patient <- patient_new %>% 
#   mutate(Dødsdato = if_else(is.na(Dødsdato), as.Date(FOLLOW_UP_DATE), as.Date(Dødsdato)),
#          Fødselsdato = as.Date(Fødselsdato)) %>%
#   select(patientid, 
#          sex = Køn,
#          date_birth = Fødselsdato,
#          date_death_fu = Dødsdato,
#          status = status)

### ACUTE KIDNEY INJURY

#CREATININE_CLEAN = laboratory_measurements %>% filter(NPU %in% NPU.KREA)
#AKI = CREATININE_CLEAN %>%  AE_AKI(value = value2)

#AKI_OUTCOME = patient %>% left_join(AKI)

# =============================================================================
# CTCAE v5.0  Fibrinogen decreased  per-measurement grading
# =============================================================================
# Emits a CTCAE grade (0-4) per fibrinogen measurement, to be consumed by the
# shared threshold-crossing / time-to-event construction (first time >= grade 2,
# first time >= grade 3) the same way the other lab outcomes are.
#
# Only "decreased" is graded: CTCAE has no graded "fibrinogen increased" term.
#
# CTCAE v5 logic (dual axis, baseline-conditional):
#   baseline NORMAL   (baseline >= LLN)  -> grade on value / LLN ratio
#   baseline ABNORMAL (baseline <  LLN)  -> grade on % decrease from baseline
#   PLUS an absolute floor that forces grade 4 (specified as < 50 mg/dL).
# Final grade = max(axis grade, floor grade).
#
#   LLN-fraction axis:        % decrease axis (baseline abnormal):   floor:
#     value <  0.25*LLN -> 4    decrease >= 0.75 -> 4                 value < g4_floor -> 4
#     <  0.50*LLN       -> 3    >= 0.50          -> 3
#     <  0.75*LLN       -> 2    >= 0.25          -> 2
#     <  1.00*LLN       -> 1    >  0.00          -> 1
#     >= 1.00*LLN       -> 0    <= 0.00          -> 0
#
# UNIT WARNING: the grade-4 absolute floor is defined as 50 mg/dL. Danish labs
# report fibrinogen in g/L or umol/L, NOT mg/dL. 50 mg/dL = 0.5 g/L ~= 1.47
# umol/L. Hardcoding 50 against g/L or umol/L values would flag everyone as
# grade 4. Set `unit` correctly.
#
# INTEGRATION POINTS (align these with the rest of the pipeline):
#   - rename to your handler convention (e.g. grade_fibrinogen / ctcae_fibrinogen)
#   - swap the column names in grade_fibrinogen_table() to your schema
#   - confirm how baseline + LLN are supplied (per-patient baseline, lab LLN);
#     the other baseline-conditional handlers (ALT/ALP/creatinine) already
#     resolve these, so reuse that resolver rather than re-deriving here
#   - confirm the coverage-gate / eligibility mask is applied upstream, as for
#     the other labs (this function does NOT gate; it only grades)
# =============================================================================
# =============================================================================
# AE_hypofibrinogenemia  (CTCAE v5.0  "Fibrinogen decreased")
# Drop-in dalycare-style grader, modelled on AE_hypomagnesemia.
# =============================================================================
# CONVENTION NOTES / DELIBERATE CHOICES
#
# 1. LLN-fraction axis only. CTCAE v5 fibrinogen decreased has two axes: an
#    LLN-fraction axis (baseline normal) and a percent-decrease-from-baseline
#    axis (baseline abnormal). The dalycare graders never carry a patient
#    baseline, so this implements the LLN-fraction axis only and drops the
#    baseline-relative axis. That axis only governs patients whose fibrinogen
#    was already below LLN pre-treatment; for those, the absolute LLN-ratio is
#    the more relevant bleeding-risk signal anyway. Revisit only if the
#    baseline-abnormal subset (e.g. APL/DIC) turns out to matter.
#
# 2. CTCAE cuts are FRACTIONS of LLN, not absolute concentrations. Unlike
#    magnesium (absolute mmol/L cuts on value_numeric), fibrinogen cuts on
#    value_numeric / LLN:  <0.25 -> 4, <0.50 -> 3, <0.75 -> 2, <1.00 -> 1.
#
# 3. UNITS. The ratio axis is unit-safe (value and LLN share the row's unit, so
#    the ratio cancels units). Only two pieces are unit-sensitive: the
#    missing-LLN fallback and the grade-4 absolute floor. SET THESE to match the
#    unit this NPU code is stored in in the Forskertabel:
#       g/L:    fallback ~2.0,  floor 0.5    (CTCAE floor is 50 mg/dL = 0.5 g/L)
#       umol/L: fallback ~5.9,  floor 1.47   (using MW ~340 kDa; verify locally)
#
# 4. NPU code is a placeholder. Define it alongside the other NPU.* constants
#    from load_npu_common(); confirm against the NPU dictionary which fibrinogen
#    assay you want (Clauss functional fibrinogen is the usual choice).
# =============================================================================

# <-- SET ME: fibrinogen NPU code(s), confirmed against load_npu_common()

#subset = laboratory_measurements %>% 
#  filter(NPU %in% NPU.FIBR)
# <-- SET ME: unit-sensitive constants (see note 3). Defaults assume g/L.
FIB_LLN_FALLBACK = 2.0   # used only when referenceinterval_lowerlimit is NA
FIB_G4_FLOOR     = 0.5   # CTCAE grade-4 absolute floor (50 mg/dL equivalent)

AE_hypofibrinogenemia = function(data,
                                 c_value = c_value,
                                 analysiscode = analysiscode,
                                 date_sample = samplingdate,
                                 referenceinterval_lowerlimit = referenceinterval_lowerlimit,
                                 days_grade_5 = 30) {

  if (!exists('patient'))  { load_dataset('patient') }
  if (!exists('NPU.FIBR')) { load_npu_common() }
  if (!exists('NPU.FIBR')) stop('NPU.FIBR is required for fibrinogen grading')

  data %>%
    filter({{ analysiscode }} %in% NPU.FIBR,
           !is.na({{ c_value }})) %>%
    left_join(patient %>% select(patientid, status, date_death_fu), 'patientid') %>%
    mutate(
      time_to_death = diff_days({{ date_sample }}, date_death_fu),
      value_numeric = as.numeric({{ c_value }}),
      LLN           = as.numeric({{ referenceinterval_lowerlimit }}),
      LLN           = ifelse(is.na(LLN), FIB_LLN_FALLBACK, LLN),

      # CTCAE v5 LLN-fraction axis (cut on the ratio, right-closed as elsewhere)
      ae_hypofibrinogenemia = as.character(cut(value_numeric / LLN,
                                               c(-Inf, 0.25, 0.50, 0.75, Inf),
                                               c('4', '3', '2', '1'))),
      # everyone at or above LLN is grade 0
      ae_hypofibrinogenemia = ifelse(value_numeric >= LLN, '0', ae_hypofibrinogenemia),
      # grade-4 absolute floor (usually subsumed by the ratio grade 4)
      ae_hypofibrinogenemia = ifelse(value_numeric < FIB_G4_FLOOR, '4', ae_hypofibrinogenemia),

      ae_hypofibrinogenemia_g5 = factor(
        ifelse(time_to_death < {{ days_grade_5 }} & status == 1, '5', ae_hypofibrinogenemia),
        c('0', '1', '2', '3', '4', '5')),
      ae_hypofibrinogenemia = factor(ae_hypofibrinogenemia, c('0', '1', '2', '3', '4'))
    )
}

FIBRINOGEN <- laboratory_measurements %>% AE_hypofibrinogenemia(analysiscode = NPU, c_value = value2)

AKI <- read_delim(aki_table_path, delim = "\t")

AKI = AKI %>% select(
  patientid, 
  samplingdate = aki_start_time,
  ae_AKI = aki_episode # note, this doesn't really make sense, should probably just always be 3 or something
)

# NEED TO ASK CHRISTIAN HOW TO UNDERSTAND WHAT THIS OUTPUT IS

ALAT = laboratory_measurements %>% 
  AE_ALAT_increased(analysiscode = NPU, c_value = value2)

ANEMIA = laboratory_measurements %>% 
  AE_anemia(analysiscode = NPU, c_value = value2)

ASAT = laboratory_measurements %>%
  AE_ASAT_increased(analysiscode = NPU, c_value = value2)

BASP = laboratory_measurements %>%
  AE_BASP_increased(analysiscode = NPU, c_value = value2)

BILI = laboratory_measurements %>%
  AE_bilirubin_increased(analysiscode = NPU, c_value = value2)

CREA = laboratory_measurements %>%
  AE_creatinine_increased(analysiscode = NPU, c_value = value2)

EOS = laboratory_measurements %>%
  AE_eosinophilia(analysiscode = NPU, c_value = value2)

HAPT = laboratory_measurements %>%
  AE_haptoglobin_decreased(analysiscode = NPU, c_value = value2)

HEMO = laboratory_measurements %>%
  AE_hemoglobin_increased(analysiscode = NPU, c_value = value2) #NOTE; need baseline date to work

HYPERCALCEMIA = laboratory_measurements %>%
  AE_hypercalcemia(analysiscode = NPU, c_value = value2)

HYPERKALEMIA = laboratory_measurements %>%
  AE_hyperkalemia(analysiscode = NPU, c_value = value2)

AE_hypermagnesemia <- function(data, c_value = c_value, analysiscode = analysiscode,
                               date_sample = samplingdate,
                               referenceinterval_upperlimit = referenceinterval_upperlimit,
                               days_grade_5 = 30) {
  if (!exists('patient'))  load_dataset('patient')
  if (!exists('NPU.MG2'))  load_npu_common()
  data %>%
    filter({{ analysiscode }} %in% NPU.MG2, !is.na({{ c_value }})) %>%
    left_join(patient %>% select(patientid, status, date_death_fu), 'patientid') %>%
    mutate(
      time_to_death = diff_days({{ date_sample }}, date_death_fu),
      value_numeric = as.numeric({{ c_value }}),
      ULN = as.numeric({{ referenceinterval_upperlimit }}),
      ULN = ifelse(is.na(ULN), 1.10, ULN),
      ae_hypermagnesemia = as.character(cut(value_numeric, c(-Inf, 1.23, 3.30, Inf),
                                            c('1', '3', '4'))),   # G2 not defined for hypermagnesemia
      ae_hypermagnesemia = ifelse(value_numeric <= ULN, '0', ae_hypermagnesemia),
      ae_hypermagnesemia_g5 = factor(
        ifelse(time_to_death < {{ days_grade_5 }} & status == 1, '5', ae_hypermagnesemia),
        c('0', '1', '3', '4', '5')),
      ae_hypermagnesemia = factor(ae_hypermagnesemia, c('0', '1', '3', '4')))
}

HYPERMAGNESEMIA = laboratory_measurements %>%
  AE_hypermagnesemia(analysiscode = NPU, c_value = value2)

HYPERNATREMIA = laboratory_measurements %>%
  AE_hypernatremia(analysiscode = NPU, c_value = value2)

HYPERTRIGLY = laboratory_measurements %>%
  AE_hypertriglyceridemia(analysiscode = NPU, c_value = value2)

HYPOALBU = laboratory_measurements %>%
  AE_hypoalbuminemia(analysiscode = NPU, c_value = value2)

HYPOCALC = laboratory_measurements %>%
  AE_hypocalcemia(analysiscode = NPU, c_value = value2)

HYPOGLYC = laboratory_measurements %>%
  AE_hypoglycemia(analysiscode = NPU, c_value = value2)

INR = laboratory_measurements %>%
  AE_INR_increased(analysiscode = NPU, c_value = value2)

LEUKO = laboratory_measurements %>%
  AE_leukocytosis(analysiscode = NPU, c_value = value2)

LYMPH_DEC = laboratory_measurements %>%
  AE_lymphocyte_count_decreased(analysiscode = NPU, c_value = value2)

LYMPH_INC = laboratory_measurements %>%
  AE_lymphocyte_count_increased(analysiscode = NPU, c_value = value2)

NEU_DEC = laboratory_measurements %>%
  AE_neutrophil_count_decreased(analysiscode = NPU, c_value = value2)

PLATELETS_DEC = laboratory_measurements %>%
  AE_platelets_decreased(analysiscode = NPU, c_value = value2)

WBC_DEC = laboratory_measurements %>%
  AE_WBC_decreased(analysiscode = NPU, c_value = value2)

## new stuff

HYPERURICEMIA = laboratory_measurements %>%
  AE_hyperuricemia(analysiscode = NPU, c_value = value2)

LDH_INCREASED = laboratory_measurements %>%
  AE_LDH_increased(analysiscode = NPU, c_value = value2)

HYPOMAGNESEMIA = laboratory_measurements %>%
  AE_hypomagnesemia(analysiscode = NPU, c_value = value2)

HYPOKALEMIA = laboratory_measurements %>%
  AE_hypokalemia(analysiscode = NPU, c_value = value2)

HYPONATREMIA = laboratory_measurements %>%
  AE_hyponatremia(analysiscode = NPU, c_value = value2)


## NOTE; We need the additional outcomes + we need to have align all these so they have the same columns + are like we'd expect them to be for outcomes



# NOTE; IDEA COULD BE THAT CONTRASTIVE LEARNING IS DONE ON GRADE 2 AND EVAL IS ON GRADE 3


# ============================================================
# OPERA outcome file pipeline
# Produces one .parquet per outcome in OPERA format:
#   subject_id | split | outcome_date | index_date | censor_date
#
# Design:
#   index_date   = first line treatment start (from treatment pipeline)
#   outcome_date = first post-index date at grade >= threshold
#                  (NA if never reached before censor_date)
#   censor_date  = min(date_death_fu, STUDY_CUTOFF)
#                  date_death_fu is already min(death, last_follow_up)
#                  in the patient table — no further adjustment needed
#   split        = time-based on index_date (cutoffs below)
#                  TO BE CONFIRMED with collaborators before final run
#
# Outcome types:
#   Graded (G1-4):  two files per outcome — g2plus (contrastive
#                   training) and g3plus (evaluation target)
#   Binary (G1 only, e.g. LDH): one file, binary = TRUE in call
#
# One parquet per outcome written to OUTPUT_DIR.
# ============================================================

library(dplyr)
library(arrow)

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR   <- Sys.getenv("OPERA_OUTCOMES_DIR", "")
if (!nzchar(OUTPUT_DIR) || dir.exists(OUTPUT_DIR))
  stop("Set OPERA_OUTCOMES_DIR to a new, nonexistent directory for this run")
STUDY_CUTOFF <- as.Date("2026-03-19")   # administrative registry cutoff

# Split cutoffs — TO BE CONFIRMED with collaborators.
# Current proposal: time-based split on index_date.
#   train   : index_date <  TUNING_START
#   tuning  : index_date >= TUNING_START & < TEST_START
#   held_out: index_date >= TEST_START
# Extend TUNING_START back if tuning patient count is too low
# (check n per split in the console output after running).
TUNING_START <- as.Date("2022-01-01")
TEST_START   <- as.Date("2023-01-01")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)


# ── Helpers ───────────────────────────────────────────────────────────────────

# Null coalescing (base R doesn't have it)
`%||%` <- function(x, y) if (is.null(x)) y else x

assign_splits <- function(df, index_date_col = index_date) {
  df %>%
    mutate(split = case_when(
      {{ index_date_col }} >= TEST_START   ~ "held_out",
      {{ index_date_col }} >= TUNING_START ~ "tuning",
      TRUE                                 ~ "train"
    ))
}


# ── Core helper ───────────────────────────────────────────────────────────────
#
# Arguments:
#   ae_data       output of an AE_ function; one row per lab measurement
#   grade_col     unquoted name of the grade factor column
#   date_col      unquoted name of the sample date column
#   threshold     minimum numeric grade to count as event (ignored if binary)
#   first_line    data frame with patientid + index_date
#   patient_data  patient table with patientid + date_death_fu
#   outcome_name  string used for output filename (no extension)
#   binary        TRUE for outcomes with no grading beyond G1 (LDH,
#                 hyperuricemia, haptoglobin). Forces threshold = 1
#                 and warns if caller passed something else.
#   study_cutoff  administrative censoring date

make_opera_outcome <- function(ae_data,
                               grade_col,
                               date_col,
                               threshold,
                               first_line,
                               patient_data,
                               outcome_name,
                               binary       = FALSE,
                               study_cutoff = STUDY_CUTOFF) {
  
  # Guard: binary outcomes only have G1
  if (binary && threshold != 1L) {
    warning(
      outcome_name, ": binary = TRUE but threshold = ", threshold,
      ". Forcing threshold to 1. If grading exists, set binary = FALSE."
    )
  }
  effective_threshold <- if (binary) 1L else as.integer(threshold)
  
  # 1. Attach index_date and censor_date to AE measurements
  df <- ae_data %>%
    select(-any_of(c("date_death_fu", "status", "time_to_death", "index_date", "censor_date"))) %>%
    inner_join(
      first_line %>% select(patientid, index_date),
      by = "patientid"
    ) %>%
    inner_join(
      patient_data %>% select(patientid, date_death_fu),
      by = "patientid"
    ) %>%
    mutate(
      censor_date   = pmin(date_death_fu, study_cutoff, na.rm = TRUE),
      grade_numeric = as.integer(sub("^(\\d+).*", "\\1", as.character({{ grade_col }}))),
      sample_date   = as.Date({{ date_col }})
    )
  
  # 2. First post-index event at or above threshold, before censor
  events <- df %>%
    filter(
      sample_date >  index_date,
      sample_date <= censor_date,
      grade_numeric >= effective_threshold
    ) %>%
    group_by(patientid) %>%
    slice_min(sample_date, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    select(patientid, outcome_date = sample_date)
  
  # 3. One row per patient in first_line cohort
  #    outcome_date is NA for patients who never reached threshold
  result <- first_line %>%
    select(patientid, index_date) %>%
    inner_join(
      patient_data %>% select(patientid, date_death_fu),
      by = "patientid"
    ) %>%
    mutate(
      censor_date = pmin(date_death_fu, study_cutoff, na.rm = TRUE)
    ) %>%
    left_join(events, by = "patientid") %>%
    rename(subject_id = patientid) %>%
    assign_splits(index_date_col = index_date) %>%
    select(subject_id, split, outcome_date, index_date, censor_date)
  
  # 4. Sanity checks
  n_total  <- nrow(result)
  n_events <- sum(!is.na(result$outcome_date))
  n_splits <- table(result$split)
  
  stopifnot(
    "outcome_date must not exceed censor_date" =
      all(is.na(result$outcome_date) |
            result$outcome_date <= result$censor_date),
    "outcome_date must be after index_date" =
      all(is.na(result$outcome_date) |
            result$outcome_date > result$index_date),
    "censor_date must not be NA" =
      !any(is.na(result$censor_date)),
    "index_date must not be NA" =
      !any(is.na(result$index_date))
  )
  
  # 5. Write parquet
  out_path <- file.path(OUTPUT_DIR, paste0(outcome_name, ".parquet"))
  write_parquet(result, out_path)
  
  message(sprintf(
    "%-45s  n=%d  events=%d (%.1f%%)  train=%d  tuning=%d  held_out=%d",
    outcome_name,
    n_total,
    n_events,
    100 * n_events / n_total,
    n_splits["train"]    %||% 0L,
    n_splits["tuning"]   %||% 0L,
    n_splits["held_out"] %||% 0L
  ))
  
  invisible(result)
}

# ==============================================================================
# make_opera_outcome_lab.R
#
# Coverage-gated laboratory outcomes for OPERA. Drops into calculate_adverse_events.R
# in place of the inline make_opera_outcome() definition. The administrative
# engines (_malignancy, _procedure, _infection, overall_survival) are NOT touched.
#
# What changes vs the old lab engine:
#   old: first_line LEFT JOIN events  -> every patient written, no-event = control
#        (silent negatives for patients with no lab coverage)
#   new: eligibility = source_covered AND baseline_covered (any lab in 365d before)
#        AND entry_covered (any lab in 30d after). Only eligible patients written;
#        everyone else goes to an audit parquet with a reason. An observed event
#        is always eligible. censor is outcome-specific (last draw + max_gap,
#        capped at death/cutoff), so an adequately observed death becomes a
#        competing death and an unmonitored stretch is not credited event-free.
#
# Output schema is UNCHANGED: subject_id, split, outcome_date, index_date, censor_date.
# A second file <outcome>__audit.parquet is written alongside each outcome.
#
# Relies on objects already defined in calculate_adverse_events.R:
#   assign_splits(), `%||%`, STUDY_CUTOFF, OUTPUT_DIR, write_parquet(), lubridate
# ==============================================================================

LAB_GO_LIVE <- as.Date("2017-01-01")

LAB_DEFAULTS <- list(
  baseline_window_days = 365L,   # any lab in [index - this, index]   -> baseline_covered
  entry_window_days    = 30L,    # any lab in [index, index + this]   -> entry_covered
  max_gap_days         = 180L,   # event-free credit ends this long after the last draw
  baseline_required    = TRUE    # uniform rule: both windows required for a negative
)

.safe_min_date <- function(x) if (length(x) == 0L) as.Date(NA) else min(x)
.safe_max_date <- function(x) if (length(x) == 0L) as.Date(NA) else max(x)

.calendar_period <- function(index_date, go_live) dplyr::case_when(
  index_date <  go_live                     ~ "pre_go_live",
  index_date <  go_live %m+% lubridate::years(3L) ~ "early_post_go_live",
  TRUE                                      ~ "late_post_go_live")


# ------------------------------------------------------------------------------
# Shared coverage table. Build ONCE from all lab activity; reused by every lab
# outcome. One row per first-line patient.
#   lab_data  = laboratory_measurements (all analytes; only patientid + date used)
# ------------------------------------------------------------------------------

build_lab_coverage <- function(lab_data,
                               first_line,
                               patient_data,
                               date_col             = samplingdate,
                               analyte_col          = NPU,
                               routine_analytes     = NULL,
                               go_live_date         = LAB_GO_LIVE,
                               baseline_window_days = LAB_DEFAULTS$baseline_window_days,
                               entry_window_days    = LAB_DEFAULTS$entry_window_days,
                               recent_baseline_window_days = 90L,
                               study_cutoff         = STUDY_CUTOFF) {
  
  fl <- first_line %>% select(patientid, index_date, any_of("cohort"))
  if (!"cohort" %in% names(fl)) fl$cohort <- "all"
  
  base <- fl %>%
    inner_join(patient_data %>% select(patientid, date_death_fu, status), by = "patientid") %>%
    mutate(go_live        = go_live_date,
           admin_end      = pmin(date_death_fu, study_cutoff, na.rm = TRUE),
           death_observed = status == 1L & date_death_fu <= study_cutoff)
  
  draws <- lab_data
  if (!is.null(routine_analytes)) draws <- draws %>% filter({{ analyte_col }} %in% routine_analytes)
  
  draws <- draws %>%
    transmute(patientid, draw_date = as.Date({{ date_col }})) %>%
    filter(!is.na(draw_date)) %>% distinct() %>%
    inner_join(base %>% select(patientid, index_date, admin_end), by = "patientid") %>%
    mutate(is_baseline = draw_date <= index_date & draw_date >= index_date - baseline_window_days,
           is_recent_baseline = draw_date <= index_date & draw_date >= index_date - recent_baseline_window_days,
           is_entry    = draw_date >= index_date & draw_date <= index_date + entry_window_days,
           is_post     = draw_date >= index_date & draw_date <= admin_end,   # monitoring span
           is_window   = draw_date >= index_date - baseline_window_days & draw_date <= admin_end)
  
  cov <- draws %>% group_by(patientid) %>% summarise(
    baseline_covered = any(is_baseline),
    recent_baseline_covered = any(is_recent_baseline),
    entry_covered    = any(is_entry),
    n_draws_post     = sum(is_post),
    first_draw_post  = .safe_min_date(draw_date[is_post]),
    last_draw_post   = .safe_max_date(draw_date[is_post]),
    any_window_draw  = any(is_window),
    .groups = "drop")
  
  base %>% left_join(cov, by = "patientid") %>%
    mutate(baseline_covered = coalesce(baseline_covered, FALSE),
           recent_baseline_covered = coalesce(recent_baseline_covered, FALSE),
           entry_covered    = coalesce(entry_covered, FALSE),
           n_draws_post     = coalesce(n_draws_post, 0L),
           any_window_draw  = coalesce(any_window_draw, FALSE),
           # death truncated the entry window before a qualifying draw could occur
           died_within_entry = death_observed & !is.na(date_death_fu) &
             date_death_fu <= index_date + entry_window_days,
           source_covered   = index_date >= go_live | any_window_draw,
           coverage_basis   = case_when(index_date >= go_live ~ "post_go_live",
                                        any_window_draw       ~ "pre_go_live_rescue_labs",
                                        TRUE                  ~ "none"))
}

# ------------------------------------------------------------------------------
# Event extractors -> one row per patient (patientid, event_date) for the FIRST
# qualifying post-index event. Patients with no qualifying event do not appear.
# ------------------------------------------------------------------------------

events_from_graded <- function(ae_data, grade_col, threshold, coverage,
                               date_col = samplingdate) {
  ae_data %>%
    transmute(patientid,
              event_date    = as.Date({{ date_col }}),
              grade_numeric = suppressWarnings(
                as.integer(sub("^(\\d+).*", "\\1", as.character({{ grade_col }}))))) %>%
    filter(!is.na(event_date), !is.na(grade_numeric), grade_numeric >= threshold) %>%
    inner_join(coverage %>% select(patientid, index_date, admin_end), by = "patientid") %>%
    filter(event_date > index_date, event_date <= admin_end) %>%
    group_by(patientid) %>% summarise(event_date = .safe_min_date(event_date), .groups = "drop")
}

events_from_table <- function(event_tbl, date_col, coverage) {
  event_tbl %>%
    transmute(patientid, event_date = as.Date({{ date_col }})) %>%
    filter(!is.na(event_date)) %>%
    inner_join(coverage %>% select(patientid, index_date, admin_end), by = "patientid") %>%
    filter(event_date > index_date, event_date <= admin_end) %>%
    group_by(patientid) %>% summarise(event_date = .safe_min_date(event_date), .groups = "drop")
}


# ------------------------------------------------------------------------------
# Core engine: eligibility, outcome + audit, validation, write, one-line report.
# ------------------------------------------------------------------------------

.lab_outcome_core <- function(events_by_patient, coverage, outcome_name,
                              baseline_required = LAB_DEFAULTS$baseline_required,
                              max_gap_days      = LAB_DEFAULTS$max_gap_days,
                              pre_golive_recent_baseline_exempt = FALSE,
                              write_audit       = TRUE) {
  
  cand <- coverage %>%
    left_join(events_by_patient, by = "patientid") %>%
    mutate(
      had_event         = !is.na(event_date),
      baseline_adequate = !baseline_required | baseline_covered,
      
      # Death can truncate the entry window before a qualifying draw. Where
      # capture is complete (post go-live) absence of a draw then means the
      # patient died before being drawn, not that an event was missed, so they
      # are kept as a competing death rather than dropped as "unmonitored".
      # Pre go-live, capture is ambiguous, so this is only allowed (and only when
      # switched on) with tight recent-baseline evidence of active monitoring.
      entry_exempt = died_within_entry &
        (coverage_basis == "post_go_live" |
           (pre_golive_recent_baseline_exempt & recent_baseline_covered)),
      
      # eligibility is coverage-based and outcome-independent; events never override the gate.
      eligible = source_covered & baseline_adequate & (entry_covered | entry_exempt),
      
      eligibility_reason = case_when(
        !source_covered & index_date < go_live ~ "before_registry_coverage",
        !source_covered                        ~ "coverage_unknown",
        baseline_required & !baseline_covered  ~ "missing_baseline",
        entry_exempt                           ~ "eligible_competing_death",
        !entry_covered & n_draws_post == 0L    ~ "no_post_index_measurement",
        !entry_covered                         ~ "insufficient_monitoring",
        had_event                              ~ "eligible_event",
        death_observed                         ~ "eligible_competing_death",
        TRUE                                   ~ "eligible_event_free"
      ),
      
      ascertain_horizon = pmin(last_draw_post + max_gap_days, admin_end, na.rm = TRUE),
      outcome_date      = if_else(eligible & had_event, event_date, as.Date(NA)),
      censor_date       = case_when(
        !eligible ~ as.Date(NA),
        had_event ~ admin_end,
        TRUE      ~ ascertain_horizon)
    ) %>%
    assign_splits(index_date_col = index_date) %>%
    mutate(calendar_period = .calendar_period(index_date, go_live))
  
  stopifnot("coverage must be unique by patientid" = !any(duplicated(cand$patientid)))
  
  # outcome parquet: eligible patients only, SAME schema as before
  result <- cand %>% filter(eligible) %>%
    rename(subject_id = patientid) %>%
    select(subject_id, split, outcome_date, index_date, censor_date)
  
  audit <- cand %>% transmute(
    subject_id = patientid, cohort, outcome = outcome_name, split, index_date,
    source_covered, coverage_basis, baseline_adequate, entry_covered,
    died_within_entry, entry_exempt,
    eligible, eligibility_reason,
    outcome_observed = eligible & !is.na(outcome_date), death_observed,
    measurement_count = n_draws_post,
    first_measurement_date = first_draw_post, last_measurement_date = last_draw_post,
    censor_date = if_else(eligible, censor_date, ascertain_horizon), calendar_period)
  
  validate_lab_outcome(result, audit, outcome_name)
  
  write_parquet(result, file.path(OUTPUT_DIR, paste0(outcome_name, ".parquet")))
  if (write_audit)
    write_parquet(audit, file.path(OUTPUT_DIR, paste0(outcome_name, "__audit.parquet")))
  
  n_cand <- nrow(cand)
  n_elig <- sum(cand$eligible)
  n_ev   <- sum(cand$eligibility_reason == "eligible_event")
  sp     <- table(result$split)
  message(sprintf(
    "%-45s  cand=%d  elig=%d  events=%d (%.1f%% of elig)  train=%d  tuning=%d  held_out=%d",
    outcome_name, n_cand, n_elig, n_ev,
    if (n_elig > 0) 100 * n_ev / n_elig else NA_real_,
    sp["train"] %||% 0L, sp["tuning"] %||% 0L, sp["held_out"] %||% 0L))
  
  invisible(result)
}

validate_lab_outcome <- function(result, audit, outcome_name) {
  stopifnot(
    "one row per patient"          = !any(duplicated(result$subject_id)),
    "censor_date not NA"           = !any(is.na(result$censor_date)),
    "index_date not NA"            = !any(is.na(result$index_date)),
    "censor_date >= index_date"    = all(result$censor_date >= result$index_date),
    "outcome_date > index_date"    =
      all(is.na(result$outcome_date) | result$outcome_date > result$index_date),
    "outcome_date <= censor_date"  =
      all(is.na(result$outcome_date) | result$outcome_date <= result$censor_date),
    "no excluded patient written"  =
      length(intersect(result$subject_id, audit$subject_id[!audit$eligible])) == 0L)
  invisible(TRUE)
}


# ------------------------------------------------------------------------------
# Backwards-compatible wrapper. IDENTICAL signature to the old make_opera_outcome,
# so every existing graded/binary call site runs unchanged. `coverage` defaults
# to the global `lab_coverage`; first_line / patient_data / study_cutoff are kept
# for signature compatibility (coverage already encodes them).
# ------------------------------------------------------------------------------

make_opera_outcome <- function(ae_data,
                               grade_col,
                               date_col,
                               threshold,
                               first_line,
                               patient_data,
                               outcome_name,
                               binary       = FALSE,
                               study_cutoff = STUDY_CUTOFF,
                               coverage     = lab_coverage) {
  
  if (missing(coverage) && !exists("lab_coverage"))
    stop("Build `lab_coverage <- build_lab_coverage(laboratory_measurements, first_line, patient)` before calling make_opera_outcome().")
  
  if (binary && threshold != 1L)
    warning(outcome_name, ": binary = TRUE but threshold = ", threshold,
            ". Forcing threshold to 1.")
  effective_threshold <- if (binary) 1L else as.integer(threshold)
  
  ev <- events_from_graded(ae_data, {{ grade_col }}, effective_threshold,
                           coverage, {{ date_col }})
  .lab_outcome_core(ev, coverage, outcome_name)
}



# ── Outcome calls ─────────────────────────────────────────────────────────────
# Grouped by clinical domain.
# Grade column names follow the AE_ function convention (ae_<name>).
# Verify these match your functions before running — particularly the
# ones from Christian which may differ slightly.
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: SEEMS LIKE SOME OUTCOMES HERE ARE MISSING

first_line <- firstline_tbl_mapped %>%
  select(patientid, index_date = firstline_date) %>%
  mutate(index_date = as.Date(index_date))
message("First-line rows on/after outcome administrative cutoff: ",
        sum(first_line$index_date >= STUDY_CUTOFF, na.rm = TRUE))
first_line <- first_line %>% filter(index_date < STUDY_CUTOFF)

lab_coverage <- build_lab_coverage(laboratory_measurements, first_line, patient)

events_from_table(AKI, samplingdate, lab_coverage)

.lab_outcome_core(
  events_from_table(AKI, samplingdate, lab_coverage),
  lab_coverage, "aki_first_documented"
)

# ============================================================
# ADD-ON 1 of 2:  Hyperphosphatemia
#
# WHERE TO PUT IT: source() or paste this near the END of
# calculate_adverse_events_update.R, after make_opera_outcome()
# is defined and after `laboratory_measurements`, `first_line`
# and `patient` exist (i.e. after the block of lab
# make_opera_outcome() calls, around line 1136). It needs
# nothing that is not already in scope by then.
#
# DESIGN: binary G1 = >ULN, exactly like AE_hyperuricemia and
# AE_LDH_increased. CTCAE has no clean single-value numeric
# grading for hyperphosphatemia (higher grades depend on
# clinical consequences labs cannot see), so a lab-only grader
# is binary. This binary threshold also IS the Cairo-Bishop
# laboratory-TLS phosphate criterion: adult P-Phosphate ULN is
# about 1.45 mmol/L, equal to the Cairo-Bishop cutoff. So the
# same object feeds both the standalone phosphate outcome and
# the TLS co-occurrence in add-on 2.
# ============================================================

# NPU CODE. P-Phosphate is routine chemistry and is very likely
# ALREADY in load_npu_common(). CONFIRM the constant there and
# use it if present. The value below is a PLACEHOLDER in the
# same spirit as the NPU.FIBR placeholder elsewhere in this
# script: verify before trusting, a wrong code fails silently
# (the filter just returns zero rows).
if (!exists("NPU.FOS")) NPU.FOS <- c("NPU03096")   # P-Phosphate; subst.c. (mmol/L) -- CONFIRM

AE_hyperphosphatemia = function(data,
                                c_value = c_value,
                                analysiscode = analysiscode,
                                date_sample = samplingdate,
                                referenceinterval_upperlimit = referenceinterval_upperlimit,
                                days_grade_5 = 30) {
  
  if (!exists('patient'))    { load_dataset('patient') }
  if (!exists('NPU.FOS'))   { load_npu_common() }
  
  data %>%
    filter({{ analysiscode }} %in% NPU.FOS,
           !is.na({{ c_value }})) %>%
    left_join(patient %>% select(patientid, status, date_death_fu), 'patientid') %>%
    mutate(
      time_to_death        = diff_days({{ date_sample }}, date_death_fu),
      value_numeric        = as.numeric({{ c_value }}),
      ULN                  = as.numeric({{ referenceinterval_upperlimit }}),
      # Adult ULN ~1.45 mmol/L (also the Cairo-Bishop TLS cutoff)
      ULN                  = ifelse(is.na(ULN), 1.45, ULN),
      # Lab-ascertainable grade only (G1 = >ULN)
      ae_hyperphosphatemia = ifelse(is.na(ULN) | value_numeric <= ULN, '0', '1'),
      ae_hyperphosphatemia_g5 = factor(
        ifelse(time_to_death < {{ days_grade_5 }} & status == 1, '5', ae_hyperphosphatemia),
        c('0', '1', '5')),
      ae_hyperphosphatemia = factor(ae_hyperphosphatemia, c('0', '1'))
    )
}

HYPERPHOSPHATEMIA = laboratory_measurements %>%
  AE_hyperphosphatemia(analysiscode = NPU, c_value = value2)

# Standalone binary phosphate outcome (mirrors the LDH / hyperuricemia calls)
make_opera_outcome(HYPERPHOSPHATEMIA, ae_hyperphosphatemia, samplingdate, 1,
                   first_line, patient, "hyperphosphatemia", binary = TRUE)


# ── Cytopenias ───────────────────────────────────────────────────────────────

# NOTE; currently only 1914 patients in tuning - could consider allocating more to tuning if needed 

make_opera_outcome(ANEMIA,        ae_anemia,                    samplingdate, 2, first_line, patient, "anemia_g2plus")
make_opera_outcome(ANEMIA,        ae_anemia,                    samplingdate, 3, first_line, patient, "anemia_g3plus")

make_opera_outcome(NEU_DEC,       ae_neutrophil_count_decreased, samplingdate, 2, first_line, patient, "neutropenia_g2plus")
make_opera_outcome(NEU_DEC,       ae_neutrophil_count_decreased, samplingdate, 3, first_line, patient, "neutropenia_g3plus")

make_opera_outcome(PLATELETS_DEC, ae_platelets_decreased,       samplingdate, 2, first_line, patient, "thrombocytopenia_g2plus")
make_opera_outcome(PLATELETS_DEC, ae_platelets_decreased,       samplingdate, 3, first_line, patient, "thrombocytopenia_g3plus")

# The external graded lymphocyte-count-decreased function is v5-derived.
# CTCAE v6 lists "Lymphopenia" only as present (G1), without G2/G3 tiers.
# Keep these historical labels for v4 reproducibility, not as v6 grades.
make_opera_outcome(LYMPH_DEC,     ae_lymphocyte_count_decreased, samplingdate, 2, first_line, patient, "lymphopenia_g2plus")
make_opera_outcome(LYMPH_DEC,     ae_lymphocyte_count_decreased, samplingdate, 3, first_line, patient, "lymphopenia_g3plus")

make_opera_outcome(WBC_DEC,       ae_WBC_decreased,             samplingdate, 2, first_line, patient, "WBC_decreased_g2plus")
make_opera_outcome(WBC_DEC,       ae_WBC_decreased,             samplingdate, 3, first_line, patient, "WBC_decreased_g3plus")


# ── Liver / hepatic ──────────────────────────────────────────────────────────

make_opera_outcome(ALAT, ae_ALAT_increased,      samplingdate, 2, first_line, patient, "ALT_increased_g2plus")
make_opera_outcome(ALAT, ae_ALAT_increased,      samplingdate, 3, first_line, patient, "ALT_increased_g3plus")

make_opera_outcome(ASAT, ae_ASAT_increased,      samplingdate, 2, first_line, patient, "AST_increased_g2plus")
make_opera_outcome(ASAT, ae_ASAT_increased,      samplingdate, 3, first_line, patient, "AST_increased_g3plus")

make_opera_outcome(BILI, ae_bilirubin_increased, samplingdate, 2, first_line, patient, "bilirubin_increased_g2plus")
make_opera_outcome(BILI, ae_bilirubin_increased, samplingdate, 3, first_line, patient, "bilirubin_increased_g3plus")

make_opera_outcome(BASP, ae_BASP_increased,      samplingdate, 2, first_line, patient, "ALP_increased_g2plus")
make_opera_outcome(BASP, ae_BASP_increased,      samplingdate, 3, first_line, patient, "ALP_increased_g3plus")


# ── Renal ────────────────────────────────────────────────────────────────────

make_opera_outcome(CREA, ae_creatinine_increased, samplingdate, 2, first_line, patient, "creatinine_increased_g2plus")
make_opera_outcome(CREA, ae_creatinine_increased, samplingdate, 3, first_line, patient, "creatinine_increased_g3plus")


# ── Coagulation ──────────────────────────────────────────────────────────────

make_opera_outcome(INR,        ae_INR_increased,        samplingdate, 2, first_line, patient, "INR_increased_g2plus")
make_opera_outcome(INR,        ae_INR_increased,        samplingdate, 3, first_line, patient, "INR_increased_g3plus")

make_opera_outcome(FIBRINOGEN, ae_hypofibrinogenemia,  samplingdate, 2, first_line, patient, "hypofibrinogenemia_g2plus")
make_opera_outcome(FIBRINOGEN, ae_hypofibrinogenemia,  samplingdate, 3, first_line, patient, "hypofibrinogenemia_g3plus")


# ── Nutritional / protein ────────────────────────────────────────────────────

make_opera_outcome(HYPOALBU, ae_hypoalbuminemia, samplingdate, 2, first_line, patient, "hypoalbuminemia_g2plus")
make_opera_outcome(HYPOALBU, ae_hypoalbuminemia, samplingdate, 3, first_line, patient, "hypoalbuminemia_g3plus")


# ── Electrolytes — hypo directions ──────────────────────────────────────────

make_opera_outcome(HYPOCALC,      ae_hypocalcemia,    samplingdate, 2, first_line, patient, "hypocalcemia_g2plus")
make_opera_outcome(HYPOCALC,      ae_hypocalcemia,    samplingdate, 3, first_line, patient, "hypocalcemia_g3plus")

make_opera_outcome(HYPONATREMIA,  ae_hyponatremia,    samplingdate, 2, first_line, patient, "hyponatremia_g2plus")
make_opera_outcome(HYPONATREMIA,  ae_hyponatremia,    samplingdate, 3, first_line, patient, "hyponatremia_g3plus")

make_opera_outcome(HYPOKALEMIA,   ae_hypokalemia,     samplingdate, 2, first_line, patient, "hypokalemia_g2plus")
make_opera_outcome(HYPOKALEMIA,   ae_hypokalemia,     samplingdate, 3, first_line, patient, "hypokalemia_g3plus")

make_opera_outcome(HYPOMAGNESEMIA, ae_hypomagnesemia, samplingdate, 2, first_line, patient, "hypomagnesemia_g2plus")
make_opera_outcome(HYPOMAGNESEMIA, ae_hypomagnesemia, samplingdate, 3, first_line, patient, "hypomagnesemia_g3plus")

make_opera_outcome(HYPOGLYC,      ae_hypoglycemia,    samplingdate, 2, first_line, patient, "hypoglycemia_g2plus")
make_opera_outcome(HYPOGLYC,      ae_hypoglycemia,    samplingdate, 3, first_line, patient, "hypoglycemia_g3plus")


# ── Electrolytes — hyper directions ─────────────────────────────────────────
# Lower clinical priority at first line; included for completeness.

make_opera_outcome(HYPERCALCEMIA,   ae_hypercalcemia,         samplingdate, 2, first_line, patient, "hypercalcemia_g2plus")
make_opera_outcome(HYPERCALCEMIA,   ae_hypercalcemia,         samplingdate, 3, first_line, patient, "hypercalcemia_g3plus")

make_opera_outcome(HYPERKALEMIA,    ae_hyperkalemia,          samplingdate, 2, first_line, patient, "hyperkalemia_g2plus")
make_opera_outcome(HYPERKALEMIA,    ae_hyperkalemia,          samplingdate, 3, first_line, patient, "hyperkalemia_g3plus")

make_opera_outcome(HYPERNATREMIA,   ae_hypernatremia,         samplingdate, 2, first_line, patient, "hypernatremia_g2plus")
make_opera_outcome(HYPERNATREMIA,   ae_hypernatremia,         samplingdate, 3, first_line, patient, "hypernatremia_g3plus")

#make_opera_outcome(HYPERMAGNESEMIA, ae_hypermagnesemia,       samplingdate, 2, first_line, patient, "hypermagnesemia_g2plus")
make_opera_outcome(HYPERMAGNESEMIA, ae_hypermagnesemia,       samplingdate, 3, first_line, patient, "hypermagnesemia_g3plus")

make_opera_outcome(HYPERTRIGLY,     ae_hypertriglyceridemia,  samplingdate, 2, first_line, patient, "hypertriglyceridemia_g2plus")
make_opera_outcome(HYPERTRIGLY,     ae_hypertriglyceridemia,  samplingdate, 3, first_line, patient, "hypertriglyceridemia_g3plus")


# ── Hemolysis markers ────────────────────────────────────────────────────────
# Haptoglobin: CTCAE v6 grades as present/absent only — binary.

make_opera_outcome(HAPT, ae_haptoglobin_decreased, samplingdate, 1,
                   first_line, patient, "haptoglobin_decreased",
                   binary = TRUE)


# ── Binary outcomes (no CTCAE grading beyond G1) ────────────────────────────

make_opera_outcome(LDH_INCREASED,      ae_LDH_increased, samplingdate, 1,
                   first_line, patient, "LDH_increased",
                   binary = TRUE)

make_opera_outcome(HYPERURICEMIA, ae_hyperuricemia, samplingdate, 1,
                   first_line, patient, "hyperuricemia",
                   binary = TRUE)


# ── Low-priority / provisional ───────────────────────────────────────────────
# Review event rates before including as OPERA contrastive targets.

# CTCAE v6 excludes leukemia-related leukocytosis and has no G2 category.
# This historical file is an exploratory disease-state marker, not a primary
# toxicity endpoint; removing it from training requires a versioned registry.
make_opera_outcome(LEUKO,     ae_leucocytosis,              samplingdate, 3, first_line, patient, "leukocytosis_g2plus")
make_opera_outcome(EOS,       ae_eosinophilia,              samplingdate, 1, first_line, patient, "eosinophilia", binary = TRUE)
make_opera_outcome(LYMPH_INC, ae_lymphocyte_count_increased, samplingdate, 2, first_line, patient, "lymphocytosis_g2plus")
make_opera_outcome(LYMPH_INC, ae_lymphocyte_count_increased, samplingdate, 3, first_line, patient, "lymphocytosis_g3plus")


# ── Post-run checklist ────────────────────────────────────────────────────────
# After running, inspect the console output for each outcome:
#
#   event rate < ~2%  → too rare; consider dropping from contrastive targets
#                        or keep only for evaluation (g3plus only)
#   event rate > ~80% → too common; limited discrimination value
#
#   tuning n too small → extend TUNING_START back (e.g. to 2021-01-01)
#                        and rerun; confirm with collaborators first
#
#   split imbalance    → if disease subtype distributions differ strongly
#                        across time periods, consider stratified splitting

### 


# ============================================================
# OPERA secondary malignancy outcomes
# Produces two .parquet files:
#   next_malignancy.parquet
#   next_hematological_malignancy.parquet
#
# Operationalization:
#   A post-index diagnosis counts as a NEW malignancy only if
#   its ICD-10 3-character prefix was never observed in that
#   patient's diagnosis history on or before index_date.
#
#   This automatically excludes:
#     - The index hematological disease (pre-existing by definition)
#     - Any other pre-existing malignancies
#     - Relapse coded under the same ICD-10 block
#
# SKS code format (Danish):
#   Diagnosis codes are prefixed with 'D' (e.g. DC619 = ICD-10 C61.9)
#   Non-D prefixes (A, T, etc.) are procedure/action codes and are
#   excluded from all diagnosis matching.
#
# Sources:
#   diagnoses_all          — all diagnosis codes, all registers
#   t_dalycare_diagnoses   — hematological diagnoses, all registers
#
# After running, call diagnose_malignancy_latency() to examine
# the distribution of days from index_date to outcome_date.
# Use this to decide whether a grace period (grace_days) is needed.
# ============================================================

library(dplyr)
library(purrr)
library(arrow)


# ── ICD-10 code definitions ───────────────────────────────────────────────────

# Any malignancy: C00-C97 minus C44 (NMSC) and C80 (unspecified)
# C43 melanoma retained — clinically meaningful, immunosuppression-related
ICD10_CANCER_PREFIXES <- c(
  paste0("C", sprintf("%02d", c(0:43, 45:79, 81:97)))
)

# Haematological malignancies
# C81-C96: lymphomas, leukemias, myeloma, myeloid neoplasms
# D45:     polycythaemia vera
# D46:     myelodysplastic syndromes
# D47:     other uncertain haematological neoplasms (incl. MGUS, SMM)
ICD10_HEME_PREFIXES <- c(
  paste0("C", 81:96),
  "D45", "D46", "D47"
)


# ── SKS-aware ICD-10 helpers ──────────────────────────────────────────────────

# Strip leading SKS 'D' prefix and return 3-character ICD-10 prefix.
# DC619  -> C61
# DZ063A -> Z06
# C61    -> C61  (already bare ICD-10, handled gracefully)
to_icd10_prefix <- function(code_vec) {
  code_clean <- toupper(gsub("\\.", "", as.character(code_vec)))
  # Strip leading D from SKS diagnosis codes
  code_clean <- ifelse(
    startsWith(code_clean, "D") & nchar(code_clean) > 1,
    substr(code_clean, 2, nchar(code_clean)),
    code_clean
  )
  substr(code_clean, 1, 3)
}

# TRUE where code is a SKS diagnosis code (D-prefix) AND its ICD-10
# prefix matches at least one element of prefix_vec.
# Non-D prefix codes (procedures, actions) are excluded at this step.
matches_icd10 <- function(code_vec, prefix_vec) {
  is_dx_code <- startsWith(toupper(as.character(code_vec)), "D")
  is_dx_code & (to_icd10_prefix(code_vec) %in% prefix_vec)
}


# ── Core helper ───────────────────────────────────────────────────────────────
#
# Arguments:
#   dx_data          diagnosis data frame; one row per diagnosis event
#   code_col         unquoted column name for SKS diagnosis code
#   date_col         unquoted column name for diagnosis date
#   include_prefixes ICD-10 3-char prefixes defining the malignancy group
#   first_line       data frame with patientid + index_date
#   patient_data     patient table with patientid + date_death_fu
#   outcome_name     output filename stem
#   grace_days       exclude events within this many days after index_date
#                    (default 0; set after examining latency distribution)
#   study_cutoff     administrative censoring date

make_opera_outcome_malignancy <- function(dx_data,
                                          code_col,
                                          date_col,
                                          include_prefixes,
                                          first_line,
                                          patient_data,
                                          outcome_name,
                                          grace_days   = 0,
                                          study_cutoff = STUDY_CUTOFF) {
  
  # 0. Filter to relevant diagnosis codes and attach index_date
  dx <- dx_data %>%
    filter(matches_icd10({{ code_col }}, include_prefixes)) %>%
    inner_join(
      first_line %>% select(patientid, index_date),
      by = "patientid"
    ) %>%
    mutate(
      dx_date   = as.Date({{ date_col }}),
      dx_prefix = to_icd10_prefix({{ code_col }})
    )
  
  # 1. Per-patient set of ICD-10 prefixes seen ON OR BEFORE index_date
  #    These are known pre-existing malignancies — not new post-index events
  pre_index_prefixes <- dx %>%
    filter(dx_date <= index_date + grace_days) %>%
    group_by(patientid) %>%
    summarise(
      known_prefixes = list(unique(dx_prefix)),
      .groups = "drop"
    )
  
  # 2. Post-index candidates: after index_date + grace period, before censor
  post_index <- dx %>%
    inner_join(
      patient_data %>% select(patientid, date_death_fu),
      by = "patientid"
    ) %>%
    mutate(
      censor_date = pmin(date_death_fu, study_cutoff, na.rm = TRUE)
    ) %>%
    filter(
      dx_date >  index_date + grace_days,
      dx_date <= censor_date
    )
  
  # 3. Exclude codes whose prefix was already seen pre-index
  events <- post_index %>%
    left_join(pre_index_prefixes, by = "patientid") %>%
    mutate(
      known_prefixes = map(known_prefixes,
                           ~ if (is.null(.x)) character(0) else .x),
      is_new = map2_lgl(dx_prefix, known_prefixes, ~ !(.x %in% .y))
    ) %>%
    filter(is_new) %>%
    group_by(patientid) %>%
    slice_min(dx_date, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    select(patientid, outcome_date = dx_date)
  
  # 4. One row per patient in first_line cohort
  result <- first_line %>%
    select(patientid, index_date) %>%
    inner_join(
      patient_data %>% select(patientid, date_death_fu),
      by = "patientid"
    ) %>%
    mutate(
      censor_date = pmin(date_death_fu, study_cutoff, na.rm = TRUE)
    ) %>%
    left_join(events, by = "patientid") %>%
    rename(subject_id = patientid) %>%
    assign_splits(index_date_col = index_date) %>%
    select(subject_id, split, outcome_date, index_date, censor_date)
  
  # 5. Sanity checks
  n_total  <- nrow(result)
  n_events <- sum(!is.na(result$outcome_date))
  n_splits <- table(result$split)
  
  stopifnot(
    "outcome_date must not exceed censor_date" =
      all(is.na(result$outcome_date) |
            result$outcome_date <= result$censor_date),
    "outcome_date must be strictly after index_date" =
      all(is.na(result$outcome_date) |
            result$outcome_date > result$index_date),
    "censor_date must not be NA" = !any(is.na(result$censor_date)),
    "index_date must not be NA"  = !any(is.na(result$index_date))
  )
  
  # 6. Write parquet
  out_path <- file.path(OUTPUT_DIR, paste0(outcome_name, ".parquet"))
  write_parquet(result, out_path)
  
  message(sprintf(
    "%-50s  n=%d  events=%d (%.1f%%)  train=%d  tuning=%d  held_out=%d",
    outcome_name,
    n_total,
    n_events,
    100 * n_events / n_total,
    n_splits["train"]    %||% 0L,
    n_splits["tuning"]   %||% 0L,
    n_splits["held_out"] %||% 0L
  ))
  
  invisible(result)
}


# ── Latency diagnostic ────────────────────────────────────────────────────────
# Run after outcome calls. A spike in the 0-30 day bin suggests synchronous
# diagnoses slipping through — concurrent malignancies coded slightly after
# treatment start. Use to decide on grace_days.

diagnose_malignancy_latency <- function(parquet_path) {
  df <- read_parquet(parquet_path) %>%
    filter(!is.na(outcome_date)) %>%
    mutate(days_to_event = as.integer(outcome_date - index_date))
  
  n <- nrow(df)
  cat("\nLatency distribution:", basename(parquet_path),
      sprintf("(n events = %d)\n", n))
  cat(sprintf("  0-14 days:    %4d (%5.1f%%)\n",
              sum(df$days_to_event <= 14),
              100 * mean(df$days_to_event <= 14)))
  cat(sprintf("  15-30 days:   %4d (%5.1f%%)\n",
              sum(df$days_to_event > 14  & df$days_to_event <= 30),
              100 * mean(df$days_to_event > 14  & df$days_to_event <= 30)))
  cat(sprintf("  31-90 days:   %4d (%5.1f%%)\n",
              sum(df$days_to_event > 30  & df$days_to_event <= 90),
              100 * mean(df$days_to_event > 30  & df$days_to_event <= 90)))
  cat(sprintf("  91-180 days:  %4d (%5.1f%%)\n",
              sum(df$days_to_event > 90  & df$days_to_event <= 180),
              100 * mean(df$days_to_event > 90  & df$days_to_event <= 180)))
  cat(sprintf("  181-365 days: %4d (%5.1f%%)\n",
              sum(df$days_to_event > 180 & df$days_to_event <= 365),
              100 * mean(df$days_to_event > 180 & df$days_to_event <= 365)))
  cat(sprintf("  > 365 days:   %4d (%5.1f%%)\n",
              sum(df$days_to_event > 365),
              100 * mean(df$days_to_event > 365)))
  
  invisible(df)
}


# ── Outcome calls ─────────────────────────────────────────────────────────────
# Adjust code_col and date_col to match your actual column names.

# Any new malignancy (all cancers excl. NMSC C44 and unspecified C80)
make_opera_outcome_malignancy(
  dx_data          = diagnoses_all,
  code_col         = diagnosis,           # adjust to actual column name
  date_col         = date_diagnosis,       # adjust to actual column name
  include_prefixes = ICD10_CANCER_PREFIXES,
  first_line       = first_line,
  patient_data     = patient,
  outcome_name     = "next_malignancy",
  grace_days       = 31
)

# New haematological malignancy (C81-C96, D45-D47)
make_opera_outcome_malignancy(
  dx_data          = t_dalycare_diagnoses,
  code_col         = diagnosis,           # adjust to actual column name
  date_col         = date_diagnosis,       # adjust to actual column name
  include_prefixes = ICD10_HEME_PREFIXES,
  first_line       = first_line,
  patient_data     = patient,
  outcome_name     = "next_hematological_malignancy",
  grace_days       = 31
)


# ── Post-run diagnostics ──────────────────────────────────────────────────────
# Examine latency distributions before deciding on grace_days.
# If 0-30 day bin is large, rerun with grace_days = 90 or 180.

diagnose_malignancy_latency(
  file.path(OUTPUT_DIR, "next_malignancy.parquet"))

diagnose_malignancy_latency(
  file.path(OUTPUT_DIR, "next_hematological_malignancy.parquet"))


### let's also do some procedure code outcomes as well

load_dataset("SDS_t_sksopr", selected_columns = c("v_recnum", "d_odto", "c_opr", "v_otime", "v_ominut"))
load_dataset("SDS_t_sksube", selected_columns = c("v_recnum", "c_oprart", "d_odto", "c_opr", "v_otime", "v_ominut"))
load_dataset("SDS_t_adm", selected_columns = c("k_recnum", "patientid", "d_inddto","v_indtime","v_indminut", "v_udtime", "c_adiag","c_kontaars", "c_spec", "d_uddto", "c_sgh", "c_pattype", "c_indm"))
load_dataset("SDS_kontakter", selected_columns = c("dw_ek_kontakt", "patientid", "dato_start", "tidspunkt_start", "aktionsdiagnose", "kontaktaarsag", "hovedspeciale_ans", "region_ans", "dato_slut", "prioritet", "kontakttype"))
load_dataset("SDS_procedurer_kirurgi", selected_columns=c("procedurekode","dato_start", "tidspunkt_start", "dw_ek_kontakt", "dw_ek_forloeb"))
load_dataset("SDS_procedurer_andre", selected_columns=c("procedurekode", "proceduretype", "dato_start", "tidspunkt_start", "dw_ek_kontakt", "dw_ek_forloeb"))

LPR3_surgery <- SDS_kontakter %>%
  left_join(SDS_procedurer_kirurgi %>% mutate(dw_ek_kontakt = as.integer(dw_ek_kontakt))
            %>% transmute(dw_ek_kontakt, procedurekode,
                          procedure_date = as.Date(dato_start)),
            by = "dw_ek_kontakt", relationship = "many-to-many",
            na_matches = "never")

LPR3 <- SDS_kontakter %>%
  left_join(SDS_procedurer_andre %>% mutate(dw_ek_kontakt = as.integer(dw_ek_kontakt))
            %>% transmute(dw_ek_kontakt, procedurekode,
                          procedure_date = as.Date(dato_start)),
            by = "dw_ek_kontakt", relationship = "many-to-many",
            na_matches = "never")

LPR3 <- LPR3 %>% rbind(LPR3_surgery)

## LPR

# LPR2 record numbers can exceed R's 32-bit integer range. Casting v_recnum
# with as.integer() silently discarded matching procedures and changed the
# exported procedure outcomes. Use the same lossless key representation on
# both sides of the join; blank/missing keys must not match one another.
lpr2_key_text <- function(x) {
  result <- if (is.numeric(x)) trimws(format(x, scientific = FALSE, trim = TRUE))
            else trimws(as.character(x))
  result[is.na(x) | !nzchar(result)] <- NA_character_
  result
}
SDS_t_adm <- SDS_t_adm %>% mutate(.lpr2_join_key = lpr2_key_text(k_recnum))

LPR_surgery <- SDS_t_adm %>%
  left_join(SDS_t_sksopr %>%
              transmute(.lpr2_join_key = lpr2_key_text(v_recnum), c_opr,
                        procedure_date = as.Date(d_odto)) %>%
              filter(!is.na(.lpr2_join_key)),
            by = ".lpr2_join_key", relationship = "many-to-many", na_matches = "never")

LPR <- SDS_t_adm %>%
  left_join(SDS_t_sksube %>%
              transmute(.lpr2_join_key = lpr2_key_text(v_recnum), c_opr,
                        procedure_date = as.Date(d_odto)) %>%
              filter(!is.na(.lpr2_join_key)),
            by = ".lpr2_join_key", relationship = "many-to-many", na_matches = "never")

LPR <- LPR %>% rbind(LPR_surgery)

# A procedure occurring after treatment starts in an earlier admission/contact
# is still post-index. Use the recorded procedure date, not encounter start.
LPR3_summarized <- LPR3 %>% select(patientid, dato_start = procedure_date, procedurekode)
LPR_summarized <- LPR %>% select(patientid, dato_start = procedure_date, procedurekode = c_opr)

CODES <- LPR_summarized %>% rbind(LPR3_summarized) %>% distinct() %>% arrange(dato_start) %>% filter(!is.na(procedurekode))

### 

# ============================================================
# OPERA procedure-based and ICD-10 event outcome files
# Produces one .parquet per outcome in OPERA format:
#   subject_id | split | outcome_date | index_date | censor_date
#
# Requires sourcing first:
#   make_opera_outcomes.R        (defines assign_splits, %||%, STUDY_CUTOFF)
#   make_opera_outcomes_malignancy.R  (defines make_opera_outcome_malignancy,
#                                      to_icd10_prefix, matches_icd10)
# ============================================================

library(dplyr)
library(arrow)


# ── Core procedure helper ─────────────────────────────────────────────────────
# Takes a pre-filtered event dataset (one row per event) and finds the
# first post-index occurrence per patient.

make_opera_outcome_procedure <- function(event_data,
                                         date_col,
                                         first_line,
                                         patient_data,
                                         outcome_name,
                                         study_cutoff = STUDY_CUTOFF) {
  
  events <- event_data %>%
    inner_join(
      first_line %>% select(patientid, index_date),
      by = "patientid"
    ) %>%
    inner_join(
      patient_data %>% select(patientid, date_death_fu),
      by = "patientid"
    ) %>%
    mutate(
      event_date  = as.Date({{ date_col }}),
      censor_date = pmin(date_death_fu, study_cutoff, na.rm = TRUE)
    ) %>%
    filter(
      event_date >  index_date,
      event_date <= censor_date
    ) %>%
    group_by(patientid) %>%
    slice_min(event_date, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    select(patientid, outcome_date = event_date)
  
  result <- first_line %>%
    select(patientid, index_date) %>%
    inner_join(
      patient_data %>% select(patientid, date_death_fu),
      by = "patientid"
    ) %>%
    mutate(
      censor_date = pmin(date_death_fu, study_cutoff, na.rm = TRUE)
    ) %>%
    left_join(events, by = "patientid") %>%
    rename(subject_id = patientid) %>%
    assign_splits(index_date_col = index_date) %>%
    select(subject_id, split, outcome_date, index_date, censor_date)
  
  n_total  <- nrow(result)
  n_events <- sum(!is.na(result$outcome_date))
  n_splits <- table(result$split)
  
  stopifnot(
    "outcome_date must not exceed censor_date" =
      all(is.na(result$outcome_date) |
            result$outcome_date <= result$censor_date),
    "outcome_date must be strictly after index_date" =
      all(is.na(result$outcome_date) |
            result$outcome_date > result$index_date),
    "censor_date must not be NA" = !any(is.na(result$censor_date)),
    "index_date must not be NA"  = !any(is.na(result$index_date))
  )
  
  out_path <- file.path(OUTPUT_DIR, paste0(outcome_name, ".parquet"))
  write_parquet(result, out_path)
  
  message(sprintf(
    "%-45s  n=%d  events=%d (%.1f%%)  train=%d  tuning=%d  held_out=%d",
    outcome_name,
    n_total,
    n_events,
    100 * n_events / n_total,
    n_splits["train"]    %||% 0L,
    n_splits["tuning"]   %||% 0L,
    n_splits["held_out"] %||% 0L
  ))
  
  invisible(result)
}


# ── Overall survival ──────────────────────────────────────────────────────────

overall_survival <- first_line %>%
  select(patientid, index_date) %>%
  inner_join(
    patient %>% select(patientid, date_death_fu, status),
    by = "patientid"
  ) %>%
  mutate(
    censor_date  = pmin(date_death_fu, STUDY_CUTOFF, na.rm = TRUE),
    # A death after the administrative cutoff is censored, not converted into
    # an event on the cutoff date.
    outcome_date = if_else(status == 1 & !is.na(date_death_fu) &
                             date_death_fu <= STUDY_CUTOFF,
                           as.Date(date_death_fu), as.Date(NA))
  ) %>%
  rename(subject_id = patientid) %>%
  assign_splits(index_date_col = index_date) %>%
  select(subject_id, split, outcome_date, index_date, censor_date)

stopifnot(
  "overall survival has one row per patient" = !anyDuplicated(overall_survival$subject_id),
  "overall survival censor is not before index" =
    all(overall_survival$censor_date >= overall_survival$index_date),
  "overall survival event follows index" =
    all(is.na(overall_survival$outcome_date) |
          overall_survival$outcome_date > overall_survival$index_date),
  "overall survival event does not exceed censor" =
    all(is.na(overall_survival$outcome_date) |
          overall_survival$outcome_date <= overall_survival$censor_date)
)

write_parquet(overall_survival,
              file.path(OUTPUT_DIR, "overall_survival.parquet"))

n_os    <- nrow(overall_survival)
n_os_ev <- sum(!is.na(overall_survival$outcome_date))
n_os_sp <- table(overall_survival$split)
message(sprintf(
  "%-45s  n=%d  events=%d (%.1f%%)  train=%d  tuning=%d  held_out=%d",
  "overall_survival", n_os, n_os_ev, 100 * n_os_ev / n_os,
  n_os_sp["train"] %||% 0L, n_os_sp["tuning"] %||% 0L,
  n_os_sp["held_out"] %||% 0L))


# ── TTNT ──────────────────────────────────────────────────────────────────────
# Adjust ttnt_start_date to your actual column name in ttnt_tbl

# make_opera_outcome_procedure(
#   event_data   = ttnt_tbl %>% rename(dato_start = ttnt_start_date),
#   date_col     = dato_start,
#   first_line   = first_line,
#   patient_data = patient,
#   outcome_name = "TTNT"
# )


# ── Build hospitalisation dataset ─────────────────────────────────────────────
# Definition from "Databasen for Akutte Hospitalskontakter"
# Sources: SDS_t_adm and SDS_kontakter
#
# ADM filter:
#   Inpatient:  c_pattype == 1 AND c_indm == "1"
#   Acute:      c_pattype == 2 AND c_indm == "1"
#
# CONTACTS filter:
#   prioritet == "ATA1" AND kontakttype == "ALCA00"
#
# Exclude psychiatry specialties from both.
#
# NOTE: date column in ADM is d_inddto (admission date).
# Adjust if your R datasets use a different column name.

# ============================================================
# OPERA hospitalisation outcome
# Unified across LPR2 (SDS_t_adm) and LPR3 (SDS_kontakter)
#
# Design:
#   1. Pull acute physical contacts (+ candidate planned stays) from both eras
#   2. Merge contacts that are close in time into episodes (configurable gap)
#   3. Classify each episode:
#        - priority: acute if ANY contact in the episode is acute (precedence)
#        - hospitalisation: explicit LPR2 inpatient contact OR crosses a
#          duration / overnight threshold (LPR3 has no inpatient flag)
#   4. Split into the predicted outcome (acute hospitalisations) and a
#      separate planned-admission stream (treatment-delivery token)
# ============================================================

library(dplyr)
library(lubridate)

# ---- 0. Config ------------------------------------------------------------
MERGE_GAP_HOURS <- 4      # contacts within this gap = one episode (DrAstma uses 4h)
MIN_STAY_HOURS  <- 12     # episode is a hospitalisation if it lasts >= this ...
USE_OVERNIGHT   <- TRUE   # ... OR (if TRUE) if it crosses a calendar midnight
INCLUDE_ER      <- TRUE   # include acute ER / acute-ambulant contacts in LPR2

PSYCH_SPEC_LPR2 <- c(50, 52)                       # ADJUST to specialeklassifikation
PSYCHIATRY_SPECIALTIES <- c("psykiatri",
                            "boerne- og ungdomspsykiatri")  # ADJUST to hovedspeciale_ans values

# Time imputation when hour is missing (SDS teknikergruppe convention)
DEFAULT_IN_HOUR  <- 9
DEFAULT_OUT_HOUR <- 21

# helper: POSIXct from date + optional hour/minute
.mk_dt <- function(date, hour = NA, minute = 0L, default_hour = 9L) {
  d <- as.Date(date)
  h <- ifelse(is.na(hour),   default_hour, as.integer(hour))
  m <- ifelse(is.na(minute), 0L,           as.integer(minute))
  as.POSIXct(d, tz = "UTC") + h * 3600 + m * 60
}

# ---- 1. LPR2: SDS_t_adm ---------------------------------------------------
# c_pattype: 0 = inpatient (heldoegn), 2 = ambulant, 3 = skadestue (pre-2014)
# c_indm:    "1" = acute, "2" = planned/elective  (ADJUST if stored numeric)
# Inpatient stays use real admission/discharge; ambulant/ER are point events
# (their ambulant-forloeb end date can run far into the future and would
#  otherwise swallow the merge).
lpr2 <- SDS_t_adm %>%
  filter(
    c_pattype == 0 |
      (INCLUDE_ER & c_pattype %in% c(2, 3) & c_indm == "1"),
    !c_spec %in% PSYCH_SPEC_LPR2
  ) %>%
  transmute(
    patientid,
    inpatient_flag = c_pattype == 0,
    acute          = c_indm == "1" | c_pattype == 3,   # skadestue is acute
    planned        = c_pattype == 0 & c_indm == "2",
    specialty      = as.character(c_spec),
    source         = "LPR2",
    t_start = .mk_dt(d_inddto, v_indtime, v_indminut, DEFAULT_IN_HOUR),
    t_end   = if_else(
      c_pattype == 0,
      .mk_dt(d_uddto, v_udtime, 0L, DEFAULT_OUT_HOUR),
      t_start                                           # point event for ambulant/ER
    )
  )

# ---- 2. LPR3: SDS_kontakter -----------------------------------------------
# kontakttype "ALCA00" = physical attendance (NOT inpatient; LPR3 dropped the
#   inpatient/outpatient distinction, so inpatient is derived from duration).
# prioritet "ATA1" = acute; anything else = planned.
# Keep all acute contacts (any length) so short acute seeds survive the merge;
# keep planned contacts only if they already look like a stay.
lpr3 <- SDS_kontakter %>%
  filter(
    kontakttype == "ALCA00",
    !tolower(hovedspeciale_ans) %in% PSYCHIATRY_SPECIALTIES
  ) %>%
  transmute(
    patientid,
    inpatient_flag = NA,                               # unknown in LPR3
    acute          = prioritet == "ATA1",
    planned        = prioritet != "ATA1",
    specialty      = as.character(hovedspeciale_ans),
    source         = "LPR3",
    # ADJUST: if you have separate date + time columns, fold them with .mk_dt
    t_start = as.POSIXct(dato_start, tz = "UTC"),
    t_end   = as.POSIXct(dato_slut,  tz = "UTC")
  ) %>%
  mutate(dur_h = as.numeric(difftime(t_end, t_start, units = "hours"))) %>%
  filter(acute | dur_h >= MIN_STAY_HOURS) %>%
  select(-dur_h)

# ---- 3. Merge contacts into episodes --------------------------------------
contacts <- bind_rows(lpr2, lpr3) %>%
  filter(!is.na(t_start)) %>%
  mutate(t_end = if_else(is.na(t_end) | t_end < t_start, t_start, t_end))

merge_episodes <- function(df, gap_hours) {
  gap_s <- gap_hours * 3600
  df %>%
    arrange(patientid, t_start) %>%
    group_by(patientid) %>%
    mutate(
      prev_end_max = lag(cummax(as.numeric(t_end))),
      new_ep       = is.na(prev_end_max) |
        (as.numeric(t_start) - prev_end_max) > gap_s,
      episode_id   = cumsum(new_ep)
    ) %>%
    group_by(patientid, episode_id) %>%
    summarise(
      t_start       = min(t_start),
      t_end         = max(t_end),
      is_acute      = any(acute,         na.rm = TRUE),  # acute takes precedence
      has_inpatient = any(inpatient_flag, na.rm = TRUE),
      specialty     = specialty[which.min(t_start)],
      source        = paste(sort(unique(source)), collapse = "+"),
      n_contacts    = n(),
      .groups = "drop"
    )
}

episodes <- merge_episodes(contacts, MERGE_GAP_HOURS) %>%
  mutate(
    duration_h    = as.numeric(difftime(t_end, t_start, units = "hours")),
    crosses_night = as.Date(t_end) > as.Date(t_start),
    is_hospitalisation =
      has_inpatient |
      duration_h >= MIN_STAY_HOURS |
      (USE_OVERNIGHT & crosses_night)
  )

# ---- 4. Outputs -----------------------------------------------------------
# (a) The predicted outcome: acute hospitalisations
hospitalisation_acute <- episodes %>%
  filter(is_acute, is_hospitalisation) %>%
  transmute(
    patientid,
    dato_start = as.Date(t_start),
    dato_end   = as.Date(t_end),
    duration_h, specialty, source, n_contacts,
    event = "hosp_acute"
  )


make_opera_outcome_procedure(
  event_data   = hospitalisation_acute,
  date_col     = dato_start,
  first_line   = first_line,
  patient_data = patient,
  outcome_name = "hospitalisation"
)

# 
# # (b) Separate token stream: planned (treatment-related) admissions.
# #     Kept OUT of the outcome, kept IN the input sequence for OPERA.
# admission_planned <- episodes %>%
#   filter(!is_acute, is_hospitalisation) %>%
#   transmute(
#     patientid,
#     dato_start = as.Date(t_start),
#     dato_end   = as.Date(t_end),
#     duration_h, specialty, source, n_contacts,
#     event = "admission_planned"
#   )
# 
# PSYCHIATRY_SPECIALTIES <- c("psykiatri", "børne- og ungdomspsykiatri")
# 
# # Acute and inpatient admissions from ADM
# HOSPITALISATION_ADM <- SDS_t_adm %>%
#   filter(
#     ((c_pattype == 1) | (c_pattype == 0)) & c_indm == "1",
#     !c_spec %in% c(50, 52) # psychiatry and related
#   ) %>%
#   select(patientid, dato_start = d_inddto)   # adjust date col if needed
# 
# # Acute contacts from CONTACTS register
# HOSPITALISATION_CONTACTS <- SDS_kontakter %>%
#   filter(
#     prioritet    == "ATA1",
#     kontakttype  == "ALCA00",
#     !tolower(hovedspeciale_ans) %in% PSYCHIATRY_SPECIALTIES
#   ) %>%
#   select(patientid, dato_start = dato_start)  # adjust date col if needed
# 
# # Union: a hospitalisation event is captured in either source
# HOSPITALISATION <- bind_rows(
#   HOSPITALISATION_ADM,
#   HOSPITALISATION_CONTACTS
# ) %>%
#   distinct() %>%
#   filter(!is.na(dato_start))
# 
# make_opera_outcome_procedure(
#   event_data   = HOSPITALISATION,
#   date_col     = dato_start,
#   first_line   = first_line,
#   patient_data = patient,
#   outcome_name = "hospitalisation"
# )


# ── Transfusions ──────────────────────────────────────────────────────────────

RBC_TRANSFUSION <- CODES %>%
  filter(procedurekode %in% c("BOQA0", "BOQA00", "BOQA01", "BOQA02", "BOQA03", "BOQA04", "BOQA05"))

make_opera_outcome_procedure(
  event_data   = RBC_TRANSFUSION,
  date_col     = dato_start,
  first_line   = first_line,
  patient_data = patient,
  outcome_name = "RBC_transfusion"
)

PLATELET_TRANSFUSION <- CODES %>%
  filter(procedurekode %in% c("BOQC", "BOQC3"))

make_opera_outcome_procedure(
  event_data   = PLATELET_TRANSFUSION,
  date_col     = dato_start,
  first_line   = first_line,
  patient_data = patient,
  outcome_name = "platelet_transfusion"
)

# Composite: any transfusion
ANY_TRANSFUSION <- bind_rows(
  RBC_TRANSFUSION      %>% select(patientid, dato_start),
  PLATELET_TRANSFUSION %>% select(patientid, dato_start)
) %>%
  distinct()

make_opera_outcome_procedure(
  event_data   = ANY_TRANSFUSION,
  date_col     = dato_start,
  first_line   = first_line,
  patient_data = patient,
  outcome_name = "any_transfusion"
)


# ── Antibiotics ───────────────────────────────────────────────────────────────
# NOTE: expect very high event rate — nearly all patients on cytotoxic
# chemotherapy receive antibiotics. Consider whether this belongs in
# training or evaluation only.

ANTI_BIOTICS <- CODES %>%
  filter(str_detect(procedurekode, "^BPH"))

make_opera_outcome_procedure(
  event_data   = ANTI_BIOTICS,
  date_col     = dato_start,
  first_line   = first_line,
  patient_data = patient,
  outcome_name = "antibiotics"
)


# ── Respiratory support ───────────────────────────────────────────────────────

RESPIRATORY_SUPPORT <- CODES %>%
  filter(procedurekode %in% c("BGDA0", "BGDA7", "BGDA1", "BGDA2"))

make_opera_outcome_procedure(
  event_data   = RESPIRATORY_SUPPORT,
  date_col     = dato_start,
  first_line   = first_line,
  patient_data = patient,
  outcome_name = "respiratory_support"
)


# ── Dialysis ──────────────────────────────────────────────────────────────────

DIALYSIS <- CODES %>%
  filter(str_detect(procedurekode, "^BJFD"))

make_opera_outcome_procedure(
  event_data   = DIALYSIS,
  date_col     = dato_start,
  first_line   = first_line,
  patient_data = patient,
  outcome_name = "dialysis"
)


# ── Emergency stay ────────────────────────────────────────────────────────────

EMERGENCY_STAY <- CODES %>% 
  filter(str_detect(procedurekode, "^NAB")) %>%
  filter(!str_detect(procedurekode, "NAB1"))

make_opera_outcome_procedure(
  event_data   = EMERGENCY_STAY,
  date_col     = dato_start,
  first_line   = first_line,
  patient_data = patient,
  outcome_name = "emergency_stay"
)


# ── Bone fractures ────────────────────────────────────────────────────────────
# ICD-10 diagnosis codes in SKS format (DS02x, DM80x, etc.)
# S-codes: traumatic fractures by anatomical site
# M80:     osteoporosis with fracture
# M84.3:   stress fracture
# M84.4:   pathological fracture
# Uses diagnoses_all with SKS D-prefix stripping (from malignancy script).

bone_fractures_dx <- diagnoses_all %>%
  filter(startsWith(toupper(as.character(diagnosis)), "D")) %>%
  mutate(
    code_bare = substr(toupper(gsub("\\.", "", diagnosis)), 2, 99)
  ) %>%
  filter(
    substr(code_bare, 1, 3) %in% c("S02", "S12", "S22", "S32", "S42",
                                   "S52", "S62", "S72", "S82", "S92",
                                   "T02") |
      startsWith(code_bare, "M80") |
      startsWith(code_bare, "M843") |   # M84.3 stress fracture
      startsWith(code_bare, "M844")     # M84.4 pathological fracture
  ) %>%
  select(patientid, dato_start = date_diagnosis)  # adjust date col if needed

make_opera_outcome_procedure(
  event_data   = bone_fractures_dx,
  date_col     = dato_start,
  first_line   = first_line,
  patient_data = patient,
  outcome_name = "bone_fracture"
)


# ── Cardiovascular outcomes (ICD-10 via diagnoses_all) ────────────────────────
# Uses pre-index exclusion logic from make_opera_outcome_malignancy:
# a post-index code only counts as new if not present pre-index.
# Appropriate here since pre-existing AFib, HF etc. should not count.

ICD10_AFIB_PREFIXES   <- c("I48")
ICD10_VTE_PREFIXES    <- c("I26", "I80", "I81", "I82")
ICD10_HF_PREFIXES     <- c("I50")
ICD10_MI_PREFIXES     <- c("I21", "I22")
ICD10_STROKE_PREFIXES <- c("I63", "I64", "G45")
ICD10_MACE_PREFIXES   <- c(ICD10_AFIB_PREFIXES, ICD10_VTE_PREFIXES,
                           ICD10_HF_PREFIXES,   ICD10_MI_PREFIXES,
                           ICD10_STROKE_PREFIXES)

for (outcome in list(
  list(ICD10_AFIB_PREFIXES,   "atrial_fibrillation"),
  list(ICD10_VTE_PREFIXES,    "VTE"),
  list(ICD10_HF_PREFIXES,     "heart_failure"),
  list(ICD10_MI_PREFIXES,     "myocardial_infarction"),
  list(ICD10_STROKE_PREFIXES, "stroke_TIA"),
  list(ICD10_MACE_PREFIXES,   "MACE_composite")
)) {
  make_opera_outcome_malignancy(
    dx_data          = diagnoses_all,
    code_col         = diagnosis,
    date_col         = date_diagnosis,
    include_prefixes = outcome[[1]],
    first_line       = first_line,
    patient_data     = patient,
    outcome_name     = outcome[[2]],
    grace_days       = 0
  )
}


# ── Infection outcomes (ICD-10 via diagnoses_all) ─────────────────────────────
# Sepsis, herpes zoster, invasive fungal, PCP
# No pre-index exclusion applied here — infections are acute events,
# not chronic conditions. Using make_opera_outcome_procedure logic
# (first post-index event, no pre-index filter).
# If a patient had prior sepsis, a new post-index episode is still a
# meaningful new event.

ICD10_SEPSIS_PREFIXES    <- c("A40", "A41")
ICD10_ZOSTER_PREFIXES    <- c("B02")
ICD10_FUNGAL_PREFIXES    <- c("B44", "B45", "B46")
ICD10_PCP_PREFIXES       <- c("B59")
ICD10_SERIOUS_INF_PREFIXES <- c(ICD10_SEPSIS_PREFIXES,
                                ICD10_FUNGAL_PREFIXES,
                                ICD10_PCP_PREFIXES)

# For infections we use a stripped-down version: no pre-index exclusion
make_opera_outcome_infection <- function(dx_data, code_col, date_col,
                                         include_prefixes, first_line,
                                         patient_data, outcome_name,
                                         study_cutoff = STUDY_CUTOFF) {
  events <- dx_data %>%
    filter(matches_icd10({{ code_col }}, include_prefixes)) %>%
    inner_join(first_line %>% select(patientid, index_date), by = "patientid") %>%
    inner_join(patient_data %>% select(patientid, date_death_fu), by = "patientid") %>%
    mutate(
      dx_date     = as.Date({{ date_col }}),
      censor_date = pmin(date_death_fu, study_cutoff, na.rm = TRUE)
    ) %>%
    filter(dx_date > index_date, dx_date <= censor_date) %>%
    group_by(patientid) %>%
    slice_min(dx_date, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    select(patientid, outcome_date = dx_date)
  
  result <- first_line %>%
    select(patientid, index_date) %>%
    inner_join(patient_data %>% select(patientid, date_death_fu), by = "patientid") %>%
    mutate(censor_date = pmin(date_death_fu, study_cutoff, na.rm = TRUE)) %>%
    left_join(events, by = "patientid") %>%
    rename(subject_id = patientid) %>%
    assign_splits(index_date_col = index_date) %>%
    select(subject_id, split, outcome_date, index_date, censor_date)
  
  n_total  <- nrow(result)
  n_events <- sum(!is.na(result$outcome_date))
  n_splits <- table(result$split)
  
  stopifnot(
    "outcome_date must not exceed censor_date" =
      all(is.na(result$outcome_date) | result$outcome_date <= result$censor_date),
    "outcome_date must be strictly after index_date" =
      all(is.na(result$outcome_date) | result$outcome_date > result$index_date),
    "censor_date must not be NA" = !any(is.na(result$censor_date)),
    "index_date must not be NA"  = !any(is.na(result$index_date))
  )
  
  out_path <- file.path(OUTPUT_DIR, paste0(outcome_name, ".parquet"))
  write_parquet(result, out_path)
  message(sprintf(
    "%-45s  n=%d  events=%d (%.1f%%)  train=%d  tuning=%d  held_out=%d",
    outcome_name, n_total, n_events, 100 * n_events / n_total,
    n_splits["train"] %||% 0L, n_splits["tuning"] %||% 0L,
    n_splits["held_out"] %||% 0L))
  invisible(result)
}

for (outcome in list(
  list(ICD10_SEPSIS_PREFIXES,      "sepsis"),
  list(ICD10_ZOSTER_PREFIXES,      "herpes_zoster"),
  list(ICD10_FUNGAL_PREFIXES,      "invasive_fungal"),
  list(ICD10_SERIOUS_INF_PREFIXES, "serious_infection_composite")
)) {
  make_opera_outcome_infection(
    dx_data          = diagnoses_all,
    code_col         = diagnosis,
    date_col         = date_diagnosis,
    include_prefixes = outcome[[1]],
    first_line       = first_line,
    patient_data     = patient,
    outcome_name     = outcome[[2]]
  )
}


#### INFECTIONS

# Function to find IV infections in SMR
AE_infection_SMR_SP <- function(data, days_min_AB_duration = 1.00, days_between_separating_infections = 7.00){
  #' @title
  #' AE_infection_SMR_SP
  #' @description  
  #' Defines infections based on duration of iv. antimicrobial therapy (AB_min_duration [days]: default 1.00 days) and days between AB separating 2 infectious events (days_between_separating_infections: default 7.00 days) first using ATC_AB(c_atc).  
  #' @note
  #' Obs if you should filter some prophylactic antimicrobial therapy out before calling the function and rename SP administration column to align with SMR adm. column. 
  #' Obs for CADD pumps with no administration after 'bolus and first adm. of 24 hours' is not counted. We do not know if they are getting it 24 hours exactly
  #' Obs on data coverage. SP from 2016/2017, SMR starts May 2018
  #' 
  #' @example 
  #' load_dataset('SP_AdministreretMedicin', cohort$patientid) #SP
  #' load_dataset('SDS_indberetningmedpris', cohort$patientid) #SMR
  #' 
  #' IV_AB_SP = SP_AdministreretMedicin_subset %>% ATC_AB() %>%
  #'      #filter(!atc %in% c('J01EE01', 'J05AB01', 'J05AB11', 'P02') ) %>% # filter out prophylactic AB - maybe not necessary
  #'      rename(d_adm = taken_time) #rename to use function across these data sources SMR/SP  (taken_time is in POSIXct format as well as d_adm)
  #'      select(-date_received, -datetime_created) %>% # these are a lot of dublicates - maybe better with new filesystem for DALYCARE
  #'      distinct(across(-medication_name), .keep_all = TRUE)
  #'      
  #' 
  #' IV_AB_SMR <- SDS_indberetningmedpris_subset %>% ATC_AB(c_atc) %>%
  #'      #filter(!c_atc %in% c('J01EE01', 'J05AB01', 'J05AB11', 'P02'))  %>% # filter out prophylactic AB - maybe not necessary
  #'      select(-date_received, -datetime_created) %>%
  #'      distinct(across(-v_laegemiddelnavn), .keep_all = TRUE)
  #' 
  #' all_IV_infections <- bind_rows(IV_AB_SP, IV_AB_SMR) %>%
  #' 
  #' AE_infection_SMR_SP(days_min_AB_duration = 1, days_between_separating_infections = 7)
  #' 
  #' 
  #' To use in document: source('functions/R/AE_infection_SMR_SP.R') #load function
  
  
  data %>% 
    filter(
      if_any(
        any_of(c("c_adm_vej", "zc_admin_route_name")), #c_adm_vej from SMR, zc_admin_route_name from SP
        ~ .x %in% c("IV", "Intravenøs anvendelse", "IntravenøsAnvendelse", 'Intravenøsanvendelse')
      )) %>%
    group_by(patientid) %>% 
    arrange(patientid, d_adm) %>% 
    mutate(DIFF = diff_days(lag(d_adm), d_adm)) %>% 
    ungroup() %>% 
    mutate(N = ifelse(DIFF > {{days_between_separating_infections}}, row_number(), NA), 
           N = ifelse(row_number() == 1, 1, N)) %>% 
    mutate(N = as.numeric(factor(N))) %>% 
    fill(N) %>% 
    ungroup() %>% 
    mutate(DIFF = ifelse(is.na(DIFF), 0, DIFF),
           DIFF = ifelse(DIFF >{{days_between_separating_infections}}, 0, DIFF)) %>% 
    group_by(patientid, N) %>% 
    mutate(n_days_IVAB = round(sum(DIFF),2)) %>%
    slice(1) %>% 
    ungroup() %>% 
    filter(n_days_IVAB >= {{days_min_AB_duration}}) %>%  #>24h
    group_by(patientid) %>% 
    mutate(N = row_number())  %>%
    ungroup() %>% 
    mutate(date_infection = as_date(d_adm)) %>% 
    select(patientid, date_infection , DIFF, n_inf = N, n_days_IVAB, everything())
}
#mutate(is_CADD = str_detect(tolower(v_laegemiddelnavn), 'cadd') | str_detect(v_lagemiddelnavn, '12|14|16')) 
# If we want to find cadd administrations

# Function to find infections based on prescriped med in SDS Epikur
AE_precribed_inf <- function(data, days_min_AB_duration = 1.00, days_between_separating_infections = 7.00){
  #' @title
  #' 
  #' @description
  #' 
  #' @note
  #' Obs if you should filter some prophylactic antimicrobial therapy out before calling the function 
  #' 
  #' @example 
  #' load_dataset('SDS_epikur', cohort$patientid) #prescriped
  #' 
  #' presc_inf = SDS_epikur_subset %>% ATC_AB() %>%
  #'      filter(!atc %in% c('J01EE01', 'J05AB01', 'J05AB11', 'P02') ) %>% # filter out prophylactic AB
  #'      rename(d_adm = 'eksd') #rename  (taken_time is in POSIXct format obs format)
  #'      
  #' all_precribed_infections <- presc_inf %>%
  #' AE_precribed_inf(days_min_AB_duration = 1, days_between_separating_infections = 7) or 14 days?
  #' 
  #' 
  
  data %>% 
    group_by(patientid) %>% 
    arrange(patientid, d_adm) %>% 
    mutate(DIFF = diff_days(lag(d_adm), d_adm)) %>% 
    ungroup() %>% 
    mutate(N = ifelse(DIFF > {{days_between_separating_infections}}, row_number(), NA), 
           N = ifelse(row_number() == 1, 1, N)) %>% 
    mutate(N = as.numeric(factor(N))) %>% 
    fill(N) %>% 
    ungroup() %>% 
    mutate(DIFF = ifelse(is.na(DIFF), 0, DIFF),
           DIFF = ifelse(DIFF >{{days_between_separating_infections}}, 0, DIFF)) %>% 
    group_by(patientid, N) %>% 
    mutate(n_days_AB = round(sum(DIFF),2)) %>%
    slice(1) %>% 
    ungroup() %>% 
    filter(n_days_AB >= {{days_min_AB_duration}}) %>%  #>24h
    group_by(patientid) %>% 
    mutate(N = row_number())  %>%
    ungroup() %>% 
    mutate(date_infection = as_date(d_adm)) %>% 
    select(patientid, date_infection , DIFF, n_inf = N, everything())
}  

### first infections from SMR

load_dataset('SDS_indberetningmedpris') #SMR

IV_AB_SMR <- SDS_indberetningmedpris %>% ATC_AB(c_atc) %>%
     #filter(!c_atc %in% c('J01EE01', 'J05AB01', 'J05AB11', 'P02'))  %>% # filter out prophylactic AB - maybe not necessary
     select(-date_received, -datetime_created) %>%
     distinct(across(-v_laegemiddelnavn), .keep_all = TRUE)

all_IV_infections <- IV_AB_SMR %>%
  AE_infection_SMR_SP(days_min_AB_duration = 1, days_between_separating_infections = 7)

# then prescription infections

load_dataset('SDS_ekokur') #prescriped

presc_inf = SDS_ekokur %>% ATC_AB() %>%
     filter(!atc %in% c('J01EE01', 'J05AB01', 'J05AB11', 'P02') ) %>% # filter out prophylactic AB
     rename(d_adm = 'eksd') #rename  (taken_time is in POSIXct format obs format)

all_precribed_infections <- presc_inf %>%
  AE_precribed_inf(days_min_AB_duration = 1, days_between_separating_infections = 7) 

# Medication-episode proxies do not require a laboratory measurement, baseline,
# or last-draw censor. Use the administrative non-lab constructor; the SMR
# reporting-start eligibility mask is applied later to infections_IV_based.
make_opera_outcome_procedure(all_IV_infections, date_infection, first_line,
                             patient, "infections_IV_based")
make_opera_outcome_procedure(all_precribed_infections, date_infection, first_line,
                             patient, "infections_prescription_based")

### NEW STUFF WITH EXCLUSION LOGIC


suppressPackageStartupMessages({
  library(arrow); library(dplyr); library(purrr); library(rlang)
})

# guard-define shared config, in case this is sourced on its own
if (!exists("OUTPUT_DIR"))   OUTPUT_DIR   <- "opera_outcomes"
if (!exists("STUDY_CUTOFF")) STUDY_CUTOFF <- as.Date("2026-03-19")
if (!exists("%||%")) `%||%` <- function(x, y) if (is.null(x)) y else x
# assign_splits() is expected to already be defined by the main script.

# earliest date at which >= k distinct axes fall inside a trailing
# window of width w days. Inputs are one patient's abnormal
# (date, axis) rows. Returns a Date, or NA if never satisfied.
.first_cooccurrence <- function(dates, axes, k, w) {
  o <- order(dates); dates <- dates[o]; axes <- axes[o]
  n <- length(dates)
  if (n == 0L) return(as.Date(NA))
  for (j in seq_len(n)) {
    in_win <- dates >= (dates[j] - w) & dates <= dates[j]
    if (length(unique(axes[in_win])) >= k) return(dates[j])
  }
  as.Date(NA)
}

# distinct (patientid, sample_date) where an AXIS is abnormal.
# axis_spec is either one spec list(data=, grade_col=, threshold=)
# or a list of such specs (OR'd together into a single axis).
.axis_abnormal_days <- function(axis_spec) {
  if (!is.null(axis_spec$data)) axis_spec <- list(axis_spec)  # single -> list of one
  map_dfr(axis_spec, function(s) {
    s$data %>%
      transmute(
        patientid,
        sample_date   = as.Date(samplingdate),
        # same grade parse as make_opera_outcome()
        grade_numeric = as.integer(sub("^(\\d+).*", "\\1", as.character(.data[[s$grade_col]])))
      ) %>%
      filter(grade_numeric >= s$threshold)
  }) %>%
    distinct(patientid, sample_date)
}


make_opera_outcome_cooccurrence <- function(components,
                                            first_line,
                                            patient_data,
                                            outcome_name,
                                            k            = 2L,
                                            window_days  = 7L,
                                            horizon_days = NULL,
                                            study_cutoff = STUDY_CUTOFF) {
  
  # 1. abnormal (patientid, sample_date, axis) across all axes
  abn <- imap_dfr(components, function(axis_spec, axis_name) {
    .axis_abnormal_days(axis_spec) %>% mutate(axis = axis_name)
  })
  
  # 2. one row per first_line patient with index + censor
  cohort <- first_line %>%
    select(patientid, index_date) %>%
    inner_join(patient_data %>% select(patientid, date_death_fu), by = "patientid") %>%
    mutate(censor_date = pmin(date_death_fu, study_cutoff, na.rm = TRUE))
  
  # 3. restrict abnormal rows to the observable (and, if set, peri-index) window
  abn <- abn %>%
    inner_join(cohort, by = "patientid") %>%
    filter(sample_date > index_date, sample_date <= censor_date)
  if (!is.null(horizon_days)) {
    abn <- abn %>% filter(sample_date <= index_date + horizon_days)
  }
  
  # 4. earliest co-occurrence per patient
  events <- abn %>%
    group_by(patientid) %>%
    summarise(outcome_date = .first_cooccurrence(sample_date, axis, k, window_days),
              .groups = "drop") %>%
    filter(!is.na(outcome_date))
  
  # 5. assemble standard schema (NA outcome_date for non-events)
  result <- cohort %>%
    left_join(events, by = "patientid") %>%
    rename(subject_id = patientid) %>%
    assign_splits(index_date_col = index_date) %>%
    select(subject_id, split, outcome_date, index_date, censor_date)
  
  # 6. same sanity checks as make_opera_outcome()
  stopifnot(
    "outcome_date must not exceed censor_date" =
      all(is.na(result$outcome_date) | result$outcome_date <= result$censor_date),
    "outcome_date must be strictly after index_date" =
      all(is.na(result$outcome_date) | result$outcome_date > result$index_date),
    "censor_date must not be NA" = !any(is.na(result$censor_date)),
    "index_date must not be NA"  = !any(is.na(result$index_date))
  )
  
  out_path <- file.path(OUTPUT_DIR, paste0(outcome_name, ".parquet"))
  write_parquet(result, out_path)
  
  n_total <- nrow(result); n_events <- sum(!is.na(result$outcome_date))
  n_splits <- table(result$split)
  message(sprintf(
    "%-40s  n=%d  events=%d (%.1f%%)  train=%d  tuning=%d  held_out=%d  [k=%d/%d, w=%dd, %s]",
    outcome_name, n_total, n_events, 100 * n_events / n_total,
    n_splits["train"] %||% 0L, n_splits["tuning"] %||% 0L, n_splits["held_out"] %||% 0L,
    k, length(components), window_days,
    if (is.null(horizon_days)) "any-time" else paste0("<=idx+", horizon_days, "d")))
  invisible(result)
}


# ============================================================
# The three co-occurrence outcomes
# ============================================================
message("Building co-occurrence outcomes:")

# --- Tumor lysis syndrome -----------------------------------------------------
# Cairo-Bishop laboratory TLS: >=2 of {urate, K, phosphate, corrected Ca}
# abnormal in the peri-induction window. CTCAE-grade thresholds map to the
# Cairo-Bishop cutoffs as follows:
#   urate      threshold 1  (>ULN; Cairo-Bishop >=476 umol/L is slightly stricter)
#   K          threshold 3  (CTCAE G3 = >6.0-7.0; Cairo-Bishop >=6.0)
#   phosphate  threshold 1  (>ULN ~1.45 mmol/L = Cairo-Bishop cutoff)
#   Ca (low)   threshold 3  (CTCAE G3 <1.75-1.5; Cairo-Bishop <=1.75)
# Note the urate leniency (G1 >ULN vs Cairo-Bishop 476 umol/L); tighten by
# swapping to an absolute-value axis if you want strict Cairo-Bishop.
make_opera_outcome_cooccurrence(
  components = list(
    hyperuricemia = list(data = HYPERURICEMIA,     grade_col = "ae_hyperuricemia",       threshold = 1),
    hyperkalemia  = list(data = HYPERKALEMIA,      grade_col = "ae_hyperkalemia",        threshold = 3),
    hyperphosphat = list(data = HYPERPHOSPHATEMIA, grade_col = "ae_hyperphosphatemia",   threshold = 1),
    hypocalcemia  = list(data = HYPOCALC,          grade_col = "ae_hypocalcemia",        threshold = 3)
  ),
  first_line = first_line, patient_data = patient,
  outcome_name = "tumor_lysis_syndrome",
  k = 2L, window_days = 7L, horizon_days = 7L
)

# --- Hy's-law-style DILI ------------------------------------------------------
# (ALT or AST >3x ULN) AND (bilirubin >~2x ULN), concurrent within 14 days,
# any time on treatment. The transaminase axis OR's ALT and AST so they count
# as ONE axis; k = 2 then forces transaminase AND bilirubin (not ALT+AST alone).
#   ALT/AST threshold 2  (CTCAE G2 = >3-5x ULN; matches Hy's 3x)
#   bili    threshold 2  (CTCAE G2 ~ >1.5-3x ULN; approximates Hy's 2x)
# LAB APPROXIMATION: true Hy's law also needs ALP not markedly raised and other
# causes excluded, which labs alone cannot establish. Label accordingly.
make_opera_outcome_cooccurrence(
  components = list(
    transaminase = list(
      list(data = ALAT, grade_col = "ae_ALAT_increased", threshold = 2),
      list(data = ASAT, grade_col = "ae_ASAT_increased", threshold = 2)
    ),
    bilirubin = list(data = BILI, grade_col = "ae_bilirubin_increased", threshold = 2)
  ),
  first_line = first_line, patient_data = patient,
  outcome_name = "hys_law_dili",
  k = 2L, window_days = 14L, horizon_days = NULL
)

# --- DIC-like consumptive coagulopathy ----------------------------------------
# >=2 of {fibrinogen low G3+, INR high G2+, platelets low G3+} within 7 days,
# any time. PROXY ONLY: no D-dimer or schistocytes in the lab set, so this is
# a coagulation-derangement cluster, not ISTH-scored DIC. Worth it mainly if
# the APL / AML share of your cohorts justifies the caveat.
make_opera_outcome_cooccurrence(
  components = list(
    hypofibrinogen   = list(data = FIBRINOGEN,    grade_col = "ae_hypofibrinogenemia",  threshold = 3),
    inr_high         = list(data = INR,           grade_col = "ae_INR_increased",       threshold = 2),
    thrombocytopenia = list(data = PLATELETS_DEC, grade_col = "ae_platelets_decreased", threshold = 3)
  ),
  first_line = first_line, patient_data = patient,
  outcome_name = "dic_like_coagulopathy",
  k = 2L, window_days = 7L, horizon_days = NULL
)

message("Done. Co-occurrence outcomes written to ", OUTPUT_DIR,
        " (run the coverage gate next so these are gated with the rest).")
