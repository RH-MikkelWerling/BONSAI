# ==============================================================================
# lpr3_coverage_gate.R   (per-outcome calendar coverage gate)
#
# Post-hoc coverage gate for the SMR-based IV-medication infection proxy only.
# Hospitalisation and diagnosis-coded infections retain their full observation
# periods: their observed time trends do not justify a speculative LPR3 gate.
#
# Each source has its OWN coverage start date, so the gate keys on a per-outcome
# table rather than one global date. Gating on index_date guarantees the entire
# forward, label-defining follow-up window sits in the consistent era; pre-break
# patients are masked for THESE outcomes only and still contribute everywhere
# else and to pretraining.
#
# For each gated outcome:
#   - keep patients with index_date >= its coverage_start  -> overwrite <name>.parquet
#   - move earlier patients                                -> <name>__coverage_audit.parquet
#   - back up the ungated input (when gating changes anything) -> <name>__ungated.parquet
# Output schema is UNCHANGED: subject_id, split, outcome_date, index_date, censor_date.
# Idempotent: re-running on an already-gated parquet is a no-op.
#
# Run only through run_coverage_gate_versioned.R, which copies the sealed panel
# to a new directory before this script changes a parquet.
# ==============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(lubridate)
  library(tibble)
})

if (!identical(Sys.getenv("OPERA_VERSIONED_COVERAGE_GATE"), "1") ||
    !exists("OUTPUT_DIR") || !dir.exists(OUTPUT_DIR))
  stop("Run via run_coverage_gate_versioned.R with a new output directory")

SCHEMA <- c("subject_id", "split", "outcome_date", "index_date", "censor_date")

# ── Coverage-break dates ──────────────────────────────────────────────────────
# The SMR reporting regulation took effect on 2018-05-15, when regional
# submissions began (Danish Health Data Authority). This is a national start,
# not a claim that every medicine/department/region was complete that day.
# OPERA_SMR_COVERAGE_START permits an explicit sensitivity-analysis override.
smr_start_text <- Sys.getenv("OPERA_SMR_COVERAGE_START", "2018-05-15")
if (!grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", smr_start_text))
  stop("OPERA_SMR_COVERAGE_START must be YYYY-MM-DD")
SMR_COVERAGE_START <- tryCatch(as.Date(smr_start_text), error = function(e) NA_Date_)
if (is.na(SMR_COVERAGE_START)) stop("Invalid OPERA_SMR_COVERAGE_START")

# ── Per-outcome coverage table ────────────────────────────────────────────────
# outcome | its coverage_start | source tier (flows into audit + tag CSV).
# The repaired LPR2 procedure outcomes and reviewed LPR contact/diagnosis
# outcomes are intentionally absent: do not mask their pre-2019 patients.
COVERAGE_GATES <- tribble(
  ~outcome,                       ~coverage_start,       ~source_tier,
  # SMR-sourced IV antibiotics (infection proxy).
  "infections_IV_based",    SMR_COVERAGE_START,    "smr_gated"
)

# ── Gate one outcome (source-agnostic) ────────────────────────────────────────
apply_coverage_gate <- function(outcome_name, coverage_start,
                                source_tier = "gated", output_dir = OUTPUT_DIR) {
  
  if (is.na(coverage_start)) {
    message(sprintf("  [TODO] %-32s no coverage_start set (tier=%s); skipped",
                    outcome_name, source_tier))
    return(invisible(NULL))
  }
  
  path <- file.path(output_dir, paste0(outcome_name, ".parquet"))
  if (!file.exists(path)) {
    stop("Missing outcome to gate: ", path)
  }
  
  full  <- read_parquet(path) %>% mutate(index_date = as.Date(index_date))
  n_pre <- sum(full$index_date < coverage_start, na.rm = TRUE)
  # A rerun on an already-gated file must preserve the original exclusion audit.
  if (n_pre == 0L) {
    message(sprintf("  %-32s already eligible; audit left unchanged", outcome_name))
    return(invisible(full))
  }
  
  # Back up the ungated input whenever gating actually removes patients.
  backup <- file.path(output_dir, paste0(outcome_name, "__ungated.parquet"))
  if (n_pre > 0L) write_parquet(full %>% select(all_of(SCHEMA)), backup)
  
  eligible <- full %>% filter(index_date >= coverage_start) %>% select(all_of(SCHEMA))
  masked   <- full %>% filter(index_date <  coverage_start) %>%
    transmute(subject_id, split, outcome_date, index_date, censor_date,
              coverage_eligible = FALSE,
              reason = paste0("masked_pre_", source_tier, "_coverage"))
  
  stopifnot(
    "schema unchanged" = identical(names(eligible), SCHEMA),
    "censor_date >= index_date" = all(eligible$censor_date >= eligible$index_date),
    "outcome_date > index_date" =
      all(is.na(eligible$outcome_date) | eligible$outcome_date > eligible$index_date),
    "no masked subject remains" =
      length(intersect(eligible$subject_id, masked$subject_id)) == 0L
  )
  
  write_parquet(eligible, path)   # overwrite canonical with eligible-only
  write_parquet(masked, file.path(output_dir, paste0(outcome_name, "__coverage_audit.parquet")))
  
  n0 <- nrow(full); n1 <- nrow(eligible); ev <- sum(!is.na(eligible$outcome_date))
  message(sprintf("  %-32s [%s >= %s] kept=%6d/%-6d  masked=%5d (%.1f%%)  events=%d",
                  outcome_name, source_tier, format(coverage_start),
                  n1, n0, n0 - n1, if (n0 > 0) 100 * (n0 - n1) / n0 else 0, ev))
  invisible(eligible)
}

# Back-compat alias (older callers / muscle memory).
apply_lpr3_gate <- apply_coverage_gate

# ── Apply across the table ────────────────────────────────────────────────────
message(sprintf("Coverage gate over %d outcomes:", nrow(COVERAGE_GATES)))
for (i in seq_len(nrow(COVERAGE_GATES))) {
  r <- COVERAGE_GATES[i, ]
  apply_coverage_gate(r$outcome, r$coverage_start, r$source_tier)
}

# ── Source-tier tags for the manifest / send-off ──────────────────────────────
source_tags <- COVERAGE_GATES %>%
  transmute(outcome, source_tier, coverage_start = as.character(coverage_start))
write.csv(source_tags, file.path(OUTPUT_DIR, "coverage_gated_outcomes.csv"),
          row.names = FALSE)
message("Wrote coverage_gated_outcomes.csv (source-tier tags).")
