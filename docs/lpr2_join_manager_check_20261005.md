# LPR2 procedure-join check for the refreshed extract (internal)

The outcome builder previously converted `SDS_t_sksube$v_recnum` and
`SDS_t_sksopr$v_recnum` to R's 32-bit integer before joining to
`SDS_t_adm$k_recnum`. A valid key such as `2147483648` becomes `NA_integer_`
under `as.integer()`. Its procedure can therefore disappear from the
patient-level outcome despite being present in both source tables. The
production repair compares lossless text representations and prevents missing
keys from matching. Separately, the outcome date is the source procedure
start date (`d_odto`), not the admission start date (`d_inddto`). These field
meanings are documented in the official
[LPR2 variable catalogue](https://www.dst.dk/ext/7143686406/0/forskning/Dokumentation-af-LPR2--pdf):
for `t_sksube`, `d_odto` is *dato for procedurens påbegyndelse*; for
`t_sksopr`, it is *datoen for operationens påbegyndelse*. `d_inddto` is the
contact/admission start date. Neither is an indberetningsdato. A recorded
procedure code/date is still a registry proxy and may need code-family review
before being described as delivered treatment.

Run this **only inside the secure data environment** on the new extract. It
prints one real key and patient identifier locally; do not copy that row into
email, an issue tracker, or this repository.

```r
library(dplyr)
key_text <- function(x) {
  y <- if (is.numeric(x)) trimws(format(x, scientific = FALSE, trim = TRUE))
       else trimws(as.character(x))
  y[is.na(x) | !nzchar(y)] <- NA_character_
  y
}
admissions <- SDS_t_adm %>%
  transmute(key = key_text(k_recnum), patientid) %>%
  filter(!is.na(key))
procedures <- SDS_t_sksube %>%
  transmute(key = key_text(v_recnum), code = c_opr,
            procedure_date = as.Date(d_odto),
            old_integer_key = suppressWarnings(as.integer(v_recnum))) %>%
  filter(!is.na(key))
affected <- procedures %>%
  filter(is.na(old_integer_key)) %>%
  inner_join(admissions, by = "key", relationship = "many-to-many")
print(affected %>% select(key, code, procedure_date, patientid) %>% slice_head(n = 1))
cat("Rows joining as text but lost by 32-bit cast:", nrow(affected), "\n")
cat("Distinct affected keys:", n_distinct(affected$key), "\n")
cat("Keys mapping to multiple patients:",
    affected %>% distinct(key, patientid) %>% count(key) %>%
      summarise(n = sum(n > 1)) %>% pull(n), "\n")
```

For the printed example, verify directly in the refreshed source that the
`v_recnum` procedure row and the `k_recnum` admission row have the same exact
key, the admission resolves to the expected patient, and the exported event
uses `d_odto`. A procedure key having several procedure rows is normal; a
key mapping to multiple patients is not and needs investigation. If the
refreshed extract returns zero affected rows, record the source version and
key types: the bug may no longer be triggered there, but code should still
avoid lossy 32-bit casts.

Historical aggregate audit: the LPR2 other-procedure source had 1,576,198
distinct overflowed keys that matched an admission as text. The combined
key-and-date repair added 22,208 labels and moved 5,484 dates earlier across
seven overlapping outcome files relative to the old export; these are not
distinct patients. The analogous LPR3 key audit found no lost matching keys.

## Separate first-line treatment-table check

The upstream treatment-table snippet currently visible converts trimmed
`v_recnum` with `as.numeric()`, not `as.integer()`. In R that produces a
double-precision number, which can exactly represent integer identifiers up
to 2^53 (9,007,199,254,740,992), so the observed 32-bit overflow does not by
itself imply missing treatment rows. However, the whole join must be checked:
look for any later `as.integer(v_recnum_int)`, compare the admission-side key
type, and count nonnumeric keys, round-trip failures, and keys above 2^53 in
the *actual refreshed treatment extract*. An exact string join is preferable
for opaque record IDs, provided both sides use the same lossless canonical
representation. Do not infer that the selected first-line cohort is unchanged
until the upstream join and frozen treatment table have been audited.
