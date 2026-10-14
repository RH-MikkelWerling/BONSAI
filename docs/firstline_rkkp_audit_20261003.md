# First-line RKKP attrition audit (INTERNAL)

This is a diagnostic, not permission to add registry-dated patients to the
cohort. Sync `opera/R/run_firstline_rkkp_audit.R` and the current first-line R
script. Reuse the *same* frozen loader cache and explicit CLL/MM/treatment
Parquet environment variables as the current 37,547-person candidate run;
otherwise the comparison does not concern that candidate. The earlier 37,558-
person run had 436 more selected CLL/MM seed records and 11 more final patients;
record this version difference in the release provenance, but it need not block
the diagnostic RKKP comparison. The three RKKP registry tables are
currently loaded from SQL and are not frozen by this wrapper. Record their
extraction time and source definitions, or freeze them before final adjudication.

Run on the secure server, from the BONSAI repository, after verifying the
candidate input environment:

```bash
umask 077
mkdir -p "$PWD/data/results/firstline_rkkp_audit_runs"
export OPERA_EXPECTED_FIRSTLINE_N=37547
Rscript opera/R/run_firstline_rkkp_audit.R \
  "$PWD/opera/R/identification_first_line_treatment_v2.R" \
  "$PWD/data/results/firstline_rkkp_audit_runs" \
  /ngc/people/mikwer_r
```

The new private run directory contains nine aggregate CSVs and a README.
`rkkp_attrition_INTERNAL.csv` has mutually exclusive first-failing reasons;
`rkkp_stage_counts_INTERNAL.csv` has marginal support flags that must not be
subtracted as a single funnel; `rkkp_date_agreement_INTERNAL.csv` compares
absolute registry and exported index dates. The new
`rkkp_signed_date_agreement_INTERNAL.csv` bins the signed difference by registry
and candidate source; negative means the candidate date precedes the registry
date. `ANY_RKKP` deduplicates patients across
LYFO, CLL and DaMyDa, so it is not the sum of those three strata.
`rkkp_date_only_upper_bound_INTERNAL.csv` asks how many omitted patients would
receive a fine disease label if the earliest RKKP date were used as a
hypothetical index; this is *not* evidence of an administered treatment.
`rkkp_treatment_field_inventory_INTERNAL.csv` lists treatment-related registry
fields and how many omitted patients have a value, but exports no field values.
`rkkp_lyfo_treatment_values_INTERNAL.csv` tabulates selected structured LYFO
field values among omitted patients by first failing gate. It contains exact
small cells and must remain on the secure server. A populated chemotherapy
field is not necessarily an affirmative chemotherapy flag; inspect its values.
`rkkp_near_date_treatment_gate_INTERNAL.csv` classifies raw treatment rows
within 30 days of the RKKP date as absent, plan-only/missing-source, without
usable code, excluded by indication, or surviving row filters despite absence
from the final export. `rkkp_near_date_indication_conflicts_INTERNAL.csv`
counts rows for which an RKKP-source override retains a treatment even though
an exclusion signal is present. These are diagnostic, not validated matches
between a registry regimen and an administered treatment row.

Review internally: (1) registry-dated patients absent from the candidate,
(2) patients with no cleaned RKKP treatment row versus no eligible episode,
(3) indexed-but-unclassified patients, and (4) date discrepancies. Do not
share small exact cells or patient-level records outside the secure server.
Any proposed rescue must be checked for disease identity, actual administered
treatment, chronology, compatibility with the outcome observation window, and
whether the registry-recorded treatment would be excluded by the pipeline's
plan/source, missing-code, indication, or episode rules. Do not infer regimen
from a merely nearby but unlinked row.
Compare both patient count and date/cohort changes before approving a new
first-line policy.

For the methods, describe the common episode algorithm and its disease-
specific CLL/MM seed override separately. Those seed dates are algorithmic
line-one estimates, not a uniformly applied independent clinician adjudication.
Report their coverage, discordance, and the prespecified priority/matching
rule in the final release manifest.
