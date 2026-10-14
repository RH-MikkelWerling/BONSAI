# Early TLS candidate audit v1 (INTERNAL)

This is a separate exploratory audit, not an addition to the current primary
panel or the historical outcome parquets. It produces two main nested labels,
`tls_laboratory_early_v1` and `tls_laboratory_with_kidney_early_v1`, plus
two complete-panel sensitivity labels.
The historical `tumor_lysis_syndrome` proxy remains retired from the primary
panel. The main candidate labels now use the same analyte-agnostic lab-coverage
gate as the R outcomes, read from a same-run `__audit.parquet`. Two additional
`complete_panel` labels preserve the former strict monitoring rule as a
sensitivity analysis. Sync the updated exporter and regression test.
The **prediction horizon is 30 days**. The qualifying biochemical event is
restricted to post-index days 1–7; these are different concepts.

The laboratory label requires at least two distinct abnormal metabolic axes
on the same calendar day. The renal label additionally requires creatinine
above that laboratory's reported upper limit on the same day. Exact absolute
cutoffs are urate ≥476 µmol/L, potassium ≥6.0 mmol/L, phosphate ≥1.45 mmol/L
and calcium ≤1.75 mmol/L. This is an absolute-threshold, date-only
approximation to [CTCAE v6 TLS](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf),
**not** full CTCAE grading: it omits 25%-from-baseline alternatives,
pre-treatment/spontaneous TLS, seizures and arrhythmias. Elevated pre-existing
creatinine is not adjudicated, and the calcium axis has not yet been verified
as albumin-corrected calcium. We use “with kidney involvement,” not
“clinical TLS,” for that reason.

Main-label eligibility uses shared lab-system coverage, baseline and entry
rules, not a TLS-specific panel. A documented event still requires two
actually observed abnormal axes on the same calendar day within days 1–7;
the kidney label additionally requires elevated creatinine on that day.
An event-free label therefore means no qualifying combination was recorded,
not that every axis was measured and normal. The complete-panel sensitivity
requires all four axes on one day, plus creatinine for its kidney version.

## Sync

Sync these new files from the local repository:

- `opera/R/export_candidate_tls_labs_v1.R`
- `opera/run/export_candidate_tls_v1.py`
- `tests/test_candidate_tls_v1.py` (for the server test, optional)

The Python module also imports already-existing
`opera/run/export_candidate_outcomes_v1.py` and
`opera/run/decide_outcome_panel.py`; make sure those are synced too. All
patient-level outputs and the monitoring CSV remain internal.

## Run on the secure server

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
mkdir -p data/results

TLS_SNAP="$PWD/data/results/candidate_tls_labs_v1_$(date -u +%Y%m%dT%H%M%SZ)"
Rscript opera/R/export_candidate_tls_labs_v1.R "$TLS_SNAP"
cat "$TLS_SNAP/npu_mapping_v1_INTERNAL.csv"
cat "$TLS_SNAP/unit_inventory_v1_INTERNAL.csv"

TLS_OUT="$PWD/data/results/candidate_tls_v1_$(date -u +%Y%m%dT%H%M%SZ)"
: "${BONSAI_FINAL_SURVIVAL:?Set this to overall_survival.parquet rebuilt from the frozen V4 first-line index}"
python -m opera.run.export_candidate_tls_v1 \
  --labs "$TLS_SNAP/candidate_tls_labs_v1_INTERNAL.parquet" \
  --survival "$BONSAI_FINAL_SURVIVAL" \
  --lab-audit "$(dirname "$BONSAI_FINAL_SURVIVAL")/anemia_g2plus__audit.parquet" \
  --output-dir "$TLS_OUT"

cat "$TLS_OUT/tls_monitoring_and_events_INTERNAL.csv"
cat "$TLS_OUT/horizon_support_INTERNAL.csv"
```

Stop if the R script cannot resolve an NPU mapping or if the Python unit
contract rejects the cleaned units. Do not silently substitute an NPU code
or unit conversion. The R script expects `NPU.FOS` for inorganic phosphate
and `NPU.CA` for calcium. `BASP` denotes alkaline phosphatase, **not** the
phosphate axis of TLS. Verify both mappings against the installed DALY-CARE
loader. If the loader lacks `NPU.FOS`, the extractor uses `NPU03096`, the same
fallback in the historical outcome R script. [Sundhed.dk](https://www.sundhed.dk/sundhedsfaglig/laegehaandbogen/undersoegelser-og-proever/klinisk-biokemi/blodproever/fosfat/)
identifies that code as plasma inorganic phosphate in mmol/L. Still inspect
the emitted NPU and unit inventories before proceeding.

The extractor accepts the cleaned code column `NPU` (or retained raw
`analysiscode`) and records whether units came from `unit2` or `unit`.
It prints the number of cleaned measurements that did not parse numerically;
review that count before interpreting the TLS coverage audit. If `unit2` is
absent, the Python unit contract is especially important because raw `unit`
may not reflect any normalization performed on `value2`.

The Python exporter now explicitly converts urate values reported in mmol/L
to µmol/L before applying the 476 µmol/L absolute threshold. This is a
measured-value conversion only; the urate reference upper limit is not used
for the TLS label. Other unrecognized units still stop the run. In particular,
do not apply an automatic conversion to the creatinine reference limit until
the raw-versus-cleaned unit relationship is verified.

The most useful files to share back are the two aggregate CSVs, the NPU/unit
inventories and `manifest_INTERNAL.json`. They are marked INTERNAL and need
disclosure review before transfer. **Do not upload the lab snapshot or outcome
parquets.** If coverage leaves too few cases, report that result; the next
step would be a clinician-reviewed ascertainment sensitivity, not immediate
training.

The runbook's `overall_survival.parquet` input must be rebuilt from the frozen
37,540-person V4 first-line index before this candidate is considered for the
final release. An older survival file would silently apply the old index and
split assignments. The candidate is not primary-panel approved by a successful
export alone; review shared-gate and complete-panel support, unit mapping,
event counts and the renal-definition limitations first.

The shared audit must come from the *same outcome run* as overall survival.
The exporter verifies identical patient IDs, index dates and temporal splits.
The event definition remains days 1–7 even though the existing support export
reports a 30-day horizon. Since eligibility uses post-index lab activity,
these are retrospectively ascertainable labels, not a prospectively known
baseline subgroup. This needs explicit modelling and methods review.
