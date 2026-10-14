# Isolated candidate outcome export v1 (INTERNAL)

This is a numerical **what-if** audit, not a new primary panel or a claim of
fully adjudicated CTCAE v6 adverse events. It writes 12 candidate lab-outcome
parquets (neutrophils, ALT, AST, bilirubin, creatinine at G2+/G3+, plus
eosinophilia and one separate creatinine-rise outcome) to a **new** directory. AST is included because the candidate
`hepatotoxicity` composite depends on it even though AST is not a primary
candidate. The old parquets are read-only. The output is a partial panel;
never point a training run at it directly.

## Where the selection inventory is

`OperaContrastiveModule._log_competing_risk_validation()` writes
`endpoint_selection_inventory.json` to `trainer.default_root_dir`.
`contrastive_multicohort.py` sets that to `model_save_dir`, which is the run
directory. For the reviewed `weightinggroups` run the exact path is:

```text
/ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia/data/models/endpoint_weighting_pilot_v1/weightinggroups__direct_cr_bce_lowrank_endpoint_groups_v1__seed42/endpoint_selection_inventory.json
```

The validation bundler omits this file; it includes the training interval
support, not the tuning selection inventory. If absent at this exact path,
inspect the run's `hydra/config.yaml` and the server checkout's
`OperaContrastiveModule.py` for the actual `default_root_dir` and whether the
inventory writer was present in that historical run. Do not synthesize it from
training support; the uploaded horizon-support table gives tuning event counts
but not the exact validation cases/controls after competing deaths/censoring.

## Inputs and safeguards

`opera/R/export_candidate_lab_snapshot_v1.R` loads the same raw lab table and
`clean_lab_values()` helper as the historical script, but exports **only six
analytes** into an internal normalized snapshot. It also writes NPU and
cleaned-unit inventories. Inspect those inventories and compare the server
loader/package code hash with the public revision before interpreting labels.
The Python exporter fails if units differ from
`opera/configs/candidate_lab_units_v1.yaml`, input parquets/audits disagree,
dates are invalid, or the destination already exists.

The baseline policy is the median of draws on the latest strictly pre-index day within
365 days. If an analyte-specific baseline or required reference range is
missing, that patient is **excluded**, never assigned an event-free label.
The creatinine G2+/G3+ candidates implement the numerical branches of
[CTCAE v6 Creatinine increased](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf):
when pretreatment creatinine is below its LLN, G2+ is >1.5 times baseline and
G3+ is >3 times baseline; otherwise the cutoffs are >1.5 and >3 times the
post-treatment measurement's ULN. The distinct
`creatinine_rise_gt1p5x_preindex_baseline` candidate tests >1.5 times the
patient's own pretreatment creatinine **regardless** of its reference-range
position. That is a clinical change-from-baseline target, not CTCAE v6 grading
or an acute-kidney-injury diagnosis. It needs a positive analyte-specific
baseline but not a valid reference interval. A missing baseline excludes the
patient; neither definition treats missing values as normal.
Candidate censoring is recomputed from the all-lab last-draw date and the
administrative censor in `overall_survival.parquet`, including when regrading
removes an old event. This preserves the old all-lab monitoring assumption
for a clean regrading comparison; `baseline_availability_INTERNAL.csv` also
counts patients without a post-index or first-30-day *analyte-specific* draw.
That monitoring sensitivity must be reviewed before finalizing a label.

## Server commands

Run on the secure server, from the synced repository root, in the usual R and
Python environments. The source input directory should be the one named in
the uploaded `summary_INTERNAL.json`. Outputs contain patient IDs and are
**not disclosure-cleared**.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh

RUN_DIR="$PWD/data/models/endpoint_weighting_pilot_v1/weightinggroups__direct_cr_bce_lowrank_endpoint_groups_v1__seed42"
ls -l "$RUN_DIR/endpoint_selection_inventory.json"

SNAP_DIR="$PWD/data/results/candidate_lab_snapshot_v1_$(date -u +%Y%m%dT%H%M%SZ)"
Rscript opera/R/export_candidate_lab_snapshot_v1.R "$SNAP_DIR"
cat "$SNAP_DIR/unit_inventory_v1_INTERNAL.csv"

CAND_DIR="$PWD/data/results/candidate_outcomes_v1_$(date -u +%Y%m%dT%H%M%SZ)"
python -m opera.run.export_candidate_outcomes_v1 \
  --labs "$SNAP_DIR/candidate_labs_v1_INTERNAL.parquet" \
  --old-outcomes-dir "$PWD/data/processed_data/daly_care/outcomes" \
  --output-dir "$CAND_DIR" \
  --unit-contract opera/configs/candidate_lab_units_v1.yaml
```

Do not proceed past the unit inventory if it shows mixed or unexpected units;
resolve those mappings first. `Rscript` requires the package loader's normal
environment and read access to its underlying raw table. The Python exporter
requires each existing outcome's `__audit.parquet` and `overall_survival.parquet`.
It fails closed if they are absent.

Review `manifest_INTERNAL.json`, `comparison_counts_INTERNAL.csv`,
`event_overlap_INTERNAL.csv`, `baseline_availability_INTERNAL.csv` and
`horizon_support_INTERNAL.csv` first. Also review
`creatinine_definition_overlap_INTERNAL.csv`: in that file `old` means the
candidate CTCAE G2+ definition and `candidate` means the relative-rise
definition, not an earlier release. Both candidates are exploratory until the
doctor confirms the threshold/window and the event support and eligibility
tradeoff have been inspected.
The comparison reports old/new
eligible patients, gained/lost events and shifted event dates by split. It
also reports total old/new events. The overlap report separates changed event
membership among shared-eligible patients from old events lost because a
baseline was unavailable, at 30 days, 365 days and all available follow-up;
it also reports event-set Jaccard overlap. Horizon support covers 30, 60, 90, 180,
365 and 730 days, including full-follow-up-or-event denominators. It
does **not** release patient-level transition rows outside the secure server.
For a compact handoff without exact small cells or held-out counts, run
`python -m opera.run.summarize_candidate_lab_overlap --lab-candidates-dir
"$CAND_DIR" --output-dir "$PWD/data/results/lab_overlap_bands_$(date -u
+%Y%m%dT%H%M%SZ)"` in the Python environment. Its
`overlap_bands_INTERNAL.md` compares old versus candidate definitions at 30
and 365 days, plus candidate creatinine G2+ versus relative creatinine rise.
Bands are descriptive, not a selection threshold; denominators below 20 are
withheld. Event overlap refers to patients eligible for both definitions;
eligibility loss and gain are reported separately. The output remains INTERNAL
and is not automatically disclosure-cleared.
To adopt any definition, additionally review patient-level transitions there,
check all numeric boundaries, assay units and reference ranges, rebuild
`hepatotoxicity` against the candidate ALT/AST/bilirubin component dates, and
rerun the structural and 30/60/365-day support audits on a coherent new release.

## Extra refinements worth considering, not automatic changes

- Audit whether all-lab monitoring overstates observation of a particular
  analyte. The availability CSV exposes this, but v1 does **not** silently
  change eligibility and regrading at once. A separate strict-monitoring
  sensitivity can require a first-30-day draw of that analyte.
- Check raw `<`/`>`-qualified values, duplicate same-day draws and missing
  reference limits. The snapshot's cleaned numeric `value2` strips comparison
  signs; such results should be quantified before final clinical sign-off.
- Consider repeat-confirmed abnormalities as a sensitivity for noisy lab
  endpoints, not a default: a second draw can miss clinically important acute
  events. Do not add new primary endpoints until the current definitions are
  signed off. Febrile neutropenia or microbiology-confirmed infection could be
  valuable later only if temperature/culture timing and capture are reliable.
