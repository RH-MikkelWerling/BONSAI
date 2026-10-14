# Outcome release adjudication (working decision, 4 October 2026)

Clinical terminology reference: [NCI CTCAE v6.0](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf).

Separate two decisions for every exported outcome: **include as an operational
prediction target?** and **what clinical terminology does its implementation
actually support?** Event support never upgrades a proxy into a CTCAE grade.

## Early laboratory TLS

- Retain the old `tumor_lysis_syndrome` parquet as historical/retired; do not
  relabel it or silently substitute it in a trained model.
- Keep `tls_laboratory_early_v1` and
  `tls_laboratory_with_kidney_early_v1` as two nested, exploratory operational
  candidates. Their shared lab-source gate matches the existing lab-outcome
  eligibility policy. Events require two observed abnormal axes on the same
  post-index day 1–7; kidney involvement additionally requires measured
  creatinine above its reported ULN on that day. Negatives mean *no qualifying
  combination documented under the shared observation process*, not a
  complete normal metabolic panel. The strict complete-panel variants are
  ascertainment sensitivities, not four independent headline outcomes.
- Current split counts are 243/45/114 events for the lab signal and
  207/41/100 for the nested kidney signal (train/tuning/held-out). Both pass
  the existing 100/20 train/tuning event floor numerically, but these are
  counts within days 1–7, not a 30-day event process. Document the event
  window separately from any 30-day support reporting.
- These are **CTCAE-v6-related laboratory approximations**, not direct v6 TLS
  grades. The implementation omits the 25%-from-baseline route, pre-index
  spontaneous TLS, clinical arrhythmia/seizure routes, creatinine severity
  ratios and adjudication of pre-existing kidney elevation; same calendar day
  is a local operational choice. Do not call the kidney label "clinical TLS"
  or "CTCAE grade 1+". Clinician/source sign-off is required before either
  becomes a primary headline endpoint.
- The two main labels are strongly nested. They may both be retained for a
  prespecified analysis, but should be one endpoint family for macro-averaging
  and multiplicity accounting; do not present them as independent discoveries.

## Terminology classes for the final endpoint dictionary

Assign every outcome **exactly one implementation class**, plus separate
status and evidence fields:

1. `ctcae_v6_direct_verified`: exact v6 term and grade threshold/qualifiers
   reproduced by the installed server grader, with NPU, unit, reference and
   boundary tests. A matching filename alone never qualifies.
2. `ctcae_v6_related_partial`: term/threshold partly mapped but clinical
   qualifiers, baseline branches or severity components missing. State the
   measurable rule, not a v6 grade claim. The new TLS candidates belong here.
3. `ctcae_legacy_or_local_threshold`: a v5-derived, local numeric or
   differently graded lab rule. Preserve the implemented threshold and version
   explicitly; never relabel an existing parquet as v6 without regrading.
4. `registry_diagnosis_or_procedure`: coded event with registry capture, not
   CTCAE severity or treatment attribution.
5. `treatment_care_use_or_survival`: treatment transition, medication proxy,
   transfusion, hospitalization, death, etc.; outside CTCAE grading.
6. `derived_composite`: explicitly list constituent outcomes, temporal logic,
   eligibility/censor rule and whether any component is a proxy.

For each outcome record: source table; source code set/function and revision;
operational event rule; index-relative event window and prediction horizon;
eligibility and censoring; baseline/new-onset convention; implementation
class; CTCAE term/version/grade claim if any; clinical limitations; train and
tuning support; disposition (`primary`, `secondary`, `exploratory`, `hold`, or
`retired`); and reviewer/date. The support floor is a necessary operational
screen, not clinical validation.

## Short closeout sequence

1. Reconcile the outcome inventory: 81 canonical files passed structural
   checks, while six downstream outcomes were not yet generated. Build and
   audit the second-line-dependent and other composites only after their
   inputs/cutoffs are settled. Do not confuse an inventory failure with 81
   structurally bad parquets.
2. Set the 4 versus 19 March administrative source-completeness policy and
   resolve the procedure `v_recnum` overflow/many-to-many warning before
   freezing affected endpoints. Review the R coverage gate's source dates.
3. Run the versioned outcome-panel decision on the complete, new directory.
   Apply the existing clinical holds/retirements before support thresholds;
   use train/tuning only for selection. Keep held-out counts descriptive.
4. Fill the endpoint dictionary from actual installed function bodies, code
   sets, NPU/unit/boundary checks and the run's manifests; sign off only the
   few unresolved high-priority definitions. Freeze versioned files and
   methods language before rebuilding model targets.

Read-only checks available before any parquet mutation:

```bash
OUTCOME_WORK="$PWD/data/results/outcomes_v4_audit_20261003T153741Z"
python -m opera.run.audit_outcome_cutoff_window \
  --outcomes-dir "$OUTCOME_WORK/opera_outcomes" \
  --output "$OUTCOME_WORK/cutoff_window_04_to_19_march_INTERNAL.csv"
Rscript opera/R/audit_procedure_join_keys.R
```

The cutoff report counts potentially affected event *rows across outcomes*,
not distinct patients and not proof of source validity. The procedure audit
reports whether integer overflow loses join keys and whether key-to-patient
multiplicity can misassign a procedure. Neither check changes the outcome
directory. Do not run the mutating LPR3/SMR coverage gate or composites until
their cutoff/source policies and required inputs are agreed.

This document does not change any existing outcome parquet or release roster.

## LPR2 procedure-join adjudication (4 October 2026)

The audit found 355,581 distinct `SDS_t_sksopr` keys and 1,576,198 distinct
`SDS_t_sksube` keys that overflow `as.integer(v_recnum)` but match an LPR2
admission as text. The corresponding LPR3 check found zero lost matching keys.
The targeted final-cohort audit found no eligible lost-key surgery events but
47,650 eligible patient-outcome-date rows from `SDS_t_sksube`. Against the
existing outcome exports, lost-key records reveal the following patients with
no current event / an event later than the lost-key record:

| Outcome | No current event | Current event later |
| --- | ---: | ---: |
| RBC transfusion | 3,722 | 568 |
| Platelet transfusion | 571 | 48 |
| Any transfusion | 3,761 | 595 |
| Antibiotics | 5,686 | 2,130 |
| Respiratory support | 1,828 | 67 |
| Dialysis | 791 | 93 |
| Emergency stay | 2,939 | 179 |

This is an outcome-label defect, not merely expected one-admission-to-many-
procedures multiplicity. The LPR2 joins in `calculate_adverse_events_update.R`
now use matched text keys without the 32-bit cast. Existing parquet exports
remain stale until a fresh, versioned outcome run and differential audit are
complete. Do not silently overwrite the old audit directory or freeze the
procedure-derived labels yet.

After generating a fresh **ungated** outcome directory with the repaired join,
compare it to the old ungated export before deciding whether the 2019 coverage
gate remains justified:

```bash
python -m opera.run.audit_lpr3_procedure_eras \
  --old-dir data/results/outcomes_v4_audit_20261003T153741Z/opera_outcomes \
  --new-dir /absolute/path/to/new_run/opera_outcomes \
  --output-dir /absolute/path/to/new_run/lpr3_era_audit
```

The audit refuses differing patient sets, index dates or censor dates, so the
comparison isolates the join repair. Review `event_rates_by_index_year.csv`,
`event_rates_by_index_era.csv`, `time_to_event_bins_by_index_era.csv`, and both
PDFs. The 30-/365-day crude fractions use patients with full horizon follow-up
or an earlier event; they are descriptive, not competing-risk incidence.
Calendar-year first-event counts are also exported, but changes in cohort size
make raw calendar counts unsuitable as the sole coverage criterion. Keep the
old gate until corrected pre/post-LPR3 ascertainment is judged by endpoint.

The first repaired-join comparison returned 19,298 additional event labels and
3,680 earlier event dates summed across the seven outcomes (correlated labels,
not distinct patients), with no lost labels and seven nonidentical file hashes.
The sharp pre-2019 procedure dip largely disappeared in the visual audit.
The original 365-day plot's 2025/2026 points are invalid for interpretation:
late cohorts have little/no full-horizon follow-up, so an events-or-full-
follow-up denominator can approach 100% events. The plotter now restricts to
complete calendar index years through 2024 for the March 2026 cutoff and
exports full-horizon-follow-up counts in the CSV. Do not use the late-year
spikes to set coverage policy.

Before removing the old LPR3 gate, inspect event-date semantics. The current
generator joins procedure records to admissions/contacts but dates `CODES`
with the *admission/contact start* rather than the source procedure date, even
though `d_odto`/procedure `dato_start` are loaded. This could shift events
earlier and omit a post-index procedure during a pre-index encounter. Run the
read-only `opera/R/audit_procedure_event_dates.R` on the repaired run's
first-line input, review missing procedure dates and post-index differences,
then decide whether the final generator should date events by the procedure.
The aggregate date audit returned zero missing procedure dates for the selected
code families and showed substantial post-index reclassification when the
procedure date is used; for example, LPR2 other-procedure records add 4,469
RBC-transfusion and 9,527 antibiotic patient-date pairs that fall post-index
only by procedure date (1,575 and 2,895 distinct patients within those source
rows). These are candidate record counts, not final per-patient label changes.
The local generator now dates the seven procedure outcomes from `d_odto`/
procedure `dato_start`; admission/contact dates remain for hospitalization.
The LPR2 variable catalogue defines `d_odto` as the start date of the
recorded operation/procedure, and `d_inddto` as admission/contact start, not
as the reporting date. The source-date audit now also counts procedures after
the recorded encounter end and those more than 30 days after encounter start;
review these plausibility checks before freezing procedure event dates.
This change and the already-local sodium/magnesium/ATC corrections require a
new versioned outcome run after sync and a differential audit; they did not
affect the earlier repaired-join run. The source-specific LPR3 gate decision
is resolved for these procedure outcomes below.

## Final procedure-date comparison and LPR3 gate decision

The source-date-corrected run compared with the repaired-join-only run has
2,910 additional event labels and 4,582 earlier event dates, summed across
seven overlapping outcomes, with no lost labels. Against the original export,
the combined repair has 22,208 additional labels and 5,484 earlier dates, with
none lost. These counts are not distinct patients and must not be added across
the comparisons.

The 365-day plot restricted to complete index years (through 2024) no longer
shows the old near-zero pre-2019 period for transfusions, antibiotics,
respiratory support, dialysis, or emergency stay. Antibiotics retain a gradual
calendar trend; the plot alone cannot establish equal capture across eras or
separate case mix from practice change. The sharp LPR3 discontinuity that
motivated masking the procedure outcomes was an artifact of the LPR2 join and
event-date defects. Thus `lpr3_coverage_gate.R` no longer gates antibiotics,
any/RBC/platelet transfusion, or emergency stay. Respiratory support and
dialysis were not in that gate. Keep the distinct hospitalisation,
infection-diagnosis, and SMR gates pending their own source/era review; this
procedure audit does not validate those outcomes.

Run the revised gate only on a fresh ungated, date-corrected outcome directory.
It does not restore pre-2019 rows in any previously gated parquet. Before
finetuning, complete the remaining grading, cutoff, and outcome-inventory
checks and record the final source-tier decisions in the release manifest.

## 5 October 2026: final source-coverage decision

Keep hospitalisation and all diagnosis-coded outcomes ungated across LPR3.
Hospitalisation has continuous event ascertainment in the reviewed era plots;
diagnosis-coded infections show changing rates, but no established abrupt
source failure or defensible common cutoff. Report calendar-era trends and
stratified model performance instead of inventing an LPR3 eligibility date.
The repaired procedure outcomes likewise remain ungated.

The only post-hoc source gate is `infections_IV_based`, an IV antimicrobial-
administration proxy derived from SMR. Use index date >= **2018-05-15**. The
Danish Health Data Authority states that the reporting regulation took effect
on that date and regional submission began then. This is a documented national
reporting start, *not* a guarantee of complete capture in every medicine,
department or region. Its data declaration specifically describes ongoing
incompleteness, including some antibiotic coverage gaps. Preserve that caveat
in the manuscript. Source: [SMR register description](https://sundhedsdatastyrelsen.dk/data-og-registre/nationale-sundhedsregistre/sygehusmedicinregisteret),
[SMR data declaration](https://sundhedsdatastyrelsen.dk/-/media/sds/filer/rammer-og-retningslinjer/sygehusmedicinregisteret/smr_datadeklaration_regioner.pdf).

Additional implementation finding before the final run: `infections_IV_based`
and `infections_prescription_based` had called the laboratory outcome wrapper,
which wrongly required lab coverage and censored non-events at the last lab
draw. They now use the existing non-lab administrative constructor; this
changes their eligible populations, event labels and follow-up. Consequently,
the prior sealed outcome panel remains provenance, not the final modelling
panel. Regenerate a *new versioned* full outcome run from the sealed first-
line cohort, then inspect the prescription-based endpoint by index/event era.
No prescription-register start date has been established from the evidence at
hand; do not impose a speculative calendar gate. Apply the SMR gate to a copy
of the new outcome directory, then build composites from that copy. Do not
retrofit old parquets in place.

The final modelling directory is assembled by
`python -m opera.run.assemble_outcome_panel` from the gated base outcomes, the
verified 19 March second-line export, and the five post-hoc composites. This
step requires the registry's exact canonical inventory, rejects duplicate
names, copies and hashes every parquet, and leaves the three source stages
untouched. Run the structural audit and provisional panel-decision audit on
that assembled directory before setting `BONSAI_OUTCOMES_DIR` for finetuning.

The earlier sealed-run `secondline/second_line_treatment.parquet` was discovered
to have maximum censor date **2026-03-04**, reflecting the former second-line
script default. It must not be reused with the 2026-03-19 primary outcome panel:
that would miss second-line events from 5–19 March and invalidate downstream
failure composites. The primary second-line script now defaults to 2026-03-19,
and the sealed-cohort runner enforces that date; 2026-03-04 remains available
only through an explicit standalone historical/sensitivity invocation. Rebuild
second line against the sealed first-line cache in a new output directory, then
check the maximum censor date before building composites.

## 5 October 2026: interpretation of panel-decisions report

`panel_decisions_INTERNAL.csv` is an advisory support/definition screen, not the
actual finetuning roster. Its thresholds use train/tuning events at day 365;
`opera/configs/experiment_registry.yaml` still lists all historical outcomes,
including retirements and definition holds, and the sweep generator does not
consume the decisions CSV. Do not launch a generated full-panel sweep assuming
that this report automatically removed anything.

The rubric was corrected where facts had become stale: ALP G2/G3 are now
definition holds rather than permanent retirements, because the measured ALP
elevations may be useful under verified legacy/numeric-threshold names; the
hyponatremia exact-boundary fix is acknowledged, while the symptom limitation
remains; the corrected prescription prophylaxis-code typo is acknowledged.
The potassium thresholds and leukocytosis export name were *not* fixed by
earlier comments: the local potassium bins still differ from CTCAE v6, and the
R leukocytosis call still exports a `g2plus` filename while requesting grade 3.

The two newer TLS labels are produced by the separate candidate exporter and
are not in the 87-outcome production registry or `final_panel` assembly.
Therefore they cannot appear in `panel_decisions_INTERNAL.csv`. Regenerate
them against this exact first-line/outcome run and review the NPU/unit and
monitoring audits before deciding whether to add either to a versioned
exploratory or primary model roster. The historical `tumor_lysis_syndrome`
proxy remains retired.
