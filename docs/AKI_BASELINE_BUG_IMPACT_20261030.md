# AKI baseline-window bug: OPERA impact assessment

30 October 2026. The author now confirms that the reversed-window implementation
was used for OPERA’s AKI input. Treat the error as confirmed. The exact server
writer/path is still useful for reproducing the repair. No frozen phenotype,
checkpoint, analysis recipe or manuscript result was changed in this audit.
See the [recovery and compute plan](AKI_CORRECTION_RECOVERY_PLAN_20261030.md).

## Confirmed source and exact error

Inspected DALY-CARE commit `b9c12e7261df868d7e5c9653841b95dea9b05862`:

- [core_scripts/AE_AKI.R](https://github.com/RH-CLL-LAB/dalycare_package_v2/blob/b9c12e7261df868d7e5c9653841b95dea9b05862/core_scripts/AE_AKI.R)
  contains the reported `base_mean_7d365` calculation. It identifies its origin
  as `AKI_function_toNGC_v2.R` and explicitly distinguishes itself from the
  other AKI implementation under `functions/`.
- [functions/AE_AKI.R](https://github.com/RH-CLL-LAB/dalycare_package_v2/blob/b9c12e7261df868d7e5c9653841b95dea9b05862/functions/AE_AKI.R)
  is a different algorithm, calling a historical median helper. Finding that
  function loaded somewhere does not identify the producer of the imported CSV.

For a current sample at time `t`, the published annual baseline selects:

```text
sample_time >= t - 365 days AND sample_time >= t - 7 days
```

The first condition is redundant. The effective window is **from seven days
before t onward, without an upper limit**. It includes the current sample and
all later measurements for the same patient supplied to the function. Sorting
by datetime does not restrict the vectors inside the per-patient calculation
to previously visited rows.

The proposed correction selects:

```text
sample_time >= t - 365 days AND sample_time < t - 7 days
```

Swapping equivalent expressions (`x < y` and `y > x`) would be harmless. The
published expression is not equivalent: its inequality is reversed. This is a
future-dependent outcome-ascertainment error. It does not, on its own, establish
that future measurements entered the model's pre-treatment features.

## Demonstrated consequences

The independent base-R reproduction in
[reproduce_window_effect.R](../scratch/aki_baseline_audit_20261030/reproduce_window_effect.R)
uses only synthetic histories and reproduces the four published detection
branches plus Boolean missing-value handling and episode starts. It passed.

| Synthetic current sample | Original annual baseline | Corrected annual baseline | Detection consequence |
|---|---:|---:|---|
| Creatinine 150, with an older stable history and gradual recent rise | 150 | 91 | Original misses the episode onset. Correction detects it despite all three other branches remaining false. |
| First observed creatinine 160, followed 30 days later by 40 | 100 | Missing | Original flags the first sample using the future low value. The intended historical algorithm has no qualifying reference at that time. |

These examples establish that both additions and removals are possible under
the intended definition. They do not establish clinical truth for those
hypothetical patients or estimate the frequency of the problem in DALY-CARE.
If the annual branch is redundant with another positive branch, a measurement's
AKI flag may stay unchanged. Episode onset can still change when earlier flags
change, so comparing only counts of positive measurements is insufficient.

## How OPERA uses the table

The actual import is:

- `opera/R/calculate_adverse_events_update.R:23`: obtains the path from
  **OPERA_AKI_TABLE**.
- The required filename is **AKI_table_all_dalycare_simplified.csv**. It is read
  with a tab delimiter, despite its `.csv` suffix. Required fields are
  `patientid`, `aki_start_time`, `aki_episode`.
- At line 449 the external table is read. `aki_start_time` becomes
  `samplingdate`.
- At line 1047 its onset dates enter `aki_first_documented` through
  `events_from_table`. This helper selects the first dated onset strictly after
  treatment index and no later than administrative follow-up.

`aki_episode` is renamed to an apparent grade field, but that field is **not
used** in this call. Changing episode numbers or stage alone would therefore
not change the exported target. Changing onset dates can change it.

The local repository, downloaded aggregate bundle and available Git history do
not supply the absolute path of the CSV or its original writer. The environment
variable's value, original outcome-job settings or the server source lock are
needed. The author has since confirmed that this reversed-window implementation was used.
The original writer is still needed to reproduce the correction exactly.

The downloaded historical tuning `model_cells.csv` does contain **First
documented AKI**, including 15 disease rows at one year. This confirms that the
endpoint is actually represented in the available results. It is not an unused
legacy name. These rows do not identify the CSV's producing algorithm.

## Direct and indirect scope

| Component | Effect of this specific condition |
|---|---|
| First documented AKI | Confirmed affected definition; patient-level label/date changes still need quantification. |
| Creatinine increased G2+/G3+ | Different grader comparing against laboratory upper reference limits. This annual AKI mean is not an input. |
| Creatinine >1.5 times pre-index baseline | Separate Python baseline from the latest pre-index day within the preceding 365 days. It does not use the rolling annual AKI mean. |
| Dialysis | Separate procedure-defined endpoint. No dependency on this annual mean found. |
| Composites in build_composites.R | None of the listed composites uses AKI as a component. |
| Mortality and clinical-index outcome dates | Not directly defined using this AKI calculation. |
| Shared supervised models | Potential indirect effect because AKI contributes supervised loss and may supply borrowed outcome corrections. The size of this effect is unknown. |
| Raw-record pretraining | No dependency on this calculation found in the OPERA outcome code. This finding alone is not a reason to stop national pretraining. If an external vocabulary build included derived AKI tokens, inspect that separately. |

Coverage eligibility in the R lab engine is based on coverage/baseline/entry,
not rescued by an event. However, the engine uses administrative censoring for
event cases and monitoring-based censoring for event-free cases. A corrected
AKI status can therefore change censor dates as well as event dates. Apply the
whole existing export rule when measuring the effect, not just a new event flag.

Dropping AKI from an evaluation table does not remove its influence from a
previously trained shared encoder or routed outcome bank. Conversely, one
suspect target does not demonstrate that the other targets or the overall
sharing result are invalid.

## Smallest useful next audit

1. Find the external CSV and the exact script/version that wrote it. The public
   core source's `AKI_function_toNGC_v2.R` provenance is an additional search clue.
2. Recompute that implementation with **only this window correction**, in an
   isolated directory. Keep mean/median choice and other detection criteria
   unchanged in this first comparison so the cause of differences is clear.
3. Re-derive episodes from the full relevant patient histories, including
   pre-index samples. Then apply the same post-index outcome/censor rules.
4. On development data first, compare original versus corrected measurement
   flags, onsets, first post-index dates, censor dates and event status at all
   six horizons. Report both directions of change, native patient denominators
   and the fraction of cases captured independently by the other branches.
5. Only then decide whether a limited corrected-label sensitivity is sufficient
   or affected supervised fits need revision. Frozen-model evaluation against
   corrected labels is useful as a sensitivity, but it is not equivalent to
   training against the corrected definition.

This phenotype comparison can be done before opening test predictions or
performance. Existing router/layout audits can finish while provenance is
resolved. The AKI definition should be resolved before final empirical sign-off.

## Separate definition questions

Correcting this inequality does not by itself establish exact NHS/KDIGO
concordance. The core implementation uses a historical **mean** and explicitly
adds a most-recent-sample ratio without a maximum time gap. The
[NHS England algorithm](https://www.england.nhs.uk/wp-content/uploads/2014/06/psa-aki-alg.pdf)
uses the median for the 8–365-day reference and a minimum for the recent
seven-day reference. These are separate definition choices to document or audit,
not additional changes to bundle silently with this one-line bug fix.
