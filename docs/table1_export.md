# Table 1 extraction

## Disclosure-reviewed exports

`--disclosure-policy primary-only` provides a surgical INTERNAL REVIEW draft:
only counts 1–4 (and their percentages), and continuous summaries with fewer
than five observed patients, are masked. No complementary suppression is applied.
This is NOT disclosure-cleared and can reveal masked counts through subtraction.
Use only for deciding the final reporting layout with the institutional reviewer.
`python -m opera.run.format_table1_release --input <exact aggregate CSV>
--output-dir <new directory> --disclosure-policy primary-only` reformats existing
server-side exact aggregates without reprocessing MEDS histories.

New runs write exact aggregates, metadata inventory and provenance under
`LOCAL_ONLY_internal/`. Keep that directory on the server. Only
`release_review/` is intended for release review. Older exports directly in
the output root are NOT sanitized retroactively; do not bundle them.

The default policy is now `--disclosure-policy targeted`; the older broad
policy remains available as `--disclosure-policy conservative`.
The targeted policy suppresses individual small cells, adds complementary
cells along explicit category, Overall/split and pooled/grouped/fine additive
margins, and protects small local missingness and binary complements. Common
cells elsewhere remain visible. It does not prove safety against arbitrary
combinations of equations or external releases; the full bundle must be reviewed.
NMZL and SMZL are not merged, renamed or excluded.

The threshold is 5 (`--suppression-threshold`). Counts 1–4 display
`<5`; complementary suppressed cells display `Suppressed`. A characteristic
with any small count, complementary binary/category count, observed denominator
or missingness count is withheld across ALL cohorts and splits. Score-system,
score-value and paired-score releases are suppressed together. Tiny cohort
partitions trigger suppression of patient totals and cohort distributions.
Numeric counts, percentages and summary statistics are blanked in suppressed
rows; denominator fields are withheld throughout the release CSV. The
variable-wide rules above describe the optional conservative policy, not the
targeted default. Neither policy is optimal minimum-information suppression.
Review the complete intended publication bundle (including external tables and
previous releases) under institutional disclosure rules. This tool is not a
GDPR compliance certificate and cannot account for arbitrary external knowledge.

## Raw MEDS history summaries

Optionally add `--meds-dir /path/to/event-level/meds` and
`--meds-code-prefixes /path/to/verified_prefixes.json`. The MEDS root can contain
`data/train`, `data/tuning`, `data/held_out`, or these split directories directly.
These are **physical pretraining/preprocessing shards**, not the supervised
train/tuning/held-out groups. The exporter pools events by subject ID and uses
the sealed population metadata's `split` column for Table 1 columns. It prints
event coverage by supervised split and refuses a split with zero pre-index
history. Parquet shards are streamed in batches. Use event-level MEDS rows;
verify any binned-code effects before naming a statistic a raw measurement.

Prefix JSON must contain `distinct_medications`, `distinct_diagnoses`,
and `laboratory_measurements`, with optional `n_unique_specialties` and
`predominant_region`; each key maps to a nonempty list of verified code
prefixes. Do not add the optional keys until the raw MEDS vocabulary confirms
their token families. For example, ONLY if those namespaces exist in your source:

```json
{
  "distinct_medications": ["RX//"],
  "distinct_diagnoses": ["DX//"],
  "laboratory_measurements": ["LAB//"]
}
```

These examples are not assumed production mappings.
The optional specialty summary counts distinct complete specialty codes;
the region summary takes the modal complete region code among pre-index MEDS
events. Ties are missing with `tied_mode` status. A configured context family
with zero pre-index matches fails rather than yielding an all-missing table.
Inspect the region code values and event multiplicity before calling the mode
"predominant region"; repeated tokens might weight some contacts more heavily.
The inspected `t100_joined_bins_with_sor` event source uses `MEDICATION//`,
`DIAGNOSIS//`, `LABTEST//`, `CONTACT_SPECIALTY//` and `REGION//`; these are
configured in `opera/configs/table1_meds_prefixes.json`. The vocabulary
inspection sampled 30 batches, rather than an exhaustive codebook, but found
examples of all five configured families.
Its lab codes contain `/BIN_` suffixes. Counting these rows does not recover
raw measurements: use them only as laboratory-coded record counts, after
confirming preprocessing preserved one row per source event. Medication and
diagnosis counts exclude `DIAGNOSIS_TYPE//`, procedure and contact namespaces.
The namespace `SEX//F`/`SEX//M` belongs to event codes; Table 1 instead uses
the population metadata's explicit `sex` values Female/Male.

Unique complete raw codes
are counted, so diagnosis/medication ontology granularity must be checked.
All timed coded rows strictly before treatment count as recorded events;
laboratory counts are matching raw rows, not unique tests or visits. Duplicate
source rows are counted as supplied. `--history-lookback-days 1095` optionally
uses a fixed lookback; default is all available pre-index history.
Recorded history span is time from the first retained event to index, NOT a
registry coverage duration. Patients with no retained events get missing
summaries, not zeros. Hospital encounters and comorbidity scores are not guessed.
Sex is categorical as supplied; no redundant female binary row is emitted when
`sex` exists.

Use a frozen included-patient manifest, not the union of model predictions or
outcome-eligible populations. Provide the existing temporal split labels; the
script deliberately does not guess splits from calendar dates.
Use `--inspect-inputs` with the usual arguments to print aggregate row counts,
column names, split values and population coverage, without exporting tables.
`--split-manifest` can supply existing assignments; `--split-map` accepts explicit
JSON aliases such as `{"validation":"tuning","test":"held_out"}`. Neither
option authorizes inventing assignments or excluding unmatched patients.
All joins are left joins onto this population, requiring one record per patient. Duplicate or
conflicting records fail; no patients are silently added or dropped.

If the overall-survival patient list defines the frozen analysis cohort, add
`--restrict-to-overall-survival` to explicitly use its intersection with the
supplied population. Excluded population IDs and survival-only counts are
reported and recorded in provenance. Split labels remain mandatory for every
included patient. Do not use this option merely to hide missing assignments if
the full membership population is the intended study cohort.

`PS`, `IPI`, `Disease` and `arm_primary` are retained as
separate categorical characteristics. Age is derived from birth/index dates
when an explicit age field is absent, using elapsed days divided by 365.25.
`arm_day1_primary` is retained only in the internal column inventory, not in
Table 1 summaries. No patients are excluded because of rare treatment categories.

```bash
python -u -m opera.run.export_table1 \
  --population "$BONSAI_COHORT_MEMBERSHIP" \
  --overall-survival "$BONSAI_OUTCOMES_DIR/overall_survival.parquet" \
  --output-dir "$PWD/data/results/table1"
```

Add repeatable `--metadata /path/to/baseline_characteristics.parquet` for baseline
age, sex, history summaries and disease-specific variables. `--population-metadata`
is an alias. Recognized treatment groups, named IPI
systems, performance status and stage fields are reported categorically, retaining
their source names and codes. Case/space differences in supported column names
are recognized without recoding values. NCCN-IPI, CLL-IPI, FLIPI and MIPI remain
separate fields. `metadata_inventory.csv` flags which columns were included;
use an explicit column map for unsupported equivalent names. Prefer disease-specific
supplements for the score distributions rather than treating them as one pooled IPI.
The population must
be the actual frozen analysis cohort; if the membership file contains additional
patients, use the correct locked included manifest instead. Descriptive held-out
demographics do not involve model selection or evaluation predictions.

Outputs:

- `table1_pooled_display.csv`: manuscript-friendly Overall/train/tuning/held_out.
- `table1_values_long.csv`: numeric counts, denominators, median/IQR and missingness
  across pooled/grouped/fine cohorts and splits, for local table design.
- `LOCAL_ONLY_internal/metadata_inventory.csv`: available field names, types and coverage, without
  patient-level values. Send this first if column names need mapping.
- `LOCAL_ONLY_internal/provenance.json`: source paths, mapping and denominator conventions.

Optional JSON `--column-map` maps canonical names to actual column names, e.g.
`{"age": "age_at_treatment", "ecog": "performance_status"}`. Only use clinically
equivalent definitions. Binary columns `female`, `death_event` and
`prior_non_hematological_malignancy` require explicit 0/1 coding. `sex` is reported
categorically as supplied, never guessed from numeric codes. Missing binary values
are not interpreted as absence. Continuous canonical fields and optional disease
fields are listed at the top of the script. Entirely unavailable continuous/binary
fields appear as Not available; missing categorical fields appear in the inventory
only when present. Percentages use nonmissing denominators; inspect missingness
before formatting the final table. No imputation is performed. Disclosure
suppression is applied to the release-review outputs as described above; apply
your institutional data-release policy before moving outputs off the server.

History/diagnosis/medication/lab counts must be supplied as pre-index summaries;
the script does not derive them from the sequenced tokens. Baseline clinical fields
must refer to index or earlier. Follow-up is raw observed duration, not a
reverse-Kaplan–Meier estimate, and must come from overall survival.
The survival input can contain explicit event/time_days or the standard outcome
date/censor-date schema; the latter uses the existing BONSAI label conversion.
Disease stages, IPI/ISS and cytogenetic risk are not comparable across all diseases: use the fine
cohort supplementary rows where appropriate. Ascertainment passes belong in the
CONSORT diagram, not Table 1; the manuscript disease-profile section is the fine
disease distribution.
