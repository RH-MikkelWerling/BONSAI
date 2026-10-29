# Danish registry code assignments — audit for clinical review

**29 October follow-up:** the OPERA R folder, active 69-output release, ehr2meds lymphoma configuration and DALY-CARE package have now been inspected. Several questions below are superseded. See [the source audit and figure redesign](FIGURE_REDESIGN_AND_SOURCE_AUDIT_20261029.md) and the updated open-question CSV.

Audit date: 28 October 2026. This audits code meanings used by the attribution
configuration, clinical display groups, code dictionary and figure prototypes.
It also inventories explicit code references and generated malignancy code sets
in `opera/R/calculate_adverse_events_update.R`. It does **not** establish which
revision generated the frozen outcomes or validate the complete phenotyping
algorithms. No patient data, fitted vocabulary or model weights were available.

## Review files

- [All assignments, official definitions and sources](DANISH_CODE_ASSIGNMENT_AUDIT_20261028.csv).
  Repeated uses of the same code are kept, with their source file and context.
- [One row per system/code](DANISH_VERIFIED_CODE_EXTRACT_20261028.csv).
  Includes units, validity dates, historical codes and rejected former assignments.
- [24 attribution concepts](ATTRIBUTION_CONCEPT_MAPPING_REVIEW_20261028.csv).
- [Questions requiring your audit](DANISH_CODE_AUDIT_OPEN_QUESTIONS_20261028.csv).
  Each question has an identifier, evidence and the requested decision/check.
- [Source URLs, record counts and SHA-256 checksums](DANISH_CODE_SOURCES_20261028.json).

Public identity verification and clinical interpretation are separate. A code can
exist and still be too broad for the label assigned to it. Likewise, a verified
NPU code does not establish that it covers every local assay for that analyte.

## What was verified

The public sources are the [full Danish SKS catalogue](https://filer.sundhedsdata.dk/sks/data/skscomplete/SKScomplete.txt),
the [IFCC/IUPAC NPU release](https://cms.ifcc.org/wp-content/uploads/npu-codes-latest.csv)
linked by the [NPU terminology organisation](https://npu-terminology.org/npu-database/),
and Patobank's [molecular](https://www.patobank.dk/kodebog/?tag=V),
[lymphoid](https://www.patobank.dk/kodebog/?tag=E) and
[blood/bone-marrow](https://www.patobank.dk/kodebog/?tag=D) codebooks.
SKS was parsed according to [Sundhedsdatastyrelsen's fixed-width specification](https://filer.sundhedsdata.dk/sks/docs/SKS-tabelstruktur.pdf).
The download contains 107,495 records, including historical versions and closed
codes. SKS diagnosis codes carry `D`, and SKS medication codes can carry `M`
before the ATC code. Both forms are now handled consistently in attribution
concept matching and roll-up.

The assignment ledger covers the 24 configured concepts, every proposed summary
label, the built-in clinical groups and ATC group names, concrete registry-code
examples in the synthetic prototypes, five neutral morphology families, and the
R code references described above. It is not an audit of every code that may
exist in an unseen server vocabulary. The 22 broad ICD chapter ranges remain
structural chapter labels, rather than individual diagnosis assignments.

The only unmatched former proposed identities are NPU19123 and KTSF. Neither is
an active assignment now. Numerical category ranges also contain unused slots
such as D54 and I14. Those slots match no official code and do not indicate
missing patients or missing study records.

The ledger contains 520 uses of 345 distinct system/code pairs. These comprise
281 current/historical SKS codes, 17 NPU codes, 29 exact Patobank codes, five
neutral morphology prefixes, 11 unused numerical-range positions and the two
rejected former assignments. These counts describe the code audit, not patients,
endpoints or a measure of clinical validation.

## Molecular findings: concrete codes found

| Finding used in the atlas | Included positive-result codes | Limits |
|---|---|---|
| TP53 mutation/deletion recorded | `FE16E3`, `FE16E4`, `FK17D0` | Mutation, deletion and del(17p13) records. This does not establish variant pathogenicity or allele fraction. |
| MYC translocation recorded | `FE18UA`, `J3N8U1`, `J3N8U3`, `JFR8U1` | Generic positive MYC translocation and positive IGH::MYC, IGL::MYC or IGK::MYC records. Does not include mutation, amplification or protein expression. |
| Unmutated IGHV recorded | `FE13N8` | The specific IGHV-not-hypermutated result. Generic IGH mutation not detected (`FE13N0`) is not substituted for this. |

These identities are from the [Danish molecular codebook](https://www.patobank.dk/kodebog/?tag=V).
The config uses complete `PATHOLOGY//<code>` tokens and exact membership, rather
than a broad molecular prefix. For example, negative TP53 deletion `FK17D1`,
negative MYC translocation `FE18UB`, hypermutated IGHV `FE13N7`, and TP53 VUS
`FE16E9` cannot satisfy the corresponding positive-result concepts. The ledger
includes excluded codes alongside the included ones. TP53 insertion `FE16E2`
is recorded separately for review rather than treated automatically as a
pathogenic mutation. Missing testing is not a negative result.

Generic positive t(8;14) code `J08141` is an additional candidate for your audit.
Its public label does not specify genes or breakpoints. It is not included in
the MYC concept without confirmation of how it was used in the local assay data.

Historical introduction/closure dates for these pathology codes remain to be
checked. The Patobank HTML codebooks give current identities but not the full
historical validity intervals. The official SNOcomplete download failed TLS
certificate validation during this audit; the SKS download succeeded. No dates
were invented, and TLS verification was not disabled. Patobank provides
[historical updates and the official historical-catalogue link](https://www.patobank.dk/snomed/).
The actual `PATHOLOGY//` vocabulary and the available assay modalities also
remain to be checked on the server.

## Laboratory identities and units

All 15 NPU identities configured for the attribution atlas/curves are present in
the public NPU release. The two additional literal R constants, NPU02647 for
magnesium and NPU03096 for inorganic phosphate, are also verified.

| Analyte | NPU | System and unit |
|---|---|---|
| LDH | NPU19658 | Plasma, U/L, IFCC reference procedure |
| Albumin | NPU19673 | Plasma, g/L |
| Haemoglobin(Fe) | NPU02319 | Blood, mmol/L |
| Neutrophils | NPU02902 | Blood, 10⁹/L |
| Platelets | NPU03568 | Blood, 10⁹/L |
| Creatinine | NPU18016 | Plasma, µmol/L |
| CRP | NPU19748 | Plasma, mg/L |
| ALT | NPU19651 | Plasma, U/L, IFCC reference procedure |
| ALP | NPU27783 | Plasma, U/L, specified procedure at 37°C |
| Bilirubins | NPU01370 | Plasma, µmol/L |
| Ferritin | NPU19763 | Plasma, µg/L |
| Lymphocytes | NPU02636 | Blood, 10⁹/L |
| Eosinophils | NPU01933 | Blood, 10⁹/L |
| Sodium | NPU03429 | Plasma, mmol/L |
| Potassium | NPU03230 | Plasma, mmol/L |
| Magnesium | NPU02647 | Plasma, mmol/L |
| Inorganic phosphate(P) | NPU03096 | Plasma, mmol/L |

Source: [IFCC/IUPAC NPU definitions](https://cms.ifcc.org/wp-content/uploads/npu-codes-latest.csv).
Full formal definitions, procedure qualifications and units are retained in the
CSV extract. They describe the NPU definitions, not a check of the stored values
or their unit conversions. The Danish LabTerm/local laboratory aliases and any
additional equivalent NPU codes need comparison with the source data.

The earlier draft's NPU19123 assignment to LDH is not supported by this public
release and has not been retained. This is not proof that NPU19123 never existed.
NPU19658 is a verified LDH code, but its presence in the actual fitted vocabulary
still needs confirmation. No stored tokens were rewritten.

The atlas labels now say **lower/upper bins**, rather than clinically low/high.
The selection uses relative bin positions. Clinical abnormality needs actual
bin boundaries, units and appropriate reference intervals. Expected directions
remain plausibility checks on the fitted model, not clinical-effect estimates.

## Display corrections made

| Assignment | Correction and evidence |
|---|---|
| `BOQA` | Blood transfusion generally. Its volume-based children do not establish the component transfused. Removed the automatic red-cell label. |
| `I48` | Fibrillation **or flutter**, rather than fibrillation alone. |
| `I80` | Broad thrombophlebitis/phlebitis category, including superficial disease. Removed the narrow VTE claim from the display group. |
| `G45` | Transient cerebral ischaemia and related syndromes. Display label reflects this broader scope. |
| `B44–B46` | Selected fungal diagnoses, rather than proof of invasive fungal disease. `DB448A` explicitly denotes Aspergillus colonisation. |
| `B59` / `B485` | Historical and current Danish Pneumocystosis codes are both included in the display group. B59 closed 31 December 2018; B485 applies from 1 January 2019. |
| `C88` | Separate malignant immunoproliferative-disease display group. |
| `J01D` | Other beta-lactam antibacterials. The earlier cephalosporin/carbapenem label was incomplete. |
| `H02` | Systemic-steroid display group uses H02A/H02B. H02C antiadrenal preparations are separate. The official ATC umbrella heading is retained as an official heading. |
| `H02AB` | Systemic glucocorticoids, rather than every corticosteroid. |
| `C03C`, `A10A`, `J05A` | High-ceiling diuretics; insulins and analogues; direct-acting antivirals. |
| `M95`–`M99` | Neutral morphology-family names. M95 includes precursor findings and M97 includes mast-cell disorders. Removed numeric-range guesses at exact diagnoses. |
| Prototype `KTSF` | No matching SKS code/prefix found. Replaced the synthetic biopsy example with verified `KTNE25A`, iliac-crest marrow biopsy. Its four-character roll-up `KTNE` is labelled minor pelvic procedures, not biopsy. |

SKS findings are supported by the [historical Danish catalogue](https://filer.sundhedsdata.dk/sks/data/skscomplete/SKScomplete.txt).
Morphology findings are supported by the [blood](https://www.patobank.dk/kodebog/?tag=D)
and [lymphoid](https://www.patobank.dk/kodebog/?tag=E) codebooks. The source extract
preserves exact official wording. Current labels should not be read as a claim
that all historical diagnostic concepts were identical.

The synthetic prototypes now use verified molecular codes and relative-bin
labels too. Their patients, event dates, effects and counts remain synthetic.
No prototype result is evidence about the fitted model.

## Endpoint questions that need priority review

These concern **definitions in the repository R script**. They are not yet
evidence that the frozen outcomes used in the current results contain these
problems. Establish that provenance first. No endpoint membership, event time,
training label or trained model was changed in this audit.

1. **Pneumocystosis after 2018.** `ICD10_PCP_PREFIXES` contains B59 alone. Review
   whether the frozen serious-infection outcome includes the replacement B485.
   The table display is now corrected, but changing a display cannot recover
   an event that was omitted from outcome construction.
2. **Invasive fungal disease.** Full B44 includes non-invasive disease and
   colonisation. Decide whether the intended endpoint is broader fungal
   diagnoses or a clinically justified invasive subset.
3. **VTE.** Full I80 includes superficial disease. Decide whether that is part
   of the intended endpoint or whether a more specific deep-VTE definition was
   used elsewhere in the frozen pipeline.
4. **Red-cell transfusion.** BOQA0/BOQA00–05 alone establish blood transfusion,
   not the component. Check whether a product-specific source or local coding
   convention supports the existing RBC endpoint name.
5. **“MACE”.** The code composite includes AF/flutter and VTE alongside heart
   failure, MI and stroke/TIA. Report its actual components rather than implying
   a conventional three-point MACE endpoint. Review the G45 inclusion too.
6. **New haematological malignancy.** D47 includes uncertain/unknown-behaviour
   conditions. Confirm the intended endpoint, exclusions and event validation.
   This audit verifies the code meaning, not whether a newly recorded code
   represents a clinically new malignancy.

The open-questions CSV also covers molecular assay interpretation, historical
pathology validity, local NPU aliases, unit/bin boundaries and external
`load_npu_common()` constants that are not defined in this repository.
The R diagnosis helpers also assume SKS-prefixed input. Their attribution
counterparts now support bare and SKS forms, but that does not change the R
helpers or establish the format of the frozen endpoint inputs.

## Implementation checks and next use

Attribution matching now applies the same SKS/WHO normalisation as dictionary
lookup. Previously `MEDICATION//MJ01CA04` could receive a correct name but fail
the configured `MEDICATION//J01` membership test, or roll up to J01 rather than
J01C. Raw and SKS-prefixed medication codes, and bare/SKS diagnosis codes, now
have the same membership and roll-up. Molecular result codes remain exact and
namespace-specific. Vocabulary coverage, summary coverage and concept effects
use the same matcher.

Regression tests exercise positive versus negative molecular results, VUS and
mutation/translocation distinctions, namespace/suffix boundaries, SKS/WHO
equivalence, subcategory grouping for B485, safe morphology labels, historical
SKS parsing and preservation of Danish result polarity.

The relevant six test files pass (59 tests). Four attribution design proofs were
rebuilt with the verified-code proposals and inspected visually. Their layout
checks found no text-collision candidates or overflowing legends. These are
synthetic previews under
`scratch/danish_code_audit_20261028/prototypes/`, not refreshed fitted-model
results. The actual server summaries/figures still need rebuilding.

The configuration remains `concept_mapping_reviewed: false`. The new mappings
are explicit, source-backed tuning-rehearsal proposals. Check the fitted
vocabulary and bin metadata before locking them for held-out use. These changes
require attribution summaries and relevant figures to be rebuilt from the
existing token-attribution shards. They do not require recomputing IG merely to
change names or concept membership. Their clinical consequences cannot be
settled by an image rerender alone.

To reproduce the public-code ledger from the downloaded sources:

```bash
PYTHONPATH=. python -m opera.run.audit_danish_code_assignments \
  --sources scratch/danish_code_audit_20261028 --output-dir docs
```

To check against the actual vocabulary in the secure environment:

```bash
python -m opera.run.audit_vocabulary \
  --vocabulary /actual/path/to/vocabulary.pt \
  --config opera/configs/attribution_analysis.example.yaml \
  --output-dir /actual/analysis/path/danish_code_vocabulary_audit \
  --search 'TP53|MYC|IGHV|IgHV|17p|FE16E|FE18U|FE13N|FK17D|J3N8U|JFR8U'
```

Use actual paths, then inspect `concept_coverage.csv`, `code_meanings.csv`,
`vocabulary_formats.csv` and the original bin-boundary metadata. An absent
token is not a negative molecular finding. The vocabulary command alone cannot
establish assay sensitivity, phenotype validity or stored laboratory units.
