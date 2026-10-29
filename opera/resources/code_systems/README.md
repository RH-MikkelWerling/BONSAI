# Code systems: from model tokens to clinical meaning

The model's tokens are register codes (`DIAGNOSIS//DC833`,
`MEDICATION//J01CA04`, `LABTEST//NPU19673//BIN_3`). The attributions are read
through `opera.interpretability.code_meaning.CodeDictionary`, which gives every
token a source, a code system, a concept (the level the analysis summarises
at), a name, a clinical group and a chapter.

Names come from, in order:

1. reviewed labels in the analysis config (`summary.concept_labels`), used only
   when `concept_mapping_reviewed: true`;
2. official code tables listed under `code_tables` in the analysis config
   (below);
3. the built-in structure in this folder: ICD-10 chapters (WHO), ATC groups
   at levels 1 and 2 (WHO), neutral morphology code families,
   and `clinical_groups.yaml`;
4. a readable form of the code (`ICD-10 C83`).

`python -m opera.run.audit_vocabulary` reports, for the real vocabulary, which
token formats occur, how many tokens each step names, and which unnamed codes
matter most for the attributions, so that review effort goes where it counts.

## Tables from DALY-CARE itself (simplest)

Most source tables carry the text next to the code (laboratory: NPU code and
analysis name; prescriptions and administrations: ATC code and drug name;
pathology: SNOMED code and text). `python -m opera.run.build_code_table`
reads those two columns and writes one `code,label` row per code (its most
frequent name; codes seen fewer than 5 times are dropped, so one-off free text
never reaches the table):

```bash
python -m opera.run.build_code_table --system npu --input <lab table> \
  --code-column <NPU column> --label-column <analysis name column> \
  --output resources/npu_daly_care.csv
```

and list it under `code_tables` with `code: code, label: label`. Diagnoses
(ICD-10) and procedures (SKS) have no text in the national patient register;
use the official SKS file below for those, or rely on the built-in chapters
and the reviewed `concept_labels`.

## Official tables

Classifications are public and may be copied into the secure environment. Any
delimited file (csv, tsv, txt) or Excel file with a code column and a name
column works. Codes are normalised (case, dots and the SKS `D` prefix of
diagnoses), so SKS and WHO forms both match.

| System | Source | Columns to point at |
|---|---|---|
| ICD-10 (English) | WHO ICD-10, checked against the Danish SKS codes actually used | code, short description |
| ICD-10 / SKS (Danish) | Sundhedsdatastyrelsen, SKS-browseren, download "SKS komplet" | code, text |
| SKS procedures | the same SKS file (procedure codes K, B, U, N, Z ...) | code, text |
| ATC | WHO ATC/DDD index, or the Danish Medicines Agency ATC list | code, name |
| NPU | Sundhedsdatastyrelsen, NPU terminology download | code, short definition |

```yaml
code_tables:
  - {system: icd10, path: /project/resources/icd10_en.csv, code: code, label: name}
  - {system: atc, path: /project/resources/atc.csv, code: code, label: name}
  - {system: npu, path: /project/resources/npu.xlsx, code: NPU code, label: Short definition}
  - {system: sks_procedure, path: /project/resources/sks_procedures.csv, code: code, label: text}
```

NPU definitions (`P—Albumin; mass c. = ? g/L`) are shortened to the component
and, outside blood, plasma and serum, its system (`Albumin (urine)`).

The built-in names are short English versions of the WHO titles. Clinical
groups are a draft for review.

## Danish public-code audit, 28 October 2026

See [the audit and unresolved clinical questions](../../../docs/DANISH_CODE_AUDIT_20261028.md)
and [the assignment-by-assignment ledger](../../../docs/DANISH_CODE_ASSIGNMENT_AUDIT_20261028.csv).
The audit uses the full historical Danish SKS catalogue, the public IFCC/IUPAC
NPU release and the Danish Patobank molecular, lymphoid and blood codebooks.
It retains historical codes rather than discarding codes that are closed today.
The source manifest records download URLs and checksums. The full downloads
are under `scratch/danish_code_audit_20261028/` and are not patient data.

The SKS download is fixed-width cp1252 data, not a delimited CSV. Use
`opera.run.audit_danish_code_assignments.read_sks` to parse it before creating
per-system code tables. The public NPU CSV uses `npu_code` and
`short_definition`; NPU code identity is international, while study coverage
and local laboratory aliases need the Danish source data. Danish Patobank's
classification is not interchangeable with SNOMED CT or a generic ICD-O lookup.

Exact molecular result codes distinguish detected, not detected, normal and
unknown-significance findings. Configured molecular concepts use
`exact_codes: true`. Neither absent testing nor a negative result is evidence
for an unrecorded positive finding. Broad morphology prefixes retain their
code-family names until an official table or reviewed exact label is available.
Laboratory lower/upper bins describe the model vocabulary, not clinical
reference intervals. Public identity verification does not set
`concept_mapping_reviewed: true` without the server vocabulary/bin audit.
