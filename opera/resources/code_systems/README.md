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
   at levels 1 and 2 (WHO), lymphoid and myeloid morphology ranges (ICD-O-3),
   and `clinical_groups.yaml`;
4. a readable form of the code (`ICD-10 C83`).

`python -m opera.run.audit_vocabulary` reports, for the real vocabulary, which
token formats occur, how many tokens each step names, and which unnamed codes
matter most for the attributions, so that review effort goes where it counts.

## Official tables

Classifications are public and may be copied into the secure environment. Any
delimited file (csv, tsv, txt) or Excel file with a code column and a name
column works. Codes are normalised (case, dots and the SKS `D` prefix of
diagnoses), so SKS and WHO forms both match.

| System | Source | Columns to point at |
|---|---|---|
| ICD-10 (English) | WHO ICD-10 or the public ICD-10-CM order file (3-character categories match WHO) | code, short description |
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
