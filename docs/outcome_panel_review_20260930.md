# Outcome panel crosswalk and support review (internal, 30 September 2026)

This is a **complete registry-level inventory** of the 87 outcomes in
`opera/configs/experiment_registry.yaml`, compared with the supplied
`opera/R` source and [NCI CTCAE v6.0](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf).
The active production DALY-CARE loader and source data are unavailable locally.
Therefore "potentially alignable" is not a certification of the installed
grader. No existing parquet, model config or primary-panel membership was
changed by this review.

## What the four categories mean

- **Direct v6 lab overlap:** the locally visible rule captures a v6 lab
  finding, subject to assay, unit and reference-range validation. It does not
  establish treatment attribution or incident toxicity.
- **Alignable after verification/regrading:** v6 has the term and an operational
  rule could be built, but the current threshold, clinical qualifier,
  baseline branch, boundary or imported grader needs checking/changing.
- **Legacy/proxy, not a v6 grade:** it may be clinically useful, but the current
  `g2plus`/`g3plus` name or syndrome name does not describe a v6 grade.
- **Outside CTCAE grading:** clinical diagnosis, treatment, care-use, survival
  or composite outcome. A similar CTCAE term may exist; these files do not
  implement its grades.

## Complete inventory

| Status | Registered outcomes | R/source assessment |
| --- | --- | --- |
| Direct v6 lab overlap, conditional | `LDH_increased`, `haptoglobin_decreased` | Binary >ULN / <LLN matches the v6 G1-only lab terms in structure. Verify NPU, cleaned units and patient-specific reference interval. |
| Lab proxy of v6 term | `hyperuricemia`, `hyperphosphatemia` | Binary >ULN; v6 higher grades require clinical consequences/intervention. Phosphate G1 additionally assumes intervention is not indicated. Do not claim full v6 grading. |
| Local grader, regrade/boundary check | `hyponatremia_g2plus`, `hyponatremia_g3plus`, `hypokalemia_g2plus`, `hypokalemia_g3plus`, `hypomagnesemia_g2plus`, `hypomagnesemia_g3plus`, `hypermagnesemia_g3plus` | Potassium-low bins differ materially from v6; sodium-low requires symptom information to split G2/G3; `cut()` right-closed exact boundaries affect sodium/magnesium. Hypermagnesemia has no G2 in v6, consistent with the R call. |
| Imported grader, v6 term exists; installed implementation unverified | `ALT_increased_g2plus`, `ALT_increased_g3plus`, `AST_increased_g2plus`, `AST_increased_g3plus`, `anemia_g2plus`, `anemia_g3plus`, `bilirubin_increased_g2plus`, `bilirubin_increased_g3plus`, `creatinine_increased_g2plus`, `creatinine_increased_g3plus`, `hypercalcemia_g2plus`, `hypercalcemia_g3plus`, `hyperkalemia_g2plus`, `hyperkalemia_g3plus`, `hypernatremia_g2plus`, `hypernatremia_g3plus`, `hypertriglyceridemia_g2plus`, `hypertriglyceridemia_g3plus`, `hypoalbuminemia_g2plus`, `hypoalbuminemia_g3plus`, `hypocalcemia_g2plus`, `hypocalcemia_g3plus`, `hypoglycemia_g2plus`, `hypoglycemia_g3plus`, `INR_increased_g2plus`, `INR_increased_g3plus`, `lymphocytosis_g2plus`, `lymphocytosis_g3plus`, `neutropenia_g2plus`, `neutropenia_g3plus`, `thrombocytopenia_g2plus`, `thrombocytopenia_g3plus`, `WBC_decreased_g2plus`, `WBC_decreased_g3plus` | Must inspect the server's exact `AE_*` function bodies and units. ALT/AST, bilirubin and creatinine can depend on baseline; INR on anticoagulation. V6 ANC G3 begins <0.5 x 10^9/L; the inspected public package revision was v5-derived. V6 platelet G4 begins <10 x 10^9/L. These are **potentially alignable**, not confirmed aligned. |
| Imported grader, v6 term exists but the current named grade does not | `ALP_increased_g2plus`, `ALP_increased_g3plus`, `lymphopenia_g2plus`, `lymphopenia_g3plus`, `leukocytosis_g2plus` | V6 ALP and lymphopenia have only G1, so their current G2/G3 labels cannot be v6. `leukocytosis_g2plus` actually requests threshold 3; v6 lacks G2 and warns against leukemia-related increases. Retire or rename these, rather than claim v6 alignment. |
| Imported grader, limited v6 overlap | `eosinophilia` | V6 G1 requires >ULN **and >baseline**; check whether the imported binary grader tests the baseline increase. Higher v6 grades are treatment-based, not measured here. |
| Local grader, related v6 term but incomplete | `hypofibrinogenemia_g2plus`, `hypofibrinogenemia_g3plus` | V6 *Fibrinogen decreased* includes LLN-relative and abnormal-baseline branches. The local grader implements only the first and assumes units for fixed cutoffs. The small event counts support retirement from the next primary panel rather than rebuilding it now. |
| Local clinical/lab definition, not a CTCAE grade | `aki_first_documented`, `dic_like_coagulopathy`, `hepatotoxicity`, `hys_law_dili`, `severe_cytopenia`, `tumor_lysis_syndrome` | AKI uses a separate renal/creatinine ascertainment; the others are unions or multi-lab proxies, not adjudicated v6 syndromes. Preserve definitions if exploratory, but do not call them CTCAE grades. |
| Diagnosis-coded clinical events, not CTCAE graded | `atrial_fibrillation`, `bone_fracture`, `heart_failure`, `herpes_zoster`, `invasive_fungal`, `MACE_composite`, `myocardial_infarction`, `next_hematological_malignancy`, `next_malignancy`, `sepsis`, `serious_infection_composite`, `stroke_TIA`, `VTE` | ICD-code events and diagnosis composites; some concepts have CTCAE terms, but ICD presence does not yield v6 severity or attribution. `invasive_fungal` uses B44/B45/B46 only. `serious_infection_composite` is sepsis or selected fungal/PCP codes, not all serious infections. |
| Medication, procedure, care-use or disease-course outcome; outside CTCAE grading | `antibiotics`, `any_transfusion`, `dialysis`, `emergency_stay`, `hospitalisation`, `infections_all_definitions`, `infections_IV_based`, `infections_prescription_based`, `overall_survival`, `platelet_transfusion`, `RBC_transfusion`, `respiratory_support`, `second_line_treatment`, `treatment_failure`, `treatment_failure_transformation` | Valuable possible endpoints, but not v6 grades. The two medication infection outcomes are antimicrobial-use proxies, not confirmed infections; the all-definitions union inherits them. Failure outcomes are unions of second-line, death and (for transformation) next hematological malignancy. |

The full threshold-level discrepancies and remediation options are in
`docs/ctcae_v6_endpoint_review_20260930.md`. This inventory deliberately
does not declare an imported grader correct from its function name alone.

## Overlaps that affect interpretation or headline weighting

1. Most G3+ lab endpoints are nested within their G2+ counterpart: including
   both gives one biological axis multiple votes in an unweighted macro score.
2. `neutropenia`, `WBC_decreased`, `lymphopenia`, `anemia` and `severe_cytopenia`
   overlap biologically; the last is the union of severe neutropenia,
   thrombocytopenia and anemia.
3. `creatinine_increased`, `aki_first_documented` and `dialysis` cover related
   renal dysfunction at different levels of ascertainment/severity.
4. `hepatotoxicity` is a union of G3+ ALT, AST and bilirubin; `hys_law_dili`
   co-occurs on lower transaminase/bilirubin thresholds but is *not* adjudicated
   Hy's law. `dic_like_coagulopathy` can be INR plus platelets without low
   fibrinogen. `tumor_lysis_syndrome` combines lab axes with approximate cutoffs.
5. `serious_infection_composite` contains sepsis plus selected fungal/PCP
   diagnoses. `infections_all_definitions` then unions that result with IV and
   prescription antimicrobial proxies, mixing diagnosis and medication-use
   ascertainment. Count overlap before retaining both in the headline panel.
6. `MACE_composite` combines AF, VTE, heart failure, MI and stroke/TIA codes.
   That is **not a conventional narrow MACE definition**; rename or redefine
   it for the next release. `treatment_failure` overlaps second-line and death;
   `treatment_failure_transformation` adds next hematological malignancy.

## What the supplied event audit establishes

The internal `outcome_export_audit_20260930T085117Z.csv` has **87/87** registry
outcomes and split counts for each. Across those exports it reports zero
duplicate IDs, missing index/censor dates, invalid event/index/censor order,
invalid splits or split-date mismatches. These checks **do not validate** the
clinical definition, assay, registry capture or fixed-horizon support.

| Priority example | Train events | Tuning events | Held-out events | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `hypofibrinogenemia_g3plus` | 72 | 8 | 25 | Remove from next primary panel; assay and grading also unverified. |
| `invasive_fungal` | 30 | 5 | 17 | Important concept, but very weak training support and selected-code definition. |
| `hypoglycemia_g3plus` | 109 | 4 | 11 | Sparse tuning support; inspect 365-day support and unit/threshold. |
| `hyperkalemia_g3plus` | 795 | 51 | 125 | Do **not** drop for rarity alone; verify grader and 365-day held-out support. |
| `ALP_increased_g3plus` | 632 | 49 | 101 | Plenty of support does not rescue a non-v6 grade label. |

These are *ever post-index within observed follow-up* counts, not 365-day
events. Eligible denominators vary by outcome because of source-coverage and
composite-intersection rules, so compare rates only with their risk sets.
Do not publish small cells without the project's disclosure review.
For the full count sheet, regenerate or inspect the internal CSV from
`python -m opera.run.audit_outcome_exports --outcomes-dir ... --output ...`.
Risk-set exclusion details live alongside individual outcome parquets as
`<outcome>__audit.parquet` (lab/composite) and
`<outcome>__coverage_audit.parquet` (calendar-gated sources); their existence
and provenance must be checked on the server, not inferred from the aggregate
CSV. The local audit source is not a substitute for the final MEDS release.

### Historical sweep heatmap reconciliation (internal)

The supplied `sweep_test_events.pdf` (dated 26 June 2026) plots **cumulative
test events** for candidate `TEST_START` years 2019–2024. The **2023-onward**
column is the closest historical analogue of the present held-out split;
**2022-onward minus 2023-onward** approximates the 2022 tuning-year event
count. Neither is a 365-day event count. The PDF contains 75 of today's 87
registered outcomes; later medication/composite definitions are missing.

Across the 75 shared endpoints, the old 2023-onward counts differ from the
current held-out counts by a median of 15 events (32/75 differ by at most 10;
8/75 match exactly). The reconstructed 2022-only counts differ from today's
tuning counts by a median of 3 events (60/75 differ by at most 10; 10/75 match
exactly). This supports using the heatmap as a **historical support screen**,
not as current final-release provenance: source tables and label rules can
change, and the sizable second-line-treatment difference needs separate
lineage review.

| Endpoint | PDF 2022-only / 2023+ | Current tuning / held-out | Reading |
| --- | ---: | ---: | --- |
| `hypoglycemia_g3plus` | 4 / 11 | 4 / 11 | Sparse in both periods. |
| `invasive_fungal` | 4 / 16 | 5 / 17 | Sparse in both periods. |
| `hypofibrinogenemia_g3plus` | 8 / 25 | 8 / 25 | Sparse in both periods. |
| `herpes_zoster` | 27 / 47 | 27 / 48 | Limited tuning support, especially at short horizons. |
| `hyperkalemia_g3plus` | 50 / 118 | 51 / 125 | More held-out events, but retain only after grader/assay check. |

All cells above are internal review figures; apply disclosure rules before
publication. The final MEDS release needs its own split/horizon support table.

## Proposed decision rule for the next version

Do **not** target a particular number of endpoints. Keep an endpoint in the
primary panel only if (1) its clinical estimand is clear, (2) source and
eligibility support reliable ascertainment, and (3) enough distinct training
and evaluation events exist at the intended horizons. Otherwise relabel,
rebuild, move to exploratory analysis or retire it. This need not be exhaustive
of clinically interesting complications; a smaller well-defined panel is
preferable to an apparently comprehensive but ambiguous one.
