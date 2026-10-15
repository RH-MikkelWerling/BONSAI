# Traceability: `opera_data_definitions_supplement.tex`

Maps each definition stated in the supplement to the repository location that
implements or records it. Line numbers refer to commit `8bdd49e`. Since then
only comments/diagnostic config changed: the hospitalisation header comment in
`AE` was corrected (lines after 1758 are shifted by +3 in the working tree) and
`opera/configs/endpoint_dependencies.yaml` was corrected to match the R
composites (see `paper/opera_data_definitions_open_questions.md`, items 27 and
31). Section headings give the corresponding supplement sections (S1-S10).
Verification items and code/document conflicts removed from the supplement are
listed in `paper/opera_data_definitions_open_questions.md`. Abbreviations:

- `FL` = `opera/R/identification_first_line_treatment_v2.R`
- `SL` = `opera/R/identification_second_line_treatment_v5.R`
- `AE` = `opera/R/calculate_adverse_events_update.R`
- `COMP` = `opera/R/build_composites.R`
- `GATE` = `opera/R/lpr3_coverage_gate.R`
- `REL` = `opera/run/build_approved_outcome_release_v2.py`
- `CAND` = `opera/run/export_candidate_outcomes_v1.py`
- `REG` = `opera/configs/experiment_registry.yaml`

"Doc" rows are statements taken from a design document rather than code.

## S1. Data sources and linkage (incl. data cutoff)

| Statement | Evidence |
| --- | --- |
| Loader, `patient`, `t_dalycare_diagnoses`, ICD-10 codebook from verified cache; live SQL refused for release | FL:108-145; `opera/R/run_firstline_candidate_export.R`:7-8; `opera/R/freeze_firstline_loader_inputs.R`:17-59 |
| RKKP diagnosis rows re-prioritised to priority 1 | FL:223-224 |
| `diagnoses_all`, `t_dalycare_diagnoses` loaded for outcomes | AE:46-47 |
| Treatment file name enforced; `SP beh` rows removed | FL:157-187 |
| RKKP sources LYFO, DaMyDa, DCLLR; `TRUST_RKKP <- TRUE` | FL:153-155 |
| CLL V4 / MM V5 seed files enforced | FL:170-175; SL:873-880 |
| RKKP_LYFO second-line date used; CLL/DaMyDa loaded, cross-check commented out | SL:848-859, 912-926 |
| Laboratory table, `clean_lab_values()`, `load_npu_common()` | AE:40-43 |
| External AKI table and required columns | AE:23-31, 449-455 |
| LPR2/LPR3 tables loaded | AE:1541-1546 |
| LPR2 lossless text-key join; missing keys never match | AE:1566-1592 |
| LPR3 join still casts `dw_ek_kontakt` with `as.integer()` | AE:1548-1562 |
| Procedures dated by procedure start (`d_odto`, `dato_start`) | AE:1551, 1558, 1581, 1588, 1594-1599 |
| SMR source `SDS_indberetningmedpris` | AE:2340-2345 |
| Prescription source loaded as `SDS_ekokur` | AE:2352-2356 |
| IPI used for metadata only | `opera/R/build_sealed_population_metadata.R`:25-82; `docs/approved_outcome_release_v2_runbook.md`:103-107 (Doc) |
| `date_death_fu` described as min(death, last follow-up) | AE:588-590 |
| LPR2 overflow audit: 355,581 / 1,576,198 keys; LPR3 zero lost keys | `docs/outcome_release_adjudication_20261004.md`:106-108 (Doc); audit script `opera/R/audit_procedure_join_keys.R` |
| Repaired run: +22,208 labels, 5,484 earlier dates (7 outcomes, not patients) | `docs/outcome_release_adjudication_20261004.md`:190-195 (Doc); `docs/lpr2_join_manager_check_20261005.md`:59-63 (Doc) |
| Outcome cutoff 2026-03-19 | AE:610 |
| Second-line cutoff default 2026-03-19, only 03-04/03-19 allowed | SL:835-839 |
| Sealed runner requires 2026-03-19 | `opera/R/run_secondline_from_sealed_firstline.R`:19-22 |
| Composite builder refuses differing max censor (2L vs OS) | COMP:58-61 |
| Death after cutoff censored, not converted to event | AE:1700-1712 |
| Diagnoses-coverage audit exists, results not archived | `opera/R/audit_diagnoses_all_coverage.R`:1-52 |

## S2-S3. Study population, index date and disease classification

| Statement | Evidence |
| --- | --- |
| Mapped table exported at counts marker; FROZEN branch not exported | `opera/R/run_firstline_candidate_export.R`:16-42; FL:2508-2528 (marker), 2904-2924 (FROZEN) |
| Registry-date fallback not exported | FL:3592-3602 |
| Downstream scripts read the mapped CSV | AE:290-293; SL:789-792 |
| Diagnosis anchor = earliest `t_dalycare_diagnoses` record | FL:810-819 |
| Record filter precedence (RKKP, heme ICD, excl ICD, excl text/drug, keep) | FL:575-581 |
| Heme ICD prefixes DC81-DC89, DC90-DC96, DD47 | FL:437-441 |
| Non-heme ICD prefixes list | FL:448-458 |
| Excluded drug text list | FL:463-473 |
| Danish indication exclusion terms | FL:479-503 |
| Search window dx-30 d to dx+3650 d (pass 1) | FL:683-689 (filter), 821-823 (`pre_dx_grace_days = 30`), 651 (`max_days_from_dx` default 3650) |
| 21-day episode gap rule | FL:652, 694-701 |
| Therapy-like gate rules (a)-(e) | FL:739-749 |
| Anti-CD20 suspect detection incl. `"cd29 antibody"` regex | FL:718 |
| CD20 monotherapy enabled in release branch | FL:821-823, 1509-1516 (`accept_cd20_mono = TRUE`) |
| Anchor code lists (CLL, aggressive B, MM text, RT, additional) | FL:234-246, 251-282, 290-347, 350-399, 404 |
| Prophylaxis-suspect list removed from anchors | FL:406-424 |
| Seeds: earliest `line == 1` date; earliest across CLL/MM | FL:192-220 |
| Seed episode = seed within or <=30 d of episode | FL:752-764, 821-822 (`seed_max_gap = 30`) |
| Seed episodes bypass therapy gate | FL:772 |
| Selection: seeded closest, else earliest; index = episode start | FL:777-802 |
| ICD-10 group dictionaries and local supplements; DD472A -> EXCLUDE_SECONDARY | FL:848-907 |
| SLL/LPL/NMZL/SMZL exact overrides; MZL emptied | FL:918-935 |
| Non-assignable groups (DALYCARE, UNS, *_UNS, MGUS, MBL) | FL:951 |
| Aggressiveness ranks; derived RT/TRANSFORMED_FL | FL:952-984 |
| Pass-1 window: lookback 2*3650 d, grace +14 d, priority threshold 3 | FL:1084-1086, 1343-1350 (call), 1284-1287 (grace default 14) |
| Confirmation rules | FL:1105-1112 |
| Derived Richter / transformed FL, 90-day rule, full mapped history | FL:1167-1230 |
| Fine beats supergroup, aggressive beats non-aggressive, tie-break order | FL:1241-1247 |
| Grouped mapping; RT_DERIVED -> RT in pass 1 | FL:1370-1388 |
| Pass 2: anchor = earliest confirmed assignable dx | FL:1458-1501 |
| Pass 2: treatment window anchor-90 d to +20 y, seeds retained | FL:1509-1517 |
| Pass 2: cohort lookback 3650 d, grace 90 d | FL:1533-1541 |
| Pass 2 does not rename RT_DERIVED | FL:1546-1563 (compare 1375) |
| Pass-2 rows replace pass-1 rows (new index) | FL:1590-1599 |
| Arm rescue | FL:2187-2264 |
| `arm_exclude_primary` not used to filter export | FL:2172-2173, 2508-2511 (only EXCLUDE_SECONDARY filtered) |
| EXCLUDE_SECONDARY removed | FL:2508-2511 |
| 24 fine / 10 grouped labels | `opera/analysis_manifest.yaml`:47-85; REG:29-60 |
| Registry per-label counts (sum 38,905, historical) | REG:29-60 |
| Flow counts 78,668 ... 37,557 | `paper/figures/cohort_flow.tex`:2-40 (Doc); `docs/opera_paper_next_actions_20261007.md`:13-21 (Doc) |
| 37,557 / 37,531 / 29,105 / 1,980 / 6,446; 26 without split | `docs/supervisor_validation_cohort_table1_20261007.md`:437-443 (Doc); `docs/manuscript_result_lock_20261007.md`:104-106 (Doc); `docs/table1_supervisor_handoff_20261014.md`:6-7 (Doc) |
| Historical counts 37,540 (V4/V4), 37,482 (30 Sept) | `docs/firstline_release_decision_20261003.md`:24-28 (Doc); `docs/ascertainment_review.md`:223-232 (Doc) |
| Outcome population: patient row and index < `date_death_fu` | AE:307-313 |
| Outcome population: index < 2026-03-19 | AE:1035-1040 |
| Reasons enumerated by audit (not archived) | `opera/R/audit_outcome_population_gate.R`:21-35 |
| 26 kept in metadata with missing split | `opera/R/build_sealed_population_metadata.R`:57-64, 101; runbook:84-86 (Doc) |

## S4. Temporal split and prediction time

| Statement | Evidence |
| --- | --- |
| train < 2022-01-01, tuning 2022, held-out >= 2023-01-01 | AE:619-637; `opera/R/run_secondline_from_sealed_firstline.R`:45-54; `opera/configs/manifests/temporal_split.yaml`:1-16 |
| Composites refuse disagreeing index/split | COMP:110-115; REL:39-41 |
| Sequences truncated strictly before index (non-inclusive) | `bonsai/functional/censoring.py`:8-35; `opera/modules/datasets/ContrastiveDataset.py`:105-119; `opera/functional/outcomes.py`:37-53 |
| Pretraining cutoff 2022-01-01, non-inclusive | `opera/configs/daly_care_pretrain.yaml`:66-69; `opera/configs/dapt.yaml`:149-152; `bonsai/modules/datasets/PretrainDataset.py`:44-58, 77-78 |
| Base pretrain config leaves cutoff empty | `configs/pretrain.yaml`:47 |
| Vocabulary cutoff 2022-01-01 | `configs/daly_care_data.yaml`:17-21 |
| Pre-2022 histories of later tuning/held-out patients included | `docs/ascertainment_review.md`:138-145 (Doc) |

## S5. Disease-course endpoints

| Statement | Evidence |
| --- | --- |
| Rebuild first line and require byte-identical mapped CSV | `opera/R/run_secondline_from_sealed_firstline.R`:23-40 |
| Disease configuration table | SL:91-117 |
| CLL_SLL/MM excluded from episodes; BL_LBL/HCL unreliable | SL:119-120 |
| CD20-only, maintenance and ASCT code sets | SL:126-148 |
| Single-day first-line end imputation | SL:156-182 |
| Episodes after first-line end + min gap | SL:269-323 (279-280) |
| ASCT/consolidation/maintenance/CD20-single classification | SL:350-425 |
| Retreatment-like gate incl. RKKP >=2 codes or dates | SL:388-394 |
| Confidence levels; secondline = retreatment + >= moderate | SL:243-261, 404-410 |
| Earliest secondline episode = event | SL:513-535 |
| Censor = min(death if status 1, cutoff) | SL:572-576, 832-833 |
| Eligibility mask (no follow-up, unreliable cohort, weak-only) | SL:617-625 |
| RKKP-LYFO 2L override/rescue | SL:637-673, 855-859 |
| CLL/MM curated line-2 events; absent = censored negative | SL:700-782, 884-910 |
| Export writes eligible only | SL:1438-1499 |
| Header claims anti-CD20 alone excluded | SL:17-19 (contrast 388-394, 413-421) |
| Availability exclusion AMYLOIDOSIS/BL_LBL/HCL | REG:61-67; `opera/run/generate_sweep_configs.py`:119-124 |
| Treatment failure composites and membership | COMP:161-167 |
| next_hematological_malignancy definition | AE:1308-1311, 1516-1525 |

## S5-S6. Outcome definitions

| Statement | Evidence |
| --- | --- |
| Laboratory engine (first post-index grade >= threshold; censor rule) | AE:867-878, 894-975, 1000-1022 |
| Administrative engine | AE:1622-1695 |
| New-code engine (block not seen on/before index + grace) | AE:1320-1337, 1354-1461 |
| Infection-diagnosis engine (no pre-index exclusion) | AE:2157-2222 |
| Python candidate engine: baseline median, latest day <365 d before index | CAND:98-112 |
| Candidate gradable baseline rules | CAND:115-123 |
| Candidate thresholds (ALT, creatinine relative) | CAND:126-157 |
| Candidate eligibility = R audit eligible set; censor rule | CAND:224-312 (230-235, 290-297) |
| Horizons 30-730; primary 365 | REG:7; `opera/configs/outcome_panel_decision_v1.yaml`:4-5 |
| `n_hours_start_include = 1`; competing OS for non-death outcomes | `opera/run/generate_sweep_configs.py`:170-191 |
| `outcomes_containing_death` | REG:12-15 |
| Upstream grader revision a0742b0 cites v5; platelet 25e9/L | `docs/ascertainment_review.md`:147-154 (Doc) |
| overall_survival | AE:1700-1731 |
| next_malignancy C00-C97 minus C44, C80; grace 31 | AE:1299-1301, 1504-1513 |
| sepsis, zoster, fungal, serious_infection_composite codes | AE:2148-2154, 2207-2222 |
| antibiotics `^BPH` | AE:2018-2027 |
| IV proxy: IV route, 7-day gap, >=1 day span; prophylaxis filter commented | AE:2228-2289, 2340-2348, 2364-2365 |
| Prescription proxy: exclusions J01EE01, J05AB01, J05AB11, P02 | AE:2294-2336, 2352-2359, 2366-2367 |
| `ATC_AB()` external | AE:2342, 2354 |
| Release renames and notes | `opera/configs/outcome_release_primary_v2.yaml`:26-59 |
| External grader calls (anaemia, neutropenia, WBC, platelets, lymphocytosis, eosinophilia, haptoglobin, creatinine, bilirubin, INR, ALP, hypernatraemia, hyperkalaemia, calcium, triglycerides, albumin) | AE:459-551 (objects), 1121-1241 (calls) |
| severe_cytopenia, hepatotoxicity, infections_all_definitions composites | COMP:170-180 |
| hepatotoxicity rebuilt with candidate ALT | `opera/configs/outcome_release_primary_v2.yaml`:34-43; REL:27-49, 160-164 |
| Transfusion codes BOQA0, BOQA00-05; BOQC, BOQC3; any_transfusion union | AE:1975-2010 |
| aki_first_documented via lab engine | AE:1044-1049 |
| creatinine_rise definition, eligibility | CAND:23, 115-123, 135, 243-283, 355-363 |
| dialysis `^BJFD` | AE:2046-2055 |
| ALT candidate rules | CAND:136-141, 282-283 |
| ALP legacy grader behaviour (server check) | `docs/approved_outcome_release_v2_runbook.md`:12-16 (Doc); `opera/configs/outcome_release_primary_v2.yaml`:51-52 |
| hyponatraemia local grader (<130 / <125, LLN default 136, right-open) | AE:82-113, 1181-1182 |
| potassium <3.0 candidate | `opera/run/export_potassium_below_3_v1.py`:22-80 |
| hypomagnesaemia (`NPU02647`, <0.5, LLN default 0.7) | AE:162-192, 1187 |
| hyperphosphataemia (`NPU03096` fallback, ULN default 1.45) | AE:1078-1114 |
| hyperuricaemia, LDH (> ULN; missing ULN -> 0) | AE:208-233, 252-279, 1223-1229 |
| Urate unit comment without conversion | AE:269-271 |
| Cardiovascular codes | AE:2110-2137 |
| Hospitalisation algorithm | AE:1790-1922 |
| Hospitalisation comment `c_pattype == 1` vs code 0 (comment corrected; resolved) | AE:1760 vs 1819 (working tree: 1759-1764, 1822) |
| emergency_stay, respiratory_support | AE:2032-2041, 2060-2070 |
| bone_fracture codes, no pre-index exclusion | AE:2081-2102 |

## S7. Eligibility, ascertainment and censoring

| Statement | Evidence |
| --- | --- |
| Go-live 2017-01-01; windows 365/30/180 | AE:779-786 |
| Coverage table built from any analyte | AE:803-860 |
| Source covered / baseline / entry definitions (index day in both) | AE:831-835, 856 |
| Entry exemption for early death post go-live; pre-go-live exemption off | AE:853-855, 897, 912-914 |
| Events never override eligibility | AE:916-917 (comment at 768 contradicts) |
| Reasons list | AE:919-929 |
| Censoring: event -> admin end; else last draw + 180 | AE:931-936 |
| SMR gate 2018-05-15 only gated outcome | GATE:43-57; `opera/R/run_coverage_gate_versioned.R`:9-29 |
| Release builder verifies SMR start for IV and union | REL:24, 173-176 |
| LPR3 procedure gate removed; hosp/dx ungated (decision) | `docs/outcome_release_adjudication_20261004.md`:199-221 (Doc) |
| Sidecar contract `eligible` / `ascertainment_eligible` | `DATA_FORMAT.md`:138-182; `opera/evaluation/cohort_flow.py`:130-160; `opera/functional/outcomes.py`:95-172 |
| No sidecars configured; `registry_start_date: null` | REG:157-162; `opera/run/generate_sweep_configs.py`:175-185 |
| Known-status horizon mask | `opera/modules/networks/competing_risk.py`:438-455 |
| Competing death only if <= outcome censor and in window | `bonsai/functional/outcomes.py`:202-327 (275-300) |

## S8. Selection of the outcome panel

| Statement | Evidence |
| --- | --- |
| 87 registry outcomes | REG:69-156 |
| Floor >=100 train, >=20 tuning at 365 d | `opera/configs/outcome_panel_decision_v1.yaml`:4-8; `opera/configs/outcome_release_primary_v2.yaml`:4 |
| Support = events within 365 d among eligible rows | `opera/run/decide_outcome_panel.py`:68-102 |
| Retirements / holds with reasons | `opera/configs/outcome_panel_decision_v1.yaml`:13-67 |
| Clinical actions override support; held-out descriptive only | `opera/run/decide_outcome_panel.py`:105-145 |
| include_held / exclude_held / rename / replace / add | `opera/configs/outcome_release_primary_v2.yaml`:8-48 |
| Builder resolves holds, checks floor, writes manifest `held_out_used_for_selection: False` | REL:114-125, 179-191, 215-221 |
| 60 primary candidates (derived set) | `opera/configs/outcome_display_groups_60_candidate_v1.yaml` |
| Nine below-floor outcomes | Derived: REG outcomes minus decision-rule actions minus 60-candidate taxonomy (not read from a server report) |
| TLS candidate definition and counts 243/45/114, 207/41/100 | `opera/run/export_candidate_tls_v1.py`:21-170; `docs/outcome_release_adjudication_20261004.md`:22-24 (Doc) |

## S9. Composites, endpoint groups and display groups

| Statement | Evidence |
| --- | --- |
| Intersection / earliest censor / earliest event <= censor; death by membership | COMP:9-26, 72-156; REL:27-49 |
| `endpoint_dependencies.yaml` memberships (corrected to match R; resolved) | `opera/configs/endpoint_dependencies.yaml`:5-14 (working tree: 5-18) |
| R MACE membership | AE:2110-2117 |
| R serious infection membership | AE:2152-2154 |
| Endpoint groups v1 (grade suffix, ALP pair, sodium pair) | `opera/functional/endpoint_groups.py`:6-18 |
| 52 endpoint groups | Derived by applying `endpoint_groups_v1` to the 69 names |
| Endpoint-group macro aggregation and selection metric | `opera/run/generate_sweep_configs.py`:530-535; `opera/run/materialize_primary_training_families_v1.py`:117-124 |
| Display groups: 17, additions placed | `opera/run/materialize_primary_display_groups_v2.py`:18-47; `opera/configs/outcome_display_groups_v1.yaml` |

## S10 and open questions. Limitations and conflicts

| Statement | Evidence |
| --- | --- |
| Manifest status pending sign-off | REL:215 |
| Existing supplement's LPR3 gate, lab-gated proxies, 03-04 cutoff, JO1EE01 | `paper/opera_ascertainment_supplement.tex`:164-175, 210-211; current code GATE:53-57, AE:2361-2367, SL:835, AE:2355 |
| Pass-1 window -20 y / +14 d (not +-14 d) | FL:1084-1086, 1347 |
