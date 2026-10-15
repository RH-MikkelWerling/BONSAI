# Open questions: `opera_data_definitions_supplement.tex`

Internal file. Collects every verification flag and code/document conflict that
was removed from the supplement when it was rewritten as journal-style
Supplementary Methods (14 October 2026). The supplement now states each
definition as implemented, in neutral language; the items below are what still
needs confirming before submission. File:line references follow
`paper/opera_data_definitions_traceability.md` and refer to commit `8bdd49e`
(abbreviations as in that file: `FL`, `SL`, `AE`, `COMP`, `GATE`, `REL`,
`CAND`, `REG`). Lines in `AE` after 1758 are shifted by +3 in the working tree
because of the header fix in item 31.

Section references (S1-S10) point to the rewritten supplement.

## Verification items (previously "To verify 1-26")

1. **Source snapshots and loader revision (S1).** The SQL extraction date /
   snapshot of every source table, and the DALY-CARE loader and package
   revision used for the sealed release, are not recorded in the repository.
   Record them in the release manifest. Evidence: FL:108-145;
   `opera/R/run_firstline_candidate_export.R`:7-8;
   `opera/R/freeze_firstline_loader_inputs.R`:17-59.
2. **Prescription dataset name (S1, S6).** The prescription source is loaded as
   `SDS_ekokur`; design documents (and a code comment) refer to `SDS_epikur`.
   Confirm the dataset name and that it is the Danish National Prescription
   Registry. Evidence: AE:2352-2356.
3. **Meaning of `date_death_fu` and `status` (S1, Conventions).** Code
   describes `date_death_fu` as "already min(death, last follow-up)". Confirm
   what "last follow-up" means for living patients (extraction date,
   emigration or last contact) and the coding of `status`. The supplement
   states "earlier of death and last follow-up". Evidence: AE:588-590.
4. **Completeness of sources to the cutoff (S1).** 2026-03-19 is a literal
   constant, not a computed source maximum. Confirm that every source is
   complete up to this date (`opera/R/audit_diagnoses_all_coverage.R`:1-52
   checks diagnoses only; its result is not archived). Evidence: AE:610.
5. **"cd29 antibody" regex (S2, step 5c).** The anti-CD20 text pattern reads
   `"rituximab|ofatumumab|cd29 antibody"`; "cd29" is probably a typo for
   "cd20", so text-only "cd20 antibody" records do not trigger the
   anti-CD20-monotherapy rule. The supplement lists only rituximab/ofatumumab
   text. Evidence: FL:718.
6. **Diagnosis `priority` semantics (S3).** The meaning of DALY-CARE diagnosis
   priority values (other than RKKP = 1) is defined externally; the threshold
   "priority <= 3" (supplement: "reliability rank of 3 or better") cannot be
   audited from this repository. Evidence: FL:223-224, 1105-1112.
7. **Richter / transformed-FL timing (S3).** Timing uses the earliest
   *mapped* (not necessarily confirmed) code over the patient's full history,
   although the code comment says "earliest confirmed". Supplement states the
   implemented rule ("earliest recorded ... over the full history"). Evidence:
   FL:1167-1230.
8. **`RT_DERIVED` label in pass 2 (S3).** Pass 1 renames `RT_DERIVED` to `RT`;
   pass 2 does not. Confirm that no released patient carries `RT_DERIVED`.
   Evidence: FL:1546-1563 (compare FL:1375).
9. **Arm rescue contributed no patients (S3).** The flow implies the arm-rescue
   stage is empty; confirm from the release run's `pipeline_counts.csv` that
   `arm_rescued_cohort` is empty. Evidence: FL:2187-2264.
10. **DD472A = IgM MGUS (S3).** Confirm that SKS DD472A denotes IgM MGUS (the
    cohort flow labels the 9 excluded patients as IgM MGUS). Evidence:
    FL:848-907, 2508-2511.
11. **No PCD/MZL fine labels; historical registry counts (S3).** Confirm that
    no sealed patient carries `PCD` or `MZL` as fine label. Per-label counts in
    `experiment_registry.yaml` sum to 38,905 and are historical, not the sealed
    release. Evidence: REG:29-60; `opera/analysis_manifest.yaml`:47-85.
12. **Archive release `pipeline_counts.csv` (S2, Supplementary Table S2).** The
    pass-level flow counts (78,668 ... 37,557) were supplied by the study team
    and are not archived as `pipeline_counts.csv`. Archive and reconcile.
    Earlier runs (V4/V4 seeds: 37,540; 30 Sept 2026: 37,482) are historical and
    must not be mixed with this flow. Evidence: `paper/figures/cohort_flow.tex`:2-40;
    `docs/opera_paper_next_actions_20261007.md`:13-21;
    `docs/firstline_release_decision_20261003.md`:24-28;
    `docs/ascertainment_review.md`:223-232.
13. **Reasons for the 26 patients without a survival row (S2).** The outcome
    gate (patient-table row, index < `date_death_fu`, index < 2026-03-19)
    explains absence from all R outcomes, and the supplement attributes the 26
    exclusions to it, but the per-reason breakdown
    (`opera/R/audit_outcome_population_gate.R`:21-35) and the clinical
    explanation are not recorded in the repository. These 26 remain in the
    population metadata with a missing split. Unresolved. Evidence: AE:307-313,
    1035-1040; `opera/R/build_sealed_population_metadata.R`:57-64, 101.
14. **Pretraining cutoff of the final checkpoint (S4).** Confirm from the
    resolved Hydra configuration of the final 256-wide checkpoint that
    `training.cutoff_date` = 2022-01-01 was active (the base
    `configs/pretrain.yaml` leaves it empty). Evidence:
    `opera/configs/daly_care_pretrain.yaml`:66-69; `opera/configs/dapt.yaml`:149-152;
    `configs/pretrain.yaml`:47.
15. **Availability rule applied (S5).** Confirm that the approved
    training/evaluation configuration applies the `availability_rules`
    exclusion of BL_LBL, HCL and AMYLOIDOSIS from second-line-dependent
    outcomes; otherwise BL_LBL/HCL contribute event-only risk sets. Evidence:
    REG:61-67; `opera/run/generate_sweep_configs.py`:119-124.
16. **Second-line censoring for living patients (S5, S10).** For living
    patients the second-line censor is the cutoff, whereas overall survival and
    all other R outcomes use `date_death_fu`. If `date_death_fu` precedes the
    cutoff for living patients (e.g. emigration), second-line follow-up extends
    beyond observed follow-up. Also confirm whether any of the 26 patients
    without a survival row appear in the second-line file. Evidence:
    SL:572-576, 832-833; AE:588-590.
17. **External grader cut-points (S6).** For every outcome graded by the
    DALY-CARE `AE_*` functions, archive the installed function body, NPU codes,
    units and numeric cut-points; the release names ("g2plus", "g3plus")
    describe the grader's output, not a verified CTCAE v6 grade. The inspected
    upstream revision `a0742b0` documents several as CTCAE v5-derived.
    Evidence: AE:459-551, 1121-1241; `docs/ascertainment_review.md`:147-154.
18. **Antimicrobial ATC list; P02 exact match (S6).** `ATC_AB()` (the
    antimicrobial ATC selection) is an external DALY-CARE function; archive its
    code list. The exact-match exclusion of `P02` does not remove detailed P02
    sub-codes. Evidence: AE:2294-2336, 2342, 2354.
19. **Continuation coding in diagnosis-based infections (S6, S10).**
    Diagnosis-based infection outcomes take the first post-index code without
    excluding continuation coding of a pre-index episode. Evidence:
    AE:2140-2222.
20. **External AKI algorithm (S6).** The AKI table algorithm (baseline, KDIGO
    criteria, meaning of `aki_episode`) is external and undocumented in the
    repository; the script comment asks to "ask Christian". Evidence:
    AE:23-31, 449-455, 1044-1049.
21. **ALT candidate abnormal-baseline multipliers (S6).** Confirm the
    1.5x / 2.0x baseline multipliers against the intended reference before
    describing them as CTCAE-aligned. Evidence: CAND:136-141.
22. **Local NPU constants; urate units (S6).** Confirm the NPU constants
    (`NPU.NAT`, `NPU.KAL`, `NPU.FOS`, `NPU.URAT`, `NPU.LDH`; `NPU02647` and
    `NPU03096` are hard-coded fallbacks) and unit distributions on the server.
    The urate grader does not convert units although its comment says it
    should (not mentioned in the supplement). Evidence: AE:82-113, 162-192,
    208-233, 252-279 (269-271), 1078-1114.
23. **Hospitalisation timing and SKS code families (S6).** LPR3 contacts are
    timed from `dato_start`/`dato_slut` without clock times, so the 12-hour
    rule reduces to "ends on a later calendar day" (stated as a limitation in
    S10 for contacts without clock times). Confirm the clinical meaning of the
    BPH (antibiotic treatment), NAB (emergency stay) and BGDA (respiratory
    support) code families. The `c_pattype` part of this item is resolved
    (item 31). Evidence: AE:1790-1922 (working tree +3); AE:2018-2070.
24. **Laboratory go-live 2017-01-01 (S1, S7).** The 2017-01-01 laboratory
    go-live is an implemented constant; its source-capture justification is
    not in the repository. Evidence: AE:779-786.
25. **No eligibility sidecars in the release configuration (S7).**
    `DATA_FORMAT.md` defines optional sidecars (`eligible`,
    `ascertainment_eligible`); in the approved release the outcome parquet
    itself is the risk set and `registry_start_date` is null. Confirm that the
    generated release configuration contains no `eligibility_file` entries.
    Evidence: REG:157-162; `opera/run/generate_sweep_configs.py`:175-185;
    `DATA_FORMAT.md`:138-182.
26. **Nine below-floor outcomes (S8).** Derived as the set difference between
    the registry, the clinical rules and the checked-in 60-candidate taxonomy;
    confirm against the server `panel_decisions_INTERNAL.csv`, which also holds
    the support counts. Evidence: `opera/run/decide_outcome_panel.py`:68-145;
    `opera/configs/outcome_display_groups_60_candidate_v1.yaml`.

## Code/document conflicts (previously "Code/doc conflict" section)

27. **`endpoint_dependencies.yaml` composite membership — RESOLVED.** The YAML
    listed MACE = MI + stroke/TIA, serious infection = sepsis + IV +
    prescription proxies, `infections_all_definitions` = IV + prescription, and
    `treatment_failure_transformation` = OS + second line. It has been corrected
    to match the R composites: MACE = AF + VTE + HF + MI + stroke/TIA; serious
    infection = sepsis + invasive fungal + Pneumocystis codes (one diagnosis
    code set A40, A41, B44-B46, B59); `infections_all_definitions` includes the
    serious-infection composite; hepatotoxicity added; and
    `treatment_failure_transformation` includes next haematological malignancy.
    The file is used for diagnostics/scaffold discovery only, so labels and
    released outcomes were never affected. Evidence:
    `opera/configs/endpoint_dependencies.yaml`:5-18 (working tree); AE:2110-2117,
    2148-2154; COMP:161-180.
28. **`opera_ascertainment_supplement.tex` is out of date.** It describes an
    LPR3 gate (2019-02-03 plus 30 days) on designated outcomes (now only the
    SMR gate), laboratory gating of the IV/prescription proxies (now
    administrative), a 2026-03-04 second-line cutoff (now 2026-03-19, enforced
    by the sealed runner; 2026-03-04 remains selectable only in standalone
    historical/sensitivity runs) and the `JO1EE01` typo (code now excludes
    `J01EE01`). Evidence: `paper/opera_ascertainment_supplement.tex`:164-175,
    210-211; GATE:53-57; AE:2355, 2361-2367; SL:835-839;
    `opera/R/run_secondline_from_sealed_firstline.R`:19-22.
29. **Second-line header comment on anti-CD20.** The script header states that
    anti-CD20 alone is "systematically excluded"; the code counts repeated
    (>= 2 dates) anti-CD20-only episodes as retreatment and, with obinutuzumab
    (an anchor) or a haematological indication plus two distinct codes, as
    second line, unless a maintenance window applies (MCL, indolent B-NHL
    only). The supplement describes the implemented behaviour. Evidence:
    SL:17-19 vs SL:388-394, 413-421.
30. **Laboratory-engine header comment.** The comment states "an observed event
    is always eligible"; the code (correctly) never lets an event override
    eligibility. Evidence: AE:768 vs AE:916-917.
31. **Hospitalisation header comment — RESOLVED.** The header in
    `opera/R/calculate_adverse_events_update.R` previously stated LPR2
    inpatient = `c_pattype == 1`. It has been corrected; the code was already
    correct: LPR2 patient type 0 = inpatient, 2 = outpatient, 3 = emergency
    room before 2014; from 2014 ER = outpatient with acute admission mode. The
    outcome keeps only episodes containing an acute contact, so planned
    admissions count only when merged with an acute contact. The supplement
    (S6, acute hospitalisation) describes this. Evidence: AE:1759-1764
    (working tree), AE:1822 (working tree; `c_pattype == 0`).
32. **Pass-1 disease-assignment window.** Documents describe it as "+-14 days";
    the code uses -20 years to +14 days (pass 2: -10 years to +90 days). The
    supplement states the implemented windows. Evidence: FL:1084-1086, 1347,
    1533-1541.

## Other items removed from the supplement (implementation provenance)

33. **LPR3 procedure join key.** LPR3 procedures are joined to contacts on
    `dw_ek_kontakt`, still cast with `as.integer()`; the key audit reported
    zero lost keys (supplement: "found no lost records"). Consider switching to
    the lossless text join used for LPR2. Evidence: AE:1548-1562;
    `opera/R/audit_procedure_join_keys.R`.
34. **CLL/DaMyDa second-line cross-check disabled.** `RKKP_CLL_CLEAN` and
    `RKKP_DaMyDa_CLEAN` are loaded but their cross-check is commented out
    (supplement: "not used for second-line dating"). Evidence: SL:848-859,
    912-926.
35. **IV prophylaxis exclusion commented out.** The IV proxy's prophylaxis
    filter (J01EE01, J05AB01, J05AB11, P02) is commented out (supplement: "No
    prophylaxis exclusion was applied to intravenous administrations").
    Evidence: AE:2340-2348, 2364-2365.
36. **ALP legacy grader behaviour** was established by a server check on
    5 October 2026, not from code in this repository. Evidence:
    `docs/approved_outcome_release_v2_runbook.md`:12-16;
    `opera/configs/outcome_release_primary_v2.yaml`:51-52.
37. **Clinical sign-off pending.** The release manifest status is
    `primary_release_candidate_pending_final_signoff`; the cohort flow and
    Table 1 await supervisor sign-off. Evidence: REL:215.
38. **Endpoint-group count (52)** is derived by applying `endpoint_groups_v1`
    to the 69 names, not read from a release artifact. Evidence:
    `opera/functional/endpoint_groups.py`:6-18.
