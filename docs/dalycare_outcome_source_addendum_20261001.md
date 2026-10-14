# Outcome source-audit addendum (internal, 1 October 2026)

This review compares the uploaded 30 September outcome audit with the public
[DALY-CARE package](https://github.com/RH-CLL-LAB/dalycare_package) main commit
`a0742b0de4d0984d7a8491bd103144236a1f27b9` and the local
`opera/R/calculate_adverse_events_update.R` call sites, against
[NCI CTCAE v6.0](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf). The public functions
label themselves CTCAE v5-derived. **The server's sourced loader and installed
function bodies have not yet been matched to this commit.** Findings below are
source-level hypotheses, not claims about the exact exported v4 labels.

## Uploaded audit and early-horizon support

- `summary_INTERNAL.json` records 87 exports: 60 primary candidates, 10
  definition holds, nine low-support exploratories, eight retirements.
  `horizon_support_INTERNAL.csv` has all 87 x 3 splits x 6 horizons (1,566)
  unique cells. The 87 structural audits have zero chronology, duplicate-ID,
  missing-date or split flags. The 87 file hashes are an output manifest, not
  upstream source provenance.
- The figure taxonomy's 60 names exactly match `panel_decisions_INTERNAL.csv`.
  At 30 days, 17/60 candidates have fewer than 20 tuning events and seven
  have fewer than ten. Examples: stroke/TIA 5, VTE 7, heart failure 4,
  atrial fibrillation 16, hypertriglyceridemia G3+ 4. The two later-malignancy
  outcomes have zero 30-day events because their R calls use a 31-day grace
  period; their missing 30-day AUCs must not be interpreted as model failure.
  Training events are not tuning AUC precision. Obtain the run's
  `endpoint_selection_inventory.json` for actual tuning cases/controls per
  outcome-horizon and, for formal contrasts, paired patient bootstrap intervals.

## Source-level triage for the 60 candidate outcomes

| Decision | Endpoint(s) | Evidence and action |
| --- | --- | --- |
| Regrade or transparently relabel before final release | Neutropenia G2+/G3+ | Public `AE_neutrophil_count_decreased.R` uses cutpoints 1.5, 1.0, 0.5, whereas v6 G2 is <1.0–0.5 and G3 is <0.5–0.1 x 10^9/L. The G3+ membership can change substantially; audit units and regrade with boundary tests. |
| Audit baseline handling and regrade if confirmed | ALT G2+/G3+ | Public `AE_ALAT_increased.R` uses `lag(cummean(value_numeric))` over a patient's full lab history, so later post-index draws can change the reference used at each event. Its abnormal-baseline multipliers (G3 >5x, G4 >20x) differ from v6 (G3 >2–4x; G4 >4x, subject to v6 qualifiers). A fixed pre-index baseline should be evaluated on the server before replacing labels. AST is exploratory, but is still needed for the `hepatotoxicity` composite. |
| Audit/regrade conditionally | Bilirubin G2+/G3+, creatinine G2+/G3+ | Public bilirubin/creatinine functions use fixed multiples of ULN; v6 includes baseline-dependent branches. Quantify how many patients have abnormal pre-index baseline and how many labels would change. Preserve a clearly named legacy rule if a faithful v6 rule cannot be supported. |
| Audit clinical qualifier, likely keep as lab proxy | INR G2+/G3+, hyperkalemia G2+/G3+ | Public INR is absolute-threshold only, while v6 has anticoagulation/baseline branches. Public potassium uses numerical bins; v6 also permits intervention/hospitalization criteria. Check exposure/clinical data availability; do not claim complete CTCAE grading from lab bins alone. |
| Likely retain binary membership after unit/boundary check | Anemia G2+/G3+, thrombocytopenia G2+/G3+, hypomagnesemia G2+ | The public anemia and platelet functions have imperfect grade details (e.g. platelet G4 at <25 rather than v6 <10), but the currently requested G3+ platelet threshold (<50) can remain the same. The locally defined hypomagnesemia uses v6-style 0.5/0.4/0.3 mmol/L cutoffs. Test exact boundaries and assay units; avoid unnecessary re-export when binary membership is unchanged. |
| Retain as a named proxy or refine if simple | Eosinophilia | Public code checks >ULN, while v6 G1 also requires >baseline. If baseline can be identified reliably, add it; otherwise label the outcome explicitly as eosinophil elevation, not full v6 eosinophilia. |
| Retain clinically, with precise names | Sepsis, serious-infection composite, antibiotics, lymphocytosis | Diagnosis-coded infection and antimicrobial use are distinct constructs. `antibiotics` is medication use, not confirmed infection. Lymphocytosis is a disease-burden marker in this population, not necessarily treatment toxicity; keep only with that interpretation and audit its strong performance for leakage/ascertainment. |
| Retain but show timing caveat | Next malignancy and next haematological malignancy | R calls use `grace_days = 31`, explaining zero day-30 cases. Validate the clinical intent of this grace period and source codes; omit day-30 AUC rather than forcing a value. |

Other imported graders (calcium, sodium-high, albumin, triglycerides, WBC,
lymphocytosis) still need unit/reference-range and threshold-boundary checks.
No action above authorizes overwriting old parquets or silently relabelling
historical results. Retired and held outcomes remain outside the primary panel
unless a separate clinical decision revises their definition.

## What to run next

1. On the server, save the loader's real file path/hash, package checkout
   commit and `body(AE_...)`/NPU mappings to a secure provenance directory.
   Compare those bodies with the public commit before treating this crosswalk
   as executable truth. Also record source table paths/hashes and the exact
   administrative cutoff.
2. Retrieve `endpoint_selection_inventory.json` from the completed neural run;
   align it with `horizon_support_INTERNAL.csv` for 30/60-day case/control
   support. This is read-only and needs no training rerun.
3. Create a versioned *candidate* outcome output, initially for neutrophils,
   ALT/bilirubin/creatinine and eosinophilia. Unit-test numeric boundaries and
   baseline selection, then compare old/new eligibility, event dates and
   split/horizon counts on-server. Export aggregates only.
4. Once these decisions and the final MEDS/source release are frozen, rebuild
   first-line/second-line outcomes and dependent composites in a fresh release
   directory. Re-run structural/panel audits before launching finalist models.

The public GitHub comparison alone cannot determine which revised endpoints
improve clinical validity or how many patients change labels; the server-side
loader and patient-level transition audit are the decisive checks.
