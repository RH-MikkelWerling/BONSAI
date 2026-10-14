# CTCAE v6 endpoint review (30 September 2026)

Scope: `opera/R/calculate_adverse_events_update.R`, the 87-outcome registry,
the previously inspected DALY-CARE package revision and NCI [CTCAE v6.0](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf)
and [v5.0](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v5-8x11.pdf).
This is a source audit, not a patient-level validation. The server's installed
DALY-CARE revision, NPU mappings and measurement units still require checking.
No existing outcome parquet or model result was changed by this review.

| Priority | Endpoint(s) | What the R code currently does | v6 interpretation / action |
| --- | --- | --- | --- |
| High | `leukocytosis_g2plus` | Calls external leukocytosis grader at threshold **3**, despite `g2plus` filename; first post-index documentation | v6 has no G2 and says not to use leukocytosis for increases caused by leukemias, steroids or growth factors. Recommend removing from the *primary adverse-event* panel in a future registry version. Retain old parquet only for v4 reproducibility and perhaps exploratory disease-state analysis. |
| High | `lymphopenia_g2plus/g3plus` | Calls external graded `ae_lymphocyte_count_decreased` | The graded v5 **Lymphocyte count decreased** term became v6 **Lymphopenia**, listed only as present/G1. These can remain valuable numeric lymphocyte-depletion endpoints, but not as v6 G2/G3. Either retain explicitly v5-derived lab thresholds with transparent names or omit from a strictly v6 panel. |
| High | `hypokalemia_g2plus/g3plus` | Local bins use <3.0, <2.5, <2.0 mmol/L for G2/G3/G4 | These are not v6 bins: v6 G2 needs symptoms/intervention, G3 is <3.0–2.5, G4 <2.5. A v6 lab-only G3+ candidate is feasible; G2+ is not completely observable from potassium alone. Compare event dates/counts before versioned relabelling. |
| High | `neutropenia_g3plus` | External grader documented as v5-derived in inspected revision | v5 G3 begins below 1.0 x 10^9/L; v6 G3 begins below 0.5 x 10^9/L. Confirm actual server grader and units; this is potentially a substantial label shift. |
| Medium | `thrombocytopenia_g2plus/g3plus` | External v5-derived platelet grader | v6 moves the G3/G4 boundary from 25 to 10 x 10^9/L, but the G3+ entry threshold remains 50. G3+ event membership may be similar if the grader otherwise agrees; grade distribution and clinical-intervention clauses still need checking. |
| Medium | local sodium, magnesium, fibrinogen | Uses `cut()` with default right-closed intervals | Exact values at 120/125 sodium, 0.3/0.4 magnesium and 0.25/0.5/0.75 of fibrinogen LLN can fall on the more severe side than v6's strict inequalities. Use explicit boundary tests and an updated candidate implementation for final data; do not edit old parquets in place. Sodium also lacks symptom-based escalation. |
| Medium | fibrinogen | Uses LLN-ratio bins, fixed 2.0 missing-LLN fallback and 0.5 absolute floor | v6 also has a baseline-relative axis for already-abnormal patients. Validate NPU code, assay, units, missing-LLN frequency and baseline availability before calling it v6. |
| Medium | `lymphocytosis_g2plus/g3plus` | First post-index lymphocyte increase; external grader | v6 retains graded **Lymphocyte count increased**, but in CLL the post-index value may reflect known disease rather than treatment toxicity. Consider retaining as disease-state/response endpoint only; audit pre-index abnormality and persistence by cohort. |
| Low | LDH, haptoglobin, phosphate, urate | Binary lab abnormalities/proxies | v6 supports lab-only G1 for LDH and haptoglobin. Phosphate and urate higher grades require clinical consequences/intervention; continue to describe the files as lab abnormalities, not full graded toxicities. |

The local potassium comment was corrected because it falsely described legacy
bins as v6. Local NPU-name guards and a duplicate composite execution were
also corrected separately; no threshold was changed. The other main lab AE
functions come from an external package, so a truthful uniform-v6 claim needs
the exact server package revision and a per-endpoint crosswalk. The generic
outcome constructor finds first post-index documentation rather than new onset
or worsening from normal baseline; this distinction applies even to otherwise
correct CTCAE thresholds.

## Decision for the primary panel

Recommend **excluding leukocytosis** from the next primary adverse-event
training/evaluation panel, while preserving the v4 result as a documented
exploratory disease-state prediction. Do not interpret its high CLL AUROC as
proof of treatment-toxicity prediction. Check by cohort the fraction already
above threshold before index, time to first post-index positive (especially
first 1/7/30 days), and whether the same pre-index abnormality persists. Apply
the same baseline-persistence check to lymphocytosis; v6 permits that term, but
the scientific estimand is still disease status unless incident/worsening is
defined. Leave the existing 87-outcome registry unchanged until a versioned
final-release decision, so v4 checkpoints and validation summaries remain
reproducible.

## Adding outcomes

No new CTCAE endpoint currently clears both proposed gates: clear added
clinical value **and** straightforward, validated ascertainment in the sources
already loaded. Febrile neutropenia is clinically important but requires fever
and ANC timing plus infection/clinical context; a neutropenia lab result or
diagnosis code alone is not enough. Registry-coded events such as VTE, sepsis,
hospitalisation, transfusion and overall survival are already present. Defer
new additions until a source-code/coverage audit demonstrates a specific
underrepresented clinical domain and a reliable ascertainment rule.

## Primary-panel pruning decision and pending checks

The study team favours a smaller, clinically defensible primary panel for the
next *versioned* outcome release. Retain old outcome files, configuration and
validation results for reproducibility; do not silently alter v4 runs.

| Disposition | Outcome(s) | Evidence and remaining action |
| --- | --- | --- |
| Remove from next primary training/headline panel | `hypofibrinogenemia_g2plus`, `hypofibrinogenemia_g3plus` | Only 176/72 training events across available follow-up, and 8/5 positives in the 365-day tuning comparison. Assay/unit and baseline grading also remain unverified. Preserve as historical/exploratory files. |
| Keep, conditional on grader/assay check | `hyperkalemia_g2plus`, `hyperkalemia_g3plus` | 1,851/795 training events, 140/51 tuning and 319/125 held-out across follow-up. Verify installed grader, NPU and units. Do not require an unrelated all-cause admission to infer CTCAE grade. Obtain 365-day held-out support separately. |
| Retire from primary panel unless rebuilt with a defensible definition | `dic_like_coagulopathy`, `hys_law_dili`, `tumor_lysis_syndrome` | These are laboratory co-occurrence proxies, not adjudicated clinical syndromes. DIC-like permits INR + platelets without fibrinogen, so removing fibrinogen endpoints alone does not remove the composite. All three lack ordinary lab-coverage eligibility for negative examples. Preserve historic outputs and audit component overlaps before any exploratory reuse. |
| Review for support/ascertainment before primary inclusion | `invasive_fungal`, `hypoglycemia_g3plus` | 30/109 training events across follow-up and 5/4 tuning events, respectively. Rarity does not make them clinically unimportant, but it weakens standalone training and evaluation claims. Check 365-day training and held-out support, source validity and disease-specific concentration. |

The v6 term **Neutrophil count decreased** exists under Investigations, with
ANC-based grade bands; **Febrile neutropenia** is a separate term. The concern
is whether the *installed* external grader matches v6, not whether the term is
absent. CTCAE-based endpoint names should be qualified individually where the
operational rule is a proxy or uses older thresholds.

For antimicrobial proxies, the active prescription filter contains `JO1EE01`
where the upstream codebook uses `J01EE01`, and the IV exclusion is commented
out. The episode constructors require administration/dispensing dates at least
one day apart within gaps of at most seven days; this measures repeated records,
not days supplied, continuous therapy or diagnosed infection. Source-specific
capture/eligibility, prophylaxis code intent, treatment indication and overlap
with diagnosis-defined serious infection need an aggregate comparison before
the new release. Do not change the old infection labels in place.

## CTCAE v6 crosswalk against the active R calls

Reference: [NCI CTCAE v6.0](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf). Many `AE_*` functions are imported from an external DALY-CARE loader. For those rows, the local call site establishes the function and requested grade, **not** that the installed function implements v6. Archive the server's actual package revision before final labels are made. This compares grading, not causality or new onset.

| R outcomes / grader | v6 comparison | Release action |
| --- | --- | --- |
| `ALP_increased_g2plus/g3plus` via external `AE_BASP_increased` | **Confirmed term mismatch:** v6 *Alkaline phosphatase increased* has only G1 (`>baseline and ULN`), no G2/G3. | Remove as v6-grade endpoints. A useful legacy/threshold-based ALP elevation would need a different name and documented thresholds. |
| `lymphopenia_g2plus/g3plus` via external lymphocyte-decreased grader | **Confirmed term mismatch:** v6 *Lymphopenia* is present/G1 only. | Use a transparent absolute-lymphocyte-count threshold name or mark as v5-derived; not v6 G2/G3. |
| `leukocytosis_g2plus` | R requests grade **3**; v6 has no G2 and advises against this term for leukemia-related increases. | Retire from primary adverse-event panel; preserve old result as exploratory. |
| `hypokalemia_g2plus/g3plus` via local grader | **Confirmed threshold mismatch:** local G3 begins below 2.5 mmol/L; v6 G3 covers `<3.0–2.5`, with G4 `<2.5`. v6 G2 also needs symptoms/intervention. | Rebuild with explicit v6 lab-severity bands where observable, or rename legacy thresholds. Do not relabel existing files. |
| `hyponatremia_g2plus/g3plus` via local grader | v6 G2 is asymptomatic 125–<130; G3 is symptomatic 125–<130 or 120–<125 regardless of symptoms. R cannot observe symptoms; `cut(..., right=TRUE)` also misplaces exact 120/125/130 boundaries. | Call it a lab-only severity proxy; use explicit boundary tests in a versioned regrade. |
| `hypomagnesemia_g2plus/g3plus`, `hypermagnesemia_g3plus` via local graders | Broad numerical bands reflect v6, but `cut()` right-closure misplaces exact 0.3/0.4 and 3.30 mmol/L boundaries; units/NPU and reference limits remain unverified. | Boundary-test and validate units/code before v6 claim. |
| `hyperkalemia_g2plus/g3plus` via external grader | v6 includes bands `>5.5–6.0` and `>6.0–7.0 mmol/L`, plus intervention/hospitalization descriptors. | Keep provisionally as lab hyperkalemia; inspect installed grader, assay units and exact boundaries. Do not use all-cause admission as proof of attribution. |
| `neutropenia_g2plus/g3plus` via external ANC grader | v6 *Neutrophil count decreased* exists: G2 `<1.0–0.5`, G3 `<0.5–0.1` x 10^9/L. *Febrile neutropenia* is separate. | Retain if useful, but compare installed thresholds and units; regrade or mark legacy rule. |
| `thrombocytopenia_g2plus/g3plus` via external grader | v6 *Thrombocytopenia* G3 starts `<50` and G4 `<10` x 10^9/L; inspected external revision had older G4 boundary. | Check actual grader. G3+ membership may stay similar; exact grade labels need a crosswalk. |
| `ALT/AST/bilirubin/creatinine` graded calls | v6 has baseline-dependent branches for some, particularly ALT/AST and creatinine. Baseline handling is not proven by the local script. | Audit installed source, baseline branch, units and pre-index availability. |
| `anemia`, `WBC_decreased`, `INR`, calcium, sodium-high, triglycerides, albumin, glucose and other external graded lab calls | v6 contains corresponding terms, but call sites alone cannot verify thresholds, clinical qualifiers or reference-unit handling. INR also varies with anticoagulation status. | Build an installed-grader threshold table before claiming v6 alignment; use “CTCAE-based laboratory endpoints” meanwhile. |
| `LDH_increased`, `haptoglobin_decreased` | v6 has only G1 lab findings; R exports binary abnormalities. | Broadly appropriate after NPU/reference-range checks; do not infer hemolysis from haptoglobin alone. |
| `hyperuricemia`, `hyperphosphatemia` | R exports binary lab elevations; higher v6 grades rely on consequences or intervention. | Keep explicitly as lab-elevation proxies, not full graded clinical AEs. |
| DIC-like, TLS, Hy's-law-style and medication-derived infection endpoints | Locally constructed proxies, not direct CTCAE grades or fully adjudicated diagnoses. | Remove questionable composites from the next primary panel pending validation; describe any exploratory definitions precisely. |

This establishes concrete mismatches but is **not** patient-level or installed-package validation. For final data, record the server loader/package commit, function bodies, NPU mappings, unit distributions and before/after event counts by split and horizon. ALP and lymphopenia cannot be fixed by changing a version label alone.
