# Where Carsten Niemann may push back, 18 October 2026

Drawn from his last ten years of publications (CLL-TIM 2020 and 2024,
DALY-CARE 2025, "One for the ages" 2025, the domain-adaptation, polypharmacy,
pre-diagnostic infection, atrial fibrillation and DLBCL papers, and the CLL-IPI
reassessment). This is a map of likely arguments, not a decision about
inclusion. Quoted fragments were extracted from PMC full texts; check them
against the PDFs before quoting.

| # | Likely argument | Where it comes from | Possible answer | Status |
|---|---|---|---|---|
| A | No explainability: "opening the black box" is key to clinical trust | CLL-TIM (Nat Commun 2020) showed personalised risk factors; the deployed version shows the top factors and a confidence; MLAll and the Acta Oncol paper use SHAP | Input-restriction analysis (code ready). Limitation stated: no patient-level explanations, needed before clinical use. Option: SHAP on XGBoost to show sensible features | Limitation added; analysis your call |
| B | No confidence per prediction | CLL-TIM deployment: "high-confidence predictions were generally more correct" | Neighbourhood calibration and rarity in training (Discussion, now linked to CLL-TIM's confidence); seed or ensemble spread later | Text added |
| C | Registers only, no EHR (labs, vitals, IV antimicrobials) | DALY-CARE was built around EHR for two of five regions | National uniformity, portability and national pretraining. Optional: an eastern-Denmark EHR sensitivity analysis | Justification in Methods; analysis optional |
| D | Infection proxies mix in prophylaxis | His group's definitions: blood culture drawn (CLL-TIM), IV antimicrobials (polypharmacy), IV plus culture (Acta Oncol); BJC 2024 used outpatient prescriptions as a proxy | Prophylactic agents excluded (now stated; list the ATC codes); stricter label as an evaluation; cite BJC 2024 as precedent | Text added; codes needed |
| E | What action does a treatment-start prediction trigger? | CLL-TIM predicts 3 months after diagnosis to select for early treatment (PreVent-ACaLL) | Intended use names prophylaxis, growth factors, monitoring and dose planning. Could add IgG replacement and vaccination for CLL | In place |
| F | Beating an index on its own endpoint is a low bar; age alone beats the indices | "One for the ages" (Blood Adv 2025); CLL-IPI weaker in the targeted era (Blood 2024) | Add an age-only (or age and sex) baseline, and index plus continuous age | **Decision: add simple baselines?** |
| G | Index computability and missingness | CLL-TIM: CLL-IPI computable for 48% (2020) and 0.63% at deployment (2024) | Report the share of patients with each index, complete-case and full-cohort comparisons | Results placeholder exists; extend to every index |
| H | Deployment realities: harmonisation, MDR class, drift monitoring | CLL-TIM deployment: harmonisation "by far the most time-consuming", MDR class IIa, quarterly drift checks | Discussion covers harmonisation and drift; medical-device regulation now stated | Text added |
| I | Treatment era: targeted agents dominate the test period | CLL-IPI reassessment; CIRI2-CLL (JCO 2026) | Results by calendar year (prespecified); optionally by regimen class (chemoimmunotherapy vs targeted) | **Decision: regimen-class stratum?** |
| J | Does a simple model do as well (age, medication count, prior antimicrobials)? | Polypharmacy (HemaSphere 2025): medication count predicted as well as the IPIs; BJC 2024: prior antimicrobials prognostic | A simple-baseline row in Table 2 (cheap, logistic regression on three variables) | **Decision: add?** |
| K | Pooling and negative transfer; sparse histories | Acta Oncol 2026: the gain vanished in patients without prior hospitalisation | Per-subtype gains and losses (Fig. 4, S7), performance by recorded history (Fig. 2E, S20) | In place |
| L | International transport, OMOP, federated learning | DALY-CARE maps about 90% of data to OMOP and envisions federated models | One sentence added in the limitations | Text added |

## Where his work supports the paper
- Infection and supportive care are the unmet need he has argued for since
  CLL-TIM.
- DALY-CARE's stated rationale is joint modelling across lymphoid cancers,
  now cited in the Introduction.
- Prediction at first-line treatment start, as in the Acta Oncol, atrial
  fibrillation and DLBCL papers.
- Data-rich models cope with missingness that defeats the indices (the
  CLL-IPI computability figures).
- His group frames models as decision support, not treatment selection,
  matching the intended use.
- The temporal held-out test set goes beyond the random split of the DLBCL
  paper.

## Citations to consider adding (metadata to verify first)
- Packness et al., Br J Cancer 2024: precedent for prescription-based
  infection proxies.
- Brieghel et al., HemaSphere 2025 (polypharmacy): medication count as a
  strong single predictor.
- Parviz et al., Leuk Lymphoma 2024 (atrial fibrillation): treatment-aware
  prediction at treatment start.
- Langerbeins et al., Blood 2024: CLL-IPI in the era of targeted therapy.
