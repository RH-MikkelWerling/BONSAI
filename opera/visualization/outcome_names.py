"""Presentation labels for outcome identifiers; never used to define labels."""

from __future__ import annotations

import re


OUTCOME_DISPLAY_NAMES = {
    "overall_survival": "Overall survival",
    "aki_first_documented": "First documented AKI",
    "hospitalisation": "Hospital admission",
    "emergency_stay": "Emergency care",
    "bone_fracture": "Bone fracture",
    "treatment_failure": "Treatment failure",
    "treatment_failure_transformation": "Treatment failure / transformation",
    "second_line_treatment": "Second-line treatment",
    "next_hematological_malignancy": "Subsequent hematologic malignancy",
    "next_malignancy": "Subsequent malignancy",
    "RBC_transfusion": "Red-cell transfusion",
    "any_transfusion": "Any transfusion",
    "platelet_transfusion": "Platelet transfusion",
    "respiratory_support": "Respiratory support",
    "serious_infection_composite": "Serious infection composite",
    "infection_diagnosis_or_antimicrobial_use_composite": "Infection diagnosis or antimicrobial use",
    "iv_antimicrobial_use_proxy": "IV antimicrobial-use proxy",
    "prescription_antimicrobial_use_proxy": "Antimicrobial-prescription proxy",
    "antibiotics": "Antibiotic use",
    "alp_elevation_legacy_tier2plus": "ALP elevation (legacy tier 2+)",
    "alp_elevation_legacy_tier3plus": "ALP elevation (legacy tier 3+)",
    "creatinine_rise_gt1p5x_preindex_baseline": "Creatinine >1.5× pre-index baseline",
    "potassium_below_3p0_mmol_l": "Potassium <3.0 mmol/L",
    "sodium_below_125_mmol_l": "Sodium <125 mmol/L",
    "hyponatremia_g2plus": "Low sodium (G2+ threshold)",
    "hypomagnesemia_g2plus": "Low magnesium (G2+ threshold)",
    "hypomagnesemia_g3plus": "Low magnesium (G3+ threshold)",
    "hypernatremia_g2plus": "High sodium (G2+ threshold)",
    "hypernatremia_g3plus": "High sodium (G3+ threshold)",
    "hepatotoxicity": "Hepatotoxicity composite",
    "hys_law_dili": "Hy's law liver injury",
    "dic_like_coagulopathy": "DIC-like coagulopathy",
    "MACE_composite": "Major cardiovascular events",
    "stroke_TIA": "Stroke / TIA",
    "VTE": "Venous thromboembolism",
    "LDH_increased": "Elevated LDH",
    "WBC_decreased_g2plus": "Low white-cell count (G2+)",
    "WBC_decreased_g3plus": "Low white-cell count (G3+)",
    "ALT_increased_g2plus": "Elevated ALT (G2+)",
    "ALT_increased_g3plus": "Elevated ALT (G3+)",
    "AST_increased_g2plus": "Elevated AST (G2+)",
    "AST_increased_g3plus": "Elevated AST (G3+)",
    "ALP_increased_g2plus": "Elevated ALP (G2+)",
    "ALP_increased_g3plus": "Elevated ALP (G3+)",
    "INR_increased_g2plus": "Elevated INR (G2+)",
    "INR_increased_g3plus": "Elevated INR (G3+)",
}


def outcome_display_name(identifier: str) -> str:
    """Use curated clinical wording, with a readable fallback for new IDs."""
    if identifier in OUTCOME_DISPLAY_NAMES:
        return OUTCOME_DISPLAY_NAMES[identifier]
    match = re.fullmatch(r"(.+)_g([23])plus", identifier)
    if match:
        term = match.group(1).replace("_", " ")
        return f"{term.capitalize()} (G{match.group(2)}+)"
    return identifier.replace("_", " ").capitalize()
