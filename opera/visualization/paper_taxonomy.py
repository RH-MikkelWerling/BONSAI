"""Display names and groupings shared by the paper figures and their inputs.

Analysis tables keep registry identifiers (fine cohorts such as ``SolM`` and
outcomes such as ``anemia_g3plus``); figures show clinical names. Every plot
table is translated here, so prototypes and real figures read identically.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from opera.visualization.outcome_names import outcome_display_name

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "opera/configs/experiment_registry.yaml"
DISPLAY_GROUPS = REPO / "opera/configs/outcome_display_groups_paper_v1.yaml"

GROUP_NAMES = {"AMYLOIDOSIS": "Amyloidosis", "BL_LBL": "BL/LBL", "CLL_SLL": "CLL/SLL",
               "DLBCL_like": "DLBCL-like", "HCL": "HCL", "HL": "Hodgkin",
               "Indolent_B_NHL": "Indolent B-NHL", "MCL": "MCL", "MM": "Myeloma",
               "T_NHL": "T-cell NHL"}
FINE_NAMES = {"AMYLOIDOSIS": "Amyloidosis", "TRANSFORMED_FL": "Transformed FL",
              "SolM": "Solitary plasmacytoma"}


# Full clinical names for tables (abbreviation as used in the figures).
GROUP_LONG = {
    "DLBCL_like": "Large B-cell lymphomas",
    "BL_LBL": "Burkitt and lymphoblastic lymphomas",
    "MCL": "Mantle-cell lymphoma",
    "Indolent_B_NHL": "Indolent B-cell lymphomas",
    "CLL_SLL": "CLL and SLL",
    "HCL": "Hairy-cell leukaemia",
    "HL": "Hodgkin lymphoma",
    "T_NHL": "T-cell lymphomas",
    "MM": "Plasma-cell neoplasms",
    "AMYLOIDOSIS": "Amyloidosis",
}
# Clinical order for tables: aggressive B-cell, indolent B-cell, Hodgkin, T-cell,
# plasma-cell (not alphabetical, so related groups sit together).
GROUP_ORDER = ("DLBCL_like", "BL_LBL", "MCL", "Indolent_B_NHL", "CLL_SLL", "HCL", "HL",
               "T_NHL", "MM", "AMYLOIDOSIS")
FINE_LONG = {
    "DLBCL": "Diffuse large B-cell lymphoma (DLBCL)",
    "RT": "Richter transformation (RT)",
    "TRANSFORMED_FL": "Transformed follicular lymphoma (tFL)",
    "BL": "Burkitt lymphoma (BL)",
    "LBL": "Lymphoblastic lymphoma (LBL)",
    "MCL": "Mantle-cell lymphoma (MCL)",
    "FL": "Follicular lymphoma (FL)",
    "LPL": "Lymphoplasmacytic lymphoma (LPL)",
    "EMZL": "Extranodal marginal-zone lymphoma (EMZL)",
    "SMZL": "Splenic marginal-zone lymphoma (SMZL)",
    "NMZL": "Nodal marginal-zone lymphoma (NMZL)",
    "BCL": "B-cell lymphoma, unspecified (BCL)",
    "CLL": "Chronic lymphocytic leukaemia (CLL)",
    "SLL": "Small lymphocytic lymphoma (SLL)",
    "HCL": "Hairy-cell leukaemia (HCL)",
    "HL": "Hodgkin lymphoma (HL)",
    "PTCL": "Peripheral T-cell lymphoma, NOS (PTCL)",
    "AITL": "Angioimmunoblastic T-cell lymphoma (AITL)",
    "ALCL": "Anaplastic large-cell lymphoma (ALCL)",
    "TCL": "T-cell lymphoma, unspecified (TCL)",
    "MM": "Multiple myeloma (MM)",
    "PCL": "Plasma-cell leukaemia (PCL)",
    "SolM": "Solitary plasmacytoma",
    "AMYLOIDOSIS": "Amyloidosis",
}
# Under its group heading a subtype drops what the heading already says
# ("T-cell lymphomas" > "Angioimmunoblastic (AITL)"), so Table 1 rows fit one
# line; a group with one subtype is a standalone row and keeps FINE_LONG.
FINE_IN_GROUP = {
    "DLBCL": "Diffuse large B-cell (DLBCL)",
    "RT": "Richter transformation (RT)",
    "TRANSFORMED_FL": "Transformed follicular (tFL)",
    "BL": "Burkitt (BL)",
    "LBL": "Lymphoblastic (LBL)",
    "FL": "Follicular (FL)",
    "LPL": "Lymphoplasmacytic (LPL)",
    "EMZL": "Extranodal marginal-zone (EMZL)",
    "SMZL": "Splenic marginal-zone (SMZL)",
    "NMZL": "Nodal marginal-zone (NMZL)",
    "BCL": "Unspecified B-cell (BCL)",
    "CLL": "Chronic lymphocytic leukaemia (CLL)",
    "SLL": "Small lymphocytic lymphoma (SLL)",
    "PTCL": "Peripheral T-cell, NOS (PTCL)",
    "AITL": "Angioimmunoblastic (AITL)",
    "ALCL": "Anaplastic large-cell (ALCL)",
    "TCL": "Unspecified T-cell (TCL)",
    "MM": "Multiple myeloma (MM)",
    "PCL": "Plasma-cell leukaemia (PCL)",
    "SolM": "Solitary plasmacytoma",
}
# Subtypes without a dedicated prognostic index (prespecified stratum): no
# multivariable index derived for the entity and externally validated in an
# independent cohort; indices borrowed from other entities (the IPI applied to
# RT, tFL, ALCL) or broader families (MZL-IPI) do not count. The unspecified
# categories have none by construction. Evidence per subtype: supplementary
# table "Prognostic indices by lymphoid subtype" (tab:index-availability).
# Sensitivity analyses: without NMZL; with AITL and BL added.
NO_ROUTINE_INDEX = frozenset({"LBL", "HCL", "ALCL", "TRANSFORMED_FL", "PCL", "RT", "SolM",
                              "NMZL", "BCL", "TCL"})


def disease_long_name(fine: str) -> str:
    return FINE_LONG.get(fine, disease_name(fine))


def disease_in_group_name(fine: str) -> str:
    """Subtype name under its group heading (Table 1)."""
    return FINE_IN_GROUP.get(fine, disease_long_name(fine))


def disease_group_long_name(group: str) -> str:
    return GROUP_LONG.get(group, disease_group_name(group))


def disease_name(fine: str) -> str:
    return FINE_NAMES.get(fine, fine)


def disease_group_name(group: str) -> str:
    return GROUP_NAMES.get(group, group)


def fine_cohort_groups(registry_path: Path = REGISTRY) -> dict[str, str]:
    """Registry fine cohort -> registry cohort group."""
    registry = yaml.safe_load(Path(registry_path).read_text())
    return {fine: group for group, spec in registry["cohort_groups"].items()
            for fine in spec["fine"]}


def outcome_endpoint_groups(groups_path: Path = DISPLAY_GROUPS) -> dict[str, str]:
    """Registry outcome -> clinical display group."""
    groups = yaml.safe_load(Path(groups_path).read_text())
    return {outcome: group for group, members in groups.items() for outcome in members}


def to_display(frame: pd.DataFrame, *, registry_path: Path = REGISTRY) -> pd.DataFrame:
    """Translate registry identifiers in ``cohort``/``cohort_group``/``outcome``.

    ``cohort_group`` is filled from the registry when absent, so tables need
    only carry the fine cohort.
    """
    frame = frame.copy()
    if "cohort" in frame:
        if "cohort_group" not in frame:
            frame["cohort_group"] = frame.cohort.map(fine_cohort_groups(registry_path))
        frame["cohort"] = frame.cohort.map(disease_name)
    if "cohort_group" in frame:
        frame["cohort_group"] = frame.cohort_group.map(
            lambda g: disease_group_name(g) if isinstance(g, str) else g)
    if "outcome" in frame:
        frame["outcome"] = frame.outcome.map(outcome_display_name)
    return frame
