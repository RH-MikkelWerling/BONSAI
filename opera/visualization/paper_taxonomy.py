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
