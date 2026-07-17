"""Hematology registry cohort grouping definitions.

Training uses the 10 ``cohort_grouped`` groups defined in the production
experiment registry. Evaluation breaks results out by the 24 ``cohort_fine``
diagnoses. This module mirrors ``opera/configs/experiment_registry.yaml`` and
is guarded by regression tests so stale cohort labels cannot re-enter runs.

The mapping is derived from the RKKP/hematology registry classification and
the patient counts confirmed by the study team.  See OPERA_EXPERIMENTS.md for
the experimental rationale.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional

# ── Canonical mapping: fine diagnosis → grouped training cohort ──────────────

FINE_TO_GROUPED: Dict[str, str] = {
    # DLBCL-like (large B-cell and transformation)
    "DLBCL": "DLBCL_like",
    "RT": "DLBCL_like",
    "TRANSFORMED_FL": "DLBCL_like",
    # Indolent B-cell NHL
    "FL": "Indolent_B_NHL",
    "BCL": "Indolent_B_NHL",
    "LPL": "Indolent_B_NHL",
    "EMZL": "Indolent_B_NHL",
    "NMZL": "Indolent_B_NHL",
    "SMZL": "Indolent_B_NHL",
    "MCL": "MCL",
    # T-cell NHL
    "AITL": "T_NHL",
    "ALCL": "T_NHL",
    "PTCL": "T_NHL",
    "TCL": "T_NHL",
    # CLL / SLL spectrum
    "CLL": "CLL_SLL",
    "SLL": "CLL_SLL",
    # BL / LBL spectrum
    "BL": "BL_LBL",
    "LBL": "BL_LBL",
    # Plasma-cell disorders
    "MM": "MM",
    "PCL": "MM",
    # One-to-one groups
    "HL": "HL",
    "HCL": "HCL",
    "AMYLOIDOSIS": "AMYLOIDOSIS",
    "SolM": "MM",
}

# ── Inverse mapping: grouped cohort → frozenset of fine diagnoses ────────────

GROUPED_TO_FINE: Dict[str, FrozenSet[str]] = {}
for _fine, _grouped in FINE_TO_GROUPED.items():
    GROUPED_TO_FINE.setdefault(_grouped, set()).add(_fine)  # type: ignore[arg-type]
GROUPED_TO_FINE = {k: frozenset(v) for k, v in GROUPED_TO_FINE.items()}

# ── Ordered lists for iteration ───────────────────────────────────────────────

ALL_GROUPED: tuple[str, ...] = (
    "DLBCL_like",
    "Indolent_B_NHL",
    "T_NHL",
    "CLL_SLL",
    "BL_LBL",
    "MM",
    "HL",
    "HCL",
    "AMYLOIDOSIS",
    "MCL",
)

ALL_FINE: tuple[str, ...] = (
    "AITL",
    "ALCL",
    "AMYLOIDOSIS",
    "BCL",
    "BL",
    "CLL",
    "DLBCL",
    "EMZL",
    "FL",
    "HCL",
    "HL",
    "LBL",
    "LPL",
    "MCL",
    "MM",
    "NMZL",
    "PCL",
    "PTCL",
    "RT",
    "SLL",
    "SMZL",
    "SolM",
    "TCL",
    "TRANSFORMED_FL",
)

EXCLUDED_FINE: FrozenSet[str] = frozenset()
ALL_EVALUATED_FINE: tuple[str, ...] = ALL_FINE


# ── Helper functions ──────────────────────────────────────────────────────────


def fine_to_grouped(cohort_fine: str) -> Optional[str]:
    """Return the grouped cohort for a fine diagnosis, or None if unknown."""
    return FINE_TO_GROUPED.get(cohort_fine)


def grouped_to_fine(cohort_grouped: str) -> FrozenSet[str]:
    """Return the frozenset of fine diagnoses within a grouped cohort."""
    return GROUPED_TO_FINE.get(cohort_grouped, frozenset())


def is_valid_fine(cohort_fine: str) -> bool:
    """Return True if the string names a known fine cohort."""
    return cohort_fine in FINE_TO_GROUPED


def is_valid_grouped(cohort_grouped: str) -> bool:
    """Return True if the string names a known grouped cohort."""
    return cohort_grouped in GROUPED_TO_FINE


def resolve_training_cohort(
    cohort_name: str,
    *,
    training_cohort_override: Optional[str] = None,
) -> str:
    """Return the grouped training cohort for a (possibly fine) cohort name.

    If ``training_cohort_override`` is given it is returned as-is.  Otherwise:
    - a fine cohort name is mapped to its grouped parent;
    - a grouped cohort name is returned unchanged;
    - an unknown name raises ``KeyError``.
    """
    if training_cohort_override is not None:
        return training_cohort_override
    if cohort_name in GROUPED_TO_FINE:
        return cohort_name
    if cohort_name in FINE_TO_GROUPED:
        return FINE_TO_GROUPED[cohort_name]
    raise KeyError(
        f"Unknown cohort name {cohort_name!r}.  "
        "Must be a known fine diagnosis or grouped cohort."
    )
