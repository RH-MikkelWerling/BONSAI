"""Hematology registry cohort grouping definitions.

Two disease-taxonomy levels are used by different pipeline stages:

- ``cohort_grouped`` (10 groups) gates the *upstream self-supervised*
  contrastive/DAPT/pretrain encoder pretraining population (built once via
  ``opera.run.contrastive_multicohort``, before any outcome labels exist).
  Pooling broadly there is legitimate and unchanged — it's the foundation-
  model backbone.
- ``cohort_fine`` (24 diagnoses) gates the *supervised finetuning* (train/
  val/test split) and evaluation population for the sweep, via
  ``cohort_fine_col``/``cohort_fine_value`` applied in
  ``opera.run.survival_finetune`` before any split happens. Fine-level sweep
  cells train and evaluate strictly on their own fine diagnosis — never the
  pooled grouped parent.

This module mirrors ``opera/configs/experiment_registry.yaml`` and is guarded
by regression tests so stale cohort labels cannot re-enter runs.

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
