"""OPERA outcome-frame helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from bonsai.functional.features import compute_abspos

LOGGER = logging.getLogger(__name__)


def resolve_registry_start_date(
    cohort_config: Mapping[str, Any],
    outcome_config: Mapping[str, Any],
) -> Optional[Any]:
    """Resolve outcome coverage, allowing an explicit outcome-level exemption.

    Outcome-level coverage takes precedence when the key is present. This
    includes an explicit ``null``, which means the outcome is not restricted by
    a cohort-level registry date. If the outcome omits the key, the cohort
    default is used.
    """
    if "registry_start_date" in outcome_config:
        return outcome_config.get("registry_start_date")
    return cohort_config.get("registry_start_date")


def _pandas_abspos(timestamps: pd.Series) -> pd.Series:
    """Convert timestamps to float32 hours since Unix epoch."""
    return compute_abspos(timestamps)


def attach_prediction_censor_abspos(
    outcomes: pd.DataFrame,
    prediction_time_col: str = "index_date",
) -> pd.DataFrame:
    """Set sequence truncation position from the prediction origin.

    ``censor_date`` is retained for follow-up and time-to-event calculations.
    The model input should be truncated at ``index_date`` so downstream
    finetuning and evaluation do not see post-index information.
    """
    if prediction_time_col not in outcomes.columns:
        raise ValueError(
            f"Outcome frame is missing prediction-time column {prediction_time_col!r}."
        )
    outcomes = outcomes.copy()
    outcomes["censor_abspos"] = _pandas_abspos(outcomes[prediction_time_col])
    return outcomes


def filter_registry_eligible_outcomes(
    outcomes: pd.DataFrame,
    registry_start_date: Optional[Any] = None,
    *,
    date_col: str = "index_date",
    cohort: Optional[str] = None,
    outcome_name: Optional[str] = None,
) -> pd.DataFrame:
    """Restrict supervised outcome rows to registry-covered prediction times.

    Patients with prediction times before registry coverage begins can still
    contribute to MLM/DAPT pretraining, but they should not be treated as
    supervised cases, controls, or ordinary RKKP-missing rows.
    """
    if registry_start_date in (None, "", "null"):
        return outcomes.copy()
    if date_col not in outcomes.columns:
        raise ValueError(
            f"Cannot apply registry_start_date; outcome frame is missing {date_col!r}."
        )
    start = pd.Timestamp(registry_start_date)
    dates = pd.to_datetime(outcomes[date_col])
    eligible = dates >= start
    n_before = int(len(outcomes))
    filtered = outcomes.loc[eligible].copy()
    n_excluded = n_before - int(len(filtered))
    LOGGER.info(
        "registry_eligibility_filter cohort=%s outcome=%s start=%s "
        "excluded=%s/%s retained=%s",
        cohort or "unknown",
        outcome_name or "unknown",
        start.date(),
        n_excluded,
        n_before,
        len(filtered),
    )
    return filtered


def filter_outcome_eligibility(
    outcomes: pd.DataFrame,
    eligibility: Optional[pd.DataFrame | str | Path],
    *,
    cohort: Optional[str] = None,
    outcome_name: Optional[str] = None,
    eligibility_scope: str = "final",
) -> pd.DataFrame:
    """Apply a patient-level outcome ascertainment sidecar.

    The sidecar must satisfy the repository eligibility schema and contain an
    entry for every row in the outcome frame. Rows marked ``eligible=false``
    are excluded before binarization, giving multi-outcome datasets a genuine
    per-patient, per-outcome missingness mask.
    """
    if eligibility is None or (
        isinstance(eligibility, str) and eligibility in ("", "null")
    ):
        return outcomes.copy()

    from opera.evaluation.cohort_flow import (
        eligibility_mask,
        load_eligibility_frame,
        validate_eligibility_frame,
    )

    frame = (
        eligibility.copy()
        if isinstance(eligibility, pd.DataFrame)
        else load_eligibility_frame(eligibility)
    )
    issues = validate_eligibility_frame(frame)
    if issues:
        raise ValueError(
            f"Invalid eligibility sidecar for {cohort or 'unknown'}/"
            f"{outcome_name or 'unknown'}: {'; '.join(issues)}"
        )
    frame = frame.copy()
    frame["eligible"] = eligibility_mask(frame, scope=eligibility_scope)

    required = {"subject_id", "split"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(
            "Outcome eligibility filtering requires columns: "
            f"{sorted(required)}; missing {sorted(missing)}."
        )

    outcome_keys = outcomes[["subject_id", "split"]].copy()
    eligibility_keys = frame[["subject_id", "split", "eligible"]].copy()
    merged = outcome_keys.merge(
        eligibility_keys,
        on=["subject_id", "split"],
        how="left",
        validate="many_to_one",
    )
    missing_sidecar = merged["eligible"].isna()
    if missing_sidecar.any():
        examples = merged.loc[missing_sidecar, ["subject_id", "split"]].head(10)
        raise ValueError(
            f"Eligibility sidecar for {cohort or 'unknown'}/"
            f"{outcome_name or 'unknown'} is missing {int(missing_sidecar.sum())} "
            f"outcome rows; examples={examples.to_dict('records')}."
        )

    eligible_mask = merged["eligible"].astype(bool).to_numpy()
    filtered = outcomes.loc[eligible_mask].copy()
    LOGGER.info(
        "outcome_eligibility_filter cohort=%s outcome=%s scope=%s "
        "excluded=%s/%s retained=%s",
        cohort or "unknown",
        outcome_name or "unknown",
        eligibility_scope,
        int((~eligible_mask).sum()),
        len(outcomes),
        len(filtered),
    )
    return filtered
